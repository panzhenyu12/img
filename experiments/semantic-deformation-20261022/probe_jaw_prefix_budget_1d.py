"""One 1D retained-target experiment with the six original Plateau prefix caps.

Restores the original D2 objective exactly. Adds only six untapered Dirichlet
SOC constraints; the original plateau preference, area, fixed endpoints,
signs and continuous retained-bend domains remain unchanged. Never renders
pixels or solves a 2D spatial deformation field.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import clarabel
from scipy.sparse import csc_matrix, vstack

from probe_source_affine_bending import (assemble as assemble_original_pair, checks, metrics,
    sha, save, check_source_binding, snapshot_local_code)
from jaw_prefix_budget_1d import prefix_energy_operator, reduce_fixed_area
from measure_jaw_preference_start import measure_preference_start


def assemble(baseline,joint):
    # Reuse the independently checked original 1D assembly, then explicitly
    # select its original linear term. The affine shift is never applied.
    data=assemble_original_pair(baseline)
    data['linear']=data['old_linear'].copy()
    data.pop('reference')
    p=data['problem'];caps=[]
    if joint['source_geometry']!=baseline['source_geometry'] or joint['amount']!=100.:
        raise ValueError('prefix budgets must belong to the same frozen source and signed amount')
    original=joint['jaw_start_intent_constraints']['budgets']
    for i,a in enumerate(p.arcs):
        for end in [2.,6.,12.]:
            measured=measure_preference_start(p,a,100.,.13,end,'equal_area_plateau')
            budget=measured['unweighted_energy_px3']*(12./end)**2
            proof=next(b for b in original if b['id']==a['id'] and b['prefix_px']==end)
            if proof['energy_cap_px3']!=budget or proof['outer_scale_px']!=12.:
                raise ValueError('original joint-field prefix budget changed')
            local=prefix_energy_operator(a['nodes'],end,12.)
            operator=np.zeros((len(local),len(data['linear'])))
            operator[:,data['offsets'][i]:data['offsets'][i+1]]=local
            caps.append({'operator':operator,'budget':budget,'id':a['id'],'end':end,'report':proof})
    data['caps']=caps
    return data


def prefix_checks(data,x):
    rows=[]
    for c in data['caps']:
        residual=c['operator']@x;energy=float(residual@residual)
        rows.append({'id':c['id'],'prefix_px':c['end'],'outer_scale_px':12.,'energy_cap_px3':c['budget'],
                     'actual_energy_px3':energy,'margin_px3':c['budget']-energy,'accepted':bool(energy<=c['budget'])})
    return {'accepted':all(r['accepted']for r in rows),'constraints':rows,'gate':'literal actual energy <= original cap; no added tolerance'}


def solve_round(data,matrix,rhs,time_limit):
    reduction=reduce_fixed_area(data['H'],data['linear'],data['area'],data['wanted'],data['lower'],data['upper'],data['normalization'])
    base,T=reduction['base'],reduction['transform'];n=T.shape[1]
    if not n:raise ValueError('this two-jaw experiment expects a nonempty reduced variable space')
    identity=np.eye(len(base));finite_lower=np.isfinite(data['lower'])&reduction['free'];finite_upper=np.isfinite(data['upper'])&reduction['free']
    original_rows=np.vstack([matrix,identity[finite_lower],-identity[finite_upper]])
    original_rhs=np.r_[rhs,data['lower'][finite_lower],-data['upper'][finite_upper]]
    reduced=original_rows@T;shifted=original_rhs-original_rows@base;norms=np.maximum(np.linalg.norm(reduced,axis=1),1.)
    A=[csc_matrix(-reduced/norms[:,None])];b=[-shifted/norms];cones=[clarabel.NonnegativeConeT(len(shifted))]
    for c in data['caps']:
        R=c['operator'];A.append(csc_matrix(np.vstack([np.zeros(n),-R@T])))
        b.append(np.r_[np.sqrt(c['budget']),R@base]);cones.append(clarabel.SecondOrderConeT(1+len(R)))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200
    settings.tol_feas=1e-12;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.time_limit=float(time_limit)
    solved=clarabel.DefaultSolver(csc_matrix(2*np.eye(n)),reduction['q'],vstack(A,format='csc'),np.concatenate(b),cones,settings).solve()
    if np.shape(solved.x)!=(n,) or not np.isfinite(solved.x).all():return None,str(solved.status)
    return base+T@np.asarray(solved.x),str(solved.status)


def solver_child(baseline,joint,out):
    out=Path(out);data=assemble(baseline,joint);started=time.monotonic();matrix=data['matrix'];rhs=data['rhs'];rounds=[]
    for iteration in range(4):
        remaining=30.-(time.monotonic()-started)
        if remaining<=0.:break
        begin=time.monotonic();x,status=solve_round(data,matrix,rhs,remaining)
        if x is None:
            save(out/'solve.json',{'accepted':False,'solver_status':status,'rounds':rounds});break
        old=checks(data,x);cap=prefix_checks(data,x)
        accepted=status=='Solved' and old['all_original_target_constraints_pass'] and cap['accepted']
        rounds.append({'round':iteration,'solver_status':status,'seconds':time.monotonic()-begin,'original_check':old,'prefix_check':cap})
        np.save(out/'candidate-nodes.npy',x)
        save(out/'solve.json',{'accepted':accepted,'solver_status':status,'rounds':rounds,'seconds':time.monotonic()-started})
        if accepted or status not in ['Solved','AlmostSolved']:break
        added=[];bounds=[]
        for i,a in enumerate(data['problem'].arcs):
            u=x[data['offsets'][i]:data['offsets'][i+1]];positions,margin,_=data['problem'].continuous_minima(a,u);bad=margin < -1e-8
            if bad.any():
                op,source=data['problem'].turn_rows(a,positions[bad]);full=np.zeros((len(op),len(x)))
                full[:,data['offsets'][i]:data['offsets'][i+1]]=op;added.append(full);bounds.append(-source)
        # A failed existing prefix gate is not a license to move its budget.
        if not added:break
        matrix=np.vstack([matrix,*added]);rhs=np.r_[rhs,*bounds]


def overlay(data,values,source_path,path):
    from PIL import Image,ImageDraw
    source=Image.open(source_path).convert('RGB');box=(365,500,655,620);scale=3;panels=[]
    for label,vector in [('Original source',None),('Original retained target',values['baseline']),('Retained + original prefix caps: diagnostic',values.get('candidate'))]:
        panel=source.crop(box).resize(((box[2]-box[0])*scale,(box[3]-box[1])*scale));draw=ImageDraw.Draw(panel)
        for i,a in enumerate(data['problem'].arcs):
            s=np.unique(np.r_[a['knots'],np.arange(0.,a['length'],.25),a['length']]);xy=data['problem'].source(a,s)
            draw.line([tuple(p)for p in (xy-[box[0],box[1]])*scale],fill=(0,220,255),width=2)
            if vector is not None:
                u=vector[data['offsets'][i]:data['offsets'][i+1]];target=xy+np.interp(s,a['nodes'],u)[:,None]*data['problem'].across
                draw.line([tuple(p)for p in (target-[box[0],box[1]])*scale],fill=(255,60,180),width=3)
        tile=Image.new('RGB',(panel.width,panel.height+30),'white');tile.paste(panel,(0,30));ImageDraw.Draw(tile).text((6,8),label,fill='black');panels.append(tile)
    canvas=Image.new('RGB',(sum(p.width for p in panels),panels[0].height),'white');left=0
    for p in panels:canvas.paste(p,(left,0));left+=p.width
    canvas.save(path)


def run(baseline_path,joint_path,source_path,output):
    baseline_path,joint_path,source_path,output=[Path(p).resolve()for p in [baseline_path,joint_path,source_path,output]]
    if output.exists():raise FileExistsError(output)
    baseline=json.loads(baseline_path.read_text());joint=json.loads(joint_path.read_text())
    evidence=check_source_binding(baseline,source_path)
    data=assemble(baseline,joint)
    code=[Path(__file__),Path(__file__).with_name('jaw_prefix_budget_1d.py'),Path(__file__).with_name('probe_source_affine_bending.py'),
          Path(__file__).with_name('source_affine_bending_reference.py'),Path(__file__).with_name('probe_jaw_retained_bend.py'),Path(__file__).with_name('jaw_curve_qp.py'),
          Path(__file__).with_name('probe_jaw_plateau.py'),Path(__file__).with_name('jaw_bilateral.py'),Path(__file__).with_name('measure_jaw_preference_start.py')]
    bindings={**evidence,**{str(p):sha(p)for p in [baseline_path,joint_path,source_path,*code]}}
    output.mkdir(parents=True)
    manifest,code_hashes=snapshot_local_code(output);bindings.update(code_hashes)
    np.savez_compressed(output/'objective.npz',hessian=data['H'],linear=data['linear'],original_linear=data['old_linear'],area=data['area'],lower=data['lower'],upper=data['upper'])
    save(output/'cap-assembly.json',{'caps':[c['report']for c in data['caps']],'original_linear_exact':bool(np.array_equal(data['linear'],data['old_linear'])),'affine_reference_used':False,'original_baseline_caps':prefix_checks(data,data['old_values'])})
    ctx=mp.get_context('spawn');process=ctx.Process(target=solver_child,args=(baseline,joint,str(output)))
    started=time.monotonic();process.start();process.join(30.);timed_out=process.is_alive()
    if timed_out:process.terminate();process.join()
    elapsed=time.monotonic()-started
    solve=json.loads((output/'solve.json').read_text())if (output/'solve.json').exists()else {'accepted':False,'solver_status':'no_complete_finite_round'}
    values={'baseline':data['old_values']}
    if (output/'candidate-nodes.npy').exists():values['candidate']=np.load(output/'candidate-nodes.npy')
    report={'schema':'retained-jaw-target-plus-original-prefix-caps-v1','new_objective':False,'new_constraints':'six original joint-field Plateau prefix SOCs, evaluated on original 1D piecewise-linear controls',
        'affine_reference_used':False,'original_hessian_unchanged':True,'original_plateau_linear_unchanged':bool(np.array_equal(data['linear'],data['old_linear'])),
        'original_area_endpoint_sign_and_retained_bend_unchanged':True,'maximum_rounds':4,'maximum_wall_seconds':30.,'wall_seconds':elapsed,'timed_out':timed_out,
        'child_exit_code':process.exitcode,'solve':solve,'accepted_target_constraints':bool(not timed_out and solve['accepted']),
        'curves':metrics(data,values),'spatial_field_computed':False,'photo_warped':False,'image_approval':False,'executable':False,'inputs_sha256':bindings,
        'execution_code_snapshot':str(manifest)}
    if 'candidate'in values:
        report['original_target_check']=checks(data,values['candidate']);report['prefix_check']=prefix_checks(data,values['candidate'])
    overlay(data,values,source_path,output/'source-target-curves.png')
    assert all(sha(p)==h for p,h in bindings.items())
    report['outputs_sha256']={p.name:sha(p)for p in output.iterdir()if p.is_file()}
    save(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'accepted':report['accepted_target_constraints'],'seconds':elapsed}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['baseline','joint','source','output']:parser.add_argument('--'+arg,type=Path,required=True)
    a=parser.parse_args();run(a.baseline,a.joint,a.source,a.output)
