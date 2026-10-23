"""One bounded 1D jaw experiment: center only D2 on source-affine width.

The original retained-bend QP is reconstructed with identical H, Plateau
position preference, area, signs and hard-zero endpoints. Only its linear
bending term changes. No spatial field, photo warp or application edit occurs.
"""
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
from scipy.linalg import block_diag

from probe_jaw_retained_bend import RetainedProblem
from probe_jaw_plateau import OriginalProblem, PlateauProblem
from source_affine_bending_reference import source_affine_reference
from jaw_bilateral import build_bilateral_coupling
from jaw_curve_qp import _basis, _solve_precise_qp
from interface_bend_oracle import chord_metrics
from measure_jaw_preference_start import measure_preference_start
from export_jaw_curve_profile import check_protocol
from jaw_target_profile import _geometry_sha


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_source_binding(baseline,source_path):
    """Bind an overlay to the original image role, not any mask in evidence."""
    if not baseline.get('source_protocol') or not baseline.get('source_protocol_sha256'):
        raise ValueError('frozen source protocol binding required')
    protocol_path=Path(baseline['source_protocol']).resolve()
    if sha(protocol_path)!=baseline['source_protocol_sha256']:
        raise ValueError('frozen source protocol changed')
    protocol,evidence=check_protocol(protocol_path)
    evidence={str(Path(path).resolve()):digest for path,digest in evidence.items()}
    geometry_hash=_geometry_sha(baseline.get('source_geometry'))
    if (baseline.get('source_geometry_sha256')!=geometry_hash
            or _geometry_sha(protocol.get('source_geometry'))!=geometry_hash):
        raise ValueError('source protocol geometry differs from baseline')
    source_path=Path(source_path).resolve()
    digest=sha(source_path)
    if evidence.get(str(source_path))!=digest:
        raise ValueError('source image is not bound to the frozen protocol')
    # Portable protocols bind preparation reports rather than duplicating their
    # image role. Only a hash-bound report with the same geometry can supply it.
    declarations=[]
    for name in evidence:
        if Path(name).name!='report.json':continue
        report=json.loads(Path(name).read_text())
        if (report.get('source_image') and report.get('source_geometry') is not None
                and _geometry_sha(report['source_geometry'])==geometry_hash):
            declared=str(Path(report['source_image']).resolve())
            report_hashes={str(Path(path).resolve()):digest for path,digest in report.get('hashes',{}).items()}
            if declared in evidence and report_hashes.get(declared)==evidence[declared]:
                declarations.append(declared)
    if not declarations or set(declarations)!={str(source_path)}:
        raise ValueError('source image differs from the frozen preparation image role')
    return {str(protocol_path):sha(protocol_path),**evidence}


def snapshot_local_code(output):
    """Use the portable experiment's loaded-local-module snapshot convention."""
    root=Path(__file__).resolve().parents[2]
    paths={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value).resolve()
            if path.suffix=='.py' and any(path.is_relative_to(root/folder) for folder in ['experiments','scripts']):
                paths.add(path)
    folder=Path(output)/'executed-code';folder.mkdir()
    records=[];hashes={}
    for path in sorted(paths):
        target=folder/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(path);shutil.copy2(path,target)
        if sha(target)!=digest:raise ValueError('source code changed during snapshot')
        records.append({'source':str(path),'snapshot':str(target),'sha256':digest})
        hashes[str(path)]=digest;hashes[str(target)]=digest
    manifest=folder/'manifest.json'
    save(manifest,{'scope':'Loaded local Python modules from experiments/scripts, including transitive dependencies; external libraries are version-recorded, not copied.',
                   'files':records})
    hashes[str(manifest)]=sha(manifest)
    return manifest,hashes


def assemble(baseline):
    parameters=baseline['parameters']
    if parameters!={'amount':100.,'max_radius_fraction':.13,'node_spacing_px':2.,
                    'smoothing_length_px':12.,'window_radius_px':16.,'bilateral_weight':1.}:
        raise ValueError('the experiment is frozen to the original retained +100 parameters')
    p=RetainedProblem(baseline['source_geometry'],2.,16.)
    ref=source_affine_reference(p,100.,.13,12.)
    offsets=np.r_[0,np.cumsum([len(a['nodes'])for a in p.arcs])];size=int(offsets[-1])
    weights=[];betas=[];hs=[];rows=[];rhs=[];old_values=[]
    for i,(a,c) in enumerate(zip(p.arcs,ref['curves'])):
        n=len(a['nodes']);h=a['length']/(n-1);w=np.full(n,h);w[[0,-1]]*=.5
        d2=np.diff(np.eye(n),n=2,axis=0)/h**2
        original_bend=12.**4*h*(d2.T@d2)
        np.testing.assert_array_equal(c['hessian'],original_bend)
        hs.append(np.diag(w)+original_bend);weights.append(w);betas.append(p.beta(a,a['nodes'],100.,.13))
        regions=p.convex_regions(a);centers=np.unique(np.r_[regions.ravel(),regions.mean(axis=1)])
        local,source=p.turn_rows(a,centers);full=np.zeros((len(centers),size));full[:,offsets[i]:offsets[i+1]]=local
        rows.append(full);rhs.append(-source)
        row=next(row for row in baseline['profiles']if row['arc_id']==a['id'])
        old_values.append(np.interp(a['nodes'],row['arclength_px'],row['across_displacement_px']))
    beta=np.concatenate(betas);H=block_diag(*hs)+build_bilateral_coupling(p,_basis).hessian
    old_linear=-np.concatenate(weights)*beta;new_linear=old_linear+ref['linear_shift']
    lower=np.where(beta>0.,0.,-np.inf);upper=np.where(beta<0.,0.,np.inf)
    lower[beta==0.]=upper[beta==0.]=0.
    for i in range(len(p.arcs)):
        lower[offsets[i]]=upper[offsets[i]]=lower[offsets[i+1]-1]=upper[offsets[i+1]-1]=0.
    area=np.concatenate([p.area_row(a)for a in p.arcs]);wanted=p.beta_area(100.,.13);norm=np.linalg.norm(area)
    return dict(problem=p,reference=ref,offsets=offsets,H=H,old_linear=old_linear,linear=new_linear,
        old_values=np.concatenate(old_values),matrix=np.vstack(rows),rhs=np.concatenate(rhs),lower=lower,upper=upper,
        area=area,wanted=wanted,equality=area[None,:]/norm,equality_rhs=np.array([wanted/norm]),
        normalization=sum(a['length']for a in p.arcs))


def checks(data,x):
    p=data['problem'];rows=[]
    for i,a in enumerate(p.arcs):
        u=x[data['offsets'][i]:data['offsets'][i+1]];positions,margins,_=p.continuous_minima(a,u)
        # OriginalProblem directly supplies target numerator, independent of
        # the RetainedProblem override used to find each margin minimum.
        target,_=OriginalProblem.numerator(p,a,positions,u)
        rows.append({'id':a['id'],'retained_bend_margin_min_px2':float(margins.min(initial=np.inf)),
            'retained_bend_violations':int(np.count_nonzero(margins < -1e-8)),
            'target_numerator_min_at_margin_witnesses_px2':float(target.min(initial=np.inf))})
    area=float(data['area']@x);fixed=data['lower']==data['upper']
    endpoint=float(abs(x[fixed]-data['lower'][fixed]).max(initial=0.))
    sign=float(np.maximum(np.maximum(data['lower']-x,x-data['upper']),0.).max(initial=0.))
    accepted=abs(area-data['wanted'])<=max(1e-8,abs(data['wanted'])*1e-10) and endpoint<=1e-12 and sign<=1e-10 and not any(r['retained_bend_violations']for r in rows)
    return {'area_px2':area,'area_error_px2':area-data['wanted'],'hard_zero_max_px':endpoint,
        'sign_violation_px':sign,'curves':rows,'all_original_target_constraints_pass':bool(accepted)}


def solver_child(baseline,out):
    """Killed by the parent after 30 seconds; no restart or weight scan."""
    out=Path(out);data=assemble(baseline);started=time.monotonic();rows=data['matrix'];rhs=data['rhs'];rounds=[];x=None;status=None
    for iteration in range(8):
        if time.monotonic()-started>=30.:status='budget_exhausted';break
        solved=_solve_precise_qp(data['H'],data['linear'],rows,rhs,data['equality'],data['equality_rhs'],
            data['lower'],data['upper'],data['normalization'])
        x=np.asarray(solved.x);status=solved.message
        if x.shape!=data['linear'].shape or not np.isfinite(x).all():status='nonfinite';x=None;break
        checked=checks(data,x);rounds.append({'round':iteration,'solver_status':status,'check':checked})
        np.save(out/'candidate-nodes.npy',x)
        save(out/'solve.json',{'solver_status':status,'rounds':rounds,'seconds':time.monotonic()-started,
             'original_constraints_accepted':bool(solved.success and checked['all_original_target_constraints_pass'])})
        if not solved.success or checked['all_original_target_constraints_pass']:break
        added=[];bounds=[]
        for i,a in enumerate(data['problem'].arcs):
            local=x[data['offsets'][i]:data['offsets'][i+1]]
            positions,margin,_=data['problem'].continuous_minima(a,local);bad=margin < -1e-8
            if bad.any():
                op,source=data['problem'].turn_rows(a,positions[bad]);full=np.zeros((len(op),len(x)))
                full[:,data['offsets'][i]:data['offsets'][i+1]]=op;added.append(full);bounds.append(-source)
        if not added:break
        rows=np.vstack([rows,*added]);rhs=np.r_[rhs,*bounds]


def metrics(data,values):
    p=data['problem'];result=[]
    for i,a in enumerate(p.arcs):
        start,stop=data['offsets'][i:i+2];row={'id':a['id'],'turns':[],'prefix_energies':[],'motion':[]}
        curves={name:np.asarray(x[start:stop])for name,x in values.items()}
        for center,radius in [(18.75,2.),(20.875,6.),(26.875,2.),(36.75,2.)]:
            query=np.array([center-radius,center,center+radius]);source=p.source(a,query)
            angles={'source':float(chord_metrics(source[None],[p.across,p.down])['turn_degrees'][0])}
            for name,u in curves.items():
                target=source+np.interp(query,a['nodes'],u)[:,None]*p.across
                angles[name]=float(chord_metrics(target[None],[p.across,p.down])['turn_degrees'][0])
            row['turns'].append({'s':center,'radius_px':radius,'angles_degrees':angles})
        for end in [2.,6.,12.]:
            budget=measure_preference_start(p,a,100.,.13,end,'equal_area_plateau')['unweighted_energy_px3']*12.**2/end**2
            cuts=np.unique(np.r_[0.,a['nodes'][(a['nodes']>0)&(a['nodes']<end)],end]);entry={'end_px':end,'original_plateau_budget_px3':budget}
            for name,u in curves.items():
                q=np.interp(cuts,a['nodes'],u);energy=float(12.**2*np.sum(np.diff(q)**2/np.diff(cuts)))
                entry[name]={'energy_px3':energy,'budget_ratio':energy/budget,'passes':bool(energy<=budget)}
            row['prefix_energies'].append(entry)
        low,high=12.,28.;cuts=np.unique(np.r_[low,a['nodes'][(a['nodes']>low)&(a['nodes']<high)],a['knots'][(a['knots']>low)&(a['knots']<high)],high]);g,w=np.polynomial.legendre.leggauss(8);lo,hi=cuts[:-1],cuts[1:];s=((lo+hi)[:,None]/2+(hi-lo)[:,None]*g/2).ravel();weights=((hi-lo)[:,None]*w/2).ravel();beta=p.beta(a,s,100.,.13)
        pref_mass=float(weights@abs(beta));row['motion_integral_12_28']={'plateau_px2':pref_mass}
        for name,u in curves.items():
            sampled=np.interp(s,a['nodes'],u);row['motion_integral_12_28'][name]={'px2':float(weights@abs(sampled)),'ratio_to_plateau':float(weights@abs(sampled)/pref_mass),'rms_error_to_plateau_px':float(np.sqrt(weights@(sampled-beta)**2/16.))}
        for s in [0.,2.,6.,12.,16.,20.,24.,28.,40.,60.,80.,100.,120.]:
            row['motion'].append({'s':s,'plateau':float(p.beta(a,np.array([s]),100.,.13)[0]),**{name:float(np.interp(s,a['nodes'],u))for name,u in curves.items()}})
        result.append(row)
    return result


def overlay(data,values,source_path,path):
    from PIL import Image,ImageDraw
    source=Image.open(source_path).convert('RGB');box=(365,500,655,620);scale=3;panels=[]
    for label,vector in [('Original source',None),('Original retained target',values['baseline']),('Affine curvature reference: diagnostic',values.get('candidate'))]:
        panel=source.crop(box).resize(((box[2]-box[0])*scale,(box[3]-box[1])*scale));draw=ImageDraw.Draw(panel)
        for i,a in enumerate(data['problem'].arcs):
            s=np.unique(np.r_[a['knots'],np.arange(0.,a['length'],.25),a['length']]);xy=data['problem'].source(a,s)
            src=(xy-[box[0],box[1]])*scale;draw.line([tuple(p)for p in src],fill=(0,220,255),width=2)
            if vector is not None:
                u=vector[data['offsets'][i]:data['offsets'][i+1]];target=xy+np.interp(s,a['nodes'],u)[:,None]*data['problem'].across
                dst=(target-[box[0],box[1]])*scale;draw.line([tuple(p)for p in dst],fill=(255,60,180),width=3)
        tile=Image.new('RGB',(panel.width,panel.height+30),'white');tile.paste(panel,(0,30));ImageDraw.Draw(tile).text((6,8),label,fill='black');panels.append(tile)
    canvas=Image.new('RGB',(sum(p.width for p in panels),panels[0].height),'white');left=0
    for p in panels:canvas.paste(p,(left,0));left+=p.width
    canvas.save(path)


def run(baseline_path,source_path,output):
    baseline_path=Path(baseline_path).resolve();source_path=Path(source_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    baseline=json.loads(baseline_path.read_text())
    if baseline['controller']!='probe-plateau-retain-half-source-bend-v1' or not baseline['report']['accepted_numerically']:
        raise ValueError('frozen original retained target solution required')
    bindings={str(baseline_path):sha(baseline_path),**check_source_binding(baseline,source_path)}
    data=assemble(baseline);x0=data['old_values'];reference=data['reference'];preflight={'reference':reference['report'],'baseline_original_constraints':checks(data,x0),'curves':[]}
    for i,(a,c) in enumerate(zip(data['problem'].arcs,reference['curves'])):
        u=x0[data['offsets'][i]:data['offsets'][i+1]];d2=c['d2'];r=c['reference_displacement'];preflight['curves'].append({'id':a['id'],
            'old_bending_energy_px3':float(c['weight']*np.sum((d2@u)**2)),
            'reference_bending_energy_px3':float(c['weight']*np.sum((d2@(u-r))**2)),
            'reference_endpoint_values_px':[float(r[0]),float(r[-1])],
            'actual_hard_zero_values_px':[float(u[0]),float(u[-1])],
            'reference_curvature_min_max_per_px':[float((d2@r).min()),float((d2@r).max())]})
    output.mkdir(parents=True)
    code_manifest,code_hashes=snapshot_local_code(output);bindings.update(code_hashes)
    save(output/'preflight.json',preflight)
    np.savez_compressed(output/'objective.npz',hessian=data['H'],old_linear=data['old_linear'],new_linear=data['linear'],linear_shift=reference['linear_shift'],
        unchanged_area=data['area'],unchanged_lower=data['lower'],unchanged_upper=data['upper'])
    # A separate child enforces a hard wall-clock ceiling even inside native code.
    ctx=mp.get_context('spawn');process=ctx.Process(target=solver_child,args=(baseline,str(output)))
    started=time.monotonic();process.start();process.join(30.)
    timed_out=process.is_alive()
    if timed_out:process.terminate();process.join()
    elapsed=time.monotonic()-started
    solve=json.loads((output/'solve.json').read_text())if (output/'solve.json').exists()else {'original_constraints_accepted':False,'solver_status':'no_complete_finite_round'}
    values={'baseline':x0}
    if (output/'candidate-nodes.npy').exists():values['candidate']=np.load(output/'candidate-nodes.npy')
    report={'schema':'target-only-source-affine-bending-reference-experiment-v1','new_objective_experiment':True,'reference':reference['report'],
        'same_original_hessian':True,'same_plateau_position_preference':True,'same_hard_constraints':True,
        'prefix_budgets_scope':'Measured unchanged joint-field prefix budgets; these were not constraints in the original retained 1D baseline and are not silently introduced into this QP.',
        'max_wall_seconds':30.,'child_wall_seconds':elapsed,'timed_out':timed_out,'child_exit_code':process.exitcode,'solve':solve,
        'target_original_constraints_accepted':bool(not timed_out and solve['original_constraints_accepted']),
        'spatial_field_computed':False,'photo_warped':False,'image_approval':False,'executable':False,
        'curves':metrics(data,values),'inputs_sha256':bindings,'preflight':preflight,
        'executed_code_manifest':str(code_manifest),'executed_code_manifest_sha256':sha(code_manifest),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'clarabel':version('clarabel')}}
    overlay(data,values,source_path,output/'source-target-curves.png')
    if 'candidate'in values:
        report['diagnostic_candidate_original_constraints']=checks(data,values['candidate'])
        report['prefix_budgets_all_pass']=all(x['candidate']['passes']for r in report['curves']for x in r['prefix_energies'])
        report['joint_field_admissible']=False
        report['quality_status']='unapproved target-only result; original joint-field prefix failure remains rejection' if not report['prefix_budgets_all_pass']else 'unapproved; only target-only diagnostics available'
    if any(sha(p)!=h for p,h in bindings.items()):
        raise ValueError('bound source evidence or executed code changed during probe')
    report['outputs_sha256']={p.name:sha(p)for p in output.iterdir()if p.is_file()}
    save(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'original_constraints_accepted':report['target_original_constraints_accepted'],'prefix_budgets_all_pass':report.get('prefix_budgets_all_pass'),'wall_seconds':elapsed}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['baseline','source','output']:parser.add_argument('--'+arg,type=Path,required=True)
    a=parser.parse_args();run(a.baseline,a.source,a.output)
