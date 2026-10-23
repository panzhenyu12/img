"""One same-QP numerical experiment: normalize each original prefix SOC.

Every entire SOC is divided by its original positive radius. No data weight,
physical gate, objective or reference changes. The original failed run remains
immutable, and its wall time is debited from the same 30 second budget.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import clarabel
from scipy.sparse import diags,save_npz

from probe_jaw_prefix_budget_1d import assemble,solve_round,checks,prefix_checks,metrics,overlay,sha,save


def child(baseline,joint,out,remaining):
    out=Path(out);data=assemble(baseline,joint);original_solver=clarabel.DefaultSolver;capture={}
    def normalized(P,q,A,b,cones,settings):
        scale=np.ones(len(b));offset=0;original_radii=[]
        for cone in cones:
            if isinstance(cone,clarabel.SecondOrderConeT):
                radius=float(b[offset])
                if radius<=0.:raise ValueError('normalization requires unchanged positive original cap')
                scale[offset:offset+cone.dim]=1./radius;original_radii.append(radius)
            offset+=cone.dim
        if len(original_radii)!=6:raise ValueError('only the original six SOCs are normalized')
        save_npz(out/'unnormalized-P.npz',P);save_npz(out/'unnormalized-A.npz',A)
        np.savez_compressed(out/'unnormalized-vectors.npz',q=q,b=b,row_positive_scale=scale)
        capture['original_soc_radii']=original_radii
        solver=original_solver(P,q,(diags(scale)@A).tocsc(),scale*b,cones,settings)
        class Observed:
            def solve(self):
                solved=solver.solve()
                capture.update({key:float(getattr(solved,key)) for key in ['r_prim','r_dual','obj_val','obj_val_dual','solve_time']})
                capture['absolute_gap']=abs(capture['obj_val']-capture['obj_val_dual']);capture['iterations']=int(solved.iterations)
                return solved
        return Observed()
    started=time.monotonic()
    with patch.object(clarabel,'DefaultSolver',normalized):x,status=solve_round(data,data['matrix'],data['rhs'],remaining)
    report={'solver_status':status,'solver_seconds':time.monotonic()-started,'solver_diagnostics':capture,'normalization':'entire original SOC divided by its existing positive sqrt(cap); P/q and all halfspaces unchanged','accepted':False}
    if x is not None:
        np.save(out/'candidate-nodes.npy',x);report['original_check']=checks(data,x);report['prefix_check']=prefix_checks(data,x)
        report['accepted']=status=='Solved' and report['original_check']['all_original_target_constraints_pass'] and report['prefix_check']['accepted']
    save(out/'solve.json',report)


def run(reference_path,output):
    reference_path,output=Path(reference_path).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    first=json.loads(reference_path.read_text());bindings=first['inputs_sha256']
    baseline_path=next(Path(p)for p in bindings if p.endswith('jaw-retained-bend-probe-v1/user-plus100/candidate-profile.json'))
    joint_path=next(Path(p)for p in bindings if p.endswith('joint-jaw-visible-interface-source-probe-v13/user-plus100/report.json'))
    source_path=next(Path(p)for p in bindings if p.endswith('user-jaw-source-v1/source.png'))
    baseline,joint=[json.loads(p.read_text())for p in [baseline_path,joint_path]]
    if any(sha(p)!=bindings[str(p)]for p in [baseline_path,joint_path,source_path]):raise ValueError('same frozen original inputs required')
    remaining=30.-first['wall_seconds']
    if remaining<=0.:raise ValueError('original experiment budget exhausted')
    output.mkdir(parents=True);snapshot=output/'executed-code';snapshot.mkdir()
    sources=[Path(__file__),Path(__file__).with_name('probe_jaw_prefix_budget_1d.py'),Path(__file__).with_name('jaw_prefix_budget_1d.py')]
    for p in sources:(snapshot/p.name).write_bytes(p.read_bytes())
    started=time.monotonic();ctx=mp.get_context('spawn');process=ctx.Process(target=child,args=(baseline,joint,str(output),remaining));process.start();process.join(remaining)
    timedout=process.is_alive()
    if timedout:process.terminate();process.join()
    elapsed=time.monotonic()-started;solve=json.loads((output/'solve.json').read_text())if (output/'solve.json').exists()else {'accepted':False,'solver_status':'no_complete_finite_round'}
    data=assemble(baseline,joint);values={'baseline':data['old_values']}
    if (output/'candidate-nodes.npy').exists():values['candidate']=np.load(output/'candidate-nodes.npy')
    report={'schema':'same-retained-prefix-qp-soc-unit-normalization-v1','first_attempt':str(reference_path),'first_attempt_sha256':sha(reference_path),
        'prior_wall_seconds':first['wall_seconds'],'polish_wall_seconds':elapsed,'combined_wall_seconds':elapsed+first['wall_seconds'],'maximum_combined_seconds':30.,
        'timed_out':timedout,'child_exit_code':process.exitcode,'solve':solve,'accepted_target_constraints':bool(not timedout and solve['accepted']),
        'original_objective_and_physical_constraints_unchanged':True,'affine_reference_used':False,'curves':metrics(data,values),
        'spatial_field_computed':False,'photo_warped':False,'image_approval':False,'executable':False,
        'inputs_sha256':{str(p):sha(p)for p in [reference_path,baseline_path,joint_path,source_path,*sources]}}
    overlay(data,values,source_path,output/'source-target-curves.png')
    report['outputs_sha256']={p.name:sha(p)for p in output.iterdir()if p.is_file()};save(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'accepted':report['accepted_target_constraints'],'combined_seconds':report['combined_wall_seconds'],'solver':solve.get('solver_diagnostics')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.reference,a.output)
