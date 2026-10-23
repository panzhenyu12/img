"""Freeze a rejected optimizer's physically checked target for offline diagnosis.

This never promotes AlmostSolved, never grants executable/image approval, and
never changes any original numerical or physical gate. Loading the profile
checks only the existing source/node protocol; it is not optimizer acceptance.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from probe_jaw_prefix_budget_1d import assemble,checks,prefix_checks,sha,save
from jaw_target_profile import load_profile


def export(first_path,second_path,output):
    first_path,second_path,output=[Path(p).resolve()for p in [first_path,second_path,output]]
    if output.exists():raise FileExistsError(output)
    first=json.loads(first_path.read_text());second=json.loads(second_path.read_text())
    if second['first_attempt']!=str(first_path) or second['first_attempt_sha256']!=sha(first_path):
        raise ValueError('the exact two original attempts must be bound')
    if first['solve']['solver_status']!='AlmostSolved' or second['solve']['solver_status']!='AlmostSolved' or first['accepted_target_constraints'] or second['accepted_target_constraints']:
        raise ValueError('this export retains the two failed AlmostSolved diagnostic attempts')
    bound=second['inputs_sha256']
    baseline_path=next(Path(p)for p in bound if p.endswith('jaw-retained-bend-probe-v1/user-plus100/candidate-profile.json'))
    joint_path=next(Path(p)for p in bound if p.endswith('joint-jaw-visible-interface-source-probe-v13/user-plus100/report.json'))
    baseline,joint=[json.loads(p.read_text())for p in [baseline_path,joint_path]]
    if any(sha(p)!=bound[str(p)]for p in [baseline_path,joint_path]):raise ValueError('original baseline/joint source evidence changed')
    first_nodes=first_path.parent/'candidate-nodes.npy';nodes=second_path.parent/'candidate-nodes.npy'
    if sha(first_nodes)!=first['outputs_sha256']['candidate-nodes.npy'] or sha(nodes)!=second['outputs_sha256']['candidate-nodes.npy']:
        raise ValueError('a frozen finite target state changed')
    data=assemble(baseline,joint);x=np.load(nodes)
    if x.shape!=data['old_values'].shape or not np.isfinite(x).all():raise ValueError('complete finite original node vector required')
    old=checks(data,x);prefix=prefix_checks(data,x)
    if not old['all_original_target_constraints_pass'] or not prefix['accepted']:
        raise ValueError('diagnostic target must pass every unchanged actual-u physical constraint')
    keep=['schema','amount','max_radius_fraction','area_reference','required_signed_area_delta_px2','source_geometry_sha256',
          'source_geometry','geometry_hash_encoding','across_axis','retained_bend_ratio','parameters','source_protocol',
          'source_protocol_sha256','source_evidence_sha256']
    profile={key:copy.deepcopy(baseline[key])for key in keep}
    profile['controller']='diagnostic-retained-jaw-original-prefix-caps-almost-solved-v1'
    profile['profiles']=[]
    for i,a in enumerate(data['problem'].arcs):
        original=next(row for row in baseline['profiles']if row['arc_id']==a['id']);row=copy.deepcopy(original)
        positions=np.unique(np.r_[a['nodes'],a['knots']]);u=x[data['offsets'][i]:data['offsets'][i+1]]
        row['arclength_px']=positions.tolist();row['across_displacement_px']=np.interp(positions,a['nodes'],u).tolist()
        if row['across_displacement_px'][0]!=0. or row['across_displacement_px'][-1]!=0.:raise ValueError('hard-zero endpoint changed')
        profile['profiles'].append(row)
    profile.update(accepted_target_constraints=False,all_actual_physical_constraints_pass=True,
        solver_status='AlmostSolved',executable=False,image_approval=False,diagnostic_only=True,
        amount_already_baked=True,candidate_signed_area_delta_px2=old['area_px2'])
    bindings={str(p):sha(p)for p in [first_path,second_path,first_nodes,nodes,baseline_path,joint_path,Path(__file__),
        Path(__file__).with_name('probe_jaw_prefix_budget_1d.py'),Path(__file__).with_name('jaw_prefix_budget_1d.py')]}
    profile['inputs_sha256']={**baseline.get('source_evidence_sha256',{}),**bindings}
    profile['report']={'accepted_numerically':False,'accepted_target_constraints':False,'all_actual_physical_constraints_pass':True,
        'solver_status':'AlmostSolved','executable':False,'aesthetic_approval':False,'image_approval':False,
        'purpose':'offline finite-shape request comparison only; optimization convergence remains rejected',
        'original_target_constraints':old,'original_prefix_constraints':prefix,
        'solver_diagnostics':copy.deepcopy(second['solve']['solver_diagnostics']),
        'profile_load_scope':'source identity, signed amount and original knot protocol only; no optimizer or image approval',
        'original_acceptance_gates_changed':False,'first_attempt':str(first_path),'second_attempt':str(second_path)}
    save(output,profile)
    loaded=load_profile(output,profile['source_geometry'],100.)
    if loaded!=profile:raise ValueError('profile source protocol validation changed data')
    if not all(sha(p)==h for p,h in bindings.items()):raise ValueError('bound original evidence changed during export')
    save(output.with_name(output.stem+'-validation.json'),{'profile':str(output),'profile_sha256':sha(output),
        'load_profile_source_and_node_protocol_passed':True,'optimizer_accepted':False,'solver_status':'AlmostSolved',
        'all_actual_physical_constraints_pass':True,'image_approval':False,'executable':False,'inputs_sha256':bindings})
    print(json.dumps({'profile':str(output),'sha256':sha(output),'source_protocol_passed':True,'solver_status':'AlmostSolved','optimizer_accepted':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['first','second','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();export(a.first,a.second,a.output)
