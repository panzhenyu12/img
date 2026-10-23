"""Freeze one whole-interface target candidate against an existing f32 field."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from boundary_curve_qp import solve_boundary_curve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'face-source-forward-prototype-20261021'))
from forward import inverse_points


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def run(interface_path, baseline_case, profile_path, output, *,
        preserve_through_y=535., smoothing_length_px=16., neck_weight=4.,
        target_density=1., source_sign_threshold=1e-8,upper_preservation='pointwise',
        node_spacing_px=8.,preferred_audit=None,controller='source_turn_preserving',
        local_slope_change_max=.01):
    interface_path, baseline_case, profile_path, output = [Path(p).resolve()
        for p in [interface_path, baseline_case, profile_path, output]]
    if output.exists():raise FileExistsError('refusing to replace a frozen candidate')
    interface=read(interface_path);protocol=read(baseline_case.parent/'protocol.json')
    decision=read(baseline_case/'decision.json');profile=read(profile_path)
    if (interface.get('schema')!='reviewed-visible-source-interface-v1'
        or interface.get('automatic_edit_authorized') is not False
        or interface.get('mode')!='model_source_visual_review_only'
        or interface.get('visibility_status')!='unknown'):
        raise ValueError('reviewed source-only visible interface required')
    visual_path=interface_path.parent/'visual-source-review.json';visual=read(visual_path)
    if (visual.get('mode')!='model_source_visual_review_only'
        or visual.get('automatic_edit_authorized') is not False
        or visual.get('visibility_status')!='unknown'
        or visual.get('source_image_sha256')!=interface['source_image_sha256']
        or visual.get('hashes',{}).get(str(interface_path))!=sha(interface_path)):
        raise ValueError('source interface is not the independently viewed and frozen source evidence')
    if not decision['final_geometry']['passed_strict'] or decision['amount'] != profile['amount']:
        raise ValueError('passing field at the identical signed profile amount required')
    historical_profile_hash=protocol['inputs_sha256'].get(str(Path(protocol['frozen_target_profile']).resolve()))
    if historical_profile_hash != sha(profile_path):
        raise ValueError('preferred field must use the same frozen jaw profile')
    if profile['source_geometry']!=protocol['source_geometry']:
        raise ValueError('profile source geometry differs from preferred field protocol')
    if preferred_audit is None:raise ValueError('independently frozen preferred field audit required')
    preferred_audit=Path(preferred_audit).resolve();audit=read(preferred_audit)
    field_paths=[baseline_case/'inverse-field.npy',baseline_case/'decision.json',baseline_case.parent/'protocol.json']
    if any(audit['inputs_sha256'].get(str(p))!=sha(p) for p in field_paths):
        raise ValueError('preferred field/protocol/decision differs from independently frozen audit')
    prior=interface['prior_review_consistency']
    neck_path=Path(prior['old_neck_source_path'])
    if sha(neck_path)!=prior['old_neck_source_sha256']:
        raise ValueError('prior source-only neck review changed')
    neck_review=read(neck_path)
    if (neck_review.get('schema')!='reviewed-source-neck-boundary-candidate-v1'
        or neck_review.get('source_image_sha256')!=interface['source_image_sha256']
        or neck_review.get('automatic_edit_authorized') is not False
        or len(neck_review['segments'])!=len(interface['segments'])):
        raise ValueError('independent source-only low-boundary preference evidence required')
    paths=[interface_path,baseline_case/'inverse-field.npy',baseline_case/'decision.json',
        baseline_case.parent/'protocol.json',profile_path,
        interface_path.parent/'visual-source-review.json',neck_path,preferred_audit]
    hashes={**interface['inputs_sha256'],**neck_review['input_hashes'],**visual['hashes']}
    hashes.update({str(p):sha(p) for p in paths})
    source_path=Path(interface['source_image'])
    if sha(source_path)!=interface['source_image_sha256']:
        raise ValueError('source image identity changed')
    if protocol['inputs_sha256'].get(str(source_path))!=sha(source_path):
        raise ValueError('preferred field belongs to a different source image')
    if any(sha(p)!=digest for p,digest in hashes.items()):
        raise ValueError('source evidence changed')
    field=np.load(baseline_case/'inverse-field.npy').astype(float)
    if field.shape != (protocol['crop'][3],protocol['crop'][2],2) or not np.isfinite(field).all():
        raise ValueError('preferred field shape must match frozen crop exactly')
    offset=np.asarray(protocol['crop'][:2]);frame=protocol['source_geometry']['frame']
    across,down=[np.asarray(frame[k]) for k in ['across','down']]
    output.mkdir(parents=True)
    for p in [Path(__file__).resolve(),Path(__file__).with_name('boundary_curve_qp.py').resolve()]:
        target=output/p.name;shutil.copy2(p,target);hashes[str(target)]=sha(target)
    cases=[];sources=[];targets=[];weights=[]
    for segment,neck_segment in zip(interface['segments'],neck_review['segments']):
        source=np.asarray(segment['source_points'])
        low_source=np.asarray(neck_segment['source_points'])
        if not np.allclose(np.interp(low_source[:,1],source[:,1],source[:,0]),low_source[:,0],rtol=0.,atol=1e-10):
            raise ValueError('prior low-boundary review does not coincide with this visible interface')
        actual,closure=inverse_points(field,source-offset);actual+=offset
        preferred=(actual-source)@across
        preferred_down=(actual-source)@down
        def height(y):
            return float(np.interp(y,source[:,1],source@down))
        result=solve_boundary_curve(source,preferred,frame,
            preserve_through_down=height(preserve_through_y),neck_start_down=height(float(low_source[0,1])),
            neck_end_down=height(float(low_source[-1,1])),smoothing_length_px=smoothing_length_px,
            neck_weight=neck_weight,target_density=target_density,
            source_sign_threshold=source_sign_threshold,upper_preservation=upper_preservation,
            node_spacing_px=node_spacing_px,controller=controller,
            local_slope_change_max=local_slope_change_max)
        result['report'].update({'source_segment_id':segment['id'],
            'preferred_forward_closure':closure,
            'preferred_discarded_down_max_px':float(abs(preferred_down).max()),
            'preserve_through_source_y':preserve_through_y,
            'low_boundary_preference_review':None if controller=='transport_only_motion_fairing' else str(neck_path),
            'legacy_low_boundary_identity_review':str(neck_path)})
        cases.append(result['report'])
        np.savez_compressed(output/(segment['id']+'-source-preferred.npz'),
            source=source,preferred_across=preferred,preferred_down=preferred_down)
        if result['report']['accepted_numerically']:
            sources.append(result['source']);targets.append(result['target']);weights.append(result['weights'])
            np.savez_compressed(output/(segment['id']+'-candidate.npz'),
                source=result['source'],target=result['target'],weights=result['weights'],
                arclength_px=result['arclength_px'],preferred_across=result['preferred_across'])
    report={'controller':controller+'-whole-visible-interface-v1',
        'accepted_numerically':all(case['accepted_numerically'] for case in cases),
        'aesthetic_approval':False,'source_interface':str(interface_path),
        'preferred_field':str(baseline_case/'inverse-field.npy'),
        'anatomical_role_status':interface['anatomical_role_status'],
        'no_new_jaw_points':True,'post_solve_field_taper':False,
        'source_2px_band_is_inspection_only':True,'curves':cases}
    if controller=='local_chord_motion_fairing':
        report['local_slope_change_max']=float(local_slope_change_max)
    artifact={'schema':'source-boundary-motion-targets-v1',
        'source_image_sha256':interface['source_image_sha256'],
        'source_size':interface['source_size'],'amount':decision['amount'],
        'jaw_profile_sha256':sha(profile_path),'inputs_sha256':hashes,'report':report}
    if profile.get('diagnostic_only'):
        # Boundary-target convergence cannot promote a separately supplied,
        # rejected jaw optimization request or grant photographic approval.
        report.update(jaw_target_solver_status=profile.get('solver_status'),
            jaw_target_optimization_accepted=profile.get('accepted_target_constraints') is True,
            diagnostic_only=True,image_approval=False,executable=False)
        artifact.update(diagnostic_only=True,image_approval=False,executable=False)
    if report['accepted_numerically']:
        artifact.update({'source':np.vstack(sources).tolist(),'target':np.vstack(targets).tolist(),
                         'weights':np.concatenate(weights).tolist()})
        save(output/'boundary-motion.json',artifact)
    else:save(output/'rejected-candidate.json',artifact)
    if any(sha(p)!=digest for p,digest in hashes.items()):raise ValueError('inputs changed during solve')
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['interface','baseline','profile','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preserve-through-y',type=float,default=535.)
    p.add_argument('--smoothing-length',type=float,default=16.)
    p.add_argument('--neck-weight',type=float,default=4.)
    p.add_argument('--target-density',type=float,default=1.)
    p.add_argument('--source-sign-threshold',type=float,default=1e-8)
    p.add_argument('--upper-preservation',choices=['pointwise','integral_and_anchor'],default='pointwise')
    p.add_argument('--node-spacing',type=float,default=8.)
    p.add_argument('--local-slope-change-max',type=float,default=.01,
        help='pointwise correction chord-slope difference limit; not a true angle bound')
    p.add_argument('--preferred-audit',type=Path,required=True)
    p.add_argument('--controller',choices=['source_turn_preserving','gradient_bounded_motion_fairing',
        'chord_bounded_motion_fairing','local_chord_motion_fairing',
        'transport_only_motion_fairing'],default='source_turn_preserving')
    a=p.parse_args()
    result=run(a.interface,a.baseline,a.profile,a.output,
        preserve_through_y=a.preserve_through_y,smoothing_length_px=a.smoothing_length,
        neck_weight=a.neck_weight,target_density=a.target_density,
        source_sign_threshold=a.source_sign_threshold,upper_preservation=a.upper_preservation,
        node_spacing_px=a.node_spacing,preferred_audit=a.preferred_audit,controller=a.controller,
        local_slope_change_max=a.local_slope_change_max)
    print(json.dumps(result,indent=2))
