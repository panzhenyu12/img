"""Run a reviewed new-photo jaw experiment without old native trace inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from portable_jaw import jaw_geometry, pixels, fixed_support, request_targets
from portable_field import build_problem, solve_problem
from portable_result_gate import evaluate_result
from evaluate import inverse_points, inspect, summarize
from resampling import render_frozen_field
from jaw_collocation import sample_jaw_targets, independent_jaw_probes


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')


def snapshot_local_code(output):
    """Retain the loaded local Python implementation, not mutable paths alone."""
    root=Path(__file__).resolve().parents[2]
    paths={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value).resolve()
            if path.suffix=='.py' and any(path.is_relative_to(root/folder) for folder in ['experiments','scripts']):
                paths.add(path)
    folder=output/'executed-code';folder.mkdir(exist_ok=False)
    files=[];hashes={}
    for path in sorted(paths):
        relative=path.relative_to(root);target=folder/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(path);shutil.copy2(path,target)
        if sha(target)!=digest:raise ValueError('algorithm source changed while taking its execution snapshot')
        files.append({'source':str(path),'snapshot':str(relative),'sha256':digest})
        hashes[str(path)]=digest;hashes[str(target.resolve())]=digest
    manifest=folder/'snapshot-manifest.json'
    save(manifest,{'scope':'Local Python modules loaded from experiments/scripts at run start; external libraries and the separately hash-bound Rust binary are not copied.',
                   'files':files})
    hashes[str(manifest.resolve())]=sha(manifest)
    return manifest,hashes


def validate_review_boundaries(geometry,review):
    """Both selected sides and their common chin proxy require source review."""
    crossings=review.get('mouth_crossings')
    if not isinstance(crossings,list) or len(crossings)!=2 or len(geometry['arcs'])!=2:
        raise ValueError('source review requires exactly two mouth boundaries')
    for actual,reviewed in zip(geometry['arcs'],crossings):
        if actual['start']['edge']!=reviewed['edge'] or abs(actual['start']['t']-reviewed['t'])>1e-10:
            raise ValueError('source review did not inspect these mouth boundaries')
    old=review['chin152_projection']
    for arc in geometry['arcs']:
        end=arc['end']
        if end['edge']!=old['edge'] or abs(end['t']-old['t'])>1e-10:
            raise ValueError('source review chin projection differs')


def run(evidence, proof_path, review_path, binary, output, amounts=(0.,60.,-60.),
        additional_protection=None, collocation_spacing=None, field_stride=4,
        target_profile=None, soft_source_boundary=None,
        soft_boundary_density=.05, soft_boundary_fade=16., boundary_motion=None,
        curve_correspondence='source-arclength'):
    started=time.perf_counter()
    if output.exists(): raise FileExistsError('refusing to overwrite portable jaw results')
    if curve_correspondence not in ['source-arclength','uniform-target-arclength']:
        raise ValueError('explicit source or uniform target arclength correspondence required')
    if curve_correspondence!='source-arclength' and (target_profile is None or boundary_motion is not None or soft_source_boundary is not None):
        raise ValueError('sliding diagnostic requires a frozen jaw profile and a free outer interface')
    proof,review=read(proof_path),read(review_path)
    source_path=Path(proof['source_image']); source=Image.open(source_path).convert('RGBA')
    size=source.size; rgba=np.asarray(source)
    record=read(evidence)
    if list(size)!=proof['source_size'] or [record['source']['width'],record['source']['height']]!=list(size):
        raise ValueError('source evidence dimensions differ')
    hashes={**proof['hashes'],**review['hashes']}
    if hashes.get(str(evidence))!=sha(evidence) or hashes.get(str(source_path))!=sha(source_path):
        raise ValueError('reviewed source/detection identity differs')
    if (review.get('mode')!='model_source_visual_review_only'
            or review.get('automatic_edit_authorized') is not False
            or review.get('visibility_status')!='unknown'):
        raise ValueError('explicit offline source review required without automatic visibility upgrade')
    for path in [evidence,proof_path,review_path,binary,Path(__file__),
                 Path(__file__).with_name('portable_jaw.py'),Path(__file__).with_name('portable_field.py')]:
        hashes[str(path.resolve())]=sha(path)
    if any(sha(p)!=digest for p,digest in hashes.items()): raise ValueError('frozen source evidence changed')
    rows=[c for c in record['checks'] if c.get('key')=='faceWidth']
    if len(rows)!=1: raise ValueError('one cached face evidence record required')
    operations=rows[0]['graph']['operations']
    if len(operations)!=1 or operations[0]['type']!='face.adjust':raise ValueError('one source face operation required')
    params=operations[0]['params']
    contour=pixels(params['semanticMasks']['faceContour']['points'],size)
    oval=pixels(params['landmarks'],size)
    eyes=[pixels(params['semanticMasks'][key]['points'],size) for key in ['leftEye','rightEye']]
    lips=pixels(params['semanticMasks']['lips']['points'],size)
    geometry=jaw_geometry(contour,oval,eyes,lips)
    validate_review_boundaries(geometry,review)
    profile=None
    if target_profile is not None:
        from jaw_target_profile import load_profile, apply_profile_samples, profile_contour_targets
        if len(amounts)!=1 or collocation_spacing is None:
            raise ValueError('one signed amount and continuous collocation required for a frozen profile')
        profile=load_profile(target_profile,geometry,amounts[0])
        if curve_correspondence=='uniform-target-arclength':
            from jaw_curve_correspondence import (apply_arclength_samples as apply_profile_samples,
                arclength_contour_targets as profile_contour_targets)
        bound_inputs=read(Path(profile['source_protocol']))['inputs_sha256']
        if any(bound_inputs.get(str(path.resolve()))!=sha(path)
               for path in [source_path,evidence,proof_path,review_path]):
            raise ValueError('profile source identity differs from this reviewed image and detection')
        hashes[str(target_profile.resolve())]=sha(target_profile)
        for key in ['source_protocol','qp_source','qp_report','export_source']:
            path=profile.get(key+'_path',profile.get(key))
            if path:
                hashes[str(Path(path).resolve())]=profile[key+'_sha256']
    soft=None
    if soft_source_boundary is not None and boundary_motion is not None:
        raise ValueError('compare boundary controllers separately before combining objectives')
    if soft_source_boundary is not None:
        from source_boundary_soft_targets import sample_boundary_soft_targets
        boundary=read(soft_source_boundary)
        if (boundary.get('schema')!='reviewed-source-neck-boundary-candidate-v1'
                or boundary.get('source_image_sha256')!=sha(source_path)
                or boundary.get('source_size')!=list(size)
                or boundary.get('mode')!='model_source_visual_review_only'
                or boundary.get('automatic_edit_authorized') is not False):
            raise ValueError('soft boundary requires this image and explicit offline source review')
        for path,digest in boundary['input_hashes'].items():
            if sha(path)!=digest:raise ValueError('soft boundary source evidence changed')
            hashes[path]=digest
        hashes[str(soft_source_boundary.resolve())]=sha(soft_source_boundary)
        soft=sample_boundary_soft_targets([s['source_points'] for s in boundary['segments']],
            uncertainty_px=[s['uncertainty_radius_px'] for s in boundary['segments']],
            spacing=2.,density=soft_boundary_density,fade_length=soft_boundary_fade)
    if boundary_motion is not None:
        boundary=read(boundary_motion)
        if (boundary.get('schema')!='source-boundary-motion-targets-v1'
                or boundary.get('source_image_sha256')!=sha(source_path)
                or boundary.get('source_size')!=list(size)
                or len(amounts)!=1 or boundary.get('amount')!=amounts[0]
                or target_profile is None or boundary.get('jaw_profile_sha256')!=sha(target_profile)):
            raise ValueError('boundary motion must bind the same image, signed amount and jaw profile')
        bound=boundary.get('inputs_sha256',{})
        if bound.get(str(source_path.resolve()))!=sha(source_path):
            raise ValueError('boundary motion must retain the original image identity')
        for path,digest in bound.items():
            if sha(path)!=digest:raise ValueError('boundary motion inputs changed')
            hashes[path]=digest
        hashes[str(boundary_motion.resolve())]=sha(boundary_motion)
        soft={key:np.asarray(boundary[key],float) for key in ['source','target','weights']}
        if (soft['source'].ndim!=2 or soft['source'].shape[1]!=2 or not len(soft['source'])
                or soft['target'].shape!=soft['source'].shape
                or soft['weights'].shape!=(len(soft['source']),)
                or not all(np.isfinite(soft[key]).all() for key in ['source','target','weights'])
                or np.any(soft['weights']<=0.)):
            raise ValueError('finite boundary source/target/positive weights with matching rows required')
        soft['report']=boundary['report']
    fixed,crop,roles,radius=fixed_support(size,params,geometry)
    refinement=None
    if additional_protection is not None:
        refinement=read(additional_protection)
        if (refinement['reference_source_sha256'] != sha(source_path)
                or refinement.get('operation') not in ['union_of_reference_and_native_fixed_support',
                                                       'source_reviewed_neck_boundary_protection']):
            raise ValueError('additional protection is not bound to this reference source')
        for path,digest in refinement['inputs_sha256'].items():
            if sha(path)!=digest:raise ValueError('additional protection inputs changed')
            hashes[path]=digest
        mask_path=Path(refinement['mask'])
        if sha(mask_path)!=refinement['mask_sha256']:
            raise ValueError('additional protection mask changed')
        extra=np.asarray(Image.open(mask_path).convert('L'))>0
        if extra.shape!=fixed.shape or np.any(fixed & ~extra):
            raise ValueError('additional protection must retain every original fixed pixel')
        roles['additional_protection']=extra & ~fixed
        fixed |= extra
        hashes[str(additional_protection.resolve())]=sha(additional_protection)
        hashes[str(mask_path.resolve())]=sha(mask_path)
    yy,xx=np.where(fixed);fixed_points=np.column_stack([xx,yy])
    output.mkdir(parents=True)
    code_manifest,code_hashes=snapshot_local_code(output);hashes.update(code_hashes)
    save(output/'protocol.json',{'inputs_sha256':hashes,'source_geometry':geometry,'crop':crop,
        'amounts':list(amounts),'max_radius_fraction':.13,'support_radius_px':radius,
        'target_weight':4.,'bending_weight':.05,'membrane_weight':.001,
        'collocation_spacing':collocation_spacing,
        'target_controller':profile['controller'] if profile else 'jaw_arc_v1',
        'curve_correspondence':curve_correspondence,
        'correspondence_scope':('same frozen jaw target polyline set and endpoints; source material pairs by normalized target arclength; two-component displacement, outer interface remains free'
            if curve_correspondence=='uniform-target-arclength' else 'original source-arclength pairing'),
        'target_optimization_accepted':profile.get('accepted_target_constraints',profile.get('report',{}).get('accepted_numerically')) if profile else None,
        'target_solver_status':profile.get('solver_status') if profile else None,
        'diagnostic_only':bool(profile and profile.get('diagnostic_only')),
        'frozen_target_profile':str(target_profile) if profile else None,
        'profile_amount_already_baked':True if profile else None,
        'soft_source_boundary':soft['report'] if soft is not None else None,
        'soft_source_boundary_evidence':str(soft_source_boundary) if soft_source_boundary is not None else None,
        'boundary_motion_evidence':str(boundary_motion) if boundary_motion is not None else None,
        'field_reduction_stride':field_stride,
        'continuous_arc_probe_spacing_px':.5,
        'problem':'new portable source quadratic; not the historical whole-face objective',
        'rendering':'bilinear, no fixed-mask pixel replacement, no solved-field taper',
        'execution_code_snapshot':str(code_manifest),
        'additional_protection':refinement,
        'execution_limits':{'all_contour_error_max_px':.25,'active_requested_motion_gt_px':.5,
                            'each_active_projection_gain_min':.90,'each_active_projection_gain_max':1.10,
                            'zero_coordinates_and_image_bytes':'exact',
                            'scope':'portable jaw numerical execution in source pixel units; not high-resolution detection or beauty validation'},
        'source_review':'model-reviewed offline selection; visibility remains unknown',
        'quality_score':None})
    Image.fromarray(fixed.astype('uint8')*255).save(output/'fixed-source.png')
    results=[];images=[source.convert('RGB')];labels=['Original']
    for amount in amounts:
        case=output/('zero' if amount==0 else ('plus' if amount>0 else 'minus')+f'{abs(amount):g}')
        case.mkdir();tick=time.perf_counter()
        if profile is None:
            target,request,controller=request_targets(geometry,amount,binary)
        else:
            target=profile_contour_targets(geometry,profile)
            request={'source_geometry':geometry,'amount':amount,
                     'frozen_profile':str(target_profile),'profile_sha256':sha(target_profile)}
            controller={'controller':profile['controller'],'amount_already_baked':True,
                        'targets':target.tolist(),'aesthetic_approval':False,
                        'curve_correspondence':curve_correspondence}
        save(case/'request.json',request);save(case/'controller.json',controller)
        controls=None
        if collocation_spacing is not None:
            controls=sample_jaw_targets(geometry,amount,spacing=collocation_spacing)
            if profile is not None:controls=apply_profile_samples(controls,profile,geometry)
        fit_source=contour if controls is None else controls['source']
        fit_target=target if controls is None else controls['target']
        fit_weights=np.full(len(contour),4.) if controls is None else controls['weights']
        if soft is not None:
            fit_source=np.vstack([fit_source,soft['source']])
            fit_target=np.vstack([fit_target,soft['target']])
            fit_weights=np.r_[fit_weights,soft['weights']]
        problem=build_problem((size[1],size[0]),fit_source,fit_target,fixed_points,crop=crop,
            target_weights=fit_weights,stride=field_stride)
        if controls is not None:
            save(case/'collocation.json',controls['report'])
        result=solve_problem(problem)
        np.save(case/'source-forward.npy',result['source_displacement'])
        decision={'amount':amount,'problem':problem['diagnostics'],'solver':result['report'],
            'source_geometry':result['geometry'],'target_max_error_source_px':result['target_max_error_px'],
            'fixed_source_motion_max_px':result['fixed_source_motion_max_px'],
            'aesthetic_approval':False,'rendered':False}
        if not result['inverse_evaluation_allowed']:
            decision['status']='rejected_source_geometry';save(case/'decision.json',decision)
            results.append(decision);continue
        cy,cx=np.mgrid[:crop[3],:crop[2]]
        q=np.column_stack([cx.ravel(),cy.ravel()])/2.
        inv,closure=inverse_points(result['source_displacement']/2.,q)
        field=((inv-q)*2.).reshape(crop[3],crop[2],2).astype('<f4')
        np.save(case/'inverse-field.npy',field)
        final=inspect(field);decision.update(final_geometry=final,inverse_closure=closure)
        if not final['passed_strict']:
            decision['status']='rejected_final_geometry';save(case/'decision.json',decision)
            results.append(decision);continue
        edited,sampling=render_frozen_field(rgba,field,crop,method='bilinear')
        if soft is not None:
            soft_actual,soft_closure=inverse_points(field.astype(float),soft['source']-np.array(crop[:2]))
            soft_actual+=crop[:2]
            soft_motion=np.linalg.norm(soft_actual-soft['source'],axis=1)
            np.savez_compressed(case/'soft-boundary-motion.npz',source=soft['source'],actual=soft_actual,
                                target=soft['target'],weights=soft['weights'])
            decision['soft_boundary']={'max_motion_px':float(soft_motion.max()),
                'mean_motion_px':float(soft_motion.mean()),'closure':soft_closure,
                'max_target_error_px':float(np.linalg.norm(soft_actual-soft['target'],axis=1).max()),
                'soft_pre_solve_targets_only':True,'fixed_boundary_claim':False}
        changed=np.any(edited!=rgba,axis=2)
        # Measure the actual sampled f32 inverse as its own continuous map.
        actual,actual_closure=inverse_points(field.astype(float),contour-np.array(crop[:2]))
        actual+=crop[:2]
        targets=summarize(actual,target,range(len(contour)),contour)
        save(case/'contour.json',targets)
        np.savez_compressed(case/'contours.npz',source=contour,target=target,actual=actual)
        outside=np.setdiff1d(np.arange(len(contour)),geometry['selected_rows'])
        role_changes={key:int(np.count_nonzero(changed & value)) for key,value in roles.items()}
        fixed_changed=int(np.count_nonzero(changed & fixed));total_changed=int(changed.sum())
        execution=evaluate_result(contour,target,actual,amount=amount,
            fixed_changed_pixels=fixed_changed,whole_image_changed_pixels=total_changed)
        probes=independent_jaw_probes(geometry,amount,spacing=.5,
            collocation_spacing=2. if collocation_spacing is None else collocation_spacing)
        if profile is not None:probes=apply_profile_samples(probes,profile,geometry)
        probe_actual,probe_closure=inverse_points(field.astype(float),probes['source']-np.array(crop[:2]))
        probe_actual+=crop[:2]
        dense_execution=evaluate_result(probes['source'],probes['target'],probe_actual,amount=amount,
            fixed_changed_pixels=fixed_changed,whole_image_changed_pixels=total_changed)
        np.savez_compressed(case/'arc-probes.npz',source=probes['source'],target=probes['target'],actual=probe_actual)
        save(case/'arc-probes.json',{'execution':dense_execution,'sampling':probes['report'],'closure':probe_closure})
        passed=execution['passed'] and dense_execution['passed']
        if boundary_motion is not None:
            boundary_gate=evaluate_result(soft['source'],soft['target'],soft_actual,amount=amount,
                fixed_changed_pixels=fixed_changed,whole_image_changed_pixels=total_changed)
            decision['boundary_motion_execution_gate']=boundary_gate
            passed=passed and boundary_gate['passed']
        decision.update(status='requires_visual_review' if passed else 'rejected_numerical_execution',
            fixed_pixels=int(fixed.sum()),fixed_changed_pixels=fixed_changed,
            role_changed_pixels=role_changes,whole_image_changed_pixels=total_changed,
            zero_image_exact=bool(not changed.any()) if amount==0 else None,
            contour_max_error_px=targets['max_error_px'],contour_mean_gain=targets['mean_active_projection_gain'],
            contour_worst_gain=targets['worst_active_projection_gain'],
            outside_arc_actual_motion_max_px=float(np.linalg.norm(actual[outside]-contour[outside],axis=1).max()),
            image_contract_pass=bool(passed),execution_gate=execution,
            continuous_arc_gate=dense_execution,
            sampling=sampling,actual_contour_closure=actual_closure,
            elapsed_seconds=time.perf_counter()-tick)
        if passed:
            Image.fromarray(edited).save(case/'photo.png');decision['rendered']=True
            if amount!=0: images.append(Image.fromarray(edited).convert('RGB'));labels.append(f'Local jaw {amount:+g}')
        save(case/'decision.json',decision);results.append(decision)
        print({k:decision[k] for k in ['amount','status','fixed_changed_pixels','contour_max_error_px','contour_worst_gain','elapsed_seconds']},flush=True)
    lo=np.maximum(np.floor(contour.min(0)-16).astype(int),0)
    hi=np.minimum(np.ceil(contour.max(0)+40).astype(int),size)
    for name,box in [('face',(*lo,*hi)),('full',(0,0,*size))]:
        w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(w*len(images),h+28),'white');draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(images,labels)):
            panel.paste(im.crop(box),(i*w,28));draw.text((i*w+3,5),label,fill='black')
        panel.save(output/(name+'.png'))
    if any(sha(p)!=digest for p,digest in hashes.items()):raise ValueError('frozen inputs changed during run')
    save(output/'report.json',{'cases':results,'source':str(source_path),'inputs_sha256':hashes,
                             'aesthetic_approval':False,'elapsed_seconds':time.perf_counter()-started,
                             'curve_correspondence':curve_correspondence,
                             'target_optimization_accepted':profile.get('accepted_target_constraints',profile.get('report',{}).get('accepted_numerically')) if profile else None,
                             'target_solver_status':profile.get('solver_status') if profile else None,
                             'diagnostic_only':bool(profile and profile.get('diagnostic_only'))})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['evidence','proof','review','binary','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--additional-protection',type=Path)
    p.add_argument('--collocation-spacing',type=float)
    p.add_argument('--field-stride',type=int,default=4)
    p.add_argument('--amounts',type=float,nargs='+',default=[0.,60.,-60.])
    p.add_argument('--target-profile',type=Path)
    p.add_argument('--soft-source-boundary',type=Path)
    p.add_argument('--soft-boundary-density',type=float,default=.05)
    p.add_argument('--soft-boundary-fade',type=float,default=16.)
    p.add_argument('--boundary-motion',type=Path)
    p.add_argument('--curve-correspondence',choices=['source-arclength','uniform-target-arclength'],default='source-arclength')
    a=p.parse_args();run(a.evidence.resolve(),a.proof.resolve(),a.review.resolve(),a.binary.resolve(),a.output.resolve(),
                        additional_protection=a.additional_protection.resolve() if a.additional_protection else None,
                        collocation_spacing=a.collocation_spacing,field_stride=a.field_stride,
                        amounts=tuple(a.amounts),
                        target_profile=a.target_profile.resolve() if a.target_profile else None,
                        soft_source_boundary=a.soft_source_boundary.resolve() if a.soft_source_boundary else None,
                        soft_boundary_density=a.soft_boundary_density,soft_boundary_fade=a.soft_boundary_fade,
                        boundary_motion=a.boundary_motion.resolve() if a.boundary_motion else None,
                        curve_correspondence=a.curve_correspondence)
