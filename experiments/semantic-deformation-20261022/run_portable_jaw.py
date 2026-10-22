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
        additional_protection=None, collocation_spacing=None, field_stride=4):
    started=time.perf_counter()
    if output.exists(): raise FileExistsError('refusing to overwrite portable jaw results')
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
        target,request,controller=request_targets(geometry,amount,binary)
        save(case/'request.json',request);save(case/'controller.json',controller)
        controls=None
        if collocation_spacing is not None:
            controls=sample_jaw_targets(geometry,amount,spacing=collocation_spacing)
        problem=build_problem((size[1],size[0]),
            contour if controls is None else controls['source'],
            target if controls is None else controls['target'],fixed_points,crop=crop,
            target_weights=4. if controls is None else controls['weights'],stride=field_stride)
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
        probe_actual,probe_closure=inverse_points(field.astype(float),probes['source']-np.array(crop[:2]))
        probe_actual+=crop[:2]
        dense_execution=evaluate_result(probes['source'],probes['target'],probe_actual,amount=amount,
            fixed_changed_pixels=fixed_changed,whole_image_changed_pixels=total_changed)
        np.savez_compressed(case/'arc-probes.npz',source=probes['source'],target=probes['target'],actual=probe_actual)
        save(case/'arc-probes.json',{'execution':dense_execution,'sampling':probes['report'],'closure':probe_closure})
        passed=execution['passed'] and dense_execution['passed']
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
                             'aesthetic_approval':False,'elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['evidence','proof','review','binary','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--additional-protection',type=Path)
    p.add_argument('--collocation-spacing',type=float)
    p.add_argument('--field-stride',type=int,default=4)
    a=p.parse_args();run(a.evidence.resolve(),a.proof.resolve(),a.review.resolve(),a.binary.resolve(),a.output.resolve(),
                        additional_protection=a.additional_protection.resolve() if a.additional_protection else None,
                        collocation_spacing=a.collocation_spacing,field_stride=a.field_stride)
