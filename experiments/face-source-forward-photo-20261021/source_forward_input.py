"""Independent source-forward input construction; never solves or renders."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix,csc_matrix,diags,eye,hstack,kron,save_npz,vstack

from adapt import point_operators,read,save,sha,square_nodes

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'experiments/face-feature-transport-20261020'))
sys.path.insert(0,str(ROOT/'experiments/face-user-source-roles-20261021'))
from evidence import resolve
from context_refinement import expanded_mask,guaranteed_zero_patches
from whole_face_width import derive_whole_face_width
from prepare_bounded_face import touched_nodes


def grid_for_crop(crop,image_size):
    if len(crop)!=4 or len(image_size)!=2 or any(type(v) is not int for v in crop):
        raise ValueError('integer original crop and image dimensions required')
    original=list(crop);lo=np.array(crop[:2],int);span=np.array(crop[2:],int)-1
    image_size=np.array(image_size,int)
    if np.any(span<2) or np.any(lo<0) or np.any(lo+span>=image_size):
        raise ValueError('crop outside source photograph')
    for axis in range(2):
        if span[axis]%2:
            if lo[axis]+span[axis]+1<image_size[axis]:span[axis]+=1
            elif lo[axis]>0:lo[axis]-=1;span[axis]+=1
            else:raise ValueError('exact 2px grid cannot cover original crop inside source photograph')
    extended=[*lo.tolist(),*(span+1).tolist()]
    grid={'origin':lo.tolist(),'spacing':[2.,2.],
          'width':int(span[0]//2+1),'height':int(span[1]//2+1)}
    return grid,extended,{'original_crop':original,'source_field_crop':extended,
        'image_resampling':False,'extension_rule':'at most one source pixel per odd-span dimension, positive side preferred; no crop truncation',
        'added_source_pixels':int(np.prod(span+1)-crop[2]*crop[3])}


def line_constraints(lines,grid,mapping,nvars):
    """Exact piecewise-quadratic zero-normal equalities in source space.

    The original hard line fixes normal displacement on its original spatial
    segment. Restricting a bilinear field to each crossed cell gives degree 2;
    three Bernstein coefficients per interval fix the whole continuous segment,
    even when a requested sample count of two would miss an interior bend.
    Vector packing is [all x variables, all y variables], not interleaved.
    """
    rows=[];reports=[];row_offset=0
    origin=np.array(grid['origin'],float);spacing=np.array(grid['spacing'],float)
    for index,line in enumerate(lines):
        if line.get('weight') is not None:
            raise ValueError('soft background line energy requires an explicit source-weight conversion; not silently dropped')
        start=np.array(line['start'],float);end=np.array(line['end'],float);delta=end-start
        length=float(np.linalg.norm(delta));count=line['samples']
        if start.shape!=(2,) or end.shape!=(2,) or not np.isfinite([*start,*end]).all() or length<=1e-9 or type(count) is not int or not 2<=count<=100000:
            raise ValueError('finite nondegenerate line with original sample count required')
        normal=np.array([-delta[1],delta[0]])/length;breaks=[0.,1.]
        for axis,n in enumerate([grid['width'],grid['height']]):
            if delta[axis]!=0:
                t=(origin[axis]+np.arange(1,n-1)*spacing[axis]-start[axis])/delta[axis]
                breaks.extend(t[(t>0)&(t<1)].tolist())
        ordered=sorted(breaks);breaks=[ordered[0]]
        for t in ordered[1:]:
            if t-breaks[-1]>=1e-12:breaks.append(t)
        pieces=[]
        for a,b in zip(breaks[:-1],breaks[1:]):
            times=np.array([a,(a+b)*.5,b]);points=start+times[:,None]*delta
            basis,_,_=point_operators(points,grid,mapping,nvars)
            bernstein=vstack([basis[0],2*basis[1]-.5*(basis[0]+basis[2]),basis[2]],format='csr')
            pieces.append(hstack([normal[0]*bernstein,normal[1]*bernstein],format='csr'))
        matrix=vstack(pieces,format='csr');matrix.sum_duplicates();matrix.eliminate_zeros();rows.append(matrix)
        reports.append({'line_index':index,'original':copy.deepcopy(line),'hard':True,
            'normal':normal.tolist(),'breaks':breaks,'polynomial_degree':2,
            'bernstein_rows':matrix.shape[0],'row_start':row_offset,'row_end_exclusive':row_offset+matrix.shape[0],
            'structurally_zero_rows_after_original_locks':int(np.count_nonzero(np.diff(matrix.indptr)==0)),
            'normal_displacement_rhs':0.,'original_requested_samples':count,
            'source_line_semantics':'zero normal displacement on complete original source segment; tangent motion remains free',
            'not_old_inverse_field_representation':True})
        row_offset+=matrix.shape[0]
    return (vstack(rows,format='csr') if rows else csc_matrix((0,2*nvars))),reports


def replace_targets(old_targets,source_contour,requested):
    source_contour=np.asarray(source_contour,float);requested=np.asarray(requested,float)
    if requested.shape!=source_contour.shape or requested.ndim!=2 or requested.shape[1]!=2 or len(requested)>len(old_targets):
        raise ValueError('same finite source contour/request shape required')
    if not np.isfinite([source_contour,requested]).all():raise ValueError('finite targets required')
    rows=[]
    for index,old in enumerate(old_targets):
        destination=np.asarray(old['position'],float);source=destination+old['delta']
        if index<len(source_contour):
            np.testing.assert_array_equal(source,source_contour[index]);destination=requested[index]
        rows.append({'original_index':index,'source':source.tolist(),'destination':destination.tolist(),
            'forward_delta':(destination-source).tolist(),'weight':old['weight'],
            'original_destination':old['position'],'original_inverse_delta':old['delta']})
    return rows


def zero_patches(old_grid,node_ids):
    raw=np.zeros((old_grid['height'],old_grid['width']),bool)
    raw.ravel()[node_ids]=True;raw[[0,-1]]=True;raw[:,[0,-1]]=True
    mask=guaranteed_zero_patches(expanded_mask(raw));y,x=np.where(mask)
    lo=np.column_stack([x,y])*old_grid['spacing']+old_grid['origin']
    return lo,lo+old_grid['spacing']


def allocate_mapping(grid,locked,groups):
    """Reject conflicting complete supports instead of editing their masks."""
    fixed=[int(np.count_nonzero(locked[ids])) for ids in groups]
    pairs={f'{i}-{j}':len(np.intersect1d(groups[i],groups[j]))
           for i in range(len(groups)) for j in range(i+1,len(groups))}
    if any(fixed) or any(pairs.values()):
        raise ValueError(f'complete source support conflicts: fixed={fixed}, groups={pairs}')
    material=np.zeros(len(locked),bool)
    for ids in groups:material[ids]=True
    free=np.flatnonzero(~locked&~material);count=len(free)+len(groups)
    mapping=np.full(len(locked),-1,dtype=np.int64);mapping[free]=np.arange(len(free))
    group_ids=np.arange(len(free),count)
    for ids,value in zip(groups,group_ids):mapping[ids]=value
    return mapping,group_ids,{'fixed_collisions':fixed,'group_collisions':pairs,
        'free_nodes':len(free),'source_group_count':len(groups),'scalar_variables':count}


def signed_target_intent(applicability,baseline_amount,amount_override=None):
    """Derive each signed request from the same source, never a solved field."""
    amount=float(baseline_amount if amount_override is None else amount_override)
    if not np.isfinite(amount) or not -100<=amount<=100:
        raise ValueError('finite faceWidth amount in [-100,100] required')
    matches=[case for case in applicability['cases'] if case['amount']==amount]
    if len(matches)!=1:raise ValueError('one matching independently evidenced source amount required')
    contour=np.asarray(applicability['source_128'],float)
    intent=derive_whole_face_width(contour,applicability['frame'],applicability['source_oval'],
                                  applicability['source_eyes'],amount)
    np.testing.assert_array_equal(intent['requested'],matches[0]['new_requested_128'])
    return amount,intent


def prepare(applicability_path,output,oracle_evidence,amount_override=None):
    applicability=read(applicability_path);base=Path(applicability['base'])
    for path,digest in applicability['inputs_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError(f'frozen source applicability evidence changed: {path}')
    old=read(base/'controls.json');p=read(base/'provenance.json')
    if old.get('material_strain') or p.get('ear_orientation'):
        raise ValueError('this independent feature-only constructor requires explicit adaptation of any existing ear/strain objective')
    source,image,graph_path,hard_path=resolve(base,p)
    image_size=list(image.size);iw,ih=image_size
    graph=read(graph_path)['operations'][0]['params']
    oracle=read(oracle_evidence/'report.json')
    if (sha(oracle_evidence/'masks.npz')!=oracle['masks_sha256'] or sha(source)!=oracle['source_sha256']
            or sha(graph_path)!=oracle['graph_sha256'] or sha(base/'provenance.json')!=oracle['base_provenance_sha256']
            or sha(Path(oracle['original_audit']))!=oracle['original_audit_sha256']):
        raise ValueError('frozen original RGB oracle evidence changed')
    oracle_arrays=dict(np.load(oracle_evidence/'masks.npz'))
    baseline_amount=float(p['source_relative_width']['amount'])
    amount,intent=signed_target_intent(applicability,baseline_amount,amount_override)
    if amount!=baseline_amount and any(g['target_weight']!=0 for g in old['rigid_groups']):
        raise ValueError('signed amount override requires explicit adaptation of nonzero material priors')
    contour=np.asarray(applicability['source_128'],float)
    targets=replace_targets(old['targets'],contour,intent['requested'])
    original_sources=np.array([row['position'] for row in old['targets']])+np.array([row['delta'] for row in old['targets']])
    np.testing.assert_array_equal(original_sources,[r['source'] for r in targets])
    np.testing.assert_array_equal([r['weight'] for r in old['targets']],[r['weight'] for r in targets])
    if len(targets)!=applicability['original_target_count']:raise ValueError('target row count changed')
    grid,crop,grid_proof=grid_for_crop(p['crop'],image_size)
    left,top,cw,ch=p['crop'];w,h=grid['width'],grid['height'];n=w*h
    classified=np.asarray(Image.open(hard_path))>0
    if classified.shape!=(ch,cw):raise ValueError('original classifier crop mismatch')
    masks={'classified_fixed':np.zeros((ih,iw),bool)}
    masks['classified_fixed'][top:top+ch,left:left+cw]=classified
    core_reconstruction={}
    for name in ['nose','lips']:
        for key in [name+'_full_polygon',name+'_reconstructed_core']:
            if oracle_arrays[key].shape!=(ih,iw) or oracle_arrays[key].dtype!=np.dtype(bool):
                raise ValueError('complete bool masks on original source dimensions required')
            masks[key]=oracle_arrays[key].copy()
        core=masks[name+'_reconstructed_core'][top:top+ch,left:left+cw]
        yy,xx=np.where(core)
        old_nodes=touched_nodes(np.column_stack([xx+left,yy+top]),old['grid']['origin'],
            old['grid']['spacing'],old['grid']['width'],old['grid']['height'])
        recorded=p['feature_cores'][name]
        core_reconstruction[name]={'recorded':recorded,'reconstructed_pixels':int(core.sum()),
            'reconstructed_old_grid_nodes':len(old_nodes),
            'matches_recorded_counts':bool(core.sum()==recorded['core_pixels'] and len(old_nodes)==recorded['nodes']),
            'reconstructed_pixels_outside_complete_fixed_polygon':int(np.count_nonzero(masks[name+'_reconstructed_core']&~masks[name+'_full_polygon'])),
            'original_runtime_oracle':str(oracle_evidence),
            'pillow_version':oracle['pillow_version'],'complete_frozen_mask_sha256':oracle['masks_sha256']}
        if not core_reconstruction[name]['matches_recorded_counts']:
            raise ValueError('frozen historical raster does not reproduce original core pixel/node evidence')
    fixed_mask=np.logical_or.reduce(list(masks.values()));y,x=np.where(fixed_mask)
    fixed_points=np.column_stack([x,y]);fixed_ids=square_nodes(fixed_points,grid,identity_extension=True)
    locked=np.zeros(n,bool);locked[fixed_ids]=True
    collar=np.zeros((h,w),bool);collar[:2]=collar[-2:]=True;collar[:,:2]=collar[:,-2:]=True
    locked|=collar.ravel()
    context_min,context_max=zero_patches(old['grid'],p['context_fixed_nodes'])
    historical_min,historical_max=zero_patches(old['grid'],old['fixed_nodes'])
    context=np.zeros((h,w),bool)
    for lo,hi in zip(context_min,context_max):
        a=np.floor((lo-grid['origin'])/grid['spacing']).astype(int)
        b=np.ceil((hi-grid['origin'])/grid['spacing']).astype(int)
        # The new crop encloses the complete old crop. Only old finite-domain
        # coordinates are processed; clipping does not truncate a source role.
        if np.any(a<0) or np.any(b>=[w,h]):raise ValueError('original context patch outside new enclosing grid')
        context[a[1]:b[1]+1,a[0]:b[0]+1]=True
    locked|=context.ravel()
    groups=[{'name':'accessory_'+str(i),'source_material':e['source_material']+e['source_contact'],
             'evidence_kind':'original_accessory_material_and_contact'} for i,e in enumerate(p['rigid_evidence'])]
    groups += [{'name':'+'.join(e['features']),'features':e['features'],
                'source_material':e['source_material'],'evidence_kind':'original_complete_feature_material'}
               for e in p['feature_transport']['groups']]
    if len(groups)!=len(old['rigid_groups']):raise ValueError('original rigid group identity count differs')
    group_nodes=[]
    for index,(group,prior) in enumerate(zip(groups,old['rigid_groups'])):
        points=np.asarray(group['source_material'],float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or not np.array_equal(points,np.round(points)):
            raise ValueError('complete original integer source material required')
        if np.any(points<0) or np.any(points>=[iw,ih]):raise ValueError('original material outside photo')
        nodes=square_nodes(points,grid);group_nodes.append(nodes)
        group.update(group_index=index,original_source_rows=len(points),unique_source_texels=len(np.unique(points,axis=0)),
            forward_translation_prior=(-np.asarray(prior['delta'])).tolist(),translation_prior_weight=prior['target_weight'],
            original_inverse_nodes=prior['nodes'],source_support_node_count=len(nodes))
    changed=np.flatnonzero(np.any(np.array([r['destination'] for r in targets])!=np.array([r['position'] for r in old['targets']]),axis=1)).tolist()
    source_contract={'source_photo':str(source),'source_photo_sha256':sha(source),'source_dimensions':image_size,
        'base':str(base),'original_controls':str(base/'controls.json'),'requested_amount':amount,
        'target_definition':intent['name'],'target_contour_count':len(contour),'targets':targets,'groups':groups,
        'background_lines':copy.deepcopy(old.get('background_lines',[])),
        'original_grid':old['grid'],'original_crop':p['crop'],'source_grid':grid,'source_crop':crop,
        'grid_cover_proof':grid_proof,'changed_target_indices':changed,
        'source_fixed_masks':{k:int(v.sum()) for k,v in masks.items()},
        'historical_core_reconstruction_audit':core_reconstruction,
        'original_rgb_oracle_raster':oracle,
        'source_ear_material':'none declared in original baseline; graph ear mask is not substituted for a reviewed material',
        'source_neck_targets':'none declared in original baseline; none borrowed from another photograph',
        'original_unknown_prior':p['unknown_motion_prior'],
        'all_original_target_rows_weights_materials_background_lines_retained':True,
        'source_target_rule':'only the evidenced leading contour rows receive the whole-face-width intent at the original amount; every other original source/destination/weight remains unchanged',
        'source_line_rule':'each original hard segment keeps zero source normal displacement continuously; source reformulation, not a same inverse-objective assertion',
        'source_fixed_rule':'complete original classifier fixed pixels and complete nose/lip oracle polygons; reconstructed erosion cores are redundant subsets, and historical core counts are audited separately',
        'historical_fixed_nodes_rule':'original fixed B3 nodes/zero patches retained as numerical ledger; distant-context zero patches remain hard, feature margins are not invented semantic locks',
        'inverse_jacobian_contract':{'symmetric_min':.52,'symmetric_max':1.8,'absolute_skew_max':.24},
        'no_solver_or_photo_edit':True,'not_aesthetic_approval':True}
    if amount_override is not None:
        source_contract.update(baseline_amount=baseline_amount,explicit_amount_override=amount,
            signed_target_generation='fresh whole_face_width(source_128, own frame/oval/eyes, explicit amount); no field or prior-output scaling',
            source_target_rule='only the evidenced leading contour rows receive the explicit signed whole-face-width intent; every other original source/destination/weight remains unchanged',
            signed_group_prior_rule='unchanged original zero-weight priors; no new nonzero position prior')
    paths={Path(path) for path in applicability['inputs_sha256']}
    paths.update([applicability_path,base/'controls.json',base/'provenance.json',source,graph_path,hard_path,hard_path.parent/'report.json',
                  oracle_evidence/'report.json',oracle_evidence/'masks.npz',Path(oracle['original_audit'])])
    hashes={str(path):sha(path) for path in sorted(paths)}
    output.mkdir(parents=True,exist_ok=False)
    save(output/'source-contract.json',source_contract)
    save(output/'original-controls.json',old);save(output/'original-provenance.json',p)
    save(output/'target-intent.json',{'amount':amount,'name':intent['name'],
        'source':contour.tolist(),'requested':intent['requested'].tolist(),'changed_target_indices':changed,
        'applicability_evidence':str(applicability_path),'source_formula_sha256':sha(ROOT/'experiments/face-user-source-roles-20261021/whole_face_width.py')})
    np.savez_compressed(output/'source-evidence.npz',**masks,
        source_fixed_pixels=fixed_points,fixed_source_nodes=fixed_ids,locked_nodes=np.flatnonzero(locked),
        context_min=context_min,context_max=context_max,historical_identity_min=historical_min,
        historical_identity_max=historical_max,**{f'group_nodes_{i}':ids for i,ids in enumerate(group_nodes)})
    report={'status':'Preparing','inputs_sha256':hashes,'script_sha256':sha(Path(__file__)),
        'original_target_count':len(old['targets']),'source_target_count':len(targets),
        'contour_rows':len(contour),'other_original_rows':len(targets)-len(contour),'changed_target_rows':len(changed),
        'original_group_count':len(old['rigid_groups']),'source_group_count':len(groups),
        'original_hard_background_lines':sum(line.get('weight') is None for line in old.get('background_lines',[])),
        'source_fixed_pixels':len(fixed_points),'source_fixed_nodes':len(fixed_ids),
        'preserved_context_patches':len(context_min),'preserved_context_nodes':int(context.sum()),
        'old_numerical_identity_patches':len(historical_min),'grid_cover_proof':grid_proof,
        'historical_core_reconstruction_audit':core_reconstruction,
        'no_solve_no_render':True,'solving_authorized':False,'aesthetic_approval':False}
    try:
        mapping,group_ids,conflicts=allocate_mapping(grid,locked,group_nodes)
    except ValueError as error:
        report.update(status='RejectedSourceSupportConflict',reason=str(error));save(output/'report.json',report)
        raise
    nvars=conflicts['scalar_variables'];active=np.flatnonzero(mapping>=0)
    prolong=coo_matrix((np.ones(len(active)),(active,mapping[active])),shape=(n,nvars)).tocsr()
    points=np.array([r['source'] for r in targets]);dest=np.array([r['destination'] for r in targets])
    deltas=np.array([r['forward_delta'] for r in targets]);weights=np.array([r['weight'] for r in targets])
    target,_,_=point_operators(points,grid,mapping,nvars)
    hx,hy=grid['spacing']
    lx=diags([-np.ones(w-1),2*np.ones(w),-np.ones(w-1)],[-1,0,1])/hx**2
    ly=diags([-np.ones(h-1),2*np.ones(h),-np.ones(h-1)],[-1,0,1])/hy**2
    laplacian=((kron(eye(h),lx)+kron(ly,eye(w)))@prolong).tocsr()
    dx=(kron(eye(h),diags([-np.ones(w-1),np.ones(w-1)],[0,1],shape=(w-1,w))/hx)@prolong).tocsr()
    dy=(kron(diags([-np.ones(h-1),np.ones(h-1)],[0,1],shape=(h-1,h))/hy,eye(w))@prolong).tocsr()
    priors=np.array([g['forward_translation_prior'] for g in groups],float).reshape(-1,2)
    prior_weights=np.array([g['translation_prior_weight'] for g in groups],float)
    prior=coo_matrix((np.ones(len(groups)),(np.arange(len(groups)),group_ids)),shape=(len(groups),nvars)).tocsr()
    equalities,line_reports=line_constraints(old.get('background_lines',[]),grid,mapping,nvars)
    matrices={'prolongation':prolong,'target':target,'laplacian':laplacian,'Dx':dx,'Dy':dy,
              'group_prior':prior,'hard_line_equalities':equalities}
    for name,matrix in matrices.items():save_npz(output/(name+'.npz'),matrix)
    for name,array in [('target_deltas',deltas),('target_weights',weights),('group_priors',priors),('group_prior_weights',prior_weights)]:
        np.save(output/(name+'.npy'),array)
    np.save(output/'hard_line_rhs.npy',np.zeros(equalities.shape[0]))
    blocks=[{'name':f'all_{len(targets)}_source_targets','matrix_file':'target.npz',
             'rhs_file':'target_deltas.npy','weights_file':'target_weights.npy'},
            {'name':'source_physical_laplacian','matrix_file':'laplacian.npz','weights':old['smoothness']*hx*hy},
            {'name':'source_displacement','matrix_file':'prolongation.npz','weights':old['regularization']*hx*hy},
            {'name':'all_original_rigid_translation_priors','matrix_file':'group_prior.npz',
             'rhs_file':'group_priors.npy','weights_file':'group_prior_weights.npy'}]
    assembly={'nvars':nvars,'blocks':blocks,'hard_equalities':[
        {'matrix_file':'hard_line_equalities.npz','rhs_file':'hard_line_rhs.npy','rows':equalities.shape[0],
         'vector_packing':'[all x variables, all y variables]','all_original_lines':len(line_reports),
         'must_not_be_omitted_or_downgraded_to_soft_penalty':True}],
        'objective_convention':'J=z_x^T H z_x+z_y^T H z_y-2 sum(rhs*z)+constant; hard lines are separate equalities',
        'physical_energy':{'smoothness':old['smoothness'],'regularization':old['regularization'],'cell_area':hx*hy,
            'same_2px_definition_as_user_forward_probe':True,'ear_energy_not_invented_when_absent':True},
        'review_only_not_authorized_to_solve':True}
    save(output/'assembly.json',assembly);save(output/'background-line-contract.json',line_reports)
    hessian=(target.T@diags(weights)@target+old['smoothness']*hx*hy*(laplacian.T@laplacian)
             +old['regularization']*hx*hy*(prolong.T@prolong)+prior.T@diags(prior_weights)@prior).tocsc()
    rhs=target.T@(weights[:,None]*deltas)+prior.T@(prior_weights[:,None]*priors)
    (output/'quadratic').mkdir();save_npz(output/'quadratic/hessian.npz',hessian);np.save(output/'quadratic/rhs.npy',rhs)
    np.savez_compressed(output/'data.npz',target_sources=points,target_destinations=dest,target_deltas=deltas,
        target_weights=weights,node_to_variable=mapping,source_fixed_pixels=fixed_points,
        group_variable_ids=group_ids,group_priors=priors,group_prior_weights=prior_weights,
        preserved_context_nodes=np.flatnonzero(context),old_context_patch_min=context_min,
        old_context_patch_max=context_max,old_identity_patch_min=historical_min,old_identity_patch_max=historical_max,
        **{f'group_nodes_{i}':ids for i,ids in enumerate(group_nodes)})
    contract={'source_contract':str(output/'source-contract.json'),'source_contract_sha256':sha(output/'source-contract.json'),
        'source_photo':str(source),'source_photo_sha256':sha(source),'baseline':str(base),
        'grid':grid,'crop':crop,'original_crop':p['crop'],'scalar_variables':nvars,'node_count':n,
        'target_count':len(targets),'changed_target_indices':changed,'requested_amount':amount,
        'target_definition':intent['name'],'source_fixed_rows':len(fixed_points),
        'group_count':len(groups),'background_line_count':len(line_reports),
        'hard_line_rows':equalities.shape[0],'hard_equalities_required':True,
        'target_intent_sha256':sha(output/'target-intent.json'),
        'inverse_geometry_gate':source_contract['inverse_jacobian_contract'],
        'state':'source_input_prepared_for_independent_review_only','solving_authorized':False,
        'not_aesthetic_approval':True}
    save(output/'contract.json',contract)
    report.update(status='PreparedForIndependentReview',**conflicts,hard_line_rows=equalities.shape[0],
        hard_line_nnz=equalities.nnz,hard_line_structural_zero_rows=int(np.count_nonzero(np.diff(equalities.tocsr().indptr)==0)),
        hessian_nnz=hessian.nnz,original_input_hashes_unchanged=all(sha(Path(k))==v for k,v in hashes.items()),
        constructed_outputs_sha256={name:sha(output/name) for name in ['contract.json','source-contract.json','data.npz','source-evidence.npz','assembly.json','hard_line_equalities.npz','hard_line_rhs.npy','quadratic/hessian.npz','quadratic/rhs.npy']},
        source_limits=['No ear or neck source contract exists here; no user-photo material or neck targets were copied.',
            'Complete nose/lip RGB and core masks are frozen from the historical Pillow 11.1.0 runtime; do not rerasterize with another Pillow version.',
            'Full old inverse B3 numerical margins remain a ledger, not source semantic labels.',
            'Source straight-line equalities preserve the original physical segments, but final inverse line extrema still need separate verification after an authorized solve.',
            'A generic scalar-H direct runner that ignores coupled hard lines cannot consume this input.'])
    save(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['status','source_target_count','source_group_count','source_fixed_pixels',
        'scalar_variables','original_hard_background_lines','hard_line_rows','hard_line_nnz','hard_line_structural_zero_rows',
        'fixed_collisions','group_collisions','original_input_hashes_unchanged']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--applicability',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oracle-evidence',type=Path,required=True)
    parser.add_argument('--amount',type=float,default=None,help='explicit source-derived strength; omission preserves baseline amount')
    args=parser.parse_args();prepare(args.applicability.resolve(),args.output.resolve(),args.oracle_evidence.resolve(),args.amount)
