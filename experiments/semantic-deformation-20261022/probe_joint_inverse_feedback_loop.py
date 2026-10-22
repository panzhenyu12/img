"""Four-step, sixty-second actual inverse feedback; no photograph rendering.

This isolates source-to-output reconstruction errors. Source shape naturalness
is a separate problem; even numerical success is not image approval.
"""
import argparse
from pathlib import Path
import time

import clarabel
import numpy as np
from PIL import Image
from scipy.sparse import save_npz

from joint_inverse_feedback_controller import (ExactCutLedger,FeedbackState,
    assemble_trial,geometry_merit,load_frozen)
from joint_source_constraints import SourceConstraintAudit
from joint_jaw_boundary_field import qp_fingerprint
from inverse_field_linearization import linearize_inverse_samples
from portable_field import point_operators
from probe_joint_inverse_feedback import (bad_final_corners,final_corner_linearization,
    final_jacobian_soc_block,inverse_field,sampled_actual_curve_audit,save,sha)
from forward import inverse_points
from evaluate import inspect
from resampling import sample_rgba
from run_portable_jaw import snapshot_local_code


def fixed_audit(field,grid,rgba,fixed):
    left,top=map(int,grid['origin']);height,width=field.shape[:2]
    local=fixed[top:top+height,left:left+width];yy,xx=np.where(local)
    points=np.column_stack([xx+left,yy+top])
    values,_=sample_rgba(rgba,points+field[yy,xx],method='bilinear')
    sampled=np.floor(np.clip(values,0.,255.)+.5).astype(np.uint8)
    return {'checked':len(points),'changed':int(np.any(sampled!=rgba[yy+top,xx+left],axis=1).sum()),
        'inverse_max_px':float(np.linalg.norm(field[yy,xx],axis=1).max(initial=0.)),
        'outside_crop':'identity by construction; no photograph rendered'}


def curve_response_audit(field,z,grid,across,curves):
    """Original execution thresholds at unchanged observed source samples."""
    rows=[]
    for curve in curves:
        source=curve['source'];requested=source+(curve['operator']@z)[:,None]*across
        actual,closure=inverse_points(field.astype(float),source-grid['origin'])
        actual+=grid['origin'];delta=requested-source;motion=actual-source
        lengths=np.linalg.norm(delta,axis=1);active=lengths>.5
        gains=np.sum(delta[active]*motion[active],axis=1)/lengths[active]**2
        errors=np.linalg.norm(actual-requested,axis=1)
        bad_gain=(gains<.9)|(gains>1.1);bad_error=errors>.25
        rows.append({'index':curve['index'],'sample_count':len(source),'active_count':int(active.sum()),
            'active_gain_min':float(gains.min()) if len(gains) else None,
            'active_gain_max':float(gains.max()) if len(gains) else None,
            'gain_failed_rows':np.flatnonzero(active)[bad_gain].tolist(),
            'contour_error_max_px':float(errors.max(initial=0.)),
            'error_failed_rows':np.flatnonzero(bad_error).tolist(),
            'passed':not bool(bad_gain.any() or bad_error.any()),'closure':closure})
    return {'passed':bool(rows) and all(row['passed'] for row in rows),'curves':rows,
        'limits':{'active_requested_motion_gt_px':.5,'projection_gain':[.9,1.1],'maximum_error_px':.25},
        'target':'current audited source field at unchanged original source samples',
        'continuous_curve_approval':False,'aesthetic_approval':False}


def actual_audit(z,frozen,ctx,rgba,fixed,curves):
    grid=frozen['reference']['grid'];across=ctx.across
    field,closure=inverse_field(z,frozen['mapping'],grid,across)
    report={'geometry':inspect(field),'inverse_closure':closure,
            'fixed':fixed_audit(field,grid,rgba,fixed)}
    bad=bad_final_corners(field)
    # A failure to invert the reconstructed output map is retained alongside
    # its geometry failure, not allowed to hide the full-domain measurement.
    try:report['sampled_curves']=sampled_actual_curve_audit(field,grid,ctx.specifications)
    except ValueError as error:report['sampled_curves']={'sampled_passed':False,'error':str(error)}
    try:report['curve_response']=curve_response_audit(field,z,grid,across,curves)
    except ValueError as error:report['curve_response']={'passed':False,'error':str(error)}
    return report,field,bad


def prediction_diagnostic(linear,z,field,grid,mapping,across):
    """Fresh relocation exposes source-cell crossings and tangent error."""
    points=linear['destinations'];before=linear['linearization']
    fresh=linearize_inverse_samples(grid,mapping,z,across,points)
    def cells(source):
        uv=(source-np.asarray(grid['origin']))/grid['spacing']
        return np.minimum(np.maximum(np.floor(uv).astype(int),0),[grid['width']-2,grid['height']-2])
    predicted=before['constant']+before['rows']@z
    pixel=(points-grid['origin']).astype(int)
    stored=field[pixel[:,1],pixel[:,0]].astype(float)@across
    e=linear['e_const']+linear['e_rows']@z;c=linear['c_const']+linear['c_rows']@z
    disc=np.hypot(e-1.,c)
    predicted_merit=float(np.maximum.reduce([.52-(e+1.-disc)/2.,(e+1.+disc)/2.-1.8,
                                           abs(c)/2.-.24,-e,np.zeros_like(e)]).max(initial=0.))
    return {'required_output_pixels':len(points),
        'source_cell_changes':int(np.any(cells(before['source'])!=cells(fresh['source']),axis=1).sum()),
        'inverse_scalar_prediction_max_error_px':float(abs(predicted-fresh['inverse_scalar']).max(initial=0.)),
        'stored_f32_prediction_max_error_px':float(abs(predicted-stored).max(initial=0.)),
        'f32_scalar_rounding_max_error_px':float(abs(stored-fresh['inverse_scalar']).max(initial=0.)),
        'predicted_targeted_corner_merit':predicted_merit,
        'fresh_sample_closure_max_source_px':fresh['closure_max_source_px'],
        'minimum_anchor_source_cell_boundary_distance_px':float(before['source_cell_boundary_distance_px'].min()),
        'cells_frozen_for_acceptance':False}


def _settings(metadata,remaining):
    settings=clarabel.DefaultSettings()
    for key,value in metadata['settings'].items():setattr(settings,key,value)
    settings.time_limit=min(15.,remaining)
    return settings


def finite_solver_state(solved,total_variables,constraint_rows):
    vectors=[np.asarray(value,float) for value in [solved.x,solved.s,solved.z]]
    return all(value.shape==(size,) and np.isfinite(value).all()
               for value,size in zip(vectors,[total_variables,constraint_rows,constraint_rows]))


def run(reference_path,interface_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('refusing to overwrite bounded feedback evidence')
    frozen=load_frozen(reference_path,interface_path);reference=frozen['reference'];interface=frozen['interface']
    if reference['amount']!=100.:raise ValueError('this fixed experiment accepts original +100 only')
    ctx=SourceConstraintAudit(reference,interface,frozen['mapping'])
    source_path=Path(interface['source_image']);fixed_paths=[Path(p) for p in reference['inputs_sha256'] if p.endswith('/fixed-source.png')]
    if len(fixed_paths)!=1:raise ValueError('one original fixed mask required')
    fixed_path=fixed_paths[0];hashes=dict(frozen['hashes'])
    for path in [source_path,fixed_path]:hashes[str(path)]=reference['inputs_sha256'][str(path)]
    curves=[];grid=reference['grid'];mapping=frozen['mapping'];size=len(frozen['z0']);total=len(frozen['q'])
    for row in reference['diagnostic_curves']:
        path=frozen['reference_path'].parent/f"diagnostic-curve-{row['index']}.npz"
        hashes[str(path)]=row['sha256']
        if sha(path)!=row['sha256']:raise ValueError('frozen curve source samples changed')
        with np.load(path) as values:source=values['source']
        basis,_,_=point_operators(source,grid,mapping,size)
        curves.append({'index':row['index'],'source':source,'operator':basis})
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('frozen inputs changed')
    rgba=np.asarray(Image.open(source_path).convert('RGBA'));fixed=np.asarray(Image.open(fixed_path).convert('L'))>0
    spans=(np.array([grid['width'],grid['height']])-1)*grid['spacing']+1
    if np.any(spans!=np.round(spans)) or np.any(np.asarray(grid['origin'])!=np.round(grid['origin'])):
        raise ValueError('original integer output lattice required')
    left,top=map(int,grid['origin']);width,height=spans.astype(int)
    if fixed.shape!=rgba.shape[:2] or min(left,top)<0 or left+width>rgba.shape[1] or top+height>rgba.shape[0]:
        raise ValueError('source, mask and output crop dimensions differ')
    output.mkdir(parents=True);manifest,code_hashes=snapshot_local_code(output)
    started=time.monotonic();initial_source=ctx.audit(frozen['z0'])
    initial,field,bad=actual_audit(frozen['z0'],frozen,ctx,rgba,fixed,curves)
    np.save(output/'initial-inverse-field.npy',field)
    state=FeedbackState(frozen['z0'],initial,bad);ledger=ExactCutLedger(frozen)
    initial_cuts=ledger.add(ctx.additional_cuts(frozen['z0'],total))
    report={'experiment':'bounded-actual-inverse-feedback-v2','reference':str(frozen['reference_path']),
        'reference_sha256':hashes[str(frozen['reference_path'])],'original_qp_sha256':frozen['metadata']['sha256'],
        'source_variables':size,'auxiliary_variables':total-size,'maximum_solves':4,'total_budget_seconds':60.,
        'maximum_solve_seconds':15.,'maximum_total_source_change_px':.25,'initial_step_radius_px':.25,
        'initial_source_audit':initial_source,'initial_actual_audit':initial,'initial_source_cut_exchange':initial_cuts,
        'initial_actual_corner_indices':bad.tolist(),'rounds':[],
        'source_reference_approved':reference['source_continuous_accepted'],
        'merit':'max(0, .52-min final symmetric eigenvalue, max final symmetric eigenvalue-1.8, max final skew-.24, -min final determinant)',
        'source_cut_policy':'exact cone/row deduplication against frozen QP and accumulated exact source cuts; no rounded keys',
        'inverse_tangent_policy':'rebuild from fresh inverse sample locations at the current accepted iterate each round; retain indices only',
        'trust_policy':'intersect frozen source ±.25 with accepted iterate ±radius; halve radius on rejected nonlinear progress, never expand',
        'intermediate_policy':'AlmostSolved is diagnostic only; it may advance only after full source/fixed gates and strict actual geometry merit decrease',
        'inputs_sha256':hashes,'executed_code_manifest':str(manifest),
        'rendered':False,'executable':False,'image_approval':False,'continuous_final_curve_approval':False}
    reason='maximum_solve_count'
    for iteration in range(4):
        if not state.corners:reason='no_initial_geometry_feedback_direction';break
        remaining=60.-(time.monotonic()-started)
        if remaining<=0:reason='total_budget_exhausted';break
        tracked=np.array(sorted(state.corners),int)
        linear=final_corner_linearization(grid,mapping,state.anchor,ctx.across,ctx.down,tracked)
        feedback=final_jacobian_soc_block(linear['e_rows'],linear['e_const'],linear['c_rows'],linear['c_const'],total)
        trial=assemble_trial(frozen,ledger.blocks,feedback,frozen['z0'],state.anchor,state.radius)
        description=[{'type':type(c).__name__,'dim':c.dim} for c in trial['cones']]
        trial_path=output/f'round-{iteration}';trial_path.mkdir()
        np.save(trial_path/'anchor-source.npy',state.anchor);np.save(trial_path/'targeted-corners.npy',tracked)
        save_npz(trial_path/'A.npz',trial['A']);np.savez_compressed(trial_path/'vectors.npz',b=trial['b'])
        save(trial_path/'cones.json',description)
        remaining=60.-(time.monotonic()-started)
        if remaining<=0:reason='total_budget_exhausted_before_solve';break
        settings=_settings(frozen['metadata'],remaining);solve_start=time.monotonic()
        solved=clarabel.DefaultSolver(frozen['P'],frozen['q'],trial['A'],trial['b'],trial['cones'],settings).solve()
        finite_number=lambda value:float(value) if np.isfinite(value) else None
        row={'round':iteration,'solver':{'status':str(solved.status),'seconds':time.monotonic()-solve_start,
            'iterations':solved.iterations,'primal_residual':finite_number(solved.r_prim),
            'dual_residual':finite_number(solved.r_dual),'time_limit':settings.time_limit},
            'subproblem_sha256':qp_fingerprint(frozen['P'],frozen['q'],trial['A'],trial['b'],description)}
        np.savez_compressed(trial_path/'solver-state.npz',primal_vector=solved.x,primal_slack=solved.s,dual_vector=solved.z)
        vector=np.asarray(solved.x,float);z=vector[:size];source=None;actual=None;bad=[];new=0
        complete_state=finite_solver_state(solved,total,len(trial['b']))
        row['solver']['complete_finite_primal_slack_dual']=complete_state
        if vector.shape==(total,) and np.isfinite(vector).all():
            np.savez_compressed(trial_path/'source-state.npz',scalar_solution=z,mapping=mapping)
            source=ctx.audit(z);row['source_audit']=source
            cut_report=ledger.add(ctx.additional_cuts(z,total));row['source_cut_exchange']=cut_report
            new=cut_report['new_cone_blocks']
            try:
                actual,field,bad=actual_audit(z,frozen,ctx,rgba,fixed,curves)
                np.save(trial_path/'inverse-field.npy',field);row['actual_audit']=actual
                row['actual_bad_corner_indices']=bad.tolist()
                try:row['prediction']=prediction_diagnostic(linear,z,field,grid,mapping,ctx.across)
                except ValueError as error:row['prediction_error']=str(error)
            except ValueError as error:row['fresh_inverse_rejection']=str(error)
        else:z=np.full(size,np.nan)
        decision_z=z if complete_state else np.full(size,np.nan)
        decision=state.observe(str(solved.status),decision_z,source,actual,bad,new);row['decision']=decision
        if not complete_state:decision['reason']='nonfinite_or_incomplete_solver_state'
        row['elapsed_seconds']=time.monotonic()-started
        if row['elapsed_seconds']>60.:
            state.accepted=False;decision.update(action='stop',reason='total_budget_exhausted_during_audit')
        report['rounds'].append(row);save(trial_path/'report.json',row)
        reason=decision['reason']
        if decision['action'] in ['stop','accept']:break
    else:reason='maximum_solve_count'
    report.update(termination=reason,solves=state.attempts,elapsed_seconds=time.monotonic()-started,
        numerical_gates_passed=bool(state.accepted),final_actual_corner_index_union=[list(c) for c in sorted(state.corners)],
        limits='Local reconstruction correction only. Full source bands, independent prefix caps, areas and fixed support are retained. Actual curves are sampled, not continuously certified; shape naturalness and photographic quality are unapproved.')
    np.save(output/'last-accepted-source-anchor.npy',state.anchor)
    if any(sha(path)!=digest for path,digest in {**hashes,**code_hashes}.items()):
        raise ValueError('frozen inputs or executed source code changed during feedback')
    save(output/'report.json',report)
    print({key:report[key] for key in ['termination','solves','elapsed_seconds','numerical_gates_passed']},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','interface','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.interface,args.output)
