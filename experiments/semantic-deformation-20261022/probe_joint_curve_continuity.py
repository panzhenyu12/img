"""One bounded source-shape experiment: remove jaw trace grid-slope jumps.

Append analytic C1 trace equalities to the complete frozen QP. The old target,
area, prefix budgets, supports and band stay fixed. The optional curvature
variance experiment additionally repairs unresolved bending in the objective;
it retains all old energy terms. No image export or application integration.
"""
import argparse
import copy
from pathlib import Path
import time

import clarabel
import numpy as np
from scipy.sparse import csr_matrix, hstack, save_npz, vstack, block_diag

from curve_grid_continuity import curve_grid_continuity
from joint_inverse_feedback_controller import ExactCutLedger, load_frozen
from joint_jaw_boundary_field import qp_fingerprint
from joint_source_constraints import SourceConstraintAudit
from probe_joint_inverse_feedback import save, sha
from run_portable_jaw import snapshot_local_code


def run(reference_path,interface_path,output,*,repair_curvature_variance=False,
        global_source_skew=False,interval_certificates=False):
    preparation_started=time.monotonic()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('refusing to overwrite frozen curve-continuity evidence')
    frozen=load_frozen(reference_path,interface_path)
    reference=frozen['reference'];size=len(frozen['z0']);total=frozen['P'].shape[0]
    if reference['amount']!=100.:raise ValueError('this fixed shape experiment requires original +100')
    audit=SourceConstraintAudit(reference,frozen['interface'],frozen['mapping'])
    interfaces=[row[1] for row in audit.specifications if row[0].endswith(':source-bend-lower')]
    observed=[*audit.jaws,*interfaces]
    if len(observed)!=len(reference['diagnostic_curves']):
        raise ValueError('same original observed source curves required')
    frozen_samples=[];sample_hashes={}
    for curve,row in zip(observed,reference['diagnostic_curves']):
        path=frozen['reference_path'].parent/f"diagnostic-curve-{row['index']}.npz"
        if sha(path)!=row['sha256']:raise ValueError('original curve samples changed')
        with np.load(path) as data:
            source=data['source'].copy();positions=data['source_arclength_px'].copy()
        if not np.array_equal(curve.source(positions),source):
            raise ValueError('original sampled positions differ from the unchanged curve')
        frozen_samples.append((curve,source,positions));sample_hashes[str(path)]=row['sha256']
    traces=[curve_grid_continuity(curve) for curve in audit.jaws]
    variance=[];p=frozen['P']
    if repair_curvature_variance:
        from curve_curvature_variance import curvature_variance
        for curve in audit.jaws:
            variance.append(curvature_variance(curve,
                reference['jaw_start_displacement_regularization']['length_px'],
                reference['interface_displacement_bending']['requested_node_spacing_px']))
        added=sum((item['hessian'] for item in variance),csr_matrix((size,size)))
        p=(p+block_diag([2.*added,csr_matrix((total-size,total-size))],format='csc')).tocsc()
    continuity=vstack([trace['A'] for trace in traces],format='csr')
    active=continuity.getnnz(axis=1)>0
    equality=continuity[active]
    equality=hstack([equality,csr_matrix((equality.shape[0],total-size))],format='csc')
    base={'A':vstack([frozen['A'],equality],format='csc'),
          'b':np.r_[frozen['b'],np.zeros(equality.shape[0])],
          'cones':[*frozen['cones'],clarabel.ZeroConeT(equality.shape[0])]}
    interval_exchange=None
    if interval_certificates:
        from source_interval_exchange import SourceIntervalExchange
        interval_exchange=SourceIntervalExchange(audit,
            reference['refinement'][-1]['interval_certificate_piece_keys'])
    global_skew=None
    if global_source_skew:
        from all_source_skew_constraints import all_source_skew_constraints
        global_skew=all_source_skew_constraints(audit.problem,audit.across,audit.down)
        skew_rows=hstack([global_skew['A'],csr_matrix((len(global_skew['b']),total-size))],format='csc')
        base={'A':vstack([base['A'],skew_rows],format='csc'),
              'b':np.r_[base['b'],global_skew['b']],
              'cones':[*base['cones'],*global_skew['cones']]}
    ledger=ExactCutLedger(base)
    # Existing source corner constraints must not be recounted as progress.
    initial_cut=ledger.add(audit.additional_cuts(frozen['z0'],total))
    if initial_cut['new_cone_blocks']:
        raise ValueError('reference source constraints must already cover its frozen failures')
    output.mkdir(parents=True)
    manifest,code_hashes=snapshot_local_code(output)
    started=time.monotonic()
    parameters=['amount','grid','source_geometry','jaw_radius_px','interface_radius_px',
        'interface_source_relative_bend_band','requested_jaw_and_full_contour_area_px2',
        'interface_displacement_bending','jaw_start_displacement_regularization',
        'jaw_start_intent_constraints']
    report={key:copy.deepcopy(reference[key]) for key in parameters}
    for key in ['interface_displacement_bending','jaw_start_displacement_regularization']:
        for old_metric in ['curves','total_energy_px3','comparison']:
            report[key].pop(old_metric,None)
    report['jaw_start_intent_constraints'].pop('final_check',None)
    report.update(schema='joint-jaw-grid-continuity-diagnostic-v1',reference=str(frozen['reference_path']),
        original_qp_sha256=frozen['metadata']['sha256'],inputs_sha256={**reference['inputs_sha256'],**frozen['hashes'],**sample_hashes},
        executed_code_manifest=str(manifest),executed_code_manifest_sha256=sha(manifest),
        source_continuous_accepted=False,executable=False,image_approval=False,
        geometry_only_source_acceptance=False,source_curve_continuity_accepted=False,
        maximum_rounds=4,total_solve_budget_seconds=60.,maximum_round_seconds=15.,refinement=[],
        original_constraints_removed=0,objective_changed=bool(repair_curvature_variance),requested_strength_changed=False,
        global_source_skew=global_skew['report'] if global_skew is not None else {'enabled':False},
        complete_new_interval_certificates=bool(interval_certificates),
        preparation_seconds=started-preparation_started,
        curvature_variance_repair={'enabled':bool(repair_curvature_variance),
            'original_discrete_bending_retained':True,'new_tunable_weights':0,
            'curves':[{'id':arc['id'],**item['report'],
                'baseline_energy_px3':float(np.linalg.norm(item['operator']@frozen['z0'])**2)}
                for arc,item in zip(audit.old.arcs,variance)]},
        known_source_jaw_grid_continuity={'scope':'jaw arcs only; inside each original edge, excluding original vertices',
            'curves':[{'id':arc['id'],**trace['report']} for arc,trace in zip(audit.old.arcs,traces)],
            'scalar_equations':int(equality.shape[0]),'maximum_allowed_equality_residual':1e-10,
            'baseline_maximum_abs_slope_jump':float(abs(continuity@frozen['z0']).max(initial=0.))},
        limits='C1 trace inside source edges only, not global C1 and not across original source vertices. '
               'Final pixel inverse geometry, actual curves and photographic beauty require independent inspection.')
    z=None;last_qp=None
    for iteration in range(4):
        remaining=60.-(time.monotonic()-started)
        if remaining<=0:report['rejection']='total_source_budget_exhausted';break
        blocks=[base,*ledger.blocks]
        a=vstack([block['A'] for block in blocks],format='csc')
        b=np.concatenate([block['b'] for block in blocks]);cones=[c for block in blocks for c in block['cones']]
        current_p=p;current_q=frozen['q']
        if interval_exchange is not None:
            expanded=interval_exchange.augment({'P':p,'q':frozen['q'],'A':a,'b':b,'cones':cones})
            current_p,current_q,a,b,cones=[expanded[k] for k in ['P','q','A','b','cones']]
        current_total=current_p.shape[0]
        description=[{'type':type(c).__name__,'dim':c.dim} for c in cones]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=160
        settings.tol_feas=1e-12;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10
        settings.time_limit=min(15.,remaining)
        begin=time.monotonic()
        solved=clarabel.DefaultSolver(current_p,current_q,a,b,cones,settings).solve()
        current_qp={'P':current_p,'q':current_q,'A':a,'b':b,'cones':description,'settings':{
            key:getattr(settings,key) for key in dir(settings)
            if not key.startswith('_') and not callable(getattr(settings,key))}}
        number=lambda value:float(value) if np.isfinite(value) else None
        row={'round':iteration,'status':str(solved.status),'solver_seconds':time.monotonic()-begin,
             'primal_residual':number(solved.r_prim),'dual_residual':number(solved.r_dual),
             'subproblem_sha256':qp_fingerprint(current_p,current_q,a,b,description),
             'total_variables':current_total,
             'added_interval_piece_keys':list(interval_exchange.keys) if interval_exchange is not None else []}
        report['refinement'].append(row)
        if (np.shape(solved.x)!=(current_total,) or np.shape(solved.s)!=(len(b),) or np.shape(solved.z)!=(len(b),)
                or not all(np.isfinite(value).all() for value in [solved.x,solved.s,solved.z])):
            report['rejection']='nonfinite_solver_state';break
        last_qp=current_qp;report['last_finite_round']=iteration
        z=np.asarray(solved.x[:size]);row['source_audit']=audit.audit(z)
        row['maximum_abs_slope_jump']=float(abs(continuity@z).max(initial=0.))
        row['continuity_passed']=row['maximum_abs_slope_jump']<=1e-10
        np.savez_compressed(output/'diagnostic-solver-state.npz',primal_vector=solved.x,
                            primal_slack=solved.s,dual_vector=solved.z)
        new=ledger.add(audit.additional_cuts(z,total));row['new_original_source_constraints']=new
        interval_new=interval_exchange.discover(z) if interval_exchange is not None else {}
        row['new_original_interval_constraints']=interval_new
        if (str(solved.status)=='Solved' and row['source_audit']['accepted'] and row['continuity_passed']):
            report['source_continuous_accepted']=True;break
        if str(solved.status) not in ['Solved','AlmostSolved']:
            report['rejection']='terminal_solver_status';break
        if not (new['new_cone_blocks'] or interval_new.get('new_certificates',0)
                or interval_new.get('new_factorized_pieces',0)):
            report['rejection']='failed_source_or_continuity_without_new_constraint';break
    if not report['source_continuous_accepted'] and 'rejection' not in report:
        report['rejection']='source_exchange_round_budget_exhausted'
    if z is not None:
        final=audit.audit(z)
        report.update(final_source_audit=final,jaw_area_px2=final['jaw_area_px2'],
            full_contour_area_px2=final['full_contour_area_px2'],
            geometry_only_source_acceptance=final['accepted'],
            source_curve_continuity_accepted=bool(abs(continuity@z).max(initial=0.)<=1e-10))
        report['jaw_start_intent_constraints']['final_check']=final['jaw_start_intent']
        report['known_source_jaw_grid_continuity']['final_maximum_abs_slope_jump']=float(abs(continuity@z).max(initial=0.))
        for row,item in zip(report['curvature_variance_repair']['curves'],variance):
            row['final_energy_px3']=float(np.linalg.norm(item['operator']@z)**2)
        np.savez_compressed(output/'diagnostic-source-state.npz',scalar_solution=z,mapping=frozen['mapping'])
        report['diagnostic_source_state_sha256']=sha(output/'diagnostic-source-state.npz')
        report['diagnostic_solver_state_sha256']=sha(output/'diagnostic-solver-state.npz')
        report['diagnostic_curves']=[]
        for index,(curve,source,positions) in enumerate(frozen_samples):
            target=source+(curve.op(positions)@z)[:,None]*audit.across
            path=output/f'diagnostic-curve-{index}.npz'
            np.savez_compressed(path,source=source,target=target,source_arclength_px=positions)
            report['diagnostic_curves'].append({'index':index,'sha256':sha(path)})
    if last_qp is not None:
        folder=output/'diagnostic-qp';folder.mkdir()
        save_npz(folder/'P.npz',last_qp['P']);save_npz(folder/'A.npz',last_qp['A'])
        np.savez_compressed(folder/'vectors.npz',q=last_qp['q'],b=last_qp['b'])
        metadata={'sha256':qp_fingerprint(last_qp['P'],last_qp['q'],last_qp['A'],last_qp['b'],last_qp['cones']),
                  'scalar_variables':size,'cones':last_qp['cones'],
                  'settings':last_qp['settings'],
                  'files_sha256':{name:sha(folder/name) for name in ['P.npz','A.npz','vectors.npz']}}
        save(folder/'metadata.json',metadata)
        report['diagnostic_qp_metadata_sha256']=sha(folder/'metadata.json')
    report['total_source_solve_seconds']=time.monotonic()-started
    if report['total_source_solve_seconds']>60.:
        report['source_continuous_accepted']=False;report['rejection']='total_source_budget_exhausted'
    if any(sha(path)!=digest for path,digest in {**report['inputs_sha256'],**code_hashes}.items()):
        raise ValueError('original evidence or executed source changed during experiment')
    save(output/'report.json',report)
    print({key:report.get(key) for key in ['source_continuous_accepted','rejection','total_source_solve_seconds']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curvature-variance',action='store_true',help='retain original bending and add its unresolved continuous-curvature variance')
    parser.add_argument('--global-source-skew',action='store_true',help='preassemble the unchanged skew bound on every active source corner')
    parser.add_argument('--interval-certificates',action='store_true',help='certify newly violated original curve pieces over their whole intervals')
    for name in ['reference','interface','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.interface,args.output,repair_curvature_variance=args.curvature_variance,
                               global_source_skew=args.global_source_skew,interval_certificates=args.interval_certificates)
