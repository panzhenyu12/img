"""One diagnostic source-QP step driven by actual stored inverse-cell failures.

The original objective and every frozen source constraint remain unchanged.
No photograph is rendered; every finite trial is independently re-inverted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import clarabel
import numpy as np
from PIL import Image
from scipy.sparse import csc_matrix, csr_matrix, eye, hstack, load_npz, save_npz, vstack

from inverse_field_linearization import linearize_inverse_samples
from joint_jaw_boundary_field import (GridCurve, OriginalProblem, _refine_regions,
    all_corner_inverse_check, qp_fingerprint, source_sign_regions)
from forward import cell_jacobians, inverse_points, source_geometry
from evaluate import inspect
from resampling import sample_rgba
from run_portable_jaw import snapshot_local_code


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def final_jacobian_soc_block(e_rows,e_const,c_rows,c_const,total_variables):
    """Exact FINAL rank-one Jacobian eigen/skew cones for affine E and C."""
    e_rows,c_rows=csr_matrix(e_rows),csr_matrix(c_rows)
    e_const,c_const=np.asarray(e_const,float),np.asarray(c_const,float)
    n,size=e_rows.shape
    if (c_rows.shape!=(n,size) or e_const.shape!=(n,) or c_const.shape!=(n,)
            or total_variables<size or not np.isfinite(e_rows.data).all()
            or not np.isfinite(c_rows.data).all() or not np.isfinite([e_const,c_const]).all()):
        raise ValueError('matching finite affine final-Jacobian coefficients required')
    rows=[];rhs=[];cones=[]
    for e,c,e0,c0 in zip(e_rows,c_rows,e_const,c_const):
        # s=b-Ax: radii E-.04 and 2.6-E, common vector [E-1,C].
        rows.extend([vstack([-e,-e,-c]),vstack([e,-e,-c]),vstack([c,-c])])
        rhs.extend([[e0-.04,e0-1.,c0],[2.6-e0,e0-1.,c0],[.48-c0,.48+c0]])
        cones.extend([clarabel.SecondOrderConeT(3),clarabel.SecondOrderConeT(3),clarabel.NonnegativeConeT(2)])
    matrix=vstack(rows,format='csr') if rows else csr_matrix((0,size))
    return {'A':hstack([matrix,csr_matrix((matrix.shape[0],total_variables-size))],format='csc'),
            'b':np.concatenate(rhs) if rhs else np.empty(0),'cones':cones}


def inverse_field(z,mapping,grid,across):
    scalar=np.zeros(len(mapping));active=mapping>=0;scalar[active]=z[mapping[active]]
    forward=(scalar[:,None]*across).reshape(grid['height'],grid['width'],2)
    spacing=np.asarray(grid['spacing']);source_check=source_geometry(forward/spacing)
    if source_check['strong_monotonicity_lower_bound']<=0 or not source_check['boundary_zero']:
        raise ValueError('source trial cannot be certified for inversion')
    width,height=((np.array([grid['width'],grid['height']])-1)*spacing+1).astype(int)
    yy,xx=np.mgrid[:height,:width];q=np.column_stack([xx.ravel(),yy.ravel()])/spacing
    inv,closure=inverse_points(forward/spacing,q)
    field=((inv-q)*spacing).reshape(height,width,2).astype('<f4')
    return field,{'closure_grid_units':closure,'closure_max_source_px':closure['closure_max_px']*spacing[0]}


def bad_final_corners(field):
    jac=cell_jacobians(field.astype(float));eigen=np.linalg.eigvalsh((jac+jac.swapaxes(-1,-2))/2.)
    skew=abs(jac[...,0,1]-jac[...,1,0])/2.;det=np.linalg.det(jac)
    return np.argwhere((eigen[...,0]<.52)|(eigen[...,1]>1.8)|(skew>.24)|(det<=0.))


def final_corner_linearization(grid,mapping,z,across,down,indices):
    # Each row is TL/TR/BL/BR: derive x and y edges from actual output pixels.
    pixels=[]
    for corner,y,x in indices:
        row_y=y+corner//2;column_x=x+corner%2
        pixels.extend([[x,row_y],[x+1,row_y],[column_x,y],[column_x,y+1]])
    points,inverse=np.unique(np.asarray(pixels,int),axis=0,return_inverse=True)
    ids=inverse.reshape(-1,4);destinations=points+np.asarray(grid['origin'])
    linear=linearize_inverse_samples(grid,mapping,z,across,destinations)
    r=linear['rows'];k=linear['constant']
    dx=r[ids[:,1]]-r[ids[:,0]];dy=r[ids[:,3]]-r[ids[:,2]]
    dx0=k[ids[:,1]]-k[ids[:,0]];dy0=k[ids[:,3]]-k[ids[:,2]]
    return {'e_rows':across[0]*dx+across[1]*dy,'e_const':1.+across[0]*dx0+across[1]*dy0,
        'c_rows':down[0]*dx+down[1]*dy,'c_const':down[0]*dx0+down[1]*dy0,
        'destinations':destinations,'linearization':linear}


def audit_source(reference,interface,mapping,z):
    geometry=reference['source_geometry'];frame=geometry['frame'];across=np.array(frame['across']);down=np.array(frame['down'])
    problem={'grid':reference['grid'],'mapping':mapping,'hessian':eye(len(z),format='csc')}
    radius=reference['jaw_radius_px'];old=OriginalProblem(geometry,2.,radius)
    jaws=[GridCurve(old.source(a,a['knots']),problem,across,down,source_knots=a['knots']) for a in old.arcs]
    full=GridCurve(np.vstack([geometry['contour'],geometry['contour'][0]]),problem,across,down)
    specifications=[];signs=[];zero_max=0.
    for arc,curve in zip(old.arcs,jaws):
        regions=_refine_regions(old.convex_regions(arc),curve.chord_partition(radius),arc['side'])
        specifications.append((arc['id'],curve,regions,radius,.5))
        regions,zeros=source_sign_regions(curve,old.origin,old.midline,reference['amount'])
        checked,_=curve.continuous_sign_check(z,None,regions);signs.append({'id':arc['id'],**checked})
        zero_max=max(zero_max,float(abs(curve.op(zeros)@z).max()))
    for segment in interface['segments']:
        curve=GridCurve(segment['source_points'],problem,across,down);radius=reference['interface_radius_px']
        specifications.append((segment['id'],curve,curve.source_turn_regions(radius),radius,0.))
    curves=[]
    for name,curve,regions,radius,rho in specifications:
        checked,_,_=curve.continuous_chord_check(z,regions,radius,rho);curves.append({'id':name,**checked})
    wanted=old.beta_area(reference['amount'],.13)
    if wanted!=reference['requested_jaw_and_full_contour_area_px2']:raise ValueError('original area request changed')
    jaw=sum(arc['direction']*float((curve.area_row()@z)[0]) for arc,curve in zip(old.arcs,jaws))
    full_area=float((full.area_row()@z)[0]);geometry_check,_=all_corner_inverse_check(problem,z,across)
    area_ok=max(abs(jaw-wanted),abs(full_area-wanted))<=max(1e-8,abs(wanted)*1e-10)
    result={'curves':curves,'sign':signs,'zero_max_px':zero_max,
        'jaw_area_px2':jaw,'full_contour_area_px2':full_area,'requested_area_px2':wanted,
        'jaw_area_residual_px2':jaw-wanted,'full_area_residual_px2':full_area-wanted,
        'all_source_corner_geometry':geometry_check,'accepted':bool(area_ok and zero_max<=1e-10
            and geometry_check['accepted'] and not any(c['violations'] for c in curves+signs))}
    return result,specifications


def sampled_actual_curve_audit(field,grid,specifications):
    """Failure-finding actual f32 chord samples; not a continuous certificate."""
    queries=[];pieces=[]
    for name,curve,regions,radius,rho in specifications:
        lo,hi,sign=regions.T;t=np.linspace(0.,1.,5)
        centers=((1.-t[:,None])*lo+t[:,None]*hi).ravel();sign=np.tile(sign,5)
        before=curve.source(centers-radius);middle=curve.source(centers)
        upper=centers+radius;upper=np.where(centers==curve.length-radius,curve.length,upper)
        after=curve.source(upper);source=np.vstack([before,middle,after])
        ns,_,_=curve.chord(centers,radius)
        queries.append(source);pieces.append((name,curve,len(centers),sign,ns,rho))
    actual,closure=inverse_points(field.astype(float),np.vstack(queries)-grid['origin'])
    actual+=grid['origin'];offset=0;rows=[]
    for name,curve,n,sign,ns,rho in pieces:
        p=actual[offset:offset+3*n].reshape(3,n,2);offset+=3*n
        u=p@curve.across;v=p@curve.down;h0=v[1]-v[0];h1=v[2]-v[1]
        numerator=(u[2]-u[1])*h0-(u[1]-u[0])*h1
        margin=sign*(numerator-rho*ns)
        rows.append({'id':name,'sampled_chords':n,'minimum_margin_px2':float(margin.min()),
            'violations':int(np.count_nonzero(margin < -1e-8))})
    return {'curves':rows,'sampled_passed':not any(row['violations'] for row in rows),
        'closure':closure,'continuous_approval':False,'sampling':'5 normalized centers per original source/grid chord piece'}


def run(reference_path,interface_path,output):
    reference_path,interface_path,output=[Path(p).resolve() for p in [reference_path,interface_path,output]]
    if output.exists():raise FileExistsError('refusing to overwrite frozen inverse feedback diagnostic')
    reference=read(reference_path);interface=read(interface_path)
    if reference['amount']!=100.:raise ValueError('this fixed experiment only accepts original +100')
    for path,digest in [(interface_path,reference['inputs_sha256'][str(interface_path)]),
            (reference_path.parent/'diagnostic-source-state.npz',reference['diagnostic_source_state_sha256']),
            (reference_path.parent/'diagnostic-qp/metadata.json',reference['diagnostic_qp_metadata_sha256'])]:
        if sha(path)!=digest:raise ValueError('frozen source/QP evidence changed')
    folder=reference_path.parent/'diagnostic-qp';metadata=read(folder/'metadata.json')
    for name,digest in metadata['files_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('frozen QP data changed')
    p,a=load_npz(folder/'P.npz'),load_npz(folder/'A.npz');vectors=np.load(folder/'vectors.npz');q,b=vectors['q'],vectors['b']
    if qp_fingerprint(p,q,a,b,metadata['cones'])!=metadata['sha256']:raise ValueError('QP fingerprint mismatch')
    state=np.load(reference_path.parent/'diagnostic-source-state.npz');z0=state['scalar_solution'];mapping=state['mapping']
    size=metadata['scalar_variables'];total=p.shape[0]
    if len(z0)!=size:raise ValueError('source scalar and auxiliary dimensions differ')
    grid=reference['grid'];across=np.array(reference['source_geometry']['frame']['across']);down=np.array(reference['source_geometry']['frame']['down'])
    source_path=Path(interface['source_image']);fixed_paths=[Path(path) for path in reference['inputs_sha256'] if path.endswith('/fixed-source.png')]
    if len(fixed_paths)!=1:raise ValueError('one original fixed mask required')
    fixed_path=fixed_paths[0]
    for path in [source_path,fixed_path]:
        if sha(path)!=reference['inputs_sha256'][str(path)]:raise ValueError('original source/fixed pixels changed')
    output.mkdir(parents=True);manifest,code_hashes=snapshot_local_code(output);started=time.monotonic()
    report={'experiment':'one-step-actual-inverse-jacobian-feedback-v1','reference':str(reference_path),
        'reference_sha256':sha(reference_path),'original_qp_sha256':metadata['sha256'],
        'source_variables':size,'auxiliary_variables':total-size,'trust_box_source_px':.25,
        'maximum_trials':1,'solve_budget_seconds':15.,'total_budget_seconds':60.,
        'source_reference_approved':reference['source_continuous_accepted'],
        'rendered':False,'executable':False,'image_approval':False,'executed_code_manifest':str(manifest)}
    initial,closure=inverse_field(z0,mapping,grid,across);initial_check=inspect(initial);bad=bad_final_corners(initial)
    report.update(initial_actual_f32_geometry=initial_check,initial_inverse_closure=closure,
        targeted_actual_corner_indices=bad.tolist())
    np.save(output/'initial-inverse-field.npy',initial)
    if not len(bad):raise ValueError('no actual stored inverse corner violates the original contract')
    linear=final_corner_linearization(grid,mapping,z0,across,down,bad)
    feedback=final_jacobian_soc_block(linear['e_rows'],linear['e_const'],linear['c_rows'],linear['c_const'],total)
    trust=hstack([eye(size,format='csr'),csr_matrix((size,total-size))],format='csr')
    trial_a=vstack([a,feedback['A'],trust,-trust],format='csc');trial_b=np.r_[b,feedback['b'],z0+.25,.25-z0]
    cones=[getattr(clarabel,c['type'])(c['dim']) for c in metadata['cones']]
    cones.extend(feedback['cones']);cones.append(clarabel.NonnegativeConeT(2*size))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=160
    # Use the successful v6 exchange precision, with every physical gate intact.
    settings.tol_feas=1e-12;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10
    remaining=60.-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('initial inversion/linearization exceeded total budget')
    settings.time_limit=min(15.,remaining)
    report['linearization']={'required_output_pixels':len(linear['destinations']),
        'minimum_source_cell_boundary_distance_px':float(linear['linearization']['source_cell_boundary_distance_px'].min()),
        'closure_max_source_px':linear['linearization']['closure_max_source_px'],
        'limitations':linear['linearization']['limits']}
    solve_started=time.monotonic();solved=clarabel.DefaultSolver(p,q,trial_a,trial_b,cones,settings).solve()
    report['solve']={'status':str(solved.status),'seconds':time.monotonic()-solve_started,
        'primal_residual':float(solved.r_prim),'dual_residual':float(solved.r_dual),
        'iterations':solved.iterations,'tol_feas':settings.tol_feas,'tol_gap_abs':settings.tol_gap_abs,'tol_gap_rel':settings.tol_gap_rel}
    np.savez_compressed(output/'diagnostic-solver-state.npz',primal_vector=solved.x,primal_slack=solved.s,dual_vector=solved.z)
    save_npz(output/'trial-A.npz',trial_a);np.savez_compressed(output/'trial-b.npz',b=trial_b)
    save(output/'trial-cones.json',[{'type':type(c).__name__,'dim':c.dim} for c in cones])
    if np.isfinite(solved.x).all():
        z=np.asarray(solved.x[:size]);np.savez_compressed(output/'diagnostic-source-state.npz',scalar_solution=z,mapping=mapping)
        source_check,specifications=audit_source(reference,interface,mapping,z)
        report['source_continuous_audit']=source_check;report['maximum_source_coefficient_step_px']=float(abs(z-z0).max())
        try:
            trial,closure=inverse_field(z,mapping,grid,across);np.save(output/'trial-inverse-field.npy',trial)
            report['actual_f32_geometry']=inspect(trial);report['trial_inverse_closure']=closure
            fixed=np.asarray(Image.open(fixed_path))>0;rgba=np.asarray(Image.open(source_path).convert('RGBA'))
            left,top=map(int,grid['origin']);height,width=trial.shape[:2]
            local_fixed=fixed[top:top+height,left:left+width];yy,xx=np.where(local_fixed)
            points=np.column_stack([xx+left,yy+top]);values,_=sample_rgba(rgba,points+trial[yy,xx],method='bilinear')
            sampled=np.floor(np.clip(values,0.,255.)+.5).astype(np.uint8)
            report['fixed_pixels']={'checked':len(points),'changed':int(np.any(sampled!=rgba[yy+top,xx+left],axis=1).sum()),
                'inverse_max_px':float(np.linalg.norm(trial[yy,xx],axis=1).max(initial=0.)),
                'outside_crop':'identity by construction; no image formed'}
            report['actual_sampled_curves']=sampled_actual_curve_audit(trial,grid,specifications)
        except ValueError as error:report['trial_inverse_rejection']=str(error)
    elapsed=time.monotonic()-started;report['elapsed_seconds']=elapsed
    report['numerical_gates_passed']=bool(solved.status==clarabel.SolverStatus.Solved
        and report.get('source_continuous_audit',{}).get('accepted',False)
        and report.get('actual_f32_geometry',{}).get('passed_strict',False)
        and report.get('maximum_source_coefficient_step_px',np.inf)<=.25
        and report.get('fixed_pixels',{}).get('changed',-1)==0
        and report.get('fixed_pixels',{}).get('inverse_max_px',-1)==0.
        and report.get('actual_sampled_curves',{}).get('sampled_passed',False) and elapsed<=60.)
    report['continuous_final_curve_approval']=False
    report['limits']='One local linearization before f32 rounding; all gates are measured after fresh inversion. Sampled actual curve checks cannot certify continuous final curves or beauty.'
    if any(sha(path)!=digest for path,digest in code_hashes.items()):raise ValueError('executed code changed during probe')
    save(output/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ['numerical_gates_passed','solve','actual_f32_geometry','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','interface','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.interface,args.output)
