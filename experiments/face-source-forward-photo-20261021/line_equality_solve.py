"""Direct quadratic solve with all original homogeneous background-line rows."""
import time

import numpy as np
from scipy.linalg import qr
from scipy.sparse import block_diag,bmat,csc_matrix,diags
from scipy.sparse.linalg import splu


def stationarity_evidence(hessian,rhs,solution,line_force):
    """Certify x/y separately: a large component cannot hide the other."""
    h=csc_matrix(hessian);rhs=np.asarray(rhs,float);solution=np.asarray(solution,float)
    line_force=np.asarray(line_force,float)
    if rhs.shape!=solution.shape or line_force.shape!=rhs.shape or rhs.ndim!=2 or rhs.shape[1]!=2:
        raise ValueError('matching two-component source stationarity arrays required')
    residual=h@solution-rhs+line_force
    hnorm=float(np.asarray(abs(h).sum(axis=1)).max())
    denominator=hnorm*np.max(abs(solution),axis=0)+np.max(abs(rhs),axis=0)+np.max(abs(line_force),axis=0)
    maximum=np.max(abs(residual),axis=0)
    backward=maximum/np.maximum(denominator,np.finfo(float).tiny)
    return {'residual_max_by_component':maximum.tolist(),'denominator_by_component':denominator.tolist(),
        'backward_error_by_component':backward.tolist(),
        'passed':bool(np.isfinite(backward).all() and np.all(backward<=1e-10)),
        'tolerance':1e-10,'component_order':['x','y'],
        'shared_denominator_not_used':True}


def independent_rows(equalities):
    """RRQR on unit rows, retaining an audit of every original equality.

    Only columns occurring in a line enter the small dense rank problem.
    Zero rows and dependencies are excluded from the KKT factorization, never
    from the final complete-row residual check. No photograph is inspected.
    """
    original=csc_matrix(equalities,dtype=float).tocsr()
    if not np.isfinite(original.data).all():raise ValueError('finite equality rows required')
    norms=np.sqrt(np.asarray(original.multiply(original).sum(axis=1)).ravel())
    active=np.flatnonzero(norms>0)
    if not len(active):
        return csc_matrix((0,original.shape[1])),np.array([],int),norms,{
            'original_rows':original.shape[0],'exact_zero_rows':original.shape[0],
            'independent_rows':0,'active_columns':0,'selected_original_rows':[],
            'relative_rank_tolerance':None,'max_normalized_row_space_error':0.}
    normalized=(diags(1/norms[active])@original[active]).tocsr()
    columns=np.unique(normalized.indices);dense=normalized[:,columns].toarray().T
    q,r,pivots=qr(dense,mode='economic',pivoting=True,check_finite=True)
    diagonal=abs(np.diag(r));tolerance=np.finfo(float).eps*max(dense.shape)*diagonal.max(initial=0.)
    rank=int(np.count_nonzero(diagonal>tolerance));selected=active[pivots[:rank]]
    error=dense-q[:,:rank]@(q[:,:rank].T@dense)
    row_error=float(np.linalg.norm(error,axis=0).max(initial=0.))
    if row_error>max(1e-12,32*tolerance):
        raise ValueError('RRQR omitted a row outside the verified numerical row space')
    reduced=(diags(1/norms[selected])@original[selected]).tocsc()
    return reduced,selected,norms,{'original_rows':original.shape[0],
        'exact_zero_rows':original.shape[0]-len(active),'independent_rows':rank,
        'active_columns':len(columns),'selected_original_rows':selected.tolist(),
        'rank_threshold':float(tolerance),'rank_rule':'eps * max(local_matrix_dimensions) * max(abs(diag(R))) on unit original rows',
        'qr_diagonal_smallest_selected':float(diagonal[rank-1]) if rank else None,
        'qr_diagonal_largest_omitted':float(diagonal[rank]) if rank<len(diagonal) else None,
        'max_normalized_row_space_error':row_error,
        'final_verification_includes_all_original_rows':True}


def solve_equality_quadratic(hessian,rhs,equalities):
    """Solve the SPD source Gram objective with background-line equalities.

    No Jacobian cone is enforced. Caller must supply the validated positive
    definite Gram objective (source assembly has positive displacement energy
    on every independent variable). The status never implies geometric or
    aesthetic acceptance. Original H/RHS/equality coefficients remain unchanged.
    """
    started=time.monotonic();h=csc_matrix(hessian,dtype=float);rhs=np.asarray(rhs,float)
    n=h.shape[0];lines=csc_matrix(equalities,dtype=float)
    if n<1 or h.shape!=(n,n) or rhs.shape!=(n,2) or lines.shape[1]!=2*n:
        raise ValueError('matching scalar H/two-component RHS and packed line rows required')
    if not np.isfinite(h.data).all() or not np.isfinite(rhs).all():raise ValueError('finite original objective required')
    skew=h-h.T
    if skew.nnz and abs(skew.data).max()>1e-12*max(1.,abs(h.data).max()):raise ValueError('symmetric original H required')
    if np.any(h.diagonal()<=0):raise ValueError('positive source Gram diagonal required')
    reduced,selected,norms,reduction=independent_rows(lines)
    objective=block_diag([h,h],format='csc');linear=rhs.T.ravel();m=reduced.shape[0]
    kkt=bmat([[objective,reduced.T],[reduced,csc_matrix((m,m))]],format='csc')
    b=np.r_[linear,np.zeros(m)]
    factor_started=time.monotonic();factor=splu(kkt);factor_seconds=time.monotonic()-factor_started
    answer=factor.solve(b)
    if not np.isfinite(answer).all():raise ValueError('nonfinite equality quadratic result')
    vector=answer[:2*n];solution=vector.reshape(2,n).T
    multipliers=np.zeros(lines.shape[0]);multipliers[selected]=answer[2*n:]/norms[selected]
    full_residual=lines@vector
    normalized_residual=np.divide(full_residual,norms,out=np.zeros_like(full_residual),where=norms>0)
    line_force=lines.T@multipliers
    stationarity=objective@vector-linear+line_force
    residual=kkt@answer-b
    denominator=float(np.asarray(abs(kkt).sum(axis=1)).max())*abs(answer).max()+abs(b).max(initial=0.)
    backward=float(abs(residual).max(initial=0.)/max(denominator,np.finfo(float).tiny))
    component_stationarity=stationarity_evidence(h,rhs,solution,line_force.reshape(2,n).T)
    stationarity_backward=max(component_stationarity['backward_error_by_component'])
    normalized_max=float(abs(normalized_residual).max(initial=0.));raw_max=float(abs(full_residual).max(initial=0.))
    accepted=backward<=1e-10 and component_stationarity['passed'] and normalized_max<=1e-9
    report={'status':'EqualityQuadraticSolved' if accepted else 'EqualityQuadraticResidualRejected',
        'solver':'SciPy pivoted QR row-space reduction + sparse SuperLU KKT',
        'optimization_scope':'original SPD source quadratic with all original homogeneous background-line equalities; no Jacobian geometry constraints',
        'objective_convention':'original J=z^T H z-2 b^T z+constant; KKT uses J/2, so returned line multipliers are half-objective multipliers',
        'positive_definiteness_scope':'validated source Gram plus strictly positive displacement term on every independent variable; not inferred from LU success',
        'line_row_reduction':reduction,'all_original_line_rows_checked':lines.shape[0],
        'all_original_line_residual_max':raw_max,'all_original_unit_line_residual_max':normalized_max,
        'linear_backward_error':backward,'half_objective_stationarity_backward_error':stationarity_backward,
        'half_objective_stationarity_by_component':component_stationarity,
        'half_objective_stationarity_max':float(abs(stationarity).max(initial=0.)),
        'objective_without_constant':float(np.sum(solution*(h@solution))-2*np.sum(rhs*solution)),
        'factor_seconds':factor_seconds,'elapsed_seconds':time.monotonic()-started,
        'kkt_shape':list(kkt.shape),'kkt_nnz':kkt.nnz,'factor_L_nnz':factor.L.nnz,'factor_U_nnz':factor.U.nnz,
        'actual_geometry_pending':True,'source_convex_gate_not_enforced':True,
        'material_rgb_pending':True,'aesthetic_approval':False,'no_scaling_clipping_or_objective_reweighting':True}
    return {'solution':solution,'half_objective_line_multipliers':multipliers,
        'original_line_residuals':full_residual,'report':report}
