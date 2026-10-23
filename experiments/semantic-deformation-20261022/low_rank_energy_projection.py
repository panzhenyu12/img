"""Small constraint-space correction of a frozen quadratic field.

Minimize (z-z0)' H (z-z0), subject to ||R_i z||² <= cap_i and
E(z-z0)=0. The correction lies in the constrained Green operator's
range applied to R'. This is a numerical Schur reduction, not a new
image prior, a ridge approximation, or a certificate for resampled pixels.
"""
import time
import clarabel
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, vstack
from scipy.sparse.linalg import splu


def _positive_modes(matrix):
    symmetric=(matrix+matrix.T)*.5
    values,vectors=np.linalg.eigh(symmetric)
    tolerance=64*np.finfo(float).eps*max(1,len(values))*float(abs(values).max(initial=0.))
    if values.min(initial=0.) < -tolerance:
        raise ValueError('constraint Gram matrix is not positive semidefinite')
    active=values>tolerance
    return values[active],vectors[:,active],tolerance


def project_energy_balls(hessian, baseline, blocks, *, preserve=None):
    """Return a finite diagnostic candidate and independent actual-row checks.

    H must be the caller's positive definite Dirichlet energy. Small eigenmodes
    of the constraint Gram are discarded at a reported roundoff scale; no
    diagonal regularizer is added. A candidate is rechecked against every
    original row. The preserved values are those of z0, not new requested values.
    """
    started=time.perf_counter();H=csc_matrix(hessian,dtype=float)
    z0=np.asarray(baseline,float)
    if z0.ndim!=1 or H.shape!=(len(z0),len(z0)) or not np.isfinite(z0).all() or not np.isfinite(H.data).all():
        raise ValueError('finite matching scalar field and Hessian required')
    asymmetry=H-H.T
    if abs(asymmetry.data).max(initial=0.)>1e-12*max(1.,abs(H.data).max(initial=0.)):
        raise ValueError('symmetric field energy required')
    operators=[csr_matrix(b['operator'],dtype=float) for b in blocks]
    caps=np.asarray([b['cap'] for b in blocks],float)
    if not len(operators) or not np.isfinite(caps).all() or np.any(caps<=0.) or any(
            r.shape[1]!=len(z0) or not r.shape[0] or not np.isfinite(r.data).all() for r in operators):
        raise ValueError('finite nonempty original operators and positive energy caps required')
    # Normalize whole balls before constructing the Gram matrix. Truncating
    # first would make numerical rank depend on the caller's physical units.
    R=vstack([r/np.sqrt(c) for r,c in zip(operators,caps)],format='csr')
    sizes=np.r_[0,np.cumsum([r.shape[0] for r in operators])]
    E=csr_matrix((0,len(z0))) if preserve is None else csr_matrix(preserve,dtype=float)
    if E.shape[1]!=len(z0) or not np.isfinite(E.data).all():
        raise ValueError('finite matching preserved linear measurements required')
    factor=splu(H);green=factor.solve(R.T.toarray())
    preserved_rank=0;etol=0.;Eunit=E
    if E.shape[0]:
        norms=np.sqrt(np.asarray(E.multiply(E).sum(axis=1)).ravel());nonzero=norms>0.
        Eunit=diags(1./norms[nonzero])@E[nonzero]
        egreen=factor.solve(Eunit.T.toarray());ev,eq,etol=_positive_modes(np.asarray(Eunit@egreen))
        preserved_rank=len(ev)
        if len(ev):green-=egreen@((eq/ev)@(eq.T@np.asarray(Eunit@green)))
    gram=np.asarray(R@green);values,vectors,threshold=_positive_modes(gram)
    transform=green@(vectors/np.sqrt(values)) if len(values) else np.empty((len(z0),0))
    reduced=np.asarray(R@transform);reference=np.asarray(R@z0)
    rank=len(values);solver_status='no_free_constraint_modes';coordinate=np.zeros(rank)
    if rank:
        rows=[];rhs=[];cones=[]
        for i in range(len(caps)):
            sl=slice(sizes[i],sizes[i+1])
            rows.append(csc_matrix(np.vstack([np.zeros(rank),-reduced[sl]])))
            rhs.append(np.r_[1.,reference[sl]])
            cones.append(clarabel.SecondOrderConeT(1+sizes[i+1]-sizes[i]))
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
        settings.time_limit=5.;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
        solved=clarabel.DefaultSolver(2*eye(rank,format='csc'),np.zeros(rank),
            vstack(rows,format='csc'),np.concatenate(rhs),cones,settings).solve()
        solver_status=str(solved.status)
        if np.shape(solved.x)!=(rank,) or not np.isfinite(solved.x).all():
            return {'solution':None,'report':{'solver_status':solver_status,'accepted':False}}
        coordinate=np.asarray(solved.x)
    z=z0+transform@coordinate;delta=z-z0
    energies=np.array([float((r@z)@(r@z)) for r in operators])
    preserved_error=float(abs(E@delta).max(initial=0.))
    preserved_unit_error=float(abs(Eunit@delta).max(initial=0.))
    metric=np.asarray(transform.T@(H@transform))
    orthogonality_error=float(abs(metric-np.eye(rank)).max(initial=0.))
    # Reconstruction checks are numerical diagnostics, never an image gate.
    report={'solver_status':solver_status,'accepted':bool(solver_status=='Solved' and
        np.all(energies<=caps) and preserved_unit_error<=1e-9 and orthogonality_error<=1e-6),
        'field_variables':len(z0),'constraint_rows':R.shape[0],'constraint_space_rank':rank,
        'gram_roundoff_rank_threshold':threshold,'preserved_rank':preserved_rank,
        'preserved_gram_roundoff_rank_threshold':etol,'preserved_unit_row_change_max':preserved_unit_error,
        'preserved_change_max':preserved_error,'energy_basis_error_max':orthogonality_error,
        'correction_energy':float(delta@(H@delta)),'correction_max':float(abs(delta).max(initial=0.)),
        'blocks':[{'cap':float(c),'actual_energy':float(e),'passed':bool(e<=c)} for c,e in zip(caps,energies)],
        'seconds':time.perf_counter()-started,'final_pixel_geometry_checked':False,'image_approval':False}
    return {'solution':z,'report':report}
