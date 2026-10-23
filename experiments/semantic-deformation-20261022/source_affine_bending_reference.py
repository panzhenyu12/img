"""Source-affine reference for the existing discrete jaw bending term.

Only q (and an irrelevant energy constant) changes. The reference is neither
new position data nor new endpoint data; its nonzero endpoints must not be
clamped before differentiation. No constraints or Hessian weights change.
"""
import numpy as np


def discrete_reference_bending(nodes, reference, smoothing_length_px):
    nodes,reference=np.asarray(nodes,float),np.asarray(reference,float)
    ell=float(smoothing_length_px)
    if (nodes.ndim!=1 or len(nodes)<3 or reference.shape!=nodes.shape
            or not np.isfinite(nodes).all() or not np.isfinite(reference).all()
            or not np.isfinite(ell) or ell<0. or np.any(np.diff(nodes)<=0.)):
        raise ValueError('finite uniform source nodes, reference and nonnegative original smoothing length required')
    h=(nodes[-1]-nodes[0])/(len(nodes)-1)
    if not np.allclose(np.diff(nodes),h,atol=64*np.finfo(float).eps*max(1.,abs(nodes).max()),rtol=0.):
        raise ValueError('existing discrete bending requires uniform original source arclength')
    d2=np.diff(np.eye(len(nodes)),n=2,axis=0)/h**2
    weight=ell**4*h
    hessian=weight*(d2.T@d2)
    curvature=d2@reference
    return {'hessian':hessian,'linear_shift':-weight*(d2.T@curvature),
        'constant':float(weight*(curvature@curvature)),'d2':d2,
        'reference_curvature':curvature,'weight':float(weight)}


def source_affine_reference(problem,amount,fraction,smoothing_length_px):
    """Return one shared equal-original-jaw-area affine width coefficient.

    ``problem`` must be the unchanged OriginalProblem or PlateauProblem API.
    Its beta_area must retain the original continuous beta-cubed hard area.
    k is computed by two-point exact integration of linear radius on each
    original edge, not from sampled controls or a fitted current solution.
    The returned displacement is used only inside D2(u-reference).
    """
    if not np.isfinite([amount,fraction]).all() or abs(amount)>100. or not 0.<=fraction<=.45:
        raise ValueError('original finite amount and radius-fraction bounds required')
    across=np.asarray(problem.across,float);down=np.asarray(problem.down,float)
    if not np.allclose(np.array([across,down])@np.array([across,down]).T,np.eye(2),atol=1e-10,rtol=0.):
        raise ValueError('original orthonormal source frame required')
    unit_area=0.;per_side=[]
    for arc in problem.arcs:
        points=problem.source(arc,arc['knots'])
        radius=(points-problem.origin)@across-problem.midline
        delta=np.diff(points,axis=0);dv=across[0]*delta[:,1]-across[1]*delta[:,0]
        side=float(arc['direction']*np.sum(-.5*(radius[:-1]+radius[1:])*dv))
        unit_area+=side;per_side.append(side)
    if not np.isfinite(unit_area) or unit_area==0.:
        raise ValueError('original jaw area cannot determine an affine width reference')
    wanted=float(problem.beta_area(amount,fraction));k=wanted/unit_area
    curves=[]
    for arc,area in zip(problem.arcs,per_side):
        points=problem.source(arc,arc['nodes'])
        reference=-k*((points-problem.origin)@across-problem.midline)
        item=discrete_reference_bending(arc['nodes'],reference,smoothing_length_px)
        curves.append({'id':arc['id'],'reference_displacement':reference,**item,
            'continuous_affine_jaw_area_px2':float(k*area)})
    return {'curves':curves,'linear_shift':np.concatenate([c['linear_shift']for c in curves]),
        'constant':float(sum(c['constant']for c in curves)),
        'report':{'schema':'source-affine-jaw-discrete-bending-reference-v1',
          'width_fraction':float(k),'original_requested_jaw_area_px2':wanted,
          'unit_affine_jaw_area_px2':float(unit_area),'reference_jaw_area_px2':float(k*unit_area),
          'formula':'ell^4*h*||D2(u-delta_aff)||^2; delta_aff=-k*original_source_radius',
          'linear_shift_convention':'E=u.T H u + 2 l.T u + constant; delta_l=-H_bend*delta_aff',
          'solver_q_shift':'2*delta_l for 0.5*u.T P u + q.T u',
          'units':'u/reference in px, D2u in 1/px, ell^4*h in px^5, energy in px^3; linear coefficients in px^2',
          'hessian_changed':False,'plateau_position_preference_changed':False,
          'hard_zero_endpoints_changed':False,'new_weights':0,
          'reference_endpoints_clamped':False,'reference_adopted_as_position_targets':False}}
