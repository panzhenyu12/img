"""Whole visible-interface target QP, isolated from rendering and anatomy.

A frozen preferred across displacement is interpolated with PCHIP. A compact
C2 cubic-spline correction may act only after an explicitly preserved upper
interval and before a preserved lower interval. This says nothing about C2
regularity of the original polyline, preferred interpolation, or final f32 map.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix


def _solve_whitened_qp(hessian,linear,matrix,thresholds,equality,normalization):
    """Solve the convex QP using the existing Clarabel dependency.

    The change of variables removes bending-energy conditioning. Row scaling
    changes neither feasible targets nor physical constraint tolerances; the
    continuous checks below still operate in original image coordinates.
    """
    import clarabel
    size=len(linear)
    factor=np.linalg.cholesky(hessian/normalization)
    transform=solve_triangular(factor.T,np.eye(size),lower=False)
    a=matrix@transform
    # Do not magnify almost-fixed rows. Near a compact spline endpoint their
    # small coefficients can otherwise turn ordinary physical bounds into
    # enormous RHS values, making a relative solver residual misleading.
    norms=np.maximum(np.linalg.norm(a,axis=1),1.)
    a=a/norms[:,None];b=thresholds/norms
    cones=[clarabel.NonnegativeConeT(len(b))]
    a=-a;b=-b
    if len(equality):
        eq=equality@transform;eq/=np.linalg.norm(eq,axis=1)[:,None]
        a=np.vstack([eq,a]);b=np.r_[np.zeros(len(eq)),b]
        cones.insert(0,clarabel.ZeroConeT(len(eq)))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200
    settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-10
    result=clarabel.DefaultSolver(csc_matrix(2*np.eye(size)),
        2*transform.T@linear/normalization,csc_matrix(a),b,cones,settings).solve()
    return SimpleNamespace(x=transform@np.asarray(result.x),
        success=result.status==clarabel.SolverStatus.Solved,message=str(result.status))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def _smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3.-2.*t)


def solve_boundary_curve(source, preferred_across, frame, *,
        preserve_through_down, neck_start_down, neck_end_down,
        preserve_suffix_px=8., node_spacing_px=8., smoothing_length_px=16.,
        neck_weight=4., neck_ramp_px=16., window_radius_px=12.,
        spacing_px=2., target_density=1., source_sign_threshold=1e-8,
        upper_preservation='pointwise',controller='source_turn_preserving',
        chord_radii_px=(4.,8.,16.),local_slope_change_max=.01):
    """Return source/target/weights, or target=None when constraints fail.

    Down coordinates never change. The upper preserved interval must be chosen
    from the edit's actual moving source region, not merely a stationary end.
    Lower known-neck preference is soft in the existing controllers. Explicit
    transport-only fairing has no neck-to-zero preference and requires zero
    neck weight; both end neighborhoods retain the frozen preferred displacement.
    Source finite-chord signs are geometric
    proxies; unknown anatomy is not inferred or relabelled.

    Curve-sign and displacement-bound constraints are linear at each location.
    Their between-location extrema are found from degree-four (turn numerator)
    and degree-three (displacement) polynomials; violating extrema add rows to
    the convex QP. Infeasibility returns no executable targets or attenuation.
    """
    p = np.asarray(source, float)
    preferred = np.asarray(preferred_across, float)
    across, down = [np.asarray(frame[k], float) for k in ['across', 'down']]
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or preferred.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(preferred).all()):
        raise ValueError('at least three finite source points and matching preferred scalars required')
    if (across.shape != (2,) or down.shape != (2,) or not np.isfinite([across, down]).all()
            or not np.allclose(np.array([across, down])@np.array([across, down]).T,
                               np.eye(2), rtol=0., atol=1e-10)):
        raise ValueError('orthonormal across/down frame required')
    options = [preserve_through_down, neck_start_down, neck_end_down,
        preserve_suffix_px, node_spacing_px, smoothing_length_px, neck_weight,
        neck_ramp_px, window_radius_px, spacing_px, target_density, source_sign_threshold]
    if not np.isfinite(options).all() or min(options[3:6]+options[7:]) <= 0.:
        raise ValueError('finite heights and strictly positive solver scales required')
    uv = p@np.array([across, down]).T
    if upper_preservation not in ['pointwise','integral_and_anchor']:
        raise ValueError('explicit pointwise or integral_and_anchor upper preservation required')
    if controller not in ['source_turn_preserving','gradient_bounded_motion_fairing',
                          'chord_bounded_motion_fairing','local_chord_motion_fairing',
                          'transport_only_motion_fairing']:
        raise ValueError('explicit isolated curve controller required')
    transport_only=controller=='transport_only_motion_fairing'
    if (transport_only and neck_weight!=0.) or (not transport_only and neck_weight<=0.):
        raise ValueError('transport-only requires zero neck weight; existing controllers require positive neck weight')
    motion_fairing=controller!='source_turn_preserving'
    local_chord_bounded=controller=='local_chord_motion_fairing'
    chord_bounded=controller in ['chord_bounded_motion_fairing','local_chord_motion_fairing']
    if local_chord_bounded and (not np.isfinite(local_slope_change_max)
            or local_slope_change_max<=0.):
        raise ValueError('positive finite local chord slope-change limit required')
    chord_radii=np.asarray(chord_radii_px,float)
    if chord_bounded and (chord_radii.ndim!=1 or not len(chord_radii)
            or not np.isfinite(chord_radii).all() or np.any(chord_radii<=0.)):
        raise ValueError('positive finite chord radii required')
    if np.any(np.diff(uv[:, 1]) <= 0.):
        raise ValueError('source interface must be strictly increasing along frame.down')
    edge_length = np.linalg.norm(np.diff(p, axis=0), axis=1)
    knots = np.r_[0., edge_length.cumsum()]; length = float(knots[-1])
    if chord_bounded and np.any(2*chord_radii>=length):
        raise ValueError('each chord diameter must fit the visible interface')
    # Independent BLAS projection calls can differ by an ulp at the same
    # endpoint; accept only this coordinate-roundoff margin, then snap back.
    if not uv[0, 1] < preserve_through_down < neck_start_down < neck_end_down <= uv[-1, 1]+1e-10:
        raise ValueError('ordered preserved-upper and known-neck down intervals required')
    neck_end_down=min(neck_end_down,float(uv[-1,1]))
    upper_end = float(np.interp(preserve_through_down, uv[:, 1], knots))
    active_start = upper_end if upper_preservation=='pointwise' else 0.
    active_end = length-preserve_suffix_px
    if active_end <= active_start or length <= 2*window_radius_px:
        raise ValueError('positive active interval and finite-chord support required')
    active_nodes = np.linspace(active_start, active_end,
        max(6, int(np.ceil((active_end-active_start)/node_spacing_px)))+1)
    spline_knots = np.r_[np.repeat(active_start, 4), active_nodes[1:-1], np.repeat(active_end, 4)]
    n_full = len(spline_knots)-4
    # Three zero endpoint coefficients give zero value, first and second
    # derivatives, so the correction extends by zero with C2 continuity.
    transform = np.eye(n_full)[:, 3:-3]
    basis_spline = BSpline(spline_knots, transform, 3, extrapolate=False)
    preferred_spline = PchipInterpolator(knots, preferred, extrapolate=False)
    size = transform.shape[1]
    def source_at(s):
        s = np.asarray(s, float)
        return np.column_stack([np.interp(s, knots, p[:, i]) for i in range(2)])
    def basis(s, derivative=0):
        s = np.asarray(s, float)
        inside = (s >= active_start)&(s <= active_end)
        result = np.zeros((len(s), size))
        result[inside] = basis_spline(s[inside], nu=derivative)
        return result
    def displacement(s, x):
        return preferred_spline(s)+basis(s)@x
    critical = np.unique(np.r_[knots, active_nodes])
    # PCHIP's first derivative is quadratic on each source edge. Its extrema
    # are the edge endpoints and roots of its linear second derivative.
    gradient_locations=list(knots)
    for i,(lo0,hi0) in enumerate(zip(knots[:-1],knots[1:])):
        c3,c2=preferred_spline.c[:2,i]
        if abs(c3)>1e-20:
            stationary=-c2/(3*c3)
            if 0.<stationary<hi0-lo0:gradient_locations.append(lo0+stationary)
    reference_gradient=preferred_spline(np.asarray(gradient_locations),1)
    gradient_lower=float(reference_gradient.min());gradient_upper=float(reference_gradient.max())
    # Exact-enough Gauss quadrature: cubic neck ramps times squared cubics have
    # degree nine on these intervals; six points also cover all other terms.
    neck_breaks = [neck_start_down, min(neck_end_down, neck_start_down+neck_ramp_px), neck_end_down]
    quadrature_cuts = np.unique(np.r_[critical, upper_end, np.interp(neck_breaks, uv[:, 1], knots)])
    gauss, gw = np.polynomial.legendre.leggauss(6)
    lo, hi = quadrature_cuts[:-1], quadrature_cuts[1:]
    q = ((lo+hi)[:, None]/2+(hi-lo)[:, None]/2*gauss).ravel()
    w = ((hi-lo)[:, None]/2*gw).ravel()
    b, d2 = basis(q), basis(q, 2)
    pref, pref2 = preferred_spline(q), preferred_spline(q, 2)
    qdown = source_at(q)@down
    neck = (np.zeros_like(q) if transport_only else
            _smoothstep((qdown-neck_start_down)/neck_ramp_px)*(qdown <= neck_end_down))
    hessian = b.T@((w*(1+neck_weight*neck))[:, None]*b)
    hessian += smoothing_length_px**4*d2.T@(w[:, None]*d2)
    linear = b.T@(w*neck_weight*neck*pref)+smoothing_length_px**4*d2.T@(w*pref2)
    equality=np.empty((0,size))
    upper_integral=float(w[q<upper_end]@pref[q<upper_end])
    if upper_preservation=='integral_and_anchor':
        equality=np.vstack([(w[q<upper_end,None]*b[q<upper_end]).sum(0),basis(np.array([upper_end]))[0]])
        norms=np.linalg.norm(equality,axis=1)
        equality=equality[norms>1e-13]/norms[norms>1e-13,None]
    normalization = length
    # Every shift-dependent source and cubic knot partitions the finite-chord
    # numerator into an exact degree-at-most-four polynomial in its center.
    r = window_radius_px
    turn_cuts = np.unique(np.r_[r, length-r,
        *[(critical+shift)[(critical+shift > r)&(critical+shift < length-r)] for shift in [-r, 0., r]]])
    def turn_data(centers, radius=None):
        r=window_radius_px if radius is None else radius
        a, b0, c = [source_at(centers+offset)@np.array([across, down]).T for offset in [-r, 0., r]]
        h0, h1 = b0[:, 1]-a[:, 1], c[:, 1]-b0[:, 1]
        original = (c[:, 0]-b0[:, 0])*h0-(b0[:, 0]-a[:, 0])*h1
        bp, bm, bz = basis(centers+r), basis(centers-r), basis(centers)
        row = (bp-bz)*h0[:, None]-(bz-bm)*h1[:, None]
        u0, u1, u2 = [preferred_spline(centers+offset) for offset in [-r, 0., r]]
        constant = original+(u2-u1)*h0-(u1-u0)*h1
        return original, constant, row
    known = []
    for a, z in zip(turn_cuts[:-1], turn_cuts[1:]):
        t = np.array([0., .5, 1.]); values = turn_data(a+(z-a)*t)[0]
        poly = np.polynomial.Polynomial.fit(t, values, 2, domain=[0., 1.]).convert()
        cuts = [0., 1.]
        for threshold in [-source_sign_threshold, source_sign_threshold]:
            for root in (poly-threshold).roots():
                if abs(root.imag) < 1e-9 and 0. < root.real < 1.:
                    cuts.append(float(root.real))
        cuts = sorted(set(cuts))
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            value = float(poly((x0+x1)/2))
            if abs(value) > source_sign_threshold:
                known.append((a+(z-a)*x0, a+(z-a)*x1, float(np.sign(value))))
    lower, upper = min(0., float(preferred.min())), max(0., float(preferred.max()))
    rows, rhs, row_info = [], [], []
    def add_turn(centers, signs):
        original, constant, matrix = turn_data(np.asarray(centers))
        rows.extend(np.asarray(signs)[:, None]*matrix)
        rhs.extend(-np.asarray(signs)*constant)
        row_info.extend({'kind':'source_turn','source_arclength_px':float(s),
            'source_turn_numerator_px2':float(a),'preferred_turn_numerator_px2':float(b)}
            for s,a,b in zip(centers,original,constant))
    def add_bound(positions):
        matrix = basis(np.asarray(positions)); value = preferred_spline(positions)
        rows.extend(matrix); rhs.extend(lower-value)
        rows.extend(-matrix); rhs.extend(value-upper)
        row_info.extend({'kind':kind,'source_arclength_px':float(s)}
            for kind in ['lower_bound','upper_bound'] for s in positions)
    def add_gradient(positions):
        matrix=basis(np.asarray(positions),1);value=preferred_spline(positions,1)
        rows.extend(matrix);rhs.extend(gradient_lower-value)
        rows.extend(-matrix);rhs.extend(value-gradient_upper)
        row_info.extend({'kind':kind,'source_arclength_px':float(s)}
            for kind in ['gradient_lower','gradient_upper'] for s in positions)
    if known and controller=='source_turn_preserving':
        add_turn([s for a,z,sgn in known for s in [a, (a+z)/2, z]],
                 [sgn for a,z,sgn in known for _ in range(3)])
    add_bound(np.unique(np.r_[critical, (critical[:-1]+critical[1:])/2]))
    if motion_fairing:
        add_gradient(np.unique(np.r_[critical,(critical[:-1]+critical[1:])/2,gradient_locations]))
    def extrema(a, z, evaluator, degree):
        t = np.linspace(0., 1., degree+1)
        coeff = np.polynomial.Polynomial.fit(t, evaluator(a+(z-a)*t), degree, domain=[0., 1.]).convert()
        real = [float(root.real) for root in coeff.deriv().roots()
                if abs(root.imag) < 1e-7 and 0. < root.real < 1.]
        positions = a+(z-a)*np.array([0., 1., *real])
        values = evaluator(positions)
        return positions, values
    # D_r is the source-relative finite-chord turn numerator, in px^2.
    # Its reference and candidate are degree <=4 between all shifted source,
    # PCHIP and correction knots. These are per-scale GLOBAL signed budgets,
    # not pointwise angle/curvature bounds or an anatomical convexity prior.
    chord_budgets=[]
    def induced_turn(s,radius,x=None):
        original,constant,matrix=turn_data(np.asarray(s),radius)
        return constant-original if x is None else constant-original+matrix@x
    def add_chord(positions,budget):
        positions=np.asarray(positions)
        original,constant,matrix=turn_data(positions,budget['radius_px'])
        preferred_turn=constant-original
        low,high=budget['preferred_signed_interval_px2']
        rows.extend(matrix);rhs.extend(low-preferred_turn)
        rows.extend(-matrix);rhs.extend(preferred_turn-high)
        row_info.extend({'kind':kind,'radius_px':budget['radius_px'],
            'source_arclength_px':float(s)} for kind in ['chord_lower','chord_upper']
            for s in positions)
    def chord_height_product(positions,radius):
        a,b,c=[source_at(np.asarray(positions)+offset)@down
               for offset in [-radius,0.,radius]]
        return (b-a)*(c-b)
    def local_turn(positions,radius,x):
        # Down coordinates are fixed, so only the correction contributes to
        # D_candidate-D_preferred. The positive denominator converts this
        # numerator to a difference of the two chord slopes, not an angle.
        matrix=turn_data(np.asarray(positions),radius)[2]
        return matrix@x,chord_height_product(positions,radius)
    def add_local_chord(positions,budget):
        positions=np.asarray(positions);radius=budget['radius_px']
        matrix=turn_data(positions,radius)[2]
        limit=local_slope_change_max*chord_height_product(positions,radius)
        rows.extend(matrix);rhs.extend(-limit)
        rows.extend(-matrix);rhs.extend(-limit)
        row_info.extend({'kind':kind,'radius_px':radius,'source_arclength_px':float(s)}
            for kind in ['local_chord_lower','local_chord_upper'] for s in positions)
    def local_ratio_extrema(a,z,radius,x):
        # Numerator degree <=4, positive denominator degree <=2. The roots of
        # N'H-NH' (degree <=5) locate extrema of the normalized local change.
        # This is also a check when very small down spans magnify raw errors.
        t=np.linspace(0.,1.,5);numerator,denominator=local_turn(a+(z-a)*t,radius,x)
        numerator=np.polynomial.Polynomial.fit(t,numerator,4,domain=[0.,1.]).convert()
        denominator=np.polynomial.Polynomial.fit(t,denominator,2,domain=[0.,1.]).convert()
        derivative=numerator.deriv()*denominator-numerator*denominator.deriv()
        roots=[float(root.real) for root in derivative.roots()
               if abs(root.imag)<1e-7 and 0.<root.real<1.]
        positions=a+(z-a)*np.array([0.,1.,*roots])
        actual,heights=local_turn(positions,radius,x)
        return positions,actual/heights
    if chord_bounded:
        for radius in np.unique(chord_radii):
            cuts=np.unique(np.r_[radius,length-radius,*[
                (critical+shift)[(critical+shift>radius)&(critical+shift<length-radius)]
                for shift in [-radius,0.,radius]]])
            locations=[];low=0.;high=0.
            for a,z in zip(cuts[:-1],cuts[1:]):
                position,value=extrema(a,z,lambda s:induced_turn(s,radius),4)
                locations.extend(position);low=min(low,float(value.min()));high=max(high,float(value.max()))
            budget={'radius_px':float(radius),'cuts':cuts,
                'preferred_signed_interval_px2':[low,high]}
            chord_budgets.append(budget)
            add_chord(np.unique(locations),budget)
            if local_chord_bounded:
                add_local_chord(np.unique(np.r_[locations,(cuts[:-1]+cuts[1:])/2]),budget)
    report = {'accepted_numerically': False, 'aesthetic_approval': False,
        'controller':controller,
        'qp_solver':'clarabel_cholesky_whitened' if chord_bounded else 'scipy_slsqp',
        'source_turn_signs_are_hard_constraints':controller=='source_turn_preserving',
        'preferred_signed_gradient_interval':[gradient_lower,gradient_upper],
        'chord_budget_kind':'global signed source-relative numerator, not pointwise angle' if chord_bounded else None,
        'chord_budgets':[{'radius_px':b['radius_px'],
            'preferred_signed_interval_px2':b['preferred_signed_interval_px2']} for b in chord_budgets],
        'source_geometry_role': 'visible interface only; no anatomical relabelling',
        'source_input_sha256': _digest({'source':p.tolist(), 'preferred_across':preferred.tolist(), 'frame':frame}),
        'correction_continuity': 'C2 compact spline only; no final field regularity claim',
        'upper_preservation':upper_preservation,
        'upper_reference_down_interval': [float(uv[0, 1]), float(preserve_through_down)],
        'upper_preferred_integral_px2':upper_integral,
        'upper_preferred_anchor_px':float(preferred_spline(upper_end)),
        'preserved_lower_arclength_interval_px': [active_end, length],
        'known_neck_down_interval': None if transport_only else [float(neck_start_down), float(neck_end_down)],
        'known_source_turn_subintervals':len(known), 'window_radius_px':r,
        'source_sign_threshold_px2':source_sign_threshold,
        'source_contour_unmodified':True, 'down_displacement_exact_zero':True,
        'neck_is_hard_fixed':False, 'edit_attenuation_allowed':False,
        'neck_zero_preference_used':not transport_only,
        'motion_objective':('integral (u-preferred)^2 + ell^4*(d2u/ds2)^2; no neck zero term'
                            if transport_only else 'preferred deviation, motion bending and soft neck-to-zero preference'),
        'smoothing_length_px':smoothing_length_px, 'neck_weight':neck_weight,
        'node_spacing_px':node_spacing_px, 'target_density':target_density,
        'refinement':[]}
    if local_chord_bounded:
        report.update({'local_chord_budget_kind':
            'pointwise correction chord-slope difference, not a true angle bound',
            'local_slope_change_max':float(local_slope_change_max),
            'local_slope_check_tolerance':1e-8})
        for record in report['chord_budgets']:
            record['local_slope_change_max']=float(local_slope_change_max)
    samples = np.unique(np.r_[np.linspace(0., length, int(np.ceil(length/spacing_px))+1), knots, active_start, active_end])
    output = {'source':source_at(samples), 'target':None, 'weights':None, 'report':report}
    x = np.zeros(size)
    for iteration in range(10):
        matrix, thresholds = np.asarray(rows), np.asarray(rhs)
        norms = np.linalg.norm(matrix, axis=1)
        fixed = norms <= 1e-13
        if np.any(thresholds[fixed] > 1e-7):
            report['rejection']='preserved preferred neighborhood conflicts with geometric constraints'
            report['fixed_constraint_conflicts']=[dict(row_info[i],required_slack=float(thresholds[i]))
                for i in np.flatnonzero(fixed&(thresholds>1e-7))]
            return output
        matrix, thresholds, norms = matrix[~fixed], thresholds[~fixed], norms[~fixed]
        scales=np.maximum(norms,1.) if chord_bounded else norms
        matrix, thresholds = matrix/scales[:, None], thresholds/scales
        feasible = linprog(np.zeros(size), A_ub=-matrix, b_ub=-thresholds,
                           A_eq=equality if len(equality) else None,
                           b_eq=np.zeros(len(equality)) if len(equality) else None,
                           bounds=[(None,None)]*size, method='highs')
        if not feasible.success:
            report['rejection']='constraint set infeasible; no attenuation or fallback'
            report['feasibility_message']=feasible.message
            if upper_preservation=='integral_and_anchor':
                turn_rows=np.asarray([item['kind']=='source_turn' for item in row_info])[~fixed]
                checks={}
                for name,select,eq in [
                    ('source_turns_and_upper_integral_without_range',turn_rows,equality[:1]),
                    ('source_turns_and_upper_anchor_without_range',turn_rows,equality[1:]),
                    ('source_turns_and_both_upper_equalities_without_range',turn_rows,equality),
                    ('range_and_upper_integral_without_anchor',np.ones(len(matrix),bool),equality[:1])]:
                    probe=linprog(np.zeros(size),A_ub=-matrix[select],b_ub=-thresholds[select],
                        A_eq=eq,b_eq=np.zeros(len(eq)),bounds=[(None,None)]*size,method='highs')
                    checks[name]={'feasible':bool(probe.success),'status':int(probe.status)}
                report['diagnostic_feasibility_only_no_targets']=checks
            return output
        if chord_bounded:
            solved=_solve_whitened_qp(hessian,linear,matrix,thresholds,equality,normalization)
        else:
            constraints=[LinearConstraint(matrix,thresholds,np.inf)]
            if len(equality):constraints.append(LinearConstraint(equality,0.,0.))
            solved = minimize(lambda z:float((z@hessian@z+2*linear@z)/normalization),
                x, jac=lambda z:2*(hessian@z+linear)/normalization,
                constraints=constraints, method='SLSQP',
                options={'maxiter':1000,'ftol':1e-11})
        x = solved.x
        violations = []; signed_min = np.inf
        for a,z,sign in known:
            position, value = extrema(a,z,lambda s:sign*(turn_data(s)[1]+turn_data(s)[2]@x),4)
            signed_min = min(signed_min, float(value.min()))
            violations.extend((float(s),sign) for s,v in zip(position,value) if v < -1e-7)
        diagnostic_turn_violations=len(violations)
        if motion_fairing:violations=[]
        bad_bound=[];bad_gradient=[];global_min,global_max=np.inf,-np.inf
        actual_gradient_min,actual_gradient_max=np.inf,-np.inf
        for a,z in zip(critical[:-1],critical[1:]):
            position,value=extrema(a,z,lambda s:displacement(s,x),3)
            global_min=min(global_min,float(value.min()));global_max=max(global_max,float(value.max()))
            bad_bound.extend(float(s) for s,v in zip(position,value) if v<lower-1e-7 or v>upper+1e-7)
            position,value=extrema(a,z,lambda s:preferred_spline(s,1)+basis(s,1)@x,2)
            actual_gradient_min=min(actual_gradient_min,float(value.min()))
            actual_gradient_max=max(actual_gradient_max,float(value.max()))
            if motion_fairing:
                bad_gradient.extend(float(s) for s,v in zip(position,value)
                    if v<gradient_lower-1e-8 or v>gradient_upper+1e-8)
        bad_chords=[];bad_local_chords=[]
        for budget,record in zip(chord_budgets,report['chord_budgets']):
            low,high=budget['preferred_signed_interval_px2'];bad=[]
            actual_low,actual_high=np.inf,-np.inf
            for a,z in zip(budget['cuts'][:-1],budget['cuts'][1:]):
                position,value=extrema(a,z,lambda s:induced_turn(s,budget['radius_px'],x),4)
                actual_low=min(actual_low,float(value.min()));actual_high=max(actual_high,float(value.max()))
                bad.extend(float(s) for s,v in zip(position,value) if v<low-1e-7 or v>high+1e-7)
            record['actual_signed_interval_px2']=[actual_low,actual_high]
            if bad:bad_chords.append((bad,budget))
            if local_chord_bounded:
                local_bad=[];local_max=0.;minimum_slack=np.inf;radius=budget['radius_px']
                for a,z in zip(budget['cuts'][:-1],budget['cuts'][1:]):
                    for sign in [-1.,1.]:
                        def slack(s):
                            correction,height=local_turn(s,radius,x)
                            return local_slope_change_max*height-sign*correction
                        position,value=extrema(a,z,slack,4)
                        minimum_slack=min(minimum_slack,float(value.min()))
                        local_bad.extend(float(s) for s,v in zip(position,value) if v < -1e-7)
                    position,value=local_ratio_extrema(a,z,radius,x)
                    local_max=max(local_max,float(abs(value).max()))
                    local_bad.extend(float(s) for s,v in zip(position,value)
                        if abs(v)>local_slope_change_max+1e-8)
                record['actual_max_abs_local_slope_change']=local_max
                record['min_local_budget_slack_px2']=minimum_slack
                if local_bad:bad_local_chords.append((np.unique(local_bad),budget))
        report['refinement'].append({'round':iteration,'solver_success':bool(solved.success),
            'solver_message':str(solved.message),'turn_violations':len(violations),
            'bound_violations':len(bad_bound),'gradient_violations':len(bad_gradient),
            'chord_budget_violations':sum(len(bad) for bad,_ in bad_chords),
            'local_chord_budget_violations':sum(len(bad) for bad,_ in bad_local_chords),
            'source_turn_violations_diagnostic':diagnostic_turn_violations})
        if not solved.success:
            report['rejection']='QP solver did not converge; no executable targets'
            return output
        if not violations and not bad_bound and not bad_gradient and not bad_chords and not bad_local_chords:
            report['accepted_numerically']=True
            report['min_signed_known_turn_numerator_px2']=None if not known else signed_min
            report['displacement_range_px']=[global_min,global_max]
            report['actual_signed_gradient_interval']=[actual_gradient_min,actual_gradient_max]
            report['source_turn_violations_diagnostic']=diagnostic_turn_violations
            break
        if violations:add_turn(*zip(*violations))
        if bad_bound:add_bound(bad_bound)
        if bad_gradient:add_gradient(bad_gradient)
        for bad,budget in bad_chords:add_chord(bad,budget)
        for bad,budget in bad_local_chords:add_local_chord(bad,budget)
    if not report['accepted_numerically']:
        report['rejection']='continuous constraint refinement exhausted'
        return output
    gradient_max=0.;preferred_gradient_max=0.
    for a,z in zip(critical[:-1],critical[1:]):
        _,vals=extrema(a,z,lambda s:preferred_spline(s,1)+basis(s,1)@x,2)
        gradient_max=max(gradient_max,float(abs(vals).max()))
        _,vals=extrema(a,z,lambda s:preferred_spline(s,1),2)
        preferred_gradient_max=max(preferred_gradient_max,float(abs(vals).max()))
    displacement_values=displacement(samples,x)
    line_weights=np.zeros(len(samples)); steps=np.diff(samples)
    line_weights[:-1]+=steps/2;line_weights[1:]+=steps/2
    output['target']=output['source']+displacement_values[:,None]*across
    # Every reviewed sample remains executable, including both endpoints.
    # Endpoint quadrature weights are positive half-interval measures.
    output['weights']=target_density*line_weights
    output['arclength_px']=samples
    output['preferred_across']=preferred_spline(samples)
    report.update({'max_abs_displacement_gradient':gradient_max,
        'preferred_max_abs_displacement_gradient':preferred_gradient_max,
        'upper_reference_displacement_range_px':[float(preferred_spline(samples[samples<=upper_end]).min()),
            float(preferred_spline(samples[samples<=upper_end]).max())],
        'upper_actual_integral_px2':float(w[q<upper_end]@displacement(q[q<upper_end],x)),
        'upper_actual_anchor_px':float(displacement(np.array([upper_end]),x)[0]),
        'sample_count':len(samples),'total_target_weight':float(output['weights'].sum()),
        'profile':{'source_arclength_px':knots.tolist(), 'preferred_across_px':preferred.tolist(),
                   'spline_knots_px':spline_knots.tolist(),'correction_coefficients':(transform@x).tolist()}})
    return output
