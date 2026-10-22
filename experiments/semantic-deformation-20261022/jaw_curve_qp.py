"""Experimental equal-area jaw target controller, without image rendering.

The source polyline is unchanged. Numerical constraints are not an aesthetic
approval and do not assign anatomy to unknown source geometry. This extracts
the v3 target-only experiment; its area reference now uses exact degree-seven
Gauss integration on each original source edge instead of a dense trapezoid.
NumPy/SciPy are offline research dependencies, not a Web runtime dependency.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from scipy.linalg import block_diag, null_space, solve_triangular
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.sparse import csc_matrix

from jaw_collocation import _prepare
from jaw_bilateral import build_bilateral_coupling


def _solve_precise_qp(hessian, linear, matrix, rhs, equality, equality_rhs,
                      lower, upper, normalization):
    """Numerically solve the same convex QP after SLSQP precision stalls.

    Fixed coordinates and the one already-feasible nonzero jaw-area equality
    are eliminated first; this is not a generic multi-equality feasibility API.
    Cholesky whitening changes variables, without a ridge, altered preference,
    relaxed inequality, or attenuated amount. Acceptance remains a separate
    check of the original physical constraints after reconstruction.
    """
    import clarabel
    fixed = lower == upper; free = ~fixed
    x = np.zeros(len(linear)); x[fixed] = lower[fixed]
    hf = hessian[np.ix_(free, free)]
    lf = linear[free]+hessian[np.ix_(free, fixed)]@x[fixed]
    eq = equality[:, free]; wanted = equality_rhs-equality[:, fixed]@x[fixed]
    particular = np.linalg.lstsq(eq, wanted, rcond=None)[0]
    null = null_space(eq)
    reduced_hessian = null.T@hf@null
    reduced_linear = null.T@(hf@particular+lf)
    if not len(reduced_linear):
        x[free] = particular
        feasible = (np.all(matrix@x >= rhs-1e-12) and np.all(x >= lower-1e-12)
                    and np.all(x <= upper+1e-12)
                    and np.allclose(equality@x, equality_rhs, rtol=0., atol=1e-12))
        return SimpleNamespace(x=x, success=bool(feasible), version=clarabel.__version__,
                               message='zero-dimensional equality solution checked in original coordinates')
    factor = np.linalg.cholesky(reduced_hessian/normalization)
    whiten = solve_triangular(factor.T, np.eye(len(reduced_linear)), lower=False)
    transform = null@whiten
    finite_lower = np.isfinite(lower[free]); finite_upper = np.isfinite(upper[free])
    identity = np.eye(int(free.sum()))
    physical_rows = np.vstack([matrix[:, free], identity[finite_lower], -identity[finite_upper]])
    physical_rhs = np.r_[rhs-matrix[:, fixed]@x[fixed],
                         lower[free][finite_lower], -upper[free][finite_upper]]
    a = physical_rows@transform
    b = physical_rhs-physical_rows@particular
    # As in the tested boundary QP: never magnify nearly fixed rows into a
    # huge RHS. Scaling does not replace the original-coordinate checks.
    norms = np.maximum(np.linalg.norm(a, axis=1), 1.)
    a = -a/norms[:, None]; b = -b/norms
    settings = clarabel.DefaultSettings(); settings.verbose = False; settings.max_iter = 200
    settings.tol_gap_abs = 1e-10; settings.tol_gap_rel = 1e-10; settings.tol_feas = 1e-12
    solved = clarabel.DefaultSolver(csc_matrix(2.*np.eye(len(reduced_linear))),
        2.*whiten.T@reduced_linear/normalization, csc_matrix(a), b,
        [clarabel.NonnegativeConeT(len(b))], settings).solve()
    x[free] = particular+transform@np.asarray(solved.x)
    return SimpleNamespace(x=x, success=solved.status == clarabel.SolverStatus.Solved,
                           message=str(solved.status), version=clarabel.__version__)


def _geometry_hash(geometry):
    return hashlib.sha256(json.dumps(geometry, sort_keys=True,
                                   separators=(',', ':')).encode('utf-8')).hexdigest()


def _merge_intervals(regions):
    merged = []
    for lo, hi in regions:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(hi))
        else:
            merged.append([float(lo), float(hi)])
    return merged


def _unknown_intervals(known, length):
    result, start = [], 0.
    for lo, hi in known:
        if lo > start:
            result.append([start, lo])
        start = hi
    if start < length:
        result.append([start, float(length)])
    return result


def _basis(nodes, s):
    s = np.asarray(s, float)
    i = np.clip(np.searchsorted(nodes, s, side='right')-1, 0, len(nodes)-2)
    t = (s-nodes[i])/(nodes[i+1]-nodes[i])
    result = np.zeros((s.size, len(nodes))); rows = np.arange(s.size)
    result[rows, i], result[rows, i+1] = 1.-t, t
    return result


def _quadratic(values):
    q0, qm, q1 = values
    a = 2.*(q1+q0-2.*qm)
    return a, q1-q0-a, q0


class _Problem:
    def __init__(self, geometry, spacing, window):
        prepared = _prepare(geometry, 0., .13, spacing)
        (self.p, self.edge, self.edge_length, self.cumulative,
         self.origin, self.across, self.midline, given_arcs, _) = prepared
        self.window = window
        self.down = np.asarray(geometry['frame'].get('down'), float)
        if (self.down.shape != (2,) or not np.isfinite(self.down).all()
                or not np.isclose(self.across@self.across, 1., rtol=0, atol=1e-10)
                or not np.isclose(self.down@self.down, 1., rtol=0, atol=1e-10)
                or not np.isclose(self.down@self.across, 0., rtol=0, atol=1e-10)
                or np.linalg.det(np.array([self.across, self.down])) <= 0.):
            raise ValueError('orthonormal across/down frame with positive orientation required')
        if set(a['id'] for a in given_arcs) != {'left-jaw', 'right-jaw'}:
            raise ValueError('exactly left-jaw and right-jaw reviewed arcs required')
        self.arcs = []
        for given in given_arcs:
            arc = dict(given)
            distance = ((self.cumulative[:-1]-arc['start'])*arc['direction']) % self.cumulative[-1]
            arc['knots'] = np.r_[0., np.sort(distance[(distance > 0.) & (distance < arc['length'])]), arc['length']]
            arc['nodes'] = np.linspace(0., arc['length'], max(2, int(np.ceil(arc['length']/spacing)))+1)
            arc['side'] = 1. if arc['id'] == 'left-jaw' else -1.
            v = (self.source(arc, arc['knots'])-self.origin)@self.down
            arc['nonmonotone'] = [[float(lo), float(hi)]
                for lo, hi, dv in zip(arc['knots'][:-1], arc['knots'][1:], np.diff(v)) if dv <= 0.]
            self.arcs.append(arc)
        # The shared parser protects original interior rows; additionally reject
        # overlap entirely within a long original edge, which has no interior row.
        cuts = np.unique(np.r_[0., self.cumulative[-1], *[
            [a['start'], (a['start']+a['direction']*a['length']) % self.cumulative[-1]] for a in self.arcs]])
        for s in (cuts[:-1]+cuts[1:])/2:
            if sum(0. < ((s-a['start'])*a['direction']) % self.cumulative[-1] < a['length'] for a in self.arcs) > 1:
                raise ValueError('reviewed jaw arc interiors overlap')

    def source(self, arc, s):
        s = np.asarray(s, float)
        position = (arc['start']+arc['direction']*s) % self.cumulative[-1]
        ids = np.minimum(np.searchsorted(self.cumulative, position, side='right')-1, len(self.p)-1)
        points = self.p[ids]+((position-self.cumulative[ids])/self.edge_length[ids])[:, None]*self.edge[ids]
        points[s == 0.] = arc['first']; points[s == arc['length']] = arc['last']
        return points

    def beta(self, arc, s, amount, fraction):
        t = np.asarray(s)/arc['length']
        radius = (self.source(arc, s)-self.origin)@self.across-self.midline
        return -amount/100.*fraction*radius*64.*(t*(1.-t))**3

    def beta_area(self, amount, fraction):
        # On an original straight edge radius is affine in s, beta is degree
        # seven and dv/ds constant. Four Gauss nodes integrate exactly in real
        # arithmetic; adding collinear detector rows cannot change the intent.
        xi, weights = np.polynomial.legendre.leggauss(4)
        area = 0.
        for arc in self.arcs:
            for lo, hi in zip(arc['knots'][:-1], arc['knots'][1:]):
                s = (lo+hi)/2.+(hi-lo)/2.*xi
                edge = (self.source(arc, np.array([hi]))-self.source(arc, np.array([lo])))[0]
                dv = float(self.across[0]*edge[1]-self.across[1]*edge[0])
                area += arc['direction']*dv/2.*float(weights@self.beta(arc, s, amount, fraction))
        return float(area)

    def area_row(self, arc):
        union = np.unique(np.r_[arc['knots'], arc['nodes']])
        b = _basis(arc['nodes'], union)
        delta = np.diff(self.source(arc, union), axis=0)
        dv = self.across[0]*delta[:, 1]-self.across[1]*delta[:, 0]
        return arc['direction']*(.5*dv[:, None]*(b[1:]+b[:-1])).sum(0)

    def chord_data(self, arc, centers):
        q0, q1, q2 = [self.source(arc, centers+shift) for shift in [-self.window, 0., self.window]]
        uv0, uv1, uv2 = [(q-self.origin)@np.array([self.across, self.down]).T for q in [q0, q1, q2]]
        return uv0, uv1, uv2, uv1[:, 1]-uv0[:, 1], uv2[:, 1]-uv1[:, 1]

    def numerator(self, arc, centers, values=None):
        uv0, uv1, uv2, h0, h1 = self.chord_data(arc, centers)
        if values is not None:
            for uv, shift in [(uv0, -self.window), (uv1, 0.), (uv2, self.window)]:
                uv[:, 0] += _basis(arc['nodes'], centers+shift)@values
        return arc['side']*((uv2[:, 0]-uv1[:, 0])*h0-(uv1[:, 0]-uv0[:, 0])*h1), h0*h1

    def turn_rows(self, arc, centers):
        _, _, _, h0, h1 = self.chord_data(arc, centers)
        rows = arc['side']*((_basis(arc['nodes'], centers+self.window)-_basis(arc['nodes'], centers))/h1[:, None]
               -(_basis(arc['nodes'], centers)-_basis(arc['nodes'], centers-self.window))/h0[:, None])
        numerator, denominator = self.numerator(arc, centers)
        return rows, numerator/denominator

    def convex_regions(self, arc):
        if arc['length'] <= 2*self.window:
            return np.empty((0, 2))
        critical = np.unique(np.concatenate([np.r_[arc['knots'], arc['nodes']]+shift
                                            for shift in [-self.window, 0., self.window]]))
        critical = np.unique(np.r_[self.window,
            critical[(critical > self.window) & (critical < arc['length']-self.window)], arc['length']-self.window])
        regions = []
        for lo, hi in zip(critical[:-1], critical[1:]):
            if hi-lo < 1e-10:
                continue
            middle = (lo+hi)/2.
            if any(middle+self.window > a and middle-self.window < b for a, b in arc['nonmonotone']):
                continue
            values, denominator = self.numerator(arc, np.array([lo, middle, hi]))
            if denominator.min() <= 0.:
                continue
            a, b, c = _quadratic(values)
            roots = np.roots([a, b, c]) if abs(a) > 1e-10 else (np.array([-c/b]) if abs(b) > 1e-10 else np.array([]))
            inside = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0. < r.real < 1.)
            for t0, t1 in zip([0., *inside], [*inside, 1.]):
                m = (t0+t1)/2.
                if (a*m+b)*m+c > 1e-9:
                    regions.append([lo+(hi-lo)*t0, lo+(hi-lo)*t1])
        return np.asarray(regions).reshape(-1, 2)

    def continuous_minima(self, arc, values):
        regions = self.convex_regions(arc)
        if len(regions) == 0:
            return np.array([]), np.array([]), np.array([])
        lo, hi = regions.T
        a, b, c = _quadratic(self.numerator(arc, np.concatenate([lo, (lo+hi)/2, hi]), values)[0].reshape(3, -1))
        vertex = np.divide(-b, 2*a, out=np.full_like(a, .5), where=abs(a) > 1e-14)
        vertex = np.where((a > 0.) & (vertex > 0.) & (vertex < 1.), vertex, 0.)
        ts = np.stack([np.zeros_like(a), np.ones_like(a), vertex])
        q = (a[None, :]*ts+b[None, :])*ts+c[None, :]
        t = ts[q.argmin(0), np.arange(len(a))]
        centers = lo+(hi-lo)*t
        numerator, denominator = self.numerator(arc, centers, values)
        return centers, numerator, numerator/denominator


def solve_jaw_curve_profile(geometry, amount, *, max_radius_fraction=.13,
                            node_spacing_px=2., smoothing_length_px=4.,
                            window_radius_px=16., bilateral_weight=0.):
    """Return hash-bound piecewise-linear targets and a numerical report.

    Positive amount shrinks toward the reviewed frame's midline. The signed
    continuous beta area is a hard equality; smoothing redistributes that area,
    never silently reduces amount. Only previously source-convex, height-
    monotone finite-chord windows receive no-new-reversal constraints. A failed
    solve returns no executable profiles, not a relaxed or attenuated edit.
    """
    values = [amount, max_radius_fraction, node_spacing_px, smoothing_length_px, window_radius_px, bilateral_weight]
    if (not np.isfinite(values).all() or abs(amount) > 100. or not 0. <= max_radius_fraction <= .45
            or node_spacing_px <= 0. or smoothing_length_px < 0. or window_radius_px <= 0. or bilateral_weight < 0.):
        raise ValueError('finite amount/radius bounds and positive spacing/window, nonnegative smoothing/bilateral weight required')
    problem = _Problem(geometry, node_spacing_px, window_radius_px)
    arcs = problem.arcs
    offset = np.r_[0, np.cumsum([len(a['nodes']) for a in arcs])]; size = int(offset[-1])
    area = np.concatenate([problem.area_row(a) for a in arcs])
    wanted = problem.beta_area(amount, max_radius_fraction)
    beta = np.concatenate([problem.beta(a, a['nodes'], amount, max_radius_fraction) for a in arcs])
    hessians, weights, rows, rhs = [], [], [], []
    for j, arc in enumerate(arcs):
        n = len(arc['nodes']); h = arc['length']/(n-1)
        w = np.full(n, h); w[[0, -1]] *= .5
        d2 = np.diff(np.eye(n), n=2, axis=0)/h**2
        hessians.append(np.diag(w)+smoothing_length_px**4*h*(d2.T@d2)); weights.append(w)
        regions = problem.convex_regions(arc)
        if len(regions):
            centers = np.unique(np.r_[regions.ravel(), regions.mean(axis=1)])
            local, original = problem.turn_rows(arc, centers)
            full = np.zeros((len(centers), size)); full[:, offset[j]:offset[j+1]] = local
            rows.append(full); rhs.append(-original)
    matrix = np.vstack(rows) if rows else np.empty((0, size))
    bounds_rhs = np.concatenate(rhs) if rhs else np.array([])
    hessian = block_diag(*hessians); linear = -np.concatenate(weights)*beta
    pairing = build_bilateral_coupling(problem, _basis) if bilateral_weight > 0. else None
    if pairing is not None:
        hessian += bilateral_weight*pairing.hessian
    normalization = sum(a['length'] for a in arcs)
    lower, upper = np.where(beta > 0., 0., -np.inf), np.where(beta < 0., 0., np.inf)
    lower[beta == 0.] = upper[beta == 0.] = 0.
    for j in range(len(arcs)):
        lower[offset[j]] = upper[offset[j]] = lower[offset[j+1]-1] = upper[offset[j+1]-1] = 0.
    area_norm = np.linalg.norm(area)
    if area_norm <= 1e-14:
        raise ValueError('jaw area constraint has no observable height support')
    equality = area[None, :]/area_norm
    norms = np.linalg.norm(matrix, axis=1)
    feasible = linprog(np.zeros(size), A_ub=-matrix/norms[:, None] if len(matrix) else None,
        b_ub=-bounds_rhs/norms if len(matrix) else None, A_eq=equality, b_eq=[wanted/area_norm],
        bounds=list(zip(lower, upper)), method='highs')
    report = {'feasibility_success': bool(feasible.success), 'feasibility_message': feasible.message,
              'accepted_numerically': False, 'aesthetic_approval': False, 'image_approval': False,
              'smoothing_length_px': smoothing_length_px, 'node_spacing_px': node_spacing_px,
              'window_radius_px': window_radius_px, 'source_contour_unmodified': True,
              'area_quadrature': '4-point Gauss on every original edge; exact degree-seven beta integrand',
              'constraints': 'continuous beta area; endpoints; edit sign; no new reverse finite-chord slope only on source-convex height-monotone windows',
              'refinement': []}
    if pairing is not None:
        report['bilateral_coupling'] = pairing.describe()
        report['bilateral_coupling'].update(weight=float(bilateral_weight),
            reference_kind='piecewise-linear beta control preference; not a lambda-zero solution',
            reference_mismatch=pairing.mismatch(beta))
    output = {'schema': 'jaw-arclength-piecewise-linear-profile-v1',
              'controller': 'equal-area-source-convex-window-qp-v4',
              'amount': float(amount), 'max_radius_fraction': float(max_radius_fraction),
              'area_reference': 'continuous_beta', 'required_signed_area_delta_px2': wanted,
              'source_geometry_sha256': _geometry_hash(geometry), 'source_geometry': copy.deepcopy(geometry),
              'geometry_hash_encoding': "UTF-8 json.dumps(source_geometry, sort_keys=True, separators=(',', ':')); default ensure_ascii=True",
              'across_axis': problem.across.tolist(), 'profiles': [], 'report': report}
    if pairing is not None:
        output['controller'] = 'equal-area-source-convex-window-bilateral-qp-v1'
    if not feasible.success:
        return output
    objective = lambda x: float((x@hessian@x+2.*linear@x)/normalization)
    gradient = lambda x: 2.*(hessian@x+linear)/normalization
    x = beta.copy(); solver_success = amount == 0. or max_radius_fraction == 0.
    if not solver_success:
        for iteration in range(8):
            norms = np.linalg.norm(matrix, axis=1)
            constraints = [LinearConstraint(equality, [wanted/area_norm], [wanted/area_norm])]
            if len(matrix):
                constraints.append(LinearConstraint(matrix/norms[:, None], bounds_rhs/norms, np.inf))
            solved = minimize(objective, x, jac=gradient, bounds=Bounds(lower, upper),
                constraints=constraints, method='SLSQP', options={'ftol': 1e-11, 'maxiter': 1000})
            x = solved.x; solver_success = bool(solved.success)
            added, added_rhs, counts = [], [], []
            for j, arc in enumerate(arcs):
                centers, numerator, _ = problem.continuous_minima(arc, x[offset[j]:offset[j+1]])
                bad = numerator < -1e-8; counts.append(int(bad.sum()))
                if np.any(bad):
                    local, original = problem.turn_rows(arc, centers[bad])
                    full = np.zeros((int(bad.sum()), size)); full[:, offset[j]:offset[j+1]] = local
                    added.append(full); added_rhs.append(-original)
            report['refinement'].append({'round': iteration, 'solver_success': solver_success,
                'active_rows': len(bounds_rhs), 'continuous_violating_subintervals': counts})
            report['solver_message'] = str(solved.message)
            if not any(counts) or not solver_success:
                break
            matrix = np.vstack([matrix, *added]); bounds_rhs = np.r_[bounds_rhs, *added_rhs]
        # Retain the exact old path for candidates that already pass. A solver
        # success flag cannot override a physical continuous-constraint failure.
        if solver_success and any(counts):
            fallback = {'method': 'clarabel_cholesky_whitened_same_qp',
                'reason': 'SLSQP reported success but continuous cuts still failed after refinement',
                'same_objective_and_constraints': True, 'refinement': []}
            report['precision_fallback'] = fallback
            for iteration in range(8):
                try:
                    solved = _solve_precise_qp(hessian, linear, matrix, bounds_rhs,
                        equality, np.array([wanted/area_norm]), lower, upper, normalization)
                except (ImportError, np.linalg.LinAlgError) as error:
                    fallback['error'] = str(error); solver_success = False
                    report['solver_message'] = 'precision fallback failed: '+str(error)
                    break
                x = solved.x; solver_success = bool(solved.success)
                fallback['clarabel_version'] = solved.version
                report['solver_message'] = 'precision fallback: '+solved.message
                added, added_rhs, counts = [], [], []
                minimum_numerators = []
                for j, arc in enumerate(arcs):
                    centers, numerator, _ = problem.continuous_minima(arc, x[offset[j]:offset[j+1]])
                    minimum_numerators.append(float(numerator.min()) if len(numerator) else None)
                    bad = numerator < -1e-8; counts.append(int(bad.sum()))
                    if np.any(bad):
                        local, original = problem.turn_rows(arc, centers[bad])
                        full = np.zeros((int(bad.sum()), size)); full[:, offset[j]:offset[j+1]] = local
                        added.append(full); added_rhs.append(-original)
                fallback['refinement'].append({'round': iteration, 'solver_success': solver_success,
                    'solver_message': solved.message, 'active_rows': len(bounds_rhs),
                    'continuous_violating_subintervals': counts,
                    'continuous_min_numerator': minimum_numerators,
                    'signed_area_error_px2': float(area@x-wanted)})
                if not solver_success or not any(counts):
                    break
                matrix = np.vstack([matrix, *added]); bounds_rhs = np.r_[bounds_rhs, *added_rhs]
    candidate_area = float(area@x)
    report['solver_success'] = solver_success
    report['signed_area_error_px2'] = candidate_area-wanted
    if pairing is not None:
        coupling_report = report['bilateral_coupling']
        coupling_report['result_mismatch'] = pairing.mismatch(x)
        coupling_report['per_side'] = []
        for j, arc in enumerate(arcs):
            nodes = np.unique(np.r_[arc['knots'], arc['nodes']])
            source = problem.source(arc, nodes)
            row = {'arc_id': arc['id']}
            for name, vector in [('reference', beta), ('result', x)]:
                local = vector[offset[j]:offset[j+1]]
                displacement = _basis(arc['nodes'], nodes)@local
                peak = int(np.argmax(np.abs(displacement)))
                row[name] = {'signed_area_delta_px2': float(problem.area_row(arc)@local),
                    'peak_across_displacement_px': float(displacement[peak]),
                    'peak_source_arclength_px': float(nodes[peak]),
                    'peak_source_down_height_px': float((source[peak]-problem.origin)@problem.down),
                    'peak_source_xy_px': source[peak].tolist()}
            row['peak_down_change_from_reference_px'] = (row['result']['peak_source_down_height_px']
                                                       -row['reference']['peak_source_down_height_px'])
            coupling_report['per_side'].append(row)
    profile_rows = []
    for j, arc in enumerate(arcs):
        nodes = np.unique(np.r_[arc['knots'], arc['nodes']]); local = x[offset[j]:offset[j+1]]
        known = _merge_intervals(problem.convex_regions(arc))
        profile_rows.append({'arc_id': arc['id'], 'arclength_px': nodes.tolist(),
            'across_displacement_px': (_basis(arc['nodes'], nodes)@local).tolist(),
            'original_source_vertex_arclength_px': arc['knots'].tolist(), 'arc_length_px': arc['length'],
            'direction': arc['direction'], 'original_contour_start_arclength_px': arc['start'],
            'known_source_convex_monotone_center_intervals_px': known,
            'unknown_center_intervals_px': _unknown_intervals(known, arc['length']),
            'source_nonmonotone_height_intervals_px': arc['nonmonotone']})
    checked = evaluate_profile_constraints(geometry, profile_rows, window_radius_px=window_radius_px)
    report['continuous_check'] = checked
    endpoint_max = max(abs(p['across_displacement_px'][i]) for p in profile_rows for i in [0, -1])
    sign_violation = float(np.maximum(np.maximum(lower-x, x-upper), 0.).max(initial=0.))
    report['endpoint_displacement_max_px'] = endpoint_max; report['sign_violation_px'] = sign_violation
    report['accepted_numerically'] = bool(solver_success and np.isfinite(x).all()
        and abs(candidate_area-wanted) <= max(1e-8, abs(wanted)*1e-10) and endpoint_max <= 1e-12
        and sign_violation <= 1e-10 and all(r['continuous_violating_subintervals'] == 0 for r in checked))
    if not report['accepted_numerically']:
        report['rejection_reason'] = 'Solver/area/endpoint/sign/continuous check did not all pass; no executable profile is exported.'
    output['candidate_signed_area_delta_px2'] = candidate_area
    if report['accepted_numerically']:
        output['profiles'] = profile_rows
    return output


def _validated_profiles(problem, profiles):
    if len(profiles) != len(problem.arcs) or len({p['arc_id'] for p in profiles}) != len(profiles):
        raise ValueError('one distinct profile per source arc required')
    by_id = {p['arc_id']: p for p in profiles}
    rows = []
    for arc in problem.arcs:
        if arc['id'] not in by_id:
            raise ValueError('profile arc id does not match reviewed geometry')
        profile = by_id[arc['id']]
        s, values = np.asarray(profile['arclength_px'], float), np.asarray(profile['across_displacement_px'], float)
        if (s.ndim != 1 or values.shape != s.shape or len(s) < 2 or not np.isfinite([s, values]).all()
                or s[0] != 0. or abs(s[-1]-arc['length']) > 1e-9 or np.any(np.diff(s) <= 0.)):
            raise ValueError('finite increasing profile must span the original source arc')
        rows.append((arc, s, values))
    return rows


def evaluate_profile_constraints(geometry, profiles, *, window_radius_px=16.):
    """Continuously inspect arbitrary piecewise-linear profile constraints.

    Adds source and profile breakpoints to the partition, then checks each
    positive-denominator quadratic at endpoints and its interior minimum.
    Unknown intervals remain unknown. This does not inspect any image.
    """
    if not np.isfinite(window_radius_px) or window_radius_px <= 0.:
        raise ValueError('positive finite window required')
    problem = _Problem(geometry, 2., window_radius_px)
    result = []
    for arc, nodes, values in _validated_profiles(problem, profiles):
        arc['nodes'] = nodes
        regions = problem.convex_regions(arc)
        centers, numerator, slope = problem.continuous_minima(arc, values)
        known = _merge_intervals(regions)
        result.append({'arc_id': arc['id'], 'continuous_convex_subintervals': len(regions),
            'continuous_min_numerator': float(numerator.min()) if len(numerator) else None,
            'continuous_violating_subintervals': int(np.sum(numerator < -1e-8)),
            'violating_center_arclength_px': centers[numerator < -1e-8].tolist(),
            'continuous_min_signed_slope_at_numerator_minima': float(slope.min()) if len(slope) else None,
            'known_source_convex_monotone_center_intervals_px': known,
            'unknown_center_intervals_px': _unknown_intervals(known, arc['length']),
            'source_nonmonotone_height_intervals_px': arc['nonmonotone']})
    return result


def evaluate_jaw_curve_profile(geometry, result, arc_id, arclength_px):
    """Evaluate a hash-bound profile on unchanged source arclength positions.

    The baked-in amount is not multiplied again. This deliberately returns
    source/target points rather than a deformation field or photometric blend.
    """
    if result.get('source_geometry_sha256') != _geometry_hash(geometry):
        raise ValueError('profile geometry hash mismatch')
    problem = _Problem(geometry, 2., 16.)
    rows = _validated_profiles(problem, result['profiles'])
    for arc, nodes, values in rows:
        if arc['id'] != arc_id:
            continue
        s = np.asarray(arclength_px, float)
        if s.ndim != 1 or not np.isfinite(s).all() or np.any((s < 0.) | (s > arc['length'])):
            raise ValueError('finite query arclengths inside the original source arc required')
        source = problem.source(arc, s); displacement = np.interp(s, nodes, values)
        return {'source': source, 'target': source+displacement[:, None]*problem.across,
                'across_displacement_px': displacement}
    raise ValueError('unknown source arc id')
