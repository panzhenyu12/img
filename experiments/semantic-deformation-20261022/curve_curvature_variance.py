"""Unresolved continuous-curvature variance inside the original D2 windows.

The original sampled D2 objective must remain in place. This supplement sees
only the constant u'' inside each original edge/grid piece; it does not charge
derivative jumps at original source vertices and is not whole-curve H2.
For windows containing a source vertex, the original D2 mean/jump cross term
can still cancel; this supplement only penalizes the continuous variance.
"""
from fractions import Fraction
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack


def _fraction(value):
    return Fraction.from_float(float(value))


def _curvature_pieces(curve):
    """Analytic 2 tx ty u_xy rows and exact source/grid piece intervals."""
    points = np.asarray(curve.points, float); knots = np.asarray(curve.knots, float)
    grid = curve.problem['grid']; mapping = np.asarray(curve.problem['mapping'])
    origin = np.asarray(grid['origin'], float); spacing = np.asarray(grid['spacing'], float)
    counts = [grid['width'], grid['height']]; size = curve.size
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all() or knots.shape != (len(points),)
            or not np.isfinite(knots).all() or knots[0] != 0.
            or np.any(np.diff(knots) <= 0) or curve.length != knots[-1]):
        raise ValueError('finite original points and increasing source arclength from zero required')
    if (origin.shape != (2,) or spacing.shape != (2,) or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or any(not isinstance(n, (int, np.integer)) or n < 2 for n in counts)):
        raise ValueError('finite grid with positive physical spacing required')
    if (not isinstance(size, (int, np.integer)) or size < 0
            or mapping.shape != (counts[0]*counts[1],)
            or not np.issubdtype(mapping.dtype, np.integer)
            or np.any(mapping < -1) or np.any(mapping >= size)):
        raise ValueError('existing fixed-zero or free scalar-node mapping required')
    exact_points = [list(map(_fraction, point)) for point in points]
    exact_origin = list(map(_fraction, origin)); exact_spacing = list(map(_fraction, spacing))
    for point in exact_points:
        if any(p < o or p > o+(n-1)*h for p, o, n, h in zip(point, exact_origin, counts, exact_spacing)):
            raise ValueError('source curve outside the unchanged grid')
    rows, columns, data = [], [], []
    pieces = []
    for edge, (a, b) in enumerate(zip(exact_points[:-1], exact_points[1:])):
        delta = [last-first for first, last in zip(a, b)]
        if delta == [0, 0]:
            raise ValueError('positive original geometric edge length required')
        start = _fraction(knots[edge]); length = _fraction(knots[edge+1])-start
        tangent = (points[edge+1]-points[edge])/(knots[edge+1]-knots[edge])
        coefficient = 2.*tangent[0]*tangent[1]/(spacing[0]*spacing[1])
        cuts = {Fraction(0), Fraction(1)}
        for axis, d in enumerate(delta):
            if d == 0:
                continue
            for index in range(1, counts[axis]-1):
                parameter = (exact_origin[axis]+index*exact_spacing[axis]-a[axis])/d
                if 0 < parameter < 1:
                    cuts.add(parameter)
        cuts = sorted(cuts)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mid = (lo+hi)/2
            uv = [(p+mid*d-o)/h for p, d, o, h in zip(a, delta, exact_origin, exact_spacing)]
            # A constant axis on the high boundary uses the inside cell. No
            # evaluation point is offset or snapped to choose a grid side.
            x, y = [min(q.numerator//q.denominator, count-2) for q, count in zip(uv, counts)]
            nodes = np.array([y*counts[0]+x, y*counts[0]+x+1,
                              (y+1)*counts[0]+x, (y+1)*counts[0]+x+1])
            row = len(pieces)
            for column, sign in zip(mapping[nodes], [1., -1., -1., 1.]):
                if column >= 0 and coefficient != 0.:
                    rows.append(row); columns.append(int(column)); data.append(sign*coefficient)
            pieces.append({'lo': start+lo*length, 'hi': start+hi*length,
                'source_edge_index': edge, 'cell_xy': [x, y], 'tangent_xy': tangent.tolist()})
    operator = coo_matrix((data, (rows, columns)), shape=(len(pieces), size)).tocsr()
    operator.sum_duplicates(); operator.eliminate_zeros()
    return operator, pieces


def _tent_mass(lo, hi, center, half_width):
    """Exact integral of the triangular kernel over a source piece."""
    lo = max(lo, center-half_width); hi = min(hi, center+half_width)
    if hi <= lo:
        return Fraction(0)
    cuts = [lo, center, hi] if lo < center < hi else [lo, hi]
    total = Fraction(0)
    for a, b in zip(cuts[:-1], cuts[1:]):
        # Each side of the tent is linear, so the trapezoid is exact.
        total += (b-a)*(2*half_width-abs(a-center)-abs(b-center))/(2*half_width**2)
    return total


def curvature_variance(curve, length_scale_px, node_spacing_px):
    """Return weighted residual ``R``, ``hessian=R.T@R`` and diagnostics.

    Keep the existing uniform nodes and h=L/(n-1). In each interior-node
    window Ki=(h-|s-si|)+/h², wj=integral_piece Ki and m=sum(wj). Residual rows
    are ell² sqrt(h*wj) [Cj - sum(wk*Ck)/m]. The actual kernel weights remain
    unchanged: only the weighted mean divides by the measured mass. This PSD
    construction never subtracts two Hessians or adds a target/linear term.
    """
    if (not np.isfinite([length_scale_px, node_spacing_px]).all()
            or length_scale_px < 0 or node_spacing_px <= 0):
        raise ValueError('finite nonnegative length scale and positive node spacing required')
    curvature, pieces = _curvature_pieces(curve)
    count = max(2, int(np.ceil(curve.length/node_spacing_px)))+1
    nodes = np.linspace(0., curve.length, count); h = curve.length/(count-1)
    exact_h = _fraction(h); ell = float(length_scale_px)
    residuals = []; windows = []; offset = 0
    for node_index, center_value in enumerate(nodes[1:-1], start=1):
        center = _fraction(center_value)
        weighted = [(index, _tent_mass(piece['lo'], piece['hi'], center, exact_h))
                    for index, piece in enumerate(pieces)]
        weighted = [(index, weight) for index, weight in weighted if weight > 0]
        indices = [index for index, _ in weighted]
        weights = np.array([float(weight) for _, weight in weighted])
        mass = math.fsum(weights)
        if not np.isfinite(weights).all() or np.any(weights <= 0.) or not np.isfinite(mass) or mass <= 0.:
            raise ValueError('positive finite physical kernel-piece masses required')
        local = curvature[indices]
        mean = (csr_matrix(weights[None, :])@local)/mass
        centered = local-vstack([mean]*len(indices), format='csr')
        factor = ell**2*np.sqrt(h*weights)
        residual = (diags(factor)@centered).tocsr()
        residual.sum_duplicates(); residual.eliminate_zeros(); residuals.append(residual)
        windows.append({'node_index': node_index, 'center_source_arclength_px': float(center_value),
            'source_support_interval_px': [float(max(Fraction(0), center-exact_h)),
                                           float(min(_fraction(curve.length), center+exact_h))],
            'piece_indices': indices, 'kernel_piece_masses': weights.tolist(),
            'actual_kernel_mass': mass, 'kernel_mass_error_from_one': mass-1.,
            'residual_row_start': offset, 'residual_row_count': len(indices)})
        offset += len(indices)
    operator = vstack(residuals, format='csr')
    hessian = (operator.T@operator).tocsc()
    report = {'schema': 'original-d2-window-continuous-curvature-variance-v1',
        'node_count': count, 'window_count': len(windows), 'residual_rows': operator.shape[0],
        'scalar_variables': curve.size, 'source_piece_count': len(pieces),
        'source_length_px': curve.length, 'requested_node_spacing_px': float(node_spacing_px),
        'actual_node_spacing_px': h, 'length_scale_px': ell,
        'formula': 'ell^4 * h * sum_windows sum_pieces w_j * (C_j - sum(w_k*C_k)/actual_mass)^2',
        'curvature': 'C_j = 2*t_x*t_y*u_xy; constant within each unchanged original edge/grid piece',
        'kernel': 'K_i(s)=(h-abs(s-s_i))_+/h^2 on the original source domain; no boundary extension',
        'kernel_mass_policy': 'weights retain their actual integrals; only the weighted mean divides by sum(weights)',
        'maximum_kernel_mass_error': max(abs(w['kernel_mass_error_from_one']) for w in windows),
        'kernel_mass_range': [min(w['actual_kernel_mass'] for w in windows), max(w['actual_kernel_mass'] for w in windows)],
        'psd_construction': 'explicit weighted residual rows R; hessian=R.T@R, no Hessian subtraction',
        'operator_includes_length_scale': True, 'energy_from_operator': '||R z||^2; no additional half',
        'units': 'R coefficients sqrt(source px); z displacement px; energy px^3',
        'replaces_original_d2': False, 'original_vertex_derivative_jumps_excluded': True,
        'source_vertex_mean_cancellation_controlled': False,
        'source_vertex_limitation': 'When D2=mu+J, retaining D2^2 and adding Var(C) retains 2*mu*J; constant C with J=-mu can still have zero total bending. This supplement is not a complete vertex or whole-curve curvature bound.',
        'linear_term_added': False, 'additional_weight': 1.,
        'scope': 'supplement for unresolved continuous curvature only; retain all original D2 terms and source-vertex contributions; not whole-curve H2',
        'pieces': [{'source_edge_index': p['source_edge_index'], 'cell_xy': p['cell_xy'],
                    'source_arclength_interval_px': [float(p['lo']), float(p['hi'])],
                    'tangent_xy': p['tangent_xy']} for p in pieces],
        'windows': windows, 'image_approval': False}
    return {'operator': operator, 'hessian': hessian, 'report': report}
