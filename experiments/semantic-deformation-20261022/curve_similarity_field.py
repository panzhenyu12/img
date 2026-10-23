"""Observed-source-curve relative-similarity prior; no anatomical inference.

For source chords e1=b-a and e2=c-b, the complex displacement stencil is
    C d = h*((d_c-d_b)/e2-(d_b-d_a)/e1),
with harmonic physical chord length h. The soft energy is density*int|Cd|²ds
over centers where both source-arclength half windows are available. C d has
pixel units, ds has pixel units and density has inverse-pixel units. This is
finite-chord relative similarity, not a continuous curvature/angle guarantee.
Uniform collapse and some oscillatory motions remain in its nullspace; global
field geometry and visual quality therefore need independent verification.
"""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csc_matrix, diags, eye, vstack
from scipy.sparse.linalg import splu


def curve_windows(points, *, half_window_px=12., center_spacing_px=2.):
    """Source-arclength windows and physical complex stencil coefficients."""
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all()
            or not np.isfinite([half_window_px, center_spacing_px]).all()
            or half_window_px <= 0 or center_spacing_px <= 0):
        raise ValueError('finite source polyline and positive physical window/spacing required')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0.):
        raise ValueError('source polyline edges must have positive length')
    cumulative = np.r_[0., lengths.cumsum()]
    span = cumulative[-1]-2*half_window_px
    if span <= 0.:
        raise ValueError('source curve must exceed two half windows in arclength')
    ratio = span/center_spacing_px
    nearest = np.rint(ratio)
    # Rotating/subdividing the same source can change a summed length by a
    # few ulps. Do not let that change a nominally integral sample count.
    if abs(ratio-nearest) <= 64*np.finfo(float).eps*max(1., abs(ratio)):
        ratio = nearest
    intervals = max(1, int(np.ceil(ratio)))
    centers = np.linspace(half_window_px, cumulative[-1]-half_window_px, intervals+1)
    locations = centers[:, None]+[-half_window_px, 0., half_window_px]
    samples = np.stack([np.interp(locations, cumulative, points[:, i]) for i in range(2)], axis=-1)
    complex_samples = samples[..., 0]+1j*samples[..., 1]
    first = complex_samples[:, 1]-complex_samples[:, 0]
    second = complex_samples[:, 2]-complex_samples[:, 1]
    # A long source-arclength window can return close to its starting point.
    # Never divide by that unobservable/degenerate physical chord.
    minimum = 64*np.finfo(float).eps*half_window_px
    if np.any(abs(first) <= minimum) or np.any(abs(second) <= minimum):
        raise ValueError('source curve contains a degenerate window chord')
    h = 2*abs(first)*abs(second)/(abs(first)+abs(second))
    coefficients = h[:, None]*np.column_stack([1/first, -1/first-1/second, 1/second])
    weights = np.full(len(centers), span/intervals)
    weights[[0, -1]] *= .5
    return {'points': samples, 'coefficients': coefficients, 'weights': weights,
            'centers_s': centers, 'harmonic_chord_px': h, 'source_length_px': float(cumulative[-1]),
            'actual_center_spacing_px': float(span/intervals)}


def assemble_curve_operator(problem, interface, *, half_window_px=12., center_spacing_px=2.):
    """Assemble complex C on the original source-grid fixed-node mapping."""
    from portable_field import point_operators

    if interface.get('schema') != 'reviewed-visible-source-interface-v1':
        raise ValueError('reviewed visible source interface required')
    segments = interface.get('segments')
    if not isinstance(segments, list) or not segments:
        raise ValueError('at least one complete reviewed source polyline required')
    grid = problem['grid']
    n = problem['hessian'].shape[0]
    ih, iw = problem['image_shape']
    operators, weights, harmonic_lengths, reports = [], [], [], []
    names = set()
    offset = 0

    def sample_operator(points):
        # Match portable_field.evaluate_motion exactly: identity outside the
        # ROI, original bilinear basis inside, and zero eliminated fixed nodes.
        uv = (points-grid['origin'])/grid['spacing']
        inside = np.all((uv >= 0)&(uv <= [grid['width']-1, grid['height']-1]), axis=1)
        ids = np.flatnonzero(inside)
        if not len(ids):
            return csc_matrix((len(points), n))
        sampled, _, _ = point_operators(points[inside], grid, problem['mapping'], n)
        sampled = sampled.tocoo()
        return coo_matrix((sampled.data, (ids[sampled.row], sampled.col)), shape=(len(points), n)).tocsr()

    for segment in segments:
        name = segment.get('id')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('distinct reviewed curve identities required')
        names.add(name)
        points = np.asarray(segment.get('source_points'), dtype=float)
        windows = curve_windows(points, half_window_px=half_window_px, center_spacing_px=center_spacing_px)
        if np.any(points < 0.) or np.any(points > [iw-1, ih-1]):
            raise ValueError('reviewed source curve lies outside the source photograph')
        operator = sum(diags(windows['coefficients'][:, j]) @ sample_operator(windows['points'][:, j])
                       for j in range(3)).tocsc()
        operator.sum_duplicates(); operator.eliminate_zeros()
        count = len(windows['weights'])
        operators.append(operator); weights.append(windows['weights'])
        harmonic_lengths.append(windows['harmonic_chord_px'])
        reports.append({'id': name, 'source_point_count': len(points),
            'source_length_px': windows['source_length_px'], 'window_count': count,
            'row_start': offset, 'row_end': offset+count,
            'center_arclength_px': windows['centers_s'].tolist(),
            'actual_center_spacing_px': windows['actual_center_spacing_px'],
            'center_measure_px': float(windows['weights'].sum()),
            'zero_operator_rows': int(np.count_nonzero(np.asarray(operator.getnnz(axis=1)).ravel() == 0))})
        offset += count
    return {'operator': vstack(operators, format='csc'), 'weights': np.concatenate(weights),
            'harmonic_chord_px': np.concatenate(harmonic_lengths), 'curves': reports,
            'half_window_px': float(half_window_px), 'center_spacing_px': float(center_spacing_px)}


def assemble_endpoint_jet_operator(problem, interface):
    """Original-grid endpoint values and source-interior unit-tangent derivatives.

    The field remains C0 bilinear. At a grid line the incident source segment
    selects the derivative cell, evaluated at the original endpoint, never at
    an epsilon-displaced surrogate. The two-node fixed collar makes value and
    derivative zero at the identity exterior. No normal derivative is imposed.
    """
    if interface.get('schema') != 'reviewed-visible-source-interface-v1':
        raise ValueError('reviewed visible source interface required')
    segments = interface.get('segments')
    if not isinstance(segments, list) or not segments:
        raise ValueError('at least one complete reviewed source polyline required')
    grid, mapping = problem['grid'], problem['mapping']
    origin, spacing = np.asarray(grid['origin']), np.asarray(grid['spacing'])
    maximum = np.array([grid['width']-1, grid['height']-1])
    n = problem['hessian'].shape[0]
    rows, cols, values, reports = [], [], [], []
    names = set()
    for segment in segments:
        name = segment.get('id')
        points = np.asarray(segment.get('source_points'), dtype=float)
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('distinct reviewed curve identities required')
        names.add(name)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all() or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0)):
            raise ValueError('finite nondegenerate source polyline required')
        ih, iw = problem['image_shape']
        if np.any(points < 0.) or np.any(points > [iw-1, ih-1]):
            raise ValueError('reviewed source curve lies outside the source photograph')
        for endpoint, p, neighbor in [('start', points[0], points[1]), ('end', points[-1], points[-2])]:
            tangent = neighbor-p
            tangent /= np.linalg.norm(tangent)
            uv = (p-origin)/spacing
            # Only exact equality in physical source coordinates identifies a
            # grid line; a point one ulp away must remain in its actual cell.
            nearest = np.rint(uv)
            on_line = p == origin+nearest*spacing
            uv = np.where(on_line, nearest, uv)
            inside = bool(np.all((uv >= 0)&(uv <= maximum)))
            cell = np.floor(uv).astype(np.int64)
            cell -= on_line & (tangent < 0.)
            # Along the top/right boundary a zero component follows that
            # shared edge, whose tangential derivative is cell-independent.
            cell = np.where(on_line & (tangent == 0.), np.minimum(cell, maximum-1), cell)
            derivative_inside = inside and bool(np.all((cell >= 0)&(cell < maximum)))
            for kind in ['value', 'inward_tangent']:
                row = len(reports)
                reports.append({'curve_id': name, 'endpoint': endpoint, 'kind': kind,
                    'source_xy': p.tolist(), 'inward_unit_tangent': tangent.tolist(),
                    'incident_cell_xy': cell.tolist() if derivative_inside else None,
                    'units': 'px' if kind == 'value' else 'px/px'})
                if not inside or (kind != 'value' and not derivative_inside):
                    continue
                # Values agree across cell boundaries; choose a valid ROI
                # cell even if the incident ray immediately exits the ROI.
                active_cell = np.clip(cell, 0, maximum-1)
                x, y = uv-active_cell
                cx, cy = active_cell
                nodes = np.array([cy*grid['width']+cx, cy*grid['width']+cx+1,
                                  (cy+1)*grid['width']+cx, (cy+1)*grid['width']+cx+1])
                if kind == 'value':
                    basis = np.array([(1-x)*(1-y), x*(1-y), (1-x)*y, x*y])
                else:
                    dx = np.array([-(1-y), 1-y, -y, y])/spacing[0]
                    dy = np.array([-(1-x), -x, 1-x, x])/spacing[1]
                    basis = tangent[0]*dx+tangent[1]*dy
                free = mapping[nodes] >= 0
                rows.extend([row]*int(free.sum())); cols.extend(mapping[nodes][free]); values.extend(basis[free])
    operator = coo_matrix((values, (rows, cols)), shape=(len(reports), n)).tocsc()
    operator.sum_duplicates(); operator.eliminate_zeros(); operator.sort_indices()
    return {'operator': operator, 'rows': reports}


def _solve_jet_equalities(factor, free_solution, operator, reference):
    """Small Schur solve after numerical row-rank reduction; audit every row."""
    e = csc_matrix(operator, dtype=float)
    reference = np.asarray(reference, dtype=complex)
    if (reference.shape != (e.shape[0],) or not np.isfinite(reference).all()
            or not np.isfinite(e.data).all() or not np.isfinite(free_solution).all()):
        raise ValueError('finite matching endpoint equalities required')
    row_norm = np.sqrt(np.asarray(e.multiply(e).sum(axis=1)).ravel())
    scale = np.ones_like(row_norm)
    nonzero = row_norm > 0.
    scale[nonzero] = 1./row_norm[nonzero]
    normalized = diags(scale) @ e
    target = scale*reference
    # Each jet has at most four grid-node entries. Restrict SVD to their union
    # rather than densifying all source-grid columns; rank reduction retains
    # an explicit transform back to every original physical equality.
    active = np.unique(normalized.tocoo().col)
    u, singular, _ = np.linalg.svd(normalized[:, active].toarray(), full_matrices=False)
    tolerance = 64*np.finfo(float).eps*max(1, *normalized[:, active].shape)*singular.max(initial=0.)
    rank = int(np.count_nonzero(singular > tolerance))
    retained = u[:, :rank]
    consistency = target-retained @ (retained.conjugate().T @ target)
    consistency_max = float(abs(consistency).max(initial=0.))
    if consistency_max > 1e-10*max(1., float(abs(target).max(initial=0.))):
        raise ValueError('inconsistent endpoint equalities after rank reduction')
    multipliers = np.zeros(e.shape[0], complex)
    solution = free_solution.copy()
    if rank:
        transform = (retained/singular[:rank]).conjugate().T
        basis = (csc_matrix(transform) @ normalized).tocsc()
        basis_target = transform @ target
        response = factor.solve(basis.conjugate().T.toarray().astype(complex))
        schur = basis @ response
        asymmetry = float(abs(schur-schur.conjugate().T).max(initial=0.))
        if not np.isfinite(schur).all() or asymmetry > 1e-10*max(1., float(abs(schur).max(initial=0.))):
            raise ValueError('endpoint Schur complement is not finite Hermitian')
        schur = .5*(schur+schur.conjugate().T)
        lagrange = cho_solve(cho_factor(schur, lower=True), basis @ free_solution-basis_target)
        solution -= response @ lagrange
        multipliers = scale*(transform.conjugate().T @ lagrange)
    residual = e @ solution-reference
    if (not np.isfinite(solution).all() or not np.isfinite(multipliers).all()
            or np.any(abs(residual) > 1e-9*np.maximum(1., abs(reference)))):
        raise ValueError('endpoint physical equality residual failed')
    return solution, multipliers, {'rank': rank, 'zero_rows': int((~nonzero).sum()),
        'rank_singular_tolerance': float(tolerance), 'normalized_target_nullspace_max_residual': consistency_max,
        'physical_residual': residual, 'reference': reference, 'original_complex_rows': e.shape[0]}


def solve_problem(problem, interface, *, half_window_px=12., center_spacing_px=2., density=.05,
                  half_windows_px=None, reference_half_window_px=12., endpoint_reference='free'):
    """Solve the original quadratic plus the observed-curve soft shape prior.

    None preserves the original single-window objective exactly. An explicit
    sequence uses the sum of independent scale energies, each with C multiplied
    by (reference_half_window_px/r)**2 and the original source-arclength W.
    Density applies to every term, without averaging. On a straight source,
    C_r applied to quadratic displacement is r²*d'', making this normalization
    exactly reference_radius²*d''. Curved sources only have that small-window
    interpretation; scale center domains also differ and are reported.
    """
    from portable_field import evaluate_motion, source_geometry

    started = time.perf_counter()
    if endpoint_reference not in ('free', 'membrane-jet'):
        raise ValueError("endpoint_reference must be 'free' or 'membrane-jet'")
    if not np.isfinite(density) or density < 0.:
        raise ValueError('finite nonnegative source-arclength density required')
    if not np.isfinite(reference_half_window_px) or reference_half_window_px <= 0.:
        raise ValueError('finite positive reference half window required')
    if half_windows_px is None:
        radii = [half_window_px]
    else:
        if (not isinstance(half_windows_px, (tuple, list)) or not half_windows_px
                or not np.isfinite(half_windows_px).all() or np.any(np.asarray(half_windows_px) <= 0.)
                or len(set(half_windows_px)) != len(half_windows_px)):
            raise ValueError('explicit half windows must be distinct finite positive radii')
        radii = list(half_windows_px)
    original_h = csc_matrix(problem['hessian'])
    n = original_h.shape[0]
    reduction = csc_matrix(problem['reduction'])
    if reduction.shape != (n, n) or (reduction != eye(n, format='csc')).nnz:
        raise ValueError('curve similarity currently requires an identity reduction (stride1)')
    if problem['line_equalities'].shape != (0, 2*n):
        raise ValueError('curve similarity currently requires no background line equalities')
    rhs_components = np.asarray(problem['rhs'], dtype=float)
    if (n < 1 or original_h.shape != (n, n) or rhs_components.shape != (n, 2)
            or not np.isfinite(original_h.data).all() or not np.isfinite(rhs_components).all()):
        raise ValueError('finite matching portable objective required')
    scales = []
    for radius in radii:
        assembled = assemble_curve_operator(problem, interface, half_window_px=radius,
                                             center_spacing_px=center_spacing_px)
        normalization = 1. if half_windows_px is None else (reference_half_window_px/radius)**2
        scales.append({'radius': float(radius), 'normalization': float(normalization), 'assembled': assembled})
    # Avoid even a multiply-by-one or one-element sparse stack in the legacy
    # path, preserving the original matrix and numerical solution bitwise.
    operators = [scale['assembled']['operator'] if scale['normalization'] == 1. else
                 scale['normalization']*scale['assembled']['operator'] for scale in scales]
    c = operators[0] if len(operators) == 1 else vstack(operators, format='csc')
    weights = scales[0]['assembled']['weights'] if len(scales) == 1 else np.concatenate(
        [scale['assembled']['weights'] for scale in scales])
    augmented = (original_h+density*(c.conjugate().T @ diags(weights) @ c)).tocsc()
    augmented.sum_duplicates(); augmented.sort_indices()
    skew = augmented-augmented.conjugate().T
    if skew.nnz and abs(skew.data).max() > 1e-12*max(1., abs(augmented.data).max()):
        raise ValueError('source plus curve objective must be Hermitian')
    rhs = rhs_components[:, 0]+1j*rhs_components[:, 1]
    factor_started = time.perf_counter()
    factor = splu(augmented.astype(complex))
    factor_seconds = time.perf_counter()-factor_started
    z = factor.solve(rhs)
    if not np.isfinite(z).all():
        raise ValueError('nonfinite complex source-field solution')
    multipliers = np.empty(0)
    endpoint_report = None
    factorization_count = 1
    constraint_force = np.zeros_like(z)
    if endpoint_reference == 'membrane-jet':
        baseline_started = time.perf_counter()
        baseline_factor = splu(original_h.astype(complex))
        factor_seconds += time.perf_counter()-baseline_started
        baseline = baseline_factor.solve(rhs)
        factorization_count += 1
        if not np.isfinite(baseline).all():
            raise ValueError('nonfinite original membrane baseline')
        baseline_residual = original_h @ baseline-rhs
        original_norm = float(np.asarray(abs(original_h.copy()).sum(axis=1)).max())
        baseline_denominator = np.array([original_norm*abs(baseline.real).max()+abs(rhs.real).max(),
                                         original_norm*abs(baseline.imag).max()+abs(rhs.imag).max()])
        baseline_backward = np.array([abs(baseline_residual.real).max(), abs(baseline_residual.imag).max()]) / np.maximum(
            baseline_denominator, np.finfo(float).tiny)
        if not np.isfinite(baseline_backward).all() or np.any(baseline_backward > 1e-10):
            raise ValueError('original membrane baseline stationarity failed')
        jet = assemble_endpoint_jet_operator(problem, interface)
        e = jet['operator']
        z, multipliers, equality = _solve_jet_equalities(factor, z, e, e @ baseline)
        constraint_force = e.conjugate().T @ multipliers
        physical = equality.pop('physical_residual')
        reference = equality.pop('reference')
        value_rows = np.array([row['kind'] == 'value' for row in jet['rows']])
        endpoint_report = {**equality, 'reference': 'original unchanged membrane quadratic solution',
            'baseline_backward_error_by_component': baseline_backward.tolist(),
            'derivative': 'source-interior one-sided unit tangent at the original endpoint on the C0 bilinear grid',
            'normal_derivative_constrained': False, 'global_C1_continuity': False,
            'value_max_residual_px': float(abs(physical[value_rows]).max(initial=0.)),
            'tangent_max_residual': float(abs(physical[~value_rows]).max(initial=0.)),
            'physical_residual_real_imag': np.column_stack([physical.real, physical.imag]).tolist(),
            'reference_real_imag': np.column_stack([reference.real, reference.imag]).tolist(),
            'rows': jet['rows'],
            'baseline_augmented_objective_without_constant': float(np.vdot(baseline, augmented @ baseline).real-2*np.vdot(baseline, rhs).real)}
    linear_residual = augmented @ z-rhs+constraint_force
    # Equivalent real blocks are [Re H,-Im H; Im H,Re H]. Certify both
    # components with denominators that include their actual cross coupling.
    # Sparse real/imag views may share structural arrays; reductions can
    # canonicalize those arrays. Keep diagnostics independent of the matrix
    # used by the solve and final energy/stationarity evaluation.
    real_norm = float(np.asarray(abs(augmented.real.copy()).sum(axis=1)).max())
    imag_norm = float(np.asarray(abs(augmented.imag.copy()).sum(axis=1)).max())
    xmax, ymax = abs(z.real).max(), abs(z.imag).max()
    denominator = np.array([real_norm*xmax+imag_norm*ymax+abs(rhs.real).max(),
                            imag_norm*xmax+real_norm*ymax+abs(rhs.imag).max()])
    if endpoint_report is not None:
        # KKT stationarity is Hz-b+E^H lambda=0, not the unconstrained Hz-b.
        denominator += [abs(constraint_force.real).max(), abs(constraint_force.imag).max()]
    backward = np.array([abs(linear_residual.real).max(), abs(linear_residual.imag).max()]) / np.maximum(
        denominator, np.finfo(float).tiny)
    if not np.isfinite(backward).all() or np.any(backward > 1e-10):
        raise ValueError('complex source-field stationarity failed')
    if endpoint_report is not None:
        endpoint_report['kkt_backward_error'] = float(backward.max())
        endpoint_report['kkt_stationarity_max_residual'] = float(abs(linear_residual).max())
    solution = np.column_stack([z.real, z.imag])
    grid = problem['grid']
    displacement = (problem['prolongation'] @ solution).reshape(grid['height'], grid['width'], 2)
    actual = problem['source_points']+problem['target_operator'] @ solution
    target_error = actual-problem['target_points']
    errors = np.linalg.norm(target_error, axis=1)
    shape_residual = c @ z
    shape_energy = float(density*np.sum(weights*abs(shape_residual)**2))
    diagnostics = problem['diagnostics']
    area = diagnostics['cell_area_px2']
    target_energy = float(np.sum(problem['target_weights'][:, None]*target_error**2))
    bending_energy = float(diagnostics['bending_weight']*area*np.sum((problem['laplacian'] @ solution)**2))
    membrane_energy = float(diagnostics['membrane_weight']*area*(
        np.sum((problem['Dx'] @ solution)**2)+np.sum((problem['Dy'] @ solution)**2)))
    geometry = source_geometry(displacement/grid['spacing'])
    fixed_motion = evaluate_motion(problem, solution, problem['fixed_pixels'])
    curve_reports, scale_reports = [], []
    row_offset = 0
    for scale in scales:
        assembled = scale['assembled']
        raw = assembled['operator'] @ z
        w = assembled['weights']
        normalized = scale['normalization']*raw
        local_curves = []
        for curve in assembled['curves']:
            select = slice(curve['row_start'], curve['row_end'])
            local, local_w = raw[select], w[select]
            detail = {**curve, 'half_window_px': scale['radius'],
                'shape_residual_max_px': float(abs(local).max(initial=0.)),
                'shape_residual_weighted_rms_px': float(np.sqrt(np.sum(local_w*abs(local)**2)/local_w.sum()))}
            local_curves.append(detail)
            curve_reports.append({**detail, 'row_start': row_offset+curve['row_start'],
                                  'row_end': row_offset+curve['row_end']})
        scale_reports.append({'half_window_px': scale['radius'],
            'residual_normalization': scale['normalization'],
            'gram_normalization': scale['normalization']**2,
            'density_per_source_px': float(density), 'density_averaged_across_scales': False,
            'center_measure_px': float(w.sum()),
            'raw_shape_residual_max_px': float(abs(raw).max(initial=0.)),
            'raw_shape_residual_weighted_rms_px': float(np.sqrt(np.sum(w*abs(raw)**2)/w.sum())),
            'normalized_shape_residual_max_px': float(abs(normalized).max(initial=0.)),
            'weighted_shape_energy': float(density*np.sum(w*abs(normalized)**2)),
            'curves': local_curves})
        row_offset += len(raw)
    report = {'status': 'CurveSimilarityQuadraticSolved', 'converged': True,
        'solver': ('SciPy complex sparse SuperLU on original SPD quadratic plus Hermitian PSD curve Gram'
                   + ('; endpoint SVD rank reduction and small Hermitian Schur solve' if endpoint_report is not None else '')),
        'model': 'observed-source finite-chord relative similarity; no anatomical inference',
        'density_per_source_px': float(density), 'half_window_px': float(half_window_px),
        'half_windows_px': [float(r) for r in radii],
        'scale_normalization_enabled': half_windows_px is not None,
        'reference_half_window_px': float(reference_half_window_px),
        'scale_aggregation': 'sum; density applies independently to every scale, never averaged',
        'shape_residual_component_meaning': 'real/imag source-chord relative-similarity components, not image-coordinate displacement',
        'top_level_residuals_are_reference_normalized': half_windows_px is not None,
        'scales': scale_reports,
        'center_spacing_px': float(center_spacing_px),
        'quadrature': 'source-arclength trapezoid over centers [r,L-r]; all complete original source curves retained',
        'target_energy': target_energy, 'bending_energy': bending_energy,
        'membrane_energy': membrane_energy, 'curve_shape_energy': shape_energy,
        'total_energy': target_energy+bending_energy+membrane_energy+shape_energy,
        'objective_without_constant': float(np.vdot(z, augmented @ z).real-2*np.vdot(z, rhs).real),
        'shape_residual_max_px': float(abs(shape_residual).max(initial=0.)),
        'shape_residual_weighted_rms_px': float(np.sqrt(np.sum(weights*abs(shape_residual)**2)/weights.sum())),
        'curves': curve_reports, 'linear_backward_error_by_component': backward.tolist(),
        'projected_backward_error': float(backward.max()), 'factorization_count': factorization_count,
        'endpoint_reference': endpoint_reference, 'endpoint_constraints': endpoint_report,
        'original_hessian_unmodified': True, 'original_rhs_unmodified': True,
        'original_target_bending_membrane_weights_unchanged': True,
        'fixed_node_mapping_unchanged': True, 'identity_exterior_unchanged': True,
        'geometry_constraints_enforced': False, 'chord_angles_hard_constrained': False,
        'global_optimality_scope': ('assembled fixed-node Hermitian SPD quadratic'
            + (' subject to reported endpoint linear equalities' if endpoint_report is not None else '')
            + '; no geometry inequalities'),
        'source_role_inference': False, 'source_identity_file_validation': 'caller responsibility',
        'nullspace_limitations': 'uniform collapse and finite-window oscillatory modes are not excluded by the curve term',
        'original_line_rows_checked': 0, 'full_line_max_residual': 0., 'full_line_unit_max_residual': 0.,
        'aesthetic_approval': False, 'factor_seconds': factor_seconds,
        'complete_seconds': time.perf_counter()-started}
    return {'solution': solution, 'coefficients': solution.copy(), 'multipliers': multipliers,
        'source_displacement': displacement, 'actual_targets': actual, 'target_error_px': errors,
        'target_max_error_px': float(errors.max(initial=0.)),
        'fixed_source_motion_max_px': float(np.linalg.norm(fixed_motion, axis=1).max(initial=0.)),
        'shape_residual_xy_px': np.column_stack([shape_residual.real, shape_residual.imag]),
        'geometry': geometry, 'report': report,
        'inverse_evaluation_allowed': bool(geometry['strong_monotonicity_lower_bound'] > 0 and geometry['boundary_zero']),
        'aesthetic_approval': False}
