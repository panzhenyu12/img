"""Arc-length quadrature of the unchanged source jaw polyline.

These are evaluation/collocation locations on existing source edges, not new
detector observations or a fitted replacement contour. The continuous target
is exactly jaw_arc_v1: -amount/100 * radius_fraction * source_radius *
64[t(1-t)]**3 along the reviewed across axis.

The new target objective is 0.5/px times the squared-residual line integral
over each selected source arc, plus weight 4 at each original outside row.
The density is explicit (weight 4 per 8px of source arc), not calibrated from
old detector-point counts. Changing quadrature spacing changes approximation
error, not the intended objective or total arc weight.
"""
import numpy as np


TARGET_DENSITY = .5
OUTSIDE_WEIGHT = 4.


def _prepare(geometry, amount, radius_fraction, spacing):
    if (not np.isfinite([amount, radius_fraction, spacing]).all()
            or abs(amount) > 100. or not 0. <= radius_fraction <= .45 or spacing <= 0.):
        raise ValueError('finite amount in [-100,100], radius fraction in [0,.45] and positive spacing required')
    p = np.asarray(geometry['contour'], float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('finite reviewed source contour required')
    edge = np.roll(p, -1, axis=0)-p
    length = np.linalg.norm(edge, axis=1)
    if np.any(length <= 0.):
        raise ValueError('source contour edges must have positive length')
    cumulative = np.r_[0., length.cumsum()]; perimeter = cumulative[-1]
    frame = geometry['frame']
    origin, across = np.asarray(frame['origin'], float), np.asarray(frame['across'], float)
    midline = float(frame['midline_u'])
    if (origin.shape != (2,) or across.shape != (2,) or not np.isfinite([*origin, *across, midline]).all()
            or not np.isclose(across@across, 1., atol=2e-6, rtol=0)):
        raise ValueError('finite reviewed frame with unit across axis required')

    def boundary(value):
        i, t = value['edge'], value['t']
        if (type(i) is not int or not 0 <= i < len(p) or not np.isfinite(t) or not 0. <= t <= 1.):
            raise ValueError('source arc boundary requires an existing edge and fraction in [0,1]')
        point = p[i].copy() if t == 0. else (p[(i+1)%len(p)].copy() if t == 1. else p[i]+t*edge[i])
        return float((cumulative[i]+t*length[i])%perimeter), point

    arcs = []; selected = np.zeros(len(p), bool); ids = set()
    for arc in geometry['arcs']:
        name, direction = arc['id'], arc['direction']
        if not isinstance(name, str) or not name or name in ids or direction not in [-1, 1]:
            raise ValueError('distinct arc ids and source traversal directions required')
        ids.add(name)
        start, first = boundary(arc['start']); end, last = boundary(arc['end'])
        span = float(((end-start)*direction)%perimeter)
        if span == 0.:
            raise ValueError('source arc endpoints must differ')
        distance = ((cumulative[:-1]-start)*direction)%perimeter
        interior = (distance > 0.) & (distance < span)
        if np.any(selected & interior):
            raise ValueError('reviewed jaw arcs have overlapping source rows')
        selected |= interior
        arcs.append({'id': name, 'direction': direction, 'start': start, 'length': span,
                     'first': first, 'last': last, 'start_boundary': arc['start'], 'end_boundary': arc['end']})
    if not arcs or np.flatnonzero(selected).tolist() != list(geometry['selected_rows']):
        raise ValueError('reviewed selected rows must match the open source arcs')
    return p, edge, length, cumulative, origin, across, midline, arcs, selected


def _evaluate_arc(prepared, arc, distance, amount, radius_fraction, *, endpoints):
    p, edge, length, cumulative, origin, across, midline, _, _ = prepared
    position = (arc['start']+arc['direction']*distance)%cumulative[-1]
    indices = np.minimum(np.searchsorted(cumulative, position, side='right')-1, len(p)-1)
    fractions = (position-cumulative[indices])/length[indices]
    source = p[indices]+fractions[:, None]*edge[indices]
    t = distance/arc['length']
    gain = 64.*(t*(1.-t))**3
    if endpoints:
        # The two sides share exactly the same original chin endpoint, even
        # when cumulative-length arithmetic would round its edge fraction.
        source[0], source[-1] = arc['first'], arc['last']
        indices[0], indices[-1] = arc['start_boundary']['edge'], arc['end_boundary']['edge']
        fractions[0], fractions[-1] = arc['start_boundary']['t'], arc['end_boundary']['t']
        gain[0] = gain[-1] = 0.
    target = source.copy()
    if amount != 0. and radius_fraction != 0.:
        radius = (source-origin)@across-midline
        target += (-amount/100.*radius_fraction*radius*gain)[:, None]*across
    return source, target, indices, fractions, gain


def _sample(geometry, amount, max_radius_fraction, spacing, phase, *, probes, collocation_spacing):
    prepared = _prepare(geometry, amount, max_radius_fraction, spacing)
    p, _, _, _, _, _, _, arcs, selected = prepared
    sources, targets, weights, controls, reports = [], [], [], [], []
    if not probes:
        outside = np.flatnonzero(~selected)
        sources.append(p[outside].copy()); targets.append(p[outside].copy())
        weights.append(np.full(len(outside), OUTSIDE_WEIGHT))
        controls.extend({'role': 'original_outside_arc', 'original_index': int(i), 'arc_id': None,
                         'arc_arclength_px': None, 'arc_t': None, 'point_gain': 0.,
                         'source_edge': int(i), 'source_edge_fraction': 0.} for i in outside)
    for arc in arcs:
        span = arc['length']
        if probes:
            training_intervals = int(np.ceil(span/collocation_spacing))
            subdivisions = max(1, int(np.ceil((span/training_intervals)/spacing)))
            intervals = training_intervals*subdivisions
            distance = (np.arange(intervals, dtype=float)+phase)*(span/intervals)
            weight = np.full(intervals, TARGET_DENSITY*span/intervals)
        else:
            intervals = int(np.ceil(span/spacing))
            distance = np.linspace(0., span, intervals+1)
            weight = np.full(intervals+1, TARGET_DENSITY*span/intervals)
            weight[[0, -1]] *= .5
        source, target, indices, fractions, gain = _evaluate_arc(
            prepared, arc, distance, amount, max_radius_fraction, endpoints=not probes)
        sources.append(source); targets.append(target); weights.append(weight)
        controls.extend({'role': 'independent_probe' if probes else 'arc_collocation',
                         'original_index': None, 'arc_id': arc['id'],
                         'arc_arclength_px': float(s), 'arc_t': float(s/span),
                         'point_gain': float(g), 'source_edge': int(i), 'source_edge_fraction': float(t)}
                        for s, g, i, t in zip(distance, gain, indices, fractions))
        reports.append({'arc_id': arc['id'], 'length_px': span, 'direction': arc['direction'],
                        'interval_count': intervals, 'sample_count': len(distance),
                        'actual_spacing_px': span/intervals, 'total_weight': float(weight.sum())})
    return {'source': np.vstack(sources), 'target': np.vstack(targets), 'weights': np.concatenate(weights),
            'controls': controls, 'report': {
                'mode': 'independent_arc_probes' if probes else 'arc_length_trapezoid_collocation',
                'source_contour_count': len(p), 'original_contour_unmodified': True,
                'new_detection_points': False, 'source_definition': 'unchanged reviewed piecewise-linear source contour',
                'amount': float(amount), 'max_radius_fraction': float(max_radius_fraction),
                'target_formula': 'jaw_arc_v1: -amount/100 * radius_fraction * radius * 64[t(1-t)]^3',
                'requested_spacing_px': float(spacing), 'phase': float(phase),
                'collocation_spacing_px': float(collocation_spacing) if probes else float(spacing),
                'target_density_per_source_px': TARGET_DENSITY,
                'outside_original_weight': 0. if probes else OUTSIDE_WEIGHT,
                'quadrature': 'midpoint' if probes else 'trapezoid',
                'probes_disjoint_from_collocation_nodes': bool(probes),
                'objective': '0.5/px * integral_arc squared target residual ds + 4 * sum_original_outside squared displacement',
                'new_objective_not_old_point_weights': True, 'arcs': reports,
                'aesthetic_approval': False}}


def sample_jaw_targets(geometry, amount, max_radius_fraction=.13, spacing=2., phase=0.):
    """Source/target Nx2 arrays, weights N, per-control metadata and report.

    Each arc includes both endpoints, with ceil(length/spacing) equal source
    arclength intervals and trapezoidal weights of density 0.5/px. Original rows
    outside the open arcs retain their exact source position and weight 4.
    Endpoint quadrature has phase 0; it cannot be shifted into a probe grid.
    """
    if not np.isfinite(phase) or phase != 0.:
        raise ValueError('endpoint trapezoid collocation requires phase 0')
    return _sample(geometry, amount, max_radius_fraction, spacing, phase,
                   probes=False, collocation_spacing=spacing)


def independent_jaw_probes(geometry, amount, max_radius_fraction=.13, spacing=.5,
                           phase=.5, collocation_spacing=2.):
    """Evaluate the same continuous intent strictly between training nodes.

    Subdivide each collocation interval equally until probe pitch <= spacing,
    then take midpoint samples (phase .5). Other interior phases are allowed.
    No probe copies a training point: (integer+phase)/subdivisions is never an
    integer. This is independent sampling, not an independent target formula.
    Probe weights describe the same line measure; probes are not fit controls.
    """
    if (not np.isfinite([phase, collocation_spacing]).all() or not 0. < phase < 1.
            or collocation_spacing <= 0.):
        raise ValueError('interior probe phase and positive collocation spacing required')
    return _sample(geometry, amount, max_radius_fraction, spacing, phase,
                   probes=True, collocation_spacing=collocation_spacing)
