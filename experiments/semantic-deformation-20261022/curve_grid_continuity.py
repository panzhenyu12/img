"""Linear C1 trace constraints inside each original source-polyline edge.

Only a grid crossing strictly inside one original straight source edge is
constrained. A source vertex keeps its original corner, even on a grid line.
This is a structural experiment, not an anatomical or image-quality gate.
"""
from fractions import Fraction

import numpy as np
from scipy.sparse import coo_matrix


def _fraction(value):
    return Fraction.from_float(float(value))


def _one_sided_cell(uv, direction, counts, side):
    """Choose the limiting cell combinatorially, without moving the point."""
    cell = []
    for q, d, count in zip(uv, direction, counts):
        index = q.numerator//q.denominator
        if q.denominator == 1:
            if d*side < 0:
                index -= 1
            elif d == 0 and index == count-1:
                # A source edge on the outer grid line uses its inside cell.
                # Its tangential trace is shared by both sides when internal.
                index -= 1
        if not 0 <= index < count-1:
            raise ValueError('source crossing has no one-sided cell inside the grid')
        cell.append(index)
    return cell


def _directional_coefficients(uv, cell, tangent, spacing, width):
    """Analytic t·grad(Q1), in physical original-arclength units."""
    x, y = cell
    tx, ty = [float(q-index) for q, index in zip(uv, cell)]
    dx = np.array([-(1.-ty), 1.-ty, -ty, ty])/spacing[0]
    dy = np.array([-(1.-tx), -tx, 1.-tx, tx])/spacing[1]
    nodes = np.array([y*width+x, y*width+x+1,
                      (y+1)*width+x, (y+1)*width+x+1])
    return nodes, tangent[0]*dx+tangent[1]*dy


def curve_grid_continuity(curve):
    """Return ``A z = b=0`` for analytic after-minus-before trace derivatives.

    ``curve`` follows GridCurve's points/knots/problem/size interface. Columns
    address only its scalar free-node variables; mapping -1 is the existing
    fixed-zero node convention. No auxiliary variables, weights or solves are
    introduced. Zero rows are retained with their crossing provenance.

    Exact rational predicates on the supplied finite binary coordinates keep
    source vertices, simultaneous grid-corner crossings and genuinely nearby
    crossings distinct. Only coefficient evaluation is rounded to float64.
    """
    points = np.asarray(curve.points, float)
    knots = np.asarray(curve.knots, float)
    grid = curve.problem['grid']
    origin, spacing = np.asarray(grid['origin'], float), np.asarray(grid['spacing'], float)
    counts = [grid['width'], grid['height']]
    mapping = np.asarray(curve.problem['mapping'])
    size = curve.size
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all() or knots.shape != (len(points),)
            or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0)):
        raise ValueError('finite original points and increasing source arclength required')
    if (origin.shape != (2,) or spacing.shape != (2,) or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or any(not isinstance(n, (int, np.integer)) or n < 2 for n in counts)):
        raise ValueError('finite grid origin, positive spacing and at least two nodes per axis required')
    if (not isinstance(size, (int, np.integer)) or size < 0
            or mapping.shape != (counts[0]*counts[1],)
            or not np.issubdtype(mapping.dtype, np.integer)
            or np.any(mapping < -1) or np.any(mapping >= size)):
        raise ValueError('scalar-node mapping must contain fixed -1 or valid free-node columns')
    exact_origin = list(map(_fraction, origin))
    exact_spacing = list(map(_fraction, spacing))
    exact_points = [list(map(_fraction, point)) for point in points]
    exact_uv = [[(p-o)/h for p, o, h in zip(point, exact_origin, exact_spacing)]
                for point in exact_points]
    if any(q < 0 or q > n-1 for uv in exact_uv for q, n in zip(uv, counts)):
        raise ValueError('original source points must lie inside the unchanged grid')

    excluded = []
    for i, uv in enumerate(exact_uv):
        axes = [axis for axis, q in enumerate(uv) if q.denominator == 1]
        if axes:
            excluded.append({'source_vertex_index': i, 'point': points[i].tolist(),
                'source_arclength_px': float(knots[i]),
                'grid_axes': ['xy'[axis] for axis in axes],
                'grid_line_indices': [int(uv[axis]) for axis in axes]})

    rows, columns, values = [], [], []
    crossings, edge_reports = [], []
    for edge, (a, b) in enumerate(zip(exact_points[:-1], exact_points[1:])):
        delta = [last-first for first, last in zip(a, b)]
        if delta == [0, 0]:
            raise ValueError('original source edge must have positive geometric length')
        tangent = (points[edge+1]-points[edge])/(knots[edge+1]-knots[edge])
        events = {}
        for axis, d in enumerate(delta):
            if d == 0:
                continue
            for index in range(1, counts[axis]-1):
                line = exact_origin[axis]+index*exact_spacing[axis]
                parameter = (line-a[axis])/d
                if 0 < parameter < 1:
                    events.setdefault(parameter, []).append((axis, index))
        edge_reports.append({'source_edge_index': edge,
            'source_arclength_interval_px': [float(knots[edge]), float(knots[edge+1])],
            'tangent_xy': tangent.tolist(), 'crossing_count': len(events),
            'axis_aligned': any(d == 0 for d in delta),
            'abs_quadratic_trace_factor': float(abs(2.*tangent[0]*tangent[1]))})
        for parameter, axes in sorted(events.items()):
            point = [coordinate+parameter*d for coordinate, d in zip(a, delta)]
            uv = [(p-o)/h for p, o, h in zip(point, exact_origin, exact_spacing)]
            before = _one_sided_cell(uv, delta, counts, -1)
            after = _one_sided_cell(uv, delta, counts, 1)
            row = len(crossings)
            for cell, sign in [(before, -1.), (after, 1.)]:
                nodes, coefficients = _directional_coefficients(uv, cell, tangent, spacing, counts[0])
                for column, value in zip(mapping[nodes], sign*coefficients):
                    if column >= 0 and value != 0.:
                        rows.append(row); columns.append(int(column)); values.append(float(value))
            crossings.append({'row': row, 'source_edge_index': edge,
                'source_arclength_px': float(knots[edge]+float(parameter)*(knots[edge+1]-knots[edge])),
                'edge_parameter': float(parameter), 'point': list(map(float, point)),
                'grid_axes': ['xy'[axis] for axis, _ in axes],
                'grid_line_indices': [index for _, index in axes],
                'before_cell_xy': before, 'after_cell_xy': after,
                'tangent_xy': tangent.tolist()})

    matrix = coo_matrix((values, (rows, columns)), shape=(len(crossings), size)).tocsr()
    matrix.sum_duplicates(); matrix.eliminate_zeros()
    nonaxis_factors = [r['abs_quadratic_trace_factor'] for r in edge_reports if not r['axis_aligned']]
    report = {'schema': 'original-edge-grid-trace-c1-v1',
        'equation': 'A z = 0; t dot (grad u after - grad u before)',
        'coefficient_units': 'inverse source pixels; A z is displacement px per source-arclength px',
        'scalar_variables': int(size), 'equation_count': len(crossings),
        'zero_support_equations': int(np.count_nonzero(matrix.getnnz(axis=1) == 0)),
        'crossings': crossings, 'source_edges': edge_reports,
        'excluded_source_vertices_on_grid': excluded,
        'axis_aligned_source_edge_count': sum(r['axis_aligned'] for r in edge_reports),
        'minimum_nonaxis_abs_quadratic_trace_factor': min(nonaxis_factors, default=None),
        'topology': 'exact rational predicates on original binary coordinates; no epsilon, snapping or merging of nearby crossings',
        'derivatives': 'analytic one-sided bilinear cell gradients at the same physical crossing; no finite differences',
        'scope': 'strict interior of each unchanged original straight edge; no continuity constraint across original vertices',
        'risks': ['Axis-aligned Q1 traces become affine along each original edge; near-axis curvature freedom is small.',
                  'Multiple curves constraining different points of the same grid edge can strengthen continuity beyond a single trace.',
                  'Existing area, fixed supports and other constraints may make the nonzero request infeasible.'],
        'image_approval': False}
    return {'A': matrix, 'b': np.zeros(len(crossings)), 'report': report}
