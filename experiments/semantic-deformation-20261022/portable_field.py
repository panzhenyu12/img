"""Source-grid deformation for a new photograph with explicit geometric roles.

This creates a new quadratic problem, not a reconstructed historical contract.
No photograph semantics are inferred. Coordinates and displacements are source
pixel indices/px; image_shape is (height, width), crop is [x,y,width,height].

The scalar energy per displacement component is
    sum_i w_i * (B_i u - delta_i)^2
    + bending_weight * cell_area * ||physical_laplacian u||^2
    + membrane_weight * cell_area * (||Dx u||^2 + ||Dy u||^2).
Weights are explicit numerical controls, not learned anatomical parameters.
Default spacing is 2px. Bending weight has px^2 relative to the target term;
membrane weight is dimensionless. The image field is C0 bilinear. A successful
linear solve does not establish final inverse geometry or photographic quality.
"""
import copy
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron

from reduced_field import ROOT, build_reduction, solve_reduced
from adapt import point_operators, square_nodes
from source_forward_input import line_constraints

sys.path.insert(0, str(ROOT/'experiments/face-source-forward-prototype-20261021'))
from forward import source_geometry


def _points(value, name, *, allow_empty=False):
    result = np.asarray(value, dtype=float)
    if allow_empty and result.size == 0:
        result = np.empty((0, 2))
    if (result.ndim != 2 or result.shape[1] != 2
            or (not allow_empty and not len(result)) or not np.isfinite(result).all()):
        raise ValueError(name+' must contain finite xy points')
    return result.copy()


def build_problem(image_shape, source_points, target_points, fixed_pixels, *, crop,
                  spacing=2., target_weights=1., bending_weight=.05,
                  membrane_weight=.001, lines=None, stride=4, fine_band=2):
    """Assemble without inference, historical files, shape clipping or solving.

    Fixed pixels have a closed +/-1px source square: all its bilinear vertices
    are locked. The field is identity outside crop, with a two-node zero collar.
    A crop must exactly fit the declared spacing; it is never silently resized.
    No rigid moving material is assumed by this initial interface. Callers must
    supply a suitable fixed role or use another solver for moving attachments.
    """
    started = time.perf_counter()
    if (len(image_shape) != 2 or any(type(n) is not int or n < 3 for n in image_shape)
            or len(crop) != 4 or any(type(n) is not int for n in crop)):
        raise ValueError('integer image shape and crop are required')
    ih, iw = image_shape
    left, top, cw, ch = crop
    if (not np.isfinite(spacing) or spacing <= 0 or min(cw, ch) < 5
            or min(left, top) < 0 or left+cw > iw or top+ch > ih):
        raise ValueError('positive spacing and a crop inside the source image required')
    spans = np.array([cw-1, ch-1], float)/spacing
    if not np.array_equal(spans, np.round(spans)) or np.any(spans < 4):
        raise ValueError('crop must be exactly covered by at least five grid nodes per axis')
    if (not np.isfinite([bending_weight, membrane_weight]).all()
            or bending_weight < 0 or membrane_weight < 0
            or bending_weight+membrane_weight <= 0):
        raise ValueError('nonnegative regularization with one positive physical energy is required')
    source = _points(source_points, 'source targets')
    target = _points(target_points, 'requested targets')
    fixed = _points(fixed_pixels, 'fixed pixels', allow_empty=True)
    if target.shape != source.shape:
        raise ValueError('source and requested target dimensions differ')
    for name, points in [('source', source), ('requested', target), ('fixed', fixed)]:
        if np.any(points < 0) or np.any(points > [iw-1, ih-1]):
            raise ValueError(name+' points lie outside the source photograph')
    if np.any(target < [left, top]) or np.any(target > [left+cw-1, top+ch-1]):
        raise ValueError('requested target outside the fixed ROI boundary')
    if not np.array_equal(fixed, np.round(fixed)):
        raise ValueError('fixed pixels must be integer source pixel indices')
    weights = np.broadcast_to(np.asarray(target_weights, float), (len(source),)).copy()
    delta = target-source
    moving = np.any(delta != 0., axis=1)
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(moving & (weights == 0)):
        raise ValueError('finite nonnegative weights; every moving target needs positive weight')
    seen = {}
    for p, q, weight in zip(source, target, weights):
        if weight == 0:
            continue
        key = tuple(p)
        if key in seen and not np.array_equal(seen[key], q):
            raise ValueError('conflicting repeated source target: '+str(key))
        seen[key] = q
    grid = {'origin': [left, top], 'spacing': [float(spacing), float(spacing)],
            'width': int(spans[0])+1, 'height': int(spans[1])+1}
    w, h = grid['width'], grid['height']
    n = w*h
    locked = np.zeros((h, w), bool)
    if len(fixed):
        locked.ravel()[square_nodes(fixed, grid, identity_extension=True)] = True
    locked[:2] = locked[-2:] = True
    locked[:, :2] = locked[:, -2:] = True
    free = np.flatnonzero(~locked.ravel())
    if not len(free):
        raise ValueError('fixed source supports leave no movable grid nodes')
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[free] = np.arange(len(free))
    prolong = coo_matrix((np.ones(len(free)), (free, mapping[free])),
                        shape=(n, len(free))).tocsr()
    target_op, _, _ = point_operators(source, grid, mapping, len(free))
    no_support = np.asarray(target_op.getnnz(axis=1)).ravel() == 0
    if np.any(moving & no_support):
        raise ValueError('moving targets have fully fixed source support: '+str(np.flatnonzero(moving & no_support).tolist()))
    lx = diags([-np.ones(w-1), 2*np.ones(w), -np.ones(w-1)], [-1, 0, 1])/spacing**2
    ly = diags([-np.ones(h-1), 2*np.ones(h), -np.ones(h-1)], [-1, 0, 1])/spacing**2
    laplacian = ((kron(eye(h), lx)+kron(ly, eye(w)))@prolong).tocsr()
    one_x = diags([-np.ones(w-1), np.ones(w-1)], [0, 1], shape=(w-1, w))/spacing
    one_y = diags([-np.ones(h-1), np.ones(h-1)], [0, 1], shape=(h-1, h))/spacing
    dx = (kron(eye(h), one_x)@prolong).tocsr()
    dy = (kron(one_y, eye(w))@prolong).tocsr()
    area = spacing**2
    hessian = (target_op.T@diags(weights)@target_op
               + bending_weight*area*(laplacian.T@laplacian)
               + membrane_weight*area*(dx.T@dx+dy.T@dy)).tocsc()
    rhs = target_op.T@(weights[:, None]*delta)
    declared_lines = copy.deepcopy([] if lines is None else lines)
    equalities, line_reports = line_constraints(declared_lines, grid, mapping, len(free))
    # A homogeneous background line cannot also be requested to leave that
    # line. Reject the explicit semantic conflict before a soft target solve
    # can silently average it away. Near but off-line controls remain legal.
    for line in declared_lines:
        start, end = np.asarray(line['start'], float), np.asarray(line['end'], float)
        direction = end-start
        length = np.linalg.norm(direction)
        tangent = direction/length
        normal = np.array([-tangent[1], tangent[0]])
        along = (source-start)@tangent
        on_line = (abs((source-start)@normal) <= 1e-9)&(along >= 0)&(along <= length)
        conflict = on_line & (abs(delta@normal) > 1e-9) & (weights > 0)
        if np.any(conflict):
            raise ValueError('target conflicts with a hard background line: '+str(np.flatnonzero(conflict).tolist()))
    reduction, reduction_report = build_reduction(mapping, w, h, [], stride=stride, fine_band=fine_band)
    reduced_support = np.asarray((target_op@reduction).getnnz(axis=1)).ravel()
    if np.any(moving & (reduced_support == 0)):
        raise ValueError('reduction would silently lock a moving source target')
    return {'image_shape': [ih, iw], 'crop': list(crop), 'grid': grid,
            'source_points': source, 'target_points': target, 'target_weights': weights,
            'target_deltas': delta, 'fixed_pixels': fixed, 'locked_nodes': np.flatnonzero(locked),
            'mapping': mapping, 'prolongation': prolong, 'target_operator': target_op,
            'laplacian': laplacian, 'Dx': dx, 'Dy': dy, 'hessian': hessian, 'rhs': rhs,
            'reduction': reduction, 'line_equalities': equalities,
            'diagnostics': {'problem': 'portable-source-field-v1', 'source_role_inference': False,
                'history_or_native_trace_required': False, 'target_count': len(source),
                'moving_target_count': int(moving.sum()), 'fixed_source_pixel_count': len(fixed),
                'fixed_node_count': int(locked.sum()), 'full_scalar_variables': len(free),
                'bending_weight': float(bending_weight), 'membrane_weight': float(membrane_weight),
                'cell_area_px2': area, 'reduction': reduction_report, 'background_lines': line_reports,
                'positive_definiteness': 'Dirichlet zero collar with positive membrane or squared discrete Laplacian energy; independent free-node columns.',
                'continuity': 'C0 bilinear source field, zero collar, identity exterior; no final inverse or beauty approval.',
                'construction_seconds': time.perf_counter()-started}}


def evaluate_motion(problem, solution, points):
    """Physical source displacement at points, including the identity exterior."""
    p = _points(points, 'evaluation points', allow_empty=True)
    z = np.asarray(solution, float)
    if z.shape != (problem['hessian'].shape[0], 2) or not np.isfinite(z).all():
        raise ValueError('finite two-component solution of matching dimension required')
    grid = problem['grid']
    uv = (p-grid['origin'])/grid['spacing']
    inside = np.all((uv >= 0)&(uv <= [grid['width']-1, grid['height']-1]), axis=1)
    value = np.zeros_like(p)
    if np.any(inside):
        op, _, _ = point_operators(p[inside], grid, problem['mapping'], len(z))
        value[inside] = op@z
    return value


def solve_problem(problem):
    """Solve once with unchanged weights; expose geometry without clipping it."""
    result = solve_reduced(problem['hessian'], problem['rhs'], problem['reduction'],
                           problem['line_equalities'])
    z = result['solution']
    grid = problem['grid']
    field = (problem['prolongation']@z).reshape(grid['height'], grid['width'], 2)
    actual = problem['source_points']+problem['target_operator']@z
    error = np.linalg.norm(actual-problem['target_points'], axis=1)
    geometry = source_geometry(field/grid['spacing'])
    fixed_motion = evaluate_motion(problem, z, problem['fixed_pixels'])
    return {**result, 'source_displacement': field, 'actual_targets': actual,
            'target_error_px': error, 'target_max_error_px': float(error.max(initial=0.)),
            'fixed_source_motion_max_px': float(np.linalg.norm(fixed_motion, axis=1).max(initial=0.)),
            'geometry': geometry,
            'inverse_evaluation_allowed': bool(geometry['strong_monotonicity_lower_bound'] > 0
                                               and geometry['boundary_zero']),
            'aesthetic_approval': False}
