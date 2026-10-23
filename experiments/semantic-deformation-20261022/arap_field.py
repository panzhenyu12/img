"""Isolated graph SO(2) ARAP experiment on the original portable source grid.

Per undirected edge i,j the regularizer is
    .5 * area / edge_length² * (||q-R_i p||² + ||q-R_j p||²),
where p is the original physical edge and q=p+d_j-d_i. Setting every R=I
exactly reproduces portable_field's membrane term. This discrete graph energy
is not a continuous exact integral, a global optimum, a no-fold certificate,
or a claim about facial anatomy or photographic quality.
"""
import time

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu


def _grid(displacement, spacing):
    displacement = np.asarray(displacement, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if (displacement.ndim != 3 or displacement.shape[-1] != 2
            or min(displacement.shape[:2]) < 2 or not np.isfinite(displacement).all()
            or spacing.shape != (2,) or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise ValueError('finite 2D displacement grid and positive physical xy spacing required')
    return displacement, spacing


def graph_rotations(displacement, spacing):
    """Minimize each source vertex's incident-edge energy over proper SO(2)."""
    displacement, spacing = _grid(displacement, spacing)
    covariance = np.zeros((*displacement.shape[:2], 2, 2))
    area = float(np.prod(spacing))
    for component, axis in [(0, 1), (1, 0)]:
        first = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        last = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        rest = spacing[component]*np.eye(2)[component]
        deformed = np.diff(displacement, axis=axis)+rest
        moment = (area/spacing[component]**2)*deformed[..., :, None]*rest
        # Reversing both incident rest/deformed edges leaves their outer
        # product unchanged. Fixed nodes participate in this local fit too.
        covariance[first] += moment
        covariance[last] += moment
    cosine_term = covariance[..., 0, 0]+covariance[..., 1, 1]
    sine_term = covariance[..., 1, 0]-covariance[..., 0, 1]
    angle = np.arctan2(sine_term, cosine_term)
    # When both terms are exactly zero every SO(2) angle minimizes the local
    # objective. Pick identity deterministically; never introduce reflection.
    angle[(cosine_term == 0)&(sine_term == 0)] = 0.
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.stack([cosine, -sine, sine, cosine], axis=-1).reshape(covariance.shape)


def graph_arap_energy(displacement, spacing, rotations=None):
    """Full endpoint-rotation residual sum, with original physical edge weights."""
    displacement, spacing = _grid(displacement, spacing)
    if rotations is None:
        rotations = graph_rotations(displacement, spacing)
    rotations = np.asarray(rotations, dtype=float)
    if rotations.shape != (*displacement.shape[:2], 2, 2) or not np.isfinite(rotations).all():
        raise ValueError('one finite 2x2 rotation per source node required')
    if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(2), rtol=0, atol=1e-12):
        raise ValueError('orthogonal source rotations required')
    if np.any(np.linalg.det(rotations) < 0):
        raise ValueError('proper SO(2) source rotations required')
    area = float(np.prod(spacing))
    total = 0.
    for component, axis in [(0, 1), (1, 0)]:
        first = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        last = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        unit = np.eye(2)[component]
        # q-Rp = delta d - (R-I)p, avoiding large absolute coordinate subtraction.
        delta = np.diff(displacement, axis=axis)
        residual_i = delta-spacing[component]*(rotations[first][..., component]-unit)
        residual_j = delta-spacing[component]*(rotations[last][..., component]-unit)
        total += .5*area/spacing[component]**2 * (np.sum(residual_i**2)+np.sum(residual_j**2))
    return float(total)


def solve_problem(problem, *, max_iterations=8):
    """Run at most eight local/global steps, starting at the membrane solution.

    All target, bending, membrane coefficient, fixed support and global-H
    entries remain unchanged. Only the edge rest directions in the RHS vary.
    This isolated candidate rejects reduced bases and background equalities;
    it never silently solves a less constrained photograph problem.
    """
    from portable_field import evaluate_motion, source_geometry

    started = time.perf_counter()
    if type(max_iterations) is not int or not 1 <= max_iterations <= 8:
        raise ValueError('max_iterations must be an integer from 1 to 8')
    h = csc_matrix(problem['hessian'])
    n = h.shape[0]
    r = csc_matrix(problem['reduction'])
    if r.shape != (n, n) or (r != eye(n, format='csc')).nnz:
        raise ValueError('graph ARAP currently requires an identity reduction (stride1)')
    lines = csc_matrix(problem['line_equalities'])
    if lines.shape != (0, 2*n):
        raise ValueError('graph ARAP currently requires no background line equalities')
    original_rhs = np.asarray(problem['rhs'], dtype=float)
    if (n < 1 or h.shape != (n, n) or original_rhs.shape != (n, 2)
            or not np.isfinite(h.data).all() or not np.isfinite(original_rhs).all()):
        raise ValueError('finite matching portable objective required')
    grid = problem['grid']
    spacing = np.asarray(grid['spacing'], dtype=float)
    area = problem['diagnostics']['cell_area_px2']
    membrane_weight = problem['diagnostics']['membrane_weight']
    bending_weight = problem['diagnostics']['bending_weight']
    row_norm = float(np.asarray(abs(h).sum(axis=1)).max())

    def field(solution):
        return (problem['prolongation'] @ solution).reshape(grid['height'], grid['width'], 2)

    def objective(solution, rotations):
        target_error = problem['target_operator'] @ solution-problem['target_deltas']
        target_energy = float(np.sum(problem['target_weights'][:, None]*target_error**2))
        bending_energy = float(bending_weight*area*np.sum((problem['laplacian'] @ solution)**2))
        arap_energy = membrane_weight*graph_arap_energy(field(solution), spacing, rotations)
        total = target_energy+bending_energy+arap_energy
        if not np.isfinite(total):
            raise ValueError('nonfinite graph ARAP objective')
        return {'target': target_energy, 'bending': bending_energy,
                'graph_arap': arap_energy, 'total': total}

    def right_hand_side(rotations):
        # Dx/Dy contain 1/physical_spacing; divide the physical rotated edge
        # by that same spacing to obtain (Ravg-I)e_axis, with no extra area.
        shift_x = .5*(rotations[:, :-1, :, 0]+rotations[:, 1:, :, 0])-[1., 0.]
        shift_y = .5*(rotations[:-1, :, :, 1]+rotations[1:, :, :, 1])-[0., 1.]
        return original_rhs + membrane_weight*area*(
            problem['Dx'].T @ shift_x.reshape(-1, 2)
            + problem['Dy'].T @ shift_y.reshape(-1, 2))

    def backward_error(solution, rhs):
        residual = h @ solution-rhs
        denominator = row_norm*np.max(abs(solution), axis=0)+np.max(abs(rhs), axis=0)
        error = np.max(abs(residual), axis=0)/np.maximum(denominator, np.finfo(float).tiny)
        return float(error.max())

    factor_started = time.perf_counter()
    factor = splu(h)
    factor_seconds = time.perf_counter()-factor_started
    solution = factor.solve(original_rhs)
    if not np.isfinite(solution).all() or backward_error(solution, original_rhs) > 1e-10:
        raise ValueError('original membrane starting solve failed')
    rotations = graph_rotations(field(solution), spacing)
    initial_energy = objective(solution, rotations)
    initial_membrane_energy = (initial_energy['target']+initial_energy['bending']
        + membrane_weight*area*(np.sum((problem['Dx'] @ solution)**2)
                                + np.sum((problem['Dy'] @ solution)**2)))
    energy = initial_energy
    history = []
    converged = False
    step_tolerance_px = 1e-7
    relative_energy_tolerance = 1e-10
    monotonicity_tolerance = 1e-12
    last_backward = backward_error(solution, original_rhs)
    for iteration in range(1, max_iterations+1):
        rhs = right_hand_side(rotations)
        candidate = factor.solve(rhs)
        last_backward = backward_error(candidate, rhs)
        if not np.isfinite(candidate).all() or not np.isfinite(last_backward) or last_backward > 1e-10:
            raise ValueError('graph ARAP global linear solve failed')
        global_energy = objective(candidate, rotations)
        new_rotations = graph_rotations(field(candidate), spacing)
        new_energy = objective(candidate, new_rotations)
        if (global_energy['total'] > energy['total']+monotonicity_tolerance*max(1., energy['total'])
                or new_energy['total'] > global_energy['total']
                    + monotonicity_tolerance*max(1., global_energy['total'])):
            raise ValueError('graph ARAP local/global objective increased')
        step = float(np.linalg.norm(candidate-solution, axis=1).max(initial=0.))
        relative_change = abs(new_energy['total']-energy['total'])/max(
            abs(energy['total']), np.finfo(float).tiny)
        history.append({'iteration': iteration, 'before': energy, 'after_global': global_energy,
                        'after_local': new_energy, 'max_node_step_px': step,
                        'relative_energy_change': float(relative_change),
                        'global_linear_backward_error': last_backward})
        solution, rotations, energy = candidate, new_rotations, new_energy
        if step <= step_tolerance_px and relative_change <= relative_energy_tolerance:
            converged = True
            break
    displacement = field(solution)
    actual = problem['source_points']+problem['target_operator'] @ solution
    error = np.linalg.norm(actual-problem['target_points'], axis=1)
    geometry = source_geometry(displacement/spacing)
    fixed_motion = evaluate_motion(problem, solution, problem['fixed_pixels'])
    # The last local update changes rotations after the linear solve. Report
    # its final stationarity separately; do not reuse the tiny linear residual
    # to claim nonlinear convergence or original membrane optimality.
    final_stationarity = backward_error(solution, right_hand_side(rotations))
    report = {
        'status': 'ARAPConverged' if converged else 'ARAPIterationLimit',
        'solver': 'SO(2) closed-form local polar fit + reused SciPy SuperLU global factor',
        'converged': converged, 'iterations': len(history), 'max_iterations': max_iterations,
        'convergence_criterion': 'both maximum physical node step and relative full-objective change',
        'step_tolerance_px': step_tolerance_px, 'relative_energy_tolerance': relative_energy_tolerance,
        'monotonicity_roundoff_tolerance': monotonicity_tolerance,
        'initialization': 'original portable membrane solution with unchanged H and RHS',
        'initial_membrane_objective': float(initial_membrane_energy),
        'initial_graph_objective': initial_energy, 'final_graph_objective': energy,
        'energy_history': history, 'factorization_count': 1, 'linear_solve_count': 1+len(history),
        'global_hessian_unchanged': True, 'target_and_bending_terms_unchanged': True,
        'fixed_nodes_in_local_rotation_fit': True, 'local_rotations_are_proper': True,
        'last_global_linear_backward_error': last_backward,
        'final_frozen_rotation_stationarity_backward_error': final_stationarity,
        'global_optimality_claimed': False, 'full_space_optimality_claimed': False,
        'geometry_constraints_enforced': False, 'aesthetic_approval': False,
        'original_line_rows_checked': 0, 'full_line_max_residual': 0., 'full_line_unit_max_residual': 0.,
        'factor_seconds': factor_seconds, 'complete_seconds': time.perf_counter()-started,
    }
    return {'solution': solution, 'coefficients': solution.copy(), 'multipliers': np.empty(0),
            'rotations': rotations, 'report': report, 'source_displacement': displacement,
            'actual_targets': actual, 'target_error_px': error,
            'target_max_error_px': float(error.max(initial=0.)),
            'fixed_source_motion_max_px': float(np.linalg.norm(fixed_motion, axis=1).max(initial=0.)),
            'geometry': geometry,
            'inverse_evaluation_allowed': bool(geometry['strong_monotonicity_lower_bound'] > 0
                                               and geometry['boundary_zero']),
            'aesthetic_approval': False}
