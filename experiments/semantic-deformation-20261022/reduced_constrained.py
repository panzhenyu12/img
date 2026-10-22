"""One bounded SOC experiment in the existing reduced source subspace.

The old source cones and final-image thresholds are untouched. An active-set
solve is not a certificate for omitted corners or the actual f32 inverse.
"""
import time

import numpy as np
from scipy.sparse import block_diag, csc_matrix, eye, hstack, triu, vstack

from reduced_field import solve_reduced
from line_equality_solve import independent_rows

MAX_ROUNDS = 2
ROUND_TIME_LIMIT = 12.
MAX_NEW_SEEDS = 256
SOURCE_SCAN_TOLERANCE = 1e-8


def project_source_cones(grid, mapping, reduction, active):
    """Compose the original cone operator with diag(R, R, I_aux)."""
    from constrained_solve import assemble_cones
    r = csc_matrix(reduction)
    matrix, offset, cones = assemble_cones(grid, mapping, r.shape[0], active)
    lift = block_diag([r, r, eye(len(active), format='csc')], format='csc')
    projected = (matrix @ lift).tocsc()
    projected.sum_duplicates(); projected.eliminate_zeros()
    return projected, offset, cones


def _background_evidence(lines, solution):
    residual = lines @ solution.T.ravel()
    norms = np.sqrt(np.asarray(lines.multiply(lines).sum(axis=1)).ravel())
    unit = np.divide(residual, norms, out=np.zeros_like(residual), where=norms > 0)
    raw_max = float(abs(residual).max(initial=0.))
    unit_max = float(abs(unit).max(initial=0.))
    return {'original_rows': lines.shape[0], 'original_max_residual': raw_max,
            'original_unit_max_residual': unit_max,
            'passed': bool(np.isfinite(residual).all() and raw_max <= 1e-7 and unit_max <= 1e-9)}


def _solve_round(h, rhs, r, grid, mapping, active, independent_lines):
    # Clarabel is optional: the unconstrained reduced experiment must remain
    # runnable in the lean worker environment.
    try:
        import clarabel
    except ImportError as error:
        raise RuntimeError('optional --constrained branch requires Clarabel') from error
    started = time.perf_counter()
    k = r.shape[1]
    matrix, offset, cones = project_source_cones(grid, mapping, r, active)
    soc_rows = matrix.shape[0]
    if independent_lines.shape[0]:
        equality = hstack([independent_lines, csc_matrix((independent_lines.shape[0], len(active)))], format='csc')
        matrix = vstack([matrix, equality], format='csc')
        offset = np.r_[offset, np.zeros(equality.shape[0])]
        cones.append(clarabel.ZeroConeT(equality.shape[0]))
    objective = triu(block_diag([2*h, 2*h, csc_matrix((len(active), len(active)))], format='csc')).tocsc()
    linear = np.r_[-2*rhs[:, 0], -2*rhs[:, 1], np.zeros(len(active))]
    assembly_seconds = time.perf_counter() - started
    settings = clarabel.DefaultSettings()
    settings.verbose = False; settings.max_iter = 150
    settings.tol_feas = 1e-9; settings.tol_gap_abs = 1e-8; settings.tol_gap_rel = 1e-8
    settings.time_limit = ROUND_TIME_LIMIT
    settings.direct_solve_method = 'faer'; settings.max_threads = 6
    init_started = time.perf_counter()
    solver = clarabel.DefaultSolver(objective, linear, matrix, offset, cones, settings)
    initialization_seconds = time.perf_counter() - init_started
    solve_started = time.perf_counter(); answer = solver.solve()
    solve_seconds = time.perf_counter() - solve_started
    status = str(answer.status); x = np.asarray(answer.x, float)
    report = {'status': status, 'iterations': answer.iterations,
              'assembly_seconds': assembly_seconds, 'initialization_seconds': initialization_seconds,
              'solve_seconds': solve_seconds, 'complete_seconds': time.perf_counter()-started,
              'solver_time_limit_seconds': ROUND_TIME_LIMIT,
              'variables_including_auxiliaries': len(linear), 'constraint_rows': matrix.shape[0],
              'constraint_nnz': matrix.nnz, 'r_prim': float(answer.r_prim), 'r_dual': float(answer.r_dual),
              'objective_primal_without_constant': float(answer.obj_val),
              'objective_dual_without_constant': float(answer.obj_val_dual),
              'full_space_optimality_claimed': False, 'final_image_geometry_pending': True}
    certificate = status in ['PrimalInfeasible', 'DualInfeasible', 'AlmostPrimalInfeasible', 'AlmostDualInfeasible']
    if certificate or x.shape != (len(linear),) or not np.isfinite(x).all():
        report.update(primal_candidate_available=False,
                      reason='infeasibility certificate is not a primal iterate' if certificate else 'nonfinite or malformed iterate')
        return None, report
    slack = (offset[:soc_rows] - matrix[:soc_rows] @ x).reshape(len(active), 11)
    violation = max(0., *[float(np.max(np.linalg.norm(slack[:, i+1:i+3], axis=1)-slack[:, i]))
                           for i in [0, 3, 6]], float(-slack[:, 9:].min()))
    report.update(primal_candidate_available=True, active_cone_violation_max=violation,
                  active_cones_pass=violation <= 1e-7)
    return np.column_stack([x[:k], x[k:2*k]]), report


def solve_constrained_reduced(h, rhs, reduction, grid, mapping, lines=None):
    """At most two fixed-budget active solves; return a clearly scoped result.

    Initial equality-only coefficients select constraints; no Clarabel warm
    start is claimed. The caller must independently check the final f32 field,
    full material RGB, fixed pixels, physical lines and photograph quality.
    """
    started = time.perf_counter()
    try:
        from constrained_solve import effective_bounds, inspect_corners
    except ImportError as error:
        raise RuntimeError('optional --constrained branch requires Clarabel') from error
    h, r = csc_matrix(h), csc_matrix(reduction)
    rhs = np.asarray(rhs, float); mapping = np.asarray(mapping)
    lines = csc_matrix((0, 2*h.shape[0])) if lines is None else csc_matrix(lines)
    initial = solve_reduced(h, rhs, r, lines)
    z, a = initial['solution'], initial['coefficients']
    reduced_h = (r.T @ h @ r).tocsc(); reduced_h = ((reduced_h + reduced_h.T)*.5).tocsc()
    reduced_rhs = r.T @ rhs
    reduced_lines = (lines @ block_diag([r, r], format='csc')).tocsc()
    independent, _, _, line_reduction = independent_rows(reduced_lines)
    active = set(); rounds = []; scans = []
    last_status = 'ReducedSubspaceSolved'; selected_stage = 'initial_equality_quadratic'
    stopped = 'round_budget_reached'; last_active_pass = True
    cell_count = (grid['width']-1)*(grid['height']-1)
    violation, qc = inspect_corners(z, grid, mapping)
    scans.append({'stage': selected_stage, **qc})
    if not qc['boundary_exact_zero']:
        raise ValueError('original fixed source boundary is not zero')
    for iteration in range(MAX_ROUNDS):
        if qc['corners_violating_1e_8'] == 0:
            stopped = 'all_original_source_proxy_corners_pass'; break
        candidates = np.flatnonzero(violation > SOURCE_SCAN_TOLERANCE)
        candidates = candidates[~np.isin(candidates, list(active))]
        if not len(candidates):
            stopped = 'previously_active_corners_still_fail_no_identical_rerun'; break
        seeds = candidates[np.argsort(-violation[candidates], kind='stable')[:MAX_NEW_SEEDS]]
        previous_count = len(active)
        for corner in seeds:
            cell = int(corner) % cell_count
            active.update(kind*cell_count+cell for kind in range(4))
        ordered = sorted(active)
        next_a, report = _solve_round(reduced_h, reduced_rhs, r, grid, mapping, ordered, independent)
        report.update(round=iteration+1, selected_seed_count=len(seeds), selected_seed_ids=seeds.tolist(),
                      new_active_corners=len(active)-previous_count, all_active_corner_ids=ordered,
                      original_background_rows=lines.shape[0], retained_same_cell_four_corners=True)
        rounds.append(report); last_status = report['status']
        if next_a is None:
            stopped = 'solver_returned_no_finite_primal_candidate'; break
        a = next_a; z = r @ a; selected_stage = 'round-'+str(iteration+1)
        last_active_pass = report['active_cones_pass']
        violation, qc = inspect_corners(z, grid, mapping)
        scans.append({'stage': selected_stage, **qc})
        report['complete_original_background'] = _background_evidence(lines, z)
        if not qc['boundary_exact_zero']:
            raise ValueError('reconstruction changed original fixed boundary')
    if qc['corners_violating_1e_8'] == 0:
        stopped = 'all_original_source_proxy_corners_pass'
    line_evidence = _background_evidence(lines, z)
    proxy_pass = bool(qc['corners_violating_1e_8'] == 0 and qc['boundary_exact_zero'])
    selected_solver_converged = bool(selected_stage.startswith('round-') and last_status == 'Solved')
    accepted = bool(proxy_pass and line_evidence['passed'] and last_active_pass and selected_solver_converged)
    return {'solution': z, 'coefficients': a,
            'report': {'status': last_status, 'selected_stage': selected_stage, 'stop_reason': stopped,
                       'optimization_scope': 'unchanged full quadratic restricted to z=R a with source SOC active subset',
                       'full_space_optimality_claimed': False, 'subspace_optimization_converged': accepted,
                       'source_direct_proxy_pass': proxy_pass,
                       # Backward-compatible field for the local regression
                       # tests; it is explicitly a proxy, never final quality.
                       'source_proxy_pass': proxy_pass,
                       'source_convex_gate_pass': bool(accepted),
                       'source_cone_certificate_attempted': bool(rounds),
                       'diagnostic_only': not accepted, 'aesthetic_approval': False,
                       'final_actual_f32_geometry_and_RGB_pending': True,
                       'maximum_rounds': MAX_ROUNDS, 'round_solver_time_limit_seconds': ROUND_TIME_LIMIT,
                       'maximum_new_seed_corners_per_round': MAX_NEW_SEEDS,
                       'effective_source_bounds': list(effective_bounds()), 'source_scan_tolerance': SOURCE_SCAN_TOLERANCE,
                       'all_source_corner_scan': qc, 'all_original_background': line_evidence,
                       'initial_reduced_solve': initial['report'], 'background_row_reduction': line_reduction,
                       'rounds': rounds, 'complete_corner_scan_history': scans,
                       'no_strength_or_target_scaling': True, 'no_threshold_changes': True,
                       'solver_limit_excludes_operator_assembly_and_initialization': True,
                       'complete_seconds': time.perf_counter()-started}}
