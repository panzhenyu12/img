"""Constraint generation for the fixed source quadratic proposal.

Every iteration keeps H/rhs unchanged and scans every source-cell corner.
The convex sufficient inverse-contract subset is intentionally conservative.
No unconstrained field is clipped or scaled, and no photo is rendered here.
"""
import argparse
import json
import time
from pathlib import Path

import clarabel
import numpy as np
from scipy.sparse import block_diag, coo_matrix, csc_matrix, load_npz, triu

MARGIN = 1e-5


class ActiveSolveFailure(ValueError):
    """Retain a rejected numerical iterate for diagnosis, never for rendering."""
    def __init__(self, reason, solution, report):
        super().__init__(reason)
        self.solution = solution
        self.report = report


def effective_bounds():
    return .52+MARGIN, 1.8-MARGIN, .24-MARGIN


def assemble_cones(grid, mapping, nvars, active):
    """Three 3D SOCs and two scalar inequalities per source cell corner.

    For F=DG, set t=trace(F)/2, d=(F11-F22)/2, s=(F12+F21)/2,
    o=(F21-F12)/2. norm2(F-rI)=hypot(t-r,o)+hypot(d,s).
    This represents sym(F)>=I/beta and sym(F)-alpha F^T F>=0,
    with |o|<=gamma/beta^2. One norm-split auxiliary per corner.
    """
    w, h = grid['width'], grid['height']; hx, hy = grid['spacing']
    cells = (w-1)*(h-1)
    mapping = np.asarray(mapping)
    if mapping.shape != (w*h,) or np.any(mapping < -1) or np.any(mapping >= nvars):
        raise ValueError('invalid eliminated source-node mapping')
    if len(set(active)) != len(active) or any(i < 0 or i >= 4*cells for i in active):
        raise ValueError('valid unique source corner IDs required')
    alpha, beta, gamma = effective_bounds(); radius = 1/(2*alpha)
    rr, cc, vv, offset, cones = [], [], [], [], []

    def merge(*terms):
        result = {}
        for weight, row in terms:
            for col, value in row.items():
                result[col] = result.get(col, 0.)+weight*value
        return {k: v for k, v in result.items() if v != 0.}

    def row(constant, coefficients):
        index = len(offset); offset.append(constant)
        for col, value in coefficients.items():
            rr.append(index); cc.append(col); vv.append(-value)

    for ai, corner in enumerate(active):
        kind, cell = divmod(int(corner), cells); y, x = divmod(cell, w-1)
        dxnodes = [(y+int(kind >= 2))*w+x, (y+int(kind >= 2))*w+x+1]
        dynodes = [y*w+x+kind % 2, (y+1)*w+x+kind % 2]

        def gradient(nodes, scale, component):
            result = {}
            for sign, node in zip([-1., 1.], nodes):
                if mapping[node] >= 0:
                    col = int(mapping[node])+component*nvars
                    result[col] = result.get(col, 0.)+sign/scale
            return result

        xx, xy = gradient(dxnodes, hx, 0), gradient(dynodes, hy, 0)
        yx, yy = gradient(dxnodes, hx, 1), gradient(dynodes, hy, 1)
        trace = merge((.5, xx), (.5, yy))
        difference = merge((.5, xx), (-.5, yy))
        symmetric = merge((.5, xy), (.5, yx))
        skew = merge((.5, yx), (-.5, xy))
        auxiliary = {2*nvars+ai: 1.}
        row(1-1/beta, trace); row(0., difference); row(0., symmetric)
        cones.append(clarabel.SecondOrderConeT(3))
        row(0., auxiliary); row(1-radius, trace); row(0., skew)
        cones.append(clarabel.SecondOrderConeT(3))
        row(radius, merge((-1., auxiliary))); row(0., difference); row(0., symmetric)
        cones.append(clarabel.SecondOrderConeT(3))
        row(gamma/beta**2, merge((-1., skew)))
        row(gamma/beta**2, skew)
        cones.append(clarabel.NonnegativeConeT(2))
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(offset), 2*nvars+len(active))).tocsc()
    matrix.sum_duplicates(); matrix.eliminate_zeros()
    return matrix, np.asarray(offset), cones


def solve_active(hessian, rhs, grid, mapping, active, time_limit=120., backend='faer'):
    nvars = hessian.shape[0]
    if hessian.shape != (nvars, nvars) or rhs.shape != (nvars, 2) or not active:
        raise ValueError('invalid reduced quadratic or empty active set')
    matrix, b, cones = assemble_cones(grid, mapping, nvars, active)
    p = triu(block_diag([2*hessian, 2*hessian, csc_matrix((len(active), len(active)))], format='csc')).tocsc()
    q = np.r_[-2*rhs[:, 0], -2*rhs[:, 1], np.zeros(len(active))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False; settings.max_iter = 150
    settings.tol_feas = 1e-9; settings.tol_gap_abs = 1e-8; settings.tol_gap_rel = 1e-8
    settings.time_limit = time_limit
    settings.direct_solve_method = backend; settings.max_threads = 6
    start = time.perf_counter()
    solver = clarabel.DefaultSolver(p, q, matrix, b, cones, settings)
    answer = solver.solve(); elapsed = time.perf_counter()-start
    x = np.asarray(answer.x)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite constrained proposal')
    slack = (b-matrix@x).reshape(len(active), 11)
    violation = max(0., *[float(np.max(np.linalg.norm(slack[:, i+1:i+3], axis=1)-slack[:, i])) for i in [0, 3, 6]],
                    float(-slack[:, 9:].min()))
    report = {'status': str(answer.status), 'diagnostic_only': str(answer.status) != 'Solved',
        'seconds': elapsed, 'iterations': answer.iterations, 'active_corners': len(active),
        'variables_including_norm_auxiliaries': len(q), 'constraint_rows': matrix.shape[0],
        'constraint_nnz': matrix.nnz, 'cone_violation_max': violation,
        'r_prim': answer.r_prim, 'r_dual': answer.r_dual,
        'objective_primal_without_constant': answer.obj_val,
        'objective_dual_without_constant': answer.obj_val_dual,
        'linear_solver': str(solver.get_info().linsolver)}
    solution = np.column_stack([x[:nvars], x[nvars:2*nvars]])
    if str(answer.status) not in ['Solved', 'AlmostSolved']:
        raise ActiveSolveFailure(f'active source solve rejected: {answer.status}', solution, report)
    if violation > 1e-7:
        raise ActiveSolveFailure(f'active source cone violation {violation} > 1e-7', solution, report)
    return solution, report


def inspect_corners(z, grid, mapping):
    w, h = grid['width'], grid['height']; hx, hy = grid['spacing']
    u = np.zeros((w*h, 2)); valid = mapping >= 0; u[valid] = z[mapping[valid]]
    u = u.reshape(h, w, 2)
    dx, dy = np.diff(u, axis=1)/hx, np.diff(u, axis=0)/hy
    f = np.stack([np.stack([x, y], axis=-1)+np.eye(2) for x, y in
        [(dx[:-1], dy[:, :-1]), (dx[:-1], dy[:, 1:]),
         (dx[1:], dy[:, :-1]), (dx[1:], dy[:, 1:])]]).reshape(-1, 2, 2)
    s = (f+f.swapaxes(-1, -2))*.5
    alpha, beta, gamma = effective_bounds()
    low = np.linalg.eigvalsh(s)[:, 0]
    schur = np.linalg.eigvalsh(s-alpha*(f.swapaxes(-1, -2)@f))[:, 0]
    skew = abs(f[:, 1, 0]-f[:, 0, 1])*.5
    violation = np.maximum.reduce([1/beta-low, -schur, skew-gamma/beta**2, np.zeros(len(f))])
    boundary = np.concatenate([u[0], u[-1], u[1:-1, 0], u[1:-1, -1]])
    report = {'all_source_corners': len(f), 'convex_violation_max': float(violation.max()),
        'corners_violating_1e_8': int((violation > 1e-8).sum()),
        'forward_symmetric_min': float(low.min()), 'forward_skew_max': float(skew.max()),
        'forward_schur_min': float(schur.min()), 'forward_determinant_min': float(np.linalg.det(f).min()),
        'boundary_exact_zero': bool(np.all(boundary == 0))}
    return violation, report


def run(inputs, proposal, output, max_rounds, batch_size, resume=None, time_limit=120., backend='faer', halo_cells=0):
    if max_rounds < 1 or batch_size < 1:
        raise ValueError('positive round and corner budgets required')
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('finite positive solver time limit required')
    if halo_cells < 0:
        raise ValueError('nonnegative source-cell constraint halo required')
    contract = json.loads((inputs/'contract.json').read_text())
    mapping = np.load(inputs/'data.npz')['node_to_variable']; grid = contract['grid']
    hessian = load_npz(proposal/'hessian.npz'); rhs = np.load(proposal/'rhs.npy')
    z = np.load(proposal/'solution.npy')
    output.mkdir(parents=True, exist_ok=False)
    active = set(); history = []
    latest_solver_status = 'Unconstrained'
    if resume is not None:
        previous = json.loads((resume.parent/'protocol.json').read_text())
        if previous['input'] != str(inputs) or previous['original_proposal'] != str(proposal):
            raise ValueError('constraint continuation must use the same input and original objective')
        z = np.load(resume/'solution.npy')
        active = set(np.load(resume/'active_corners.npy').tolist())
        latest_solver_status = json.loads((resume/'solve.json').read_text())['status']
        if z.shape != rhs.shape or not np.isfinite(z).all():
            raise ValueError('invalid constraint-generation continuation')
    protocol = {'input': str(inputs), 'original_proposal': str(proposal),
        'fixed_H_rhs_every_round': True, 'effective_inverse_bounds': effective_bounds(),
        'source_admissible_set': 'convex sufficient subset, not exact nonconvex pullback of inverse bounds',
        'maximum_rounds': max_rounds, 'added_corners_per_round_max': batch_size,
        'resume_from_round': str(resume) if resume is not None else None,
        'solver_time_limit_seconds_per_round': time_limit,
        'linear_solver_requested': backend, 'maximum_solver_threads': 6,
        'all_original_corners_in_added_cell_halo': halo_cells,
        'full_scan_after_each_round': True, 'no_field_scaling_or_clipping': True,
        'requires_final_actual_inverse_and_original_RGB_audit': True}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for iteration in range(max_rounds+1):
        violation, qc = inspect_corners(z, grid, mapping)
        if not qc['boundary_exact_zero']:
            raise ValueError('fixed source boundary was not preserved')
        record = {'iteration': iteration, 'full_scan': qc, 'active_count': len(active)}
        history.append(record)
        (output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if qc['corners_violating_1e_8'] == 0:
            np.save(output/'solution.npy', z)
            (output/'source-solver-decision.json').write_text(json.dumps({'source_convex_gate_pass': True,
                'all_original_photo_gates_pending': True, 'aesthetic_approval': False,
                'solver_status': latest_solver_status,
                'optimization_converged': latest_solver_status == 'Solved',
                'diagnostic_only': latest_solver_status != 'Solved'})+'\n')
            return
        if iteration == max_rounds:
            break
        candidates = np.flatnonzero(violation > 1e-8)
        candidates = candidates[~np.isin(candidates, list(active))]
        if not len(candidates):
            raise ValueError('previously constrained corners still fail strict global scan')
        order = np.argsort(-violation[candidates], kind='stable')[:batch_size]
        seeds = candidates[order].tolist()
        active.update(seeds)
        if halo_cells:
            cell_width, cell_height = grid['width']-1, grid['height']-1
            cell_count = cell_width*cell_height
            for corner in seeds:
                cy, cx = divmod(corner % cell_count, cell_width)
                for y in range(max(0, cy-halo_cells), min(cell_height, cy+halo_cells+1)):
                    for x in range(max(0, cx-halo_cells), min(cell_width, cx+halo_cells+1)):
                        active.update(k*cell_count+y*cell_width+x for k in range(4))
        folder = output/f'round-{iteration+1}'
        folder.mkdir()
        np.save(folder/'active_corners.npy', np.array(sorted(active)))
        try:
            z, solve = solve_active(hessian, rhs, grid, mapping, sorted(active), time_limit, backend)
        except (ValueError, RuntimeError) as error:
            failure = {'source_convex_gate_pass': False, 'failed_round': iteration+1,
                'reason': str(error), 'aesthetic_approval': False,
                'no_accepted_field_exported': True}
            (folder/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            (output/'source-solver-decision.json').write_text(json.dumps(failure, indent=2)+'\n')
            if isinstance(error, ActiveSolveFailure):
                np.save(folder/'rejected-solution.npy', error.solution)
                (folder/'rejected-solve.json').write_text(json.dumps(error.report, indent=2)+'\n')
                _, rejected_qc = inspect_corners(error.solution, grid, mapping)
                (folder/'rejected-source-scan.json').write_text(json.dumps(rejected_qc, indent=2)+'\n')
            raise
        np.save(folder/'solution.npy', z)
        (folder/'solve.json').write_text(json.dumps(solve, indent=2)+'\n')
        latest_solver_status = solve['status']
    (output/'source-solver-decision.json').write_text(json.dumps({'source_convex_gate_pass': False,
        'reason': 'round budget reached; no accepted field exported', 'aesthetic_approval': False})+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--proposal', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-rounds', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--resume-from-round', type=Path)
    p.add_argument('--solver-time-limit', type=float, default=120.)
    p.add_argument('--linear-solver', choices=['faer', 'qdldl'], default='faer')
    p.add_argument('--constraint-halo-cells', type=int, default=0)
    a = p.parse_args()
    run(a.input.resolve(), a.proposal.resolve(), a.output.resolve(), a.max_rounds, a.batch_size,
        a.resume_from_round.resolve() if a.resume_from_round else None, a.solver_time_limit, a.linear_solver,
        a.constraint_halo_cells)
