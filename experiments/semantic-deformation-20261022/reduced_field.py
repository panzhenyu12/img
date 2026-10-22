"""Nested coarse control lattice with exact existing fixed/translation roles.

Only the variable subspace changes: the source grid, objective and targets do
not. This is an engineering candidate, not a new published deformation method.
"""
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, csc_matrix, block_diag, eye
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/face-source-forward-photo-20261021'))
from line_equality_solve import solve_equality_quadratic


def build_reduction(mapping, width, height, groups, *, stride, fine_band, extra_nodes=None):
    started = time.perf_counter()
    m = np.asarray(mapping)
    groups = np.asarray(groups, dtype=int)
    if (width < 2 or height < 2 or m.shape != (width * height,)
            or m.dtype.kind not in 'iu' or np.any(m < -1)
            or type(stride) is not int or stride < 1
            or type(fine_band) is not int or fine_band < 0):
        raise ValueError('finite rectangular integer mapping and positive stride required')
    n = int(m.max(initial=-1)) + 1
    counts = np.bincount(m[m >= 0], minlength=n)
    if n == 0 or np.any(counts == 0) or len(np.unique(groups)) != len(groups) or np.any(groups < 0) or np.any(groups >= n):
        raise ValueError('contiguous variables and unique existing groups required')
    ordinary = np.ones(n, bool)
    ordinary[groups] = False
    if np.any(counts[ordinary] != 1):
        raise ValueError('only declared translation groups may share source nodes')
    group_nodes = np.isin(m, groups)
    special = ((m < 0) | group_nodes).reshape(height, width)
    fine = ((distance_transform_edt(~special) <= fine_band).ravel()
            if special.any() else np.zeros(len(m), bool))
    xs = np.unique(np.r_[np.arange(0, width, stride), width - 1])
    ys = np.unique(np.r_[np.arange(0, height, stride), height - 1])
    coarse_nodes = (ys[:, None] * width + xs).ravel()
    anchor_mask = fine | group_nodes
    anchor_mask[coarse_nodes] = True
    extra_nodes = np.array([], dtype=int) if extra_nodes is None else np.asarray(extra_nodes)
    if (extra_nodes.ndim != 1 or extra_nodes.dtype.kind not in 'iu'
            or np.any(extra_nodes < 0) or np.any(extra_nodes >= len(m))):
        raise ValueError('extra nodes must refer to original source-grid nodes')
    anchor_mask[extra_nodes] = True
    anchors = np.unique(m[anchor_mask & (m >= 0)])
    if len(anchors) == 0:
        raise ValueError('reduction has no movable anchor variables')
    columns = np.full(n, -1, dtype=int)
    columns[anchors] = np.arange(len(anchors))
    rows = [anchors]
    cols = [np.arange(len(anchors))]
    values = [np.ones(len(anchors))]
    remaining_nodes = np.flatnonzero((m >= 0) & ~group_nodes & ~anchor_mask)
    x, y = remaining_nodes % width, remaining_nodes // width
    i = np.minimum(np.searchsorted(xs, x, side='right') - 1, len(xs) - 2)
    j = np.minimum(np.searchsorted(ys, y, side='right') - 1, len(ys) - 2)
    tx = (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    for xx, yy, weight in [(xs[i], ys[j], (1-tx)*(1-ty)),
                            (xs[i+1], ys[j], tx*(1-ty)),
                            (xs[i], ys[j+1], (1-tx)*ty),
                            (xs[i+1], ys[j+1], tx*ty)]:
        variable = m[yy * width + xx]
        selected = (variable >= 0) & (weight > 0)
        column = columns[variable[selected]]
        if np.any(column < 0):
            raise AssertionError('coarse node did not reuse its anchor/group variable')
        rows.append(m[remaining_nodes[selected]]); cols.append(column); values.append(weight[selected])
    r = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, len(anchors))).tocsc()
    r.sum_duplicates(); r.eliminate_zeros()
    row_nnz = np.asarray(r.getnnz(axis=1)).ravel()
    zero_rows = np.flatnonzero(row_nnz == 0)
    if len(zero_rows):
        raise ValueError('reduction silently locks active source nodes: ' + str(zero_rows[:8].tolist()))
    # Each column has an independent identity row: no masked-basis rank guess.
    if (r[anchors] != eye(len(anchors), format='csc')).nnz:
        raise AssertionError('independent anchor identity rows lost')
    return r, {'full_variables': n, 'reduced_variables': len(anchors),
               'stride_grid_cells': stride, 'fine_band_grid_cells': fine_band,
               'anchor_variables': anchors.tolist(), 'group_variables': groups.tolist(),
               'extra_source_nodes': len(extra_nodes),
               'zero_active_rows': 0,
               'full_column_rank_identity_rows_verified': True,
               'construction_seconds': time.perf_counter() - started,
               'continuity': 'Original fine-grid C0 bilinear reconstruction; no C1 claim.',
               'fixed_and_group_mapping_unchanged': True}


def solve_reduced(hessian, rhs, reduction, lines=None):
    started = time.perf_counter()
    h, r = csc_matrix(hessian), csc_matrix(reduction)
    rhs = np.asarray(rhs, float)
    if (h.shape != (r.shape[0], r.shape[0]) or rhs.shape != (r.shape[0], 2)
            or min(r.shape) < 1 or not all(np.isfinite(v).all() for v in [h.data, r.data, rhs])):
        raise ValueError('finite matching full objective and reduction required')
    if (h - h.T).nnz and abs((h - h.T).data).max() > 1e-12 * max(1., abs(h.data).max()):
        raise ValueError('symmetric source objective required')
    reduced_h = (r.T @ h @ r).tocsc()
    reduced_b = r.T @ rhs
    reduced_h = ((reduced_h + reduced_h.T) * .5).tocsc()
    assembly_seconds = time.perf_counter() - started
    solve_started = time.perf_counter()
    lines = csc_matrix((0, 2 * h.shape[0])) if lines is None else csc_matrix(lines)
    if lines.shape[1] != 2 * h.shape[0] or not np.isfinite(lines.data).all():
        raise ValueError('finite original vector-coupled background rows required')
    if lines.shape[0]:
        reduced_lines = (lines @ block_diag([r, r], format='csc')).tocsc()
        result = solve_equality_quadratic(reduced_h, reduced_b, reduced_lines)
        if result['report']['status'] != 'EqualityQuadraticSolved':
            raise ValueError('reduced equality solve rejected')
        a = result['solution']; multipliers = result['half_objective_line_multipliers']
        factor_seconds = result['report']['factor_seconds']
    else:
        factor_started = time.perf_counter(); factor = splu(reduced_h)
        factor_seconds = time.perf_counter() - factor_started
        a = factor.solve(reduced_b); multipliers = np.empty(0)
    solve_seconds = time.perf_counter() - solve_started
    z = r @ a
    reaction = (lines.T @ multipliers).reshape(2, -1).T
    projected = r.T @ (h @ z - rhs + reaction)
    denominator = (float(np.asarray(abs(reduced_h).sum(axis=1)).max()) * abs(a).max(axis=0)
                   + abs(reduced_b).max(axis=0) + abs(r.T @ reaction).max(axis=0))
    backward = abs(projected).max(axis=0) / np.maximum(denominator, np.finfo(float).tiny)
    line_residual = lines @ z.T.ravel()
    norms = np.sqrt(np.asarray(lines.multiply(lines).sum(axis=1)).ravel())
    normalized = np.divide(line_residual, norms, out=np.zeros_like(line_residual), where=norms > 0)
    if not all(np.isfinite(v).all() for v in [a, z, projected, backward, normalized]):
        raise ValueError('nonfinite reduced result')
    if np.any(backward > 1e-10) or abs(normalized).max(initial=0) > 1e-9:
        raise ValueError('reduced stationarity or original background equality failed')
    return {'solution': z, 'coefficients': a, 'multipliers': multipliers,
            'report': {'status': 'ReducedSubspaceSolved', 'full_space_optimality_claimed': False,
                       'geometry_constraints_enforced': False, 'aesthetic_approval': False,
                       'projected_backward_error': float(backward.max()),
                       'full_line_max_residual': float(abs(line_residual).max(initial=0)),
                       'full_line_unit_max_residual': float(abs(normalized).max(initial=0)),
                       'original_line_rows_checked': lines.shape[0],
                       'assembly_seconds': assembly_seconds, 'factor_seconds': factor_seconds,
                       'solve_seconds': solve_seconds,
                       'objective_without_constant': float(np.sum(z * (h @ z)) - 2 * np.sum(rhs * z)),
                       'complete_seconds': time.perf_counter() - started}}
