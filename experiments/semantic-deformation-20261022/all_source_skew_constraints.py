"""Preassemble the original inverse-skew bounds at every source-grid corner.

For F=I+across outer grad(u), in an orthonormal across/down frame, set
p=across dot grad(u), b=down dot grad(u), d=1+p. When d>0, inverse skew
is abs(b)/(2*d). Thus the unchanged bound .24 is exactly the pair
(-.48*p +/- b) z <= .48. Their cellwise affine slacks attain their minima
at corners, so covering all four one-sided corners also covers cell interiors.

These halfspaces alone allow d=b=0. They neither certify strict invertibility
nor replace the existing eigenvalue cones, source gates or final inverse audit.
"""
import time

import clarabel
import numpy as np
from scipy.sparse import vstack

from joint_jaw_boundary_field import corner_gradient_rows


def _validate(problem, across, down):
    grid = problem['grid']; mapping = np.asarray(problem['mapping'])
    counts = [grid['width'], grid['height']]
    origin = np.asarray(grid['origin'], float); spacing = np.asarray(grid['spacing'], float)
    shape = problem['hessian'].shape
    if (any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 2
            for n in counts) or origin.shape != (2,) or spacing.shape != (2,)
            or not np.isfinite(origin).all() or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise ValueError('finite source grid with positive physical spacing required')
    if (len(shape) != 2 or shape[0] != shape[1] or mapping.shape != (counts[0]*counts[1],)
            or mapping.dtype.kind not in 'iu' or np.any(mapping < -1)):
        raise ValueError('complete original fixed-zero/free scalar mapping required')
    active = mapping[mapping >= 0]
    if not np.array_equal(np.sort(active), np.arange(shape[0])):
        raise ValueError('independent contiguous original scalar variables required')
    axes = np.asarray([across, down], float)
    if (axes.shape != (2, 2) or not np.isfinite(axes).all()
            or not np.allclose(axes@axes.T, np.eye(2), atol=1e-10, rtol=0.)):
        raise ValueError('original orthonormal across/down frame required')
    return grid, mapping, shape[0], axes[0], axes[1]


def all_source_skew_constraints(problem, across, down):
    """Return scalar-column A,b,cones plus exact coverage and row provenance.

    The caller pads existing auxiliary columns with zeros. No existing QP row,
    objective, variable, or physical bound is changed. Only identically zero
    rows (rhs=.48) and exact coefficient/rhs duplicates are omitted. Provenance
    maps each original halfspace to its retained row or -1 for a tautology;
    its axes are [minus/plus b, TL/TR/BL/BR, cell y, cell x].
    """
    started = time.perf_counter()
    grid, mapping, size, across, down = _validate(problem, across, down)
    width, height = grid['width'], grid['height']
    corner_shape = (4, height-1, width-1)
    indices = np.indices(corner_shape).reshape(3, -1).T
    p, b = corner_gradient_rows(problem, indices, across, down)
    for matrix in (p, b):
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    active_corners = (np.diff(p.indptr) > 0) | (np.diff(b.indptr) > 0)
    raw = vstack([-.48*p-b, -.48*p+b], format='csr')
    raw.sum_duplicates(); raw.eliminate_zeros(); raw.sort_indices()
    if not np.isfinite(raw.data).all():
        raise ValueError('finite physical source gradient coefficients required')
    seen = {}; retained = []; zero_rows = 0; duplicates = 0
    provenance = np.full(raw.shape[0], -1, dtype=np.int64)
    # All rows have the same literal RHS; include its exact representation so
    # this key documents coefficient/RHS identity, not geometric proximity.
    rhs_key = np.float64(.48).tobytes()
    for row in range(raw.shape[0]):
        lo, hi = raw.indptr[row:row+2]
        if lo == hi:
            zero_rows += 1
            continue
        key = (raw.indices[lo:hi].tobytes(), raw.data[lo:hi].tobytes(), rhs_key)
        existing = seen.get(key)
        if existing is None:
            existing = len(retained); seen[key] = existing; retained.append(row)
        else:
            duplicates += 1
        provenance[row] = existing
    matrix = raw[np.asarray(retained, dtype=np.int64)].tocsc()
    rhs = np.full(matrix.shape[0], .48)
    free = (mapping >= 0).reshape(height, width)
    active_cells = free[:-1, :-1] | free[:-1, 1:] | free[1:, :-1] | free[1:, 1:]
    report = {'schema': 'all-source-corner-original-inverse-skew-v1',
        'scalar_variables': size, 'source_cells': int(active_cells.size),
        'active_source_cells': int(active_cells.sum()),
        'source_corners': int(len(indices)), 'active_source_corners': int(active_corners.sum()),
        'identically_zero_source_corners': int((~active_corners).sum()),
        'covered_source_corners': int(len(indices)),
        'covered_active_source_corners': int(active_corners.sum()),
        'halfspaces_before_reduction': int(raw.shape[0]),
        'tautological_zero_halfspaces_removed': zero_rows,
        'exact_duplicate_halfspaces_removed': duplicates,
        'retained_halfspaces': int(matrix.shape[0]), 'retained_nonzeros': int(matrix.nnz),
        'inverse_skew_bound': .24, 'halfspace_rhs': .48,
        'deduplication': 'exact canonical sparse coefficient bytes and RHS bytes; no rounding or tolerance',
        'coverage': 'both original halfspaces at every one-sided corner; affine slack extends to the whole cell',
        'active_corner_definition': 'at least one nonzero original p or b gradient row after fixed-zero elimination',
        'strict_invertibility_certified': False,
        'required_external_gates': 'retain original eigenvalue SOCs, all source/curve constraints and direct inverse audits',
        'changes_objective': False, 'changes_old_constraints': False,
        'assembly_seconds': time.perf_counter()-started}
    return {'A': matrix, 'b': rhs,
        'cones': [clarabel.NonnegativeConeT(matrix.shape[0])] if matrix.shape[0] else [],
        'original_halfspace_to_row': provenance.reshape((2, *corner_shape)),
        'retained_original_halfspace_rows': np.asarray(retained, dtype=np.int64), 'report': report}
