"""One source-SOC correction selected from actual rejected inverse cells.

The caller owns immutable output directories and the complete image audit.
This module writes no files, changes no targets or basis, and never retries a
solve. Its local active-set solution is not a full image-quality certificate.
"""
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import block_diag, csc_matrix, load_npz

from reduced_field import ROOT
from reduced_constrained import _solve_round, _background_evidence, ROUND_TIME_LIMIT
from line_equality_solve import independent_rows
from input_contract import validate_input, _matrix_same
from adapt import read, sha

sys.path.insert(0, str(ROOT / 'experiments/face-source-forward-prototype-20261021'))
from forward import cell_jacobians


def focused_source_corners(inverse, source, grid, previous_active):
    """Cover complete inverse-cell enclosures, low-source cells and one halo.

    Each coordinate of a bilinear destination cell lies within its four corner
    extrema. Its axis-aligned source enclosure therefore also covers source
    cells crossed between those corners. Merely selecting four endpoint cells
    can miss interior cells. Every selected source cell retains all 4 corners.
    """
    from constrained_solve import effective_bounds
    h, w = grid['height'], grid['width']
    field, source = np.asarray(inverse), np.asarray(source, float)
    if (grid['spacing'] != [2., 2.] or min(h, w) < 2
            or source.shape != (h, w, 2) or field.shape != (2*h-1, 2*w-1, 2)
            or not np.isfinite(field).all() or not np.isfinite(source).all()):
        raise ValueError('finite matching original 2px source and inverse fields required')
    cell_count = (h-1)*(w-1)
    if any(type(i) not in (int, np.int32, np.int64) or i < 0 or i >= 4*cell_count
           for i in previous_active):
        raise ValueError('previous active source corner outside original grid')
    j = cell_jacobians(field.astype(float))
    eigen = np.linalg.eigvalsh((j+j.swapaxes(-1, -2))*.5)
    skew = abs(j[..., 1, 0]-j[..., 0, 1])*.5
    bad = np.any((eigen[..., 0] < .52) | (eigen[..., 1] > 1.8)
                 | (skew > .24) | (np.linalg.det(j) <= 0), axis=0)
    yy, xx = np.where(bad)
    footprint = np.zeros((h-1, w-1), bool)
    if len(xx):
        vertices = np.stack([(np.column_stack([xx+dx, yy+dy])
                              + field[yy+dy, xx+dx])/2.
                             for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]], axis=1)
        if np.any(vertices < 0) or np.any(vertices > [w-1, h-1]):
            raise ValueError('failed inverse footprint leaves original source domain')
        for vertices_of_cell in vertices:
            lo = np.minimum(np.floor(vertices_of_cell.min(axis=0)).astype(int), [w-2, h-2])
            hi = np.minimum(np.floor(vertices_of_cell.max(axis=0)).astype(int), [w-2, h-2])
            footprint[lo[1]:hi[1]+1, lo[0]:hi[0]+1] = True
    f = cell_jacobians(source/2.)
    low = np.linalg.eigvalsh((f+f.swapaxes(-1, -2))*.5)[..., 0]
    _, beta, _ = effective_bounds()
    low_cells = np.any(1/beta-low > 1e-8, axis=0)
    local = footprint | low_cells
    halo = binary_dilation(local, structure=np.ones((3, 3), bool))
    old = np.zeros_like(local)
    if previous_active:
        old.ravel()[np.asarray(previous_active, int) % cell_count] = True
    selected = halo | old
    cells = np.flatnonzero(selected.ravel())
    active = np.concatenate([cells+k*cell_count for k in range(4)]).tolist()
    return active, {
        'failed_inverse_cells': len(xx), 'inverse_footprint_cells': int(footprint.sum()),
        'inverse_footprint_cell_ids': np.flatnonzero(footprint.ravel()).tolist(),
        'low_source_cells': int(low_cells.sum()), 'local_union_cells': int(local.sum()),
        'source_cell_halo': 1, 'local_with_halo_cells': int(halo.sum()),
        'previous_active_cells': int(old.sum()), 'selected_source_cells': len(cells),
        'new_source_cells': int((selected & ~old).sum()),
        'selected_source_corners': len(active), 'all_active_corner_ids': active,
        'complete_bilinear_source_enclosure': True, 'all_four_corners_per_cell': True,
        'source_low_tolerance': 1e-8, 'effective_source_bounds': list(effective_bounds()),
        'final_inverse_thresholds': {'symmetric_min': .52, 'symmetric_max': 1.8,
                                     'skew_max': .24, 'positive_determinant': True}}


def _bind_previous(h, rhs, r, grid, mapping, lines, previous_case):
    case = Path(previous_case).resolve()
    protocol = read(case/'protocol.json')
    hashes = dict(protocol['inputs_sha256'])
    if not hashes:
        raise ValueError('previous protocol has no frozen input hashes')
    for name, digest in hashes.items():
        if not Path(name).is_file() or sha(Path(name)) != digest:
            raise ValueError('previous input hash differs: '+name)
    inp = Path(protocol['input']).resolve()
    validation = validate_input(inp)
    hashes.update(validation['verified_input_hashes'])
    contract = read(inp/'contract.json')
    if grid != contract['grid'] or contract['crop'][2:] != [2*grid['width']-1, 2*grid['height']-1]:
        raise ValueError('previous original grid or crop differs')
    stored_mapping = np.load(inp/'data.npz', allow_pickle=False)['node_to_variable']
    if not np.array_equal(mapping, stored_mapping):
        raise ValueError('previous source roles/mapping differ')
    objective = inp/('quadratic' if (inp/'quadratic').exists() else 'proposal')
    _matrix_same(h, load_npz(objective/'hessian.npz'), 'original Hessian')
    if not np.array_equal(rhs, np.load(objective/'rhs.npy', allow_pickle=False)):
        raise ValueError('previous original RHS differs')
    _matrix_same(r, load_npz(case/'reduction.npz'), 'previous reduction')
    lines_path = inp/'hard_line_equalities.npz'
    stored_lines = load_npz(lines_path) if lines_path.exists() else csc_matrix((0, 2*h.shape[0]))
    _matrix_same(lines, stored_lines, 'complete original background rows')
    z = np.load(case/'solution.npy', allow_pickle=False)
    a = np.load(case/'coefficients.npy', allow_pickle=False)
    if (a.shape != (r.shape[1], 2) or z.shape != rhs.shape or not np.isfinite(a).all()
            or not np.array_equal(r @ a, z)):
        raise ValueError('previous reduced coefficients do not reproduce its source solution')
    source = np.load(case/'source-forward.npy', allow_pickle=False)
    expected = np.zeros((len(mapping), 2)); valid = mapping >= 0
    expected[valid] = z[mapping[valid]]
    if not np.array_equal(source, expected.reshape(grid['height'], grid['width'], 2)):
        raise ValueError('previous source field does not match the frozen source roles')
    inverse = np.load(case/'inverse-field.npy', allow_pickle=False)
    if inverse.dtype != np.dtype('<f4'):
        raise ValueError('previous inverse must be the stored actual float32 field')
    evidence = read(case/'solver-evidence.json')
    active = sorted({corner for row in evidence.get('rounds', [])
                     for corner in row.get('all_active_corner_ids', [])})
    for name in ['protocol.json', 'reduction.npz', 'solution.npy', 'coefficients.npy',
                 'source-forward.npy', 'inverse-field.npy', 'solver-evidence.json']:
        path = case/name; hashes[str(path)] = sha(path)
    return {'solution': z, 'coefficients': a, 'source': source, 'inverse': inverse,
            'active': active, 'hashes': hashes, 'input': str(inp)}


def solve_focused_reduced(h, rhs, r, grid, mapping, lines, previous_case):
    """One 12s-limited solve with unchanged R/H/rhs and source material roles.

    The solver limit excludes assembly/initialization. Returned source proxy
    status is separate from the caller's full f32 inverse and RGB evaluation.
    No image is rendered and no output file is created or overwritten here.
    """
    from constrained_solve import inspect_corners
    started = time.perf_counter()
    h, r = csc_matrix(h), csc_matrix(r)
    rhs, mapping = np.asarray(rhs, float), np.asarray(mapping)
    lines = csc_matrix((0, 2*h.shape[0])) if lines is None else csc_matrix(lines)
    if (ROUND_TIME_LIMIT != 12. or h.shape != (r.shape[0], r.shape[0])
            or rhs.shape != (r.shape[0], 2) or lines.shape[1] != 2*h.shape[0]
            or not all(np.isfinite(v).all() for v in [h.data, rhs, r.data, lines.data])):
        raise ValueError('finite matching source objective and unchanged 12s limit required')
    previous = _bind_previous(h, rhs, r, grid, mapping, lines, previous_case)
    active, selection = focused_source_corners(previous['inverse'], previous['source'], grid, previous['active'])
    if selection['new_source_cells'] == 0:
        raise ValueError('no new source constraints; repeating the same correction cannot add coverage')
    reduced_h = (r.T @ h @ r).tocsc(); reduced_h = ((reduced_h+reduced_h.T)*.5).tocsc()
    reduced_rhs = r.T @ rhs
    reduced_lines = (lines @ block_diag([r, r], format='csc')).tocsc()
    independent, _, _, row_reduction = independent_rows(reduced_lines)
    coefficients, solve = _solve_round(reduced_h, reduced_rhs, r, grid, mapping, active, independent)
    solve = dict(solve)
    solve.update(round=1, all_active_corner_ids=active, selected_source_cells=selection['selected_source_cells'],
                 no_repeated_solve=True, retained_same_cell_four_corners=True)
    available = coefficients is not None
    if not available:
        coefficients = previous['coefficients']
    z = r @ coefficients
    _, scan = inspect_corners(z, grid, mapping)
    if not scan['boundary_exact_zero']:
        raise ValueError('focused correction changed the original fixed boundary')
    background = _background_evidence(lines, z)
    proxy_pass = bool(scan['corners_violating_1e_8'] == 0)
    accepted = bool(available and solve['status'] == 'Solved' and solve.get('active_cones_pass', False)
                    and proxy_pass and background['passed'])
    for name, digest in previous['hashes'].items():
        if sha(Path(name)) != digest:
            raise ValueError('frozen inputs changed during focused correction: '+name)
    return {'solution': z, 'coefficients': coefficients,
            'report': {'status': solve['status'], 'selected_stage': 'focused_single_round' if available
                       else 'previous_case_no_primal_candidate',
                       'optimization_scope': 'unchanged source quadratic in identical R with one focused SOC active subset',
                       'previous_case': str(Path(previous_case).resolve()), 'input': previous['input'],
                       'verified_input_hashes': previous['hashes'], 'selection': selection,
                       'full_space_optimality_claimed': False, 'subspace_optimization_converged': accepted,
                       'source_direct_proxy_pass': proxy_pass, 'source_proxy_pass': proxy_pass,
                       'source_convex_gate_pass': accepted, 'source_cone_certificate_attempted': True,
                       'diagnostic_only': not accepted, 'aesthetic_approval': False,
                       'final_actual_f32_geometry_and_RGB_pending': True,
                       'maximum_rounds': 1, 'round_solver_time_limit_seconds': 12.,
                       'solver_limit_excludes_operator_assembly_and_initialization': True,
                       'all_source_corner_scan': scan, 'all_original_background': background,
                       'background_row_reduction': row_reduction, 'rounds': [solve],
                       'no_strength_or_target_scaling': True, 'no_threshold_changes': True,
                       'unchanged_reduction_objective_mapping_and_lines': True,
                       'complete_seconds': time.perf_counter()-started}}
