"""One source 2px bilinear candidate input; preserve every source contract row.

This explicitly changes the basis and objective. Historical inverse-zero
patches remain a comparison ledger, not newly inferred semantic locks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, diags, eye, kron, save_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/face-user-source-roles-20261021'))
from context_refinement import expanded_mask, guaranteed_zero_patches


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def square_nodes(pixels, grid, identity_extension=False):
    """Every vertex of every bilinear cell meeting a full closed ±1px square."""
    p = np.asarray(pixels, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('invalid source material points')
    lo = np.floor((p-1-grid['origin'])/grid['spacing']).astype(int)
    hi = np.ceil((p+1-grid['origin'])/grid['spacing']).astype(int)
    if (np.any(lo < 0) or np.any(hi >= [grid['width'], grid['height']])) and not identity_extension:
        raise ValueError('source material fringe outside field rectangle')
    # Only fixed material may use the explicit identity extension of this
    # field; moving material is never truncated or cropped to fit the grid.
    lo = np.maximum(lo, 0); hi = np.minimum(hi, [grid['width']-1, grid['height']-1])
    result = set()
    for a, b in zip(lo, hi):
        result.update(y*grid['width']+x for y in range(a[1], b[1]+1) for x in range(a[0], b[0]+1))
    return np.array(sorted(result), dtype=np.int64)


def point_operators(points, grid, mapping, nvars):
    p = np.asarray(points, float); q = (p-grid['origin'])/grid['spacing']
    w, h = grid['width'], grid['height']
    if np.any(q < 0) or np.any(q > [w-1, h-1]):
        raise ValueError('source query outside new field; no clipping')
    base = np.minimum(np.floor(q).astype(int), [w-2, h-2]); uv = q-base
    x, y = base.T; u, v = uv.T; hx, hy = grid['spacing']
    ids = np.stack([y*w+x, y*w+x+1, (y+1)*w+x, (y+1)*w+x+1], axis=1)
    basis = np.stack([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v], axis=1)
    dx = np.stack([-(1-v), 1-v, -v, v], axis=1)/hx
    dy = np.stack([-(1-u), -u, 1-u, u], axis=1)/hy
    columns = mapping[ids].ravel(); rows = np.repeat(np.arange(len(p)), 4)
    valid = columns >= 0
    operators = []
    for data in [basis, dx, dy]:
        op = coo_matrix((data.ravel()[valid], (rows[valid], columns[valid])), shape=(len(p), nvars)).tocsr()
        op.sum_duplicates(); op.eliminate_zeros(); operators.append(op)
    return operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); folder = args.contract.resolve(); out = args.output.resolve()
    c = read(folder/'source-contract.json'); report = read(folder/'report.json')
    for path, digest in report['inputs_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'input digest differs: {path}')
    evidence = np.load(folder/'source-evidence.npz')
    old = read(Path(c['old_controls'])); base = Path(c['old_controls']).parent
    p = read(base/'provenance.json'); left, top, cw, ch = p['crop']
    if (cw-1) % 2 or (ch-1) % 2:
        raise ValueError('exact old crop not divisible by 2px; no resizing')
    grid = {'origin': [left, top], 'spacing': [2., 2.], 'width': (cw-1)//2+1, 'height': (ch-1)//2+1}
    h, w = grid['height'], grid['width']; n = h*w
    source_fixed_core = np.logical_or.reduce([evidence[k] for k in ['classified_fixed', 'nose_core', 'lips_core']])
    # The old final RGB oracle protects complete polygons, not only eroded
    # solver cores. Match that actual photograph contract in global pixels.
    graph = read(Path(p['source_trace']).parent/'graph.json')
    image_height, image_width = source_fixed_core.shape
    fixed_masks = {'classified_fixed': evidence['classified_fixed']}
    for name in ['nose', 'lips']:
        image_mask = Image.new('L', (image_width, image_height))
        ImageDraw.Draw(image_mask).polygon([(v['x']*image_width, v['y']*image_height)
            for v in graph['operations'][0]['params']['semanticMasks'][name]['points']], fill=1)
        fixed_masks[name+'_full_polygon'] = np.asarray(image_mask) > 0
    source_fixed = np.logical_or.reduce(list(fixed_masks.values())) | source_fixed_core
    yy, xx = np.where(source_fixed); fixed_points = np.stack([xx, yy], axis=1)
    fixed_ids = square_nodes(fixed_points, grid, identity_extension=True)
    locked = np.zeros(n, bool); locked[fixed_ids] = True
    boundary = np.zeros((h, w), bool); boundary[:2] = boundary[-2:] = True; boundary[:, :2] = boundary[:, -2:] = True
    locked |= boundary.ravel()
    # Preserve only the old distant-context zero region here. The former
    # nose/eye B3 coefficient margins are not turned back into semantic locks.
    og = old['grid']; oh, ow = og['height'], og['width']
    old_context = np.zeros((oh, ow), bool)
    old_context.ravel()[p['context_fixed_nodes']] = True
    old_context[[0, -1]] = True; old_context[:, [0, -1]] = True
    context_patches = guaranteed_zero_patches(expanded_mask(old_context))
    cy, cx = np.where(context_patches)
    context_min = np.stack([cx, cy], axis=1)*og['spacing']+og['origin']
    context_max = context_min+og['spacing']
    context_nodes = np.zeros((h, w), bool)
    for lo, hi in zip(context_min, context_max):
        a = np.floor((lo-grid['origin'])/grid['spacing']).astype(int)
        b = np.ceil((hi-grid['origin'])/grid['spacing']).astype(int)
        a = np.maximum(a, 0); b = np.minimum(b, [w-1, h-1])
        context_nodes[a[1]:b[1]+1, a[0]:b[0]+1] = True
    locked |= context_nodes.ravel()
    group_nodes = [square_nodes(g['source_material'], grid) for g in c['groups']]
    group_collisions = {f'{i}-{j}': len(np.intersect1d(group_nodes[i], group_nodes[j])) for i in range(4) for j in range(i+1, 4)}
    fixed_collisions = [int(locked[g].sum()) for g in group_nodes]
    if any(fixed_collisions) or any(group_collisions.values()):
        raise ValueError(f'source support conflict: fixed={fixed_collisions}, groups={group_collisions}')
    is_group = np.zeros(n, bool)
    for ids in group_nodes:
        is_group[ids] = True
    free = np.flatnonzero(~locked & ~is_group); nvars = len(free)+4
    mapping = np.full(n, -1, dtype=np.int64); mapping[free] = np.arange(len(free))
    for gi, ids in enumerate(group_nodes):
        mapping[ids] = len(free)+gi
    actual = np.flatnonzero(mapping >= 0)
    prolong = coo_matrix((np.ones(len(actual)), (actual, mapping[actual])), shape=(n, nvars)).tocsr()
    sources = np.array([t['source'] for t in c['targets']]); destinations = np.array([t['destination'] for t in c['targets']])
    deltas = np.array([t['forward_delta'] for t in c['targets']]); weights = np.array([t['weight'] for t in c['targets']])
    if len(sources) != 6353 or [t['original_index'] for t in c['targets']] != list(range(6353)):
        raise ValueError('original target rows missing/reordered')
    np.testing.assert_allclose(sources+deltas, destinations, rtol=0, atol=1e-10)
    target, _, _ = point_operators(sources, grid, mapping, nvars)
    ear_points = np.concatenate([np.array(g['source_samples']) for g in p['ear_orientation']['groups_evidence']])
    if len(ear_points) != 5096:
        raise ValueError('complete source-ear material changed')
    _, ear_dx, ear_dy = point_operators(ear_points, grid, mapping, nvars)
    hx, hy = grid['spacing']
    lx = diags([-np.ones(w-1), 2*np.ones(w), -np.ones(w-1)], [-1, 0, 1])/hx**2
    ly = diags([-np.ones(h-1), 2*np.ones(h), -np.ones(h-1)], [-1, 0, 1])/hy**2
    laplacian = ((kron(eye(h), lx)+kron(ly, eye(w)))@prolong).tocsr()
    one_x = diags([-np.ones(w-1), np.ones(w-1)], [0, 1], shape=(w-1, w))/hx
    one_y = diags([-np.ones(h-1), np.ones(h-1)], [0, 1], shape=(h-1, h))/hy
    dx = (kron(eye(h), one_x)@prolong).tocsr()
    dy = (kron(one_y, eye(w))@prolong).tocsr()
    priors = np.array([-np.array(g['delta']) for g in c['old_inverse_translation_priors']])
    prior_weights = np.array([g['target_weight'] for g in c['old_inverse_translation_priors']])
    # These input keys do not exist in this base. A future base with background
    # line constraints must not silently lose them through this mechanism probe.
    lines = old.get('background_lines', [])
    if lines:
        raise ValueError('explicit background lines require implementation before solving')
    out.mkdir(parents=True, exist_ok=False)
    for name, op in [('prolongation', prolong), ('target', target), ('laplacian', laplacian),
                     ('ear_dx', ear_dx), ('ear_dy', ear_dy), ('Dx', dx), ('Dy', dy)]:
        save_npz(out/(name+'.npz'), op)
    group_prior = coo_matrix((np.ones(4), (np.arange(4), np.arange(len(free), nvars))), shape=(4, nvars)).tocsr()
    save_npz(out/'group_prior.npz', group_prior)
    for name, values in [('target_deltas', deltas), ('target_weights', weights),
                         ('group_priors', priors), ('group_prior_weights', prior_weights)]:
        np.save(out/(name+'.npy'), values)
    assembly = {'nvars': nvars, 'blocks': [
        {'name': 'all_6353_source_targets', 'matrix_file': 'target.npz', 'rhs_file': 'target_deltas.npy', 'weights_file': 'target_weights.npy'},
        {'name': 'source_physical_laplacian', 'matrix_file': 'laplacian.npz', 'weights': old['smoothness']*hx*hy},
        {'name': 'source_displacement', 'matrix_file': 'prolongation.npz', 'weights': old['regularization']*hx*hy},
        {'name': 'all_5096_source_ears_dx', 'matrix_file': 'ear_dx.npz', 'weights': 10.},
        {'name': 'all_5096_source_ears_dy', 'matrix_file': 'ear_dy.npz', 'weights': 10.},
        {'name': 'original_rigid_priors', 'matrix_file': 'group_prior.npz', 'rhs_file': 'group_priors.npy', 'weights_file': 'group_prior_weights.npy'}],
        'fixed_material_outside_field_fringe': 'The field is exactly identity outside its crop; boundary two-node collar is zero. Only fixed fringe can extend there; moving fringe outside is rejected.'}
    save(out/'assembly.json', assembly)
    np.savez_compressed(out/'data.npz', target_sources=sources, target_destinations=destinations,
        target_deltas=deltas, target_weights=weights, node_to_variable=mapping,
        source_fixed_pixels=fixed_points, source_ear_points=ear_points,
        source_fixed_core_pixels=np.column_stack(np.where(source_fixed_core)[::-1]),
        group_variable_ids=np.arange(len(free), nvars), group_priors=priors, group_prior_weights=prior_weights,
        old_identity_patch_min=evidence['fixed_identity_patch_min'], old_identity_patch_max=evidence['fixed_identity_patch_max'],
        old_context_patch_min=context_min, old_context_patch_max=context_max,
        preserved_context_nodes=np.flatnonzero(context_nodes),
        **{'group_nodes_'+str(i): ids for i, ids in enumerate(group_nodes)})
    contract = {'source_contract': str(folder/'source-contract.json'), 'source_contract_sha256': sha(folder/'source-contract.json'),
        'source_evidence_sha256': sha(folder/'source-evidence.npz'), 'baseline': str(base), 'crop': p['crop'], 'grid': grid,
        'source_photo': c['source_photo'], 'source_photo_sha256': c['source_photo_sha256'],
        'node_count': n, 'scalar_variables': nvars, 'unknown_components': 2,
        'target_count': len(sources), 'all_original_target_weights_and_strength_retained': True,
        'source_ear_rows': len(ear_points), 'fixed_source_rows': len(fixed_points),
        'fixed_source_oracle_masks': {name: int(mask.sum()) for name, mask in fixed_masks.items()},
        'original_fixed_core_rows': int(source_fixed_core.sum()),
        'fixed_square_support_nodes': len(fixed_ids), 'boundary_nodes': int(boundary.sum()),
        'preserved_context_source_patches': len(context_min), 'preserved_context_nodes': int(context_nodes.sum()),
        'material_square_support_nodes': [len(z) for z in group_nodes], 'fixed_collisions': fixed_collisions,
        'group_collisions': group_collisions, 'group_variable_ids': list(range(len(free), nvars)),
        'background_lines_in_original_controls': len(lines),
        'energy': {'targets': 'sum original_weight * ||G(source)-original_destination||^2',
            'source_small_strain_ears_modulus': 10., 'source_ear_term': '10 sum original 5096 source pixels ||Du||_F^2',
            'smoothness': old['smoothness'], 'smoothness_term': 'smoothness * cell_area * ||physical_laplacian u||^2',
            'regularization': old['regularization'], 'regularization_term': 'regularization * cell_area * ||u||^2',
            'cell_area': hx*hy, 'rigid_priors': 'original weights and sign-inverted source translation'},
        'objective_change': 'forward residual, physical 2px bilinear regularization, and source small-strain ear energy replace the old inverse B3/finite ear objective. No same-energy claim.',
        'historical_identity_regions': 'original all-fixed 12327 patches retained as numerical ledger. Distant context-only continuous zero patches remain hard fixed; near-feature B3 margins may differ and must be reported.',
        'source_fixed_semantics': 'all original classified texels, complete global nose/lip polygons from the actual old RGB oracle, and eroded old cores plus closed ±1 square fringe fixed; 2 node boundary collar and old distant context zero domain fixed',
        'inverse_geometry_gate': c['inverse_jacobian_contract'],
        'maximum_proposals': 1, 'no_strength_reduction_or_clipping': True,
        'source_geometry_requirement': 'positive strong-monotone lower bound and identity boundary; final inverse all-cell .52/1.8/.24 gate separately',
        'not_aesthetic_approval': True}
    save(out/'contract.json', contract)
    print(json.dumps({k: contract[k] for k in ['grid', 'node_count', 'scalar_variables', 'target_count',
        'source_ear_rows', 'fixed_source_rows', 'material_square_support_nodes', 'fixed_collisions', 'group_collisions']}, indent=2))


if __name__ == '__main__':
    main()
