"""Reject unsafe source candidates before inverse sampling or photograph output."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/face-source-forward-prototype-20261021'))
from forward import source_geometry, cell_jacobians, inverse_points, sample
sys.path.insert(0, str(ROOT/'experiments/face-monotone-socp-20261020'))
from audit_pixel_field import inspect
from compare_controls import load, forward
sys.path.insert(0, str(ROOT/'experiments/face-source-relative-width-20261020'))
from review_dark import material_stats
sys.path.insert(0, str(ROOT/'experiments/face-bilinear-footprint-audit-20261021'))
from audit import audit
from adapt import read, save, sha


def summarize(actual, requested, indices, source=None):
    indices = np.asarray(indices, int); error = np.linalg.norm(actual[indices]-requested[indices], axis=1)
    report = {'indices': indices.tolist(), 'actual': actual[indices].tolist(), 'error_px': error.tolist(),
        'max_error_px': float(error.max(initial=0.)), 'rms_error_px': float(np.sqrt(np.mean(error**2))) if len(indices) else 0.}
    if source is not None:
        want = requested[indices]-source[indices]; motion = actual[indices]-source[indices]
        length = np.linalg.norm(want, axis=1); active = length > .5
        gain = np.sum(want*motion, axis=1)/np.maximum(length**2, 1e-20)
        report.update(requested=requested[indices].tolist(), source=source[indices].tolist(),
            requested_motion_px=length.tolist(), actual_motion_px=np.linalg.norm(motion, axis=1).tolist(),
            projection_gain=gain.tolist(), active_cutoff_px=.5,
            worst_active_projection_gain=float(gain[active].min()) if active.any() else None,
            mean_active_projection_gain=float(gain[active].mean()) if active.any() else None)
    return report


def read_source_gate(case, *, expected_input=None):
    """Never confuse a final-image decision with a source-solver decision."""
    named, legacy = case/'source-solver-decision.json', case/'decision.json'
    direct = case/'direct-solver-decision.json'
    if direct.exists():
        if named.exists() or legacy.exists():
            raise ValueError('ambiguous direct and constrained source decisions')
        gate, evidence = read(direct), read(case/'solver-evidence.json')
        scan = gate['all_source_corner_scan']
        if (gate.get('solver_status') != 'DirectSolved' or gate.get('linear_system_solved') is not True
                or gate.get('source_strong_monotone_gate_pass') is not True
                or gate.get('inverse_evaluation_allowed') is not True
                or scan['boundary_exact_zero'] is not True
                or not np.isfinite(scan['forward_symmetric_min']) or scan['forward_symmetric_min'] <= 0):
            raise ValueError('direct source field is not eligible for strict inverse evaluation')
        if evidence.get('status') != 'DirectSolved' or evidence.get('linear_system_solved') is not True:
            raise ValueError('direct gate and linear solver evidence disagree')
        # This branch solves a different, unconstrained quadratic proposal.
        # Strong monotonicity permits inversion; it never substitutes for
        # the original final-pixel geometry and material gates below.
        folder = Path(evidence['assembly_manifest']).resolve().parent
        if expected_input is not None and folder != Path(expected_input).resolve():
            raise ValueError('direct linear evidence belongs to a different input')
        for path, key in [(case/'solution.npy', 'source_solution_sha256'),
                (folder/'proposal/hessian.npz', 'hessian_sha256'),
                (folder/'proposal/rhs.npy', 'rhs_sha256')]:
            if sha(path) != evidence[key]:
                raise ValueError('direct linear evidence changed: '+str(path))
        from scipy.sparse import load_npz
        matrix = load_npz(folder/'proposal/hessian.npz')
        rhs, solution = np.load(folder/'proposal/rhs.npy'), np.load(case/'solution.npy')
        n = matrix.shape[0]
        if (n < 1 or matrix.shape != (n, n) or rhs.shape != (n, 2) or solution.shape != (n, 2)
                or not all(np.isfinite(a).all() for a in [matrix.data, rhs, solution])):
            raise ValueError('direct linear evidence must contain a finite two-component reduced system')
        residual = matrix@solution-rhs
        denom = float(np.asarray(abs(matrix).sum(axis=1)).max())*np.max(abs(solution), axis=0)+np.max(abs(rhs), axis=0)
        backward = np.max(abs(residual), axis=0)/np.maximum(denom, np.finfo(float).tiny)
        if not np.isfinite(backward).all() or np.any(backward > 1e-10):
            raise ValueError('direct source quadratic failed independent backward-error check')
        return gate, direct
    selected = named if named.exists() else (legacy if legacy.exists() else None)
    if selected is None:
        return None, None
    gate = read(selected)
    if not isinstance(gate.get('source_convex_gate_pass'), bool):
        raise ValueError(f'{selected.name} is not a source-solver gate; image evaluation cannot substitute')
    if any(key in gate for key in ['inverse_geometry', 'beauty_accepted', 'source_geometry']):
        raise ValueError(f'{selected.name} contains final-evaluation fields; keep source and image decisions separate')
    if not gate['source_convex_gate_pass']:
        raise ValueError('root source candidate was rejected; do not evaluate intermediate/rejected output')
    return gate, selected


def source_solver_evidence(case, gate_path):
    """A direct linear solution never supplies a constrained-solver claim."""
    direct = gate_path is not None and gate_path.name == 'direct-solver-decision.json'
    solve_paths = sorted(case.glob('round-*/solve.json'), key=lambda path: int(path.parent.name.split('-')[-1]))
    if direct:
        final_solver = read(case/'solver-evidence.json')
    else:
        final_solver = read(solve_paths[-1]) if solve_paths else (read(case/'solver-evidence.json') if (case/'solver-evidence.json').exists() else {})
    status = final_solver.get('status', 'unverified_source_candidate')
    constrained_solved = not direct and status == 'Solved'
    direct_solved = direct and status == 'DirectSolved' and final_solver.get('linear_system_solved') is True
    return {'solver_status': status,
        'solver_diagnostic_only': final_solver.get('diagnostic_only', True),
        'optimizer_convergence_accepted': constrained_solved or direct_solved,
        'constrained_optimization_convergence_accepted': constrained_solved,
        'direct_linear_system_convergence_accepted': direct_solved,
        'source_optimization_problem': 'unconstrained source quadratic' if direct else 'source quadratic with conservative inverse constraints',
        'source_parent': read(case/'source-parent.json') if (case/'source-parent.json').exists() else None,
        'source_solver_gate': {'path': str(gate_path), 'sha256': sha(gate_path)} if gate_path else None}


def target_ledgers(contract, baseline, source, requested):
    """Keep current, A1, and earliest requests in separate complete ledgers."""
    count = len(source)
    a1 = read(baseline/'contract.json')
    old = read(baseline/'original-controls.json')
    old_targets = np.array([t['position'] for t in old['targets']])
    a1_targets = np.array([t['position'] for t in read(baseline/'controls.json')['targets']])
    if requested.shape != (count, 2) or old_targets.shape != (count, 2) or a1_targets.shape != (count, 2):
        raise ValueError('current/A1/legacy target ledgers must retain every source row')
    changed = contract.get('changed_target_indices', a1['changed_target_indices'])
    if (not isinstance(changed, list) or any(type(i) is not int or i < 0 or i >= count for i in changed)
            or len(set(changed)) != len(changed)):
        raise ValueError('changed_target_indices must be unique valid original integer row indices')
    changed_set = set(changed)
    unchanged = [i for i in range(count) if i not in changed_set]
    specs = [(f'all_{count}_new_contract', requested, range(count)),
        (f'changed_{len(changed)}_new_contract', requested, changed),
        (f'original_{count}_ledger', old_targets, range(count)),
        ('original_128_ledger', old_targets, range(128)),
        (f'unchanged_{len(unchanged)}_ledger', requested, unchanged),
        ('five_root_original', old_targets, [28, 29, 30, 31, 32])]
    if 'changed_target_indices' in contract:
        specs.extend([(f'a1_all_{count}_ledger', a1_targets, range(count)),
            ('a1_128_ledger', a1_targets, range(128)),
            ('legacy_128_ledger', old_targets, range(128))])
    metadata = {'declared_changed_indices': changed, 'changed_count': len(changed),
        'unchanged_count': len(unchanged), 'total_count': count,
        'changed_indices_source': 'input contract' if 'changed_target_indices' in contract else 'A1 baseline default',
        'unchanged_reference': 'current requested positions for the declared unchanged rows; legacy equivalence is not assumed',
        'a1_reference_controls': str(baseline/'controls.json'),
        'legacy_reference_controls': str(baseline/'original-controls.json')}
    return specs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args(); inp = args.input.resolve(); case = args.case.resolve()
    root_gate, gate_path = read_source_gate(case, expected_input=inp)
    solver_evidence = source_solver_evidence(case, gate_path)
    c = read(inp/'contract.json'); data = np.load(inp/'data.npz'); z = np.load(case/'solution.npy')
    grid = c['grid']; h, w = grid['height'], grid['width']; origin = np.asarray(grid['origin'], float)
    if grid['spacing'] != [2., 2.]:
        raise ValueError('this evaluator has explicit isotropic 2px coordinates')
    if sha(Path(c['source_contract'])) != c['source_contract_sha256'] or sha(Path(c['source_photo'])) != c['source_photo_sha256']:
        raise ValueError('source contract or photograph changed')
    if z.shape != (c['scalar_variables'], 2) or not np.isfinite(z).all():
        raise ValueError('invalid reduced source solution')
    mapping = data['node_to_variable']; u = np.zeros((len(mapping), 2)); movable = mapping >= 0
    u[movable] = z[mapping[movable]]; u = u.reshape(h, w, 2)
    unit = u/2.; geometry = source_geometry(unit); np.save(case/'source-forward.npy', u)
    save(case/'source-geometry.json', geometry)
    f = cell_jacobians(unit); sym = (f+f.swapaxes(-1, -2))*.5
    eig = np.linalg.eigvalsh(sym); skew = abs(f[..., 1, 0]-f[..., 0, 1])*.5
    schur = np.linalg.eigvalsh(sym-.52*(f.swapaxes(-1, -2)@f))[..., 0]
    np.savez_compressed(case/'source-corner-violations.npz',
        nonpositive=np.argwhere(eig[..., 0] <= 0),
        sufficient_lower=np.argwhere(eig[..., 0] < 1/1.8),
        sufficient_schur=np.argwhere(schur < 0),
        sufficient_skew=np.argwhere(skew > .24/(1.8**2)))
    source = data['target_sources']; requested = data['target_destinations']
    mapped, _ = sample(unit, (source-origin)/2.); mapped = mapped*2.+origin
    base = Path(c['baseline']); ledger_specs, ledger_metadata = target_ledgers(c, base, source, requested)
    target_report = {'mapping': 'continuous source bilinear map before any inverse quantization',
        'ledger_metadata': ledger_metadata,
        **{name: summarize(mapped, wanted, ids, source) for name, wanted, ids in ledger_specs}}
    save(case/'source-target-ledger.json', target_report)
    if geometry['strong_monotonicity_lower_bound'] <= 0 or not geometry['boundary_zero']:
        result = {'decision': 'reject_source_geometry_no_inverse_or_render', 'beauty_accepted': False,
            'source_geometry': geometry, 'source_target_count': len(source), 'source_contract_complete': True, **solver_evidence}
        save(case/'evaluation-decision.json', result); print(json.dumps(result, indent=2)); return
    left, top, cw, ch = c['crop']; yy, xx = np.mgrid[:ch, :cw]
    local_q = np.stack([xx.ravel(), yy.ravel()], axis=1)/2.
    start = time.monotonic(); inverse, closure = inverse_points(unit, local_q)
    offsets = ((inverse-local_q)*2.).reshape(ch, cw, 2).astype('<f4')
    closure['seconds'] = time.monotonic()-start
    closure['logical_grid_spacing_px'] = 2.
    closure['physical_closure_max_px'] = closure['closure_max_px']*2.
    actual_map, _ = sample(unit, local_q+offsets.reshape(-1, 2).astype(float)/2.)
    closure['final_f32_physical_closure_max_px'] = float(np.linalg.norm((actual_map-local_q)*2., axis=1).max())
    final_geometry = inspect(offsets)
    save(case/'inverse-closure.json', closure); save(case/'final-inverse-geometry.json', final_geometry)
    trace = case/'photo-trace'; trace.mkdir(exist_ok=False)
    offsets.tofile(trace/'inverse.f32'); np.ones((ch, cw), dtype='<f4').tofile(trace/'color.f32')
    Image.fromarray(np.zeros((ch, cw), dtype='uint8')).save(trace/'frozen.png')
    save(trace/'metadata.json', {'left': left, 'top': top, 'width': cw, 'height': ch,
        'representation': 'strict inverse of source 2px C0 bilinear field, float32 pixel samples',
        'coordinate_clamp_used': False, 'coefficient_scaling_after_solve': False})
    p = copy.deepcopy(read(base/'provenance.json')); p['source_forward_candidate'] = c
    save(case/'provenance.json', p)
    translations = z[data['group_variable_ids']]
    save(case/'field.json', {'representation': 'source-forward-strict-inverse-pixel-samples',
        'rigid_translations': [{'actual': (-v).tolist()} for v in translations],
        'source_forward': str(case/'source-forward.npy'), 'grid': grid})
    if not final_geometry['passed_strict']:
        result = {'decision': 'reject_final_inverse_geometry_no_render', 'beauty_accepted': False,
            'source_geometry': geometry, 'inverse_geometry': final_geometry, **solver_evidence}
        save(case/'evaluation-decision.json', result); print(json.dumps(result, indent=2)); return
    renderer = ROOT/'.run/face-current-20261019-target3/release/examples/face-photo-transport'
    with (case/'render.log').open('w') as log:
        subprocess.run([str(renderer), '--input', c['source_photo'], '--trace', str(trace),
            '--output', str(case/'photo.png')], stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
    rgb = audit(case); save(case/'full-footprint-rgb.json', rgb)
    original_image = np.asarray(Image.open(c['source_photo']).convert('RGB'))
    photo = np.asarray(Image.open(case/'photo.png').convert('RGB'))
    fixed = data['source_fixed_pixels'].astype(int)
    fixed_error = abs(photo[fixed[:, 1], fixed[:, 0]].astype(int)-original_image[fixed[:, 1], fixed[:, 0]].astype(int))
    # The final delivered inverse map, not source coefficients, defines this ledger.
    final_points, final_closure = forward(load(case), source)
    final_targets = {name: summarize(final_points, wanted, ids, source) for name, wanted, ids in ledger_specs}
    save(case/'final-target-ledger.json', dict(final_targets, forward_inverse_closure=final_closure,
        ledger_metadata=ledger_metadata))
    ear_rows = []
    for image_case in [base, case]:
        inverse_sampler = load(image_case)
        ear_rows.append({'case': str(image_case), 'complete_source_ears': {
            group['ear']: material_stats(inverse_sampler, np.asarray(group['source_samples'], float))
            for group in p['ear_orientation']['groups_evidence']}})
    save(case/'complete-source-ear-comparison.json', {'source_pixel_count': 5096, 'cases': ear_rows,
        'limitation': 'Actual final inverse map finite differences and same source pixels; not an aesthetic rating.'})
    # Report how the old numerical zero domain changed without silently
    # reinstating near-feature numerical margins as source semantic locks.
    starts, ends = data['old_identity_patch_min'], data['old_identity_patch_max']
    probes = np.concatenate([starts, ends, (starts+ends)*.5])
    logical = (probes-origin)/2.
    inside = np.all((logical >= 0) & (logical <= [w-1, h-1]), axis=1)
    # Old rectangle endpoints can exceed the crop by a JSON-roundtrip ULP.
    # Evaluate the declared identity extension outside, never clamp queries.
    delta = np.zeros_like(probes)
    actual, _ = sample(unit, logical[inside])
    delta[inside] = (actual-logical[inside])*2.
    historical = {'old_all_fixed_patch_count': len(starts), 'probe_count': len(probes),
        'nonzero_probe_count': int(np.count_nonzero(np.linalg.norm(delta, axis=1) > 1e-9)),
        'probe_max_displacement_px': float(np.linalg.norm(delta, axis=1).max()),
        'identity_extension_probe_count': int((~inside).sum()),
        'preserved_context_patch_count': c['preserved_context_source_patches'],
        'meaning': 'Corners and centers only for historical full-fixed numerical ledger; hard context domain is separately retained structurally.'}
    save(case/'historical-zero-region-ledger.json', historical)
    images = [Image.fromarray(original_image), Image.open(base/'photo.png').convert('RGB'), Image.fromarray(photo)]
    for name, box, scale in [('full', (0, 0, 1024, 1024), 1), ('face', (320, 185, 705, 635), 2),
        ('upper-face', (320, 185, 705, 430), 2),
        ('left-ear', (315, 285, 425, 515), 4), ('right-ear', (605, 285, 710, 515), 4), ('neck', (350, 480, 680, 730), 2)]:
        width, height = (box[2]-box[0])*scale, (box[3]-box[1])*scale
        panel = Image.new('RGB', (width*3, height+26), 'white'); draw = ImageDraw.Draw(panel)
        for i, (image, label) in enumerate(zip(images, ['Source', 'A1-step1 inverse B3', 'Source C0 candidate / diagnostic'])):
            panel.paste(image.crop(box).resize((width, height), Image.Resampling.NEAREST), (i*width, 26))
            draw.text((i*width+4, 4), label, fill='black')
        panel.save(case/(name+'.png'))
    result = {'decision': 'requires_independent_rgb_and_photo_review', 'beauty_accepted': False,
        'complete_fixed_source_rgb_max': int(fixed_error.max(initial=0)),
        'complete_fixed_source_changed_pixels': int(np.count_nonzero(fixed_error.max(-1))),
        'full_footprint_strict_f64_exact': rgb['all_full_footprint_strict_f64_legacy_oracles_exact'],
        'actual_f32_rgb_exact': rgb['all_actual_f32_renderer_order_oracles_exact'],
        'final_inverse_geometry_pass': True, 'source_basis': 'C0 bilinear', 'new_objective_not_old_equivalent': True, **solver_evidence}
    if not result['full_footprint_strict_f64_exact'] or result['complete_fixed_source_changed_pixels']:
        result['decision'] = 'reject_strict_source_rgb_diagnostic_photo_only'
    save(case/'evaluation-decision.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
