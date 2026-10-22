"""One frozen-request subspace experiment, then original image checks.

The reduced quadratic has its own identity. Never labels it an optimum of the
old full-dimensional problem. Does not change strength, masks or rendering.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import load_npz, save_npz

from reduced_field import build_reduction, solve_reduced, ROOT
from input_contract import validate_input
from reduced_constrained import solve_constrained_reduced
from adapt import read, save, sha
from evaluate import source_geometry, inverse_points, sample, inspect, summarize, audit, load, forward
from audit_same_input_protection import sampler, line_maximum


def run(inp, out, stride=8, fine_band=4, refinement=None, constrained=False):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    validation = validate_input(inp)
    save(out / 'input-validation.json', validation)
    c = read(inp / 'contract.json')
    original = read(inp / 'source-contract.json')
    grid = c['grid']
    if grid['spacing'] != [2., 2.]:
        raise ValueError('this experiment retains the original 2px representation')
    data = dict(np.load(inp / 'data.npz'))
    folder = inp / ('quadratic' if (inp / 'quadratic').exists() else 'proposal')
    files = [inp / 'contract.json', inp / 'source-contract.json', inp / 'data.npz',
             folder / 'hessian.npz', folder / 'rhs.npy', Path(c['source_photo'])]
    lines_file = inp / 'hard_line_equalities.npz'
    if lines_file.exists():
        files.append(lines_file)
    base = Path(c['baseline'])
    files.append(base / 'provenance.json')
    if refinement is not None:
        files.append(refinement)
    hashes = dict(validation['verified_input_hashes'])
    hashes.update({str(p): sha(p) for p in files})
    if hashes[str(Path(c['source_photo']))] != c['source_photo_sha256']:
        raise ValueError('source photograph changed')
    if hashes[str(inp / 'source-contract.json')] != c['source_contract_sha256']:
        raise ValueError('source contract changed')
    code = out / 'executed-code'; code.mkdir()
    for name in ['run_reduced_photo.py', 'reduced_field.py', 'reduced_constrained.py', 'input_contract.py']:
        shutil.copy2(Path(__file__).with_name(name), code / name)
    save(out / 'protocol.json', {'input': str(inp), 'inputs_sha256': hashes,
         'stride_grid_cells': stride, 'fine_band_grid_cells': fine_band,
         'source_requests_masks_weights_and_fine_grid_unchanged': True,
         'new_search_space': 'z=R a; minimize original quadratic over a with all original line equalities',
         'bounded_active_source_SOC': constrained,
         'old_full_space_optimality_claimed': False, 'old_geometry_thresholds_unchanged': True,
         'no_strength_scaling_clipping_or_output_interpolation': True,
         'historical_failures_not_reclassified': True})
    extra = None if refinement is None else np.load(refinement)
    r, reduction = build_reduction(data['node_to_variable'], grid['width'], grid['height'],
                                  data['group_variable_ids'], stride=stride, fine_band=fine_band, extra_nodes=extra)
    save(out / 'reduction.json', reduction); save_npz(out / 'reduction.npz', r)
    h, rhs = load_npz(folder / 'hessian.npz'), np.load(folder / 'rhs.npy')
    lines = load_npz(lines_file) if lines_file.exists() else None
    result = (solve_constrained_reduced(h, rhs, r, grid, data['node_to_variable'], lines)
              if constrained else solve_reduced(h, rhs, r, lines))
    z = result['solution']; np.save(out / 'solution.npy', z)
    np.save(out / 'coefficients.npy', result['coefficients'])
    if 'multipliers' in result:
        np.save(out / 'line-multipliers.npy', result['multipliers'])
    save(out / 'solver-evidence.json', result['report'])
    print({'stage': 'solver_returned', 'full': r.shape[0], 'reduced': r.shape[1],
           'build_seconds': reduction['construction_seconds'],
           **{k: v for k, v in result['report'].items()
              if isinstance(v, (str, float, int, bool))}}, flush=True)
    mapping = data['node_to_variable']; active = mapping >= 0
    u = np.zeros((len(mapping), 2)); u[active] = z[mapping[active]]
    u = u.reshape(grid['height'], grid['width'], 2); unit = u / 2.
    np.save(out / 'source-forward.npy', u)
    geometry = source_geometry(unit); save(out / 'source-geometry.json', geometry)
    mapped, _ = sample(unit, (data['target_sources'] - grid['origin']) / 2.)
    mapped = mapped * 2. + grid['origin']
    count = original.get('target_contour_count', 128)
    target_report = summarize(mapped, data['target_destinations'], range(count), data['target_sources'])
    save(out / 'source-contour.json', target_report)
    decision = {'source_geometry': geometry, 'solver_scope': 'reduced_subspace_only',
                'aesthetic_approval': False, 'requested_amount': c['requested_amount'],
                'bounded_active_source_SOC': constrained,
                'solver_status': result['report'].get('status', 'reduced_equality_solution'),
                'solver_diagnostic_only': result['report'].get('diagnostic_only', False),
                'full_variables': r.shape[0], 'reduced_variables': r.shape[1]}
    def finish(reason):
        if any(sha(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError('frozen experiment inputs mutated during execution')
        decision.update(decision=reason, elapsed_seconds=time.perf_counter()-started,
                        inputs_unchanged=True)
        save(out / 'decision.json', decision)
        print({k: v for k, v in decision.items() if k != 'source_geometry'}, flush=True)
    if geometry['strong_monotonicity_lower_bound'] <= 0 or not geometry['boundary_zero']:
        finish('reject_source_geometry_no_render'); return
    left, top, width, height = c['crop']
    yy, xx = np.mgrid[:height, :width]
    q = np.column_stack([xx.ravel(), yy.ravel()]) / 2.
    tick = time.perf_counter(); inverse, closure = inverse_points(unit, q)
    field = ((inverse-q)*2.).reshape(height, width, 2).astype('<f4')
    np.save(out / 'inverse-field.npy', field)
    closure.update(seconds=time.perf_counter()-tick, physical_closure_max_px=2*closure['closure_max_px'])
    save(out / 'inverse-closure.json', closure)
    qc = inspect(field); save(out / 'final-inverse-geometry.json', qc)
    decision['final_geometry_pass'] = qc['passed_strict']
    if not qc['passed_strict']:
        finish('reject_final_geometry_no_render'); return
    trace = out / 'photo-trace'; trace.mkdir()
    field.tofile(trace / 'inverse.f32')
    np.ones((height, width), dtype='<f4').tofile(trace / 'color.f32')
    Image.fromarray(np.zeros((height, width), dtype='uint8')).save(trace / 'frozen.png')
    save(trace / 'metadata.json', {'left': left, 'top': top, 'width': width, 'height': height,
         'representation': 'strict inverse of original 2px field restricted to semantic subspace',
         'coordinate_clamp_used': False, 'coefficient_scaling_after_solve': False})
    p = copy.deepcopy(read(base / 'provenance.json')); p['source_forward_candidate'] = c
    save(out / 'provenance.json', p)
    save(out / 'field.json', {'representation': 'source-forward-strict-inverse-pixel-samples',
         'rigid_translations': [{'actual': (-v).tolist()} for v in z[data['group_variable_ids']]],
         'source_forward': str(out / 'source-forward.npy'), 'grid': grid})
    tick = time.perf_counter()
    with (out / 'render.log').open('w') as log:
        subprocess.run([str(ROOT / '.run/face-current-20261019-target3/release/examples/face-photo-transport'),
                        '--input', c['source_photo'], '--trace', str(trace), '--output', str(out / 'photo.png')],
                       check=True, timeout=60, stdout=log, stderr=subprocess.STDOUT)
    decision['render_seconds'] = time.perf_counter() - tick
    rgb = audit(out); save(out / 'full-footprint-rgb.json', rgb)
    before = np.asarray(Image.open(c['source_photo']).convert('RGB'))
    after = np.asarray(Image.open(out / 'photo.png').convert('RGB'))
    fixed = data['source_fixed_pixels'].astype(int)
    errors = abs(after[fixed[:, 1], fixed[:, 0]].astype(int) - before[fixed[:, 1], fixed[:, 0]].astype(int))
    decision.update(fixed_pixels=len(fixed), fixed_changed_pixels=int(np.any(errors, axis=1).sum()),
                    material_f64_exact=rgb['all_full_footprint_strict_f64_legacy_oracles_exact'],
                    material_f32_exact=rgb['all_actual_f32_renderer_order_oracles_exact'])
    decision['material_domains_preserved'] = all(not v['actual_footprint_outside_both_translation_domains_xy']
            and not v['f64_footprint_missing_from_actual_xy'] for v in rgb['groups'])
    actual, _ = forward(load(out), data['target_sources'])
    target_report = summarize(actual, data['target_destinations'], range(count), data['target_sources'])
    save(out / 'final-contour.json', target_report)
    decision.update(contour_max_error_px=target_report['max_error_px'],
                    contour_rms_error_px=target_report['rms_error_px'],
                    contour_mean_gain=target_report['mean_active_projection_gain'],
                    contour_worst_gain=target_report['worst_active_projection_gain'])
    originals = original.get('background_lines', [])
    line_reports = []
    source_sampler = sampler(u, np.asarray(grid['origin']), np.asarray(grid['spacing']))
    for line in originals:
        source_max = line_maximum(source_sampler, line['start'], line['end'], grid['origin'], grid['spacing'])
        final_max = line_maximum(load(out), line['start'], line['end'], [left, top], [1., 1.])
        line_reports.append({'source_normal_max_px': source_max, 'final_inverse_normal_max_px': final_max,
                             'original_tolerance_px': 1e-7, 'passed': source_max <= 1e-7 and final_max <= 1e-7})
    save(out / 'background-lines.json', line_reports)
    decision['background_lines_pass'] = all(v['passed'] for v in line_reports)
    decision['image_contract_pass'] = bool(decision['fixed_changed_pixels'] == 0
            and decision['material_f64_exact'] and decision['material_f32_exact']
            and decision['material_domains_preserved'] and decision['background_lines_pass'])
    contour = data['target_sources'][:count]
    lower = np.maximum(np.floor(contour.min(0)-35), [0, 0]).astype(int)
    upper = np.minimum(np.ceil(contour.max(0)+45), [after.shape[1], after.shape[0]]).astype(int)
    for name, box in [('full', (0, 0, after.shape[1], after.shape[0])), ('face', (*lower, *upper))]:
        pw, ph = box[2]-box[0], box[3]-box[1]
        panel = Image.new('RGB', (pw*2, ph+32), 'white'); draw = ImageDraw.Draw(panel)
        for i, im in enumerate([before, after]):
            panel.paste(Image.fromarray(im).crop(box), (i*pw, 32))
        draw.text((4, 4), 'Original', fill='black')
        draw.text((pw+4, 4), f'Reduced {c["requested_amount"]:+.0f}; offline candidate', fill='black')
        panel.save(out / (name + '.png'))
    finish('requires_independent_visual_review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--fine-band', type=int, default=4)
    parser.add_argument('--refinement', type=Path)
    parser.add_argument('--constrained', action='store_true',
                        help='bounded source SOC experiment; final image gates still apply')
    args = parser.parse_args()
    run(args.input.resolve(), args.output.resolve(), args.stride, args.fine_band,
        args.refinement.resolve() if args.refinement else None, args.constrained)
