"""Keep one reviewed jaw map across preview/native output resolutions.

Both inputs must be passing portable-jaw cases of the same photo and request.
The independently solved native case is retained as a diagnostic comparison.
Only geometry is transferred: output pixels are sampled from the native source.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from run_portable_jaw import read, save, sha, snapshot_local_code
from resolution_field import lift_inverse_field
from portable_result_gate import evaluate_result
from evaluate import inverse_points, inspect
from resampling import render_frozen_field


def run(reference_case, native_case, output):
    if output.exists():
        raise FileExistsError('refusing to overwrite resolution experiment')
    started = time.perf_counter()
    paths = []
    records = []
    for case in [reference_case, native_case]:
        for name in ['decision.json', 'inverse-field.npy', 'contours.npz']:
            paths.append(case/name)
        paths += [case.parent/name for name in ['protocol.json', 'report.json', 'fixed-source.png']]
        decision = read(case/'decision.json')
        if not decision.get('execution_gate', {}).get('passed') or not decision['final_geometry']['passed_strict']:
            raise ValueError('both frozen cases must pass their execution and final geometry checks')
        protocol = read(case.parent/'protocol.json')
        source_path = Path(read(case.parent/'report.json')['source'])
        if protocol['inputs_sha256'].get(str(source_path)) != sha(source_path):
            raise ValueError('frozen source photograph changed')
        paths.append(source_path)
        records.append((decision, protocol, source_path))
    rd, rp, rsource = records[0]
    nd, np_, nsource = records[1]
    if not rd.get('continuous_arc_gate', {}).get('passed'):
        raise ValueError('the reference needs an independent continuous-arc execution check')
    paths += [reference_case/'arc-probes.npz', reference_case/'arc-probes.json']
    # Require common original image bytes, not merely matching dimensions or landmarks.
    rh, nh = rp['inputs_sha256'], np_['inputs_sha256']
    common_images = [p for p in rh.keys() & nh.keys()
                     if Path(p).suffix.lower() in ['.jpg', '.jpeg'] and rh[p] == nh[p]]
    if len(common_images) != 1 or sha(common_images[0]) != rh[common_images[0]]:
        raise ValueError('one common unchanged original JPEG is required')
    paths.append(Path(common_images[0]))
    if rd['amount'] != nd['amount'] or rp['max_radius_fraction'] != np_['max_radius_fraction']:
        raise ValueError('reference and native requests differ')
    rsize = Image.open(rsource).size
    source = np.asarray(Image.open(nsource).convert('RGBA'))
    nsize = (source.shape[1], source.shape[0])
    field = np.load(reference_case/'inverse-field.npy')
    hashes = {str(p): sha(p) for p in paths}
    lifted = lift_inverse_field(field, rp['crop'], rsize, nsize)
    candidate, crop, scale = lifted['field'], lifted['crop'], lifted['scale']
    old_contours, new_contours = [np.load(case/'contours.npz') for case in [reference_case, native_case]]
    for key in ['source', 'target']:
        if not np.allclose((old_contours[key]+.5)*scale-.5, new_contours[key], atol=1e-8, rtol=0):
            raise ValueError('source/target geometry was not transferred with pixel-center coordinates')
    output.mkdir(parents=True)
    snapshot, code_hashes = snapshot_local_code(output)
    hashes.update(code_hashes)
    np.save(output/'inverse-field.npy', candidate)
    final = inspect(candidate)
    result = {'inputs_sha256': hashes, 'source': str(nsource), 'crop': crop,
              'amount': nd['amount'], 'scale': scale, 'transfer': lifted['report'],
              'final_geometry': final, 'execution_code_snapshot': str(snapshot),
              'new_deformation_solve': False, 'native_source_sampled_directly': True,
              'aesthetic_approval': False}
    if not final['passed_strict']:
        result['status'] = 'rejected_final_geometry'
        save(output/'report.json', result)
        return result
    source_points, target = new_contours['source'], new_contours['target']
    actual, closure = inverse_points(candidate.astype(float), source_points-np.array(crop[:2]))
    actual += crop[:2]
    probe_data = np.load(reference_case/'arc-probes.npz')
    probe_source = (probe_data['source']+.5)*scale-.5
    probe_target = (probe_data['target']+.5)*scale-.5
    probe_actual, probe_closure = inverse_points(candidate.astype(float), probe_source-np.array(crop[:2]))
    probe_actual += crop[:2]
    fixed = np.asarray(Image.open(native_case.parent/'fixed-source.png')) > 0
    fixed_roi = fixed[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2]]
    fixed_motion = float(np.linalg.norm(candidate[fixed_roi].astype(float), axis=1).max(initial=0.))
    photos, gates = {}, {}
    for method in ['bilinear', 'bounded_lanczos3']:
        pixels, sampling = render_frozen_field(source, candidate, crop, method=method)
        changed = np.any(pixels != source, axis=2)
        gate = evaluate_result(source_points, target, actual, amount=nd['amount'],
            fixed_changed_pixels=int((changed & fixed).sum()),
            whole_image_changed_pixels=int(changed.sum()))
        continuous = evaluate_result(probe_source, probe_target, probe_actual, amount=nd['amount'],
            fixed_changed_pixels=int((changed & fixed).sum()), whole_image_changed_pixels=int(changed.sum()))
        gates[method] = {'execution': gate, 'sampling': sampling,
                        'continuous_arc_execution': continuous,
                        'fixed_changed_pixels': int((changed & fixed).sum())}
        if gate['passed'] and continuous['passed'] and fixed_motion == 0.:
            Image.fromarray(pixels).save(output/(method+'.png'))
            photos[method] = pixels
    # Native output pixel nodes independently evaluate the reference inverse.
    # Compare both the old re-solve and transferred f32 map in preview-pixel units.
    yy, xx = np.mgrid[:crop[3], :crop[2]]
    q = np.stack([xx+crop[0], yy+crop[1]], axis=-1)
    ref_q = ((q+.5)/scale-.5-rp['crop'][:2])
    expected = np.stack([map_coordinates(field[...,i].astype(float),
                         [ref_q[...,1], ref_q[...,0]], order=1, mode='constant', cval=0., prefilter=False)
                         for i in range(2)], axis=-1)
    independent_field = np.load(native_case/'inverse-field.npy')
    native_q = q-np_['crop'][:2]
    independent = np.stack([map_coordinates(independent_field[...,i].astype(float),
                            [native_q[...,1], native_q[...,0]], order=1, mode='constant', cval=0., prefilter=False)
                            for i in range(2)], axis=-1)/scale
    result.update(sampling_gates=gates, fixed_pixels=int(fixed.sum()),
        fixed_native_inverse_motion_max_px=fixed_motion, actual_contour_closure=closure,
        continuous_arc_closure=probe_closure,
        preview_inverse_difference_max_px=float(np.linalg.norm(candidate.astype(float)/scale-expected, axis=-1).max()),
        independent_resolve_difference_max_px=float(np.linalg.norm(independent-expected, axis=-1).max()),
        status='requires_visual_review' if fixed_motion == 0. and all(g['execution']['passed'] and g['continuous_arc_execution']['passed'] for g in gates.values()) else 'rejected_execution',
        elapsed_seconds=time.perf_counter()-started)
    if 'bounded_lanczos3' in photos:
        independent, _ = render_frozen_field(source, independent_field, np_['crop'], method='bounded_lanczos3')
        lo = np.maximum(np.floor(source_points.min(0)-25).astype(int), 0)
        hi = np.minimum(np.ceil(source_points.max(0)+30).astype(int), nsize)
        box = (*lo, *hi); w, h = (hi-lo).tolist()
        panel = Image.new('RGB', (3*w, h+28), 'white'); draw = ImageDraw.Draw(panel)
        for i, (pixels, label) in enumerate(zip([source, independent, photos['bounded_lanczos3']],
                ['Original native source', 'Native independent re-solve', 'Same preview map, native source'])):
            panel.paste(Image.fromarray(pixels).convert('RGB').crop(box), (i*w, 28))
            draw.text((i*w+3, 5), label, fill='black')
        panel.save(output/'face.png')
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError('frozen inputs or executing code changed')
    save(output/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference-case', 'native-case', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.reference_case.resolve(), args.native_case.resolve(), args.output.resolve())
    print({key: result[key] for key in ['status', 'amount', 'elapsed_seconds'] if key in result})
