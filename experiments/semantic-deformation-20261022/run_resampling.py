"""Compare two reconstruction kernels on one unchanged accepted float32 map."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from resampling import render_frozen_field, sample_rgba


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def panel(images, labels, destination, box=None, scale=1):
    crops = [image if box is None else image.crop(box) for image in images]
    crops = [image.resize((image.width*scale, image.height*scale), Image.Resampling.NEAREST) for image in crops]
    height = max(image.height for image in crops); width = sum(image.width for image in crops)
    result = Image.new('RGB', (width, height+24), '#eeeeee'); draw = ImageDraw.Draw(result)
    x = 0
    for image, label in zip(crops, labels):
        result.paste(image.convert('RGB'), (x, 24)); draw.text((x+5, 5), label, fill='#111111'); x += image.width
    result.save(destination)


def gradient_energy(image):
    rgb = image[..., :3].astype(float)
    return float(np.mean(np.diff(rgb, axis=0)**2)+np.mean(np.diff(rgb, axis=1)**2))


def sinusoid_probe():
    yy, xx = np.mgrid[:80, :96]; py, px = np.mgrid[5:74, 5:90]
    points = np.column_stack([px.ravel()+.37, py.ravel()-.23]); records = []
    for fx, fy in [(.05, .03), (.14, .09), (.24, 0.)]:
        values = 128.+60.*np.sin(2*np.pi*(fx*xx+fy*yy))
        source = np.dstack([values, values, values, np.full_like(values, 255.)])
        exact = 128.+60.*np.sin(2*np.pi*(fx*points[:, 0]+fy*points[:, 1]))
        row = {'frequency_cycles_per_pixel': [fx, fy], 'subpixel_translation': [.37, -.23]}
        for kernel in ['bilinear', 'bounded_lanczos3']:
            sampled, _ = sample_rgba(source, points, kernel)
            row[kernel+'_rmse'] = float(np.sqrt(np.mean((sampled[:, 0]-exact)**2)))
        records.append(row)
    return {'known_continuous_signal': '128+60*sin(2*pi*(fx*x+fy*y)), unquantized source samples',
            'cases': records, 'portrait_quality_inferred': False}


def compare(case, output):
    case, output = Path(case).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite reconstruction evidence: '+str(output))
    protocol = read(case/'protocol.json'); inp = Path(protocol['input'])
    contract = read(inp/'contract.json'); decision = read(case/'decision.json')
    if decision.get('final_geometry_pass') is not True:
        raise ValueError('this reconstruction experiment requires an already passing final map')
    paths = [case/'protocol.json', case/'decision.json', case/'inverse-field.npy', case/'photo.png',
             case/'photo-trace/inverse.f32', inp/'contract.json', inp/'data.npz',
             inp/'source-contract.json', Path(contract['source_photo']),
             Path(__file__), Path(__file__).with_name('resampling.py')]
    hashes = {str(path): sha(path) for path in paths}
    if hashes[contract['source_photo']] != contract['source_photo_sha256']:
        raise ValueError('source photo differs from frozen original')
    source = np.asarray(ImageOps.exif_transpose(Image.open(contract['source_photo'])).convert('RGBA'))
    field = np.load(case/'inverse-field.npy', allow_pickle=False)
    actual = np.fromfile(case/'photo-trace/inverse.f32', dtype='<f4').reshape(field.shape)
    if field.dtype != np.dtype('<f4') or not np.array_equal(actual, field):
        raise ValueError('saved field differs from the actual rendered float32 trace')
    crop = contract['crop']
    baseline, baseline_report = render_frozen_field(source, field, crop, 'bilinear')
    old_photo = np.asarray(Image.open(case/'photo.png').convert('RGBA'))
    baseline_error = abs(baseline.astype(int)-old_photo.astype(int))
    if np.any(baseline_error):
        raise ValueError('comparison baseline does not exactly reproduce the existing Rust photograph')
    candidate, candidate_report = render_frozen_field(source, field, crop, 'bounded_lanczos3')
    if not np.array_equal(candidate[..., 3], baseline[..., 3]):
        raise ValueError('candidate altered baseline alpha')
    fixed = np.load(inp/'data.npz', allow_pickle=False)['source_fixed_pixels'].astype(int)
    fixed_error = abs(candidate[fixed[:, 1], fixed[:, 0]].astype(int)-source[fixed[:, 1], fixed[:, 0]].astype(int))
    if np.any(fixed_error):
        raise ValueError('candidate changed original fixed source pixels')
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(candidate).save(output/'candidate.png')
    images = [Image.fromarray(source), Image.fromarray(baseline), Image.fromarray(candidate)]
    labels = ['Original source', 'Frozen-map bilinear', 'Same-map bounded Lanczos3']
    panel(images, labels, output/'full.png')
    source_contract = read(inp/'source-contract.json'); details = []
    for index, group in enumerate(source_contract['groups']):
        if not any('eye' in feature.lower() for feature in group.get('features', [group['name']])):
            continue
        points = np.asarray(group['source_material'], float)
        lo = np.maximum(np.floor(points.min(axis=0)).astype(int)-14, [0, 0])
        hi = np.minimum(np.ceil(points.max(axis=0)).astype(int)+15, [source.shape[1], source.shape[0]])
        box = [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]
        panel(images, labels, output/f'eye-{index}.png', box, scale=4)
        details.append({'group': group['name'], 'source_box': box,
                        'source_gradient_energy': gradient_energy(source[lo[1]:hi[1], lo[0]:hi[0]]),
                        'bilinear_gradient_energy': gradient_energy(baseline[lo[1]:hi[1], lo[0]:hi[0]]),
                        'candidate_gradient_energy': gradient_energy(candidate[lo[1]:hi[1], lo[0]:hi[0]]),
                        'metric_scope': 'unregistered box texture diagnostic; greater energy is not an aesthetic score'})
    contour = np.asarray([row['source'] for row in source_contract['targets'][:128]])
    lo = np.maximum(np.floor(contour.min(axis=0)).astype(int)-22, [0, 0])
    hi = np.minimum(np.ceil(contour.max(axis=0)).astype(int)+23, [source.shape[1], source.shape[0]])
    panel(images, labels, output/'face.png', [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])], scale=2)
    diff = abs(candidate.astype(int)-baseline.astype(int))
    report = {'source_case': str(case), 'requested_amount': contract['requested_amount'],
              'input_hashes': hashes, 'geometry_map': 'exact existing float32 inverse, no solve or modification',
              'source_and_field_unchanged': all(sha(Path(path)) == digest for path, digest in hashes.items()),
              'baseline_rust_byte_exact': True, 'baseline_max_byte_error': int(baseline_error.max(initial=0)),
              'baseline': baseline_report, 'candidate': candidate_report,
              'fixed_source_pixels': len(fixed), 'fixed_source_max_byte_error': int(fixed_error.max(initial=0)),
              'alpha_byte_identical': True, 'changed_pixels_vs_bilinear': int(np.any(diff != 0, axis=2).sum()),
              'mean_absolute_rgb_difference': float(diff[..., :3].mean()),
              'max_rgb_difference': int(diff[..., :3].max()), 'eye_texture_diagnostics': details,
              'known_signal_comparison': sinusoid_probe(), 'old_rgb_oracle_pass_claimed': False,
              'new_sampling_contract': '6x6 Lanczos3 opaque interior; per-channel 2x2 min/max limiter; original bilinear alpha and boundary/transparency fallback; zero/integer samples byte exact',
              'limitations': ['A fixed Lanczos kernel is not anisotropic minification anti-aliasing.',
                             'Local bounds suppress extrema and may change thin-line contrast; no independent photographic ground truth.',
                             'Encoded RGB interpolation retains the old color domain and is not linear-light reconstruction.',
                             'The 6x6 footprint is wider than the historical bilinear material contract; old RGB oracle is not reused.',
                             'Clipping color channels separately can change color near edges; visual inspection remains required.'],
              'aesthetic_approval': False, 'production_renderer_changed': False}
    if not report['source_and_field_unchanged']:
        raise ValueError('frozen inputs changed during reconstruction comparison')
    save(output/'report.json', report)
    print({'output': str(output), 'baseline_seconds': baseline_report['seconds'],
           'candidate_seconds': candidate_report['seconds'], 'baseline_rust_byte_exact': True,
           'candidate_bounded_samples': candidate_report['bounded_sample_count'],
           'changed_pixels_vs_bilinear': report['changed_pixels_vs_bilinear']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.case, args.output)
