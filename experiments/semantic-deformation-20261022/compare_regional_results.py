"""Compare frozen regional and area-matched uniform edits without solving.

Uses original source geometry and existing bilinear photographs. Polygon area
is measured in complete source-image coordinates, separately for requested and
actually reached contours; equal requested area is not assumed equal output.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from diagnose_face_proportions import diagnose
from reduced_field import ROOT
from adapt import read, save, sha


RESULTS = ROOT/'test/results/semantic-deformation-20261022'
REGIONAL = RESULTS/'regional-results-v1'
GROUPS = {
    'hair-plus60': ('hair', [
        ('Old whole +60', RESULTS/'hair-plus60-focused-v2'),
        ('Area-matched uniform +60', REGIONAL/'hair-plus60-uniform-matched-focused1'),
        ('Regional +60', REGIONAL/'hair-plus60-regional-focused2')]),
    'woman-plus60': ('woman', [
        ('Old whole +60', RESULTS/'woman-plus60-focused-v1'),
        ('Area-matched uniform +60', REGIONAL/'woman-plus60-uniform-matched'),
        ('Regional +60', REGIONAL/'woman-plus60-regional-focused1')]),
    'hair-minus60': ('hair', [
        ('Old whole -60', RESULTS/'hair-minus60-v1'),
        ('Regional -60', REGIONAL/'hair-minus60-regional')]),
}


def polygon_area(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('finite complete polygon required')
    centered = points-points.mean(axis=0)
    return float(abs(np.sum(centered[:, 0]*np.roll(centered[:, 1], -1)
                            - centered[:, 1]*np.roll(centered[:, 0], -1)))*.5)


def panel(images, labels, output, box=None):
    images = [image.convert('RGB') if box is None else image.convert('RGB').crop(box) for image in images]
    if len({image.size for image in images}) != 1:
        raise ValueError('comparison images must share original dimensions')
    w, h = images[0].size
    canvas = Image.new('RGB', (len(images)*w, h+28), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image, (index*w, 28)); draw.text((index*w+4, 5), label, fill='black')
    canvas.save(output)


def compare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite regional comparison')
    output.mkdir(parents=True, exist_ok=False)
    result = {'script_sha256': sha(Path(__file__)), 'groups': {}, 'aesthetic_approval': False,
              'new_targets_or_solver_runs': False, 'sampling': 'existing bilinear photos; no Lanczos',
              'area_scope': 'complete 128-point closed contour in original image coordinates; not physical face surface area'}
    bound_hashes = {str(Path(__file__).resolve()): sha(Path(__file__))}
    for group_name, (name, cases) in GROUPS.items():
        images, labels, measurements = [], ['Original source'], []
        first_source, first_hash, original_image = None, None, None
        for label, case in cases:
            protocol = read(case/'protocol.json'); inp = Path(protocol['input'])
            contract, sc = read(inp/'contract.json'), read(inp/'source-contract.json')
            decision = read(case/'decision.json')
            if decision.get('final_geometry_pass') is not True or not (case/'photo.png').is_file():
                raise ValueError('comparison requires an existing geometry-passing photograph: '+str(case))
            source_path = Path(contract['source_photo']); digest = sha(source_path)
            if digest != contract['source_photo_sha256']:
                raise ValueError('source photograph hash mismatch')
            source = np.asarray([t['source'] for t in sc['targets'][:128]], float)
            requested = np.asarray([t['destination'] for t in sc['targets'][:128]], float)
            actual = np.asarray(read(case/'final-contour.json')['actual'], float)
            if first_source is None:
                first_source, first_hash = source, digest
                original_image = ImageOps.exif_transpose(Image.open(source_path)).convert('RGB')
                images.append(original_image)
            elif digest != first_hash or not np.array_equal(first_source, source):
                raise ValueError('comparison arms differ in source image or source contour')
            record = diagnose(name, case)
            source_area = polygon_area(source)
            areas = {stage: polygon_area(points) for stage, points in
                     [('source', source), ('requested', requested), ('actual', actual)]}
            neck = sc.get('regional_variant', {}).get('neck_arcs')
            if neck is None:
                provenance = read(Path(contract['baseline'])/'provenance.json')
                old_neck = provenance.get('source_neck_support')
                neck = [] if old_neck is None else [{
                    'profile_entry_gain': 1., 'rule': 'unchanged signed source baseline; no regional gain',
                    'source_rows': (np.asarray(arc['source_indices'])+old_neck['base_target_count']).tolist(),
                } for arc in old_neck['continuation']]
            record.update(label=label, polygon_areas_px2=areas,
                          requested_area_change_px2=areas['requested']-source_area,
                          actual_area_change_px2=areas['actual']-source_area,
                          actual_minus_requested_area_px2=areas['actual']-areas['requested'],
                          neck_entry_factors=neck,
                          profile=sc.get('regional_variant', {}).get('profile'),
                          decision=decision)
            for path in [source_path, case/'photo.png', case/'decision.json']:
                record['input_hashes'][str(path)] = sha(path)
            bound_hashes.update(record['input_hashes'])
            measurements.append(record)
            images.append(Image.open(case/'photo.png').convert('RGB')); labels.append(label)
        lo = np.maximum(np.floor(first_source.min(0)-35), [0, 0]).astype(int)
        hi = np.minimum(np.ceil(first_source.max(0)+45), original_image.size).astype(int)
        folder = output/group_name; folder.mkdir()
        panel(images, labels, folder/'full.png')
        panel(images, labels, folder/'face.png', (*lo, *hi))
        comparison = {'arms': measurements, 'source_contour_identical': True, 'source_photo_identical': True,
                      'face_crop_source_coordinates': [*lo.tolist(), *hi.tolist()]}
        if len(measurements) == 3:
            uniform, regional = measurements[1:]
            comparison['regional_minus_uniform'] = {
                'requested_area_px2': regional['polygon_areas_px2']['requested']-uniform['polygon_areas_px2']['requested'],
                'actual_area_px2': regional['polygon_areas_px2']['actual']-uniform['polygon_areas_px2']['actual'],
                'actual_lips_face_width_ratio': regional['features']['lips']['actual']['feature_to_face_width_ratio']-uniform['features']['lips']['actual']['feature_to_face_width_ratio'],
                'actual_face_width_at_lips_px': regional['features']['lips']['actual']['face_width_at_feature_height_px']-uniform['features']['lips']['actual']['face_width_at_feature_height_px'],
                'actual_height_width_ratio': regional['contour_extent']['actual']['height_width_ratio']-uniform['contour_extent']['actual']['height_width_ratio'],
            }
        save(folder/'report.json', comparison)
        result['groups'][group_name] = comparison
    if any(sha(Path(path)) != digest for path, digest in bound_hashes.items()):
        raise ValueError('frozen comparison inputs changed during measurement')
    result['all_inputs_unchanged'] = True
    result['input_hashes'] = bound_hashes
    save(output/'report.json', result)
    print({'output': str(output), 'groups': list(result['groups']), 'all_inputs_unchanged': True})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.output)
