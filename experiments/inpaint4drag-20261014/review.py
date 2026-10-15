#!/usr/bin/env python3
"""Archive verified counterexamples; do not treat mask IoU as image quality."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.run/inpaint4drag-deps'))
sys.path.insert(0, str(ROOT / 'scripts'))
import cv2
import numpy as np
from PIL import Image
import torch
from replay_user_face_contour import raster_coverage
from vendor_drag import bi_warp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    results = ROOT / 'test/results/inpaint4drag-20261014'
    archive = ROOT / 'doc/qa/inpaint4drag-20261014'
    archive.mkdir(exist_ok=False)
    report = json.loads((results / 'report.json').read_text())
    source_path = ROOT / 'test/results/user-face-contour-20261014/1024/canonical-source.png'
    source = np.array(Image.open(source_path).convert('RGBA'))
    assert sha(source_path) == report['source_sha256']
    graph = json.loads((results / 'faceWidth100/graph.json').read_text())
    semantics = graph['operations'][0]['params']['semanticMasks']
    masks = {}
    for name in ['nose', 'lips', 'leftEye', 'rightEye', 'leftBrow', 'rightBrow']:
        mask = semantics[name]
        if mask['shape'] == 'raster':
            masks[name] = raster_coverage(mask, source.shape[1], source.shape[0]) >= .5
        else:
            value = np.zeros(source.shape[:2], dtype=np.uint8)
            points = np.rint([[p['x'] * source.shape[1], p['y'] * source.shape[0]] for p in mask['points']]).astype(np.int32)
            cv2.fillPoly(value, [points], 1)
            masks[name] = value > 0
    review = {'status': 'REJECTED for production; partial geometry/completion study only',
              'source_sha256': report['source_sha256'], 'experiment_report_sha256': sha(results / 'report.json'),
              'mask_monitor_scope': 'Rounded source-anatomical polygon interior; not Rust feathered coverage or output redetection.',
              'feature_collateral': []}
    for amount in [30, -30, 60, -60, 100]:
        case = results / f'faceWidth{amount}'
        row = {'amount': amount, 'variants': {}}
        for name in ['native.png', 'official-warp.png', 'protected-warp-lama.png']:
            path = case / name
            output = np.array(Image.open(path).convert('RGBA'))
            changed = np.any(output != source, axis=2)
            row['variants'][name] = {'sha256': sha(path), 'changed_feature_pixels': {key: int((changed & value).sum()) for key, value in masks.items()}}
        review['feature_collateral'].append(row)
    # The original runner's zero branch did not call bi_warp. Verify it here
    # independently instead of turning a static placeholder into test evidence.
    geo = json.loads((results / 'faceWidth100/native-debug/geometry.json').read_text())
    controls = np.empty((len(geo['source']) * 2, 2), dtype=np.float32)
    controls[::2] = controls[1::2] = np.rint(geo['source'])
    mask = np.array(Image.open(results / 'faceWidth100/source-mask.png')) > 0
    torch.set_num_threads(3)
    donors, destinations, completion = bi_warp(mask.astype(np.uint8), controls, 5)
    zero = source.copy()
    zero[destinations[:, 1], destinations[:, 0]] = source[donors[:, 1], donors[:, 0]]
    assert np.array_equal(zero, source)
    review['independent_zero_test'] = {'warped_changed_pixels': 0, 'official_completion_mask_pixels': int(np.count_nonzero(completion)),
        'scope': 'Same frozen source polygon, equal source/target integer controls; unmodified official bi_warp executed.',
        'implication': 'Even identity controls produce a contour inpaint band; the caller must skip completion at zero adjustment.'}
    case = results / 'faceWidth100'
    names = ['native.png', 'official-warp.png', 'protected-warp-lama.png']
    images = {name: np.array(Image.open(case / name).convert('RGBA')) for name in names}
    target = np.array(Image.open(case / 'expected-target-mask.png')) > 0
    completion = np.array(Image.open(case / 'protected-completion-mask.png')) > 0
    review['target_exterior_probes'] = []
    for x in [613, 616, 619, 624, 630, 634, 637, 640]:
        y = 480
        review['target_exterior_probes'].append({'xy': [x, y], 'inside_expected_target': bool(target[y, x]), 'completion_writable': bool(completion[y, x]),
            'source_rgba': source[y, x].tolist(), **{name: pixels[y, x].tolist() for name, pixels in images.items()}})
    for name in ['report.json']:
        shutil.copy2(results / name, archive / 'experiment-report.json')
    for amount, filename in [(30, 'right-comparison-2x.png'), (-30, 'right-comparison-2x.png'), (60, 'right-comparison-2x.png'), (-60, 'jaw-comparison-2x.png'), (100, 'previous-lama-comparison-2x.png')]:
        shutil.copy2(results / f'faceWidth{amount}' / filename, archive / f'faceWidth{amount}-{filename}')
    for name, image in {'source': source, **images}.items():
        Image.fromarray(image).crop((590, 420, 662, 540)).save(archive / f'right-outline-native-pixels-{name.removesuffix(".png")}.png')
    review['archived_files_sha256'] = {path.name: sha(path) for path in archive.iterdir() if path.is_file()}
    (archive / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review['independent_zero_test']))
    for row in review['feature_collateral']:
        print(row['amount'], row['variants']['official-warp.png']['changed_feature_pixels'])


if __name__ == '__main__':
    run()
