#!/usr/bin/env python3
"""Replay pinned Inpaint4Drag pixel warping on a source-only portrait.

The default native binary is an isolated, frozen comparison candidate. Its
FACE_DEBUG_DIR export supplies each actual source/target contour; no edited
image is redetected. LaMa is optional and is a completion baseline, not the
paper's SD+LCM model or proof that this warp produces natural beauty edits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.run/inpaint4drag-deps'))
sys.path.insert(0, str(ROOT / 'scripts'))

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from replay_user_face_contour import raster_coverage
from vendor_drag import bi_warp

COMMIT = 'bfb0fafd456ac01ee23ade0cc04b64c9781eaa87'
BASE = ROOT / 'test/results/user-face-contour-20261014/1024'
PREVIOUS = ROOT / 'test/results/user-face-contour-20261014/local-refinement-ablation-20261014'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGBA')).copy()


def polygon_mask(points: np.ndarray, shape: tuple[int, int]):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask > 0


def contracts(before, after, frozen, allowed, eyes):
    changed = np.any(before != after, axis=2)
    return {
        'changed_pixels': int(changed.sum()),
        'hair_ear_accessory_changed': int(np.count_nonzero(changed & frozen)),
        'outside_source_target_band_changed': int(np.count_nonzero(changed & ~allowed)),
        'eyes_changed': int(np.count_nonzero(changed & eyes)),
        'alpha_changed': int(np.count_nonzero(before[:, :, 3] != after[:, :, 3])),
    }


def montage(images, labels, box, path):
    zoom = 2
    width, height = box[2] - box[0], box[3] - box[1]
    panel = Image.new('RGB', (width * zoom * len(images), height * zoom + 28), '#181c23')
    draw = ImageDraw.Draw(panel)
    for index, (img, label) in enumerate(zip(images, labels)):
        crop = Image.fromarray(img).convert('RGB').crop(box)
        panel.paste(crop.resize((width * zoom, height * zoom), Image.Resampling.NEAREST), (index * width * zoom, 28))
        draw.text((index * width * zoom + 5, 7), label, fill='white')
    panel.save(path)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'test/results/inpaint4drag-20261014')
    parser.add_argument('--binary', type=Path, default=ROOT / '.run/face-side-v12-trimap-candidate-image-render')
    parser.add_argument('--source', type=Path, default=BASE / 'canonical-source.png')
    parser.add_argument('--graph', type=Path, default=BASE / 'face-graph.json')
    parser.add_argument('--amounts', default='0,30,-30,60,-60,100')
    parser.add_argument('--lama', action='store_true')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Choose a new output directory; frozen results must not be overwritten')
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    source = rgba(args.source)
    height, width = source.shape[:2]
    graph = json.loads(args.graph.read_text())
    semantics = graph['operations'][0]['params']['semanticMasks']
    coverages = {name: raster_coverage(mask, width, height) for name, mask in semantics.items() if mask['shape'] == 'raster'}
    frozen = np.maximum.reduce([coverages[name] for name in ['hair', 'ears', 'accessories']]) >= .5
    # Eye classes in this frozen graph are anatomical polygons rather than
    # parser rasters. Monitor their integer polygon interior without using it
    # as a protection mask or pretending it is Rust's feathered coverage.
    eyes = np.zeros((height, width), dtype=bool)
    for name in ['leftEye', 'rightEye']:
        mask = semantics[name]
        if mask['shape'] == 'raster':
            eyes |= raster_coverage(mask, width, height) >= .5
        else:
            points = np.array([[p['x'] * width, p['y'] * height] for p in mask['points']])
            eyes |= polygon_mask(points, (height, width))
    save_mask(frozen, args.out / 'frozen-semantic-half.png')
    source_sha, binary_sha = sha(args.source), sha(args.binary)
    report = {
        'status': 'experimental; no production integration or quality acceptance',
        'official_commit': COMMIT,
        'official_source_url': f'https://raw.githubusercontent.com/Visual-AI/Inpaint4Drag/{COMMIT}/utils/drag.py',
        'vendor_sha256': sha(Path(__file__).with_name('vendor_drag.py')),
        'runner_sha256': sha(Path(__file__)), 'source_sha256': source_sha,
        'graph_sha256': sha(args.graph), 'native_binary_sha256': binary_sha,
        'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__, 'device': 'cpu', 'threads': args.threads},
        'notes': [
            'Actual contours are exported by the same frozen native candidate independently for each strength.',
            'geometry source and target are absolute coordinates; origin is only the diagnostic image crop offset.',
            'Official UI uses integer clicks: contour handles and targets are rounded, with exact floats archived.',
            'Official warp is unmodified. Protected compositing and using LaMa are explicit experiment adaptations.',
            'Far background contract uses source union expected target plus a 5px morphology band; expansion needs a writable destination.',
            'LaMa frozen pixels remain known conditioning; permission enforcement happens at the final composite.',
            'Integer sample finite differences are not interpreted as a continuous Jacobian or proof against folding.',
            'Eye collateral monitor uses rounded anatomical polygon interiors; no eye-freeze or feather-coverage claim.',
        ], 'rows': []}
    worker = None
    if args.lama:
        worker_path = ROOT / 'services/background-worker/worker.py'
        spec = importlib.util.spec_from_file_location('inpaint4drag_lama_baseline', worker_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        worker = module.LamaWorker(threads=args.threads)
        report['lama_worker_sha256'] = sha(worker_path)

    for amount in [int(value) for value in args.amounts.split(',')]:
        case = args.out / f'faceWidth{amount}'
        case.mkdir()
        native_debug = case / 'native-debug'
        current = json.loads(json.dumps(graph))
        current['operations'][0]['params']['adjustments'] = {'faceWidth': amount}
        graph_path = case / 'graph.json'
        graph_path.write_text(json.dumps(current))
        if sha(args.binary) != binary_sha:
            raise RuntimeError('Native comparison binary changed during replay')
        started = time.perf_counter()
        result = subprocess.run([str(args.binary), '--input', str(args.source), '--graph', str(graph_path), '--output', str(case / 'native.png')],
            env={**os.environ, 'FACE_DEBUG_DIR': str(native_debug)}, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        native_ms = (time.perf_counter() - started) * 1000
        native = rgba(case / 'native.png')
        if amount == 0:
            assert np.array_equal(source, native)
            row = {'amount': amount, 'native_identity_pixels': 0, 'official_identity_pixels': 0, 'native_ms': round(native_ms, 2)}
            report['rows'].append(row)
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            continue
        geometry_path = native_debug / 'geometry.json'
        geo = json.loads(geometry_path.read_text())
        original_points = np.asarray(geo['source'], dtype=np.float32)
        target_points = np.asarray(geo['target'], dtype=np.float32)
        source_mask = polygon_mask(original_points, (height, width))
        expected_mask = polygon_mask(target_points, (height, width))
        control = np.empty((len(original_points) * 2, 2), dtype=np.float32)
        control[::2] = np.rint(original_points)
        control[1::2] = np.rint(target_points)
        np.save(case / 'integer-ui-control-points.npy', control)
        allowed = cv2.dilate((source_mask | expected_mask).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        save_mask(source_mask, case / 'source-mask.png')
        save_mask(expected_mask, case / 'expected-target-mask.png')
        save_mask(allowed, case / 'allowed-source-target-band.png')
        started = time.perf_counter()
        handles, targets, completion_mask = bi_warp(source_mask.astype(np.uint8), control, 5)
        warp_ms = (time.perf_counter() - started) * 1000
        warped = source.copy()
        warped[targets[:, 1], targets[:, 0]] = source[handles[:, 1], handles[:, 0]]
        actual_mask = np.zeros_like(source_mask)
        actual_mask[targets[:, 1], targets[:, 0]] = True
        save_mask(actual_mask, case / 'actual-target-mask.png')
        save_mask(completion_mask > 0, case / 'official-inpaint-mask.png')
        Image.fromarray(warped).save(case / 'official-warp.png')
        # Official preview displays a diagnostic grid on missing pixels. Keep
        # raw pixels for objective contracts; a separate grid visual matches UI.
        grid = np.full((height, width), 255, dtype=np.uint8)
        grid[::10] = 0; grid[:, ::10] = 0
        preview = warped.copy()
        preview[completion_mask > 0, :3] = grid[completion_mask > 0, None]
        Image.fromarray(preview).save(case / 'official-grid-preview.png')
        protected = warped.copy()
        protected[frozen | ~allowed] = source[frozen | ~allowed]
        Image.fromarray(protected).save(case / 'protected-warp.png')
        selected = (completion_mask > 0) & ~frozen & allowed
        save_mask(selected, case / 'protected-completion-mask.png')
        intersection = np.count_nonzero(actual_mask & expected_mask)
        union = np.count_nonzero(actual_mask | expected_mask)
        row = {
            'amount': amount, 'geometry_sha256': sha(geometry_path), 'native_output_sha256': sha(case / 'native.png'),
            'native_ms': round(native_ms, 2), 'official_warp_cpu_ms': round(warp_ms, 2),
            'source_contour_points': len(original_points), 'mapped_pixels': len(targets),
            'target_mask_iou': float(intersection / max(union, 1)),
            'missing_expected_pixels': int(np.count_nonzero(expected_mask & ~actual_mask)),
            'outside_expected_pixels': int(np.count_nonzero(actual_mask & ~expected_mask)),
            'official_completion_pixels': int(np.count_nonzero(completion_mask)),
            'protected_completion_pixels': int(np.count_nonzero(selected)),
            'max_unrounded_target_motion_px': float(np.linalg.norm(target_points - original_points, axis=1).max()),
            'contracts_official_warp': contracts(source, warped, frozen, allowed, eyes),
            'contracts_protected_warp': contracts(source, protected, frozen, allowed, eyes),
            'official_warp_sha256': sha(case / 'official-warp.png'),
        }
        comparisons = [source, native, protected]
        labels = ['Original', f'Native {amount}', f'Official warp + protection {amount}']
        if worker:
            filled, metadata = worker.inpaint(Image.fromarray(protected), selected.astype(np.uint8) * 255,
                max_side=512, context_mode='full', mask_dilation=0, seam_blend='none')
            filled_pixels = np.asarray(filled).copy()
            filled_pixels[frozen | ~allowed] = source[frozen | ~allowed]
            filled = Image.fromarray(filled_pixels)
            filled.save(case / 'protected-warp-lama.png')
            row['lama_baseline'] = metadata
            row['contracts_protected_lama'] = contracts(source, filled_pixels, frozen, allowed, eyes)
            row['lama_sha256'] = sha(case / 'protected-warp-lama.png')
            comparisons.append(filled_pixels); labels.append('Warp + LaMa baseline')
        assert row['contracts_protected_warp']['hair_ear_accessory_changed'] == 0
        assert row['contracts_protected_warp']['outside_source_target_band_changed'] == 0
        assert row['contracts_protected_warp']['alpha_changed'] == 0
        for name, box in {'face': (280, 160, 750, 665), 'left': (290, 285, 425, 585), 'right': (598, 285, 738, 585), 'jaw': (340, 400, 680, 660)}.items():
            montage(comparisons, labels, box, case / f'{name}-comparison-2x.png')
        if amount == 100 and PREVIOUS.exists():
            old = rgba(PREVIOUS / 'full-128-none.png')
            images = [source, native, old] + comparisons[2:]
            montage(images, ['Original', 'Native100', 'Previous disocclusion LaMa'] + labels[2:], (340, 400, 680, 660), case / 'previous-lama-comparison-2x.png')
            row['previous_lama_source_sha256'] = sha(PREVIOUS / 'candidate.png')
            row['previous_lama_output_sha256'] = sha(PREVIOUS / 'full-128-none.png')
        report['rows'].append(row)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'amount': amount, 'warp_ms': row['official_warp_cpu_ms'], 'target_iou': row['target_mask_iou'], 'raw_contracts': row['contracts_official_warp'], 'protected_contracts': row['contracts_protected_warp']}), flush=True)


if __name__ == '__main__':
    run()
