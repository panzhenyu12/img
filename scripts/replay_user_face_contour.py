"""Freeze source-only detection and inspect user portrait contour artifacts.

Run diagnose_user_face_contour.mjs first. Image comparisons and mask contracts
are evidence for review; they do not constitute an aesthetic quality score.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'test/results/user-face-contour-20261014'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raster_coverage(mask, width, height):
    """Pixel-center bilinear sampling used by Rust PreparedMask."""
    data = np.asarray(mask['data'], dtype=np.float32).reshape(mask['height'], mask['width']) / np.float32(255)
    x = np.clip(((np.arange(width, dtype=np.float32) + np.float32(.5)) / np.float32(width)) * np.float32(mask['width']) - np.float32(.5), 0, mask['width'] - 1)
    y = np.clip(((np.arange(height, dtype=np.float32) + np.float32(.5)) / np.float32(height)) * np.float32(mask['height']) - np.float32(.5), 0, mask['height'] - 1)
    x0 = x.astype(int); y0 = y.astype(int)
    x1 = np.minimum(x0 + 1, mask['width'] - 1); y1 = np.minimum(y0 + 1, mask['height'] - 1)
    tx = x - x0.astype(np.float32); ty = y - y0.astype(np.float32)
    top = data[y0[:, None], x0] * (1 - tx) + data[y0[:, None], x1] * tx
    bottom = data[y1[:, None], x0] * (1 - tx) + data[y1[:, None], x1] * tx
    return np.clip(top * (1 - ty[:, None]) + bottom * ty[:, None], 0, 1)


def montage(images, labels, path, zoom=1):
    width, height = images[0].size
    canvas = Image.new('RGB', (width * zoom * len(images), height * zoom + 28), '#181c23')
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.convert('RGB').resize((width * zoom, height * zoom), Image.Resampling.NEAREST), (index * width * zoom, 28))
        draw.text((index * width * zoom + 6, 8), label, fill='white')
    canvas.save(path)


def main():
    provenance = json.loads((OUTPUT / 'provenance.json').read_text())
    binary = Path(provenance['baseline'])
    if sha(binary) != provenance['baselineSha256']:
        raise RuntimeError('Frozen native baseline SHA changed')
    cases = [('faceWidth', n) for n in [60, -60, 100, -100, 30, -30]]
    cases += [(key, n) for key in ['jaw', 'chin'] for n in [-100, 100, -60, 60]]
    cases += [('faceWidth', 0)]
    report = {'baselineSha256': provenance['baselineSha256'],
              'scope': 'One native baseline, original source-only fresh graph per resolution; full RGBA contracts and original-size edge crops. No edited-output redetection. Native timings include launch/PNG/JSON and concurrent development load.',
              'rows': [], 'maskReviews': []}
    for side in [1024, 768]:
        directory = OUTPUT / str(side)
        source = directory / 'canonical-source.png'
        base_graph = json.loads((directory / 'face-graph.json').read_text())
        original = Image.open(source).convert('RGBA')
        before = np.asarray(original)
        width, height = original.size
        scale = width / 1024
        crop_boxes = {name: tuple(round(v * scale) for v in box) for name, box in {
            'face': (280, 160, 750, 665), 'left': (290, 285, 425, 585),
            'right': (598, 285, 738, 585), 'chin': (375, 510, 640, 650)}.items()}
        baseline = directory / 'baseline'; baseline.mkdir(exist_ok=True)
        masks = base_graph['operations'][0]['params']['semanticMasks']
        coverages = {key: raster_coverage(mask, width, height) for key, mask in masks.items() if mask['shape'] == 'raster'}
        frozen = np.maximum.reduce([coverages[key] for key in ['ears', 'accessories', 'hair']]) >= .5
        face = coverages['faceContourMask']
        review = {'resolution': side, 'counts': {}, 'faceControlOverlapsFixed': int(np.count_nonzero((face >= .5) & frozen))}
        overlay_dir = directory / 'masks'; overlay_dir.mkdir(exist_ok=True)
        colors = {'faceContourMask': (0, 215, 255), 'face': (230, 0, 255), 'ears': (255, 70, 0), 'accessories': (255, 205, 0), 'hair': (100, 255, 40), 'neck': (255, 0, 90)}
        overlays = {}
        for key, color in colors.items():
            values = coverages[key]
            raw = masks[key]
            Image.fromarray(np.asarray(raw['data'], dtype=np.uint8).reshape(raw['height'], raw['width'])).save(overlay_dir / f'{key}-parser.png')
            Image.fromarray(np.rint(values * 255).astype(np.uint8)).save(overlay_dir / f'{key}-image-space.png')
            alpha = np.clip(values * .58, 0, .58)[..., None]
            pixels = np.rint(before[:, :, :3] * (1 - alpha) + np.asarray(color)[None, None, :] * alpha).astype(np.uint8)
            overlays[key] = Image.fromarray(pixels)
            overlays[key].save(overlay_dir / f'{key}-overlay.png')
            review['counts'][key] = {'greaterOrEqualHalf': int(np.count_nonzero(values >= .5)), 'nonzero': int(np.count_nonzero(values > 1 / 255))}
        for location, box in crop_boxes.items():
            if location == 'chin':
                continue
            montage([original.crop(box)] + [overlays[key].crop(box) for key in ['faceContourMask', 'ears', 'accessories', 'hair', 'neck']],
                    ['Original', 'Face control cyan', 'Ears red', 'Accessories yellow', 'Hair green', 'Neck magenta'],
                    overlay_dir / f'{location}-classes-2x.png', zoom=2)
        report['maskReviews'].append(review)
        for key, amount in cases:
            graph = json.loads(json.dumps(base_graph))
            graph['operations'][0]['params']['adjustments'] = {key: amount}
            graph_path = baseline / f'{key}{amount}.json'
            graph_path.write_text(json.dumps(graph))
            output = baseline / f'{key}{amount}.png'
            if sha(binary) != provenance['baselineSha256']:
                raise RuntimeError('Frozen native baseline SHA changed during replay')
            start = time.perf_counter()
            result = subprocess.run([str(binary), '--input', str(source), '--graph', str(graph_path), '--output', str(output)], capture_output=True, text=True)
            elapsed = (time.perf_counter() - start) * 1000
            if result.returncode:
                raise RuntimeError(result.stderr)
            edited = Image.open(output).convert('RGBA')
            after = np.asarray(edited)
            changed = np.any(before[:, :, :3] != after[:, :, :3], axis=2)
            row = {'resolution': side, 'parameter': key, 'amount': amount, 'changedPixels': int(changed.sum()),
                   'fixedClassChangedPixels': int(np.count_nonzero(changed & frozen)),
                   'alphaChangedPixels': int(np.count_nonzero(before[:, :, 3] != after[:, :, 3])),
                   'nativeTotalMs': elapsed, 'inputSha256': sha(source), 'graphSha256': sha(graph_path), 'outputSha256': sha(output)}
            for location, box in crop_boxes.items():
                montage([original.crop(box), edited.crop(box)], ['Original', f'{key} {amount}'], baseline / f'{key}{amount}-{location}-2x.png', zoom=2)
            report['rows'].append(row)
            (OUTPUT / 'baseline-report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    print(json.dumps({'completedCases': len(report['rows']), 'fixedClassChanges': sum(row['fixedClassChangedPixels'] for row in report['rows']), 'alphaChanges': sum(row['alphaChangedPixels'] for row in report['rows'])}), flush=True)


if __name__ == '__main__':
    main()
