"""Compare bounded exemplar search against a frozen native baseline.

Requires NumPy/Pillow. Benchmarks include graph parsing and mask compositing,
but exclude image I/O, detection and models. Pixel equality is an acceptance
condition; these timings do not establish aesthetic quality or browser speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_pixels(points, width, height):
    yy, xx = np.indices((height, width), dtype=np.float32)
    xx = (xx + np.float32(.5)) / np.float32(width)
    yy = (yy + np.float32(.5)) / np.float32(height)
    inside = np.zeros((height, width), dtype=bool)
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        ax, ay, bx, by = map(np.float32, (a['x'], a['y'], b['x'], b['y']))
        if ay != by:
            inside ^= ((ay > yy) != (by > yy)) & (xx < (bx - ax) * (yy - ay) / (by - ay) + ax)
    return inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'test/results/inpaint-efficiency-v1')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binaries = {name: path.resolve() for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    hashes = {name: sha(path) for name, path in binaries.items()}
    rows = []
    data = ROOT / 'doc/qa/background-lama'
    for sid in ['street', 'crowd', 'architecture', 'street-768-polygon']:
        # Use the established real-photo corpus and identical canonical masks.
        # The architecture source is an explicitly synthetic obstruction.
        source_id = 'street' if sid == 'street-768-polygon' else sid
        image = Image.open(data / f'{source_id}-source.png').convert('RGBA')
        selected = Image.open(data / f'{source_id}-mask.png').convert('L')
        protection_file = data / f'{source_id}-protect.png'
        protection = Image.open(protection_file).convert('L') if protection_file.exists() else Image.new('L', image.size)
        # Public raster masks are bounded to 512px per axis. Use one mask
        # texel per prepared photo pixel to make the edit-contract check exact.
        ratio = 1 if sid == 'street-768-polygon' else min(1, 512 / max(image.size))
        size = tuple(round(v * ratio) for v in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
        selected = selected.resize(size, Image.Resampling.NEAREST)
        protection = protection.resize(size, Image.Resampling.NEAREST)
        source = args.out / f'{sid}-source.png'
        image.save(source)
        mask = np.asarray(selected) > 0
        protect = np.asarray(protection) > 0
        original = np.asarray(image)
        params = {'method': 'exemplar', 'radius': .008, 'iterations': 32,
                  'mask': {'shape': 'raster', 'width': size[0], 'height': size[1], 'data': (mask.astype(np.uint8) * 255).ravel().tolist()},
                  'protectMask': {'shape': 'raster', 'width': size[0], 'height': size[1], 'data': (protect.astype(np.uint8) * 255).ravel().tolist()}}
        if sid == 'street-768-polygon':
            params = json.loads((data / 'street-exemplar-graph.json').read_text())['operations'][0]['params']
            params.update({'radius': .008, 'iterations': 32})
            mask = polygon_pixels(params['mask']['points'], *size)
            region = params['protectMask']['region']
            yy, xx = np.indices((size[1], size[0]), dtype=np.float32)
            xx = (xx + np.float32(.5)) / np.float32(size[0])
            yy = (yy + np.float32(.5)) / np.float32(size[1])
            x, y, w, h = (np.float32(region[k]) for k in ['x', 'y', 'width', 'height'])
            protect = (xx >= x) & (xx <= x + w) & (yy >= y) & (yy <= y + h)
        graph = args.out / f'{sid}-graph.json'
        graph.write_text(json.dumps({'operations': [{'type': 'repair.inpaint', 'params': params}]}) + '\n')
        row = {'sample': sid, 'width': size[0], 'height': size[1], 'selected_pixels': int((mask & ~protect).sum()),
               'original_source_sha256': sha(data / f'{source_id}-source.png'), 'prepared_source_sha256': sha(source), 'graph_sha256': sha(graph)}
        outputs = {}
        for name, binary in binaries.items():
            assert sha(binary) == hashes[name]
            output = args.out / f'{sid}-{name}.png'
            run = subprocess.run([str(binary), str(source), str(graph), str(output)], check=True, capture_output=True, text=True)
            row[name] = json.loads(run.stdout)
            pixels = np.asarray(Image.open(output).convert('RGBA'))
            changed = np.any(pixels != original, axis=2)
            row[name].update({'output_sha256': sha(output), 'outside_changes': int(changed[~mask].sum()),
                              'protection_changes': int(changed[protect].sum()), 'alpha_changes': int((pixels[:, :, 3] != original[:, :, 3]).sum())})
            assert row[name]['outside_changes'] == row[name]['protection_changes'] == row[name]['alpha_changes'] == 0
            outputs[name] = pixels
        row['rgba_byte_differences'] = int(np.count_nonzero(outputs['baseline'] != outputs['candidate']))
        assert row['rgba_byte_differences'] == 0, row
        row['hot_speedup'] = row['baseline']['hot_median_ms'] / row['candidate']['hot_median_ms']
        ys, xs = np.where(mask)
        crop = (max(0, int(xs.min()) - 25), max(0, int(ys.min()) - 25), min(size[0], int(xs.max()) + 26), min(size[1], int(ys.max()) + 26))
        panels = [('Original', image), ('Baseline', Image.fromarray(outputs['baseline'])), ('Candidate (identical)', Image.fromarray(outputs['candidate']))]
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        comparison = Image.new('RGB', (cw * 3, ch + 24), 'white')
        draw = ImageDraw.Draw(comparison)
        for i, (name, panel) in enumerate(panels):
            draw.text((i * cw + 4, 5), name, fill='black')
            comparison.paste(panel.crop(crop), (i * cw, 24))
        comparison.save(args.out / f'{sid}-comparison.png')
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert all(sha(path) == hashes[name] for name, path in binaries.items())
    report = {'scope': 'same-process native graph parsing, validation, exemplar fill and compositing; excludes image I/O, detection, model loading and browser/WASM',
              'binaries_sha256': hashes, 'inpaint_source_sha256': sha(ROOT / 'core/image-core/src/inpaint.rs'), 'rows': rows,
              'note': 'Exact output optimization. No visual score increased. Other concurrent CPU work may affect timings; rerun in isolation before making performance promises.'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
