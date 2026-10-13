"""Frozen A/B body replay with exact output and local visual evidence.

Five photographs, fifteen saved part graphs, symmetric slider strengths. A
performance-only candidate must be byte-identical. Contract checks are never
turned into aesthetic scores.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raster(mask, width, height):
    assert mask['shape'] == 'raster', 'Cached body contracts require raster masks'
    mw, mh = mask['width'], mask['height']
    data = np.asarray(mask['data'], dtype=np.float32).reshape(mh, mw) / 255
    x = np.clip((np.arange(width, dtype=np.float32) + .5) / width * mw - .5, 0, mw - 1)
    y = np.clip((np.arange(height, dtype=np.float32) + .5) / height * mh - .5, 0, mh - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, mw - 1), np.minimum(y0 + 1, mh - 1)
    tx, ty = x - x0, y - y0
    top = data[y0[:, None], x0] * (1 - tx) + data[y0[:, None], x1] * tx
    bottom = data[y1[:, None], x0] * (1 - tx) + data[y1[:, None], x1] * tx
    return top * (1 - ty[:, None]) + bottom * ty[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--ids', default='fashion-standing,fullbody-skirt,fullbody-side-fashion,tilted-shirt,action-suit')
    parser.add_argument('--amounts', default='-60,-30,30,60')
    args = parser.parse_args()
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
    hashes = {key: sha(value) for key, value in binaries.items()}
    amounts = [int(a) for a in args.amounts.split(',')]
    results = ROOT / 'test/results' / args.label
    qa = ROOT / 'doc/qa' / args.label
    results.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    rows = []
    for ident in args.ids.split(','):
        directory = ROOT / 'test/results/body-parts-v14' / ident
        source = directory / 'source.png'
        original = Image.open(source).convert('RGBA')
        input_pixels = np.asarray(original)
        w, h = original.size
        for graph_path in sorted(directory.glob('*.json')):
            if graph_path.stem in ('layers', 'quality'):
                continue
            graph = json.loads(graph_path.read_text())
            params = graph['operations'][0]['params']
            region = params['region']
            box = (max(0, int(region['x'] * w) - 16), max(0, int(region['y'] * h) - 16),
                   min(w, int((region['x'] + region['width']) * w) + 16),
                   min(h, int((region['y'] + region['height']) * h) + 16))
            protect = raster(params['protectMask'], w, h) > 1e-6
            subject = raster(params['subjectMask'], w, h)
            yy, xx = np.indices((h, w))
            outside = (((xx + .5) / w < region['x']) | ((xx + .5) / w > region['x'] + region['width'])
                       | ((yy + .5) / h < region['y']) | ((yy + .5) / h > region['y'] + region['height']))
            for amount in amounts:
                candidate = deepcopy(graph)
                candidate['operations'][0]['params']['adjustments'] = {key: amount for key in params['adjustments']}
                path = results / f'{ident}-{graph_path.stem}-{amount}.json'
                path.write_text(json.dumps(candidate, separators=(',', ':')))
                row = {'sample': ident, 'part': graph_path.stem, 'amount': amount,
                       'source_sha256': sha(source), 'parsing_graph_sha256': sha(graph_path),
                       'graph_sha256': sha(path)}
                panels = [(original, 'Original')]
                images = {}
                order = list(binaries) if len(rows) % 2 == 0 else list(reversed(binaries))
                for key in order:
                    binary = binaries[key]
                    assert sha(binary) == hashes[key]
                    result = results / f'{ident}-{graph_path.stem}-{amount}-{key}.png'
                    start = time.perf_counter()
                    subprocess.run([str(binary), '--input', str(source), '--graph', str(path), '--output', str(result)],
                                   check=True, capture_output=True)
                    row[key + '_total_ms'] = round((time.perf_counter() - start) * 1000, 2)
                    assert sha(binary) == hashes[key]
                    rendered = Image.open(result).convert('RGBA')
                    pixels = np.asarray(rendered)
                    changed = np.any(pixels != input_pixels, axis=2)
                    row[key + '_protected_changes'] = int(changed[protect].sum())
                    row[key + '_outside_changes'] = int(changed[outside].sum())
                    row[key + '_known_background_changes'] = int(changed[subject < .01].sum()) if amount >= 0 else None
                    row[key + '_alpha_changes'] = int((pixels[:, :, 3] != input_pixels[:, :, 3]).sum())
                    row[key + '_changed_pixels'] = int(changed.sum())
                    row[key + '_output_sha256'] = sha(result)
                    assert row[key + '_protected_changes'] == 0, row
                    assert row[key + '_outside_changes'] == 0, row
                    if amount >= 0:
                        assert row[key + '_known_background_changes'] == 0, row
                    if amount == 0:
                        assert not changed.any(), row
                    images[key] = (rendered, pixels)
                different = np.any(images['before'][1] != images['after'][1], axis=2)
                row['changed_vs_baseline'] = int(different.sum())
                if args.identity:
                    assert not different.any(), row
                panels += [(images[key][0], f'{key} / {amount}') for key in binaries]
                crops = [(image.convert('RGB').crop(box), label) for image, label in panels]
                cw, ch = crops[0][0].size
                sheet = Image.new('RGB', (cw * len(crops), ch + 28), '#161b22')
                draw = ImageDraw.Draw(sheet)
                for col, (crop, label) in enumerate(crops):
                    sheet.paste(crop, (col * cw, 28))
                    draw.text((col * cw + 7, 7), label, fill='white')
                evidence = qa / f'{ident}-{graph_path.stem}-{amount}.png'
                sheet.save(evidence)
                row['evidence'] = str(evidence.relative_to(ROOT))
                rows.append(row)
            print(ident, graph_path.stem, flush=True)
    report = {'binaries_sha256': hashes, 'binary_checks': 'before/after every subprocess',
              'amounts': amounts, 'expected_identity': args.identity, 'rows': rows,
              'timing_scope': 'native total includes PNG/launch, interleaved under concurrent development load; not performance acceptance',
              'note': 'Frozen v14 masks; no new detection. Contracts do not imply visual quality. Expansion may cover background without displacing it.'}
    (qa / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(rows), 'changed_vs_baseline': sum(r['changed_vs_baseline'] for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
