"""Visual diagnostic of fabric tangents and structured contour-strip failures.

The synthetic shirt is not a photographic quality benchmark. It makes the
fixed-core tangent and missing backdrop lines visible beside frozen real-photo
replays. Both renderers are checked before and after every subprocess.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--label', default='body-c1-v18')
    args = parser.parse_args()
    results = ROOT / 'test/results' / args.label
    qa = ROOT / 'doc/qa' / args.label
    results.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
    hashes = {key: sha(path) for key, path in binaries.items()}
    width, height = 480, 640
    source = Image.new('RGBA', (width, height), (46, 62, 72, 255))
    draw = ImageDraw.Draw(source)
    for y in range(0, height, 24):
        draw.line((0, y, width, y), fill=(150, 170, 175, 255), width=2)
    for y in range(75, 565):
        for x in range(85, 395):
            stripe = (x + y * .65) % 30 < 2.5 or (x - y * .45) % 42 < 2
            source.putpixel((x, y), (155, 159, 168, 255) if stripe else (213, 219, 223, 255))
    for y in range(110, 550, 60):
        draw.ellipse((237, y - 3, 243, y + 3), fill=(30, 45, 60, 255))
    source_path = results / 'fabric-source.png'
    source.save(source_path)
    # Half-resolution raster keeps the protocol's <=512 dimensions. This is
    # the identical mask used for the retained candidate evidence.
    alpha = [255 if 42 <= i % 240 < 198 and 37 <= i // 240 < 283 else 0
             for i in range(240 * 320)]
    params = {
        'bodyPart': 'waist',
        'region': {'x': .1, 'y': .08, 'width': .8, 'height': .84},
        'landmarks': [{'x': .5, 'y': .08}, {'x': .5, 'y': .5}, {'x': .5, 'y': .92}],
        'subjectMask': {'shape': 'raster', 'width': 240, 'height': 320, 'data': alpha},
        'adjustments': {'waist': 100},
    }
    graph = {'version': '1.0', 'sourceAssetId': 'local', 'operations': [
        {'id': 'fabric-waist', 'type': 'body.adjust', 'version': '1.0', 'enabled': True, 'params': params}]}
    graph_path = results / 'fabric-graph.json'
    graph_path.write_text(json.dumps(graph, separators=(',', ':')))
    panels = [(source, 'source')]
    outputs = {}
    for key, binary in binaries.items():
        assert sha(binary) == hashes[key]
        result = results / f'fabric-{key}.png'
        subprocess.run([str(binary), '--input', str(source_path), '--graph', str(graph_path),
                        '--output', str(result)], check=True, capture_output=True)
        assert sha(binary) == hashes[key]
        rendered = Image.open(result).convert('RGBA')
        panels.append((rendered, key))
        outputs[key] = sha(result)
    sheet = Image.new('RGB', (width * 3, height + 28), '#20252b')
    draw = ImageDraw.Draw(sheet)
    for col, (panel, label) in enumerate(panels):
        sheet.paste(panel.convert('RGB'), (width * col, 28))
        draw.text((width * col + 8, 8), label, fill='white')
    sheet.save(qa / 'fabric-100.png')
    (qa / 'fabric-provenance.json').write_text(json.dumps({
        'binaries_sha256': hashes, 'source_sha256': sha(source_path),
        'graph_sha256': sha(graph_path), 'outputs_sha256': outputs,
        'note': 'Synthetic diagnostic. Tangent improvement does not imply real-photo naturalness or an 8/10 score.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
