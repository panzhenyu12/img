"""Restrict new skin ownership to the old accessory disagreement region.

This is explicitly a semantic adapter, not an evaluation of raw neural alpha.
Frozen landmark/contour geometry and V21 binary are kept identical.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'test/results/accessory-matting-zim-v1-20261014'
    out = base / 'local-support-replay'
    out.mkdir(exist_ok=False)
    source = ROOT / 'test/results/user-face-contour-20261014/1024/canonical-source.png'
    source_graph = source.with_name('face-graph.json')
    seed = base / 'physical-support-replay/faceWidth100-physical-support.json'
    binary = Path(json.loads((base / 'physical-support-replay/manifest.json').read_text())['binary'])
    raw = ROOT / 'test/results/zim-20261014/face-dense-signed-prompts/alpha-all-f32.npy'
    g = json.loads(seed.read_text())
    original_graph = json.loads(source_graph.read_text())
    sem = original_graph['operations'][0]['params']['semanticMasks']
    skin = np.array(sem['skin']['data'], dtype=np.uint8).reshape(512, 512)
    old_accessory = np.array(sem['accessories']['data'], dtype=np.float32).reshape(512, 512) / 255
    alpha = np.asarray(Image.fromarray(np.load(raw, allow_pickle=False)[2]).resize((512, 512), Image.Resampling.BOX))
    # Source alpha only proposes skin where the old parser assigned accessory.
    # Removed accessory holes can remain background because face alpha is low.
    adapted = np.maximum(skin, np.rint(alpha * old_accessory * 255).astype(np.uint8))
    new_skin = copy.deepcopy(sem['skin'])
    new_skin['data'] = adapted.ravel().tolist()
    g['operations'][0]['params']['semanticMasks']['skin'] = new_skin
    assert np.all(adapted[old_accessory == 0] == skin[old_accessory == 0])
    report = dict(scope='Explicit source-semantic adapter: add skin only in old accessory support, keep old skin elsewhere. Ear/hair masks and geometry unchanged. New accessory source alpha is retained from combined seed; no warped-image redetection.',
                  sourceSha256=sha(source), originalGraphSha256=sha(source_graph), seedGraphSha256=sha(seed),
                  rawFaceSha256=sha(raw), runnerSha256=sha(Path(__file__)), binary=str(binary), binarySha256=sha(binary),
                  changedSkinRasterPixels=int((adapted != skin).sum()), rows=[])
    original = Image.open(source).convert('RGB')
    for amount in [30, 100]:
        g['operations'][0]['params']['adjustments'] = {'faceWidth': amount}
        graph_path = out / f'faceWidth{amount}.json'
        graph_path.write_text(json.dumps(g, separators=(',', ':')))
        dest = out / f'faceWidth{amount}.png'
        debug = out / f'debug{amount}'
        debug.mkdir()
        start = time.perf_counter()
        result = subprocess.run([str(binary), '--input', str(source), '--graph', str(graph_path), '--output', str(dest)], env={**os.environ, 'FACE_V21_DEBUG_DIR': str(debug)}, capture_output=True)
        (out / f'faceWidth{amount}.log').write_bytes(result.stdout + result.stderr)
        result.check_returncode()
        row = dict(amount=amount, graphSha256=sha(graph_path), outputSha256=sha(dest), processMs=(time.perf_counter() - start) * 1000)
        assert sha(binary) == report['binarySha256']
        report['rows'].append(row)
        panels = [('Original', original), ('V21 full support', Image.open(base / f'full-replay/faceWidth{amount}-physical.png')), ('V21 local support', Image.open(dest))]
        crop = (280, 160, 750, 700)
        pw, ph = crop[2] - crop[0], crop[3] - crop[1]
        sheet = Image.new('RGB', (pw * 3, ph + 24), '#222')
        draw = ImageDraw.Draw(sheet)
        for i, (label, im) in enumerate(panels):
            sheet.paste(im.convert('RGB').crop(crop), (i * pw, 24))
            draw.text((i * pw + 3, 5), label, fill='white')
        sheet.save(out / f'faceWidth{amount}-comparison-1x.png')
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
