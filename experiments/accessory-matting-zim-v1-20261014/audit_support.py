"""Locate source-person disagreement; parser confidence is not ground truth."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from review_body_refinement import raster


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'test/results/accessory-matting-zim-v1-20261014/full-replay'
    out = base / 'low-person-audit'
    out.mkdir(exist_ok=True)
    source_path = ROOT / 'test/results/user-face-contour-20261014/1024/canonical-source.png'
    source = Image.open(source_path).convert('RGB')
    original = np.asarray(source)
    graph_path = base / 'faceWidth100.json'
    params = json.loads(graph_path.read_text())['operations'][0]['params']
    w, h = source.size
    person = raster(params['subjectMask'], w, h)
    skin = raster(params['semanticMasks']['skin'], w, h)
    report = dict(sourceSha256=sha(source_path), graphSha256=sha(graph_path),
                  runnerSha256=sha(Path(__file__)),
                  scope='Contraction changes where original person confidence < .01; neither automatic failure nor acceptance. Original-resolution review required.', rows=[])
    for amount in [30, 100]:
        path = base / f'faceWidth{amount}-physical.png'
        candidate = np.asarray(Image.open(path).convert('RGB'))
        diff = np.max(abs(candidate.astype(int) - original.astype(int)), axis=2)
        mask = (diff > 0) & (person < .01)
        labels, count = ndimage.label(mask, np.ones((3, 3)))
        row = dict(amount=amount, outputSha256=sha(path), pixels=int(mask.sum()),
                   maxChannelDifference=int(diff[mask].max()), components=[])
        for idx in range(1, count + 1):
            component = labels == idx
            yy, xx = np.nonzero(component)
            x, y, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
            area = int(component.sum())
            row['components'].append(dict(bounds=[x, y, right, bottom], pixels=area,
                maxChannelDifference=int(diff[component].max()), meanMaxChannelDifference=float(diff[component].mean()),
                maxSkin=float(skin[component].max()), meanSkin=float(skin[component].mean())))
            if area < 3:
                continue
            crop = (max(0, x - 15), max(0, y - 15), min(w, right + 15), min(h, bottom + 15))
            pw, ph = crop[2] - crop[0], crop[3] - crop[1]
            panel = Image.new('RGB', (pw * 3, ph + 20), '#222')
            draw = ImageDraw.Draw(panel)
            overlay = original.copy()
            overlay[component] = [255, 0, 255]
            for k, (label, pixels) in enumerate([('Original', original), ('Physical', candidate), ('Low person changes', overlay)]):
                panel.paste(Image.fromarray(pixels).crop(crop), (k * pw, 20))
                draw.text((k * pw + 3, 4), label, fill='white')
            panel.save(out / f'{amount}-component{idx}-1x.png')
            panel.resize((panel.width * 3, panel.height * 3), Image.Resampling.NEAREST).save(out / f'{amount}-component{idx}-3x.png')
        report['rows'].append(row)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
