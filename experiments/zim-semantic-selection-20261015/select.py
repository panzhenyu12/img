"""Select an object-consistent ZIM candidate before using its quality score.

ZIM's predicted quality is not the probability that a mask is the requested
semantic part. This experiment keeps the existing semantic rejection limits,
applies them to every candidate, and abstains if none pass. No edited image,
photo-specific coordinate, or visual preference enters selection.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = ROOT / 'experiments/hair-accessory-matting-zim-v2-20261015/outputs'


def choose(candidates, kind):
    decisions = []
    for c in candidates:
        reasons = []
        if c['positiveAlpha'] < .5:
            reasons.append('positive point excluded')
        if any(a > .1 for a in c['negativeAlpha']):
            reasons.append('negative point retained')
        if c['skinCoreRetainedFraction'] > .05:
            reasons.append('confident skin retained')
        if kind == 'hair' and c['parserCoreRetainedFraction'] < .65:
            reasons.append('hair core lost')
        decisions.append({'index': c['index'], 'eligible': not reasons, 'reasons': reasons})
    eligible = [c for c, d in zip(candidates, decisions) if d['eligible']]
    selected = max(eligible, key=lambda c: (c['score'], -c['index']))['index'] if eligible else None
    return selected, decisions


def main():
    out = HERE / 'outputs'
    out.mkdir(exist_ok=False)
    report = json.loads((RAW / 'report.json').read_text())
    result = {'scope': 'Source-only selection experiment; no production masks or images changed; semantic agreement is not alpha ground truth.',
              'scriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputReportSha256': hashlib.sha256((RAW / 'report.json').read_bytes()).hexdigest(),
              'selection': 'Unchanged rejection thresholds from ZIM-v2 applied to all candidates, then official quality score among eligible; abstain if empty.',
              'rows': []}
    for photo in report['images']:
        src = Image.open(ROOT / photo['source']).convert('RGB')
        rgb = np.asarray(src).astype('float32')
        for case in photo['rows']:
            selected, decisions = choose(case['alphas'], case['prompt']['kind'])
            directory = out / case['directory']
            directory.mkdir(parents=True)
            alpha_path = RAW / case['directory'] / 'alphas.npy'
            assert hashlib.sha256(alpha_path.read_bytes()).hexdigest() == case['alphaSha256']
            alphas = np.load(alpha_path)
            official = case['officialScoreArgmax']
            x0, y0, x1, y1 = photo['faceCrop']
            w, h = x1-x0, y1-y0
            panel = Image.new('RGB', (w*3, h+24), '#222222')
            draw = ImageDraw.Draw(panel)
            variants = [('Source', src), (f'Official {official}', Image.fromarray(np.clip(rgb*alphas[official,...,None], 0, 255).astype('uint8')))]
            if selected is not None:
                np.save(directory / 'selected-alpha.npy', alphas[selected])
                Image.fromarray(np.rint(alphas[selected]*255).astype('uint8')).save(directory / 'selected-alpha.png')
                variants.append((f'Semantic candidate {selected}', Image.fromarray(np.clip(rgb*alphas[selected,...,None], 0, 255).astype('uint8'))))
            for i, (label, im) in enumerate(variants):
                panel.paste(im.crop((x0,y0,x1,y1)), (i*w,24))
                draw.text((i*w+3,5), label, fill='white')
            if selected is None:
                draw.text((2*w+3,5), 'ABSTAIN: no compatible mask', fill='white')
            panel.save(directory / 'face-alpha-1x.png')
            row = {'photo': photo['name'], 'directory': case['directory'], 'sourceSha256': photo['sourceSha256'],
                   'alphaSha256': case['alphaSha256'], 'officialIndex': official, 'selectedIndex': selected,
                   'decisions': decisions, 'selectedMetrics': next((c for c in case['alphas'] if c['index']==selected), None)}
            result['rows'].append(row)
            print(json.dumps({k:row[k] for k in ['photo','directory','officialIndex','selectedIndex','selectedMetrics']}), flush=True)
    (out / 'report.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
