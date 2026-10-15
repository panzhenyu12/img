"""Independent read-only checks of retained V37 native artifacts.

For a fixed underlying scene B and opaque, bounded colors F0/F1/B in
[0,255], coefficients of a1*F1-a0*F0+(a0-a1)*B sum to zero. The sum of
positive coefficients is max(a0,a1), hence the per-channel physical edit
is bounded by 255*max(a0,a1), plus 1 byte of quantization. This is an
inequality of permitted behavior, not a reimplementation of the renderer.
Model alpha is not ground truth. The recorded target alpha is instrumented
renderer state, not an independently recovered physical matte.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = {
    'user': ('user-face-contour', 'test/results/user-face-contour-20261014/1024/canonical-source.png', [287,208,730,723]),
    'woman': ('portrait-woman', 'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png', [279,526,904,1288]),
    'hair': ('portrait-hair', 'test/results/face-core-review/portrait-hair-source.png', [168,170,420,492]),
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = HERE / 'independent-audit'
    out.mkdir(exist_ok=False)
    rows = []
    for name, (case, source, box) in CASES.items():
        path = ROOT / source
        original = np.asarray(Image.open(path).convert('RGBA'))
        assert (original[...,3] == 255).all(), 'bound assumes opaque input'
        seed_path = ROOT / 'experiments/joint-skin-matting-v1-20261015/prompts-v2' / case / 'crop-point/alphas-full.npy'
        raw = np.load(seed_path)[2]
        reports = {v: json.loads((HERE / v / name / 'report.json').read_text()) for v in ['outputs', 'outputs-delta']}
        for v, report in reports.items():
            assert report['sourceSha256'] == sha(path)
            assert report['modelAlphaSha256'] == sha(seed_path)
            frozen_binary = HERE / v / name / ('image-render-' + report['binarySha256'])
            assert sha(frozen_binary) == report['binarySha256']
            saved_alpha = np.fromfile(HERE / v / name / 'physical-alpha-f32le.bin', dtype='<f4').reshape(raw.shape)
            assert np.array_equal(saved_alpha, raw)
            zero = np.asarray(Image.open(HERE / v / name / '0/candidate.png').convert('RGBA'))
            assert np.array_equal(zero, original)
        for amount in [100, -100]:
            dirs = {v: HERE / v / name / str(amount) for v in reports}
            native = ROOT / 'experiments/joint-skin-matting-v1-20261015' / ('native-v2-' + name) / str(amount)
            for filename in ['flow-geometry.json', 'actual-source-width.json']:
                assert sha(dirs['outputs'] / 'debug' / filename) == sha(dirs['outputs-delta'] / 'debug' / filename) == sha(native / 'debug' / filename)
            for v in reports:
                recorded = next(r for r in reports[v]['rows'] if r['amount'] == amount)
                assert recorded['graphSha256'] == sha(native / 'graph.json')
                assert recorded['outputSha256'] == sha(dirs[v] / 'candidate.png')
            trace = dirs['outputs-delta'] / 'layers'
            state = json.loads((trace / 'state.json').read_text())
            w,h = state['width'],state['height']; x,y = state['origin']
            a = np.fromfile(trace / 'source-alpha-f32le.bin',dtype='<f4').reshape(h,w)
            b = np.fromfile(trace / 'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
            changed = np.fromfile(trace / 'changed.bin',dtype='uint8').reshape(h,w).astype(bool)
            assert np.array_equal(a, raw[y:y+h,x:x+w])
            assert np.isfinite(a).all() and np.isfinite(b).all()
            assert ((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1.000001)).all()
            max_alpha = np.maximum(a,b)
            low = max_alpha < .002
            assert low.sum() > 100, 'nonempty domain is required'
            row = {'case':name, 'amount':amount, 'source':source, 'sourceSha256':sha(path), 'geometryExact':True, 'alphaExact':True, 'lowCoveragePixels':int(low.sum()), 'instrumentedChangedLowCoveragePixels':int((low & changed).sum()), 'variants':{}}
            panels = [original]
            for v in reports:
                image = np.asarray(Image.open(dirs[v] / 'candidate.png').convert('RGBA'))
                panels.append(image)
                assert image.shape == original.shape and (image[...,3] == 255).all()
                outside = np.ones(image.shape[:2], dtype=bool); outside[y:y+h,x:x+w] = False
                assert np.array_equal(image[outside], original[outside])
                delta = np.abs(image[y:y+h,x:x+w,:3].astype(float) - original[y:y+h,x:x+w,:3].astype(float)).max(2)
                # The low-alpha domain is NOT restricted by renderer changed[].
                # Target alpha is omitted for unchanged pixels, where output
                # must be byte-identical and the weaker bound still applies.
                bound = 255 * max_alpha + 1
                interior = (a > .99) & (b > .99)
                info = {'lowCoverageFailures':int((low & (delta > bound)).sum()), 'lowCoverageMaxRGBChange':float(delta[low].max()), 'allROIboundFailures':int((delta > bound).sum()), 'nonzeroOpaqueInteriorPixels':int((interior & (delta > 0)).sum()), 'outsideROIChanged':0, 'changedPixels':int((delta > 0).sum())}
                assert info['nonzeroOpaqueInteriorPixels'] > 100, 'identity is not a valid edited output'
                if v == 'outputs-delta':
                    assert info['lowCoverageFailures'] == 0
                    assert info['allROIboundFailures'] == 0
                row['variants'][v] = info
            # Preserve source scale: every original pixel is one preview pixel.
            x0,y0,x1,y1=box; cw,ch=x1-x0,y1-y0
            canvas=Image.new('RGB',(cw*3,ch+28),'#20242a'); draw=ImageDraw.Draw(canvas)
            for i,(panel,title) in enumerate(zip(panels,['Source','Raw alpha / old compositor','Raw alpha / delta compositor'])):
                canvas.paste(Image.fromarray(panel).convert('RGB').crop(box),(i*cw,28));draw.text((i*cw+3,8),title,fill='white')
            canvas.save(out / f'{name}-{amount}-1x.png')
            rows.append(row)
    report={'scope':__doc__, 'acceptedAsAestheticFix':False, 'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
