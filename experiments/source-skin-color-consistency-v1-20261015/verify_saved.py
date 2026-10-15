"""Read-only verification of saved v3 outputs, independent of its runner.

No rerender, no threshold search. It checks persisted arrays and decoded PNGs,
then emits compact native 1x three-column review panels.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'outputs-v3-final-b'
QA = HERE/'qa/final-verification'
QA.mkdir(exist_ok=True)
TRACE = ROOT/'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-trace-inverse'
BONLY = ROOT/'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-scene-b-only-v2-feather'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = json.loads((OUT/'report.json').read_text())
source_path = ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
source = np.asarray(Image.open(source_path).convert('RGBA'))
scene_path = ROOT/'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz'
scene = np.load(scene_path)
finite = np.isfinite(scene['sceneDistance'])
opaque = np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz')['opaque']
frozen_inputs = np.load(HERE/'outputs-v2-monotonic/source-solve.npz')
assert sha(source_path) == report['sourceSha256']
assert sha(scene_path) == report['sceneSha256']
assert sha(HERE/'pipeline.py') == report['pipelineSha256']
assert sha(HERE/'solver.py') == report['solverSha256']
rows = []

for row in report['rows']:
    amount = row['amount']; trace = TRACE/str(amount if amount else 100); layer = trace/'layers'
    state = json.loads((layer/'state.json').read_text())
    x, y = state['origin']; w, h = state['width'], state['height']; crop = np.s_[y:y+h, x:x+w]
    output_path = OUT/str(amount)/'candidate.png'
    candidate = np.asarray(Image.open(output_path).convert('RGBA'))
    baseline = np.asarray(Image.open(BONLY/str(amount)/'candidate.png').convert('RGBA'))
    solved = np.load(OUT/str(amount)/'source-solve.npz')
    raw_alpha = np.fromfile(layer/'source-alpha-f32le.bin', dtype='<f4').reshape(h,w)
    native_f = np.asarray(Image.open(layer/'foreground.png').convert('RGB'))
    actual_b = np.load(BONLY/str(amount)/'background-replaced.npy')
    c = source[crop][..., :3].astype(float)
    a = raw_alpha.astype(float)[..., None]
    old_f = native_f.astype(float); new_f = solved['correctedForegroundByte'].astype(float)
    before = np.abs(c-a*old_f-(1-a)*actual_b)
    after = np.abs(c-a*new_f-(1-a)*actual_b)
    source_changed = np.any(new_f != old_f, axis=-1)
    worsened = source_changed & np.any(after > before+1e-6, axis=-1)
    frozen = np.fromfile(layer/'frozen.bin', dtype='uint8').reshape(h,w)>0
    changed = np.fromfile(layer/'changed.bin', dtype='uint8').reshape(h,w)>0 if amount else np.ones((h,w),bool)
    allowed = np.zeros(source.shape[:2],bool)
    allowed[crop] = finite[crop] & changed & ~frozen & ~opaque[crop]
    output_changed = np.any(candidate != baseline, axis=-1)
    checks = {
        'sourceAlphaExact': np.array_equal(raw_alpha, solved['sourceAlpha']) and np.array_equal(raw_alpha, frozen_inputs['alpha']),
        'originalForegroundExact': np.array_equal(native_f, solved['sourceForegroundByte']),
        'finalBackgroundExact': np.array_equal(actual_b, solved['finalBackgroundByte']),
        'sourceDomainExact': np.array_equal(solved['domain'], frozen_inputs['domain']),
        'sourceCorrectionInsideFrozenDomain': not (source_changed & ~frozen_inputs['domain']).any(),
        'finalBackgroundFitNonWorseningAfterByteQuantization': not worsened.any(),
        'outputOutsideWriteExact': not (output_changed & ~allowed).any(),
        'frozenOutputExact': not output_changed[crop][frozen].any(),
        'opaqueOutputExact': np.array_equal(candidate[opaque], baseline[opaque]),
        'outputAlphaExact': np.array_equal(candidate[...,3], baseline[...,3]),
        'sourceAlphaHashExact': sha(layer/'source-alpha-f32le.bin') == row['sourceAlphaSha256'],
        'geometryHashExact': sha(trace/'debug/flow-geometry.json') == row['geometrySha256'],
        'inverseHashExact': sha(layer/'inverse-source-xy-f32le.bin') == row['inverseSha256'],
        'outputHashExact': sha(output_path) == row['outputSha256'],
        'zeroExact': bool(np.array_equal(candidate, source)) if not amount else True,
    }
    assert all(checks.values()), (amount, checks)
    rows.append({'amount': amount, 'checks': {k: bool(v) for k,v in checks.items()},
                 'sourceChangedPixels': int(source_changed.sum()), 'worsenedSourceFitPixels': int(worsened.sum()),
                 'outputChangedFromBonly': int(output_changed.sum())})
    box = (287,197,730,709); cw, ch = box[2]-box[0], box[3]-box[1]
    panel = Image.new('RGB', (3*cw,ch+24), '#20242a'); draw = ImageDraw.Draw(panel)
    for i,(im,label) in enumerate([(source,'Source'),(baseline,'B only'),(candidate,'F v3 / exact final B')]):
        panel.paste(Image.fromarray(im).convert('RGB').crop(box),(cw*i,24))
        draw.text((cw*i+2,5),label,fill='white')
    panel.save(QA/f'{amount}-face-native-1x.png')

result = {'accepted': False, 'allChecksPassed': True, 'rows': rows}
(QA/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
