"""V3: one correction only, solve/check F using the exact final compositor B.

No alpha, geometry, gate, neighborhood or transition changes from v2.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

from compositor import compose, sample_f32
from pipeline import correct_foreground

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'outputs-v3-final-b'
OUT.mkdir(exist_ok=False)
TRACE = ROOT/'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-trace-inverse'
BONLY = ROOT/'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-scene-b-only-v2-feather'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
source_path = ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
scene_path = ROOT/'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz'
accessory_path = ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz'
source = np.asarray(Image.open(source_path).convert('RGBA'))
finite = np.isfinite(np.load(scene_path)['sceneDistance'])
opaque = np.load(accessory_path)['opaque']
saved = np.load(HERE/'outputs-v2-monotonic/source-solve.npz')
v2_report = json.loads((HERE/'outputs-v2-monotonic/report.json').read_text())
report = {'scope': __doc__, 'accepted': False, 'sourceSha256': sha(source_path),
          'sceneSha256': sha(scene_path), 'solverSha256': sha(HERE/'solver.py'),
          'pipelineSha256': sha(HERE/'pipeline.py'), 'runnerSha256': sha(Path(__file__)),
          'frozenInputSha256': sha(HERE/'outputs-v2-monotonic/source-solve.npz'),
          'algorithm': v2_report['algorithm'], 'rows': []}
report['algorithm']['foregroundBackgroundContract'] = 'solve and monotonic check use exact final background-replaced.npy'


def panel(path, images, labels, box):
    x0, y0, x1, y1 = box; w, h = x1-x0, y1-y0
    output = Image.new('RGB', (w*len(images), h+24), '#20242a')
    draw = ImageDraw.Draw(output)
    for i, (im, label) in enumerate(zip(images, labels)):
        output.paste(Image.fromarray(im).convert('RGB').crop(box), (w*i, 24))
        draw.text((w*i+2, 5), label, fill='white')
    output.save(path)
    output.resize((output.width*4, output.height*4), Image.Resampling.NEAREST).save(path.with_name(path.stem.replace('1x', '4x')+'.png'))


for amount in [100, -100, 0]:
    destination = OUT/str(amount); destination.mkdir()
    trace = TRACE/str(amount if amount else 100); layer = trace/'layers'
    state = json.loads((layer/'state.json').read_text())
    ox, oy = state['origin']; w, h = state['width'], state['height']
    region = np.s_[oy:oy+h, ox:ox+w]
    alpha = np.fromfile(layer/'source-alpha-f32le.bin', dtype='<f4').reshape(h, w)
    target = np.fromfile(layer/'target-alpha-f32le.bin', dtype='<f4').reshape(h, w)
    xy = np.fromfile(layer/'inverse-source-xy-f32le.bin', dtype='<f4').reshape(h, w, 2)
    samples = np.fromfile(layer/'samples-f32le.bin', dtype='<f4').reshape(h, w, 5)
    changed = np.fromfile(layer/'changed.bin', dtype='uint8').reshape(h, w) > 0
    frozen = np.fromfile(layer/'frozen.bin', dtype='uint8').reshape(h, w) > 0
    source_byte = np.asarray(Image.open(layer/'source-crop.png').convert('RGB')).astype('float32')
    foreground = np.asarray(Image.open(layer/'foreground.png').convert('RGB')).astype('float32')
    final_b_path = BONLY/str(amount)/'background-replaced.npy'
    background = np.load(final_b_path).astype('float32')
    assert np.array_equal(alpha, saved['alpha'])
    assert np.array_equal(foreground, saved['sourceForegroundByte'])
    result = correct_foreground(source_byte.astype(float)/255, alpha.astype(float),
                                foreground.astype(float)/255, background.astype(float)/255,
                                saved['skinCore'], saved['allowed'], saved['transition'],
                                radius=report['algorithm']['radiusPx'])
    corrected = result['correctedForegroundByte']
    source_changed = np.any(corrected != foreground, axis=-1)
    residual_before = np.abs(source_byte.astype(float)-alpha[..., None].astype(float)*foreground-(1-alpha[..., None].astype(float))*background)
    residual_after = np.abs(source_byte.astype(float)-alpha[..., None].astype(float)*corrected-(1-alpha[..., None].astype(float))*background)
    worse = source_changed & np.any(residual_after > residual_before+1e-6, axis=-1)
    assert not worse.any(), (amount, int(worse.sum()))
    np.savez_compressed(destination/'source-solve.npz', **result, sourceForegroundByte=foreground,
                        sourceAlpha=alpha, finalBackgroundByte=background, sourceChanged=source_changed)
    Image.fromarray(corrected.astype('uint8')).save(destination/'source-foreground-corrected.png')
    base = np.asarray(Image.open(BONLY/str(amount)/'candidate.png').convert('RGBA'))
    v2 = np.asarray(Image.open(HERE/'outputs-v2-monotonic'/str(amount)/'candidate.png').convert('RGBA'))
    v37 = np.asarray(Image.open(trace/'candidate.png').convert('RGBA')) if amount else source
    if amount:
        sampled_base, target_base = sample_f32(foreground, alpha, xy)
        sample_diff = float(np.max(abs(sampled_base[changed]-samples[..., :3][changed])))
        target_diff = float(np.max(abs(target_base[changed]-target[changed])))
        assert sample_diff == target_diff == 0
    else:
        yy, xx = np.indices(alpha.shape); xy = np.stack([xx, yy], -1).astype('float32')
        target = alpha.copy(); changed = np.ones_like(changed); sample_diff = target_diff = 0
    sampled_new, target_new = sample_f32(corrected, alpha, xy)
    assert np.array_equal(target_new[changed], target[changed])
    output = base.copy(); write = finite[region] & changed & ~frozen & ~opaque[region]
    rendered = compose(source_byte, corrected, background, alpha, target, sampled_new)
    patch = output[region][..., :3].copy(); patch[write] = rendered[write]; output[region][..., :3] = patch
    write_full = np.zeros(source.shape[:2], bool); write_full[region] = write
    diff = np.any(output != base, axis=-1)
    assert not (diff & ~write_full).any()
    assert not diff[region][frozen].any()
    assert np.array_equal(output[opaque], base[opaque])
    assert np.array_equal(output[..., 3], base[..., 3])
    if not amount: assert np.array_equal(output, source)
    Image.fromarray(output).save(destination/'candidate.png')
    Image.fromarray(diff.astype('uint8')*255).save(destination/'changed-from-b-only.png')
    for name, box in [('face', [287,197,730,709]), ('left', [320,400,420,550]), ('right', [590,400,700,550])]:
        panel(destination/f'{name}-1x.png', [source, v37, base, v2, output],
              ['Source', 'Frozen V37', 'B only', 'F v2', 'F v3 final B'], box)
    report['rows'].append({'amount': amount, 'sourceDomainPixels': int(result['domain'].sum()),
        'sourceChangedPixels': int(source_changed.sum()), 'nonWorseningRefusedPixels': int(result['nonWorseningRefused'].sum()),
        'sourceFitWorsenedPixelsAfterQuantization': int(worse.sum()), 'outputChangedFromBonly': int(diff.sum()),
        'outputChangedFromV2': int(np.any(output != v2, axis=-1).sum()), 'sampleBaseFloatMaxDiff': sample_diff,
        'targetAlphaMaxDiff': target_diff, 'outsideWriteChanged': 0, 'frozenChanged': 0,
        'opaqueChanged': 0, 'outputAlphaChanged': 0, 'zeroExact': bool(np.array_equal(output, source)) if not amount else None,
        'sourceAlphaSha256': sha(layer/'source-alpha-f32le.bin'), 'inverseSha256': sha(layer/'inverse-source-xy-f32le.bin'),
        'geometrySha256': sha(trace/'debug/flow-geometry.json'), 'finalBackgroundSha256': sha(final_b_path),
        'outputSha256': sha(destination/'candidate.png')})

(OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
