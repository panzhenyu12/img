"""Capture the existing guard4 model prediction in the outer 4px ring.

The network input is identical to guard_probe.py: expanded target + dilation=0
is equivalent to original target + dilation=4. Original inner32 output is kept
byte-identical; this probe only exposes the previously discarded outer prediction.
No worker/core change, no parameter search, and no alpha/geometry modification.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services/background-worker'))
from worker import LamaWorker, DEFAULT_MODEL, EXPECTED_SHA256
from guard_probe import weight

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
def read(path): return np.asarray(Image.open(path).convert('RGBA'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', required=True, choices=['portrait-614810', 'portrait-762020'])
    args = parser.parse_args()
    base = ROOT / 'test/results/face-generalization-20261118' / args.source_case
    previous = base / 'context-guard-valid-20261119'
    neural = base / 'source-preparation/head-neural-scene'
    out = base / 'context-guard-outer-capture-20261119'
    out.mkdir(exist_ok=False)
    protocol = json.loads((previous / 'protocol.json').read_text())
    assert protocol['contextGuardPixels'] == 4 and protocol['sourceInnerBand'] == 32
    assert protocol['selected'], 'Reuse exactly the frozen valid tile set'
    assert sha(neural / 'source-crop.png') == protocol['sourceSHA256']
    assert sha(neural / 'source-inpaint-mask.png') == protocol['maskSHA256']
    assert sha(DEFAULT_MODEL) == EXPECTED_SHA256
    source = read(neural / 'source-crop.png')
    mask = np.asarray(Image.open(neural / 'source-inpaint-mask.png').convert('L')) > 0
    assert (source[:, :, 3] == 255).all(), 'Opaque source required for input equivalence'
    expanded = binary_dilation(mask, iterations=4)
    outer = expanded & ~mask
    h, w = mask.shape
    valid_weights = np.zeros((h, w), float)
    all_weights = np.zeros((h, w), float)
    for item in protocol['selected']:
        x, y, r, b = item['box']
        target = binary_dilation(mask[y:b, x:r], iterations=4)
        assert not target.all()
        wt = weight(item['box'])
        valid_weights[y:b, x:r] += wt * target
        all_weights[y:b, x:r] += wt
    uncovered = outer & (valid_weights == 0)
    report = dict(sourceCase=args.source_case, originalProtocolSHA256=sha(previous / 'protocol.json'),
        sourceSHA256=sha(neural / 'source-crop.png'), maskSHA256=sha(neural / 'source-inpaint-mask.png'),
        modelExpectedSHA256=EXPECTED_SHA256, tileSize=256, stride=128,
        expandedTargetPixels=4, modelMaskDilation=0, threads=2,
        inferenceInput='per-tile binary_dilation(original mask, iterations=4); identical to guard4',
        selected=protocol['selected'], sourceInnerBandPreserved=True, coreChanged=False,
        alphaInputChanged=False, geometryInputChanged=False,
        weights='same sin² weights and same frozen valid tiles; outer denominator excludes tiles not predicting that pixel',
        outerRingPixels=int(outer.sum()), uncoveredOuterPixels=int(uncovered.sum()),
        outerPixelsWithSomeTilesNotPredicting=int((outer & (valid_weights < all_weights - 1e-12)).sum()),
        status='protocol frozen')
    dump(out / 'protocol.json', report)
    if uncovered.any():
        Image.fromarray(np.uint8(uncovered) * 255).save(out / 'uncovered.png')
        raise RuntimeError('Fixed tile set does not cover expanded outer ring; no inference run')
    worker = LamaWorker(threads=2)
    accum = np.zeros((h, w, 3), float)
    rows = []
    started = time.monotonic()
    for item in protocol['selected']:
        i, box = item['originalTileIndex'], item['box']
        x, y, r, b = box
        original_mask = mask[y:b, x:r]
        target = binary_dilation(original_mask, iterations=4)
        tile = source[y:b, x:r]
        generated, meta = worker.inpaint(Image.fromarray(tile), target,
            max_side=256, mask_dilation=0, context_mode='full', seam_blend='none')
        result = np.asarray(generated.convert('RGBA'))
        assert np.array_equal(result[~target], tile[~target])
        assert np.array_equal(result[:, :, 3], tile[:, :, 3])
        cached = read(previous / f'guard4-tile-{i:02}.png')
        delta = np.abs(result.astype(int) - cached.astype(int))
        inner_delta = delta[original_mask]
        wt = weight(box) * target
        accum[y:b, x:r] += result[:, :, :3] * wt[:, :, None]
        Image.fromarray(result).save(out / f'expanded-tile-{i:02}.png')
        row = dict(originalTileIndex=i, box=box, originalMaskPixels=int(original_mask.sum()),
            innerRGBAChangedPixels=int(np.any(inner_delta, axis=1).sum()),
            innerMaxByteDelta=int(inner_delta.max()) if inner_delta.size else 0,
            knownOutsideExpandedChangedPixels=0, model=meta,
            outputSHA256=sha(out / f'expanded-tile-{i:02}.png'))
        rows.append(row)
        dump(out / 'partial-execution.json', rows)
        print(json.dumps({'tile': i, 'completed': len(rows), 'count': len(protocol['selected']),
            'innerChanged': row['innerRGBAChangedPixels'], 'innerMaxDelta': row['innerMaxByteDelta']}), flush=True)
    result = read(previous / 'guard4-scene.png').copy()
    prior_result = result.copy()
    result[outer, :3] = np.round(accum[outer] / valid_weights[outer, None]).astype(np.uint8)
    assert np.array_equal(result[mask], prior_result[mask])
    assert np.array_equal(result[~expanded], source[~expanded])
    Image.fromarray(result).save(out / 'guard4-outer-scene.png')
    Image.fromarray(np.uint8(outer) * 255).save(out / 'outer-prediction-coverage.png')
    Image.fromarray(np.uint8(expanded) * 255).save(out / 'expanded-write-mask.png')
    full = read(previous / 'guard4-full-scene.png').copy()
    before_full = full.copy()
    x, y, r, b = json.loads((neural / 'report.json').read_text())['sourceCropBox']
    full[y:b, x:r][outer] = result[outer]
    Image.fromarray(full).save(out / 'guard4-outer-full-scene.png')
    full_coverage = np.zeros(full.shape[:2], bool)
    full_coverage[y:b, x:r] = outer
    assert np.array_equal(full[~full_coverage], before_full[~full_coverage])
    Image.fromarray(np.uint8(full_coverage) * 255).save(out / 'outer-full-coverage.png')
    descriptor = json.loads((previous / 'guard4-descriptor.json').read_text())
    descriptor['image'] = str(out / 'guard4-outer-full-scene.png')
    dump(out / 'guard4-outer-descriptor.json', descriptor)
    sheet = Image.new('RGB', (3*w, h+24), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, a) in enumerate([('source', source), ('prior guard4 inner32', prior_result), ('same inner32 + captured outer4', result)]):
        sheet.paste(Image.fromarray(a).convert('RGB'), (i*w, 24))
        draw.text((i*w+4, 4), name, fill='black')
    sheet.save(out / 'source-review.png')
    probes = [(300,571),(302,571),(304,571),(318,580)] if args.source_case == 'portrait-614810' else []
    report.update(status='completed; paired alpha replay pending', rows=rows,
        elapsedSeconds=time.monotonic()-started, originalMaskOutputChangedPixels=0,
        outsideExpandedOutputChangedPixels=0, sourceOutsideOuterFullChangedPixels=0,
        maximumInnerTileByteDelta=max(row['innerMaxByteDelta'] for row in rows),
        innerTileChangedPixelsSum=sum(row['innerRGBAChangedPixels'] for row in rows),
        points=[dict(point=[xx, yy], inOuter=bool(full_coverage[yy, xx]),
            oldRGB=before_full[yy, xx, :3].tolist(), capturedRGB=full[yy, xx, :3].tolist())
            for xx, yy in probes])
    dump(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['rows','selected']}), flush=True)

if __name__ == '__main__': main()
