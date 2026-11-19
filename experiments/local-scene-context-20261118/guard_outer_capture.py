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
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services/background-worker'))
from worker import LamaWorker, DEFAULT_MODEL, EXPECTED_SHA256
from guard_probe import weight

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
def read(path): return np.asarray(Image.open(path).convert('RGBA'))

def direct_capture(source_case):
    """One guard4 run for a regression fixture without cached guard0/guard4.

    This is a paired-candidate regression, never the matched context ablation.
    Preserve the previous paired scene outside inner32 plus outer4 support.
    """
    base = ROOT/'test/results/face-generalization-20261118'/source_case
    neural = base/'source-preparation/head-neural-scene'
    out = base/'context-guard-direct-20261119'
    out.mkdir(exist_ok=False)
    binding = json.loads((neural/'report.json').read_text())
    spec = json.loads((base/'ordinary-rlib-spec.json').read_text())
    case = spec['cases'][0]
    assert sha(case['source']) == binding['sourceSHA256']
    assert sha(neural/'source-crop.png') == binding['inputSHA256']['source-crop.png']
    assert sha(neural/'source-inpaint-mask.png') == binding['inputSHA256']['source-inpaint-mask.png']
    assert sha(DEFAULT_MODEL) == EXPECTED_SHA256
    source = read(neural/'source-crop.png')
    mask = np.asarray(Image.open(neural/'source-inpaint-mask.png').convert('L')) > 0
    assert (source[:,:,3] == 255).all()
    h,w = mask.shape
    inner = mask & (distance_transform_edt(mask) <= 32)
    expanded = binary_dilation(mask,iterations=4)
    outer = expanded & ~mask
    accepted = inner | outer
    starts = lambda extent: sorted(set([*range(0,max(1,extent-256+1),128),max(0,extent-256)]))
    selected=[];excluded=[];weights=np.zeros((h,w),float)
    for y in starts(h):
        for x in starts(w):
            r,b=min(w,x+256),min(h,y+256)
            m=mask[y:b,x:r]
            if not m.any() or m.all() or not inner[y:b,x:r].any(): continue
            target=binary_dilation(m,iterations=4)
            box=[x,y,r,b]
            if target.all():
                excluded.append({'box':box,'reason':'no readable context after fixed guard'})
                continue
            selected.append(box)
            weights[y:b,x:r] += weight(box)*target
    uncovered=accepted&(weights==0)
    report=dict(sourceCase=source_case,status='frozen before inference',sourceSHA256=sha(case['source']),
        sourceCropSHA256=sha(neural/'source-crop.png'),maskSHA256=sha(neural/'source-inpaint-mask.png'),
        tileSize=256,stride=128,sourceInnerBand=32,outerBand=4,modelMaskDilation=0,threads=2,
        selected=selected,excluded=excluded,innerPixels=int(inner.sum()),outerPixels=int(outer.sum()),
        uncoveredAcceptedPixels=int(uncovered.sum()),inferenceCount=len(selected),
        scope='one paired candidate for regression; no zero-guard ablation',
        sourceOnly=True,geometryChanged=False,motionPermissionChanged=False,modelExpectedSHA256=EXPECTED_SHA256)
    dump(out/'protocol.json',report)
    if uncovered.any(): raise RuntimeError('Frozen direct guard cannot cover inner32+outer4')
    worker=LamaWorker(threads=2);accum=np.zeros((h,w,3),float);rows=[];start=time.monotonic()
    for i,box in enumerate(selected):
        x,y,r,b=box;m=mask[y:b,x:r];target=binary_dilation(m,iterations=4)
        generated,meta=worker.inpaint(Image.fromarray(source[y:b,x:r]),target,
            max_side=256,mask_dilation=0,context_mode='full',seam_blend='none')
        a=np.asarray(generated.convert('RGBA'))
        assert np.array_equal(a[~target],source[y:b,x:r][~target])
        accum[y:b,x:r] += a[:,:,:3]*(weight(box)*target)[:,:,None]
        Image.fromarray(a).save(out/f'expanded-tile-{i:02}.png')
        rows.append({'box':box,'model':meta});dump(out/'partial-execution.json',rows)
        print(json.dumps({'tile':i+1,'count':len(selected)}),flush=True)
    desc=json.loads(Path(case['sourcePlatePrior']).read_text())
    full=read(desc['image']).copy();old=full.copy()
    x,y,r,b=binding['sourceCropBox']
    crop=full[y:b,x:r]
    crop[accepted,:3]=np.round(accum[accepted]/weights[accepted,None]).astype(np.uint8)
    full_mask=np.zeros(full.shape[:2],bool);full_mask[y:b,x:r]=accepted
    assert np.array_equal(full[~full_mask],old[~full_mask])
    Image.fromarray(full).save(out/'guard4-full-scene.png')
    Image.fromarray(crop).save(out/'guard4-scene.png')
    Image.fromarray(np.uint8(full_mask)*255).save(out/'accepted-full-mask.png')
    Image.fromarray(np.uint8(outer)*255).save(out/'outer-mask.png')
    desc['image']=str(out/'guard4-full-scene.png');dump(out/'guard4-descriptor.json',desc)
    report.update(status='completed; regression replay pending',rows=rows,elapsedSeconds=time.monotonic()-start,
        outsideAcceptedPixelsChanged=0,outputSHA256=sha(out/'guard4-full-scene.png'))
    dump(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','selected','excluded']}),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', required=True, choices=['portrait-614810', 'portrait-762020', 'user-original'])
    parser.add_argument('--direct', action='store_true', help='Single fixed paired-candidate regression without cached guard ablation')
    args = parser.parse_args()
    if args.direct:
        direct_capture(args.source_case)
        return
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
