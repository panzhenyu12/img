"""Ablate observed-hair virtual context for local neural background completion.

The reflected context is an explicit hidden-background hypothesis, not observed
pixels or recovered truth. Final writes use the frozen disocclusion permission.
No source, face geometry, transported skin or production code is modified.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/face-background-context-20261015'))
from context import compose, montage, sha
from local import tiles
sys.path.insert(0, str(ROOT / 'services/background-worker'))
from worker import LamaWorker


def virtual_context(rgb, observed):
    """Reflect into independently allowed source hair; no other RGB is read.

    Nearest-source fallback is labelled separately, so texture replication and
    constant donor extensions cannot be confused with actual neural recovery.
    """
    if not observed.any():
        raise ValueError('No observed hair donors')
    yy, xx = np.indices(observed.shape)
    _, indices = distance_transform_edt(~observed, return_indices=True)
    qy, qx = indices
    ry, rx = 2*qy-yy, 2*qx-xx
    inside = (ry >= 0) & (ry < rgb.shape[0]) & (rx >= 0) & (rx < rgb.shape[1])
    ry, rx = ry.clip(0, rgb.shape[0]-1), rx.clip(0, rgb.shape[1]-1)
    reflected = inside & observed[ry, rx] & ~observed
    dy, dx = np.where(reflected, ry, qy), np.where(reflected, rx, qx)
    assert observed[dy, dx].all()
    # All observed pixels are identity donors, including irregular holes.
    assert np.array_equal(dy[observed], yy[observed])
    assert np.array_equal(dx[observed], xx[observed])
    return rgb[dy, dx].copy(), reflected, np.stack([dx, dy], -1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    material=ROOT/'experiments/face-material-fill-v1-20261015/woman-100-full'
    prior=ROOT/'experiments/face-background-context-20261015/woman-100'
    debug=material/'layers'
    state=json.loads((debug/'state.json').read_text());h,w=state['height'],state['width'];ox,oy=state['origin']
    provenance=json.loads((material/'provenance.json').read_text())
    source_path=Path(provenance['source']);source=np.asarray(Image.open(source_path).convert('RGBA'))
    crop=np.asarray(Image.open(debug/'source-crop.png').convert('RGBA'))
    base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'))
    bg=np.asarray(Image.open(debug/'background.png').convert('RGBA'))
    samples=np.fromfile(debug/'samples-f32le.bin','<f4').reshape(h,w,5)
    frozen=np.fromfile(debug/'frozen-u8.bin','uint8').reshape(h,w)>0
    target=np.asarray(Image.open(prior/'write-target.png'))>0
    observed=np.asarray(Image.open(prior/'hair-only/inference-mask.png'))==0
    context,reflected,donors=virtual_context(crop,observed)
    corrupted=crop.copy();rng=np.random.default_rng(421)
    corrupted[~observed,:3]=rng.integers(0,256,(int((~observed).sum()),3),dtype='uint8')
    corrupt_context,_,_=virtual_context(corrupted,observed)
    assert np.array_equal(corrupt_context,context), 'Excluded foreground leaked into context'
    Image.fromarray(context).save(out/'virtual-context.png')
    Image.fromarray((reflected*255).astype('uint8')).save(out/'reflected-context-domain.png')
    np.save(out/'source-donor-xy.npy',donors)
    report=dict(scope='One frozen source; explicit reflected observed-hair context hypothesis, no physical hidden-texture claim.',
        sourceSha256=sha(source_path),source=provenance['source'],sourceCrop=state,
        scriptSha256=sha(Path(__file__)),contextHelperSha256=sha(ROOT/'experiments/face-background-context-20261015/context.py'),
        frozenInputs={str(p.relative_to(ROOT)):sha(p) for p in [material/'candidate.png',material/'graph.json',debug/'samples-f32le.bin',debug/'background.png',prior/'write-target.png',prior/'hair-only/inference-mask.png']},
        modelSha256=sha(ROOT/'.models/background/big-lama.pt'),excludedRGBPerturbationContextDifferences=0,
        observedContextPixels=int(observed.sum()),syntheticReflectedContextPixels=int(reflected.sum()),
        syntheticNearestContextPixels=int((~observed&~reflected).sum()),rows=[])
    worker=LamaWorker(threads=3)
    for size in [256,512]:
        dest=out/str(size);dest.mkdir()
        boxes=tiles(target,size);accum=np.zeros((h,w,3),np.float64);weights=np.zeros((h,w),np.float64);tile_reports=[]
        window=np.maximum(np.hanning(size),1e-3);window=np.outer(window,window)
        # The new mask differs deliberately from earlier all-foreground-unknown
        # context: the filled virtual context conditions the missing real strip.
        inference_missing=binary_dilation(target,iterations=3)&~observed
        for index,(x0,y0,x1,y1) in enumerate(boxes):
            patch=context[y0:y1,x0:x1];missing=inference_missing[y0:y1,x0:x1]
            result,meta=worker.inpaint(Image.fromarray(patch),missing.astype('uint8')*255,max_side=size,context_mode='full',mask_dilation=0,seam_blend='none')
            result.save(dest/f'tile-{index:02}.png')
            conditioned=patch.copy();conditioned[missing,:3]=0;Image.fromarray(conditioned).save(dest/f'conditioning-{index:02}.png')
            accum[y0:y1,x0:x1]+=np.asarray(result)[:,:,:3]*window[:,:,None];weights[y0:y1,x0:x1]+=window
            tile_reports.append(dict(index=index,box=[x0,y0,x1,y1],meta=meta))
            print(json.dumps(dict(size=size,tile=index+1,total=len(boxes),inferenceMs=meta['inference_ms'])),flush=True)
        assert (weights[target]>0).all()
        for mode, rgb in [('neural',np.floor(accum/np.maximum(weights[:,:,None],1e-12)+.5).clip(0,255).astype('uint8')),('prior-only',context[:,:,:3])]:
            dst=dest/mode;dst.mkdir()
            layer=bg.copy();layer[target,:3]=rgb[target]
            Image.fromarray(layer).save(dst/'background-layer.png')
            result=base.copy();patch=result[oy:oy+h,ox:ox+w];new=compose(layer,samples[:,:,:4],samples[:,:,4]);patch[target]=new[target]
            changed=np.any(result!=base,axis=2);full_target=np.zeros(source.shape[:2],bool);full_target[oy:oy+h,ox:ox+w]=target
            assert not (changed&~full_target).any()
            assert np.array_equal(patch[frozen],base[oy:oy+h,ox:ox+w][frozen])
            assert np.array_equal(result[:,:,3],base[:,:,3])
            Image.fromarray(result).save(dst/'candidate.png')
            montage([source,base,result],['Source','V36',f'{size} {mode} virtual context'],dst/'left-1x.png',box=(300,480,600,1170))
            report['rows'].append(dict(size=size,mode=mode,tiles=tile_reports if mode=='neural' else [],directory=str(dst.relative_to(out)),changedPixels=int(changed.sum()),outsideTargetChanges=0,frozenChanges=0,alphaChanges=0,outputSha256=sha(dst/'candidate.png')))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
