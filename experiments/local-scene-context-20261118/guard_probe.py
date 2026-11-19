"""Fixed 4px readable-context guard, with matched tile-set zero-guard control.

No source write-mask, geometry, optical alpha, tile size/stride or inner band is
changed. Tiles with no readable context after dilation cannot be evaluated;
the zero-guard control uses the same surviving set and weights from cached tiles.
All output is new evidence, not a replacement of the failed first attempt.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'services/background-worker'))
from worker import LamaWorker, EXPECTED_SHA256, DEFAULT_MODEL

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
def read(path): return np.asarray(Image.open(path).convert('RGBA'))
def weight(box):
    x,y,r,b=box
    yy=np.sin(np.pi*(np.arange(b-y)+.5)/(b-y))**2
    xx=np.sin(np.pi*(np.arange(r-x)+.5)/(r-x))**2
    return yy[:,None]*xx[None,:]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-case',required=True,choices=['portrait-614810','portrait-762020']);args=parser.parse_args()
    base=ROOT/'test/results/face-generalization-20261118'/args.source_case
    original=base/'local-scene-context';neural=base/'source-preparation/head-neural-scene'
    out=base/'context-guard-valid-20261119';out.mkdir(exist_ok=False)
    protocol=json.loads((original/'protocol.json').read_text())
    source=read(neural/'source-crop.png');mask=np.asarray(Image.open(neural/'source-inpaint-mask.png').convert('L'))>0
    assert protocol['sourceSHA256']==sha(neural/'source-crop.png') and protocol['maskSHA256']==sha(neural/'source-inpaint-mask.png')
    assert protocol['tileSize']==256 and protocol['stride']==128 and protocol['sourceInnerBand']==32
    band=mask&(distance_transform_edt(mask)<=32);h,w=mask.shape
    keep=[];excluded=[];weights=np.zeros((h,w),float)
    for i,box in enumerate(protocol['boxes']):
        x,y,r,b=box;m=mask[y:b,x:r];inference_mask=binary_dilation(m,iterations=4)
        if inference_mask.all():excluded.append({'originalTileIndex':i,'box':box,'reason':'no readable context after fixed guard'})
        else:
            keep.append((i,box));weights[y:b,x:r]+=weight(box)
    uncovered=band&(weights==0)
    report=dict(sourceCase=args.source_case,tileSize=256,stride=128,sourceInnerBand=32,contextGuardPixels=4,
        selection='source-mask-only, reject windows with no readable context after scipy default binary_dilation(iterations=4)',
        writeMaskChanged=False,alphaInputChanged=False,geometryInputChanged=False,targetsRead=False,
        sourceSHA256=sha(neural/'source-crop.png'),maskSHA256=sha(neural/'source-inpaint-mask.png'),
        baselineProtocolSHA256=sha(original/'protocol.json'),modelExpectedSHA256=EXPECTED_SHA256,
        selected=[{'originalTileIndex':i,'box':box} for i,box in keep],excluded=excluded,
        innerBandPixels=int(band.sum()),uncoveredBandPixels=int(uncovered.sum()),status='protocol frozen')
    dump(out/'protocol.json',report)
    if uncovered.any():
        Image.fromarray(np.uint8(uncovered)*255).save(out/'uncovered.png');report['status']='inapplicable: original band not covered';dump(out/'report.json',report);print(json.dumps(report));return
    assert sha(DEFAULT_MODEL)==EXPECTED_SHA256
    worker=LamaWorker(threads=2);accum0=np.zeros((h,w,3),float);accum4=np.zeros_like(accum0);rows=[];start=time.monotonic()
    partial=base/'local-scene-context-guard'
    for i,box in keep:
        x,y,r,b=box;m=mask[y:b,x:r];image=source[y:b,x:r];wt=weight(box)
        cached0=original/f'tile-{i:02}.png';a0=read(cached0)
        assert a0.shape==image.shape and np.array_equal(a0[~m],image[~m])
        reused=False;cached4=partial/f'tile-{i:02}.png'
        if i==0 and cached4.is_file() and (partial/'protocol.json').is_file():
            old=json.loads((partial/'protocol.json').read_text())
            assert old['contextGuardPixels']==4 and old['sourceSHA256']==report['sourceSHA256'] and old['maskSHA256']==report['maskSHA256'] and old['boxes'][i]==box
            a4=read(cached4);reused=True;meta={'reuse':'same frozen source/mask/box/settings from interrupted first guard probe','model_sha256':EXPECTED_SHA256}
        else:
            generated,meta=worker.inpaint(Image.fromarray(image),m,max_side=256,mask_dilation=4,context_mode='full',seam_blend='none');a4=np.asarray(generated.convert('RGBA'))
        assert a4.shape==image.shape and np.array_equal(a4[~m],image[~m])
        accum0[y:b,x:r]+=a0[:,:,:3]*wt[:,:,None];accum4[y:b,x:r]+=a4[:,:,:3]*wt[:,:,None]
        Image.fromarray(a4).save(out/f'guard4-tile-{i:02}.png')
        row={'originalTileIndex':i,'box':box,'cached0SHA256':sha(cached0),'guard4SHA256':sha(out/f'guard4-tile-{i:02}.png'),'reusedGuard4':reused,'model':meta}
        rows.append(row);dump(out/'partial-execution.json',rows);print(json.dumps({'tile':i,'completed':len(rows),'count':len(keep),'reusedGuard4':reused}),flush=True)
    scenes={};initial=read(neural/'scene-generated.png')
    for name,accum in [('matched0',accum0),('guard4',accum4)]:
        result=initial.copy();result[band,:3]=np.round(accum[band]/weights[band,None]).astype(np.uint8)
        assert np.array_equal(result[~mask],source[~mask]);Image.fromarray(result).save(out/f'{name}-scene.png');scenes[name]=result
    report.update(status='completed; visual review required',rows=rows,knownPixelsChanged=0,elapsedSeconds=time.monotonic()-start)
    dump(out/'report.json',report)
    sheet=Image.new('RGB',(w*4,h+24),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,a) in enumerate([('source',source),('original all-tiles0',read(original/'scene-generated.png')),('same-valid-tiles0',scenes['matched0']),('same-valid-tiles4',scenes['guard4'])]):
        sheet.paste(Image.fromarray(a).convert('RGB'),(i*w,24));draw.text((i*w+4,4),name,fill='black')
    sheet.save(out/'source-review.png');print(json.dumps({'complete':True,'selected':len(keep),'excluded':len(excluded),'uncovered':int(uncovered.sum())}))
if __name__=='__main__':main()
