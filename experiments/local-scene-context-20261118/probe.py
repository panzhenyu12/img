"""One fixed source-context experiment, not a deployed background estimator.

Same model/mask/source as full-head LaMa; only inference context is tiled.
Accept generated RGB only in the 32px source-side boundary band, with partition
of unity overlap. Known pixels are exact. No edited image or motion is read.
"""
from pathlib import Path
import json,sys,time,hashlib,argparse
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'services/background-worker'))
from worker import LamaWorker
parser=argparse.ArgumentParser();parser.add_argument('--source-case',default='portrait-614810',choices=['portrait-614810','portrait-762020']);parser.add_argument('--context-guard',action='store_true');args=parser.parse_args()
BASE=ROOT/'test/results/face-generalization-20261118'/args.source_case
NEURAL=BASE/'source-preparation/head-neural-scene'
OUT=BASE/('local-scene-context-guard' if args.context_guard else 'local-scene-context');OUT.mkdir(exist_ok=False)
source=np.asarray(Image.open(NEURAL/'source-crop.png').convert('RGBA'))
mask=np.asarray(Image.open(NEURAL/'source-inpaint-mask.png').convert('L'))>0
original=np.asarray(Image.open(NEURAL/'scene-generated.png').convert('RGBA'))
h,w=mask.shape;distance=distance_transform_edt(mask);band=mask&(distance<=32)
TILE,STRIDE=256,128
starts=lambda extent: sorted(set([*range(0,max(1,extent-TILE+1),STRIDE),max(0,extent-TILE)]))
boxes=[]
for y in starts(h):
 for x in starts(w):
  m=mask[y:y+TILE,x:x+TILE]
  if m.any() and not m.all() and band[y:y+TILE,x:x+TILE].any():boxes.append((x,y,min(w,x+TILE),min(h,y+TILE)))
report={'contextGuardPixels':4 if args.context_guard else 0,'contextGuardRationale':'existing inpaint worker default; only model-readable context changes, write mask stays frozen','scope':'same source/model/mask; only crop context changes','tileSize':TILE,'stride':STRIDE,'sourceInnerBand':32,'boxes':boxes,'sourceSHA256':hashlib.sha256((NEURAL/'source-crop.png').read_bytes()).hexdigest(),'maskSHA256':hashlib.sha256((NEURAL/'source-inpaint-mask.png').read_bytes()).hexdigest(),'targetsRead':False}
(OUT/'protocol.json').write_text(json.dumps(report,indent=2))
worker=LamaWorker(threads=2);accum=np.zeros((h,w,3),float);weights=np.zeros((h,w),float);rows=[];start=time.monotonic()
for i,(x,y,r,b) in enumerate(boxes):
 image=source[y:b,x:r];m=mask[y:b,x:r]
 generated,meta=worker.inpaint(Image.fromarray(image),m,max_side=256,mask_dilation=4 if args.context_guard else 0,context_mode='full',seam_blend='none')
 a=np.asarray(generated.convert('RGBA'));assert np.array_equal(a[~m],image[~m])
 # Strictly positive sine weights avoid unobserved zero-weight image edges.
 yy=np.sin(np.pi*(np.arange(b-y)+.5)/(b-y))**2;xx=np.sin(np.pi*(np.arange(r-x)+.5)/(r-x))**2
 weight=yy[:,None]*xx[None,:]
 accum[y:b,x:r]+=a[:,:,:3]*weight[:,:,None];weights[y:b,x:r]+=weight
 generated.save(OUT/f'tile-{i:02}.png');rows.append({'box':[x,y,r,b],'model':meta})
 print(json.dumps({'tile':i+1,'count':len(boxes),'inferenceMs':meta.get('inference_ms')}),flush=True)
if np.any(band&(weights==0)):raise ValueError('uncovered source boundary band')
result=original.copy();result[band,:3]=np.round(accum[band]/weights[band,None]).astype(np.uint8)
assert np.array_equal(result[~mask],source[~mask])
Image.fromarray(result).save(OUT/'scene-generated.png')
report.update(rows=rows,elapsedSeconds=time.monotonic()-start,knownPixelsChanged=0,bandPixels=int(band.sum()),meaning='conditional hidden background hypothesis; tiling can introduce inconsistent structure and must be reviewed')
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['elapsedSeconds','bandPixels']}))
review=Image.new('RGB',(w*3,h+24),'white');draw=ImageDraw.Draw(review)
for i,(name,pixels) in enumerate([('source',source),('full-context LaMa',original),('fixed tiled source-context LaMa',result)]):
 review.paste(Image.fromarray(pixels).convert('RGB'),(i*w,24));draw.text((i*w+5,5),name,fill='black')
review.save(OUT/'source-review.png')
