"""Replay the frozen local scene hypothesis; report its exposure coverage."""
from pathlib import Path
import json,subprocess,argparse
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--source-case',default='portrait-614810',choices=['portrait-614810','portrait-762020']);parser.add_argument('--context-guard',action='store_true');args=parser.parse_args()
BASE=ROOT/'test/results/face-generalization-20261118'/args.source_case;OUT=BASE/('local-scene-context-guard' if args.context_guard else 'local-scene-context')
spec=json.loads((BASE/'ordinary-rlib-spec.json').read_text());c=spec['cases'][0]
descriptor=json.loads(Path(c['sourcePlatePrior']).read_text());old=np.asarray(Image.open(descriptor['image']).convert('RGBA'));prior=old.copy()
left,top,right,bottom=json.loads((BASE/'source-preparation/head-neural-scene/report.json').read_text())['sourceCropBox']
mask=np.asarray(Image.open(BASE/'source-preparation/head-neural-scene/source-inpaint-mask.png').convert('L'))>0
band=mask&(distance_transform_edt(mask)<=32)
new=np.asarray(Image.open(OUT/'scene-generated.png').convert('RGBA'))
prior[top:bottom,left:right][band]=new[band]
Image.fromarray(prior).save(OUT/'full-source-scene.png');descriptor['image']=str(OUT/'full-source-scene.png')
(OUT/'descriptor.json').write_text(json.dumps(descriptor));c['sourcePlatePrior']=str(OUT/'descriptor.json');c['amounts']=[0,30,60,-30,-60]
spec['outputDir']=str(OUT/'replay');(OUT/'spec.json').write_text(json.dumps(spec))
result=subprocess.run([str(ROOT/'core/image-core/target/release/examples/face-source-replay'),str(OUT/'spec.json')],capture_output=True,text=True)
(OUT/'execution.log').write_text(result.stdout+result.stderr)
if result.returncode:raise RuntimeError(result.stderr)
source=np.asarray(Image.open(c['source']).convert('RGBA'));raw=json.loads(Path(c['physicalAlpha']).read_text());a=np.array(raw['data']).reshape(raw['height'],raw['width'])/255
full_band=np.zeros(a.shape,bool);full_band[top:bottom,left:right]=band
rows=[];panels=[('source',source)]
for index,amount,oldindex in [(1,30,1),(2,60,3),(3,-30,2),(4,-60,4)]:
 baseline=np.asarray(Image.open(BASE/f'ordinary-rlib/{BASE.name}-{oldindex}-{amount if amount>0 else "minus"+str(-amount)}-candidate.png').convert('RGBA'))
 after=np.asarray(Image.open(OUT/f'replay/{BASE.name}-{index}-{amount if amount>0 else "minus"+str(-amount)}-candidate.png').convert('RGBA'))
 exposed=(a>.5)&np.all(baseline==old,axis=2)&(np.max(abs(source[:,:,:3].astype(float)-old[:,:,:3]),axis=2)>20)
 rows.append({'amount':amount,'baselineFullyExposedPixels':int(exposed.sum()),'outsideLocalBand':int((exposed&~full_band).sum())})
 panels.append((str(amount),after))
box=(0,0,source.shape[1],source.shape[0]);pw,ph=box[2]-box[0],box[3]-box[1]
sheet=Image.new('RGB',(pw*len(panels),ph+24),'white');draw=ImageDraw.Draw(sheet)
for i,(label,pixels) in enumerate(panels):sheet.paste(Image.fromarray(pixels).crop(box).convert('RGB'),(i*pw,24));draw.text((i*pw+5,5),label,fill='black')
sheet.save(OUT/'render-review.png');(OUT/'coverage.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
