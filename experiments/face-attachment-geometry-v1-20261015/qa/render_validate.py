from pathlib import Path
import os,subprocess,json,shutil,hashlib,sys,time
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parents[1];R=E.parents[1];sys.path.insert(0,str(R/'scripts'));from replay_user_face_contour import raster_coverage
binary=next((E/'bin').glob('candidate-v2-*'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(binary)==binary.name.removeprefix('candidate-v2-')
sources={'portrait-makeup':R/'experiments/face-anatomical-v20-20261014/inputs/portrait-makeup-source.png','portrait-woman':R/'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png','portrait-hair':R/'test/results/face-core-review/portrait-hair-source.png','user-face-contour':R/'test/results/user-face-contour-20261014/1024/canonical-source.png'}
rows=[]
for name,source in sources.items():
 original=Image.open(source).convert('RGBA');a=np.array(original);w,h=original.size
 for amt in [0,30,-30,100,-100]:
  d=E/'outputs'/f'{name}-{amt}';d.mkdir(exist_ok=True)
  old=R/'experiments/face-source-curve-v36-20261015'/('qa' if name=='portrait-makeup' else 'outputs')/(f'makeup-{amt}' if name=='portrait-makeup' else f'{name}-{amt}')
  if name=='portrait-makeup' and amt==0:old=R/'experiments/face-source-curve-v35-20261015/qa/base/portrait-makeup/parser/faceWidth_0'
  if not (d/'candidate.png').exists():
   shutil.copy2(old/'graph.json',d/'graph.json');t=time.monotonic();p=subprocess.run([str(binary),'--input',str(source),'--graph',str(d/'graph.json'),'--output',str(d/'candidate.png')],env=dict(os.environ,FACE_V21_DEBUG_DIR=str(d/'debug'),FACE_V36_DEBUG_DIR=str(d/'debug')),capture_output=True);(d/'render.log').write_bytes(p.stdout+p.stderr);p.check_returncode();print('rendered',name,amt,round(time.monotonic()-t,2),flush=True)
  g=json.loads((d/'graph.json').read_text());assert g==json.loads((old/'graph.json').read_text());b=np.array(Image.open(d/'candidate.png').convert('RGBA'));previous=np.array(Image.open(old/'candidate.png').convert('RGBA'));changed=np.any(a!=b,axis=2);delta=np.any(previous!=b,axis=2);row={'name':name,'amount':amt,'sourceSha256':sha(source),'graphSha256':sha(d/'graph.json'),'outputSha256':sha(d/'candidate.png'),'v36OutputSha256':sha(old/'candidate.png'),'changedPixelsVsV36':int(delta.sum()),'alphaChanges':int(np.sum(a[:,:,3]!=b[:,:,3])),'zeroExact':bool(np.array_equal(a,b)),'fixed':{}}
  for key in ['hair','ears','accessories']:
   mask=raster_coverage(g['operations'][0]['params']['semanticMasks'][key],w,h);row['fixed'][key]={'confidentCount':int(np.sum(mask>=.5)),'confidentChanges':int(changed[mask>=.5].sum()),'softChanges':int(changed[(mask>1e-6)&(mask<.5)].sum())};assert row['fixed'][key]['confidentChanges']==0
  assert row['alphaChanges']==0
  if not amt:assert row['zeroExact']
  else:
   od=old/'debug';cd=d/'debug';row['sourcePhysicalAlphaExact']=(od/'source-physical-alpha.png').read_bytes()==(cd/'source-physical-alpha.png').read_bytes();row['targetCurveExact']=(od/'source-width.json').read_bytes()==(cd/'source-width.json').read_bytes();assert row['sourcePhysicalAlphaExact'] and row['targetCurveExact'];oc=json.loads((od/'actual-source-width.json').read_text())['rows'];cc=json.loads((cd/'actual-source-width.json').read_text())['rows'];row['v36ActualPeaks']=[max(abs(x['points'][s]['actualInward'])for x in oc if x['points'][s]['valid'])for s in [0,1]];row['candidateActualPeaks']=[max(abs(x['points'][s]['actualInward'])for x in cc if x['points'][s]['valid'])for s in [0,1]];row['maxActualContourDelta']=max(abs(x['points'][s]['actualInward']-y['points'][s]['actualInward'])for x,y in zip(oc,cc)for s in [0,1]if x['points'][s]['valid']);row['geometry']=json.loads((cd/'flow-geometry.json').read_text());row['geometry']={k:row['geometry'][k]for k in ['folds','minJac','observed','maxForwardInverseErrorPixels']};assert row['geometry']['folds']==0
  pts=g['operations'][0]['params']['semanticMasks']['faceContour']['points'];xs=[p['x']*w for p in pts];ys=[p['y']*h for p in pts];box=(int(min(xs))-50,int(min(ys))-30,int(max(xs))+51,min(h,int(max(ys))+120));sw,sh=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(3*sw,sh+24),'#111820');draw=ImageDraw.Draw(sheet)
  for i,(im,label)in enumerate([(original,'Original'),(Image.fromarray(previous),'V36'),(Image.fromarray(b),'Geometry V2')]):sheet.paste(im.crop(box),(i*sw,24));draw.text((i*sw+4,6),label,fill='white')
  sheet.save(d/'face-comparison-1x.png');rows.append(row);(E/'qa/report.json').write_text(json.dumps({'binarySha256':sha(binary),'rows':rows},indent=2));print('verified',name,amt,'changed',row['changedPixelsVsV36'],'contourDelta',row.get('maxActualContourDelta'),flush=True)
