"""Independent immutable V35 validation. Does not edit candidate source or tests."""
from pathlib import Path
import os,json,hashlib,subprocess,time,sys,copy
import numpy as np
from PIL import Image,ImageDraw
Q=Path(__file__).resolve().parent;E=Q.parent;R=E.parents[1];sys.path.insert(0,str(R/'scripts'))
from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance=json.loads((Q/'provenance.json').read_text());binary=Path(provenance['candidate_binary'])
manifest=json.loads((R/'experiments/face-anatomical-v20-20261014/inputs/manifest.json').read_text())
fixtures=[]
for item in manifest:
 name=item['name'];source=R/item['canonical'];assert sha(source)==item['canonicalSha256']
 supports=['parser','combined'] if name=='user-face-contour' else ['parser']
 for support in supports:
  seed=R/'test/results/accessory-matting-zim-v1-20261014/full-replay/faceWidth100.json' if support=='combined' else R/item['sourceGraph'];q=json.loads(seed.read_text());im=Image.open(source).convert('RGBA');a=np.array(im);w,h=im.size;p=q['operations'][0]['params'];m={k:raster_coverage(v,w,h) for k,v in p['semanticMasks'].items() if v['shape']=='raster'};person=raster_coverage(p['subjectMask'],w,h);frozen=np.maximum.reduce([m[k] for k in ['hair','ears','accessories']])>=.5;r=p['region'];yy,xx=np.indices((h,w));outside=((xx+.5)/w<r['x'])|((xx+.5)/w>r['x']+r['width'])|((yy+.5)/h<r['y'])|((yy+.5)/h>r['y']+r['height']);fixtures.append(dict(name=name,support=support,source=source,seed=seed,graph=q,image=im,pixels=a,person=person,frozen=frozen,outside=outside))
report={'binary':str(binary),'binarySha256':sha(binary),'rows':[],'continuity':[]}
def verify_frozen():
 assert sha(binary)==provenance['candidate_binary_sha256']
 for relative,digest in provenance['source_hashes'].items():assert sha(E/relative)==digest,relative

def render(fixture,adjustments,kind):
 name=fixture['name'];support=fixture['support'];token='-'.join(f'{k}_{v:g}'for k,v in adjustments.items());D=Q/kind/name/support/token;D.mkdir(parents=True,exist_ok=True);g=copy.deepcopy(fixture['graph']);g['operations'][0]['params']['adjustments']=adjustments;gp=D/'graph.json';gp.write_text(json.dumps(g,separators=(',',':')));verify_frozen();t=time.perf_counter();r=subprocess.run([str(binary),'--input',str(fixture['source']),'--graph',str(gp),'--output',str(D/'candidate.png')],env=dict(os.environ,FACE_V21_DEBUG_DIR=str(D/'debug'),FACE_V35_DEBUG_DIR=str(D/'debug')),capture_output=True);elapsed=time.perf_counter()-t;(D/'render.log').write_bytes(r.stdout+r.stderr);verify_frozen();row={'name':name,'support':support,'kind':kind,'adjustments':adjustments,'sourceSha256':sha(fixture['source']),'sourceGraphSha256':sha(fixture['seed']),'graphSha256':sha(gp),'directory':str(D.relative_to(E)),'returnCode':r.returncode,'wallMs':elapsed*1000};report['rows'].append(row)
 if r.returncode:
  row['error']=(r.stdout+r.stderr).decode(errors='replace')[-2000:];(Q/'report.json').write_text(json.dumps(report,indent=2));return row
 out=Image.open(D/'candidate.png').convert('RGBA');b=np.array(out);a=fixture['pixels'];changed=np.any(a!=b,axis=2);row.update(outputSha256=sha(D/'candidate.png'),changedPixels=int(changed.sum()),zeroExact=bool(np.array_equal(a,b)),fixedChanges=int(changed[fixture['frozen']].sum()),outsideRoiChanges=int(changed[fixture['outside']].sum()),outputAlphaChanges=int((a[:,:,3]!=b[:,:,3]).sum()),lowPersonChanges=int(changed[fixture['person']<=1/255].sum()),newCurveUsed=(D/'debug/source-width.json').exists())
 if row['newCurveUsed']:
  curve=json.loads((D/'debug/source-width.json').read_text());actual=json.loads((D/'debug/actual-source-width.json').read_text());pts=[p for r in actual['rows'] for p in r['points'] if p['valid']];requested=curve['requestedPeak'];row['motion']={'requestedPeak':requested,'projectedMaximumAbs':max(abs(p['targetInward']) for p in pts),'actualMaximumAbs':max(abs(p['actualInward']) for p in pts),'actualMeanAbs':float(np.mean([abs(p['actualInward']) for p in pts])),'maxTargetExecutionError':max(float(np.linalg.norm(np.array(p['target'])-np.array(p['actual']))) for p in pts)}
 if kind=='base':
  amount=adjustments['faceWidth'];basepath=(R/f'test/results/accessory-matting-zim-v1-20261014/full-replay/faceWidth{amount}-physical.png' if support=='combined' else R/f'experiments/face-physical-trimap-v2-20261015/replay/{amount}/baseline.png') if name=='user-face-contour' else R/f'experiments/face-physical-trimap-v2-20261015/additional/{name}/{amount}/production.png';base=Image.open(basepath).convert('RGBA');row['baselineSha256']=sha(basepath);row['baselinePath']=str(basepath.relative_to(R));
  if name=='user-face-contour':boxes={'face':(280,160,750,700),'neck':(330,450,690,735),'leftEar':(315,360,420,560),'rightEar':(595,360,710,560)}
  else:
   p=g['operations'][0]['params'];mask=p['semanticMasks']['faceContour'];points=mask['points'];w,h=out.size;xs=[v['x']*w for v in points];ys=[v['y']*h for v in points];boxes={'face':(max(0,int(min(xs))-50),max(0,int(min(ys))-30),min(w,int(max(xs))+51),min(h,int(max(ys))+120))}
  for label,box in boxes.items():
   w,h=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(w*3,h+24),'#111820');draw=ImageDraw.Draw(sheet)
   for i,(image,title) in enumerate([(fixture['image'],'Original'),(base,'V21 same masks' if support=='combined' else 'Production V19'),(out,'V35')]):sheet.paste(image.crop(box),(i*w,24));draw.text((i*w+4,6),title,fill='white')
   sheet.save(D/f'{label}-comparison-1x.png')
 (Q/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:row[k]for k in ['name','support','kind','adjustments','newCurveUsed','fixedChanges','outsideRoiChanges']}),flush=True);return row

for fixture in fixtures:
 for amount in [0,30,100,-30,-100]:render(fixture,{'faceWidth':amount},'base')
for fixture in fixtures:
 if fixture['name']!='user-face-contour':continue
 for width in [30,100]:
  for key in ['chin','jaw']:
   group=[render(fixture,{'faceWidth':width,key:amount},'cross-zero')for amount in [-.001,0,.001]]
   if any(row['returnCode'] for row in group):continue
   paths=[E/row['directory']for row in group];images=[np.array(Image.open(p/'candidate.png').convert('RGBA'))for p in paths];basecase=Q/'base'/fixture['name']/fixture['support']/f'faceWidth_{width}'/'candidate.png';base=np.array(Image.open(basecase).convert('RGBA'));item={'support':fixture['support'],'faceWidth':width,'parameter':key,'explicitZeroVsOmittedExact':bool(np.array_equal(images[1],base)),'pairs':[]}
   for left,right in [(0,1),(1,2),(0,2)]:
    delta=np.abs(images[left].astype(int)-images[right].astype(int));change=np.any(delta,axis=2);actual=[]
    for path in [paths[left],paths[right]]:
     j=json.loads((path/'debug/actual-source-width.json').read_text());actual.append(np.array([p['actual']for row in j['rows']for p in row['points']]))
    movement=np.linalg.norm(actual[0]-actual[1],axis=1);item['pairs'].append({'values':[group[left]['adjustments'][key],group[right]['adjustments'][key]],'changedPixels':int(change.sum()),'maximumChannelDifference':int(delta.max()),'maxActualContourMovement':float(movement.max()),'meanActualContourMovement':float(movement.mean())})
   report['continuity'].append(item);(Q/'report.json').write_text(json.dumps(report,indent=2));print('CONTINUITY',json.dumps(item),flush=True)
verify_frozen()
