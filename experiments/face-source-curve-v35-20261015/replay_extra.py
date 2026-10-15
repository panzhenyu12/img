from pathlib import Path
import json,hashlib,os,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=HERE/'extra-qa';OUT.mkdir(exist_ok=False)
CAND=next((HERE/'bin').glob('image-render-*'))
BASE=ROOT/'experiments/face-occlusion-v21-20261014/binaries/face-occlusion-v21-candidate-image-render-777c8948d6349e91806cec3ae02ff4fc04ff93f61843bede2744c22d2dab722e'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'candidate':sha(CAND),'baseline':sha(BASE),'rows':[],'scope':'Two additional originals and one rotated derivative with identical source supports; native isolated candidate only; timing under concurrent QA is not an interactive performance claim.'}
for name in ['portrait-roll','portrait-glasses-street','portrait-hair']:
 source=ROOT/f'test/results/face-core-review/{name}-source.png';graph=ROOT/f'test/results/face-core-review/{name}-faceWidth-60.json'
 image=Image.open(source).convert('RGBA');raw=np.asarray(image)
 seed=json.loads(graph.read_text());op=next(o for o in seed['operations'] if o['type']=='face.adjust');seed['operations']=[op]
 pts=op['params']['landmarks'];xx=[p['x']*image.width for p in pts];yy=[p['y']*image.height for p in pts]
 box=(max(0,int(min(xx)-(max(xx)-min(xx))*.3)),max(0,int(min(yy)-(max(yy)-min(yy))*.25)),min(image.width,int(max(xx)+(max(xx)-min(xx))*.3)),min(image.height,int(max(yy)+(max(yy)-min(yy))*.4)))
 for a in [0,30,100,-30,-100]:
  d=OUT/f'{name}-{a}';d.mkdir();seed['operations'][0]['params']['adjustments']={'faceWidth':a};g=d/'graph.json';g.write_text(json.dumps(seed,separators=(',',':')))
  row={'name':name,'amount':a,'sourceSha':sha(source),'graphSha':sha(g)}
  for tag,binary in [('baseline',BASE),('candidate',CAND)]:
   t=time.perf_counter();r=subprocess.run([str(binary),'--input',str(source),'--graph',str(g),'--output',str(d/f'{tag}.png')],env={**os.environ,'FACE_V21_DEBUG_DIR':str(d/f'{tag}-debug'),'FACE_V35_DEBUG_DIR':str(d/f'{tag}-debug')},capture_output=True)
   (d/f'{tag}.log').write_bytes(r.stdout+r.stderr);r.check_returncode();row[tag+'Ms']=(time.perf_counter()-t)*1000
  cand=Image.open(d/'candidate.png').convert('RGBA');base=Image.open(d/'baseline.png').convert('RGBA')
  row['newCurveUsed']=(d/'candidate-debug/source-width.json').exists();row['zeroExact']=bool(np.array_equal(np.asarray(cand),raw)) if a==0 else None
  if a==0: assert row['zeroExact']
  ca=d/'candidate-debug/source-physical-alpha.png';ba=d/'baseline-debug/source-physical-alpha.png';row['sameSourceAlpha']=sha(ca)==sha(ba) if ca.exists() and ba.exists() else None
  if row['sameSourceAlpha'] is False: raise RuntimeError('source alpha unexpectedly changed')
  w,h=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(w*3,h+24),'#222');draw=ImageDraw.Draw(sheet)
  for k,(label,im) in enumerate([('Original',image),('V21 same support',base),('V35 constrained target',cand)]):sheet.paste(im.convert('RGB').crop(box),(k*w,24));draw.text((k*w+3,5),label,fill='white')
  sheet.save(d/'face-1x.png');row['outputSha']=sha(d/'candidate.png');report['rows'].append(row);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(row,flush=True)
