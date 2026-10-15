from pathlib import Path
import shutil,hashlib,json,subprocess,os,time
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
built=ROOT/'.run/face-source-curve-v31-target/release/image-render';binary=OUT/('image-render-'+sha(built));shutil.copy2(built,binary)
report={'binary':sha(binary),'rows':[]}
for name in ['portrait-woman','portrait-hair','user-face-contour']:
 source=ROOT/('experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png' if name=='portrait-woman' else 'test/results/face-core-review/portrait-hair-source.png' if name=='portrait-hair' else 'test/results/user-face-contour-20261014/1024/canonical-source.png')
 for amount in [100,-100,30,-30,0]:
  previous=ROOT/f'experiments/face-source-curve-v35-20261015/qa/base/{name}/parser/faceWidth_{amount}' if name!='portrait-hair' else ROOT/f'experiments/face-source-curve-v35-20261015/extra-qa/portrait-hair-{amount}'
  dest=OUT/f'{name}-{amount}';dest.mkdir();g=dest/'graph.json';shutil.copy2(previous/'graph.json',g)
  t=time.perf_counter();r=subprocess.run([str(binary),'--input',str(source),'--graph',str(g),'--output',str(dest/'candidate.png')],env={**os.environ,'FACE_V21_DEBUG_DIR':str(dest/'debug'),'FACE_V36_DEBUG_DIR':str(dest/'debug')},capture_output=True);(dest/'render.log').write_bytes(r.stdout+r.stderr);r.check_returncode()
  source_im=Image.open(source).convert('RGBA');prev=Image.open(previous/'candidate.png').convert('RGBA');cand=Image.open(dest/'candidate.png').convert('RGBA')
  op=json.loads(g.read_text())['operations'][0];pts=op['params']['landmarks'];xx=[p['x']*cand.width for p in pts];yy=[p['y']*cand.height for p in pts];box=(max(0,int(min(xx)-(max(xx)-min(xx))*.22)),max(0,int(min(yy)-(max(yy)-min(yy))*.15)),min(cand.width,int(max(xx)+(max(xx)-min(xx))*.22)),min(cand.height,int(max(yy)+(max(yy)-min(yy))*.3)))
  w,h=box[2]-box[0],box[3]-box[1];im=Image.new('RGB',(w*3,h+24),'#222');draw=ImageDraw.Draw(im)
  for k,(label,img) in enumerate([('Original',source_im),('V35 hair locked',prev),('V36 hair occlusion',cand)]):im.paste(img.convert('RGB').crop(box),(k*w,24));draw.text((k*w+3,5),label,fill='white')
  im.save(dest/'face-1x.png')
  row={'name':name,'amount':amount,'sourceSha':sha(source),'graphSha':sha(g),'outputSha':sha(dest/'candidate.png'),'ms':(time.perf_counter()-t)*1000,'newCurveUsed':(dest/'debug/source-width.json').exists()}
  if amount==0:assert np.array_equal(np.asarray(source_im),np.asarray(cand))
  if row['newCurveUsed']:
   j=json.loads((dest/'debug/actual-source-width.json').read_text());row['maxActual']=[max(abs(r['points'][s]['actualInward'])for r in j['rows'] if r['points'][s]['valid'])for s in range(2)]
  report['rows'].append(row);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(row,flush=True)
