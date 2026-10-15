from pathlib import Path
import json,os,time,hashlib,subprocess,shutil,sys
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from replay_user_face_contour import raster_coverage
base=ROOT/'test/results/user-face-contour-20261014/1024'
binary=ROOT/'.run/face-rbf-v14-candidate-image-render'
output=base/'rbf-v14-candidate';output.mkdir(exist_ok=True)
source=base/'canonical-source.png';original=Image.open(source).convert('RGBA');a=np.asarray(original)
initial=json.loads((base/'face-graph.json').read_text());masks=initial['operations'][0]['params']['semanticMasks']
coverages={key:raster_coverage(mask,*original.size) for key,mask in masks.items() if mask['shape']=='raster'}
frozen=np.maximum.reduce([coverages[k] for k in ['ears','accessories','hair']])>=.5
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();frozen_binary=sha(binary)
report={'sourceSha256':sha(source),'sourceGraphSha256':sha(base/'face-graph.json'),'candidateSha256':frozen_binary,'rows':[],'review':'Numerical checks do not constitute visual acceptance.'}
for amount in [100,-100]:
 graph=json.loads(json.dumps(initial));graph['operations'][0]['params']['adjustments']={'faceWidth':amount}
 gp=output/f'faceWidth{amount}.json';gp.write_text(json.dumps(graph));png=output/f'faceWidth{amount}.png';debug=output/f'debug{amount}'
 env=os.environ.copy();env['FACE_FLOW_DEBUG_DIR']=str(debug);env['FACE_CORE_PROFILE']='1'
 t=time.perf_counter();p=subprocess.run([str(binary),'--input',str(source),'--graph',str(gp),'--output',str(png)],text=True,capture_output=True,env=env)
 if p.returncode:raise RuntimeError(p.stderr)
 (output/f'faceWidth{amount}-profile.txt').write_text(p.stderr)
 after=Image.open(png).convert('RGBA');b=np.asarray(after);changed=np.any(a[:,:,:3]!=b[:,:,:3],axis=2)
 row={'amount':amount,'fullProcessMs':(time.perf_counter()-t)*1000,'changed':int(changed.sum()),'frozenChanged':int((changed&frozen).sum()),'alphaChanged':int((a[:,:,3]!=b[:,:,3]).sum()),'outputSha256':sha(png),'flowDiagnostics':json.loads((debug/'flow-diagnostics.json').read_text())}
 report['rows'].append(row)
 for location,box in {'face':(280,160,750,665),'left':(290,285,425,585),'right':(598,285,738,585),'chin':(375,510,640,650)}.items():
  images=[original.crop(box),Image.open(base/'baseline'/f'faceWidth{amount}.png').convert('RGBA').crop(box),after.crop(box)]
  labels=['Original',f'v11 {amount}',f'RBF v14 {amount}'];width,height=images[0].size
  canvas=Image.new('RGB',(width*2*3,height*2+28),'#181c23');draw=ImageDraw.Draw(canvas)
  for j,(im,label) in enumerate(zip(images,labels)):
   canvas.paste(im.convert('RGB').resize((width*2,height*2),Image.Resampling.NEAREST),(j*width*2,28));draw.text((j*width*2+6,8),label,fill='white')
  canvas.save(output/f'faceWidth{amount}-{location}-comparison-2x.png')
 if sha(binary)!=frozen_binary:raise RuntimeError('Candidate binary changed during replay')
 (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
