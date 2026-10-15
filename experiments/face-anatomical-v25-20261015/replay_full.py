from pathlib import Path
import subprocess,os,json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/Users/modao/work/img'); HERE=ROOT/'experiments/face-anatomical-v24-20261015'; B=Path('/Users/modao/work/img/.run/face-anatomical-v24-candidate-image-render'); SRC=HERE/'inputs/user-face-contour-source.png'; support=ROOT/'test/results/accessory-matting-zim-v1-20261014/full-replay'
out=HERE/'outputs/full';out.mkdir(parents=True,exist_ok=True)
amounts=[0,-30,30,-100,100]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for amount in amounts:
 graph=support/f'faceWidth{amount if amount else 0}.json'; dst=out/f'faceWidth{amount}';dst.mkdir(exist_ok=True)
 g=json.loads(graph.read_text());g['operations'][0]['params']['adjustments']={'faceWidth':amount}; gp=dst/'graph.json';gp.write_text(json.dumps(g,separators=(',',':')))
 env=os.environ.copy();env['FACE_V24_DEBUG_DIR']=str(dst/'debug')
 t=time.perf_counter();r=subprocess.run([str(B),'--input',str(SRC),'--graph',str(gp),'--output',str(dst/'v24.png')],env=env,capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 print(amount,round((time.perf_counter()-t)*1000,2),sha(dst/'v24.png'))
 baseline=support/f'faceWidth{amount if amount else 0}-attachment-control.png'
 if not baseline.exists():baseline=support/f'faceWidth{amount if amount else 0}-physical.png'
 source=Image.open(SRC).convert('RGB');base=Image.open(baseline).convert('RGB');cand=Image.open(dst/'v24.png').convert('RGB')
 for name,box in {'face':(280,160,750,665),'jaw-neck':(330,450,690,735),'left-ear':(320,370,440,540),'right-ear':(585,370,705,540)}.items():
  ims=[im.crop(box) for im in [source,base,cand]];w,h=ims[0].size;c=Image.new('RGB',(w*3,h+30),'#181c23');d=ImageDraw.Draw(c)
  for i,(im,label) in enumerate(zip(ims,['Source','v23 support','v24'])):c.paste(im,(i*w,30));d.text((i*w+5,8),label,fill='white')
  c.save(dst/f'{name}-1x.png')
print('done')
