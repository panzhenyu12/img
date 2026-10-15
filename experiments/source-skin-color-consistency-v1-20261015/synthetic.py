from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from test_solver import KnownSoftEdge
from solver import solve
HERE=Path(__file__).resolve().parent;OUT=HERE/'qa/synthetic';OUT.mkdir(exist_ok=True)
KnownSoftEdge.setUpClass();q=KnownSoftEdge;a,f,b,c,fp,core,allow=q.a,q.f,q.b,q.c,q.prior,q.core,q.allow
shift=11
warp=lambda v:np.roll(v,-shift,axis=1)
at=warp(a);truth=warp(a[...,None]*f)+(1-at[...,None])*b
b_bad=np.zeros_like(b)+[.74,.71,.65]
delta=lambda ff,bb:c+warp(a[...,None]*ff)-a[...,None]*ff+(a-at)[...,None]*bb
red=delta(fp,b_bad);bonly=delta(fp,b);s=solve(c,a,fp,b,core,allow);green=delta(s['foreground'],b)
valid=np.zeros(a.shape,bool);valid[6:-6,6:-shift-4]=True
old=(a>.1)&(at==0)&valid
rows=[]
for name,v in [('V37-like-bad-B-and-F',red),('B-only',bonly),('bounded-F-and-B',green)]:
 rows.append({'name':name,'maxErrorByte':float(abs(v-truth)[valid].max()*255),'oldEdgeMaxErrorByte':float(abs(v-truth)[old].max()*255),'sourceTextureMaxErrorByte':float(abs((s['foreground'] if name=='bounded-F-and-B' else fp)-f)[q.band].max()*255)})
h,w=a.shape;ims=[c,truth,red,bonly,green];labels=['Source','Known truth','Bad B / F','B only','Bounded F+B'];im=Image.new('RGB',(w*5,h+24),'#20242a');d=ImageDraw.Draw(im)
for i,(v,label) in enumerate(zip(ims,labels)):im.paste(Image.fromarray(np.floor(np.clip(v,0,1)*255+.5).astype('uint8')),(i*w,24));d.text((i*w+2,5),label,fill='white')
im.save(OUT/'texture-soft-edge-1x.png');im.resize((im.width*4,im.height*4),Image.Resampling.NEAREST).save(OUT/'texture-soft-edge-4x.png');np.savez_compressed(OUT/'known-layers.npz',trueAlpha=a,trueForeground=f,trueBackground=b,source=c,truth=truth,red=red,Bonly=bonly,green=green);(OUT/'report.json').write_text(json.dumps({'rows':rows,'alphaChanged':False,'geometry':'fixed 11px translation; no optimization; exclude wrapping border'},indent=2)+'\n');print(json.dumps(rows,indent=2))
