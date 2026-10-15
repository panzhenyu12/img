from pathlib import Path
import json,numpy as np
from PIL import Image
E=Path(__file__).resolve().parents[1]
def read_grid(d):
 m=json.loads((d/'debug/velocity-meta.json').read_text());v=np.fromfile(d/'debug/velocity.xy.f32',dtype='<f4').reshape(m['height'],m['width'],2).astype(float);return m,v
def transport(d,ps):
 m,v=read_grid(d);w=m['width'];h=m['height'];origin=np.array([m['left']+.5,m['top']+.5]);steps=m['steps'];p=ps.copy().astype(float)
 def sample(points):
  local=points-origin;x=local[:,0];y=local[:,1];inside=(x>=0)&(y>=0)&(x<=w-1)&(y<=h-1);x=np.clip(x,0,w-1);y=np.clip(y,0,h-1);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);tx=(x-x0)[:,None];ty=(y-y0)[:,None];q=((1-tx)*v[y0,x0]+tx*v[y0,x1])*(1-ty)+((1-tx)*v[y1,x0]+tx*v[y1,x1])*ty;return q*inside[:,None]
 for _ in range(steps):
  k1=sample(p);k2=sample(p+k1*.5/steps);k3=sample(p+k2*.5/steps);k4=sample(p+k3/steps);p+=(k1+2*k2+2*k3+k4)/(6*steps)
 return p
rows=[]
for name in ['portrait-makeup','user-face-contour']:
 for amount in [100,-100]:
  base=E/'baseline'/f'{name}-{amount}';g=json.loads((base/'graph.json').read_text());sm=g['operations'][0]['params']['semanticMasks'];size=Image.open(base/'candidate.png').size;polygon=np.array([[p['x']*size[0],p['y']*size[1]]for p in sm['lips']['points']]);mouth_width=np.ptp(polygon[:,0]);radius=mouth_width*.12;points=[]
  for index in [np.argmin(polygon[:,0]),np.argmax(polygon[:,0])]:
   c=polygon[index];yy,xx=np.mgrid[int(c[1]-radius):int(c[1]+radius),int(c[0]-radius):int(c[0]+radius)];p=np.stack([xx.ravel()+.5,yy.ravel()+.5],axis=1);p=p[np.linalg.norm(p-c,axis=1)<=radius];points.extend(p)
  points=np.array(points);row={'name':name,'amount':amount,'sourcePoints':len(points),'statistics':{}}
  for key,root in [('V36','baseline'),('geometryV1','outputs-v1'),('geometryV2','outputs')]:
   d=E/root/f'{name}-{amount}'
   if not (d/'debug/velocity.xy.f32').exists():continue
   dx=transport(d,points+np.array([.5,0]))-transport(d,points-np.array([.5,0]));dy=transport(d,points+np.array([0,.5]))-transport(d,points-np.array([0,.5]));jac=np.stack([dx,dy],axis=2);sv=np.linalg.svd(jac,compute_uv=False);det=np.linalg.det(jac);row['statistics'][key]={'maxForwardStretch':float(sv[:,0].max()),'p95ForwardStretch':float(np.quantile(sv[:,0],.95)),'minForwardStretch':float(sv[:,1].min()),'minJacobian':float(det.min())}
  rows.append(row);print(row,flush=True)
(E/'qa/perioral-strain.json').write_text(json.dumps(rows,indent=2))
