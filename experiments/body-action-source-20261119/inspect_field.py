"""Inspect an ordinary inverse field along an original-image garment edge.

Source RGB locates the curve; no rendered output controls edge extraction.
Inverse samples are not called target motion. The private contour target must
be inspected separately in the core if a causal target comparison is needed.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
from PIL import Image,ImageDraw
from replay import sample_mask

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def at(array,x,y):
 h,w=array.shape[:2];x=np.clip(x,0,w-1);y=np.clip(y,0,h-1);x0=int(np.floor(x));y0=int(np.floor(y));x1=min(x0+1,w-1);y1=min(y0+1,h-1);dx=x-x0;dy=y-y0
 return array[y0,x0]*(1-dx)*(1-dy)+array[y0,x1]*dx*(1-dy)+array[y1,x0]*(1-dx)*dy+array[y1,x1]*dx*dy

def polygon_distance(points,x,y):
 inside=False;distance=float('inf')
 for i,a in enumerate(points):
  b=points[(i+1)%len(points)];ax,ay=a;bx,by=b
  if (ay>y)!=(by>y) and x<(bx-ax)*(y-ay)/(by-ay)+ax:inside=not inside
  vx,vy=bx-ax,by-ay;den=vx*vx+vy*vy;t=np.clip(((x-ax)*vx+(y-ay)*vy)/den if den else 0,0,1)
  distance=min(distance,float(np.hypot(x-(ax+t*vx),y-(ay+t*vy))))
 return inside,distance

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);ap.add_argument('--replay-name',required=True);ap.add_argument('--diagnosis-name',required=True);ap.add_argument('--support-name',required=True)
 args=ap.parse_args();base=args.directory.resolve();replay=base/args.replay_name;diagnosis=base/args.diagnosis_name;support=base/args.support_name
 out=diagnosis/'source-edge-field';out.mkdir(exist_ok=False)
 source=base/'source.png';src=np.array(Image.open(source).convert('RGB'));h,w=src.shape[:2];scope=json.loads((base.parent/'source-scope.json').read_text())
 graph=json.loads((replay/'+0.json').read_text());p=graph['operations'][0]['params'];part=sample_mask(p['partMask'],w,h);subject=sample_mask(p['subjectMask'],w,h);protect=sample_mask(p['protectMask'],w,h)
 polygon=[(q['x']*w,q['y']*h) for q in p['mask']['points']];reg=p['region'];arc=next(a for a in scope['sourceVisibleArcs'] if a['id']=='right-visible-jacket-side')['points'];ax,ay=np.array(arc).T
 protocol={'sourceSHA256':sha(source),'graphSHA256':sha(replay/'+0.json'),'originalArcScopeSHA256':sha(base.parent/'source-scope.json'),
  'diagnosticAfterReplay':True,'sourceEdgeMethod':'source-only color-line .5 crossing nearest frozen source arc; foreground/background anchors at xguess-16..-8 / xguess+8..16; search xguess±12',
  'yRange':[560,870],'step':1,'inverseFieldIsNotTargetDisplacement':True,'sourceCurveIsNotPhysicalAlphaTruth':True}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 rows=[]
 for y in range(560,871):
  guess=float(np.interp(y,ay,ax));gx=int(round(guess));fg=np.median(src[y-1:y+2,gx-16:gx-8].reshape(-1,3),axis=0);bg=np.median(src[y-1:y+2,gx+8:gx+16].reshape(-1,3),axis=0);direction=bg-fg;den=float(direction@direction)
  xs=np.arange(gx-12,gx+13);colors=src[y,xs].astype(float);projection=(colors-fg)@direction/max(den,1e-9);cross=[]
  for i in range(len(xs)-1):
   if projection[i]<.5<=projection[i+1]:cross.append(float(xs[i]+(.5-projection[i])/(projection[i+1]-projection[i])))
  x=min(cross,key=lambda x:abs(x-guess)) if cross else None
  row={'y':y,'sourceArcGuessX':guess,'sourceRGBEdgeX':x,'sourceCrossings':len(cross),'anchorContrast':float(np.sqrt(den))}
  if x is not None:
   inside,distance=polygon_distance(polygon,x,y);row.update(partMask=float(at(part,x,y)),subjectMask=float(at(subject,x,y)),protectMask=float(at(protect,x,y)),insideSupport=inside,supportBoundaryDistance=distance,insideRegion=reg['x']<=x/w<=reg['x']+reg['width'] and reg['y']<=y/h<=reg['y']+reg['height'])
  rows.append(row)
 (out/'source-edge.json').write_text(json.dumps(rows,indent=2))
 residual=json.loads((support/'residual-scope.json').read_text());summaries=[]
 for amount in [-30,30]:
  label=f'{amount:+d}';field=np.fromfile(diagnosis/f'{label}.f32','<f4').reshape(h,w,2);image=np.array(Image.open(replay/f'{label}.png').convert('RGB'));diff=np.abs(image.astype(int)-src.astype(int));changed=np.any(diff>0,axis=-1)
  current=[]
  for row in rows:
   item=dict(row);x=row['sourceRGBEdgeX'];y=row['y']
   if x is not None:
    d=at(field,x,y);ddx=(at(field,x+1,y)-at(field,x-1,y))*.5;ddy=(at(field,x,y+1)-at(field,x,y-1))*.5
    item.update(dx=float(d[0]),dy=float(d[1]),jacobianDet=float((1+ddx[0])*(1+ddy[1])-ddy[0]*ddx[1]))
   current.append(item)
  valid=[r for r in current if r['sourceRGBEdgeX'] is not None];xvals=np.array([r['dx'] for r in valid]);ys=np.array([r['y'] for r in valid]);slope=np.gradient(xvals,ys);second=np.gradient(slope,ys)
  for r,s,t in zip(valid,slope,second):r['alongSourceEdgeDxSlope']=float(s);r['alongSourceEdgeDxSecond']=float(t)
  (out/f'{label}.json').write_text(json.dumps(current,indent=2))
  fields=list(valid[0]);
  with (out/f'{label}.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(valid)
  probes=[]
  for probe in residual['points']:
   x,y=probe['xy'];probes.append({**probe,'partMask':float(part[y,x]),'subjectMask':float(subject[y,x]),'protectMask':float(protect[y,x]),'field':field[y,x].tolist(),'rgbDifference':diff[y,x].tolist()})
  boxes={}
  for name,box in residual['mixedInspectionBoxes'].items():
   x0,y0,x1,y1=box;sl=(slice(y0,y1),slice(x0,x1));m=np.linalg.norm(field[sl],axis=-1);d=diff[sl]
   boxes[name]={'box':box,'changedPixels':int(changed[sl].sum()),'maxRGB':int(d.max()),'displacedPixelsAbovePoint01':int((m>.01).sum()),'maxDisplacement':float(m.max()),'protectedPixels':int((protect[sl]>0).sum())}
  top=sorted(valid,key=lambda r:abs(r['alongSourceEdgeDxSecond']),reverse=True)[:12]
  summaries.append({'amount':amount,'fieldSHA256':sha(diagnosis/f'{label}.f32'),'edgeRows':len(valid),'minJacobianAtSourceEdge':min(r['jacobianDet'] for r in valid),'maxAbsoluteDxSlope':max(abs(r['alongSourceEdgeDxSlope']) for r in valid),'largestSecondChanges':top,'residualProbes':probes,'residualBoxes':boxes})
 (out/'report.json').write_text(json.dumps({'protocol':protocol,'rows':summaries},indent=2))
 box=residual['mixedInspectionBoxes']['reviewer-left-residual-band'];cw=(box[2]-box[0])*4;ch=(box[3]-box[1])*4;sheet=Image.new('RGB',(3*cw,ch+25),'#222');d=ImageDraw.Draw(sheet)
 for col,lab in enumerate(['+0','-30','+30']):
  sheet.paste(Image.open(replay/f'{lab}.png').crop(box).resize((cw,ch),Image.Resampling.NEAREST),(col*cw,25));d.text((col*cw+5,5),lab,fill='white')
 sheet.save(out/'left-residual-triptych.png')
 print(json.dumps([{k:v for k,v in row.items() if k not in ['largestSecondChanges','residualProbes','fieldSHA256']} for row in summaries],indent=2))
if __name__=='__main__':main()
