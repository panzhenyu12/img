from pathlib import Path
import json, hashlib, csv
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
source=Image.open(ROOT/'experiments/face-anatomical-v24-20261015/inputs/user-face-contour-source.png').convert('RGB')
source_a=np.array(source)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sample(a,x,y):
 x=max(0,min(a.shape[1]-1,x)); y=max(0,min(a.shape[0]-1,y));x0=int(x);y0=int(y);x1=min(x0+1,a.shape[1]-1);y1=min(y0+1,a.shape[0]-1);u=x-x0;v=y-y0
 return float((1-u)*(1-v)*a[y0,x0]+u*(1-v)*a[y0,x1]+(1-u)*v*a[y1,x0]+u*v*a[y1,x1])
def edges(a,y,ox,oy):
 r=a[int(y)-oy];mid=508-ox
 if r[mid]<127.5:return None
 l=mid;rgt=mid
 while l>0 and r[l-1]>=127.5:l-=1
 while rgt<a.shape[1]-1 and r[rgt+1]>=127.5:rgt+=1
 def crossing(i,j):
  return ox+i+.5+(127.5-float(r[i]))/(float(r[j])-float(r[i]))*(j-i)
 return [crossing(l,l-1) if l else ox+.5,crossing(rgt,rgt+1) if rgt<a.shape[1]-1 else ox+rgt+.5]
def mask_at(m,p):
 a=np.array(m['data']).reshape(m['height'],m['width']);return sample(a,p[0]*m['width']/1024-.5,p[1]*m['height']/1024-.5)/255
report={'note':'Read-only saved-output audit. No candidate/production code changed. Alpha PNGs are 8-bit diagnostic snapshots; boundaries use 127.5 crossing, center-connected row component, not anatomical ground truth. v23 full-replay output filenames retained historical v22.png names.','versions':{}}
for version in [22,23,24]:
 dpath=ROOT/f'experiments/face-anatomical-v{version}-20261015/outputs/full/faceWidth100';dp=dpath/'debug';geom=json.loads((dp/'flow-geometry.json').read_text());g=json.loads((dpath/'graph.json').read_text())['operations'][0]['params']; sm=g['semanticMasks'];a=np.asarray(Image.open(dp/'source-physical-alpha.png'));b=np.asarray(Image.open(dp/'transported-physical-alpha.png'));ox,oy=geom['origin']
 p=np.array(geom['controlPositions']);req=np.array(geom['requested']);feas=np.array(geom['feasible']);act=np.array(geom['actualControls']);qp=np.array(geom['afterQP']);original=np.array(geom['source']);actual=np.array(geom['transported']);origin=np.array(geom['anatomicalOrigin']);down=np.array(geom['anatomicalDown']);across=np.array(geom['anatomicalAcross']);bounds=geom['anatomicalBounds'];hh=bounds[3]-bounds[2]
 imagefile=dpath/(f'v{version}.png' if (dpath/f'v{version}.png').exists() else 'v22.png');render=np.array(Image.open(imagefile).convert('RGB'))
 rows=[]
 for y in range(400,651):
  ea=edges(a,y,ox,oy);eb=edges(b,y,ox,oy)
  row={'y':y,'sourceLeft':ea[0] if ea else None,'sourceRight':ea[1] if ea else None,'targetLeft':eb[0] if eb else None,'targetRight':eb[1] if eb else None}
  if ea and eb:row.update({'leftInward':eb[0]-ea[0],'rightInward':ea[1]-eb[1]})
  rows.append(row)
 with open(OUT/f'v{version}-alpha-rows.csv','w') as f:w=csv.DictWriter(f,fieldnames=['y','sourceLeft','sourceRight','targetLeft','targetRight','leftInward','rightInward']);w.writeheader();w.writerows(rows)
 ctrl=[]
 for i in range(len(p)):
  rr={'index':i,'source':p[i].tolist(),'requested':req[i].tolist(),'feasible':feas[i].tolist(),'actual':act[i].tolist(),'residual':float(np.linalg.norm(act[i]-feas[i])),'actualDisplacement':(act[i]-p[i]).tolist(),'movable':geom['movableControls'][i]}
  if i<len(qp):rr.update({'afterQP':qp[i].tolist(),'postQPChange':float(np.linalg.norm(qp[i]-feas[i]))})
  ctrl.append(rr)
 (OUT/f'v{version}-controls.json').write_text(json.dumps(ctrl,indent=2))
 joins=[]
 for side in [-1,1]:
  guides=[v for v in geom['externalGuides'] if (np.array(v['source'])-origin)@across*side>0]
  join=min(guides,key=lambda v:abs(v['v']-v['joinV']));pt=join['source'];inside=np.array(pt)-side*across*3.696
  joins.append({'side':side,**join,'originalRGBA':source_a[round(pt[1]),round(pt[0])].tolist(),'sourceMaskAtJoin':{k:mask_at(sm[k],pt) for k in ['face','faceContourMask','skin','neck','ears','accessories']},'sourceMaskInside':{k:mask_at(sm[k],inside) for k in ['face','faceContourMask','skin','neck','ears','accessories']}})
 # Pixels removed from an originally opaque source within the jaw-neck area.
 yy,xx=np.indices(a.shape);region=(yy+oy>=500)&(yy+oy<=620)&(xx+ox>=350)&(xx+ox<=675);vacated=(a>=250)&(b<=5)&region
 vy,vx=np.where(vacated);dark=(render[vy+oy,vx+ox].max(axis=1)<50);sourcebright=source_a[vy+oy,vx+ox].mean(axis=1)>80
 # Cross-version alpha cannot prove color stage causality; recording both yields explicit limit.
 largest=sorted(ctrl,key=lambda d:d['residual'],reverse=True)[:12]
 post=sorted([c for c in ctrl if 'postQPChange' in c],key=lambda d:d['postQPChange'],reverse=True)[:12]
 report['versions'][str(version)]={'inputs':{str(q.relative_to(ROOT)):digest(q) for q in [dp/'flow-geometry.json',dp/'source-physical-alpha.png',dp/'transported-physical-alpha.png',imagefile]},'joins':joins,'folds':geom['folds'],'minJac':geom['minJac'],'maxForwardInverse':geom['maxForwardInverseErrorPixels'],'sourceAlphaHash':digest(dp/'source-physical-alpha.png'),'targetAlphaHash':digest(dp/'transported-physical-alpha.png'),'globalNeckWeightStartWorld':(origin+down*(bounds[2]+hh*.70)).tolist() if version>=23 else (origin+down*(bounds[3]+hh*.04)).tolist(),'globalNeckWeightZeroWorld':(origin+down*(bounds[2]+hh*.94)).tolist() if version>=23 else (origin+down*(bounds[3]+hh*.30)).tolist(),'maxResidualControls':largest,'maxPostQPChanges':post,'vacatedOpaquePixelsJawNeck':int(vacated.sum()),'darkVacatedPixels':int((dark&sourcebright).sum()),'rowsEvery5':rows[::5]}
 overlay=source.copy();draw=ImageDraw.Draw(overlay)
 # green source physical edge, red transported physical edge, yellow source face controls, cyan final actual controls.
 for y,row in zip(range(400,651),rows):
  for names,color in [(('sourceLeft','sourceRight'),(30,255,40)),(('targetLeft','targetRight'),(255,30,30))]:
   for key in names:
    if row[key] is not None:draw.point((round(row[key]),y),fill=color)
 for q in joins:
  x,y=q['source'];draw.ellipse((x-5,y-5,x+5,y+5),fill=(255,0,255));draw.text((x-10,y-17),f'join {y:.1f}',fill=(255,0,255))
 for i in range(128):
  x,y=p[i];tx,ty=act[i]
  if 400<=y<=650:draw.line((x,y,tx,ty),fill=(0,220,255),width=1);draw.ellipse((x-2,y-2,x+2,y+2),fill=(255,255,0));draw.ellipse((tx-2,ty-2,tx+2,ty+2),fill=(0,220,255))
 overlay.crop((320,390,690,650)).resize((740,520)).save(OUT/f'v{version}-source-overlay-2x.png')
 render_image=Image.fromarray(render);d=ImageDraw.Draw(render_image)
 for row in rows:
  for key in ['targetLeft','targetRight']:
   if row[key] is not None:d.point((round(row[key]),row['y']),fill=(0,255,0))
 render_image.crop((350,490,670,625)).resize((960,405)).save(OUT/f'v{version}-render-alpha-overlay-3x.png')
 # Show actual rejected pixels, with source and both source/target alpha next to final render.
 crops=[source.crop((350,490,670,625)),Image.fromarray(render).crop((350,490,670,625)),Image.fromarray(a).convert('RGB').crop((350-ox,490-oy,670-ox,625-oy)),Image.fromarray(b).convert('RGB').crop((350-ox,490-oy,670-ox,625-oy))]
 panel=Image.new('RGB',(1280,160),(30,30,30));dr=ImageDraw.Draw(panel)
 for i,(im,label) in enumerate(zip(crops,['source',f'v{version} final','source physical alpha','transported physical alpha'])):panel.paste(im,(320*i,25));dr.text((320*i+5,6),label,fill=(255,255,255))
 panel.save(OUT/f'v{version}-neck-alpha-panel.png')
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2))
print('wrote diagnosis.json, 3 control tables, 3 alpha row tables and overlays')
