from pathlib import Path
import os,json,hashlib,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parent;R=E.parents[1];sys.path.insert(0,str(R/'scripts'))
from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=R/'test/results/user-face-contour-20261014/1024/canonical-source.png';g=E/'graph100.json';q=json.loads(g.read_text())['operations'][0]['params'];source=Image.open(src).convert('RGBA');a=np.asarray(source);w,h=source.size
binary=E/'bin/candidate-image-render';binary_sha=sha(binary)
subprocess.run([str(binary),'--input',str(src),'--graph',str(g),'--output',str(E/'candidate100.png')],env=dict(os.environ,FACE_PHYSICAL_TRACE=str(E/'candidate-trace')),check=True,capture_output=True);assert sha(binary)==binary_sha
b=np.asarray(Image.open(E/'baseline100.png').convert('RGBA'));c=np.asarray(Image.open(E/'candidate100.png').convert('RGBA'));changed=np.any(c!=a,axis=2);delta=np.any(c!=b,axis=2)
masks={k:raster_coverage(m,w,h) for k,m in q['semanticMasks'].items() if m['shape']=='raster'};person=raster_coverage(q['subjectMask'],w,h)
frozen=np.maximum.reduce([masks[k] for k in ['hair','ears','accessories']])>=.5
r=q['region'];yy,xx=np.indices((h,w));outside=((xx+.5)/w<r['x'])|((xx+.5)/w>r['x']+r['width'])|((yy+.5)/h<r['y'])|((yy+.5)/h>r['y']+r['height'])
report={'source_sha256':sha(src),'graph_sha256':sha(g),'candidate_binary_sha256':binary_sha,'baseline_binary_sha256':sha(E/'bin/baseline-image-render'),'frozen_changes':int(changed[frozen].sum()),'known_scene_changes':int(changed[(person<=1/255)&(masks['face']<=1e-6)&(masks['skin']<=1e-6)].sum()),'outside_region_changes':int(changed[outside].sum()),'alpha_changes':int((c[:,:,3]!=a[:,:,3]).sum()),'changed_vs_baseline':int(delta.sum()),'island':[]}
for y in range(547,553):
 for x in [407,408]:report['island'].append({'x':x,'y':y,'source':a[y,x].tolist(),'baseline':b[y,x].tolist(),'candidate':c[y,x].tolist(),'changed':bool(delta[y,x])})
base_trace=json.loads((E/'baseline-trace/trimap-trace.json').read_text());cand_trace=json.loads((E/'candidate-trace/trimap-trace.json').read_text());report['crop']=cand_trace['crop'];report['island_alpha']=[r for r in cand_trace['points'] if 407<=r['x']<=408 and 547<=r['y']<=552]
old_alpha=np.asarray(Image.open(E/'baseline-trace/physical-alpha.png'));new_alpha=np.asarray(Image.open(E/'candidate-trace/physical-alpha.png'));loc=np.argwhere(new_alpha>old_alpha);report['physical_alpha_increased_pixels']=len(loc);report['physical_alpha_increased_bbox']=[int(loc[:,1].min()),int(loc[:,0].min()),int(loc[:,1].max()),int(loc[:,0].max())] if len(loc) else None
report['new_physical_pixels']=[{'x':int(x+report['crop'][0]),'y':int(y+report['crop'][1]),'alpha':int(new_alpha[y,x]),'person':float(person[y+report['crop'][1],x+report['crop'][0]])} for y,x in loc]
(E/'report.json').write_text(json.dumps(report,indent=2))
for name,box,scale in [('island',(394,531,426,566),10),('neck',(360,465,560,655),1),('face',(285,155,750,685),1)]:
 images=[source,Image.fromarray(b),Image.fromarray(c)];cw,ch=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(cw*scale*3,ch*scale+28),'#141b25');d=ImageDraw.Draw(sheet)
 for i,(im,title) in enumerate(zip(images,['Original','Production baseline +100','Physical trimap +100'])):
  sheet.paste(im.convert('RGB').crop(box).resize((cw*scale,ch*scale),Image.Resampling.NEAREST),(i*cw*scale,28));d.text((i*cw*scale+5,8),title,fill='white')
 sheet.save(E/'evidence'/f'{name}-comparison-{scale}x.png')
print(json.dumps({k:v for k,v in report.items() if k not in ['island','island_alpha','new_physical_pixels']},indent=2))
