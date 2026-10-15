from pathlib import Path
import json,hashlib,sys
import numpy as np
from PIL import Image,ImageDraw
R=Path('/Users/modao/work/img');E=R/'experiments/face-physical-trimap-v2-20261015';sys.path.insert(0,str(R/'scripts'));from replay_user_face_contour import raster_coverage
source=Image.open(R/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA');A=np.array(source);w,h=source.size;report={'cases':[]}
for amount in [0,30,-30,100,-100]:
 D=E/'replay'/str(amount);B=np.array(Image.open(D/'baseline.png').convert('RGBA'));C=np.array(Image.open(D/'candidate.png').convert('RGBA'));delta=np.any(C!=B,axis=2);change=np.any(C!=A,axis=2);p=json.loads((D/'graph.json').read_text())['operations'][0]['params'];m={k:raster_coverage(v,w,h) for k,v in p['semanticMasks'].items() if v['shape']=='raster'};person=raster_coverage(p['subjectMask'],w,h);fixed=np.maximum.reduce([m[k] for k in ['hair','ears','accessories']])>=.5;box=p['region'];yy,xx=np.indices((h,w));outside=((xx+.5)/w<box['x'])|((xx+.5)/w>box['x']+box['width'])|((yy+.5)/h<box['y'])|((yy+.5)/h>box['y']+box['height']);case={'amount':amount,'baseline_sha256':hashlib.sha256((D/'baseline.png').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((D/'candidate.png').read_bytes()).hexdigest(),'changed_vs_baseline':int(delta.sum()),'max_channel_difference':int(np.max(abs(C.astype(int)-B.astype(int)))),'fixed_changes':int(change[fixed].sum()),'alpha_changes':int(np.any(C[:,:,3]!=A[:,:,3])),'outside_roi_changes':int(change[outside].sum()),'source_exact':bool(np.array_equal(C,A)),'pure_scene_delta_vs_baseline':int(delta[person<=1/255].sum())}
 if amount:
  trace=json.loads((D/'candidate-trace/trimap-trace.json').read_text());l,t,cw,ch=trace['crop']; physical=np.array(Image.open(D/'candidate-trace/physical-alpha.png'));oldphysical=np.array(Image.open(D/'baseline-trace/physical-alpha.png'));transported=np.array(Image.open(D/'candidate-trace/transported-alpha.png'));globalT=np.zeros((h,w),np.uint8);globalT[t:t+ch,l:l+cw]=transported
  added=physical>oldphysical;loc=np.argwhere(added);case['new_physical_pixels']=int(added.sum());case['added_source_bbox']=[int(loc[:,1].min()+l),int(loc[:,0].min()+t),int(loc[:,1].max()+l),int(loc[:,0].max()+t)];case['delta_inside_opaque_target']=int((delta&(globalT>=250)).sum());case['delta_outside_target']=int((delta&(globalT==0)).sum());case['old_island']=[{'xy':[x,y],'source':A[y,x,:3].tolist(),'baseline':B[y,x,:3].tolist(),'candidate':C[y,x,:3].tolist(),'source_physical':int(physical[y-t,x-l]),'target_physical':int(globalT[y,x])}for y in range(547,553)for x in [407,408]]
  if amount==100:case['target_left_boundary_by_y']=[{'y':y,'x':int(np.where(transported[y-t,185:280]>=128)[0].min()+185+l)}for y in range(547,553)]
 report['cases'].append(case)
 for name,box,scale in [('island',(394,531,450,566),8),('face',(285,155,750,685),1)]:
  ims=[source,Image.fromarray(B),Image.fromarray(C)];cw,ch=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(cw*scale*3,ch*scale+25),'#111820');d=ImageDraw.Draw(sheet)
  for i,(im,label) in enumerate(zip(ims,['Original',f'Production {amount:+}',f'Physical alpha V2 {amount:+}'])):sheet.paste(im.crop(box).resize((cw*scale,ch*scale),Image.Resampling.NEAREST),(i*cw*scale,25));d.text((i*cw*scale+5,6),label,fill='white')
  sheet.save(D/f'{name}.png')
(E/'replay/report.json').write_text(json.dumps(report,indent=2))
print(json.dumps([{k:v for k,v in q.items() if k!='old_island'}for q in report['cases']],indent=2))
