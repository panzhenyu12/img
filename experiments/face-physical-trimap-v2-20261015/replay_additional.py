from pathlib import Path
import json,os,subprocess,hashlib,sys
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parent;R=E.parents[1];sys.path.insert(0,str(R/'scripts'))
from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'experiments/face-anatomical-v20-20261014/inputs/manifest.json').read_text())[1:]
binaries={'production':R/'experiments/face-physical-trimap-v1-20261014/bin/baseline-image-render','candidate':E/'replay/candidate-image-render'}
report={'binaries':{k:{'path':str(v.relative_to(R)),'sha256':sha(v)} for k,v in binaries.items()},'cases':[]}
for item in manifest:
 source=R/item['canonical'];assert sha(source)==item['canonicalSha256'];graph=R/item['sourceGraph'];assert sha(graph)==item['graphSha256'];src=Image.open(source).convert('RGBA');a=np.array(src);w,h=src.size;q=json.loads(graph.read_text());params=q['operations'][0]['params'];m={k:raster_coverage(v,w,h) for k,v in params['semanticMasks'].items() if v['shape']=='raster'};person=raster_coverage(params['subjectMask'],w,h);frozen=np.maximum.reduce([m[k] for k in ['hair','ears','accessories']])>=.5;r=params['region'];yy,xx=np.indices((h,w));outside=((xx+.5)/w<r['x'])|((xx+.5)/w>r['x']+r['width'])|((yy+.5)/h<r['y'])|((yy+.5)/h>r['y']+r['height']);box=(max(0,int(r['x']*w)),max(0,int(r['y']*h)),min(w,int((r['x']+r['width'])*w)+1),min(h,int((r['y']+r['height'])*h)+1))
 for amount in [0,30,-30,100,-100]:
  D=E/'additional'/item['name']/str(amount);D.mkdir(parents=True,exist_ok=True);params['adjustments']={'faceWidth':amount};gp=D/'graph.json';gp.write_text(json.dumps(q));ims=[]
  for name,binary in binaries.items():
   p=subprocess.run([str(binary),'--input',str(source),'--graph',str(gp),'--output',str(D/f'{name}.png')],env=dict(os.environ,FACE_PHYSICAL_TRACE=str(D/f'{name}-trace')),capture_output=True,text=True);(D/f'{name}.log').write_text(p.stdout+p.stderr);p.check_returncode();ims.append(np.array(Image.open(D/f'{name}.png').convert('RGBA')))
  b,c=ims;delta=np.any(c!=b,axis=2);changed=np.any(c!=a,axis=2);rows,cols=np.where(delta);case={'name':item['name'],'amount':amount,'source_sha256':sha(source),'source_graph_sha256':sha(graph),'graph_sha256':sha(gp),'candidate_sha256':sha(D/'candidate.png'),'production_sha256':sha(D/'production.png'),'changed_vs_baseline':int(delta.sum()),'fixed_changes':int(changed[frozen].sum()),'outside_roi_changes':int(changed[outside].sum()),'alpha_changes':int((c[:,:,3]!=a[:,:,3]).sum()),'pure_scene_delta_vs_baseline':int(delta[person<=1/255].sum()),'source_exact':bool(np.array_equal(a,c)),'delta_bbox':None if len(rows)==0 else [int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())]};report['cases'].append(case)
  for cropname,cropbox in [('face',box)]+([]if len(rows)==0 else [('changed-region',(max(0,int(cols.min())-24),max(0,int(rows.min())-24),min(w,int(cols.max())+25),min(h,int(rows.max())+25)))]):
   cw,ch=cropbox[2]-cropbox[0],cropbox[3]-cropbox[1];sheet=Image.new('RGB',(cw*3,ch+25),'#111820');draw=ImageDraw.Draw(sheet)
   for i,(im,title) in enumerate(zip([src,Image.fromarray(b),Image.fromarray(c)],['Original',f'Production {amount:+}',f'Physical-alpha V2 {amount:+}'])):sheet.paste(im.crop(cropbox),(i*cw,25));draw.text((i*cw+5,6),title,fill='white')
   sheet.save(D/f'{cropname}-1x.png')
  print(item['name'],amount,case['changed_vs_baseline'],flush=True)
(E/'additional/report.json').write_text(json.dumps(report,indent=2))
