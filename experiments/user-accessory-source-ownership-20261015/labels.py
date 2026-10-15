"""Sparse human visual audit labels on original source, never inference truth.

The short horizontal paths connect visible cheek-edge skin to its interior.
This is deliberately not a full segmentation or physical alpha annotation.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];out=HERE/'outputs'
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
from decompose import local_prior
z=np.load(out/'source-diagnostics.npz');a={k:z[k] for k in z.files};rgb=a['sourceRGB'];h,w=rgb.shape[:2]
params=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text())['operations'][0]['params'];features=np.maximum.reduce([a['semantic_'+k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']]);old=a['semantic_accessories'];skin=a['semantic_skin'];ear=a['semantic_ears'];hair=a['semantic_hair'];cv2=probe.cv2
core=(skin>=.9)&(np.maximum.reduce([old,ear,hair,features])<.05);prior,dist=local_prior(rgb.astype(float)/255,core,2)
lab=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2LAB);plab=cv2.cvtColor(prior.astype('float32'),cv2.COLOR_RGB2LAB);delta=lab-plab
cell=max(w/params['semanticMasks']['accessories']['width'],h/params['semanticMasks']['accessories']['height']);fw=(max(p['x'] for p in params['landmarks'])-min(p['x'] for p in params['landmarks']))*w;radius=min(3*cell,.025*fw)
color=(np.linalg.norm(delta,axis=2)<=8)&(abs(delta[...,0])<=6)&(np.linalg.norm(delta[...,1:],axis=2)<=6)
# Selected after original RGB inspection at native 1x and nearest-neighbor 4x.
paths={1:{'spineX':390,'rows':[[440,374,390],[450,375,390],[460,376,390],[470,378,390],[480,379,390],[490,381,390]]},2:{'spineX':630,'rows':[[440,630,643],[450,630,641],[460,630,640],[470,630,639],[480,630,638],[490,630,635]]}}
# The transition colors can mix skin, scene, and/or metal. Do not force an owner.
uncertain={1:[[373,440],[374,450],[375,460],[376,470],[378,480],[380,490],[371,435],[372,435]],2:[[644,440],[642,450],[641,460],[640,470],[639,480],[636,490],[646,435],[647,435]]}
metal={1:[[367,435],[355,450],[347,470],[359,490]],2:[[649,435],[654,455],[663,470],[646,490]]}
scene={1:[[371,450],[372,460],[374,470],[375,480],[377,490]],2:[[646,450],[645,460],[644,470],[640,490]]}
manual=np.zeros((h,w),np.uint8)
for c,p in paths.items():
 for y,l,r in p['rows']:manual[y,l:r+1]=1
 manual[440:491,p['spineX']]=1
 for x,y in uncertain[c]:manual[y,x]=2
 for x,y in metal[c]:manual[y,x]=3
 for x,y in scene[c]:manual[y,x]=4
Image.fromarray(manual).save(out/'sparse-source-observation-labels.png');np.save(out/'sparse-source-observation-labels.npy',manual)
stat=[]
for c,p in paths.items():
 points=[[l if c==1 else r,y] for y,l,r in p['rows']]
 for x,y in points:
  stat.append({'component':c,'xy':[x,y],'sourceRGB':list(map(int,rgb[y,x])),'label':'visible cheek skin; physical coverage not measured','physicalAlpha':float(a['physical'][y,x]),'frozen':bool(a['frozen'][y,x]),'v2Support':bool(a['v2Support'][y,x]),'oldAccessory':float(old[y,x]),'rawZimAlpha':float(a['accessoryAlpha'][y,x]),'semanticSkin':float(skin[y,x]),'semanticEar':float(ear[y,x]),'anchorDistancePx':float(dist[y,x]),'v2EvidenceRadiusPx':radius,'deltaLab':list(map(float,delta[y,x])),'deltaLabNorm':float(np.linalg.norm(delta[y,x])),'deltaChromaNorm':float(np.linalg.norm(delta[y,x,1:])),'v2ColorPass':bool(color[y,x]),'v2DistancePass':bool(dist[y,x]<=radius)})
colors={1:(0,255,70),2:(255,170,0),3:(0,180,255),4:(255,0,220)}
for c,crop in [(1,[355,420,400,505]),(2,[620,420,665,505])]:
 x0,y0,x1,y1=crop;cw,ch=x1-x0,y1-y0;scale=8;im=Image.new('RGB',(cw*scale*2+85,ch*scale+34),'#20242a');d=ImageDraw.Draw(im)
 for i in range(2):
  im.paste(Image.fromarray(rgb).crop(crop).resize((cw*scale,ch*scale),Image.Resampling.NEAREST),(45+i*(cw*scale+40),34))
  d.text((45+i*(cw*scale+40),8),'Source RGB' if i==0 else 'Source observations (sparse)',fill='white')
  for y in range(420,505,10):d.text((4+i*(cw*scale+40),34+(y-y0)*scale),str(y),fill='white')
  for x in range(((x0+9)//10)*10,x1,10):d.text((45+i*(cw*scale+40)+(x-x0)*scale,20),str(x),fill='white')
  if i==1:
   for y,x in zip(*np.nonzero(manual[y0:y1,x0:x1])):
    k=int(manual[y+y0,x+x0]);xx=45+i*(cw*scale+40)+x*scale;yy=34+y*scale;d.rectangle([xx,yy,xx+scale-1,yy+scale-1],outline=colors[k],width=2)
 im.save(out/f'component-{c}-sparse-labels-8x.png')
 Image.fromarray(rgb).crop(crop).save(out/f'component-{c}-raw-1x.png')
rows=[]
for c,crop in [(1,[320,390,410,540]),(2,[610,390,700,540])]:
 x0,y0,x1,y1=crop;ms=np.zeros((h,w),bool);ms[y0:y1,x0:x1]=manual[y0:y1,x0:x1]==1;observed=ms & (a['physical']<.99);missed=observed&~a['v2Support'];rr={'component':c,'manuallyObservedSkinPathPixels':int(ms.sum()),'pathPixelsPhysicalBelow099':int(observed.sum()),'pathPixelsPhysicalZero':int((ms&(a['physical']<=1e-6)).sum()),'physicalBelow099NotInV2Support':int(missed.sum()),'physicalZeroNotInV2Support':int((missed&(a['physical']<=1e-6)).sum()),'missingAndFrozen':int((missed&a['frozen']).sum())}
 cc,n=label(ms);rr['fourConnectedComponents']=n
 rows.append(rr)
report={'scope':'Sparse original-source RGB visual observations, selected by human inspection of 1x and 4x panels; not full annotation, not training labels, not alpha ground truth, and not an accepted fix. No labels from warped images.','legend':{'0':'unlabelled','1':'visible cheek skin path to interior; alpha unknown','2':'mixed boundary/metal-adjacent uncertain','3':'observed metal example','4':'observed dark exterior scene example'},'manualRowsInclusive':paths,'manualUncertain':uncertain,'manualMetal':metal,'manualScene':scene,'sourceDiagnosticSha256':hashlib.sha256((out/'source-diagnostics.npz').read_bytes()).hexdigest(),'pixelSamples':stat,'rows':rows}
(out/'manual-observations.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'rows':rows,'samples':stat},indent=2))
