"""Controlled local B-only replacement of exact native V37 delta trace.

Source/F/raw alpha/samples/geometry are never changed. Uses only audited
outputs-v2-consistent finite sceneDistance, existing opaque metal protection.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'qa/v37-scene-b-only-v2-feather';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
scene_path=ROOT/'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz';scene=np.load(scene_path);prior=scene['scenePrior'];finite=np.isfinite(scene['sceneDistance']);source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(source_path).convert('RGBA'));acc_path=ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz';acc=np.load(acc_path);opaque=acc['opaque'];j=acc['alpha'];report={'scope':__doc__,'sourceSha256':sha(source_path),'sceneSupportSha256':sha(scene_path),'accessoryLayersSha256':sha(acc_path),'sceneDistanceFinitePixels':int(finite.sum()),'transition':'B-only smoothstep of (interior distance - 1)/8 inside finite domain; no alpha, F or geometry changes; no writes outside finite support','accepted':False,'rows':[]}
transition=np.clip((distance_transform_edt(finite)-1)/8,0,1).astype('float32');transition=transition*transition*(3-2*transition)
Image.fromarray(np.rint(transition*255).astype('uint8')).save(OUT/'background-write-weight.png')

def compose(c,f,b,a,t,s):
    out=(c+s[...,:3])-f*a[...,None];out=out+(a-t)[...,None]*b
    return np.floor(np.clip(out,0,255)+.5).astype('uint8')

def panel(path,ims,labels,box):
 x0,y0,x1,y1=box;cw,ch=x1-x0,y1-y0;out=Image.new('RGB',(cw*len(ims),ch+24),'#20242a');draw=ImageDraw.Draw(out)
 for i,(im,txt) in enumerate(zip(ims,labels)):out.paste(Image.fromarray(im).convert('RGB').crop(box),(cw*i,24));draw.text((cw*i+2,5),txt,fill='white')
 out.save(path);out.resize((out.width*4,out.height*4),Image.Resampling.NEAREST).save(path.with_name(path.stem.replace('1x','4x')+'.png'))

for amount in [100,-100,0]:
 # Zero verifies the identical-source limit using the same arbitrary B/frozen
 # domain with sampled premult exactly F*a; native zero returns source early.
 tracing=HERE/f'qa/v37-trace/{amount if amount else 100}';d=OUT/str(amount);d.mkdir();l=tracing/'layers';st=json.loads((l/'state.json').read_text());ox,oy=st['origin'];w,h=st['width'],st['height'];cs=np.s_[oy:oy+h,ox:ox+w];a=np.fromfile(l/'source-alpha-f32le.bin',dtype='<f4').reshape(h,w);t=np.fromfile(l/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w);s=np.fromfile(l/'samples-f32le.bin',dtype='<f4').reshape(h,w,5);c=np.asarray(Image.open(l/'source-crop.png').convert('RGB')).astype('float32');f=np.asarray(Image.open(l/'foreground.png').convert('RGB')).astype('float32');b=np.asarray(Image.open(l/'background.png').convert('RGB')).astype('float32');changed=np.fromfile(l/'changed.bin',dtype='uint8').reshape(h,w)>0;frozen=np.fromfile(l/'frozen.bin',dtype='uint8').reshape(h,w)>0
 baseline=np.asarray(Image.open(tracing/'candidate.png').convert('RGBA')) if amount else source.copy()
 if amount:
  replay=compose(c,f,b,a,t,s);assert np.array_equal(replay[changed],baseline[cs][...,:3][changed])
 else:
  t=a.copy();s=np.concatenate([f*a[...,None],a[...,None],a[...,None]],axis=2);changed=np.ones((h,w),bool)
 allowed=finite[cs]&changed&~frozen&~opaque[cs]
 replacement=np.floor(np.clip(prior[cs],0,1)*255+.5).astype('float32');bnew=b.copy();blended=b*(1-transition[cs][...,None])+replacement*transition[cs][...,None];bnew[allowed]=np.floor(blended[allowed]+.5)
 result=compose(c,f,bnew,a,t,s);output=baseline.copy();patch=output[cs][...,:3].copy();patch[allowed]=result[allowed];output[cs][...,:3]=patch
 full_allowed=np.zeros(source.shape[:2],bool);full_allowed[cs]=allowed;diff=np.any(output!=baseline,axis=2)
 assert not (diff&~full_allowed).any();assert np.array_equal(output[opaque],baseline[opaque]);assert np.array_equal(output[...,3],baseline[...,3]);assert np.array_equal(a,np.fromfile(l/'source-alpha-f32le.bin',dtype='<f4').reshape(h,w));assert not amount or np.array_equal(s,np.fromfile(l/'samples-f32le.bin',dtype='<f4').reshape(h,w,5))
 if not amount:assert np.array_equal(output,source)
 Image.fromarray(output).save(d/'candidate.png');Image.fromarray(full_allowed.astype('uint8')*255).save(d/'allowed.png');Image.fromarray(diff.astype('uint8')*255).save(d/'changed-from-v37.png');Image.fromarray(bnew.astype('uint8')).save(d/'background-replaced-crop.png');np.save(d/'background-replaced.npy',bnew)
 for name,box in [('face',[287,197,730,709]),('left',[320,400,420,550]),('right',[590,400,700,550])]:panel(d/f'{name}-1x.png',[source,baseline,output],['Source','Frozen V37','Scene B only'],box)
 points=[]
 for x,y in [(630,503),(631,501),(652,430),(375,450),(376,460),(640,460),(638,480)]:
  xx,yy=x-ox,y-oy;points.append({'xy':[x,y],'finite':bool(finite[y,x]),'sceneDistance':float(scene['sceneDistance'][y,x]),'allowed':bool(allowed[yy,xx]),'oldB':list(map(float,b[yy,xx])),'newB':list(map(float,bnew[yy,xx])),'oldOutput':list(map(int,baseline[y,x,:3])),'newOutput':list(map(int,output[y,x,:3]))})
 report['rows'].append({'amount':amount,'writePixels':int(allowed.sum()),'outputChangedPixels':int(diff.sum()),'outsideChanged':0,'opaqueChanged':0,'alphaChannelChanged':0,'sourceAlphaSha256':sha(l/'source-alpha-f32le.bin'),'samplesSha256':sha(l/'samples-f32le.bin'),'geometrySha256':sha(tracing/'debug/flow-geometry.json'),'sourceForegroundSha256':sha(l/'foreground.png'),'candidateSha256':sha(d/'candidate.png'),'zeroIsOriginal':bool(np.array_equal(output,source)) if not amount else None,'points':points})
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
