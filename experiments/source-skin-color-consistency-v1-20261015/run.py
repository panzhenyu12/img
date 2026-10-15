"""Fixed V37 geometry/alpha, observed-scene B, bounded local source-F solve."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from solver import solve
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'outputs-v2-monotonic';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sample_f32(f,a,xy):
 h,w=a.shape;xx,yy=xy[...,0],xy[...,1];x0=np.floor(xx).astype('int32');y0=np.floor(yy).astype('int32');tx=xx-np.floor(xx);ty=yy-np.floor(yy);out=np.zeros((*a.shape,3),'float32');membership=np.zeros(a.shape,'float32')
 for dx,dy,weight in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]:
  x,y=x0+dx,y0+dy;v=(x>=0)&(y>=0)&(x<w)&(y<h);wa=a[y[v],x[v]]*weight[v];membership[v]+=wa;out[v]+=f[y[v],x[v]]*wa[:,None]
 return out,membership

def compose(c,f,b,a,t,s):
 result=(c+s)-f*a[...,None];result=result+(a-t)[...,None]*b
 return np.floor(np.clip(result,0,255)+.5).astype('uint8')

def panel(path,ims,labels,box):
 x0,y0,x1,y1=box;cw,ch=x1-x0,y1-y0;out=Image.new('RGB',(cw*len(ims),ch+24),'#20242a');draw=ImageDraw.Draw(out)
 for i,(im,txt) in enumerate(zip(ims,labels)):out.paste(Image.fromarray(im).convert('RGB').crop(box),(cw*i,24));draw.text((cw*i+2,5),txt,fill='white')
 out.save(path);out.resize((out.width*4,out.height*4),Image.Resampling.NEAREST).save(path.with_name(path.stem.replace('1x','4x')+'.png'))

scene_path=ROOT/'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz';z=np.load(scene_path);prior=z['scenePrior'];finite=np.isfinite(z['sceneDistance']);source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2];src=source[...,:3].astype(float)/255
acc_path=ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz';acc=np.load(acc_path);j=acc['alpha'];opaque=acc['opaque'];graph_path=source_path.with_name('face-graph.json');params=json.loads(graph_path.read_text())['operations'][0]['params'];masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()};protected=np.maximum.reduce([masks[k] for k in ['ears','hair','leftEye','rightEye','leftBrow','rightBrow','lips','teeth']])>=.05
full_path=ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/user-face-contour/full-point/alphas-full.npy';full=np.load(full_path)[2]
trace_root=ROOT/'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-trace-inverse';l=trace_root/'100/layers';st=json.loads((l/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw];a=np.fromfile(l/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw);f=np.asarray(Image.open(l/'foreground.png').convert('RGB')).astype(float)/255;core=(a>=.995)&(full[cs]>=.995)&(j[cs]<.001)&~protected[cs]
# The 6px neighborhood matches three original parser cells; no per-photo
# tuning. Full/crop agreement and stationary material veto define candidates.
cell=max(w/params['semanticMasks']['skin']['width'],h/params['semanticMasks']['skin']['height']);radius=int(np.ceil(3*cell));allowed=finite[cs]&(j[cs]<.001)&~protected[cs]&(full[cs]>.05)
Bsource=np.floor(prior[cs]*255+.5)/255
r=solve(src[cs],a.astype(float),f,Bsource,core,allowed,radius=radius)
transition=np.clip((distance_transform_edt(finite)-1)/8,0,1);transition=transition*transition*(3-2*transition)
Ffloat=f+(r['foreground']-f)*transition[cs][...,None];Fnew=np.floor(np.clip(Ffloat,0,1)*255+.5).astype('float32');Fbase=np.floor(f*255+.5).astype('float32');source_changed=np.any(Fnew!=Fbase,axis=2)
Image.fromarray(Fnew.astype('uint8')).save(OUT/'source-foreground-corrected.png');Image.fromarray(source_changed.astype('uint8')*255).save(OUT/'source-foreground-changed.png');Image.fromarray(r['clipped'].astype('uint8')*255).save(OUT/'source-color-bound-clipped.png');np.savez_compressed(OUT/'source-solve.npz',**r,skinCore=core,allowed=allowed,sourceForegroundByte=Fbase,correctedForegroundByte=Fnew,alpha=a,transition=transition[cs])
report={'scope':__doc__,'accepted':False,'solverSha256':sha(HERE/'solver.py'),'runSha256':sha(Path(__file__)),'sourceSha256':sha(source_path),'sceneSha256':sha(scene_path),'sourceAlphaSha256':sha(l/'source-alpha-f32le.bin'),'fullSkinSha256':sha(full_path),'algorithm':{'radiusPx':radius,'minimumInverseAlpha':.1,'colorReliabilityRamp':[.1,.2],'opaqueCoreAlpha':.995,'jewelryExclusion':.001,'fullSkinSupport':.05,'colorBounds':'source-skin RGB min/max +/- 1 byte, fixed 3 parser cells','BWriteTransitionPx':8,'nonWorseningSourceFit':'refuse a bounded correction if any source RGB residual channel increases before byte quantization'},'sourceDomainPixels':int(r['domain'].sum()),'sourceChangedPixels':int(source_changed.sum()),'boundedColorClippedPixels':int(r['clipped'].sum()),'nonWorseningRefusedPixels':int(r['nonWorseningRefused'].sum()),'rows':[]}
for amount in [100,-100,0]:
 d=OUT/str(amount);d.mkdir();tr=trace_root/str(amount if amount else 100);l=tr/'layers';af=np.fromfile(l/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw);assert np.array_equal(a,af);assert np.array_equal(Fbase,np.asarray(Image.open(l/'foreground.png').convert('RGB')).astype('float32'));target=np.fromfile(l/'target-alpha-f32le.bin',dtype='<f4').reshape(ch,cw);samples=np.fromfile(l/'samples-f32le.bin',dtype='<f4').reshape(ch,cw,5);xy=np.fromfile(l/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2);changed=np.fromfile(l/'changed.bin',dtype='uint8').reshape(ch,cw)>0;frozen=np.fromfile(l/'frozen.bin',dtype='uint8').reshape(ch,cw)>0;c=np.asarray(Image.open(l/'source-crop.png').convert('RGB')).astype('float32')
 b_only_root=ROOT/'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-scene-b-only-v2-feather'/str(amount);b=np.load(b_only_root/'background-replaced.npy').astype('float32');base=np.asarray(Image.open(b_only_root/'candidate.png').convert('RGBA'));v37=np.asarray(Image.open(tr/'candidate.png').convert('RGBA')) if amount else source
 if amount:
  sampled_base,alpha_base=sample_f32(Fbase,a,xy);sample_diff=float(np.max(abs(sampled_base[changed]-samples[...,:3][changed])));alpha_diff=float(np.max(abs(alpha_base[changed]-target[changed])));assert sample_diff==0,sample_diff;assert alpha_diff==0,alpha_diff
 else:
  yy,xx=np.indices(a.shape);xy=np.stack([xx,yy],-1).astype('float32');target=a.copy();changed=np.ones_like(changed);sample_diff=alpha_diff=0
 sampled_new,anew=sample_f32(Fnew,a,xy);assert np.array_equal(anew[changed],target[changed]);output=base.copy();write=finite[cs]&changed&~frozen&~opaque[cs];result=compose(c,Fnew,b,a,target,sampled_new);patch=output[cs][...,:3].copy();patch[write]=result[write];output[cs][...,:3]=patch;fullwrite=np.zeros((h,w),bool);fullwrite[cs]=write;diff=np.any(output!=base,axis=2)
 assert not (diff&~fullwrite).any();assert np.array_equal(output[opaque],base[opaque]);assert np.array_equal(output[...,3],base[...,3]);assert not diff[cs][frozen].any()
 if not amount:assert np.array_equal(output,source)
 Image.fromarray(output).save(d/'candidate.png');Image.fromarray(diff.astype('uint8')*255).save(d/'changed-from-b-only.png')
 for name,box in [('face',[287,197,730,709]),('left',[320,400,420,550]),('right',[590,400,700,550])]:panel(d/f'{name}-1x.png',[source,v37,base,output],['Source','Frozen V37','B only','Bounded F + B'],box)
 points=[]
 for x,y in [(373,450),(374,450),(375,450),(640,470),(641,470),(642,470),(630,503)]:
  xx,yy=x-ox,y-oy;points.append({'xy':[x,y],'sourceAlpha':float(a[yy,xx]),'domain':bool(r['domain'][yy,xx]),'colorBoundClipped':bool(r['clipped'][yy,xx]),'oldF':list(map(int,Fbase[yy,xx])),'rawSolvedF':list(map(float,r['rawSolve'][yy,xx]*255)),'newF':list(map(int,Fnew[yy,xx])),'BonlyRGB':list(map(int,base[y,x,:3])),'newRGB':list(map(int,output[y,x,:3]))})
 report['rows'].append({'amount':amount,'sampleBaseFloatMaxDiff':sample_diff,'targetAlphaMaxDiff':alpha_diff,'outputChangedFromBonly':int(diff.sum()),'outsideWriteChanged':0,'opaqueChanged':0,'outputAlphaChanged':0,'zeroExact':bool(np.array_equal(output,source)) if not amount else None,'points':points,'sourceAlphaSha256':sha(l/'source-alpha-f32le.bin'),'inverseSha256':sha(l/'inverse-source-xy-f32le.bin'),'geometrySha256':sha(tr/'debug/flow-geometry.json'),'outputSha256':sha(d/'candidate.png')})
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
