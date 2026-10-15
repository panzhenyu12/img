"""Known-layer demonstrations and frozen original-photo local feasibility audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from solver import solve,lower_bound
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'qa'
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import sample
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path,a):Image.fromarray(np.floor(np.clip(a,0,1)*255+.5).astype('uint8')).save(path)
def panel(path,ims,labels):
 h,w=ims[0].shape[:2];im=Image.new('RGB',(w*len(ims),h+22),'#20242a');d=ImageDraw.Draw(im)
 for i,(v,s) in enumerate(zip(ims,labels)):im.paste(Image.fromarray(np.floor(np.clip(v,0,1)*255+.5).astype('uint8')),(i*w,22));d.text((i*w+2,5),s,fill='white')
 im.save(path);im.resize((im.width*4,im.height*4),Image.Resampling.NEAREST).save(path.with_name(path.stem+'-4x.png'))
def composite(a,f,b):return a[...,None]*f+(1-a[...,None])*b
def oracle(a,f,b,xy):return sample(a[...,None]*f,xy)+(1-sample(a,xy)[...,0,None])*b
def delta(c,a,f,b,xy):return c+sample(a[...,None]*f,xy)-a[...,None]*f+(a-sample(a,xy)[...,0])[...,None]*b
h,w=96,144;yy,xx=np.indices((h,w));a=np.clip((80+2*np.sin(yy/15)-xx+1)/2,0,1);xy=np.stack([xx+12,yy],-1).astype(float);valid=(yy>4)&(yy<h-5)&(xx>4)&(xx<w-20);b=np.zeros((h,w,3))+[.035,.045,.055];f=np.zeros_like(b)+[1,.56,.37];prior=np.maximum(0,a-.02*((a>.01)&(a<.99)));c=composite(a,f,b);truth=oracle(a,f,b,xy);d=np.ones_like(a,dtype=bool)
redf=f.copy();pos=prior>0;redf[pos]=np.clip(b[pos]+(c[pos]-b[pos])/prior[pos,None],0,1);red=delta(c,prior,redf,b,xy);solution=solve(c,b,prior,f,d);green=delta(c,solution['alpha'],solution['foreground'],b,xy)
synthetic=[{'name':'clipped_highlight_known_B','redMaxByte':float(np.abs(red-truth)[valid].max()*255),'greenMaxByte':float(np.abs(green-truth)[valid].max()*255),'alphaMaxErrorAfter':float(abs(solution['alpha']-a).max())}];panel(OUT/'known-highlight.png',[c,truth,red,green],['Source','Known truth','Clipped prior','Feasible prior'])
# Corrected no-clipping control: prior is perturbed only away from vanishing a.
f2=np.zeros_like(b)+[.5,.32,.2];p2=a-.02*((a>.35)&(a<.65));c2=composite(a,f2,b);s2=solve(c2,b,p2,f2,d);g2=delta(c2,s2['alpha'],s2['foreground'],b,xy);t2=oracle(a,f2,b,xy)
synthetic.append({'name':'valid_nonsaturated_no_clip_control','outputMaxErrorByte':float(abs(g2-t2)[valid].max()*255),'priorChangedPixels':int(s2['changed'].sum()),'alphaErrorStillPresent':float(abs(s2['alpha']-a).max()),'interpretation':'Exact known constant B can cancel a feasible wrong alpha through color inversion; this is not alpha truth.'})
# Feasible-but-wrong alpha produces incorrect results against a spatial B.
b3=b.copy();b3[xx<76]=[.45,.4,.35];c3=composite(a,f2,b3);s3=solve(c3,b3,p2,f2,d);g3=delta(c3,s3['alpha'],s3['foreground'],b3,xy);t3=oracle(a,f2,b3,xy)
synthetic.append({'name':'feasible_wrong_alpha_spatial_scene','sourceReconstructionMaxByte':float(abs(composite(s3['alpha'],s3['foreground'],b3)-c3).max()*255),'outputMaxErrorByte':float(abs(g3-t3)[valid].max()*255),'priorChangedPixels':int(s3['changed'].sum()),'interpretation':'A valid gamut and perfect source fit do not identify alpha; changing the background under the warp exposes the error.'});panel(OUT/'nonidentifiable-spatial-scene.png',[c3,t3,g3,abs(g3-t3)*20],['Source','Known truth','Feasible prior','Error x20'])
# A scene-color miss of only 2/5/4 bytes at an actual background point can
# require alpha .6667. A material gate must refuse this, rather than fix alpha.
cb=np.array([[[17,10,2]]],float)/255;bb=np.array([[[19,15,6]]],float)/255;ab=np.array([[.0021]]);fb=np.zeros_like(cb)+[.7,.5,.4];unsafe=solve(cb,bb,ab,fb,np.ones_like(ab,bool));refuse=solve(cb,bb,ab,fb,np.zeros_like(ab,bool))
synthetic.append({'name':'wrong_dark_background_counterexample','trueMaterial':'scene, true skin alpha=0','sourceRGBByte':[17,10,2],'estimatedBackgroundRGBByte':[19,15,6],'priorAlpha':float(ab.item()),'unsafeFeasibleAlpha':float(unsafe['alpha'].item()),'refusedAlpha':float(refuse['alpha'].item()),'interpretation':'The lower bound is conditional on background and material truth; a 5-byte dark scene error can falsely raise alpha to 2/3.'})
# Original photo is evaluated only on independently inspected source skin
# paths. Widened domain violations are saved as REJECTED, never corrected.
srcpath=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(srcpath).convert('RGB'))/255;z=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/composite-delta-v2/source-layers.npz');ra,rf,rb,rj,local=[z[k] for k in ['skinAlpha','skin','background','jewelry','local']];manual=np.load(ROOT/'experiments/user-accessory-source-ownership-20261015/outputs/sparse-source-observation-labels.npy');confirmed=manual==1;lo=lower_bound(source,rb);violation=local&(rj<.001)&(lo>ra+.0001)
# No RGB refit unless alpha is actually infeasible: preserve existing solve.
correction=confirmed&(lo>ra+1e-8);rs=solve(source,rb,ra,rf,correction)
np.savez_compressed(OUT/'real-source-feasibility.npz',confirmed=confirmed,lowerBound=lo,prior=ra,correction=correction,rejectedWiderViolations=violation)
for comp,box in [(1,(340,420,395,510)),(2,(625,420,675,510))]:
 x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];overlay=source.copy();overlay[confirmed]=overlay[confirmed]*.35+np.array([0,1,.2])*.65;reject=source.copy();reject[violation]=reject[violation]*.35+np.array([1,0,.6])*.65
 panel(OUT/f'real-component-{comp}.png',[source[sl],overlay[sl],reject[sl]],['Source','Skin path','Reject'])
coords=np.argwhere(violation);order=np.argsort((lo-ra)[violation])[-12:][::-1];examples=[]
for i in order:
 y,x=coords[i];examples.append({'xy':[int(x),int(y)],'sourceRGBByte':list(map(int,np.rint(source[y,x]*255))),'estimatedSceneRGBByte':list(map(int,np.rint(rb[y,x]*255))),'priorAlpha':float(ra[y,x]),'feasibleLowerBound':float(lo[y,x])})
real={'sourceSha256':sha(srcpath),'confirmedSourceSkinPixels':int(confirmed.sum()),'confirmedFeasibilityViolations':int(correction.sum()),'confirmedAlphaChangedPixels':int(rs['changed'].sum()),'confirmedForegroundChangedPixels':int(np.any(rs['foreground']!=rf,axis=2).sum()),'rejectedWiderViolations':int(violation.sum()),'maximumConfirmedLowerMinusPrior':float((lo-ra)[confirmed].max()),'rejectedExamples':examples,'conclusion':'No applicable correction or visual improvement on independently inspected source skin paths. Wider violations predominantly dark scene/jewelry-hole pixels, where scene estimate is not truth. Not accepted for native integration.'}
report={'algorithmAccepted':False,'limitedKnownBackgroundSolverTestsPassed':True,'scope':'Conditional feasible skin RGB inversion; no material detection, no alpha identification, no production changes.','solverSha256':sha(HERE/'solver.py'),'synthetic':synthetic,'real':real};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
