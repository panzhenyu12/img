"""Nearest same-side scene boundary continuation, as a falsifiable diagnostic.

No neural generation and no foreground writes. A row profile at three vertical
scales is validated by holding out the three closest observed scene columns.
The hidden strip is still an extrapolation hypothesis, never observed truth.
"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,gaussian_filter1d
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import scene_donor as base
from compose_real import sample,montage
from run_real import probe,sha,mask_save


def profile(rgb,safe,box,side,skip=0):
 h,w=safe.shape;x0,y0,x1,y1=box
 yy=np.arange(max(0,y0-12),min(h,y1+12));values=[];used=np.zeros_like(safe);bound=[]
 for y in yy:
  xs=np.flatnonzero(safe[y] & (np.arange(w)<x0 if side<0 else np.arange(w)>=x1))
  xs=xs[np.abs(xs-(x0 if side<0 else x1))<90]
  if side<0:xs=xs[::-1]
  if len(xs)<skip+9:raise RuntimeError('Insufficient local observed scene for a row profile')
  chosen=xs[skip:skip+9];values.append(np.median(rgb[y,chosen],axis=0));used[y,chosen]=True;bound.append(int(xs[0]))
 return yy,np.asarray(values),used,np.asarray(bound)


def main():
 out=HERE/'scene-donor-v2-local';out.mkdir(exist_ok=False)
 sp=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';gp=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json';source=np.array(Image.open(sp).convert('RGBA'));rgb=source[...,:3].astype(float);h,w=source.shape[:2]
 p=json.loads(gp.read_text())['operations'][0]['params'];m={k:probe.zim.coverage(v,w,h) for k,v in p['semanticMasks'].items()};data=np.load(HERE/'outputs-v2/combined-point/source-layers.npz');support=data['pure']|data['trusted'];alpha=data['alpha'];metal=data['metal'];skin=data['skin']
 safe=np.asarray(Image.open(HERE/'scene-donor-v1/safe-scene-mask.png'))>0;protected=np.asarray(Image.open(HERE/'scene-donor-v1/protected-mask.png'))>0;eroded=binary_erosion(safe,iterations=2)
 report={'accepted':False,'method':'Nearest same-side observed scene row-profile donor, median of 9 columns, vertical scales sigma .5/1/2 selected by 3-column observed boundary holdout. Hidden disocclusion is a hypothesis; no foreground/alpha/map changes.', 'sourceSha256':sha(sp),'graphSha256':sha(gp),'scriptSha256':sha(Path(__file__)),'rows':[]}
 for amount in [100,-100]:
  d=out/str(amount);d.mkdir();prev=HERE/'scene-donor-v1'/str(amount);target=np.asarray(Image.open(prev/'target-mask.png'))>0;initial=np.array(Image.open(prev/'frozen-v2.png').convert('RGBA'));v36=np.array(Image.open(prev/'v36.png').convert('RGBA'))
  mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';L=mat/'layers';st=json.loads((L/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
  xy=np.fromfile(L/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);under=np.fromfile(L/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype(float);fg=np.asarray(Image.open(L/'foreground.png').convert('RGB'),dtype=float)/255;under[support[cs]]=1;fg[support[cs]]=skin[cs][support[cs]];new_alpha=sample(under,xy)[...,0];new_rgb=sample(fg*under[...,None],xy)
  candidate=initial.copy();donor_mask=np.zeros_like(safe);backing=np.zeros_like(rgb);selections=[]
  for idx,box in enumerate(base.BOXES,1):
   side=-1 if idx==1 else 1;x0,y0,x1,y1=box;comp=np.zeros_like(safe);comp[y0-10:y1+10,x0-10:x1+10]=True;ct=comp&target
   yy,colors,used,bound=profile(rgb,eroded,box,side);_,train,_,_=profile(rgb,eroded,box,side,3)
   holdout=np.stack([np.mean(rgb[y,np.arange(b,b+3) if side>0 else np.arange(b,b-3,-1)],axis=0) for y,b in zip(yy,bound)])
   rows=[]
   for sig in [.5,1.,2.]:
    val=gaussian_filter1d(train,sig,axis=0,mode='nearest');res=float(np.sqrt(np.mean((val-holdout)**2)));grad=float(np.sqrt(np.mean((np.diff(val,axis=0)-np.diff(holdout,axis=0))**2)));rows.append((res+grad,sig,res,grad))
   rows.sort();_,sig,rmse,grad=rows[0];estimate=gaussian_filter1d(colors,sig,axis=0,mode='nearest');y,x=np.where(ct);rowindex=y-yy[0];backing[y,x]=estimate[rowindex];donor_mask|=used
   distance=np.abs(x-bound[rowindex]);selections.append({'component':idx,'targetPixels':int(ct.sum()),'donorPixels':int(used.sum()),'verticalSigma':sig,'boundaryHoldoutRmse':rmse,'boundaryHoldoutGradientRmse':grad,'hiddenDistanceMax':int(distance.max()) if len(x) else 0,'hiddenDistanceMedian':float(np.median(distance)) if len(x) else 0,'observedBoundaryAccepted':rmse<=8 and grad<=2,'hiddenBackgroundVerified':False})
  fixed=np.where(m['accessories'][cs]>=.05,alpha[cs],0);rendered=fixed[...,None]*metal[cs]+(1-fixed[...,None])*(new_rgb+(1-new_alpha[...,None])*backing[cs]/255)
  patch=candidate[cs][...,:3].copy();patch[target[cs]]=np.floor(np.clip(rendered[target[cs]],0,1)*255+.5).astype('uint8');candidate[cs][...,:3]=patch
  changed=np.any(candidate!=initial,axis=2);assert not (changed&~target).any();assert not (changed&protected).any();assert not (donor_mask&~safe).any();assert np.array_equal(candidate[...,3],source[...,3])
  Image.fromarray(candidate).save(d/'candidate.png');mask_save(d/'target-mask.png',target);mask_save(d/'donor-mask.png',donor_mask);Image.fromarray(np.floor(np.clip(backing,0,255)+.5).astype('uint8')).save(d/'scene-estimate.png')
  overlay=initial.copy();overlay[target,:3]=[255,0,0];overlay[donor_mask,:3]=[0,255,255];overlay[protected,:3]=(overlay[protected,:3]*.4+np.array([0,160,0])*.6).astype('uint8');Image.fromarray(overlay).save(d/'target-donor-protection-overlay.png');Image.fromarray(np.clip(np.abs(candidate[...,:3].astype(float)-initial[...,:3])*8,0,255).astype('uint8')).save(d/'difference-x8.png')
  montage(d/'source-vs-v36-1x.png',[Image.fromarray(source),Image.fromarray(v36)],['Source','V36'],[287,197,730,709]);montage(d/'v36-vs-candidate-1x.png',[Image.fromarray(v36),Image.fromarray(initial),Image.fromarray(candidate)],['V36','Frozen v2','Local scene donor'],[287,197,730,709])
  for idx,box in enumerate(base.BOXES,1):
   x0,y0,x1,y1=box;crop=[x0-12,y0-12,x1+12,y1+40];montage(d/f'component-{idx}-1x.png',[Image.fromarray(source),Image.fromarray(v36),Image.fromarray(initial),Image.fromarray(candidate),Image.fromarray(overlay)],['Source','V36','v2','Local donor','Gate'],crop)
  (d/'donor-selection.json').write_text(json.dumps(selections,indent=2)+'\n')
  row={'amount':amount,'targetPixels':int(target.sum()),'changedPixels':int(changed.sum()),'donorPixels':int(donor_mask.sum()),'protectedChangedPixels':int((changed&protected).sum()),'outsideTargetChangedPixels':int((changed&~target).sum()),'boundaryBefore':base.boundary_gradient(initial[...,:3],target),'boundaryAfter':base.boundary_gradient(candidate[...,:3],target),'inverseSha256':sha(L/'inverse-source-xy-f32le.bin'),'outputSha256':sha(d/'candidate.png'),'directory':str(d.relative_to(HERE))};report['rows'].append(row);print(json.dumps(row),flush=True)
 report['visualConclusion']='Pending independent image inspection; no auto-accept from byte invariants or observed-scene holdout.'
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
