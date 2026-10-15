"""Model-free scene donor diagnostic; frozen v2 layers/geometry/alpha.

This is a restricted backing-only experiment, not an accepted implementation.
A coherent translated patch is selected using observed scene on three scales.
Nothing is generated and no donor may include semantic foreground.
"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha,mask_save
from compose_real import sample,montage
cv2=probe.cv2
BOXES=[[338,421,385,500],[632,426,680,503]]


def boundary_gradient(image,mask):
    values=[]
    for axis in (0,1):
        a=np.take(mask,range(mask.shape[axis]-1),axis=axis)
        b=np.take(mask,range(1,mask.shape[axis]),axis=axis)
        edge=a!=b
        if edge.any():
            delta=np.diff(image.astype(float),axis=axis)
            values.extend(np.linalg.norm(delta[edge],axis=-1).tolist())
    return {'mean':float(np.mean(values)) if values else 0,'p95':float(np.percentile(values,95)) if values else 0,'count':len(values)}


def choose_donor(rgb,safe,target,box,side):
    h,w=safe.shape; y,x=np.where(target)
    if not len(x): return None,{'accepted':False,'reason':'No background-exposed old-strip pixels survive the fixed-layer gate.'}
    x0,y0,x1,y1=box
    region=np.zeros_like(safe); region[max(0,y0-22):min(h,y1+22),max(0,x0-70):min(w,x1+70)]=True
    # Context and every target donor remain on the same exterior side.
    xx=np.indices(safe.shape)[1]
    exterior=xx<x0 if side<0 else xx>=x1
    anchors=safe&region&exterior
    ay,ax=np.where(anchors)
    erosion=binary_erosion(safe,iterations=2)
    pyramids=[]
    for sigma in [0.,1.,2.]:
        if sigma==0: pyramids.append(rgb)
        else:
            weight=cv2.GaussianBlur(safe.astype('float32'),(0,0),sigma)
            blur=cv2.GaussianBlur((rgb*safe[...,None]).astype('float32'),(0,0),sigma)
            pyramids.append(blur/np.maximum(weight[...,None],1e-6))
    rows=[]
    for dx0 in range(8,101,2):
      dx=side*dx0
      for dy in range(-8,9,2):
        sx=x+dx;sy=y+dy
        if ((sx<0)|(sx>=w)|(sy<0)|(sy>=h)).any() or not erosion[sy,sx].all(): continue
        if not exterior[sy,sx].all(): continue
        bx=ax+dx;by=ay+dy
        valid=(bx>=0)&(bx<w)&(by>=0)&(by<h)
        valid[valid]&=erosion[by[valid],bx[valid]]&exterior[by[valid],bx[valid]]
        if valid.sum()<50: continue
        qx,qy=ax[valid],ay[valid];rx,ry=bx[valid],by[valid]
        # Match only observed scene. Normalize channels to byte units.
        residuals=[float(np.sqrt(np.mean((im[qy,qx]-im[ry,rx])**2))) for im in pyramids]
        dximg=cv2.Sobel(pyramids[-1],cv2.CV_32F,1,0,ksize=3)/8
        dyimg=cv2.Sobel(pyramids[-1],cv2.CV_32F,0,1,ksize=3)/8
        gradient=float(np.sqrt(np.mean((dximg[qy,qx]-dximg[ry,rx])**2+(dyimg[qy,qx]-dyimg[ry,rx])**2)))
        score=float(np.mean(residuals)+gradient*2+np.hypot(dx,dy)*.015)
        rows.append((score,dx,dy,residuals,gradient,int(valid.sum())))
    if not rows:return None,{'accepted':False,'reason':'No coherent source patch can cover the target entirely from observed same-side scene.'}
    rows.sort(key=lambda q:q[0]);score,dx,dy,res,grad,n=rows[0]
    full=np.zeros_like(rgb); full[y,x]=rgb[y+dy,x+dx]
    donors=np.zeros_like(safe);donors[y+dy,x+dx]=True
    accepted=max(res)<=8 and grad<=2
    result={'accepted':bool(accepted),'reason':None if accepted else 'Same-side observed-scene color or gradient mismatch exceeds the conservative donor gate.',
      'translation':[dx,dy],'donorBox':[int((x+dx).min()),int((y+dy).min()),int((x+dx).max()+1),int((y+dy).max()+1)],
      'targetPixels':int(target.sum()),'donorPixels':int(donors.sum()),'matchAnchorPixels':n,'colorRmseAtSigma0_1_2':res,'gradientRmse':grad,'score':score,'validCandidates':len(rows),
      'thresholds':{'colorRmseMax':8,'gradientRmseMax':2},'top5':[{'translation':[q[1],q[2]],'score':q[0],'colorRmse':q[3],'gradientRmse':q[4]} for q in rows[:5]]}
    return (full,donors),result


def main():
 out=HERE/'scene-donor-v1';out.mkdir(exist_ok=False)
 sp=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';gp=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'
 source=np.array(Image.open(sp).convert('RGBA'));rgb=source[...,:3].astype('float32');h,w=source.shape[:2]
 params=json.loads(gp.read_text())['operations'][0]['params'];m={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
 subject=probe.zim.coverage(params['subjectMask'],w,h)
 safe=(subject<=.01)&(np.maximum.reduce([m[k] for k in ['face','faceOutline','faceContourMask','skin','neck','ears','accessories','hair'] if k in m])<.01)
 data=np.load(HERE/'outputs-v2/combined-point/source-layers.npz');alpha=data['alpha'];metal=data['metal'];skin=data['skin'];support=data['pure']|data['trusted']
 # All corrected accessory alpha > .05 is protected, including soft metal.
 protected=(m['ears']>=.05)|(m['hair']>=.05)|((m['accessories']>=.05)&(alpha>.05))|data['opaque']
 report={'accepted':False,'scope':'Isolated same-side scene patch replacement in old source-supported exposed skin strip. V2 source/metal/alpha/inverse stay frozen; only background color is changed. Old semantic accessory protection is also evaluated as a strict no-op control.',
  'method':'Deterministic coherent translated donor; masked Gaussian scales sigma 0/1/2; same-side scene color and gradient scoring. No neural model, no alpha adjustment, no Poisson diffusion through foreground.',
  'sourceSha256':sha(sp),'graphSha256':sha(gp),'sourceLayersSha256':sha(HERE/'outputs-v2/combined-point/source-layers.npz'),'safeScenePixels':int(safe.sum()),'scriptSha256':sha(Path(__file__)),'rows':[]}
 mask_save(out/'safe-scene-mask.png',safe);mask_save(out/'protected-mask.png',protected)
 for amount in [100,-100]:
  d=out/str(amount);d.mkdir();mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';L=mat/'layers';st=json.loads((L/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
  old=HERE/f'outputs-v2/combined-point/{amount}-original-backing';base=np.array(Image.open(old/'candidate.png').convert('RGBA'));v36=np.array(Image.open(mat/'candidate.png').convert('RGBA'))
  write=np.asarray(Image.open(old/'allowed-write.png'))>0;unknown=np.asarray(Image.open(old/'unknown-new-coverage.png'))>0;insufficient=np.asarray(Image.open(old/'insufficient-scene-support.png'))>0
  xy=np.fromfile(L/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);physical=np.fromfile(L/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype('float64');fg=np.asarray(Image.open(L/'foreground.png').convert('RGB'),dtype='float64')/255
  under=physical.copy();under[support[cs]]=1;fg[support[cs]]=skin[cs][support[cs]]
  new_alpha=sample(under,xy)[...,0];new_rgb=sample(fg*under[...,None],xy)
  target=np.zeros((h,w),bool);target[cs]=support[cs]&write&~unknown&~insufficient&(new_alpha<.999)&~protected[cs]
  old_semantic_control=target&(m['accessories']<.05)
  candidate=base.copy();accepted_candidate=base.copy();donor_mask=np.zeros_like(safe);selections=[]
  for idx,box in enumerate(BOXES,1):
   x0,y0,x1,y1=box;comp=np.zeros_like(safe);comp[max(0,y0-10):min(h,y1+10),max(0,x0-10):min(w,x1+10)]=True;component_target=target&comp
   proposal,row=choose_donor(rgb,safe,component_target,box,-1 if idx==1 else 1);row['component']=idx;selections.append(row)
   if proposal is None:continue
   donor,used=proposal;donor_mask|=used
   backing=donor[cs]/255
   fixed=np.where(m['accessories'][cs]>=.05,alpha[cs],0)
   rendered=fixed[...,None]*metal[cs]+(1-fixed[...,None])*(new_rgb+(1-new_alpha[...,None])*backing)
   patch=candidate[cs][...,:3].copy();patch[component_target[cs]]=np.floor(np.clip(rendered[component_target[cs]],0,1)*255+.5).astype('uint8');candidate[cs][...,:3]=patch
   if row['accepted']:accepted_candidate[component_target]=candidate[component_target]
  changed=np.any(candidate!=base,axis=2)
  assert not (changed&~target).any() and not (changed&protected).any()
  assert not (donor_mask&~safe).any();assert np.array_equal(candidate[...,3],base[...,3])
  Image.fromarray(candidate).save(d/'candidate.png');Image.fromarray(accepted_candidate).save(d/'donor-gated-candidate.png')
  Image.fromarray(base).save(d/'frozen-v2.png');Image.fromarray(v36).save(d/'v36.png')
  mask_save(d/'target-mask.png',target);mask_save(d/'donor-mask.png',donor_mask);mask_save(d/'old-semantic-control-target.png',old_semantic_control)
  diff=np.clip(np.abs(candidate[...,:3].astype(float)-base[...,:3])*8,0,255).astype('uint8');Image.fromarray(diff).save(d/'difference-x8.png')
  overlay=base.copy();overlay[target,:3]=[255,0,0];overlay[donor_mask,:3]=[0,255,255];overlay[protected,:3]=(overlay[protected,:3]*.4+np.array([0,160,0])*.6).astype('uint8')
  Image.fromarray(overlay).save(d/'target-donor-protection-overlay.png')
  montage(d/'source-vs-v36-1x.png',[Image.fromarray(source),Image.fromarray(v36)],['Source','V36'],[287,197,730,709])
  montage(d/'v36-vs-candidate-1x.png',[Image.fromarray(v36),Image.fromarray(base),Image.fromarray(candidate)],['V36','Frozen v2','Scene donor diagnostic'],[287,197,730,709])
  for idx,box in enumerate(BOXES,1):
   x0,y0,x1,y1=box;crop=[x0-12,y0-12,x1+12,y1+40]
   montage(d/f'component-{idx}-1x.png',[Image.fromarray(source),Image.fromarray(v36),Image.fromarray(base),Image.fromarray(candidate),Image.fromarray(overlay)],['Source','V36','v2','Donor','Gate'],crop)
  (d/'donor-selection.json').write_text(json.dumps(selections,indent=2)+'\n')
  row={'amount':amount,'targetPixels':int(target.sum()),'changedPixels':int(changed.sum()),'oldSemanticStrictControlPixels':int(old_semantic_control.sum()),'donorPixels':int(donor_mask.sum()),'donorOutsideScenePixels':int((donor_mask&~safe).sum()),'protectedChangedPixels':int((changed&protected).sum()),'outsideTargetChangedPixels':int((changed&~target).sum()),'oldUnknownChangedPixels':int((changed[cs]&unknown).sum()),'oldInsufficientChangedPixels':int((changed[cs]&insufficient).sum()),'boundaryBefore':boundary_gradient(base[...,:3],target),'boundaryAfter':boundary_gradient(candidate[...,:3],target),'donorAcceptedComponents':sum(int(r['accepted']) for r in selections),'inverseSha256':sha(L/'inverse-source-xy-f32le.bin'),'outputSha256':sha(d/'candidate.png'),'directory':str(d.relative_to(HERE))}
  row['boundaryGradientWorsened']=row['boundaryAfter']['mean']>row['boundaryBefore']['mean']+.1
  report['rows'].append(row);print(json.dumps(row),flush=True)
 report['visualConclusion']='Pending independent image inspection; no auto-accept from invariant tests.'
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
