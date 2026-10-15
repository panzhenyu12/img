"""Source-only scene support using independent skin alpha agreement.

No warp or output image change. Exterior scene may extend through parser
accessory overreach only with direct full/crop skin absence, corrected jewelry
absence, ear/hair protection, and local RGB-connected observed scene support.
"""
from pathlib import Path
import sys,json,collections
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation,binary_propagation,distance_transform_edt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha,mask_save
from decompose import local_prior
from compose_real import montage
cv2=probe.cv2
BOXES=[[338,421,385,500],[632,426,680,503]]


def color_connected(seed,candidate,lab):
 """4-connected flood bounded by direct evidence and adjacent Lab difference."""
 allowed=seed|candidate;reached=seed.copy()
 boundary=seed&binary_dilation(candidate,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool))
 q=collections.deque(map(tuple,np.argwhere(boundary)));h,w=seed.shape
 while q:
  y,x=q.popleft()
  for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
   ny,nx=y+dy,x+dx
   if ny<0 or ny>=h or nx<0 or nx>=w or reached[ny,nx] or not allowed[ny,nx]:continue
   delta=lab[ny,nx]-lab[y,x]
   if np.linalg.norm(delta)>8 or abs(delta[0])>6 or np.linalg.norm(delta[1:])>6:continue
   reached[ny,nx]=True;q.append((ny,nx))
 return reached


def stats(values):
 return {'count':int(values.size),'min':float(values.min()) if values.size else None,'median':float(np.median(values)) if values.size else None,'p95':float(np.percentile(values,95)) if values.size else None,'max':float(values.max()) if values.size else None}


def main():
 out=HERE/'outputs-v2-consistent';out.mkdir(exist_ok=False)
 sp=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';gp=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json';source=np.asarray(Image.open(sp).convert('RGB'));rgb=source.astype(float)/255;h,w=source.shape[:2]
 p=json.loads(gp.read_text())['operations'][0]['params'];m={k:probe.zim.coverage(v,w,h) for k,v in p['semanticMasks'].items()};sub=probe.zim.coverage(p['subjectMask'],w,h)
 inputs=ROOT/'experiments/joint-skin-matting-v1-20261015/outputs';full=np.load(inputs/'full-point/alphas-full.npy')[2];crop=np.load(inputs/'crop-point/alphas-full.npy')[2]
 acc_path=ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz';acc=np.load(acc_path);newj=acc['alpha'];support=acc['pure']|acc['trusted']
 strict=(sub<=.01)&(np.maximum.reduce([m[k] for k in ['face','faceOutline','faceContourMask','skin','neck','ears','accessories','hair'] if k in m])<.01)
 features=np.maximum.reduce([m[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']]);protected=(newj>=.01)|(m['hair']>=.05)|(m['ears']>=.05)|(features>=.05)
 legacy_strict=strict.copy()
 eligibility=(full<.01)&(crop<.01)&~protected
 strict &= eligibility
 lab=cv2.cvtColor(rgb.astype('float32'),cv2.COLOR_RGB2LAB)
 report={'scope':'Source-only same-side observed scene evidence. No edited image, geometry, alpha or source skin writes. Direct full/crop skin agreement plus corrected accessory exclusion, protected ears/hair/features and RGB-connected exterior scene.',
  'sourceSha256':sha(sp),'graphSha256':sha(gp),'fullSkinAlphaSha256':sha(inputs/'full-point/alphas-full.npy'),'cropSkinAlphaSha256':sha(inputs/'crop-point/alphas-full.npy'),'accessoryLayersSha256':sha(acc_path),'scriptSha256':sha(Path(__file__)),'acceptedForProduction':False,'rows':[]}
 np.savez_compressed(out/'inputs.npz',fullSkin=full,cropSkin=crop,accessory=newj,strictScene=strict,legacyStrictScene=legacy_strict,protected=protected)
 all_new=np.zeros_like(strict);prior_full=rgb.copy();distance_full=np.full((h,w),np.inf,dtype='float32');nearest_full=np.zeros((2,h,w),dtype='int32')
 for component,box in enumerate(BOXES,1):
  x0,y0,x1,y1=box;pad=32;context=[max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad)];rx0,ry0,rx1,ry1=context;sl=np.s_[ry0:ry1,rx0:rx1]
  roi=np.zeros_like(strict);roi[sl]=True
  candidate=roi&eligibility
  # Seeds are physically observed exterior background; parser may be vetoed
  # only for newly proposed pixels supported by both independent skin passes.
  seed=strict&roi
  connected=binary_propagation(seed,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool),mask=seed|candidate)
  reached=color_connected(seed[sl],candidate[sl],lab[sl]);rgb_connected=np.zeros_like(strict);rgb_connected[sl]=reached
  added=rgb_connected&~strict;scene=(strict&roi)|added
  scene_prior,distance=local_prior(rgb,scene,2);old_prior,old_distance=local_prior(rgb,legacy_strict&roi,2)
  nearest=distance_transform_edt(~scene,return_distances=False,return_indices=True)
  prior_full[roi]=scene_prior[roi];distance_full[roi]=distance[roi];nearest_full[:,roi]=nearest[:,roi];all_new|=added
  dd=out/f'component-{component}';dd.mkdir()
  for n,mask in [('candidate',candidate),('topology-connected',connected),('rgb-connected-scene',scene),('added-scene',added),('old-scene',seed),('protected',protected&roi),('disconnected',candidate&~connected),('rgb-refused',connected&~rgb_connected)]:mask_save(dd/(n+'.png'),mask)
  np.savez_compressed(dd/'scene.npz',scene=scene,added=added,prior=scene_prior.astype('float32'),distance=distance.astype('float32'),oldPrior=old_prior.astype('float32'),oldDistance=old_distance.astype('float32'),nearest=nearest.astype('int32'),region=roi)
  Image.fromarray(np.floor(np.clip(scene_prior,0,1)*255+.5).astype('uint8')).save(dd/'scene-prior.png')
  overlay=source.copy();overlay[added]=(overlay[added]*.25+np.array([0,255,255])*.75).astype('uint8');overlay[protected&roi]=(overlay[protected&roi]*.5+np.array([0,180,0])*.5).astype('uint8')
  prior_display=source.copy();prior_display[roi]=np.floor(scene_prior[roi]*255+.5).astype('uint8')
  montage(dd/'source-evidence-1x.png',[Image.fromarray(source),Image.fromarray(overlay),Image.fromarray(prior_display)],['Source','Cyan added / green protected','Same-side scene prior'],context)
  target=roi&support
  alpha_band=roi&(full>.01)&(crop>.01)&(full<.99)&(crop<.99)&~protected
  row={'component':component,'context':context,'candidatePixels':int(candidate.sum()),'legacySeedRejectedDirectConflict':int((legacy_strict&roi&~eligibility).sum()),'addedScenePixels':int(added.sum()),'addedOldAccessoryPixels':int((added&(m['accessories']>=.05)).sum()),'addedOldSubjectPixels':int((added&(sub>.01)).sum()),'addedProtectedPixels':int((added&protected).sum()),'addedSkinAlphaMax':float(np.maximum(full,crop)[added].max()) if added.any() else None,'addedJewelryAlphaMax':float(newj[added].max()) if added.any() else None,'rgbRejectedPixels':int((connected&~rgb_connected).sum()),'disconnectedPixels':int((candidate&~connected).sum()),'sourceSupportOldDistance':stats(old_distance[target]),'sourceSupportNewDistance':stats(distance[target]),'directMixedBandOldDistance':stats(old_distance[alpha_band]),'directMixedBandNewDistance':stats(distance[alpha_band]),'distanceReductionAtSourceSupport':stats(old_distance[target]-distance[target]),'directory':str(dd.relative_to(HERE))}
  report['rows'].append(row);print(json.dumps(row),flush=True)
 np.savez_compressed(out/'scene-support.npz',addedScene=all_new,scenePrior=prior_full.astype('float32'),sceneDistance=distance_full,nearest=nearest_full,strictScene=strict,legacyStrictScene=legacy_strict,newScene=strict|all_new)
 mask_save(out/'added-scene.png',all_new);mask_save(out/'new-scene.png',strict|all_new)
 Image.fromarray(np.floor(np.clip(prior_full,0,1)*255+.5).astype('uint8')).save(out/'scene-prior.png')
 report['outputSha256']=sha(out/'scene-support.npz');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
