"""Strict hybrid: 4x local alpha only for accessory component 2.

Component 1 remains the corrected full-image alpha. Source decomposition,
known support, frozen V36 inverse, unknown and insufficient gates are frozen
from v2. This is an isolated candidate; it never changes production.
"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import label,binary_dilation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha,mask_save
from compose_real import sample,montage

def main():
 out=HERE/'hybrid-crop2';out.mkdir(exist_ok=False)
 data=np.load(HERE/'outputs-v2/combined-point/source-layers.npz');decomp_support=data['pure']|data['trusted'];metal,skin=data['metal'],data['skin'];opaque=data['opaque'];old_alpha=data['alpha']
 crop_root=ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs/component-2-scale-4x/combined-point';local_row=json.loads((ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs/report.json').read_text())['rows'][1]['rows'][0]
 # Resolve local 4x crop directly from the recorded transform, never infer it.
 cr=json.loads((ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs/report.json').read_text())['rows'][1]['crop'];x0,y0,x1,y1=cr
 local=np.load(crop_root/'alphas-local.npy')[local_row['officialScoreArgmax']]
 hybrid=old_alpha.copy();hybrid[y0:y1,x0:x1]=local
 source=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'));h,w=source.shape[:2]
 params=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text())['operations'][0]['params'];m={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()};old_acc=m['accessories'];features=np.maximum.reduce([m[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']])
 diff=np.abs(hybrid-old_alpha)>.01;comp2_box=np.zeros((h,w),bool);comp2_box[y0:y1,x0:x1]=True
 # Reproduce the independent source-evidence ownership support (6px donor
 # radius) and connected component 2. This is 81 pixels, independent of
 # whether the crop alpha happened to differ at that exact pixel.
 from scipy.ndimage import distance_transform_edt
 anchors=(m['skin']>=.9)&(np.maximum.reduce([m['accessories'],m['hair'],m['ears'],features])<.05)
 distance,nearest=distance_transform_edt(~anchors,return_indices=True)
 rgb=source[...,:3].astype('float64')/255
 lab=probe.cv2.cvtColor(rgb.astype('float32'),probe.cv2.COLOR_RGB2LAB);cell=2.;count=probe.cv2.boxFilter(anchors.astype('float32'),-1,(5,5),normalize=False)
 means=np.stack([probe.cv2.boxFilter(lab[...,i]*anchors,-1,(5,5),normalize=False)/np.maximum(count,1) for i in range(3)],axis=-1);delta=lab-means[nearest[0],nearest[1]]
 source_evidence=((m['skin']>=.5)|((np.linalg.norm(delta,axis=-1)<=8)&(np.abs(delta[...,0])<=6)&(np.linalg.norm(delta[...,1:],axis=-1)<=6)))&(distance<=6)&(m['ears']<.05)&(features<.05)
 _,labels,_,_=probe.cv2.connectedComponentsWithStats((m['accessories']>=.5).astype('uint8'));comp2=(labels==2);support=source_evidence&comp2&comp2_box
 report={'scope':'Frozen v2 decomposition/gates and V36 inverse; component2 4x combined-point alpha replacement only. Component1 full alpha. No production changes.','accepted':False,'visualConclusion':'No clear removal of +100 soft double line; -100 still has edge teeth; reject candidate.','sourceSha256':sha(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'),'graphSha256':sha(ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'),'v2SourceLayersSha256':sha(HERE/'outputs-v2/combined-point/source-layers.npz'),'cropAlphaSha256':sha(crop_root/'alphas-local.npy'),'cropAlphaReportSha256':sha(ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs/report.json'),'cropAuditReportSha256':sha(ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs/audit-report.json'),'rows':[]}
 for amount in [100,-100]:
  mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';layers=mat/'layers';st=json.loads((layers/'state.json').read_text());cw,ch=st['width'],st['height'];ox,oy=st['origin'];cs=np.s_[oy:oy+ch,ox:ox+cw]
  xy=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);physical=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype('float64');fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype='float64')/255
  under=physical.copy();under[support[cs]]=1;fg[support[cs]]=skin[cs][support[cs]];new_alpha=sample(under,xy)[...,0];new_rgb=sample(fg*under[...,None],xy);transported=sample(support[cs].astype('float64'),xy)[...,0]>1e-6
  yy,xx=np.indices((ch,cw));moved=np.hypot(xy[...,0]-xx,xy[...,1]-yy)>1e-4
  allowed=(m['region'] if False else np.ones((h,w),bool))[cs]&((m['ears'][cs]<.05)&(m['hair'][cs]<.05)&(features[cs]<.05)&~opaque[cs]&moved)
  comp2_dest=np.zeros((ch,cw),bool);pad=24;rx0=max(ox,x0-pad);ry0=max(oy,y0-pad);rx1=min(ox+cw,x1+pad);ry1=min(oy+ch,y1+pad);comp2_dest[ry0-oy:ry1-oy,rx0-ox:rx1-ox]=True
  alpha_local=hybrid[cs]
  added=(new_alpha>under+.001)&(old_acc[cs]>=.05)&(alpha_local<=.05)&allowed&comp2_dest
  labels,_=label(added,structure=np.ones((3,3)));touch=binary_dilation((support&comp2)[cs],structure=np.ones((3,3)));ids=np.unique(labels[touch&added]);ids=ids[ids>0];connected=np.isin(labels,ids) if len(ids) else np.zeros_like(added);unknown=added&~connected
  prev=HERE/f'outputs-v2/combined-point/{amount}-original-backing';base=np.asarray(Image.open(prev/'candidate.png').convert('RGBA'));old_write=np.asarray(Image.open(prev/'allowed-write.png'))>0;insufficient=np.asarray(Image.open(prev/'insufficient-scene-support.png'))>0;old_unknown=np.asarray(Image.open(prev/'unknown-new-coverage.png'))>0
  writes=((support[cs]&comp2_dest)|transported|connected)&allowed&comp2_dest&~unknown&~old_unknown&~insufficient
  fixed=np.where(old_acc[cs]>=.05,alpha_local,0);bg=np.asarray(Image.open(layers/'background.png').convert('RGB'),dtype='float64')/255;beneath=new_rgb+(1-new_alpha[...,None])*bg;rendered=fixed[...,None]*metal[cs]+(1-fixed[...,None])*beneath
  output=base.copy();patch=output[cs][:,:,:3].copy();patch[writes]=np.floor(np.clip(rendered[writes],0,1)*255+.5).astype('uint8');output[cs][:,:,:3]=patch
  full=np.zeros((h,w),bool);full[cs]=writes;changed=np.any(output!=base,axis=2)
  assert not (changed&~full).any();assert np.array_equal(output[opaque],source[opaque]);assert np.array_equal(output[...,3],source[...,3])
  d=out/str(amount);d.mkdir();Image.fromarray(output).save(d/'candidate.png')
  for n,mask in [('hybrid-alpha-difference',diff),('component2-destination',comp2_dest),('allowed-write',writes),('newly-covered-connected',connected),('unknown-new-coverage',unknown),('old-unknown',old_unknown),('old-insufficient',insufficient)]: mask_save(d/(n+'.png'),mask if mask.shape== (ch,cw) else mask[oy:oy+ch,ox:ox+cw])
  montage(d/'face-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','v2 full alpha','Hybrid crop2 4x'],[287,197,730,709])
  for name,crop in [('left',[326,409,397,548]),('right',[620,414,692,551])]:montage(d/(name+'-1x.png'),[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','v2','Hybrid'],crop)
  row={'amount':amount,'component2AlphaChangedPixels':int(diff[comp2_box].sum()),'sourceSupportPixels':int(support.sum()),'decompositionSupportOverlapPixels':int((support&decomp_support).sum()),'writePixels':int(writes.sum()),'changedPixels':int(changed.sum()),'unknownNewCoveragePixels':int(unknown.sum()),'oldUnknownBlockedPixels':int((writes&old_unknown).sum()),'oldInsufficientBlockedPixels':int((writes&insufficient).sum()),'opaqueMetalChanged':0,'outsideWritesChanged':0,'inverseSha256':sha(layers/'inverse-source-xy-f32le.bin'),'outputSha256':sha(d/'candidate.png'),'directory':str(d.relative_to(HERE))};report['rows'].append(row);print(json.dumps(row))
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'rows':len(report['rows']),'accepted':False}))
if __name__=='__main__':main()
