"""Independent audit of final scene-only accessory candidates (v5).

Reads frozen source layers and saved candidates. It does not choose a better
looking size or direction and does not alter production code.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rgba=lambda p:np.asarray(Image.open(p).convert('RGBA'))
mask=lambda p:np.asarray(Image.open(p))>0
v5=EXP/'neural-backing-v5-virtual-scene'
report=read(v5/'report.json')
source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
graph_path=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'
assert sha(source_path)==report['sourceSha256']
assert sha(graph_path)==report['graphSha256']
source=rgba(source_path)
h,w=source.shape[:2]
params=read(graph_path)['operations'][0]['params']
masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
accessory=masks['accessories']>=.05
ears=masks['ears']>=.05
features=np.maximum.reduce([masks[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']])>=.05
source_layers=EXP/'outputs-v2/combined-point/source-layers.npz'
assert sha(source_layers)==report['sourceLayersSha256']
data=np.load(source_layers)
opaque=data['opaque']
support=data['pure']|data['trusted']
rgb=source[...,:3].astype(int)
highlight=accessory&(rgb.min(axis=2)>=160)&((rgb.max(axis=2)-rgb.min(axis=2))<=30)
result={'scope':'Independent saved-pixel audit of v5 virtual-scene candidate; no aesthetics acceptance.',
 'sourceSha256':sha(source_path),'graphSha256':sha(graph_path),'sourceLayersSha256':sha(source_layers),
 'independentHighlightPixels':int(highlight.sum()),'rows':[]}
for row in report['rows']:
 size,amount=row['size'],row['amount']
 d=v5/str(size)/str(amount)
 output=rgba(d/'candidate.png')
 assert sha(d/'candidate.png')==row['outputSha256']
 material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full'
 layers=material/'layers'
 state=read(layers/'state.json')
 ox,oy=state['origin'];cw,ch=state['width'],state['height']
 area=np.s_[oy:oy+ch,ox:ox+cw]
 inverse=layers/'inverse-source-xy-f32le.bin'
 assert sha(inverse)==row['inverseSha256']
 xy=np.fromfile(inverse,'<f4').reshape(ch,cw,2)
 yy,xx=np.indices((ch,cw))
 disp=np.hypot(xy[...,0]-xx,xy[...,1]-yy)
 base=rgba(material/'candidate.png')
 changed=np.any(output!=base,axis=2)
 writes=mask(d/'allowed-write.png')
 unknown=mask(d/'unknown-new-coverage.png')
 connected=mask(d/'connected-new-coverage.png')
 full=np.zeros((h,w),bool)
 full[area]=writes
 assert not (changed&~full).any()
 assert np.array_equal(output[...,3],source[...,3])
 accessory_changes=changed&accessory
 ear_changes=changed&ears
 feature_changes=changed&features
 opaque_changes=changed&opaque
 highlight_changes=changed&highlight
 assert not opaque_changes.any(), 'Opaque metal changed'
 assert np.array_equal(output[opaque],source[opaque])
 assert np.array_equal(output[highlight],source[highlight])
 # Independently reconstruct the stored candidate from frozen source layers,
 # the saved background and the fixed inverse map. This checks no hidden
 # blending toward the original was used to reduce the apparent face motion.
 under=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype(float)
 under[support[area]]=1
 fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype=float)/255
 fg[support[area]]=data['skin'][area][support[area]]
 coords=[xy[...,1],xy[...,0]]
 na=map_coordinates(under,coords,order=1,mode='grid-constant',cval=0,prefilter=False)
 nr=np.stack([map_coordinates(fg[...,i]*under,coords,order=1,mode='grid-constant',cval=0,prefilter=False) for i in range(3)],-1)
 bg=np.asarray(Image.open(layers/'background.png').convert('RGB'),dtype=float)/255
 saved_bg=np.asarray(Image.open(v5/str(size)/'background-layer.png').convert('RGB'),dtype=float)/255
 bg[support[area]]=saved_bg[area][support[area]]
 fixed=np.where(accessory[area],data['alpha'][area],0)
 composite=fixed[...,None]*data['metal'][area]+(1-fixed[...,None])*(nr+(1-na[...,None])*bg)
 independent=np.floor(np.clip(composite,0,1)*255+.5).astype('uint8')
 error=np.abs(independent[writes].astype(int)-output[area][writes,:3].astype(int))
 assert error.max()==0
 row_out={'size':size,'amount':amount,'candidateSha256':sha(d/'candidate.png'),
  'writePixels':int(writes.sum()),'changedPixels':int(changed.sum()),
  'unknownWritePixels':int((writes&unknown).sum()),'connectedCoveragePixels':int(connected.sum()),
  'outsideWriteChangedPixels':int((changed&~full).sum()),'alphaChangedPixels':int(np.any(output[...,3]!=source[...,3])),
  'opaqueMetalChangedPixels':int(opaque_changes.sum()),'independentHighlightChangedPixels':int(highlight_changes.sum()),
  'softAccessoryChangedPixels':int(accessory_changes.sum()),'earChangedPixels':int(ear_changes.sum()),
  'featureChangedPixels':int(feature_changes.sum()),'supportPixels':int(support.sum()),
  'supportChangedPixels':int((changed&support).sum()),'supportWrittenPixels':int((writes&support[area]).sum()),
  'displacementMovedPixels':int((disp>1e-4).sum()),'displacementMaxPx':float(disp.max()),
  'displacementMeanMovedPx':float(disp[disp>1e-4].mean()),'inverseSha256':sha(inverse),
  'independentFrozenLayerReplayMaxByteError':int(error.max()),
  'unknownDomainEmpty':not unknown.any(),'oldSemanticAccessoryPixelsAllFixed':not accessory_changes.any(),
  'outputOutsideWriteExact':not (changed&~full).any()}
 result['rows'].append(row_out)
 print(json.dumps(row_out))
(HERE/'report-v5.json').write_text(json.dumps(result,indent=2)+'\n')
