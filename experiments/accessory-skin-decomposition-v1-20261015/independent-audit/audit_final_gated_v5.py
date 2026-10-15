"""Independent audit of final-gated-v5 (strict old-strip-only) outputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rgba=lambda p:np.asarray(Image.open(p).convert('RGBA'))
mask=lambda p:np.asarray(Image.open(p))>0
final=EXP/'final-gated-v5';report=read(final/'report.json');source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';graph_path=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'
assert sha(source_path)==report['sourceSha256'] and sha(graph_path)==report['graphSha256']
source=rgba(source_path);h,w=source.shape[:2];params=read(graph_path)['operations'][0]['params'];masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
accessory=masks['accessories']>=.05;ears=masks['ears']>=.05;features=np.maximum.reduce([masks[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']])>=.05
layers_np=EXP/'outputs-v2/combined-point/source-layers.npz';assert sha(layers_np)==report['sourceLayersSha256'];data=np.load(layers_np);opaque=data['opaque'];support=data['pure']|data['trusted']
rgb=source[...,:3].astype(int);highlight=accessory&(rgb.min(axis=2)>=160)&((rgb.max(axis=2)-rgb.min(axis=2))<=30)
result={'scope':'Independent saved-pixel audit of final-gated-v5; author accepted=false preserved.','sourceSha256':sha(source_path),'graphSha256':sha(graph_path),'sourceLayersSha256':sha(layers_np),'highlightPixels':int(highlight.sum()),'rows':[]}
for r in report['rows']:
 size,amount=r['size'],r['amount'];d=final/str(size)/str(amount);out=rgba(d/'candidate.png');assert sha(d/'candidate.png')==r['outputSha256']
 mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';layers=mat/'layers';st=read(layers/'state.json');ox,oy=st['origin'];cw,ch=st['width'],st['height'];area=np.s_[oy:oy+ch,ox:ox+cw]
 inv=layers/'inverse-source-xy-f32le.bin';assert sha(inv)==r['inverseSha256'];xy=np.fromfile(inv,'<f4').reshape(ch,cw,2);yy,xx=np.indices((ch,cw));disp=np.hypot(xy[...,0]-xx,xy[...,1]-yy)
 baseline=rgba(EXP/'outputs-v2/combined-point'/f'{amount}-source-local-scene/candidate.png');changed=np.any(out!=baseline,axis=2);write=mask(d/'final-write.png');unknown=mask(d/'original-unknown.png');insuf=mask(d/'original-insufficient.png');full=np.zeros((h,w),bool);full[area]=write
 assert not(changed&~full).any();assert not (write&unknown).any() and not(write&insuf).any();assert np.array_equal(out[...,3],source[...,3]);assert np.array_equal(out[opaque],source[opaque]);assert np.array_equal(out[highlight],source[highlight])
 soft=changed&accessory;ear=changed&ears;feat=changed&features
 result['rows'].append({'size':size,'amount':amount,'writePixels':int(write.sum()),'changedPixels':int(changed.sum()),'outsideWriteChanged':int((changed&~full).sum()),'unknownWrite':int((write&unknown).sum()),'insufficientWrite':int((write&insuf).sum()),'opaqueMetalChanged':int((changed&opaque).sum()),'highlightChanged':int((changed&highlight).sum()),'softAccessoryChanged':int(soft.sum()),'earChanged':int(ear.sum()),'featureChanged':int(feat.sum()),'supportChanged':int((changed&support).sum()),'inverseSha256':sha(inv),'movedPixels':int((disp>1e-4).sum()),'maxDisplacementPx':float(disp.max()),'meanMovedDisplacementPx':float(disp[disp>1e-4].mean()),'baselineSha256':sha(EXP/'outputs-v2/combined-point'/f'{amount}-source-local-scene/candidate.png'),'outputSha256':sha(d/'candidate.png')})
(HERE/'report-final-gated-v5.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
