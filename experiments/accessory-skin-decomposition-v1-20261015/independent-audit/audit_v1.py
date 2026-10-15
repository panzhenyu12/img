"""Independent saved-pixel audit of the first rejected accessory composite."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
from scipy.ndimage import map_coordinates
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rgba=lambda p:np.asarray(Image.open(p).convert('RGBA'))
pngmask=lambda p:np.asarray(Image.open(p))>0
frozen=read(HERE/'frozen/report-v1.json')
assert sha(HERE/'frozen/run-v1.py')==frozen['scriptSha256']
raw_path=ROOT/'experiments/zim-accessory-combined-20261015/outputs/report.json'
assert sha(raw_path)==frozen['rawReportSha256']
raw=read(raw_path);source_path=ROOT/raw['source'];graph_path=ROOT/raw['graph']
assert sha(source_path)==frozen['sourceSha256']
assert sha(graph_path)==frozen['graphSha256']
source=rgba(source_path);rgb=source[...,:3].astype(float);h,w=source.shape[:2]
params=read(graph_path)['operations'][0]['params']
masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
metal_semantic=masks['accessories']
highlight=(metal_semantic>=.5)&(rgb.min(axis=2)>=160)&((rgb.max(axis=2)-rgb.min(axis=2))<=30)
component_box=np.zeros((h,w),bool)
for row in raw['rows']:
    x0,y0,x1,y1=row['prompt']['box'];component_box[y0:y1+1,x0:x1+1]=True
highlight &= component_box
skin_core=(masks['skin']>=.9)&(np.maximum.reduce([masks[k] for k in ['accessories','ears','hair','leftEye','rightEye','leftBrow','rightBrow','lips','teeth']])<.05)
result={'scope':'Frozen first candidate audit; numerical contracts do not accept visible seams.',
        'scriptSha256':frozen['scriptSha256'],'sourceSha256':sha(source_path),
        'originalNeutralHighlightProbePixels':int(highlight.sum()),'rows':[]}
for row in frozen['rows']:
    d=EXP/row['directory']
    if not (d/'candidate.png').exists():
        raise RuntimeError('Frozen candidate moved; update explicit path after checking its hash')
    assert sha(d/'candidate.png')==row['outputSha256']
    data_path=d.parent/'source-layers.npz';assert sha(data_path)==row['sourceLayersSha256']
    data=np.load(data_path);opaque=data['opaque'];support=data['pure']|data['trusted']
    a=row['amount'];material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{a}-full'
    layer=material/'layers';state=read(layer/'state.json');ox,oy=state['origin'];cw,ch=state['width'],state['height'];sl=np.s_[oy:oy+ch,ox:ox+cw]
    assert sha(layer/'inverse-source-xy-f32le.bin')==row['inverseSha256']
    xy=np.fromfile(layer/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2)
    initial=rgba(material/'candidate.png');out=rgba(d/'candidate.png')
    changed=np.any(out!=initial,axis=2);delta=np.any(out!=source,axis=2)
    writes=pngmask(d/'allowed-write.png');full=np.zeros((h,w),bool);full[sl]=writes
    unknown=pngmask(d/'unknown-new-coverage.png');insufficient=pngmask(d/'insufficient-scene-support.png')
    coverage=pngmask(d/'connected-new-coverage.png')
    forbidden=np.maximum.reduce([masks[k] for k in ['ears','hair','lips','teeth','leftEye','rightEye','leftBrow','rightBrow']])[sl]>=.05
    assert not (changed&~full).any()
    assert not (writes&(unknown|insufficient|forbidden)).any()
    assert np.array_equal(out[...,3],source[...,3])
    assert np.array_equal(out[opaque],source[opaque])
    # Mapping is immutable, and identified fully interior cheek texture is
    # compared at the same output pixels; no intensity scaling is allowed.
    sampled_skin=map_coordinates(skin_core[sl].astype(float),[xy[...,1],xy[...,0]],order=1,mode='grid-constant',cval=0,prefilter=False)>.999
    under=np.fromfile(layer/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype(float)
    under[support[sl]]=1
    new_alpha=map_coordinates(under,[xy[...,1],xy[...,0]],order=1,mode='grid-constant',cval=0,prefilter=False)
    skin_output_domain=sampled_skin&skin_core[sl]&~full[sl]
    assert np.array_equal(out[sl][skin_output_domain],initial[sl][skin_output_domain])
    # Old freeze was semantic. Count releases, do not falsely call them all
    # pure skin or all genuine jewelry. Connected coverage uses an explicit
    # jewelry-in-front hypothesis and can include previously unseen skin.
    originally_frozen=np.fromfile(layer/'frozen-u8.bin',dtype='uint8').reshape(ch,cw)>0
    released=originally_frozen&writes
    highdiff=np.abs(out[highlight,:3].astype(int)-source[highlight,:3].astype(int))
    rec={'mode':row['mode'],'amount':a,'backing':row['backing'],'outputSha256':sha(d/'candidate.png'),
         'changedVsV36Pixels':int(changed.sum()),'writePixels':int(writes.sum()),
         'opaqueMetalSourceChangedPixels':int((delta&opaque).sum()),
         'neutralHighlightChangedPixels':int(delta[highlight].sum()),'neutralHighlightMaxChannelChange':int(highdiff.max()),
         'softMetalAlphaAtLeast005NewlyChangedPixels':int((changed&(data['alpha']>=.05)).sum()),
         'unknownWritePixels':int((writes&unknown).sum()),'insufficientWritePixels':int((writes&insufficient).sum()),
         'sourceSkinSupportedPixels':int(support.sum()),'supportWithSceneBackingWrittenPixels':int((writes&support[sl]).sum()),
         'effectiveSceneBackingWritePixels':int((writes&support[sl]&(new_alpha<1-1e-10)).sum()),
         'maxSceneBackingWeightAtWrittenSupport':float(np.max(np.where(writes&support[sl],1-new_alpha,0))),
         'releasedOldFrozenPixels':int(released.sum()),'releasedOutsidePureSourceEvidencePixels':int((released&~support[sl]).sum()),
         'skinInteriorComparedPixels':int(skin_output_domain.sum()),'skinInteriorChangedVsV36':0,
         'inverseMapExact':True,'inverseMapSha256':row['inverseSha256'],
         'uncoveredOutputDifferenceIsNotGeometricAttenuation':True}
    result['rows'].append(rec)
    print(json.dumps(rec))
result['backingPairsByteExact']=all(
    next(r for r in result['rows'] if r['mode']==mode and r['amount']==a and r['backing']=='original-backing')['outputSha256'] ==
    next(r for r in result['rows'] if r['mode']==mode and r['amount']==a and r['backing']=='source-local-scene')['outputSha256']
    for mode in ['combined-point','combined-box'] for a in [100,-100])
(HERE/'report-v1.json').write_text(json.dumps(result,indent=2)+'\n')
