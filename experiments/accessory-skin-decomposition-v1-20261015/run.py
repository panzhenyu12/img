"""Isolated source-supported accessory/skin correction at exact V36 geometry.

Jewelry alpha stays source-stationary. Model-excluded pixels are not assumed
skin: independent local source evidence is required. Holes remain unknown
unless genuinely covered by transported observed skin in a connected region.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,binary_dilation,label

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha,mask_save
from decompose import decompose,local_prior
from compose_real import sample,montage
cv2=probe.cv2


def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    raw_dir=ROOT/'experiments/zim-accessory-combined-20261015/outputs'
    raw=json.loads((raw_dir/'report.json').read_text())
    source_path=ROOT/raw['source'];graph_path=ROOT/raw['graph']
    source=np.asarray(Image.open(source_path).convert('RGBA'));rgb=source[...,:3].astype('float64')/255
    assert sha(source_path)==raw['sourceSha256'] and sha(graph_path)==raw['graphSha256']
    h,w=source.shape[:2];params=json.loads(graph_path.read_text())['operations'][0]['params']
    masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    old=masks['accessories'];skin=masks['skin'];ears=masks['ears'];hair=masks['hair']
    features=np.maximum.reduce([masks[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']])
    skin_core=(skin>=.9)&(np.maximum.reduce([old,ears,hair,features])<.05)
    skin_prior,skin_dist=local_prior(rgb,skin_core,2)
    cell=max(w/params['semanticMasks']['accessories']['width'],h/params['semanticMasks']['accessories']['height'])
    face_width=(max(p['x'] for p in params['landmarks'])-min(p['x'] for p in params['landmarks']))*w
    radius=min(3*cell,.025*face_width)
    lab=cv2.cvtColor(rgb.astype('float32'),cv2.COLOR_RGB2LAB)
    skin_lab=cv2.cvtColor(skin_prior.astype('float32'),cv2.COLOR_RGB2LAB)
    delta=lab-skin_lab
    color_support=(np.linalg.norm(delta,axis=2)<=8)&(abs(delta[...,0])<=6)&(np.linalg.norm(delta[...,1:],axis=2)<=6)
    region=params['region'];yy,xx=np.indices((h,w))
    roi=((xx+.5)/w>=region['x'])&((xx+.5)/w<=region['x']+region['width'])&((yy+.5)/h>=region['y'])&((yy+.5)/h<=region['y']+region['height'])
    source_skin_support=roi&(skin_dist<=radius)&((skin>=.5)|color_support)&(ears<.05)&(hair<.05)&(features<.05)
    scene_core=(np.maximum.reduce([masks[k] for k in ['skin','face','faceOutline','neck','ears','hair','accessories']])<=.01)
    scene_prior,scene_dist=local_prior(rgb,scene_core,2)
    report={'scope':'Exact V36 geometry; explicit jewelry-front hypothesis; source-supported ownership only; unknown holes preserved.',
        'scriptSha256':sha(Path(__file__)),'solverSha256':sha(ROOT/'experiments/hair-skin-decomposition-v1-20261015/decompose.py'),
        'rawReportSha256':sha(raw_dir/'report.json'),'sourceSha256':sha(source_path),'graphSha256':sha(graph_path),
        'sourceSkinEvidenceRadiusPx':radius,'sceneDonorMaxDistancePx':radius*2,'rows':[]}
    boxes=[r['prompt']['box'] for r in raw['rows']]
    pad=int(np.ceil(radius*2))
    crop=[max(0,min(b[0] for b in boxes)-pad),max(0,min(b[1] for b in boxes)-pad),min(w,max(b[2] for b in boxes)+pad),min(h,max(b[3] for b in boxes)+pad)]
    x0,y0,x1,y1=crop;sl=np.s_[y0:y1,x0:x1]
    for mode in ['combined-point','combined-box']:
        cases=[r for r in raw['rows'] if r['directory'].endswith('-'+mode)]
        assert len(cases)==2
        paths=[raw_dir/r['directory']/'alphas.npy' for r in cases]
        alpha=np.maximum.reduce([np.load(p)[r['officialScoreArgmax']] for p,r in zip(paths,cases)]).astype('float64')
        opaque=(alpha>=.95)&(old>=.05)
        pure=(old>=.5)&(alpha<=.05)&source_skin_support
        near_pure=distance_transform_edt(~pure)<=1.5
        # A mixed metal edge must touch an independently observed skin region.
        # Proximity alone never turns an entire jewelry hole into skin.
        candidate=(old>=.05)&(alpha>.05)&(alpha<.95)&((skin>=.5)|near_pure)&(skin_dist<=radius)&(ears<.05)&(hair<.05)&(features<.05)&roi
        decomp=decompose(rgb[sl],alpha[sl],opaque[sl],skin_core[sl],candidate[sl],radius,space='srgb',refine_alpha=False)
        trusted=np.zeros((h,w),bool);trusted[sl]=decomp['trusted']
        support=pure|trusted
        metal_rgb=rgb.copy();patch=metal_rgb[sl].copy();patch[decomp['trusted']]=decomp['hair'][decomp['trusted']];metal_rgb[sl]=patch
        underlying_rgb=rgb.copy();patch=underlying_rgb[sl].copy();patch[decomp['trusted']]=decomp['skin'][decomp['trusted']];underlying_rgb[sl]=patch
        # Preserve the raw alpha; a model-excluded skin pixel may still have a
        # tiny nonzero metal tail. Solve its skin color, do not set alpha to 0.
        metal_prior,_=local_prior(rgb,opaque,1)
        metal_rgb[pure]=metal_prior[pure]
        underlying_rgb[pure]=np.clip((rgb[pure]-alpha[pure,None]*metal_rgb[pure])/np.maximum(1-alpha[pure,None],.001),0,1)
        d=out/mode;d.mkdir()
        for name,mask in [('raw-accessory-alpha',alpha),('source-pure-skin-supported',pure),('source-mixed-supported',trusted),('original-ear',ears),('old-accessory',old),('metal-core',opaque),('unknown-model-excluded',(old>=.5)&(alpha<=.05)&~support),('source-scene-donor',scene_core)]:mask_save(d/(name+'.png'),mask)
        np.savez_compressed(d/'source-layers.npz',alpha=alpha,metal=metal_rgb,skin=underlying_rgb,pure=pure,trusted=trusted,opaque=opaque,scenePrior=scene_prior,sceneDistance=scene_dist)
        for amount in [100,-100]:
            material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';layers=material/'layers'
            state=json.loads((layers/'state.json').read_text());cw,ch=state['width'],state['height'];ox,oy=state['origin'];cs=np.s_[oy:oy+ch,ox:ox+cw]
            xy=np.fromfile(layers/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2)
            physical=np.fromfile(layers/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype('float64')
            original_physical=physical.copy()
            fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype='float64')/255
            under=physical.copy();under[support[cs]]=1
            fg[support[cs]]=underlying_rgb[cs][support[cs]]
            new_alpha=sample(under,xy)[...,0];new_rgb=sample(fg*under[...,None],xy)
            transported=sample(support[cs].astype('float64'),xy)[...,0]>1e-6
            yy_,xx_=np.indices((ch,cw));moved=np.hypot(xy[...,0]-xx_,xy[...,1]-yy_)>1e-4
            allowed=roi[cs]&(ears[cs]<.05)&(hair[cs]<.05)&(features[cs]<.05)&~opaque[cs]&moved
            # Only real transported coverage may fill a previously-background
            # location. An excluded alpha hole itself provides no skin evidence.
            added=(new_alpha>under+.001)&(old[cs]>=.05)&(alpha[cs]<=.05)&allowed
            labels,count=label(added,structure=np.ones((3,3)));touch=binary_dilation(support[cs],structure=np.ones((3,3)))
            ids=np.unique(labels[touch&added]);ids=ids[ids>0]
            connected=np.isin(labels,ids) if len(ids) else np.zeros_like(added)
            unknown=added&~connected
            writes=((support[cs]|transported|connected)&allowed)&~unknown
            # A source skin strip needs a hidden backing estimate only when it
            # is exposed. Refuse distant donors instead of silently painting it.
            needs_scene=support[cs]&(new_alpha<.999)
            insufficient=needs_scene&(scene_dist[cs]>radius*2)
            writes &= ~insufficient
            fixed_alpha=np.where(old[cs]>=.05,alpha[cs],0)
            base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'))
            bg=np.asarray(Image.open(layers/'background.png').convert('RGB'),dtype='float64')/255
            variants=[('original-backing',bg.copy()),('source-local-scene',bg.copy())]
            variants[1][1][support[cs]]=scene_prior[cs][support[cs]]
            for backing_name,backing in variants:
                beneath=new_rgb+(1-new_alpha[...,None])*backing
                rendered=fixed_alpha[...,None]*metal_rgb[cs]+(1-fixed_alpha[...,None])*beneath
                output=base.copy();patch=output[cs][:,:,:3].copy();patch[writes]=np.floor(np.clip(rendered[writes],0,1)*255+.5).astype('uint8');output[cs][:,:,:3]=patch
                full_write=np.zeros((h,w),bool);full_write[cs]=writes
                assert np.array_equal(output[~full_write],base[~full_write])
                assert np.array_equal(output[opaque],source[opaque])
                assert np.array_equal(output[...,3],source[...,3])
                dest=d/f'{amount}-{backing_name}';dest.mkdir()
                Image.fromarray(output).save(dest/'candidate.png')
                for n,m in [('allowed-write',writes),('connected-new-coverage',connected),('unknown-new-coverage',unknown),('insufficient-scene-support',insufficient),('original-frozen',np.fromfile(layers/'frozen-u8.bin',dtype='uint8').reshape(ch,cw)>0)]:mask_save(dest/(n+'.png'),m)
                montage(dest/'face-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','V36','Jewelry front hypothesis'],[287,197,730,709])
                for component in cases:
                    b=component['prompt']['box'];c=[max(0,b[0]-pad),max(0,b[1]-pad),min(w,b[2]+pad),min(h,b[3]+pad*4)]
                    montage(dest/f"component-{component['prompt']['component']}-1x.png",[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','V36','Candidate'],c)
                row={'mode':mode,'amount':amount,'backing':backing_name,'sourcePureSkinPixels':int(pure.sum()),'sourceTrustedMixedPixels':int(trusted.sum()),'sourceMixedCandidatePixels':int(candidate.sum()),
                    'writePixels':int(writes.sum()),'changedPixels':int(np.any(output!=base,axis=2).sum()),'newlyCoveredConnectedPixels':int(connected.sum()),'unknownNewCoveragePixels':int(unknown.sum()),'insufficientScenePixels':int(insufficient.sum()),
                    'opaqueMetalChanged':0,'outsideWritesChanged':0,'alphaRefined':False,'sourceCrop':crop,'decomposition':decomp['stats'],
                    'rawAlphaPaths':[str(p.relative_to(ROOT)) for p in paths],'rawAlphaSha256':[sha(p) for p in paths],
                    'inverseSha256':sha(layers/'inverse-source-xy-f32le.bin'),'sourceLayersSha256':sha(d/'source-layers.npz'),'outputSha256':sha(dest/'candidate.png'),
                    'originalFrozenUnskippedPixels':int((writes&(np.fromfile(layers/'frozen-u8.bin',dtype='uint8').reshape(ch,cw)>0)).sum()),'directory':str(dest.relative_to(HERE))}
                report['rows'].append(row);print(json.dumps(row),flush=True)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
