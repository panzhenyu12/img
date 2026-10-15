"""Background-only mixed-edge completion at fixed source skin and geometry.

Contrast estimated mixed-source H with neural H. The new background write
domain is exactly the union of old disocclusion and the frozen B edge writes.
B remains an explicit skin-in-front hypothesis, never automatic depth truth.
"""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from run import ROOT,HERE,virtual_context,tiles,LamaWorker,sha,montage

HAIR=ROOT/'experiments/hair-skin-decomposition-v1-20261015'
sys.path.insert(0,str(HAIR))
from compose_real import sample


def main():
    out=HERE/'mixed-edge';out.mkdir(exist_ok=False)
    material=ROOT/'experiments/face-material-fill-v1-20261015/woman-100-full';layers=material/'layers'
    prior=ROOT/'experiments/face-background-context-20261015/woman-100'
    bdir=HAIR/'occlusion-corrected-prompts/portrait-woman-100/lama-256'
    state=json.loads((layers/'state.json').read_text());h,w=state['height'],state['width'];ox,oy=state['origin']
    crop=np.asarray(Image.open(layers/'source-crop.png').convert('RGBA'))
    source_path=Path(json.loads((material/'provenance.json').read_text())['source'])
    source=np.asarray(Image.open(source_path).convert('RGBA'))
    old_b=np.asarray(Image.open(bdir/'B-skin-front.png').convert('RGBA'))
    old_target=np.asarray(Image.open(prior/'write-target.png'))>0
    writes_b=np.asarray(Image.open(bdir/'B-write.png'))>0
    target=old_target|writes_b
    observed=np.asarray(Image.open(prior/'hair-only/inference-mask.png'))==0
    virtual,_,_=virtual_context(crop,observed)
    unknown=np.asarray(Image.open(prior/'hair-scene/inference-mask.png'))>0
    estimates=json.loads((HAIR/'source-decomposition-corrected-prompts/report.json').read_text())
    estimate=next(r for r in estimates['rows'] if r['name']=='portrait-woman' and r['variant']=='source-fit-srgb')
    data_path=HAIR/estimate['directory']/'layers.npz';data=np.load(data_path)
    x0,y0,x1,y1=estimate['crop'];sl=np.s_[y0-oy:y1-oy,x0-ox:x1-ox]
    physical=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(h,w).astype('float64')
    p=physical[sl].copy();p[data['trusted']]=1-data['alpha'][data['trusted']];p[data['pureReclaim']]=1;physical[sl]=p
    skin=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype='float64')/255
    p=skin[sl].copy();p[data['trusted']]=data['skin'][data['trusted']];p[data['pureReclaim']]=source[y0:y1,x0:x1,:3][data['pureReclaim']]/255;skin[sl]=p
    xy=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(h,w,2)
    alpha=sample(physical,xy)[...,0];rgb=sample(skin*physical[...,None],xy)
    samples=np.fromfile(layers/'samples-f32le.bin','<f4').reshape(h,w,5)
    # Keep original V36 foreground outside the B domain, exactly as the earlier
    # full-background combination did. No hidden expansion of source skin.
    alpha[~writes_b]=samples[...,4][~writes_b]
    rgb[~writes_b]=samples[...,:3][~writes_b]/255
    full_target=np.zeros(source.shape[:2],bool);full_target[oy:oy+h,ox:ox+w]=target
    report=dict(scope='Fixed B hypothesis, unchanged alpha/foreground; only unknown background color replaced.',
        inputHashes={str(p.relative_to(ROOT)):sha(p) for p in [data_path,source_path,material/'graph.json',layers/'inverse-source-xy-f32le.bin',layers/'samples-f32le.bin',bdir/'B-write.png',prior/'write-target.png',Path(__file__),HERE/'run.py']},
        previousTargetPixels=int(old_target.sum()),extendedTargetPixels=int(target.sum()),rows=[])
    Image.fromarray((target*255).astype('uint8')).save(out/'write-target.png')
    boxes=tiles(target,256);window=np.outer(np.maximum(np.hanning(256),1e-3),np.maximum(np.hanning(256),1e-3))
    worker=LamaWorker(threads=3)
    for name,context,missing in [('observed-only',crop,unknown|target),('virtual-context',virtual,binary_dilation(target,iterations=3))]:
        dest=out/name;dest.mkdir();accum=np.zeros((h,w,3),np.float64);weights=np.zeros((h,w),np.float64);metas=[]
        for i,(x0,y0,x1,y1) in enumerate(boxes):
            patch=context[y0:y1,x0:x1];masked=missing[y0:y1,x0:x1]
            result,meta=worker.inpaint(Image.fromarray(patch),masked.astype('uint8')*255,max_side=256,context_mode='full',mask_dilation=0,seam_blend='none')
            result.save(dest/f'tile-{i:02}.png');conditioned=patch.copy();conditioned[masked,:3]=0;Image.fromarray(conditioned).save(dest/f'conditioning-{i:02}.png')
            accum[y0:y1,x0:x1]+=np.asarray(result)[:,:,:3]*window[...,None];weights[y0:y1,x0:x1]+=window
            metas.append(dict(box=[x0,y0,x1,y1],meta=meta));print(name,i+1,len(boxes),flush=True)
        assert (weights[target]>0).all()
        newbg=np.floor(accum/np.maximum(weights[...,None],1e-12)+.5).clip(0,255)/255
        result=old_b.copy();patch=result[oy:oy+h,ox:ox+w,:3];render=rgb+(1-alpha[...,None])*newbg
        patch[target]=np.floor(render[target].clip(0,1)*255+.5).astype('uint8')
        assert np.array_equal(result[~full_target],old_b[~full_target])
        assert np.array_equal(result[...,3],old_b[...,3])
        Image.fromarray(result).save(dest/'candidate.png');Image.fromarray(np.floor(newbg*255+.5).astype('uint8')).save(dest/'inferred-background.png')
        montage([source,old_b,result],['Source','Previous B',name],dest/'left-1x.png',box=(300,480,600,1170))
        report['rows'].append(dict(name=name,tiles=metas,outputSha256=sha(dest/'candidate.png'),outsideDomainDifferences=0,outputAlphaDifferences=0))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
