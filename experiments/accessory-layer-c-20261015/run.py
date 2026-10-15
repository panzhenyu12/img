"""Hypothesis C: subtract cropped accessory alpha from V36 moving physical alpha.

Exact frozen inverse, background, original write/freeze gates and source RGB.
This tests the proposed source-layer cut independently of expanding write gates
or regenerating backgrounds. Source-stationary jewelry remains in the backing.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import sample,montage
from run_real import probe,sha,mask_save
spec=importlib.util.spec_from_file_location('exact_compose',ROOT/'experiments/face-background-context-20261015/context.py')
context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)

def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    raw_root=ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs'
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    graph_path=source_path.with_name('face-graph.json')
    source=np.asarray(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2]
    params=json.loads(graph_path.read_text())['operations'][0]['params']
    masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    paths=[raw_root/f'component-{i}-scale-4x/combined-point/alphas-full.npy' for i in [1,2]]
    raw=[np.load(p,mmap_mode='r')[2] for p in paths]
    accessory=np.maximum.reduce(raw).astype('float64')
    opaque=(accessory>=.95)&(masks['accessories']>=.05)
    old_accessory=masks['accessories']>=.5
    light=source[...,:3].astype('float64').mean(axis=-1)
    bright=old_accessory&(light>=np.quantile(light[old_accessory],.9))
    ears=masks['ears']>=.05
    report=dict(scope=__doc__,sourceSha256=sha(source_path),graphSha256=sha(graph_path),
        rawAlphaFiles={str(p.relative_to(ROOT)):sha(p) for p in paths},scriptSha256=sha(Path(__file__)),
        accepted=False,sourceAlphaRule='A_move = A_v36 * (1-A_accessory); no other alpha changes',
        outputGate='Existing V36 changed/frozen gates only; no source/destination unskip, no backing edit',rows=[])
    for amount in [100,-100]:
        material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';layers=material/'layers'
        state=json.loads((layers/'state.json').read_text());cw,ch=state['width'],state['height'];ox,oy=state['origin'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        frozen=np.fromfile(layers/'frozen-u8.bin',dtype='uint8').reshape(ch,cw)>0
        changed=np.fromfile(layers/'changed-u8.bin',dtype='uint8').reshape(ch,cw)>0
        original_alpha=np.fromfile(layers/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw)
        xy=np.fromfile(layers/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2)
        old_samples=np.fromfile(layers/'samples-f32le.bin',dtype='<f4').reshape(ch,cw,5)
        background=np.asarray(Image.open(layers/'background.png').convert('RGBA'))
        foreground=np.asarray(Image.open(layers/'foreground.png').convert('RGBA'))
        base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'))
        original_render=context.compose(background,old_samples[...,:4],old_samples[...,4])
        replay=source.copy();patch=replay[cs].copy();patch[changed]=original_render[changed];replay[cs]=patch
        assert np.array_equal(replay,base),'Original sample/backing replay must be byte-exact'
        original_alpha64=original_alpha.astype('float64')
        cut_alpha=(original_alpha64*(1-accessory[cs])).astype('float32')
        # Update exact saved transported samples by the paired sampler delta.
        # The paired subtraction cancels Python/Rust summation-rounding drift;
        # zero source-alpha change is exactly the frozen rendered baseline.
        fa=foreground[...,3].astype('float64')/255
        old_a=sample(original_alpha64,xy)[...,0];new_a=sample(cut_alpha.astype('float64'),xy)[...,0]
        old_rgba=sample(np.concatenate([foreground[...,:3].astype('float64')*fa[...,None],fa[...,None]],axis=2)*original_alpha64[...,None],xy)
        new_rgba=sample(np.concatenate([foreground[...,:3].astype('float64')*fa[...,None],fa[...,None]],axis=2)*cut_alpha[...,None],xy)
        new_samples=old_samples.copy();new_samples[...,:4]=(old_samples[...,:4].astype('float64')+new_rgba-old_rgba).astype('float32')
        new_samples[...,4]=(old_samples[...,4].astype('float64')+new_a-old_a).astype('float32')
        new_samples=np.maximum(new_samples,0)
        rendered=context.compose(background,new_samples[...,:4],new_samples[...,4])
        output=base.copy();patch=output[cs].copy();patch[changed]=rendered[changed];output[cs]=patch
        delta=np.abs(output.astype('int16')-base.astype('int16'));difference=np.any(delta!=0,axis=-1)
        full_changed=np.zeros((h,w),bool);full_changed[cs]=changed
        assert not (difference&~full_changed).any()
        assert np.array_equal(output[opaque],source[opaque])
        assert np.array_equal(output[ears],base[ears])
        assert np.array_equal(output[...,3],source[...,3])
        removed=original_alpha64-cut_alpha.astype('float64')
        dest=out/str(amount);dest.mkdir()
        Image.fromarray(output).save(dest/'candidate-C.png')
        for name,a in [('source-v36-alpha',original_alpha),('source-C-alpha',cut_alpha),('source-cut-x1000',removed*1000),
            ('target-v36-alpha',old_a),('target-C-alpha',new_a),('target-cut-x1000',(old_a-new_a)*1000),
            ('accessory-stationary-alpha',accessory[cs]),('V36-changed',changed),('V36-frozen',frozen)]:mask_save(dest/(name+'.png'),a)
        np.savez_compressed(dest/'alpha-layers.npz',sourceOriginal=original_alpha,sourceCut=cut_alpha,targetOriginal=old_a.astype('float32'),targetCut=new_a.astype('float32'),accessory=accessory[cs].astype('float32'),changed=changed,frozen=frozen)
        for side,crop in [('left',[326,409,410,550]),('right',[612,414,696,554])]:
            montage(dest/f'{side}-comparison-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','V36','C remove accessory'],crop)
        entry=dict(amount=amount,inverseSha256=sha(layers/'inverse-source-xy-f32le.bin'),sourceAlphaSha256=sha(layers/'source-alpha-f32le.bin'),backgroundSha256=sha(layers/'background.png'),originalSampleSha256=sha(layers/'samples-f32le.bin'),
            baselineExactReplay=True,sourceCutPixelsAbove001=int((removed>.001).sum()),sourceCutAlphaSum=float(removed.sum()),sourceCutMax=float(removed.max()),
            originalAlphaOnOpaqueMetalMin=float(original_alpha[opaque[cs]].min()),originalAlphaOnOpaqueMetalMax=float(original_alpha[opaque[cs]].max()),
            targetCutMax=float(np.max(np.abs(old_a-new_a))),sourceCutEarMax=float(np.max(removed[ears[cs]])) if ears[cs].any() else None,
            outputChangedPixels=int(difference.sum()),outputMaxChannelChange=int(delta.max()),opaqueMetalChanges=int((difference&opaque).sum()),
            brightMetalChanges=int((difference&bright).sum()),earChanges=int((difference&ears).sum()),outsideWritesChanged=0,
            v36OutputSha256=sha(material/'candidate.png'),outputSha256=sha(dest/'candidate-C.png'),layersSha256=sha(dest/'alpha-layers.npz'),directory=str(dest.relative_to(ROOT)))
        report['rows'].append(entry);print(json.dumps(entry),flush=True)
    report['conclusion']='V36 already excludes the accessory alpha support. Subtraction is numerically negligible and leaves the original symptom; reject C as a repair.'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
