"""Hold V2 geometry/alpha fixed; locate reconstruction error and transport it.

This is a causal experiment, not an accepted compositing algorithm. In
particular a color residual does not tell us which material owns that color.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/source-skin-color-consistency-v1-20261015'))
from compositor import sample_f32

SOURCE = ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
TRACE = ROOT/'experiments/whole-face-shape-v2-20261015/outputs-scoped-owner/user-face-contour'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
byte = lambda x: np.floor(np.clip(x, 0, 255)+.5).astype('uint8')


def panel(path, arrays, names, box):
    x0,y0,x1,y1=box; w,h=x1-x0,y1-y0
    out=Image.new('RGB',(w*len(arrays),h+26),'#20242a'); draw=ImageDraw.Draw(out)
    for i,(array,name) in enumerate(zip(arrays,names)):
        out.paste(Image.fromarray(array).convert('RGB').crop(box),(i*w,26))
        draw.text((i*w+3,6),name,fill='white')
    out.save(path)


def main():
    out=HERE/'outputs'; out.mkdir(exist_ok=False)
    original=np.asarray(Image.open(SOURCE).convert('RGBA'))
    report={'accepted':False,'sourceSha256':sha(SOURCE),'runnerSha256':sha(Path(__file__)),
            'hypothesis':'A stationary source-decomposition residual can leave its old contour after foreground motion. Test transport only, with fixed alpha/geometry/B.', 'rows':[]}
    for amount in [30,-30]:
        native=TRACE/f'natural-{amount}'; trace=native/'debug'; dest=out/str(amount); dest.mkdir()
        state=json.loads((trace/'layers.json').read_text());ox,oy,w,h=state['crop'];region=np.s_[oy:oy+h,ox:ox+w]
        rgb=lambda name:np.asarray(Image.open(trace/name).convert('RGB')).astype('float32')
        mask=lambda name:np.asarray(Image.open(trace/name).convert('L'))>0
        alpha=np.fromfile(trace/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
        target=np.fromfile(trace/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
        samples=np.fromfile(trace/'sample-premult-rgba-f32le.bin',dtype='<f4').reshape(h,w,4)
        xy=np.fromfile(trace/'source-xy-f32le.bin',dtype='<f4').reshape(h,w,2)-np.array([ox+.5,oy+.5],dtype='float32')
        changed=mask('changed.png');frozen=mask('frozen.png');write=changed&~frozen
        source=rgb('crop-source.png');f=rgb('foreground.png');b=rgb('reconstructed-background.png')
        current=np.asarray(Image.open(native/'candidate.png').convert('RGBA'))
        native_comp=byte((samples[...,:3]+(1-target)[...,None]*b)/(samples[...,3]+1-target)[...,None].clip(1e-6))
        native_diff=np.max(np.abs(native_comp[write].astype(int)-current[region][...,:3][write].astype(int)))
        assert native_diff==0,native_diff
        residual=source-alpha[...,None]*f-(1-alpha)[...,None]*b
        # At alpha==0, source-only backdrop cannot travel as foreground.
        # Keep this explicit refusal visible instead of normalizing by alpha.
        residual[alpha==0]=0
        warped_residual,_=sample_f32(residual,np.ones_like(alpha),xy)
        stationary=current.copy(); patch=stationary[region][...,:3].copy()
        patch[write]=byte(samples[...,:3]+(1-target)[...,None]*b+residual)[write]
        stationary[region][...,:3]=patch
        transported=current.copy();patch=transported[region][...,:3].copy()
        patch[write]=byte(samples[...,:3]+(1-target)[...,None]*b+warped_residual)[write]
        transported[region][...,:3]=patch
        # Geometrical cap on physically possible premultiplied foreground.
        premult=alpha[...,None]*f+residual
        impossible=np.any((premult < -.01)|(premult > alpha[...,None]*255+.01),axis=-1)
        for name,array in [('stationary',stationary),('transported',transported)]:
            assert np.array_equal(array[...,3],original[...,3])
            assert np.array_equal(array[region][frozen],current[region][frozen])
            diff=np.any(array!=current,axis=-1);support=np.zeros(diff.shape,bool);support[region]=write
            assert not (diff&~support).any()
            Image.fromarray(array).save(dest/f'{name}.png')
        Image.fromarray(byte(np.abs(residual).max(-1)*8)).save(dest/'residual-8x.png')
        Image.fromarray(impossible.astype('uint8')*255).save(dest/'impossible-premult.png')
        np.savez_compressed(dest/'residual.npz',residual=residual,transportedResidual=warped_residual,
                            sourceAlpha=alpha,targetAlpha=target,changed=changed,frozen=frozen)
        for label,box in [('face',[287,197,730,709]),('left',[305,285,430,550]),('right',[590,285,715,550]),('hair',[300,0,715,335])]:
            panel(dest/f'{label}-1x.png',[original,current,stationary,transported],['Source','V2','Stationary R','Transport R'],box)
        report['rows'].append({'amount':amount,'nativeMaxByteDifference':int(native_diff),
            'residualMaxByte':float(np.abs(residual).max()),'impossiblePremultPixels':int(impossible.sum()),
            'transportChangedPixels':int(np.any(transported!=current,axis=-1).sum()),
            'traceSha256':{p.name:sha(p) for p in trace.iterdir() if p.is_file()},
            'sourceIdentityFromExactDecompositionMaxByte':int(np.abs(byte(alpha[...,None]*f+(1-alpha)[...,None]*b+residual)-source).max())})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({**report,'rows':[{k:v for k,v in row.items() if k!='traceSha256'} for row in report['rows']]},indent=2))

if __name__=='__main__':main()
