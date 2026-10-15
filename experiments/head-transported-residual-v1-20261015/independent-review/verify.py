"""Independent read-only audit: hashes, saved outputs and interpolation.

Uses scipy's bilinear sampler, not the probe's shared sample_f32 helper.
Writes only this audit's verification JSON; never changes candidate outputs.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parents[1]
TRACE=ROOT/'experiments/whole-face-shape-v2-20261015/outputs-scoped-owner/user-face-contour'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
to_byte=lambda a:np.floor(np.clip(a,0,255)+.5).astype('uint8')

def main():
    report=json.loads((EXP/'outputs/report.json').read_text())
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    source=np.array(Image.open(source_path).convert('RGBA'))
    assert sha(source_path)==report['sourceSha256']
    assert sha(EXP/'probe.py')==report['runnerSha256']
    result={'sourceMatches':True,'runnerMatches':True,'candidateModified':False,'rows':[]}
    for row in report['rows']:
        amount=row['amount'];native=TRACE/f'natural-{amount}';trace=native/'debug';out=EXP/f'outputs/{amount}'
        assert all(sha(trace/n)==h for n,h in row['traceSha256'].items())
        ox,oy,w,h=json.loads((trace/'layers.json').read_text())['crop'];region=np.s_[oy:oy+h,ox:ox+w]
        rgb=lambda n:np.asarray(Image.open(trace/n).convert('RGB'),dtype='float32')
        f=rgb('foreground.png');b=rgb('reconstructed-background.png');c=rgb('crop-source.png')
        assert np.array_equal(c,source[region][...,:3])
        a=np.fromfile(trace/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
        at=np.fromfile(trace/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
        samples=np.fromfile(trace/'sample-premult-rgba-f32le.bin',dtype='<f4').reshape(h,w,4)
        xy=np.fromfile(trace/'source-xy-f32le.bin',dtype='<f4').reshape(h,w,2)-np.array([ox+.5,oy+.5],dtype='float32')
        changed=np.array(Image.open(trace/'changed.png'))>0;frozen=np.array(Image.open(trace/'frozen.png'))>0;write=changed&~frozen
        current=np.array(Image.open(native/'candidate.png').convert('RGBA'))
        residual=c-a[...,None]*f-(1-a)[...,None]*b;residual[a==0]=0
        saved=np.load(out/'residual.npz')
        assert np.array_equal(residual,saved['residual'])
        # Independent library implementation of bilinear zero-padded sampling.
        scipy_sample=np.stack([map_coordinates(residual[...,i],[xy[...,1],xy[...,0]],order=1,mode='grid-constant',cval=0,prefilter=False) for i in range(3)],axis=-1)
        sampler_error=float(np.abs(scipy_sample-saved['transportedResidual']).max())
        assert sampler_error<1e-4,sampler_error
        denominator=samples[...,3]+1-at
        native_comp=to_byte((samples[...,:3]+(1-at)[...,None]*b)/denominator[...,None].clip(1e-6))
        assert np.array_equal(native_comp[write],current[region][...,:3][write])
        den_error=float(np.abs(denominator[write]-1).max())
        facts={}
        for variant,added in [('stationary',residual),('transported',saved['transportedResidual'])]:
            expected=current.copy();patch=expected[region][...,:3].copy()
            patch[write]=to_byte(samples[...,:3]+(1-at)[...,None]*b+added)[write]
            expected[region][...,:3]=patch
            actual=np.array(Image.open(out/f'{variant}.png').convert('RGBA'))
            assert np.array_equal(expected,actual)
            assert np.array_equal(actual[...,3],source[...,3])
            assert np.array_equal(actual[region][frozen],source[region][frozen])
            changed_full=np.any(actual!=current,axis=-1);allowed=np.zeros(changed_full.shape,bool);allowed[region]=write
            assert not np.any(changed_full&~allowed)
            facts[variant+'ChangedPixels']=int(changed_full.sum())
        identity=np.max(np.abs(to_byte(a[...,None]*f+(1-a)[...,None]*b+residual).astype(int)-c.astype(int)),axis=-1)
        premult=a[...,None]*f+residual
        impossible=np.any((premult<-.01)|(premult>a[...,None]*255+.01),axis=-1)
        assert int(impossible.sum())==row['impossiblePremultPixels']
        facts.update({'amount':amount,'traceHashesMatch':True,'savedResultsMatch':True,'nativeWriteByteDifference':0,'samplerMaxFloatDifference':sampler_error,'normalizationMaxDistanceFromOne':den_error,'sourceFitOnPositiveAlphaMaxByte':int(identity[a>0].max()),'sourceFitOnZeroAlphaMaxByte':int(identity[a==0].max()),'impossiblePremultPixels':int(impossible.sum()),'impossibleOpaquePixels':int((impossible&(a==1)).sum()),'impossibleFractionalPixels':int((impossible&(a>0)&(a<1)).sum()),'sourcePositiveAlphaPixels':int((a>0).sum()),'frozenChanged':0,'outsideWriteChanged':0})
        result['rows'].append(facts)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
