"""Independent endpoint/segment oracle and frozen-artifact review; no renderer changes."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;EXPERIMENT=HERE.parent;ROOT=EXPERIMENT.parents[1]
spec=importlib.util.spec_from_file_location('reviewed_project',EXPERIMENT/'project.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
byte=lambda a:np.floor(np.clip(a,0,255)+.5).astype('uint8')

def segment_oracle(c,a,f,b):
    # Construct the intersection of the observation line with the RGB square;
    # project the prior point on the geometric segment, not via F-only algebra.
    if a==0:return np.clip([f,c],0,255)
    if a==1:return np.clip([c,b],0,255)
    if c==0:return np.zeros(2)
    if c==255:return np.full(2,255.)
    q=1-a;points=[]
    for v in [(0,c/q),(255,(c-255*a)/q),(c/a,0),((c-255*q)/a,255)]:
        if min(v)>=-1e-8 and max(v)<=255+1e-8:points.append(np.clip(v,0,255))
    assert len(points)>=2,(c,a,points)
    p0,p1=max(((x,y) for x in points for y in points),key=lambda pair:np.sum((pair[0]-pair[1])**2))
    direction=p1-p0;d=float(direction@direction)
    t=np.clip(float((np.array([f,b])-p0)@direction)/d,0,1) if d else 0
    return p0+t*direction


def independent_sample(f,a,xy):
    h,w=a.shape;point=xy.astype(float);ix=np.floor(point).astype(int);t=point-ix
    color=np.zeros((*a.shape,3));coverage=np.zeros(a.shape)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        x=ix[...,0]+dx;y=ix[...,1]+dy
        weight=(t[...,0] if dx else 1-t[...,0])*(t[...,1] if dy else 1-t[...,1])
        valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        wa=weight[valid]*a[y[valid],x[valid]].astype(float)
        coverage[valid]+=wa;color[valid]+=wa[:,None]*f[y[valid],x[valid]].astype(float)
    return color,coverage


def main():
    rng=np.random.default_rng(20261015);n=2500
    alpha=np.concatenate([rng.random(n),np.geomspace(1e-10,1e-2,100),[0,1,np.nextafter(np.float32(1),np.float32(0))]]).astype('float32')
    c=rng.integers(0,256,(len(alpha),3)).astype('float32');f=rng.uniform(0,255,c.shape).astype('float32');b=rng.uniform(0,255,c.shape).astype('float32')
    got_f,got_b=module.project(c,alpha,f,b)
    reference=np.array([[segment_oracle(float(c[i,k]),float(a),float(f[i,k]),float(b[i,k])) for k in range(3)] for i,a in enumerate(alpha)])
    actual=np.stack([got_f,got_b],-1)
    max_error=float(np.abs(actual-reference).max());assert max_error<3e-4,max_error
    assert np.all(np.isfinite(actual)&(actual>=0)&(actual<=255))
    reconstruction=alpha[:,None].astype(float)*got_f+(1-alpha[:,None].astype(float))*got_b
    assert np.array_equal(byte(reconstruction),byte(c))
    report={'acceptedAsFinishedRetouch':False,'mathematicalComponent':'Correct nearest bounded F/B projection for tested valid float32 photo domain; material interpretation remains underdetermined','projectionOracle':{'channelCases':int(c.size),'maxColorErrorToIndependentSegmentOracle':max_error,'sourceReconstructionMaxByte':int(np.abs(byte(reconstruction).astype(int)-byte(c).astype(int)).max())},'artifacts':[]}
    meta=json.loads((EXPERIMENT/'outputs/report.json').read_text());src=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(src).convert('RGBA'))
    assert sha(src)==meta['sourceSha256']
    for name,digest in meta['scripts'].items():assert sha(EXPERIMENT/name)==digest
    for row in meta['rows']:
        amount=row['amount'];d=EXPERIMENT/f'outputs/{amount}';native=ROOT/f'experiments/whole-person-alpha-override-v1-20261015/outputs/raw-{amount}';trace=native/'debug'
        assert sha(native/'candidate.png')==row['nativeSha256']
        assert sha(d/'candidate.png')==row['outputSha256']
        for name,digest in row['traceSha256'].items():assert sha(trace/name)==digest,name
        ox,oy,w,h=json.loads((trace/'layers.json').read_text())['crop'];roi=np.s_[oy:oy+h,ox:ox+w]
        layers=np.load(d/'projected-layers.npz');f,b,a,target=[layers[k] for k in ['foreground','background','alpha','targetAlpha']]
        oldalpha=np.fromfile(trace/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
        oldtarget=np.fromfile(trace/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
        assert np.array_equal(a,oldalpha) and np.array_equal(target,oldtarget)
        assert np.all(np.isfinite(f)&(f>=0)&(f<=255)) and np.all(np.isfinite(b)&(b>=0)&(b<=255))
        crop=source[roi][...,:3];recon=byte(a[...,None]*f+(1-a[...,None])*b);assert np.array_equal(recon,crop)
        xy=np.fromfile(trace/'source-xy-f32le.bin',dtype='<f4').reshape(h,w,2)-np.array([ox+.5,oy+.5],dtype='float32')
        color,alpha64=independent_sample(f,a,xy)
        fixed=np.asarray(Image.open(trace/'frozen.png'))>0;changed=np.asarray(Image.open(trace/'changed.png'))>0;write=changed&~fixed
        assert np.all(np.abs(alpha64[write]-target[write])<2e-7)
        output=np.asarray(Image.open(d/'candidate.png').convert('RGBA'));old=np.asarray(Image.open(native/'candidate.png').convert('RGBA'))
        fullwrite=np.zeros(source.shape[:2],bool);fullwrite[roi]=write
        assert np.array_equal(output[~fullwrite],old[~fullwrite])
        assert np.array_equal(output[roi][fixed],source[roi][fixed])
        assert np.array_equal(output[...,3],source[...,3])
        oracle_render=byte(color+(1-target[...,None])*b)
        sampling_error=np.abs(output[roi][...,:3].astype(int)-oracle_render.astype(int)).max(-1)
        assert sampling_error[write].max()<=1
        delta=np.abs(output[roi][...,:3].astype(int)-crop.astype(int)).max(-1)
        # The source and output share B: this is tighter than the replay's sum bound.
        bound=255*np.maximum(a,target)+1
        assert np.all(delta[write]<=bound[write])
        low=write&(np.maximum(a,target)<.01)
        report['artifacts'].append({'amount':amount,'outputSha256':sha(d/'candidate.png'),'independentFloat64SamplingMaxByte':int(sampling_error[write].max()),'sourceFitMaxByte':0,'tighterMaxAlphaBoundViolations':0,'lowCoverageMaxByte':int(delta[low].max()),'lowCoverageOver8':int((delta[low]>8).sum()),'lowCoveragePixels':int(low.sum()),'changedOnlyWithinFrozenNativeWriteMask':True})
    qc=np.full((1,1,3),1,dtype='float32');qa=np.full((1,1),.5,dtype='float32')
    qf,qb=module.project(qc,qa,np.zeros_like(qc),np.ones_like(qc))
    report['prematureQuantizationCounterexample']={'observedByte':1,'alpha':.5,'priorForeground':0,'priorBackground':1,'projectedForeground':float(qf[0,0,0]),'projectedBackground':float(qb[0,0,0]),'correctFinalByte':int(byte(qa[...,None]*qf+(1-qa[...,None])*qb)[0,0,0]),'byteIfLayersRoundedBeforeComposition':int(byte(qa[...,None]*byte(qf)+(1-qa[...,None])*byte(qb))[0,0,0]),'currentImplementationAffected':False}
    report['validationGaps']=['No actual near-zero slider requests in this replay; identity recomposition is not that regression.','Only one known exploration image at +/-30; no held-out scenes, stronger amounts, non-normal blend modes or transparent input.','Python float32 layers; native/WASM must preserve layer precision and composition order before claiming parity.','No proof that alpha, background prior, ownership or geometry are anatomically/materially correct.']
    (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
