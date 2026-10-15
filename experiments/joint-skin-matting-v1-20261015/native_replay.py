"""Input intervention into frozen V36, not a production schema change.

Direct skin alpha is injected into the existing skin seed only to test whether
the complete shared pipeline benefits from recovered source ownership. Physical
alpha and semantic confidence remain distinct in the intended integration.
Geometry may adapt to the new attachment masks; actual movement is recorded.
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import montage
from run_real import probe
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main(name='user-face-contour',matte=None,out=None):
    out=out or HERE/'native-v1-512';out.mkdir(exist_ok=False)
    frozen=ROOT/'experiments/face-source-curve-v36-20261015/outputs'
    binary=frozen/'image-render-ac0f43bcdb8d4a96e78b80daab37dca1c2f8f701884bb8745b1e60a6d1c94a0b'
    assert sha(binary)==binary.name.removeprefix('image-render-')
    source=ROOT/probe.SOURCES[name][0]
    matte=matte or HERE/'outputs/crop-point/alphas-full.npy'
    skin=np.load(matte)[2]
    accessory=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz')['alpha'] if name=='user-face-contour' else None
    def raster(a):
        # Existing public raster contract is <=512 on each dimension. This
        # diagnostic includes that real quantization/resampling limitation.
        h,w=a.shape;ratio=min(1,512/max(h,w));size=(round(w*ratio),round(h*ratio))
        a=np.asarray(Image.fromarray(a.astype('float32'),mode='F').resize(size,Image.Resampling.BOX))
        h,w=a.shape
        return dict(shape='raster',width=w,height=h,data=np.rint(np.clip(a,0,1)*255).astype('uint8').ravel().tolist())
    rasters={'skin':raster(skin)}
    if accessory is not None:rasters['accessories']=raster(accessory)
    source_im=Image.open(source).convert('RGBA')
    report=dict(scope=__doc__,sourceSha256=sha(source),binarySha256=sha(binary),matteSha256=sha(matte),runnerSha256=sha(Path(__file__)),rows=[],accepted=False)
    for amount in [0,30,-30,100,-100]:
        baseline=frozen/f'{name}-{amount}'
        graph=json.loads((baseline/'graph.json').read_text())
        graph['operations'][0]['params']['semanticMasks'].update(rasters)
        d=out/str(amount);d.mkdir();gp=d/'graph.json';gp.write_text(json.dumps(graph,separators=(',',':')))
        start=time.perf_counter()
        proc=subprocess.run([str(binary),'--input',str(source),'--graph',str(gp),'--output',str(d/'candidate.png')],
                            env={**os.environ,'FACE_V36_DEBUG_DIR':str(d/'debug'),'FACE_V21_DEBUG_DIR':str(d/'debug')},capture_output=True)
        (d/'render.log').write_bytes(proc.stdout+proc.stderr);proc.check_returncode()
        candidate=Image.open(d/'candidate.png').convert('RGBA');base=Image.open(baseline/'candidate.png').convert('RGBA')
        if amount==0:assert np.array_equal(np.asarray(candidate),np.asarray(source_im))
        box=probe.face_crop(graph['operations'][0]['params'],source_im.width,source_im.height)
        panels=[('face',box)]
        if name=='user-face-contour':panels += [('left',[320,400,420,550]),('right',[590,400,700,550])]
        for panel_name,box in panels:
            montage(d/f'{panel_name}-1x.png',[source_im,base,candidate],['Source','V36 original input','Native new source seed'],box)
        row=dict(amount=amount,graphSha256=sha(gp),outputSha256=sha(d/'candidate.png'),elapsedMs=(time.perf_counter()-start)*1000)
        actual=d/'debug/actual-source-width.json'
        if actual.exists():
            j=json.loads(actual.read_text())
            row['maxActual']=[max(abs(r['points'][s]['actualInward']) for r in j['rows'] if r['points'][s]['valid'])for s in range(2)]
        report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='user-face-contour')
    parser.add_argument('--matte',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();main(args.name,args.matte,args.out)
