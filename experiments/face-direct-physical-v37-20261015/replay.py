"""Single-variable physical-alpha experiment; no production/API schema change.

Original native-v2 velocity is frozen by construction and verified by debug
hash. Direct float matte replaces only the post-velocity sampling opacity.
Raw neural opacity remains an estimate. Original source and zero are retained.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,shutil
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import montage
from run_real import probe
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(name,amounts,variant):
    label={'user-face-contour':'user','portrait-woman':'woman','portrait-hair':'hair'}[name]
    base=ROOT/f'experiments/joint-skin-matting-v1-20261015/native-v2-{label}'
    out=HERE/variant/label;out.mkdir(parents=True,exist_ok=False)
    model=ROOT/f'experiments/joint-skin-matting-v1-20261015/prompts-v2/{name}/crop-point/alphas-full.npy'
    matte=np.load(model)[2];physical=out/'physical-alpha-f32le.bin';matte.astype('<f4').tofile(physical)
    binary=ROOT/'.run/face-direct-physical-v37-target/release/image-render'
    frozen=out/f'image-render-{sha(binary)}';shutil.copy2(binary,frozen)
    src=ROOT/probe.SOURCES[name][0];orig=Image.open(src).convert('RGBA')
    report={'description':__doc__,'sourceSha256':sha(src),'binarySha256':sha(frozen),'modelAlphaSha256':sha(model),'rows':[],'accepted':False,'variant':variant}
    for amount in amounts:
        d=out/str(amount);d.mkdir();graph=base/str(amount)/'graph.json'
        t=time.perf_counter();p=subprocess.run([str(frozen),'--input',str(src),'--graph',str(graph),'--output',str(d/'candidate.png')],env={**os.environ,**({'FACE_V37_SOURCE_RESIDUAL':'1'} if variant=='outputs-delta' else {}),'FACE_V37_PHYSICAL_ALPHA':str(physical),'FACE_V36_DEBUG_DIR':str(d/'debug'),'FACE_V21_DEBUG_DIR':str(d/'debug'),'FACE_V37_LAYER_DEBUG_DIR':str(d/'layers')},capture_output=True)
        (d/'render.log').write_bytes(p.stdout+p.stderr);p.check_returncode()
        row={'amount':amount,'elapsedMs':(time.perf_counter()-t)*1000,'outputSha256':sha(d/'candidate.png'),'graphSha256':sha(graph)}
        if amount:
            assert sha(d/'debug/flow-geometry.json')==sha(base/str(amount)/'debug/flow-geometry.json')
            assert sha(d/'debug/actual-source-width.json')==sha(base/str(amount)/'debug/actual-source-width.json')
            row['geometryExact']=True
        else:assert np.array_equal(np.asarray(orig),np.asarray(Image.open(d/'candidate.png').convert('RGBA')))
        params=json.loads(graph.read_text())['operations'][0]['params'];box=probe.face_crop(params,orig.width,orig.height)
        for key,box in [('face',box)]+([('left',[320,400,420,550]),('right',[590,400,700,550])] if label=='user' else []):
            montage(d/f'{key}-1x.png',[orig,Image.open(base/str(amount)/'candidate.png'),Image.open(d/'candidate.png')],['Source','Native refined alpha','Direct alpha '+variant+', same flow'],box)
        report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--amounts',nargs='+',type=int,default=[0,100,-100]);p.add_argument('--variant',default='outputs');a=p.parse_args();main(a.name,a.amounts,a.variant)
