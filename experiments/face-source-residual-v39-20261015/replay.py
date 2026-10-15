from pathlib import Path
import argparse,hashlib,json,subprocess,time,sys,shutil
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import montage
from run_real import probe
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(name):
    out=HERE/'outputs'/name;out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'core/image-core/target/release/image-render'
    assert sha(base)=='d0fe0b5251f2f4e862ee20dc7381497fde0f198369f7f37b01eb012c8b68a434'
    candidate=ROOT/'.run/face-source-residual-v39-target/release/image-render'
    binary=out/f'image-render-{sha(candidate)}';shutil.copy2(candidate,binary)
    rows=[]
    for case in json.loads((HERE/'manifest.json').read_text()):
        if case['name']!=name:continue
        src=ROOT/case['source'];graph=ROOT/case['graph'];amount=case['amount'];d=out/str(amount);d.mkdir()
        origin=Image.open(src).convert('RGBA');a=np.asarray(origin)
        params=json.loads(graph.read_text())['operations'][0]['params'];masks={k:probe.zim.coverage(v,origin.width,origin.height) for k,v in params['semanticMasks'].items()}
        frozen=np.maximum.reduce([masks[k] for k in ['ears','hair','accessories']])>=.5
        times=[]
        for label,b in [('current',base),('candidate',binary)]:
            start=time.perf_counter();p=subprocess.run([str(b),'--input',str(src),'--graph',str(graph),'--output',str(d/f'{label}.png')],capture_output=True)
            (d/f'{label}.log').write_bytes(p.stdout+p.stderr);p.check_returncode();times.append((time.perf_counter()-start)*1000)
        before=Image.open(d/'current.png').convert('RGBA');after=Image.open(d/'candidate.png').convert('RGBA');b=np.asarray(before);c=np.asarray(after)
        assert np.array_equal(c[...,3],a[...,3]);assert np.array_equal(c[frozen],a[frozen])
        if amount==0:assert np.array_equal(c,a)
        box=probe.face_crop(params,origin.width,origin.height)
        panels=[('face',box)]+([('left',[320,400,420,550]),('right',[590,400,700,550])] if name=='user-face-contour' else [])
        for label,box in panels:montage(d/f'{label}-1x.png',[origin,before,after],['Source','Current v19 + fringe-v2','V39 source residual'],box)
        row={**case,'sourceSha256':sha(src),'graphSha256':sha(graph),'candidateSha256':sha(d/'candidate.png'),'baselineSha256':sha(d/'current.png'),'candidateBinarySha256':sha(binary),'changedFromCurrent':int(np.any(c!=b,axis=2).sum()),'maxDifferenceFromCurrent':int(np.abs(c.astype(int)-b).max()),'alphaChanged':0,'fixedChanged':0,'baselineMs':times[0],'candidateMs':times[1]}
        rows.append(row);(out/'report.json').write_text(json.dumps({'accepted':False,'rows':rows},indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();main(a.name)
