"""Native input intervention: replace only semantic ear coverage.

The original native-v2 direct skin/accessory inputs remain unchanged. Geometry
is allowed to respond to the recovered ear/lobe support; its actual width
movement is measured rather than assumed equivalent. Isolated frozen binary.
"""
from pathlib import Path
import json,os,subprocess,time,hashlib
import numpy as np
from PIL import Image
from run_pivot import montage
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=HERE/'native-v3-ear-input';out.mkdir(exist_ok=False)
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2]
    parent=ROOT/'experiments/joint-skin-matting-v1-20261015/native-v2-user'
    prov=json.loads((ROOT/'experiments/face-material-fill-v1-20261015/user-face-contour-100-full/provenance.json').read_text());binary=Path(prov['binary']);assert sha(binary)==prov['binarySha256']
    raw=HERE/'ear-matte-v6';raw_report=json.loads((raw/'report.json').read_text());paths=[];parts=[]
    for side in ['left','right']:
        row=next(r for r in raw_report['rows'] if r['side']==side and r['attention']=='point');path=raw/f'{side}-point/alphas-full.npy';paths.append(path);parts.append(np.load(path)[row['officialScoreArgmax']])
    ear=np.maximum.reduce(parts);ratio=min(1,512/max(h,w));small=np.asarray(Image.fromarray(ear.astype('float32'),mode='F').resize((round(w*ratio),round(h*ratio)),Image.Resampling.BOX));sh,sw=small.shape;raster=dict(shape='raster',width=sw,height=sh,data=np.rint(np.clip(small,0,1)*255).astype('uint8').ravel().tolist());np.save(out/'ear-alpha-full.npy',ear,allow_pickle=False)
    report={'scope':__doc__,'sourceSha256':sha(source_path),'binarySha256':sha(binary),'earRawSha256':[sha(p) for p in paths],'scriptSha256':sha(Path(__file__)),'rows':[],'accepted':False}
    for amount in [0,30,-30,100,-100]:
        d=out/str(amount);d.mkdir();graph=json.loads((parent/str(amount)/'graph.json').read_text());graph['operations'][0]['params']['semanticMasks']['ears']=raster;gp=d/'graph.json';gp.write_text(json.dumps(graph,separators=(',',':')))
        start=time.perf_counter();proc=subprocess.run([str(binary),'--input',str(source_path),'--graph',str(gp),'--output',str(d/'candidate.png')],env={**os.environ,'FACE_LAYERS_DEBUG_DIR':str(d/'layers'),'FACE_V36_DEBUG_DIR':str(d/'debug'),'FACE_V21_DEBUG_DIR':str(d/'debug')},capture_output=True);(d/'render.log').write_bytes(proc.stdout+proc.stderr);proc.check_returncode()
        candidate=np.asarray(Image.open(d/'candidate.png').convert('RGBA'));native=np.asarray(Image.open(parent/str(amount)/'candidate.png').convert('RGBA'));row={'amount':amount,'graphSha256':sha(gp),'outputSha256':sha(d/'candidate.png'),'elapsedMs':(time.perf_counter()-start)*1000,'changedPixelsVsNativeV2':int(np.any(candidate!=native,axis=2).sum()),'zeroIdentity':bool(np.array_equal(candidate,source)) if amount==0 else None}
        if amount==0:assert np.array_equal(candidate,source)
        actual=d/'debug/actual-source-width.json'
        if actual.exists():
            values=json.loads(actual.read_text());row['maxActual']=[max(abs(r['points'][s]['actualInward']) for r in values['rows'] if r['points'][s]['valid']) for s in range(2)]
        layers=d/'layers'
        if layers.exists():row['layersSha256']={p.name:sha(p) for p in layers.iterdir()}
        for name,box in [('left',[305,280,430,550]),('right',[580,280,715,550]),('face',[287,197,730,709])]:montage(d/(name+'-1x.png'),[source,native,candidate],['Source','Native v2','Native new ear input'],box)
        report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='layersSha256'}),flush=True)


if __name__=='__main__':main()
