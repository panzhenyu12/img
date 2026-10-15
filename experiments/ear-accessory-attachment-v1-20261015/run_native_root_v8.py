"""Native causal replay: source-derived ear hinges, unchanged material masks."""
from pathlib import Path
import json,os,subprocess,time,hashlib,shutil,sys
import numpy as np
from PIL import Image
from run_pivot import source_components,montage
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=HERE/'native-root-v8';out.mkdir(exist_ok=False);compiled=ROOT/'.run/ear-root-v8-target/release/image-render';digest=sha(compiled);binary=out/('image-render-'+digest);shutil.copy2(compiled,binary)
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2];parent=HERE/'native-v3-ear-input';oldnative=ROOT/'experiments/joint-skin-matting-v1-20261015/native-v2-user'
    params=json.loads((parent/'100/graph.json').read_text())['operations'][0]['params'];sem={k:raster_coverage(v,w,h) for k,v in params['semanticMasks'].items() if v['shape']=='raster'};D=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/user-face-contour/crop-point/alphas-full.npy')[2];J=np.load(HERE/'native-pivot-v4/source-attachments.npz')['jewelryAlpha'];lm=np.array([[p['x']*w,p['y']*h] for p in params['landmarks']]);c=source_components(sem['ears'],J,D,sem['skin'],lm)
    anchors=[[p['pivotSource'][0]+.5,p['pivotSource'][1]+.5,(p['semanticEarBounds'][3]-p['semanticEarBounds'][1])*.10] for p in c]
    ap=out/'source-hinge-anchors.json';ap.write_text(json.dumps(anchors)+'\n')
    report={'scope':__doc__,'sourceSha256':sha(source_path),'binarySha256':digest,'anchors':anchors,'components':[{k:v for k,v in q.items() if 'Alpha' not in k} for q in c],
      'rustSourceSha256':sha(HERE/'native-root-v8-core/src/face.rs'),'scriptSha256':sha(Path(__file__)),'rows':[],'accepted':False}
    for label,amount,enabled in [('default-control',100,False),('0',0,True),('100',100,True),('-100',-100,True)]:
        d=out/label;d.mkdir();graph=parent/str(amount)/'graph.json';shutil.copy2(graph,d/'graph.json');env={**os.environ,'FACE_LAYERS_DEBUG_DIR':str(d/'layers'),'FACE_V36_DEBUG_DIR':str(d/'debug'),'FACE_V21_DEBUG_DIR':str(d/'debug')}
        if enabled:env['EAR_ROOT_ANCHORS']=str(ap)
        start=time.perf_counter();p=subprocess.run([str(binary),'--input',str(source_path),'--graph',str(graph),'--output',str(d/'candidate.png')],env=env,capture_output=True);(d/'render.log').write_bytes(p.stdout+p.stderr);p.check_returncode();candidate=np.asarray(Image.open(d/'candidate.png').convert('RGBA'));base=np.asarray(Image.open(parent/str(amount)/'candidate.png').convert('RGBA'));old=np.asarray(Image.open(oldnative/str(amount)/'candidate.png').convert('RGBA'))
        row={'amount':amount,'label':label,'elapsedMs':(time.perf_counter()-start)*1000,'sourceIdentity':bool(np.array_equal(candidate,source)),'sameAsNewEarFixedBaseline':bool(np.array_equal(candidate,base)),'outputSha256':sha(d/'candidate.png')}
        if not enabled:assert np.array_equal(candidate,base)
        if amount==0:assert np.array_equal(candidate,source)
        actual=d/'debug/actual-source-width.json'
        if actual.exists():
            values=json.loads(actual.read_text());row['maxActual']=[max(abs(r['points'][s]['actualInward']) for r in values['rows'] if r['points'][s]['valid']) for s in range(2)]
        for name,box in [('left',[305,280,430,550]),('right',[580,280,715,550]),('face',[287,197,730,709])]:montage(d/(name+'-1x.png'),[source,old,base,candidate],['Source','Native v2','New ear fixed','New ear hinge field'],box)
        report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
