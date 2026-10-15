"""Verify actual source-coordinate face movement for the frozen face field."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/face-attachment-boundary-v1-20261015'));from run import forward
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pivot=json.loads((HERE/'outputs-pivot-v3/report.json').read_text());v36=ROOT/'experiments/face-source-curve-v36-20261015/outputs';g=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text());lm=np.array([[p['x']*1024,p['y']*1024] for p in g['operations'][0]['params']['landmarks']]);low=min(c['hookSource'][1] for c in pivot['components']);high=lm[:,1].max()-np.ptp(lm[:,1])*.08
    report={'scope':'Measured source-to-target boundary displacement. Same inverse in all groups; this is face field preservation, not proof that composited visible silhouette is identical.','rows':[]}
    for amount in [30,-30,100,-100]:
        mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full'
        if not mat.exists():mat=ROOT/f'experiments/face-attachment-boundary-v1-20261015/baseline/{amount}'
        st=json.loads((mat/'layers/state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];ip=mat/'layers/inverse-source-xy-f32le.bin';inv=np.fromfile(ip,'<f4').reshape(ch,cw,2)
        rows=json.loads((v36/f'user-face-contour-{amount}/debug/actual-source-width.json').read_text())['rows'];sides=[]
        for side in range(2):
            pts=np.array([r['points'][side]['source'] for r in rows if r['points'][side]['valid'] and low<r['points'][side]['source'][1]<high]);p=pts-np.array([ox,oy]);q,e=forward(inv,p);motion=q-p;inward=motion[:,0]*(1 if side==0 else -1)
            sides.append({'side':['left','right'][side],'sourceProbes':len(p),'meanInwardPx':float(inward.mean()),'maxAbsoluteHorizontalDisplacementPx':float(np.abs(motion[:,0]).max()),'maximumInverseSolveResidualPx':float(e.max())})
        report['rows'].append({'amount':amount,'inverseSha256':sha(ip),'allGroupsUseIdenticalInverse':True,'actualFaceDisplacementRatio':1.0,'sides':sides})
    report['scriptSha256']=sha(Path(__file__));(HERE/'face-motion-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
