"""Compare both fields at identical source coordinates, not row indices."""
from pathlib import Path
import json
import numpy as np
from attachment import bilinear
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def transport(inverse,points):
    p=np.asarray(points).reshape(1,-1,2);q=p.copy()
    for _ in range(40):q+=(p-bilinear(inverse,q))*.8
    error=np.linalg.norm(p-bilinear(inverse,q),axis=-1).ravel()
    return q.reshape(-1,2),error


def main():
    dest=HERE/'native-v3-ear-input';parent=ROOT/'experiments/joint-skin-matting-v1-20261015/native-v2-user';report={'scope':__doc__,'rows':[]}
    for amount in [30,-30,100,-100]:
        base=HERE/'native-v2-layers'/str(amount)/'layers';new=dest/str(amount)/'layers';st=json.loads((base/'state.json').read_text());origin=np.array(st['origin'])+.5
        inv0=np.fromfile(base/'inverse-source-xy-f32le.bin','<f4').reshape(st['height'],st['width'],2);inv1=np.fromfile(new/'inverse-source-xy-f32le.bin','<f4').reshape(st['height'],st['width'],2)
        rows=json.loads((parent/str(amount)/'debug/actual-source-width.json').read_text())['rows'];frame=json.loads((parent/str(amount)/'debug/flow-geometry.json').read_text());across=np.array(frame['anatomicalAcross']);sides=[]
        for s in range(2):
            probes=[r['points'][s]['source'] for r in rows if r['points'][s]['valid'] and abs(r['points'][s]['actualInward'])>=1]
            points=np.asarray(probes)-origin;a,e0=transport(inv0,points);b,e1=transport(inv1,points);d0=((a-points)@across)*(1 if s==0 else -1);d1=((b-points)@across)*(1 if s==0 else -1);ret=np.abs(d1/d0)
            sides.append({'side':['left','right'][s],'sourceProbeCount':len(points),'maxInverseSolveResidualPx':float(max(e0.max(),e1.max())),
              'meanAbsDisplacementNativeV2Px':float(np.mean(np.abs(d0))),'meanAbsDisplacementNewEarsPx':float(np.mean(np.abs(d1))),
              'meanDisplacementRetention':float(np.sum(np.abs(d1))/np.sum(np.abs(d0))),'minProbeRetention':float(ret.min()),'probesBelow80Percent':int(np.count_nonzero(ret<.8))})
        report['rows'].append({'amount':amount,'sides':sides})
    path=dest/'actual-retention-identical-source-probes.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
