"""Quantify the distributed interface lost by a single rigid-ear pivot."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from verify_native_ears import transport
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'v8-seam-audit';out.mkdir(exist_ok=False);r=json.loads((HERE/'movable-ear-v8/report.json').read_text());mat=np.load(HERE/'movable-ear-v8/source-materials.npz');E=mat['earAlpha'];J=mat['jewelryAlpha'];D=E+(1-E)*mat['skinAlpha'];source=Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGB');h,w=E.shape;report={'scope':__doc__,'rows':[],'accepted':False}
    for row in r['rows']:
        amount=row['amount'];base=HERE/'native-root-v8'/str(amount);st=json.loads((base/'layers/state.json').read_text());o=np.array(st['origin']);iv=np.fromfile(base/'layers/inverse-source-xy-f32le.bin','<f4').reshape(st['height'],st['width'],2);m=next(v for v in row['groups'] if v['group']=='pivot-pair')['motions'];entries=[]
        for c,motion in zip(r['components'],m):
            side=c['side'];pivot=np.array(c['pivotSource']);lobe=np.array(c['lobeSource']);points=[]
            for y in range(int(pivot[1]),int(lobe[1])+1):
                xs=np.flatnonzero(E[y]>.5);xs=xs[xs<w/2] if side=='left' else xs[xs>=w/2]
                if not len(xs):continue
                x=xs.max() if side=='left' else xs.min()
                if D[y,x]<.8 or J[y,x]>.05:continue
                points.append([x,y])
            points=np.asarray(points,float);face,res=transport(iv,points-o);face+=o;aff=np.array(motion['earAffine']);ear=(points-o)@aff[:,:2].T+aff[:,2]+o;gap=np.linalg.norm(ear-face,axis=1)
            v=points-pivot;target=face-pivot;angle=np.arctan2(np.sum(v[:,0]*target[:,1]-v[:,1]*target[:,0]),np.sum(v*target));co,si=np.cos(angle),np.sin(angle);R=np.array([[co,-si],[si,co]]);best=v@R.T+pivot;bestgap=np.linalg.norm(best-face,axis=1);lobeactual=aff[:,:2]@(lobe-o)+aff[:,2]+o;lobebest=R@(lobe-pivot)+pivot
            entry={'side':side,'sourceInterfaceSamples':len(points),'maxNativeInverseSolveResidualPx':float(res.max()),'currentInterfaceRmsPx':float(np.sqrt(np.mean(gap*gap))),
              'currentInterfaceMaxPx':float(gap.max()),'bestRootFixedRigidRmsPx':float(np.sqrt(np.mean(bestgap*bestgap))),'bestRigidMaxPx':float(bestgap.max()),
              'lobePositionDifferenceIfBestRigidFitPx':float(np.linalg.norm(lobebest-lobeactual)),'currentEarRotationDeg':float(np.degrees(np.arctan2(aff[1,0],aff[0,0]))),'bestRootFixedRigidRotationDeg':float(np.degrees(angle))}
            entries.append(entry)
            lo=np.floor(np.minimum(points.min(axis=0),np.minimum(face.min(axis=0),ear.min(axis=0)))-12).astype(int);hi=np.ceil(np.maximum(points.max(axis=0),np.maximum(face.max(axis=0),ear.max(axis=0)))+12).astype(int);crop=source.crop((*lo,*hi)).resize(tuple((hi-lo)*4),Image.Resampling.NEAREST);draw=ImageDraw.Draw(crop)
            for ps,color in [(points,'#ffffff'),(face,'#00ffff'),(ear,'#ff8800')]:draw.line([tuple((p-lo)*4) for p in ps],fill=color,width=2)
            crop.save(out/f'{amount}-{side}-interface-4x.png')
        report['rows'].append({'amount':amount,'sides':entries})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
