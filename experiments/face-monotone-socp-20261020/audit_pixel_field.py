#!/usr/bin/env python3
"""Independent full-cell QC of the actual float32 pixel reconstruction."""
import argparse,json
from pathlib import Path
import numpy as np


def inspect(displacement, spacing=(1., 1.), origin=(0., 0.)):
    a=np.asarray(displacement,dtype=float)
    if a.ndim!=3 or a.shape[2]!=2 or min(a.shape[:2])<2 or not np.isfinite(a).all():
        raise ValueError('invalid finite displacement grid')
    hx=np.diff(a,axis=1)/spacing[0];hy=np.diff(a,axis=0)/spacing[1]
    matrices=[(hx[:-1],hy[:,:-1]),(hx[:-1],hy[:,1:]),(hx[1:],hy[:,:-1]),(hx[1:],hy[:,1:])]
    lows=[];highs=[];skews=[];dets=[]
    for x,y in matrices:
        xx,xy,yx,yy=x[:,:,0]+1,y[:,:,0],x[:,:,1],y[:,:,1]+1
        radius=np.hypot(xx-yy,xy+yx)
        lows.append((xx+yy-radius)*.5);highs.append((xx+yy+radius)*.5)
        skews.append(np.abs(xy-yx)*.5);dets.append(xx*yy-xy*yx)
    lo,hi,sk,det=map(np.stack,[lows,highs,skews,dets])
    idx=np.unravel_index(np.argmin(lo),lo.shape);worst=[float(origin[0]+idx[2]*spacing[0]),float(origin[1]+idx[1]*spacing[1])]
    boundary=np.r_[a[0],a[-1],a[1:-1,0],a[1:-1,-1]]
    return {'symmetric_eigenvalue_min':float(lo.min()),'symmetric_eigenvalue_max':float(hi.max()),
        'skew_absolute_max':float(sk.max()),'jacobian_determinant_min':float(det.min()),'jacobian_determinant_max':float(det.max()),
        'boundary_max_displacement':float(np.linalg.norm(boundary,axis=1).max()),
        'cell_count':int(np.prod(a.shape[:2]-np.array([1,1]))),'corner_count':int(lo.size),
        'corners_below_symmetric_min_052':int((lo<.52).sum()),'corners_above_symmetric_max_18':int((hi>1.8).sum()),
        'corners_above_skew_024':int((sk>.24).sum()),'corners_nonpositive_determinant':int((det<=0).sum()),
        'worst_lower_cell_origin':worst,'worst_lower_corner':['TL','TR','BL','BR'][idx[0]],
        'passed_strict':bool(lo.min()>=.52 and hi.max()<=1.8 and sk.max()<=.24 and det.min()>0 and np.all(boundary==0)),
        'meaning':'All four one-sided Jacobians of every bilinear reconstruction cell. PSD convexity extends symmetric bounds within each cell; a coarse-grid certificate is never substituted for this measurement.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--coarse-field',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.trace/'metadata.json').read_text());pixels=np.fromfile(a.trace/'inverse.f32',dtype='<f4').reshape(meta['height'],meta['width'],2)
    field=json.loads(a.coarse_field.read_text());g=field['grid'];coarse=np.array(field['displacement']).reshape(g['height'],g['width'],2)
    report={'coarse':inspect(coarse,g['spacing'],g['origin']),'final_float32_pixel_reconstruction':inspect(pixels,(1.,1.),(meta['left'],meta['top']))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return 0 if report['final_float32_pixel_reconstruction']['passed_strict'] else 2
if __name__=='__main__':raise SystemExit(main())
