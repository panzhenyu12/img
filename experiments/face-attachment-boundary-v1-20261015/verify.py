"""Verify realized rather than requested motion for the geometry-only ablation."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from run import ROOT,HERE,SOURCE,V36,MAT,sha,bilinear,forward,raster_coverage,curvature

def main():
    report=json.loads((HERE/'report.json').read_text());src=np.asarray(Image.open(SOURCE).convert('RGBA'));h,w=src.shape[:2];yy=np.arange(h,dtype=float)
    graph=json.loads((V36/'user-face-contour-100/graph.json').read_text());params=graph['operations'][0]['params'];m=params['semanticMasks'];lm=np.array([[p['x']*w,p['y']*h] for p in params['landmarks']]);fw=float(np.ptp(lm[:,0]));fh=float(np.ptp(lm[:,1]));jaw=float(lm[:,1].max())
    acc=raster_coverage(m['accessories'],w,h);ear=raster_coverage(m['ears'],w,h)
    v2=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');opaque=v2['opaque']
    widths=json.loads((V36/'user-face-contour-100/debug/actual-source-width.json').read_text())['rows'];sources=[];roi=np.zeros((h,w),bool)
    for si,node in enumerate(report['nodes']):
        p=np.array([r['points'][si]['source'] for r in widths if r['points'][si]['valid']]);p=p[np.argsort(p[:,1])];sx=np.interp(yy,p[:,1],p[:,0]);sources.append(sx)
        y0=node['lobe'][1];roi|=(yy[:,None]>=y0-5)&(yy[:,None]<=y0+fh*.25)&(np.abs(np.arange(w)[None,:]-sx[:,None])<fw*.12)
    out={'scope':'Independent realized map verification, fixed ear ROI for all variants. Geometry-only ablation remains rejected.','accepted':False,'rows':[]}
    lum=lambda a:np.dot(a[...,:3],[.2126,.7152,.0722])
    for row in report['rows']:
        amount=row['amount'];dest=HERE/row['directory'];mat=MAT/f'user-face-contour-{amount}-full'
        if not mat.exists():mat=HERE/'baseline'/str(amount)
        layers=mat/'layers';state=json.loads((layers/'state.json').read_text());ox,oy=state['origin'];cw,ch=state['width'],state['height']
        bi=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);ci=np.fromfile(dest/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2)
        sides=[]
        for si,s in enumerate(sources):
            p=np.stack([s-ox,yy-oy],axis=-1);b,rb=forward(bi,p);c,rc=forward(ci,p);b+=np.array([ox,oy]);c+=np.array([ox,oy]);node=report['nodes'][si];y0=node['lobe'][1]
            valid=(yy>=y0)&(yy<=jaw-fh*.08)&(rb<.1)&(rc<.1)&(np.abs(b[:,0]-s)>.25)
            real_ratio=float(np.abs(c[:,0]-s)[valid].sum()/np.abs(b[:,0]-s)[valid].sum())
            curvevalid=(yy>node['root'][1])&(yy<jaw)&(rb<.1)&(rc<.1)
            bx=np.interp(yy,b[curvevalid,1],b[curvevalid,0]);cx=np.interp(yy,c[curvevalid,1],c[curvevalid,0]);region=(yy>=y0)&(yy<=y0+fh*.25)
            ks,kb,kc=curvature(s),curvature(bx),curvature(cx)
            i=int(round(y0));slope=lambda x:float(np.arctan((x[i+5]-x[i])/5)-np.arctan((x[i]-x[i-5])/5))
            sides.append({'side':node['side'],'realizedDisplacementRatio':real_ratio,'observedRows':int(valid.sum()),'maxBaselineInverseSolveResidualPx':float(rb[valid].max()),'maxCandidateInverseSolveResidualPx':float(rc[valid].max()),
                          'baselineAddedCurvatureRms':float(np.sqrt(np.mean((kb[region]-ks[region])**2))),'candidateAddedCurvatureRms':float(np.sqrt(np.mean((kc[region]-ks[region])**2))),
                          'sourceEarTangentTurnRadians':slope(s),'baselineEarTangentTurnRadians':slope(bx),'candidateEarTangentTurnRadians':slope(cx)})
        b=np.asarray(Image.open(mat/'candidate.png').convert('RGBA'));c=np.asarray(Image.open(dest/'candidate.png').convert('RGBA'));change=np.any(c!=b,axis=2)
        verify={'amount':amount,'spanFraction':row['spanFraction'],'directory':row['directory'],'sides':sides,'minimumRealizedDisplacementRatio':min(s['realizedDisplacementRatio'] for s in sides),
          'baselineBlackWedgeFixedRoiProxy':int(((lum(b)<50)&(lum(src)>65)&roi).sum()),'candidateBlackWedgeFixedRoiProxy':int(((lum(c)<50)&(lum(src)>65)&roi).sum()),
          'opaqueMetalNewChanges':int((change&opaque).sum()),'opaqueMetalChangesVsSource':int((np.any(c!=src,axis=2)&opaque).sum()),'earCoreNewChanges':int((change&(ear>=.95)).sum()),
          'softAccessoryNewChanges':int((change&(acc>1/255)&(acc<.95)).sum()),'alphaChangedPixels':int((c[...,3]!=src[...,3]).sum()),
          'sourceAlphaHashStillEqual':sha(layers/'source-alpha-f32le.bin')==row['sourceAlphaSha256'],'backgroundHashStillEqual':sha(layers/'background.png')==row['backgroundSha256'],
          'foregroundHashStillEqual':sha(layers/'foreground.png')==row['foregroundSha256']}
        out['rows'].append(verify);print(amount,row['spanFraction'],round(verify['minimumRealizedDisplacementRatio'],4),verify['baselineBlackWedgeFixedRoiProxy'],verify['candidateBlackWedgeFixedRoiProxy'],verify['opaqueMetalNewChanges'])
    out['scriptSha256']=sha(Path(__file__));(HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
