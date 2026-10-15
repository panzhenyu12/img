"""Known-pixel occlusion test of background priors, separate from aesthetics.

Automatically choose an interior source hair crop, hide its entire right half,
and evaluate the first strip against the original withheld RGB. The original
right half is available only for the explicitly labelled oracle-context run.
"""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from run import ROOT,HERE,virtual_context,LamaWorker,sha,montage
sys.path.insert(0,str(ROOT/'scripts'))
from replay_user_face_contour import raster_coverage


def error(predicted,truth,target):
    delta=np.abs(predicted.astype('float64')-truth.astype('float64'))
    py,px=np.gradient(predicted.astype('float64'),axis=(0,1));ty,tx=np.gradient(truth.astype('float64'),axis=(0,1))
    return dict(meanAbsoluteRGBByte=float(delta[target].mean()),
        p95AbsoluteRGBByte=float(np.quantile(delta[target],.95)),
        gradientAbsoluteByte=float((np.abs(py-ty)+np.abs(px-tx))[target].mean()/2))


def main():
    out=HERE/'holdout';out.mkdir(exist_ok=False)
    source_report=ROOT/'experiments/hair-skin-decomposition-v1-20261015/source-decomposition-corrected-prompts/report.json'
    cases=[r for r in json.loads(source_report.read_text())['rows'] if r['variant']=='source-fit-srgb']
    report=dict(scope='Known-pixel synthetic half-plane occlusion. No claim this proves actual hidden texture or complete retouch quality.',
        scriptSha256=sha(Path(__file__)),sourceReportSha256=sha(source_report),rows=[])
    worker=LamaWorker(threads=3)
    for case in cases:
        name=case['name'];source_path=ROOT/case['source'];graph_path=ROOT/case['graph'];alpha_path=ROOT/case['selectedAlpha']
        source=np.asarray(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2]
        params=json.loads(graph_path.read_text())['operations'][0]['params']
        hair=raster_coverage(params['semanticMasks']['hair'],w,h)
        alpha=np.load(alpha_path)[case['officialIndex']]
        interior=(hair>=.95)&(alpha>=.95)
        distance=distance_transform_edt(np.pad(interior,1))[1:-1,1:-1]
        cy,cx=np.unravel_index(np.argmax(distance),distance.shape)
        size=min(256,int(2*distance[cy,cx]/np.sqrt(2))//8*8)
        if size<32:
            report['rows'].append(dict(name=name,abstain='No sufficient observed hair interior',maximumRadius=float(distance.max())));continue
        x0,y0=cx-size//2,cy-size//2;truth=source[y0:y0+size,x0:x0+size].copy()
        assert interior[y0:y0+size,x0:x0+size].all()
        yy,xx=np.indices((size,size));known=xx<size//2;missing=~known
        target=(xx>=size//2)&(xx<size//2+max(4,size//12))&(yy>=size//8)&(yy<7*size//8)
        virtual,reflected,donors=virtual_context(truth,known)
        # Holdout is excluded even from virtual-context construction.
        damaged=truth.copy();damaged[missing,:3]=255-truth[missing,:3]
        assert np.array_equal(virtual_context(damaged,known)[0],virtual)
        dest=out/name;dest.mkdir();Image.fromarray(truth).save(dest/'truth.png');Image.fromarray((target*255).astype('uint8')).save(dest/'scored-strip.png')
        rows=[];images=[truth,virtual];labels=['Withheld truth','Reflected prior']
        rows.append(dict(mode='prior-only',**error(virtual[...,:3],truth[...,:3],target)))
        for mode,context,mask in [('unknown-half',truth,missing),('virtual-context',virtual,target),('oracle-right-context',truth,target)]:
            result,meta=worker.inpaint(Image.fromarray(context),mask.astype('uint8')*255,max_side=max(256,size),context_mode='full',mask_dilation=0,seam_blend='none')
            result.save(dest/(mode+'.png'));prediction=np.asarray(result)
            rows.append(dict(mode=mode,**error(prediction[...,:3],truth[...,:3],target),meta=meta))
            images.append(prediction);labels.append(mode)
            print(name,mode,rows[-1]['meanAbsoluteRGBByte'],flush=True)
        montage(images,labels,dest/'comparison-1x.png')
        report['rows'].append(dict(name=name,sourceSha256=sha(source_path),graphSha256=sha(graph_path),alphaSha256=sha(alpha_path),crop=[int(x0),int(y0),int(x0+size),int(y0+size)],
            cropRule='largest inscribed confident-hair circle, square within it, source labels only',hiddenRGBContextChanges=0,scoredPixels=int(target.sum()),results=rows))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
