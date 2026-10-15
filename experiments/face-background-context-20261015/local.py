"""Local context scale ablation: fixed final target, alpha, geometry and masks."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import label,binary_dilation
from context import ROOT,sha,compose,montage
sys.path.insert(0,str(ROOT/'services/background-worker'))
from worker import LamaWorker


def tiles(target,size):
    components,count=label(binary_dilation(target,iterations=2))
    boxes=[]
    for component in range(1,count+1):
        yy,xx=np.nonzero((components==component)&target)
        if not len(yy):continue
        centers=[float(np.median(yy))] if np.ptp(yy)<size/2 else np.arange(yy.min()+size/4,yy.max()+size/4,size/2)
        for cy in centers:
            band=np.abs(yy-cy)<=size/4
            cx=float(np.median(xx[band])) if band.any() else float(np.median(xx))
            x0=int(np.clip(round(cx-size/2),0,target.shape[1]-size));y0=int(np.clip(round(cy-size/2),0,target.shape[0]-size))
            boxes.append((x0,y0,x0+size,y0+size))
    covered=np.zeros(target.shape,bool)
    for x0,y0,x1,y1 in boxes:covered[y0:y1,x0:x1]=True
    if (target&~covered).any():raise RuntimeError('Deterministic tiles did not cover target')
    return sorted(set(boxes))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--diagnostic',type=Path,required=True);ap.add_argument('--ablation',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    base,out,prior=args.diagnostic.resolve(),args.output.resolve(),args.ablation.resolve()
    if out.exists():raise SystemExit('Refusing to overwrite frozen outputs')
    out.mkdir(parents=True)
    debug=base/'layers';state=json.loads((debug/'state.json').read_text());provenance=json.loads((base/'provenance.json').read_text());h,w=state['height'],state['width'];ox,oy=state['origin']
    source=np.asarray(Image.open(provenance['source']).convert('RGBA'));source_crop=np.asarray(Image.open(debug/'source-crop.png').convert('RGBA'));background=np.asarray(Image.open(debug/'background.png').convert('RGBA'));baseline=np.asarray(Image.open(base/'candidate.png').convert('RGBA'))
    sample=np.fromfile(debug/'samples-f32le.bin','<f4').reshape(h,w,5);target=np.asarray(Image.open(prior/'write-target.png'))>0;unknown=np.asarray(Image.open(prior/'hair-scene/inference-mask.png'))>0;frozen=np.fromfile(debug/'frozen-u8.bin','uint8').reshape(h,w)>0
    target_full=np.zeros(source.shape[:2],bool);target_full[oy:oy+h,ox:ox+w]=target
    report={'scope':'Local original-resolution context crop only. Same target, source, graph, foreground, known mask as full hair-scene ablation. 128/256/512 retained regardless of quality.',
        'inputHashes':{str(p.relative_to(ROOT)):sha(p) for p in [base/'candidate.png',base/'graph.json',prior/'write-target.png',prior/'hair-scene/inference-mask.png',debug/'samples-f32le.bin',debug/'background.png',Path(__file__)]},'rows':[]}
    worker=LamaWorker(threads=3)
    for size in [128,256,512]:
        dest=out/str(size);dest.mkdir();boxes=tiles(target,size);accum=np.zeros((h,w,3),np.float64);weights=np.zeros((h,w),np.float64);tile_reports=[]
        # Overlap blends only alternate predictions of the missing background.
        # It never smooths original or transported foreground pixels.
        window=np.maximum(np.hanning(size),1e-3);window=np.outer(window,window)
        for index,(x0,y0,x1,y1) in enumerate(boxes):
            patch=source_crop[y0:y1,x0:x1];missing=unknown[y0:y1,x0:x1]
            result,meta=worker.inpaint(Image.fromarray(patch),missing.astype('uint8')*255,max_side=max(256,size),context_mode='full',mask_dilation=0,seam_blend='none')
            result.save(dest/f'tile-{index:02}.png')
            conditioned=patch.copy();conditioned[missing,:3]=0;Image.fromarray(conditioned).save(dest/f'conditioning-{index:02}.png')
            accum[y0:y1,x0:x1]+=np.asarray(result)[:,:,:3]*window[:,:,None];weights[y0:y1,x0:x1]+=window
            tile_reports.append({'index':index,'box':list((x0,y0,x1,y1)),'meta':meta})
            print(json.dumps({'size':size,'tile':index+1,'total':len(boxes),'inferenceMs':meta['inference_ms']}),flush=True)
        assert np.all(weights[target]>0)
        layer=background.copy();layer[target,:3]=np.floor(accum[target]/weights[target,None]+.5).clip(0,255).astype('uint8')
        Image.fromarray(layer).save(dest/'background-layer.png')
        full_layer=source.copy();full_layer[oy:oy+h,ox:ox+w]=layer;Image.fromarray(full_layer).save(dest/'full-background-layer.png')
        new=compose(layer,sample[:,:,:4],sample[:,:,4]);result=baseline.copy();local=result[oy:oy+h,ox:ox+w];local[target]=new[target]
        changed=np.any(result!=baseline,axis=2)
        assert not (changed&~target_full).any();assert np.array_equal(result[:,:,3],baseline[:,:,3]);assert np.array_equal(local[frozen],baseline[oy:oy+h,ox:ox+w][frozen])
        Image.fromarray(result).save(dest/'candidate.png')
        row={'size':size,'tiles':tile_reports,'changedPixels':int(changed.sum()),'outsideTargetChanges':0,'frozenChanges':0,'alphaChanges':0,'foregroundBuffersUnchanged':True,'outputSha256':sha(dest/'candidate.png')};report['rows'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        montage([source,baseline,result],['Original','V36',f'Local hair-scene {size}'],dest/'face-1x.png',box=(300,480,900,1170))
        montage([source,baseline,result],['Original','V36',f'Local {size}'],dest/'left-edge-2x.png',box=(340,820,485,1110),zoom=2)
    images=[source,baseline]+[np.asarray(Image.open(out/str(s)/'candidate.png').convert('RGBA')) for s in [128,256,512]]
    montage(images,['Original','V36','Local128','Local256','Local512'],out/'face-1x.png',box=(300,480,900,1170))
    montage(images,['Original','V36','Local128','Local256','Local512'],out/'left-edge-2x.png',box=(340,820,485,1110),zoom=2)

if __name__=='__main__':main()
