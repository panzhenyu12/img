"""Separate observed below-ear strip from raw ZIM's included ear lobe."""
from audit import *
def main():
    r=json.loads((EXP/'outputs/report.json').read_text());src=np.asarray(Image.open(ROOT/r['source']).convert('RGB'));h,w=src.shape[:2]
    graph=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text());pm=graph['operations'][0]['params'];sem={k:raster_coverage(v,w,h) for k,v in pm['semanticMasks'].items() if v['shape']=='raster'}
    lm=np.array([[p['x']*w,p['y']*h] for p in pm['landmarks']]);mid=float(lm[:,0].mean());fw=float(np.ptp(lm[:,0]));fh=float(np.ptp(lm[:,1]));Y,X=np.mgrid[:h,:w]
    oldpath=ROOT/'experiments/face-material-fill-v1-20261015/user-face-contour-100-full/layers';st=json.loads((oldpath/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];old=np.zeros((h,w));old[oy:oy+ch,ox:ox+cw]=np.fromfile(oldpath/'source-alpha-f32le.bin','<f4').reshape(ch,cw)
    af=np.load(EXP/'outputs/full-point/alphas-full.npy')[2];ac=np.load(EXP/'outputs/crop-point/alphas-full.npy')[2]
    v2=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');opaque=v2['opaque'];support=v2['pure']|v2['trusted'];local=np.asarray(Image.open(HERE/'new-outside-core.png'))>0
    rows=[];alllower=np.zeros((h,w),bool)
    for side in ['left','right']:
        half=(X<mid) if side=='left' else (X>=mid)
        # Raster-pitch margin keeps below-ear evidence separate from the raw
        # model's included lobe, without coordinate choices from this image.
        py=float(Y[half&(sem['ears']>1/255)].max());margin=max(h/pm['semanticMasks']['ears']['height']*2,2)
        lower=local&half&(Y>py+margin)&(Y<lm[:,1].max()-fh*.08);alllower|=lower
        rgb=src[lower];ys,xs=np.where(lower);notv2=lower&~support
        rows.append({'side':side,'earSupportBottomY':py,'parserPitchMarginPx':margin,'newBelowEarSupportPixels':int(lower.sum()),'newBelowEarNotV2Pixels':int(notv2.sum()),'newBelowEarOpaqueMetalPixels':int((lower&opaque).sum()),
            'rgbMin':rgb.min(axis=0).tolist() if len(rgb) else None,'rgbMedian':np.median(rgb,axis=0).tolist() if len(rgb) else None,'rgbMax':rgb.max(axis=0).tolist() if len(rgb) else None,
            'sourceBounds':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
            'fullPointMeanAlpha':float(af[lower].mean()),'cropPointMeanAlpha':float(ac[lower].mean()),'oldMeanAlpha':float(old[lower].mean())})
    overlay=src.copy();overlay[alllower]=np.rint(src[alllower]*.25+np.array([0,255,255])*.75).astype(np.uint8)
    for label,box in [('left',(320,395,420,550)),('right',(600,395,705,550))]:
        panels(HERE/(label+'-below-ear-1x.png'),[('Source',src),('Below-ear support',overlay),('Full official2',cutout(src,af)),('Crop official2',cutout(src,ac))],box)
    Image.fromarray((alllower*255).astype(np.uint8)).save(HERE/'below-ear-support.png')
    out={'scope':'Read-only source support evidence below the complete observed ear support + 2 parser pitches. No production mask, no output rendering, no guarantee of material truth.','rows':rows,'scriptSha256':sha(Path(__file__))}
    (HERE/'below-ear-report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
