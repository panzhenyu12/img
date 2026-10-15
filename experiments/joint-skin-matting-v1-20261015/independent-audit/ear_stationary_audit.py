"""Source RGB audit of remaining fixed-ear boundary; no repair or masking."""
from audit import *
def main():
    source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';src=np.asarray(Image.open(source).convert('RGB'));h,w=src.shape[:2];g=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text());ear=raster_coverage(g['operations'][0]['params']['semanticMasks']['ears'],w,h);acc=raster_coverage(g['operations'][0]['params']['semanticMasks']['accessories'],w,h)
    cached=EXP/'composite-v1b';l=np.load(cached/'source-layers.npz');direct=l['skinAlpha'];j=l['jewelry'];fixed=np.asarray(Image.open(cached/'fixed.png'))>127;recon=np.asarray(Image.open(cached/'source-reconstruction.png').convert('RGB'));out=np.asarray(Image.open(cached/'100/candidate.png').convert('RGB'));res=np.abs(recon.astype(float)-src).max(axis=2)
    # Diagnostic boxes refer to the observed failure only; not an algorithm rule.
    rows=[]
    for label,box in [('right',(640,418,665,445)),('left',(339,401,367,432))]:
        x0,y0,x1,y1=box;hot=np.zeros((h,w),bool);hot[y0:y1,x0:x1]=True;hot&=res>25
        pix=[]
        for y,x in zip(*np.where(hot)):
            pix.append({'xy':[int(x),int(y)],'sourceRgb':src[y,x].tolist(),'reconstructedRgb':recon[y,x].tolist(),'outputRgb':out[y,x].tolist(),'earAlpha':float(ear[y,x]),'directSkinAlpha':float(direct[y,x]),'jewelryAlpha':float(j[y,x]),'fixed':bool(fixed[y,x]),'sourceResidualMaxByte':float(res[y,x])})
        rows.append({'side':label,'box':box,'highResidualPixels':len(pix),'pixels':pix})
        maps=[('Source',src),('Rendered +100',out),('Ear alpha',np.repeat(np.rint(ear*255).astype('uint8')[...,None],3,axis=2)),('Direct alpha',np.repeat(np.rint(direct*255).astype('uint8')[...,None],3,axis=2)),('Jewelry alpha',np.repeat(np.rint(j*255).astype('uint8')[...,None],3,axis=2)),('Fixed binary',np.repeat((fixed*255).astype('uint8')[...,None],3,axis=2))]
        panels(HERE/(label+'-ear-source-8x.png'),maps,box,8)
    report={'scope':'Diagnostic source-coordinate comparison of observed residual, not hardcoded processing windows. RGB alone does not provide unique ear-vs-cheek ground truth.','rows':rows,'sourceSha256':sha(source),'sourceLayersSha256':sha(cached/'source-layers.npz'),'scriptSha256':sha(Path(__file__))}
    (HERE/'ear-stationary-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({r['side']:r['highResidualPixels'] for r in rows}))
if __name__=='__main__':main()
