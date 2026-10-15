"""Observed raster/photographic audit; geometric success is not visual approval."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def stats(a):
    y,x=np.mgrid[:a.shape[0],:a.shape[1]];mass=a.sum();c=np.array([(a*x).sum(),(a*y).sum()])/mass;u=x-c[0];v=y-c[1];cov=np.array([[(a*u*u).sum(),(a*u*v).sum()],[(a*u*v).sum(),(a*v*v).sum()]])/mass
    return mass,c,cov

def main():
    out=HERE/'independent-real-audit-low';out.mkdir(exist_ok=False);r=json.loads((HERE/'outputs-pivot-v3-low/report.json').read_text());layers=np.load(HERE/'outputs-pivot-v3-low/source-layers.npz');source=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'));h,w=source.shape[:2];Y,X=np.mgrid[:h,:w];mid=w*.5
    report={'scope':'Single user photograph, deterministic grouped counterfactuals; no population/aesthetic score. Darkening counts are proxies, not labeled cracks.','accepted':False,'rows':[]}
    for row in r['rows']:
        amount=row['amount'];st=json.loads((ROOT/f'experiments/face-attachment-boundary-v1-20261015/baseline/{amount}/layers/state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        images={g:np.asarray(Image.open(HERE/f'outputs-pivot-v3-low/{amount}/{g}/candidate.png').convert('RGBA')) for g in ['fixed','translation-pair','pivot-pair']}
        lum=lambda a:np.dot(a[...,:3],[.2126,.7152,.0722]);fixed_l=lum(images['fixed']);records=[]
        for c in r['components']:
            side=c['side'];half=X<mid if side=='left' else X>=mid;px,py=c['pivotSource'];hx,hy=c['hookSource'];bb=c['semanticEarBounds'];radius=(bb[3]-bb[1])*.20
            region=(X-px)**2+(Y-py)**2<=radius*radius;lobe=(X-hx)**2+(Y-hy)**2<=radius*radius
            for g in ['translation-pair','pivot-pair']:
                metric=next(v for v in row['groups'] if v['group']==g);s=next(v for v in metric['shapes'] if v['side']==side);R=np.array(s['earAffine'])[:,:2];t=np.array(s['earAffine'])[:,2]
                target=np.asarray(Image.open(HERE/f'outputs-pivot-v3-low/{amount}/{g}/ear-target-alpha.png'))/255.;target*=half[cs]
                src=layers['earAlpha'][cs]*half[cs];sm,sc,sv=stats(src);tm,tc,tv=stats(target)
                delta=lum(images[g])-fixed_l
                records.append({'side':side,'group':g,'newRootDarkeningBelowMinus20Pixels':int((region&(fixed_l>50)&(delta<-20)).sum()),'newLobeDarkeningBelowMinus20Pixels':int((lobe&(fixed_l>50)&(delta<-20)).sum()),
                   'earRasterAlphaMassRelativeError':float(abs(tm-sm)/sm),'earRasterCentroidErrorPx':float(np.linalg.norm(tc-(R@sc+t))),
                   'earRasterCovarianceMaxErrorPxSquared':float(np.abs(tv-R@sv@R.T).max()),'rootDisplacementPx':s['rootDisplacementPx'],'hookSeparationPx':s['predictedHookSeparationPx'],'hookMotionRetainedRatio':s['hookMotionRetainedRatio']})
            for label,cx,cy in [('root',px,py),('lobe',hx,hy)]:
                pad=int(np.ceil(radius));box=(int(cx-pad),int(cy-pad),int(cx+pad),int(cy+pad));ww,hh=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(ww*4*3,hh*3+26),'#171b20');draw=ImageDraw.Draw(sheet)
                for i,(name,img) in enumerate([('Source',source),('Fixed',images['fixed']),('Translate',images['translation-pair']),('Pivot',images['pivot-pair'])]):
                    sheet.paste(Image.fromarray(img).convert('RGB').crop(box).resize((ww*3,hh*3),Image.Resampling.NEAREST),(i*ww*3,26));draw.text((i*ww*3+3,8),name,fill='white')
                sheet.save(out/f'{amount}-{side}-{label}-3x.png')
        report['rows'].append({'amount':amount,'components':records})
    report['scriptSha256']=sha(Path(__file__));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
