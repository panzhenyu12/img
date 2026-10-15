"""Factorial causal check: cached v1b source layers, only inverse is replaced.

Baseline replay is checked RGBA-exact against root's v1b for +/-100. New inverse
comes from independent clamped-boundary B. No root/production files are edited.
The original V36 output stays the outside-local-domain base for both variants.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sample(values,xy):
    h,w=values.shape[:2];xx,yy=xy[...,0],xy[...,1];x0,y0=np.floor(xx).astype('int32'),np.floor(yy).astype('int32');tx,ty=xx-x0,yy-y0
    if values.ndim==2:values=values[...,None]
    output=np.zeros((*xy.shape[:2],values.shape[2]),dtype='float64')
    for dx,dy,weight in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]:
        x,y=x0+dx,y0+dy;valid=(x>=0)&(y>=0)&(x<w)&(y<h);output[valid]+=values[y[valid],x[valid]]*weight[valid,None]
    return output

def montage(path,images,labels,crop,scale=1):
    cw,ch=crop[2]-crop[0],crop[3]-crop[1];canvas=Image.new('RGB',(cw*len(images)*scale,ch*scale+28),'#20242a');draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)):
        canvas.paste(im.convert('RGB').crop(crop).resize((cw*scale,ch*scale),Image.Resampling.NEAREST),(cw*i*scale,28));draw.text((cw*i*scale+3,8),label,fill='white')
    canvas.save(path)

def main():
    out=HERE/'composite-geometry-b';out.mkdir(exist_ok=False);cached=HERE/'composite-v1b';data=np.load(cached/'source-layers.npz');direct=data['skinAlpha'];face_color=data['skin'];bg=data['background'];jewelry=data['jewelry'];metal=data['metal'];local=data['local'];fixed=np.asarray(Image.open(cached/'fixed.png'))>127
    sourcepath=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(sourcepath).convert('RGBA'))
    accessory=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');opaque=accessory['opaque']
    weight=np.clip(distance_transform_edt(local)/4,0,1);weight=weight*weight*(3-2*weight)
    verify=json.loads((ROOT/'experiments/face-attachment-boundary-v1-20261015/verification.json').read_text());report={'scope':__doc__,'sourceLayersSha256':sha(cached/'source-layers.npz'),'sourceSha256':sha(sourcepath),'rows':[],'accepted':False}
    for amount in [30,-30,100,-100]:
        material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full'
        if not material.exists():material=ROOT/f'experiments/face-attachment-boundary-v1-20261015/baseline/{amount}'
        layers=material/'layers';state=json.loads((layers/'state.json').read_text());cw,ch=state['width'],state['height'];ox,oy=state['origin'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        physical=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype(float);fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB')).astype(float)/255
        alpha=physical*(1-weight[cs])+direct[cs]*weight[cs];color=fg*(1-weight[cs][...,None])+face_color[cs]*weight[cs][...,None];base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'))
        inversepaths={'v1b':layers/'inverse-source-xy-f32le.bin','geometry-b':ROOT/f'experiments/face-attachment-boundary-v1-20261015/clamp-0.08/{amount}/inverse-source-xy-f32le.bin'};images={};writes={}
        for name,path in inversepaths.items():
            xy=np.fromfile(path,'<f4').reshape(ch,cw,2);ma=sample(alpha,xy)[...,0];mc=sample(color*alpha[...,None],xy);underlying=mc+(1-ma[...,None])*bg[cs];result=jewelry[cs][...,None]*metal[cs]+(1-jewelry[cs][...,None])*underlying
            output=base.copy();transported=sample(weight[cs],xy)[...,0];write=np.maximum(weight[cs],transported);write[fixed[cs]|(jewelry[cs]>=.98)]=0
            patch=output[cs][:,:,:3].astype(float)/255;patch=patch*(1-write[...,None])+result*write[...,None];output[cs][:,:,:3]=np.floor(np.clip(patch,0,1)*255+.5).astype('uint8')
            images[name]=output;writes[name]=write
        replay_exact=None
        if abs(amount)==100:
            replay_exact=bool(np.array_equal(images['v1b'],np.asarray(Image.open(cached/str(amount)/'candidate.png').convert('RGBA'))));assert replay_exact
        dest=out/str(amount);dest.mkdir();Image.fromarray(images['v1b']).save(dest/'baseline.png');Image.fromarray(images['geometry-b']).save(dest/'candidate.png')
        Image.fromarray(np.rint(writes['geometry-b']*255).astype('uint8')).save(dest/'write-weight.png')
        for label,box in [('face',[287,197,730,709]),('left',[320,395,420,550]),('right',[590,395,705,550])]:
            ims=[Image.fromarray(a) for a in [source,base,images['v1b'],images['geometry-b']]];montage(dest/f'{label}-1x.png',ims,['Source','V36','New source/domain','New source + B'],box)
            if label!='face':montage(dest/f'{label}-3x.png',ims,['Source','V36','New source/domain','New source + B'],box,3)
        q=images['geometry-b'];b=images['v1b'];diff=np.any(q!=b,axis=2);geom=next(r for r in verify['rows'] if r['amount']==amount and r['spanFraction']==.08)
        report['rows'].append({'amount':amount,'rootBaselineRgbaReplayExact':replay_exact,'inverseSha256':{k:sha(v) for k,v in inversepaths.items()},
            'minimumRealizedDisplacementRatioOfInverse':geom['minimumRealizedDisplacementRatio'],'widthRatioNote':'Measured full B inverse; composition applies B only in root local source/destination domain, V36 base outside it.',
            'changedPixelsVsNewSourceBaseline':int(diff.sum()),'opaqueMetalChangesVsBaseline':int((diff&opaque).sum()),'opaqueMetalChangesVsSource':int((np.any(q!=source,axis=2)&opaque).sum()),
            'earHairFixedChangesVsBaseline':int((diff&fixed).sum()),'earHairFixedChangesVsSource':int((np.any(q!=source,axis=2)&fixed).sum()),'alphaChangesVsSource':int((q[...,3]!=source[...,3]).sum()),'outputSha256':sha(dest/'candidate.png')})
    report['scriptSha256']=sha(Path(__file__));report['sourceLayersHashStillEqual']=sha(cached/'source-layers.npz')==report['sourceLayersSha256'];(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
