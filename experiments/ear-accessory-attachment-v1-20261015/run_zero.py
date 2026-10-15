"""Three-way isolated ear/accessory rigid attachment experiment.

Experimental unit is one original photo, repeated editing strengths. Shared
source layers and frozen V36 face inverse across groups. No independence claim
for pixels or directions; geometric invariants are not an aesthetic score.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label,distance_transform_edt,binary_dilation
from attachment import bilinear,translate_layer,render_attachments,recompose_motion_delta
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def point_sample(a,p):
    # Same actual source-forward root solve in every group, never a slider-
    # dependent hand-picked point. Coordinates are local pixel-index samples.
    p=np.asarray(p,float);xy=p.reshape(1,1,2);return bilinear(a,xy)[0,0]

def forward(inv,p):
    p=np.asarray(p,float);q=p.copy()
    for _ in range(30):q+=(p-point_sample(inv,q))*.8
    return q,float(np.linalg.norm(p-point_sample(inv,q)))

def montage(path,ims,names,box,scale=1):
    ww,hh=box[2]-box[0],box[3]-box[1];out=Image.new('RGB',(ww*len(ims)*scale,hh*scale+28),'#171b20');d=ImageDraw.Draw(out)
    for i,(a,n) in enumerate(zip(ims,names)):
        if isinstance(a,np.ndarray):a=Image.fromarray(a)
        out.paste(a.convert('RGB').crop(box).resize((ww*scale,hh*scale),Image.Resampling.NEAREST),(i*ww*scale,28));d.text((i*ww*scale+4,8),n,fill='white')
    out.save(path)

def alpha_stats(a):
    y,x=np.mgrid[:a.shape[0],:a.shape[1]];mass=a.sum();c=np.array([(a*x).sum(),(a*y).sum()])/mass;dx=x-c[0];dy=y-c[1];cov=np.array([[(a*dx*dx).sum(),(a*dx*dy).sum()],[(a*dx*dy).sum(),(a*dy*dy).sum()]])/mass
    return float(mass),c,cov

def source_components(ears,jewelry,direct,skin,lm):
    lab,n=label(ears>=.5);mid=float(lm[:,0].mean());h,w=ears.shape;fw=float(np.ptp(lm[:,0]));pitch=w/512
    if n<2:raise RuntimeError('abstain: both source ear components not observed')
    components=sorted(range(1,n+1),key=lambda i:int((lab==i).sum()),reverse=True)[:2];components.sort(key=lambda i:float(np.where(lab==i)[1].mean()))
    rows=[]
    # Pair each observed accessory component with the nearest ear. It must have
    # a coherent upper hanging part; whole connected source J is transported.
    jlab,jn=label(jewelry>=.5);jids=sorted(range(1,jn+1),key=lambda i:int((jlab==i).sum()),reverse=True)[:2]
    Y,X=np.mgrid[:h,:w]
    for ei in components:
        ec=lab==ei;ys,xs=np.where(ec);center=np.array([xs.mean(),ys.mean()]);side='left' if center[0]<mid else 'right';half=(X<mid) if side=='left' else (X>=mid)
        edistance=distance_transform_edt(~ec)
        valid_j=[i for i in jids if ((np.where(jlab==i)[1].mean()<mid)==(side=='left'))]
        if len(valid_j)!=1:raise RuntimeError('abstain: ambiguous ear/accessory pairing')
        ji=valid_j[0];jc=jlab==ji;jy,jx=np.where(jc);jy0,jy1=jy.min(),jy.max()
        upper=jc&(Y<=jy0+max(2,(jy1-jy0)*.12));scores=np.where(upper,edistance,np.inf);y,x=np.unravel_index(np.argmin(scores),scores.shape);hook=np.array([float(x),float(y)])
        # Driver is nearest independently source-supported skin point to the
        # observed hanging anchor. Fixed V36 may make this motion very small;
        # that is reported instead of silently amplifying it.
        eligible=half&(direct>=.995)&(skin>=.8)&(ears<.005)&(jewelry<.005)
        distance,inds=distance_transform_edt(~eligible,return_indices=True);sy,sx=inds[:,y,x];driver=np.array([float(sx),float(sy)])
        if np.linalg.norm(driver-hook)>fw*.12:raise RuntimeError('abstain: no nearby source skin driver')
        earalpha=np.minimum(direct,ears)*half;jalpha=jewelry*half
        rows.append({'side':side,'earComponent':int(ei),'jewelryComponent':int(ji),'hookSource':hook.tolist(),'driverSource':driver.tolist(),'hookToDriverDistance':float(np.linalg.norm(driver-hook)),
            'earAlpha':earalpha,'jewelryAlpha':jalpha,'semanticEarBounds':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]})
    return rows

def main():
    out=HERE/'outputs-zero-check';out.mkdir(exist_ok=False)
    sourcepath=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(sourcepath).convert('RGBA'));rgb=source[...,:3]/255.;h,w=source.shape[:2]
    gp=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json';p=json.loads(gp.read_text())['operations'][0]['params'];m={k:raster_coverage(v,w,h) for k,v in p['semanticMasks'].items() if v['shape']=='raster'}
    lm=np.array([[v['x']*w,v['y']*h] for v in p['landmarks']]);fw=float(np.ptp(lm[:,0]));fh=float(np.ptp(lm[:,1]));cached=ROOT/'experiments/joint-skin-matting-v1-20261015/composite-v1b/source-layers.npz';l=np.load(cached)
    direct=l['skinAlpha'];F=l['skin'];J=l['jewelry'];M=l['metal'];BG=l['background']
    components=source_components(m['ears'],J,direct,m['skin'],lm);E=sum(c['earAlpha'] for c in components)
    # Algebraic source partition: direct == E + (1-E)*S. Ear allocation uses
    # observed semantic alpha capped by direct matte, no accessory complement.
    S=np.divide(np.maximum(direct-E,0),1-E,out=np.zeros_like(E),where=E<1-1e-8)
    source_union=E+(1-E)*S
    local=binary_dilation((m['ears']>.005)|(m['accessories']>.005),iterations=max(4,round(fw*.026)))&(m['hair']<.5)
    weight=np.clip(distance_transform_edt(local)/4,0,1);weight=weight**2*(3-2*weight)
    source_recon=J[...,None]*M+(1-J[...,None])*(E[...,None]*F+(1-E[...,None])*(S[...,None]*F+(1-S[...,None])*BG))
    np.savez_compressed(out/'source-layers.npz',earAlpha=E,faceAlpha=S,direct=direct,faceColor=F,metal=M,jewelry=J,background=BG,weight=weight)
    Image.fromarray(np.rint(np.clip(source_recon,0,1)*255).astype('uint8')).save(out/'source-reconstruction.png')
    report={'scope':__doc__,'sourceSha256':sha(sourcepath),'graphSha256':sha(gp),'cachedSourceLayersSha256':sha(cached),'sourceUnionMaxDifference':float(np.abs(source_union-direct).max()),
       'sourceReconstructionMaxByte':float(np.abs(source_recon-rgb)[local].max()*255),'sourceReconstructionMeanByte':float(np.abs(source_recon-rgb)[local].mean()*255),
       'compositionAssumption':'Source-anchored target-minus-source model delta preserves zero motion, but residual remains stationary and is not recovered hidden texture.',
       'sourceLayerAssumption':'Ear allocation E=min(direct,earSemantic); face S=(direct-E)/(1-E). This is a material partition hypothesis, not inferred hidden ear/face geometry.',
       'components':[{k:v for k,v in c.items() if 'Alpha' not in k} for c in components],'rows':[],'accepted':False}
    # Zero is source identity by explicit identity operation; no post-edit model.
    for amount in [0]:
        material=ROOT/'experiments/face-material-fill-v1-20261015/user-face-contour-100-full'
        if not material.exists():material=ROOT/f'experiments/face-attachment-boundary-v1-20261015/baseline/{amount}'
        layers=material/'layers';st=json.loads((layers/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        inv=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);physical=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw);fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB'))/255.;base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'))
        Y,X=np.mgrid[:ch,:cw];inv=np.stack([X,Y],axis=-1).astype(float);base=source.copy()
        alpha=physical*(1-weight[cs])+S[cs]*weight[cs];color=fg*(1-weight[cs][...,None])+F[cs]*weight[cs][...,None];fa=bilinear(alpha,inv)[...,0];fc=bilinear(color*alpha[...,None],inv);face_base=fc+(1-fa[...,None])*BG[cs]
        motions=[]
        for c in components:
            driver=np.array(c['driverSource'])-np.array([ox,oy]);q,res=forward(inv,driver);delta=q-driver
            motions.append({'side':c['side'],'translation':delta.tolist(),'lengthPx':float(np.linalg.norm(delta)),'driverInverseSolveResidualPx':res,
                            'hookDestination':(np.array(c['hookSource'])+delta).tolist()})
        outputs={};group_metrics=[]
        for group in ['fixed','ear-only','ear-and-jewelry']:
            ea_total=np.zeros((ch,cw));ep_total=np.zeros((ch,cw,3));ja_total=np.zeros((ch,cw));jp_total=np.zeros((ch,cw,3));write=np.maximum(weight[cs],bilinear(weight[cs],inv)[...,0]);shapes=[]
            for c,motion in zip(components,motions):
                t=np.array(motion['translation']) if group!='fixed' else np.zeros(2);jt=t if group=='ear-and-jewelry' else np.zeros(2)
                ep,ea=translate_layer(F[cs],c['earAlpha'][cs],t);jp,ja=translate_layer(M[cs],c['jewelryAlpha'][cs],jt)
                ep_total+=ep;ea_total+=ea;jp_total+=jp;ja_total+=ja
                # Expanded target support is admitted using the same source
                # domain transport, then original hair is kept as known front.
                ww=weight[cs]*((c['earAlpha'][cs]+c['jewelryAlpha'][cs])>0)
                _,ew=translate_layer(np.ones((ch,cw,3)),ww,t);_,jw=translate_layer(np.ones((ch,cw,3)),ww,jt);write=np.maximum.reduce([write,ew,jw])
                am,center,cov=alpha_stats(c['jewelryAlpha'][cs]);bm,bc,bv=alpha_stats(ja);hook=np.array(c['hookSource']);ear_hook=hook+t;j_hook=hook+jt
                shapes.append({'side':c['side'],'earTranslation':t.tolist(),'jewelryTranslation':jt.tolist(),'predictedHookSeparationPx':float(np.linalg.norm(ear_hook-j_hook)),
                     'jewelryAlphaMassRelativeError':float(abs(bm-am)/am),'jewelryCentroidShiftErrorPx':float(np.linalg.norm((bc-center)-jt)),
                     'jewelryCovarianceDeltaPixelsSquared':(bv-cov).tolist(),'predictedRigidPairwiseDistanceError':0.0})
            target_model=jp_total+(1-ja_total[...,None])*(ep_total+(1-ea_total[...,None])*face_base)
            source_face=color*alpha[...,None]+(1-alpha[...,None])*BG[cs]
            source_model=J[cs][...,None]*M[cs]+(1-J[cs][...,None])*(E[cs][...,None]*F[cs]+(1-E[cs][...,None])*source_face)
            result=recompose_motion_delta(rgb[cs],source_model,target_model)
            write[m['hair'][cs]>=.5]=0
            output=base.copy();patch=base[cs][...,:3]/255.;patch=patch*(1-write[...,None])+result*write[...,None];output[cs][...,:3]=np.floor(np.clip(patch,0,1)*255+.5).astype('uint8')
            d=out/str(amount)/group;d.mkdir(parents=True);Image.fromarray(output).save(d/'candidate.png');Image.fromarray(np.rint(write*255).astype('uint8')).save(d/'write-weight.png');Image.fromarray(np.rint(np.clip(ea_total,0,1)*255).astype('uint8')).save(d/'ear-target-alpha.png');Image.fromarray(np.rint(np.clip(ja_total,0,1)*255).astype('uint8')).save(d/'jewelry-target-alpha.png')
            outputs[group]=output;fullwrite=np.zeros((h,w),bool);fullwrite[cs]=write>0
            group_metrics.append({'group':group,'shapes':shapes,'alphaChangesVsSource':int(np.count_nonzero(output[...,3]!=source[...,3])),
                'outsideWriteChangesVsBase':int((np.any(output!=base,axis=2)&~fullwrite).sum()),'hairCoreChangesVsSource':int((np.any(output!=source,axis=2)&(m['hair']>=.95)).sum()),
                'outputSha256':sha(d/'candidate.png')})
        for name,box in [('face',[287,197,730,709]),('left',[305,280,430,550]),('right',[580,280,715,550])]:
            ims=[source,base]+[outputs[g] for g in outputs];names=['Source','V36','Attachments fixed','Ear only','Ear + jewelry'];montage(out/str(amount)/(name+'-1x.png'),ims,names,box)
            if name!='face':montage(out/str(amount)/(name+'-2x.png'),ims,names,box,2)
        report['rows'].append({'amount':amount,'inverseSha256':sha(layers/'inverse-source-xy-f32le.bin'),'faceGeometryChanged':False,'zeroIdentityOutputChangedPixels':int(np.count_nonzero(np.any(outputs['ear-and-jewelry']!=source,axis=2))),'faceWidthDisplacementRatio':1.0,'faceWidthDisplacementNote':'Same source-to-target face inverse; this is the field contract, not a post-composition silhouette measurement.','motions':motions,'groups':group_metrics})
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'amount':amount,'motions':motions}),flush=True)
    report['scriptSha256']=sha(Path(__file__));report['sourceLayersSha256']=sha(out/'source-layers.npz');report['cachedSourceLayersStillEqual']=sha(cached)==report['cachedSourceLayersSha256'];(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
