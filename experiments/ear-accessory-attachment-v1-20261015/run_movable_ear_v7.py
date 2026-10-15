"""Neural ear/lobe material with a native-aware baseline motion model.

Old semantic ear texels were source stationary; the formerly missed lobe was
sampled through native face inverse. Remove that effective old contribution
before adding the rigid pivot ear and attached jewelry, rather than counting
the missed lobe motion twice. Same native face inverse for all groups.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,binary_dilation
from attachment import bilinear,affine_layer,attachment_transforms,partition_joint_alpha
from run_pivot import forward,montage,alpha_stats
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=HERE/'movable-ear-v7';out.mkdir(exist_ok=False)
    sp=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(sp).convert('RGBA'));rgb=source[...,:3]/255.;h,w=source.shape[:2];Y,X=np.mgrid[:h,:w]
    params=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text())['operations'][0]['params'];sem={k:raster_coverage(v,w,h) for k,v in params['semanticMasks'].items() if v['shape']=='raster'}
    parts=[np.load(HERE/f'ear-matte-v6/{side}-point/alphas-full.npy')[2] for side in ['left','right']];rawE=np.maximum.reduce(parts)
    D=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/user-face-contour/crop-point/alphas-full.npy')[2].astype(float);E=np.minimum(rawE,D);S=np.divide(D-E,1-E,out=np.zeros_like(D),where=(1-E)>1e-9)
    oldmat=np.load(HERE/'native-pivot-v4/source-attachments.npz');J=oldmat['jewelryAlpha'];M=oldmat['jewelryColor'];BG=oldmat['background']
    under=np.clip((rgb-J[...,None]*M)/np.maximum(1-J[...,None],1e-7),0,1);F=np.clip((under-(1-D[...,None])*BG)/np.maximum(D[...,None],1e-7),0,1);F[D>=.98]=under[D>=.98];F[D<.001]=rgb[D<.001]
    old=json.loads((HERE/'outputs-pivot-v3/report.json').read_text());components=[];halves=[X<w/2,X>=w/2]
    for c,part,half in zip(old['components'],parts,halves):
        hx,hy=c['hookSource'];height=c['semanticEarBounds'][3]-c['semanticEarBounds'][1]
        # The hanging component's source hook is metal, not ear. Identify
        # observed lobe skin nearest to that hook, then real adjacent skin on
        # the inner (face) side of the neural ear boundary.
        lobe=(part>=.95)&(D>=.95)&(J<.05)&half
        dist=np.where(lobe,(X-hx)**2+(Y-hy)**2,np.inf);ly,lx=np.unravel_index(np.argmin(dist),dist.shape)
        face=(D>=.98)&(rawE<.02)&(J<.01)&(sem['hair']<.01)&half
        face &= (X>lx) if c['side']=='left' else (X<lx)
        radius=.2*height;face &= (X-lx)**2+(Y-ly)**2<radius*radius
        if not face.any():raise RuntimeError('no directly observed ear-face contact skin')
        dist=np.where(face,(X-lx)**2+4*(Y-ly)**2,np.inf);fy,fx=np.unravel_index(np.argmin(dist),dist.shape)
        components.append({**c,'lobeSource':[float(lx),float(ly)],'driverSource':[float(fx),float(fy)],'earAlphaAtLobe':float(part[ly,lx]),'directAlphaAtDriver':float(D[fy,fx]),'earAlphaAtDriver':float(rawE[fy,fx])})
    region=binary_dilation((E>.001)|(J>.001),iterations=6);weight=np.clip(distance_transform_edt(region)/3,0,1);weight=weight*weight*(3-2*weight)
    np.savez_compressed(out/'source-materials.npz',earAlpha=E,skinAlpha=S,jewelryAlpha=J,foreground=F,metal=M,background=BG,sourceWeight=weight)
    report={'scope':__doc__,'sourceSha256':sha(sp),'components':components,'accepted':False,'rows':[]}
    for amount in [100,-100]:
        path=HERE/'native-v2-layers'/str(amount);layers=path/'layers';st=json.loads((layers/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];origin=np.array([ox,oy]);cs=np.s_[oy:oy+ch,ox:ox+cw];inv=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);fixed=np.fromfile(layers/'frozen-u8.bin','uint8').reshape(ch,cw)>0;base=np.asarray(Image.open(path/'candidate.png').convert('RGBA'))
        # Source domain and baseline target domain are shared by all variants.
        ua=bilinear(S[cs],inv)[...,0];up=bilinear(S[cs][...,None]*F[cs],inv);U=up+(1-ua[...,None])*BG[cs]
        oldea=bilinear(E[cs],inv)[...,0];oldep=bilinear(E[cs][...,None]*F[cs],inv);oldea[fixed]=E[cs][fixed];oldep[fixed]=(E[cs][...,None]*F[cs])[fixed]
        oldmodel=J[cs][...,None]*M[cs]+(1-J[cs][...,None])*(oldep+(1-oldea[...,None])*U)
        images={};metrics=[]
        for group in ['native-fixed','pivot-ear','pivot-pair']:
            ep=np.zeros((ch,cw,3));ea=np.zeros((ch,cw));jp=np.zeros_like(ep);ja=np.zeros_like(ea);write=weight[cs].copy();motions=[]
            for c,half in zip(components,halves):
                driver=np.array(c['driverSource'])-origin;target,res=forward(inv,driver);delta=target-driver;lobe=np.array(c['lobeSource'])-origin;pivot=np.array(c['pivotSource'])-origin;em,jm=attachment_transforms(pivot,lobe,delta)
                if group=='pivot-ear':jm=np.column_stack([np.eye(2),np.zeros(2)])
                ecolor,ealpha=affine_layer(F[cs],(E*half)[cs],em);jcolor,jalpha=affine_layer(M[cs],(J*half)[cs],jm);ep+=ecolor;ea+=ealpha;jp+=jcolor;ja+=jalpha
                _,ew=affine_layer(np.ones_like(F[cs]),(weight*half)[cs],em);_,jw=affine_layer(np.ones_like(F[cs]),(weight*half)[cs],jm);write=np.maximum.reduce([write,ew,jw])
                lend=em[:,:2]@lobe+em[:,2];hook=np.array(c['hookSource'])-origin;jend=jm[:,:2]@hook+jm[:,2];sm,sc,sv=alpha_stats((J*half)[cs]);tm,tc,tv=alpha_stats(jalpha)
                motions.append({'side':c['side'],'driverMotion':delta.tolist(),'driverInverseSolveResidualPx':res,'earAffine':em.tolist(),'jewelryAffine':jm.tolist(),
                  'rootDisplacementPx':float(np.linalg.norm(em[:,:2]@pivot+em[:,2]-pivot)),
                  'lobeMotionPx':float(np.linalg.norm(lend-lobe)),'actualAttachmentVectorErrorPx':float(np.linalg.norm((jend-lend)-(hook-lobe))),
                  'earLobeMaterialSupport':c['earAlphaAtLobe'],'jewelryMassRelativeError':float(abs(tm-sm)/sm),'jewelryCentroidErrorPx':float(np.linalg.norm(tc-(sc+jm[:,2]))),'jewelryCovarianceDeltaMax':float(np.abs(tv-sv).max())})
            model=jp+(1-ja[...,None])*(ep+(1-ea[...,None])*U)
            delta=model-oldmodel
            if group=='native-fixed':delta[:]=0
            write[sem['hair'][cs]>=.5]=0;output=base.copy();output[cs][...,:3]=np.floor(np.clip(base[cs][...,:3]/255.+write[...,None]*delta,0,1)*255+.5).astype('uint8')
            d=out/str(amount)/group;d.mkdir(parents=True);Image.fromarray(output).save(d/'candidate.png');images[group]=output
            fullwrite=np.zeros((h,w),bool);fullwrite[cs]=write>0;changed=np.any(output!=base,axis=2)
            metrics.append({'group':group,'motions':motions,'outsideWriteChanges':int((changed&~fullwrite).sum()),'alphaChannelChanges':int(np.count_nonzero(output[...,3]!=source[...,3])),'changedPixelsVsNative':int(changed.sum()),'outputSha256':sha(d/'candidate.png')})
        for label,box in [('face',[287,197,730,709]),('left',[305,280,430,550]),('right',[580,280,715,550])]:
            montage(out/str(amount)/(label+'-1x.png'),[source,base,images['pivot-ear'],images['pivot-pair']],['Source','Native v2','Real lobe pivot','Real lobe + J'],box)
            if label!='face':montage(out/str(amount)/(label+'-2x.png'),[source,base,images['pivot-ear'],images['pivot-pair']],['Source','Native v2','Real lobe pivot','Real lobe + J'],box,2)
        report['rows'].append({'amount':amount,'inverseSha256':sha(layers/'inverse-source-xy-f32le.bin'),'faceInverseUnmodified':True,'groups':metrics});(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'amount':amount,'motions':metrics[-1]['motions']}),flush=True)
    report['scriptSha256']=sha(Path(__file__));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
