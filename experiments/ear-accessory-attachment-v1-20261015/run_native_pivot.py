"""Attach rigid ear/metal motion to exact native-v2 output as a layer delta.

Face physical alpha, foreground, inverse, and native reconstruction are frozen.
Only source stationary ear/jewelry contribution is removed and transformed
contribution added. The shared underlay is a hypothesis underneath attachments.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,binary_dilation
from attachment import affine_layer,attachment_transforms,bilinear,recompose_motion_delta
from run_pivot import forward,montage,alpha_stats
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=HERE/'native-pivot-v4';out.mkdir(exist_ok=False);rootexp=ROOT/'experiments/joint-skin-matting-v1-20261015';native=rootexp/'native-v2-user';srcp=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(srcp).convert('RGBA'));rgb=source[...,:3]/255.;h,w=source.shape[:2]
    original=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text())['operations'][0]['params'];sem={k:raster_coverage(v,w,h) for k,v in original['semanticMasks'].items() if v['shape']=='raster'}
    old=json.loads((HERE/'outputs-pivot-v3/report.json').read_text());components=old['components'];matte=rootexp/'prompts-v2/user-face-contour/crop-point/alphas-full.npy';direct=np.load(matte)[2];full=np.load(rootexp/'prompts-v2/user-face-contour/full-point/alphas-full.npy')[2]
    accessory=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');J=accessory['alpha'];M=np.load(rootexp/'composite-v1b/source-layers.npz')['metal'];E=np.minimum(direct,sem['ears'])
    known=(direct<.005)&(full<.005)&(J<.005)&(sem['ears']<.005)&(sem['hair']<.005);distance,indices=distance_transform_edt(~known,return_indices=True);BG=rgb[tuple(indices)]
    # Source ear foreground unmix: opaque source texels retain their own RGB.
    # Adjacent observed RGB backs only fractional ear edges. This does not
    # assert any unseen background truth under the attachment.
    EF=np.clip((rgb-(1-E[...,None])*BG)/np.maximum(E[...,None],1e-6),0,1);EF[E>=.98]=rgb[E>=.98];EF[E<.001]=rgb[E<.001]
    Y,X=np.mgrid[:h,:w];halves=[X<w/2,X>=w/2];local=binary_dilation((E>.001)|(J>.001),iterations=6);weight=np.clip(distance_transform_edt(local)/3,0,1);weight=weight**2*(3-2*weight)
    np.savez_compressed(out/'source-attachments.npz',earAlpha=E,earColor=EF,jewelryAlpha=J,jewelryColor=M,background=BG,sourceWeight=weight)
    report={'scope':__doc__,'sourceSha256':sha(srcp),'matteSha256':sha(matte),'nativeReportSha256':sha(native/'report.json'),'components':components,'rows':[],'accepted':False}
    for amount in [100,-100,30,-30]:
        mat=HERE/'native-v2-layers'/str(amount);layers=mat/'layers';st=json.loads((layers/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        inversepath=layers/'inverse-source-xy-f32le.bin';inv=np.fromfile(inversepath,'<f4').reshape(ch,cw,2);A=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw);F=np.asarray(Image.open(layers/'foreground.png').convert('RGB'))/255.;base=np.asarray(Image.open(mat/'candidate.png').convert('RGBA'))
        ma=bilinear(A,inv)[...,0];mf=bilinear(F*A[...,None],inv);U=mf+(1-ma[...,None])*BG[cs]
        source_model=J[cs][...,None]*M[cs]+(1-J[cs][...,None])*(E[cs][...,None]*EF[cs]+(1-E[cs][...,None])*U)
        motions=[]
        for c in components:
            driver=np.array(c['driverSource'])-np.array([ox,oy]);q,res=forward(inv,driver);motions.append({'side':c['side'],'translation':(q-driver).tolist(),'inverseSolveResidualPx':res})
        images={};metrics=[]
        for group in ['fixed','ear-only','pivot-pair']:
            ep=np.zeros((ch,cw,3));ea=np.zeros((ch,cw));jp=np.zeros((ch,cw,3));ja=np.zeros((ch,cw));write=weight[cs].copy();rows=[]
            for c,m,half in zip(components,motions,halves):
                pivot=np.array(c['pivotSource'])-np.array([ox,oy]);hook=np.array(c['hookSource'])-np.array([ox,oy]);delta=np.array(m['translation']) if group!='fixed' else np.zeros(2);em,jm=attachment_transforms(pivot,hook,delta)
                if group=='ear-only':jm=np.column_stack([np.eye(2),np.zeros(2)])
                ecolor,ealpha=affine_layer(EF[cs],(E*half)[cs],em);jcolor,jalpha=affine_layer(M[cs],(J*half)[cs],jm);ep+=ecolor;ea+=ealpha;jp+=jcolor;ja+=jalpha
                _,we=affine_layer(np.ones((ch,cw,3)),(weight*half)[cs],em);_,wj=affine_layer(np.ones((ch,cw,3)),(weight*half)[cs],jm);write=np.maximum.reduce([write,we,wj])
                end=em[:,:2]@hook+em[:,2];jend=jm[:,:2]@hook+jm[:,2];sm,sc,sv=alpha_stats((J*half)[cs]);tm,tc,tv=alpha_stats(jalpha)
                rows.append({'side':c['side'],'earAffine':em.tolist(),'jewelryAffine':jm.tolist(),'rootDisplacementPx':float(np.linalg.norm(em[:,:2]@pivot+em[:,2]-pivot)),'hookGapPx':float(np.linalg.norm(end-jend)),
                    'hookTargetResidualPx':float(np.linalg.norm(end-(hook+delta))),'effectiveHookMotionPx':float(np.linalg.norm(end-hook)),'jewelryAlphaMassRelativeError':float(abs(sm-tm)/sm),
                    'jewelryCentroidErrorPx':float(np.linalg.norm(tc-(sc+jm[:,2]))),'jewelryCovarianceDeltaMax':float(np.abs(tv-sv).max())})
            target=jp+(1-ja[...,None])*(ep+(1-ea[...,None])*U);result=recompose_motion_delta(base[cs][...,:3]/255.,source_model,target)
            write[sem['hair'][cs]>=.5]=0;output=base.copy();patch=base[cs][...,:3]/255.;patch=patch*(1-write[...,None])+result*write[...,None];output[cs][...,:3]=np.floor(np.clip(patch,0,1)*255+.5).astype('uint8')
            d=out/str(amount)/group;d.mkdir(parents=True);Image.fromarray(output).save(d/'candidate.png');Image.fromarray(np.rint(write*255).astype('uint8')).save(d/'write-weight.png')
            images[group]=output;fullwrite=np.zeros((h,w),bool);fullwrite[cs]=write>0
            metrics.append({'group':group,'components':rows,'changedPixelsVsNative':int(np.any(output!=base,axis=2).sum()),'outsideWriteChanges':int((np.any(output!=base,axis=2)&~fullwrite).sum()),
                'alphaChanges':int(np.count_nonzero(output[...,3]!=source[...,3])),'sourceIdentityIfFixed':bool(np.array_equal(output,base)) if group=='fixed' else None,'outputSha256':sha(d/'candidate.png')})
        for n,box in [('face',[287,197,730,709]),('left',[305,280,430,550]),('right',[580,280,715,550])]:
            montage(out/str(amount)/(n+'-1x.png'),[source,base,images['ear-only'],images['pivot-pair']],['Source','Native-v2 fixed','Ear pivot only','Ear pivot + J'],box)
            if n!='face':montage(out/str(amount)/(n+'-2x.png'),[source,base,images['ear-only'],images['pivot-pair']],['Source','Native-v2 fixed','Ear pivot only','Ear pivot + J'],box,2)
        report['rows'].append({'amount':amount,'inverseSha256':sha(inversepath),'nativeAlphaSha256':sha(layers/'source-alpha-f32le.bin'),'nativeForegroundSha256':sha(layers/'foreground.png'),'motions':motions,'groups':metrics});(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'amount':amount,'motions':motions,'fixedExact':metrics[0]['sourceIdentityIfFixed']}),flush=True)
    report['scriptSha256']=sha(Path(__file__));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
