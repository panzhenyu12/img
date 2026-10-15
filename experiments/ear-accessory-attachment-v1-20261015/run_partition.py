"""Fixed-native-inverse ear/face source support ablation, experimental only.

Four groups vary physical face alpha and foreground color separately. Source
residual is retained; zero deformation has an exactly zero model delta. The
ear's semantic ownership remains a hypothesis, while the joint neural alpha
sets total physical coverage. No user/native output is overwritten.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from attachment import bilinear,partition_joint_alpha
from run_pivot import montage
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=HERE/'native-partition-v5b';out.mkdir(exist_ok=False)
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    source=np.asarray(Image.open(source_path).convert('RGBA'));rgb=source[...,:3]/255.;h,w=source.shape[:2]
    graph=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text())
    masks=graph['operations'][0]['params']['semanticMasks'];sem={k:raster_coverage(v,w,h) for k,v in masks.items() if v['shape']=='raster'}
    layers=np.load(HERE/'native-pivot-v4/source-attachments.npz');BG=layers['background'];J=layers['jewelryAlpha']
    joint=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/user-face-contour/crop-point/alphas-full.npy')[2]
    E,S=partition_joint_alpha(joint,sem['ears'])
    C=np.clip((rgb-(1-joint[...,None])*BG)/np.maximum(joint[...,None],1e-7),0,1)
    C[joint<.001]=rgb[joint<.001]
    # Parser support uncertainty is 2 pixels in its own raster. Work only
    # next to observed ear ownership, excluding independently opaque metal.
    pitch=max(w/masks['ears']['width'],h/masks['ears']['height'])
    near=distance_transform_edt(sem['ears']<.01)
    zone=np.clip(1-near/(2*pitch),0,1)
    zone=zone*zone*(3-2*zone)
    zone[(sem['hair']>=.5)|(J>=.95)]=0
    np.savez_compressed(out/'source-model.npz',ear=E,skin=S,joint=joint,foreground=C,sourceZone=zone)
    report={'scope':__doc__,'accepted':False,'sourceSha256':sha(source_path),
      'jointPartitionMaxError':float(np.abs(E+(1-E)*S-joint).max()),
      'naiveSeparateOwnershipMaxCoverageDeficit':float(np.max(joint-(joint*sem['ears']+(1-joint*sem['ears'])*joint*(1-sem['ears'])))),
      'parserPitchPx':pitch,'rows':[]}
    for amount in [100,-100,30,-30]:
        path=HERE/'native-v2-layers'/str(amount);mat=path/'layers';state=json.loads((mat/'state.json').read_text());ox,oy=state['origin'];cw,ch=state['width'],state['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        A=np.fromfile(mat/'source-alpha-f32le.bin','<f4').reshape(ch,cw);F=np.asarray(Image.open(mat/'foreground.png').convert('RGB'))/255.
        inv=np.fromfile(mat/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2);base=np.asarray(Image.open(path/'candidate.png').convert('RGBA'))
        z=zone[cs];newA=A*(1-z)+S[cs]*z;newF=F*(1-z[...,None])+C[cs]*z[...,None]
        write=np.maximum(z,bilinear(z,inv)[...,0]);write[J[cs]>=.95]=0;write[sem['hair'][cs]>=.5]=0
        def effect(alpha,color,xy):
            target=bilinear(alpha[...,None]*color,xy)-bilinear(alpha,xy)*BG[cs]
            origin=alpha[...,None]*(color-BG[cs])
            return (1-J[cs][...,None])*(1-E[cs][...,None])*(target-origin)
        ref=effect(A,F,inv);zeroY,zeroX=np.mgrid[:ch,:cw];identity=np.stack([zeroX,zeroY],axis=-1)
        groups={'fixed':(A,F),'alpha-only':(newA,F),'color-only':(A,newF),'alpha-and-color':(newA,newF)};images={};metrics=[]
        for name,(a,f) in groups.items():
            delta=effect(a,f,inv)-ref;zero=effect(a,f,identity)-effect(A,F,identity)
            output=base.copy();patch=base[cs][...,:3]/255.+delta*write[...,None]
            output[cs][...,:3]=np.floor(np.clip(patch,0,1)*255+.5).astype('uint8')
            d=out/str(amount)/name;d.mkdir(parents=True);Image.fromarray(output).save(d/'candidate.png');images[name]=output
            changed=np.any(output!=base,axis=2);fullwrite=np.zeros((h,w),bool);fullwrite[cs]=write>0
            metrics.append({'group':name,'fixedExact':bool(np.array_equal(output,base)) if name=='fixed' else None,
               'identityMaxDifference':float(np.max(np.abs(zero))),'outsideWriteChanges':int((changed&~fullwrite).sum()),
               'opaqueMetalChangedPixels':int((changed&(J>=.95)).sum()),'changedPixels':int(changed.sum()),'sha256':sha(d/'candidate.png')})
        for name,box in [('left',[322,396,408,488]),('right',[610,396,693,488]),('face',[287,197,730,709])]:
            montage(out/str(amount)/(name+'-2x.png'),[source,base,images['alpha-only'],images['color-only'],images['alpha-and-color']],
                    ['Source','Native-v2','Alpha only','Color only','Alpha + color'],box,2 if name!='face' else 1)
        report['rows'].append({'amount':amount,'inverseSha256':sha(mat/'inverse-source-xy-f32le.bin'),'groups':metrics})
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['rows'][-1]),flush=True)
    report['scriptSha256']=sha(Path(__file__));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
