"""Freeze v2 source layers; replace only hidden scene estimate using cached LaMa.

Source-clear and transported-edge writes share the same availability contract.
No generation changes the transported foreground, source jewelry alpha, or warp.
"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
from scipy.ndimage import label,binary_dilation

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha,mask_save
from compose_real import sample,montage
sys.path.insert(0,str(ROOT/'services/background-worker'))
from worker import LamaWorker


def main():
    out=HERE/'neural-backing-v3';out.mkdir(exist_ok=False)
    frozen=HERE/'outputs-v2/combined-point';data=np.load(frozen/'source-layers.npz')
    raw=json.loads((ROOT/'experiments/zim-accessory-combined-20261015/outputs/report.json').read_text())
    source_path=ROOT/raw['source'];graph_path=ROOT/raw['graph']
    source=np.asarray(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2]
    params=json.loads(graph_path.read_text())['operations'][0]['params']
    masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    alpha,metal,skin,pure,trusted,opaque=[data[k] for k in ['alpha','metal','skin','pure','trusted','opaque']]
    support=pure|trusted;known=np.asarray(Image.open(frozen/'source-scene-donor.png'))>0
    assert not (support&known).any()
    features=np.maximum.reduce([masks[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']])
    yy,xx=np.indices((h,w));r=params['region']
    roi=((xx+.5)/w>=r['x'])&((xx+.5)/w<=r['x']+r['width'])&((yy+.5)/h>=r['y'])&((yy+.5)/h<=r['y']+r['height'])
    cases=[v for v in raw['rows'] if v['directory'].endswith('combined-point')]
    worker=LamaWorker(threads=3)
    report={'scope':'Frozen v2 source/alpha/geometry; hidden scene context only; no production changes or aesthetic acceptance.',
            'scriptSha256':sha(Path(__file__)),'sourceSha256':sha(source_path),'graphSha256':sha(graph_path),'sourceLayersSha256':sha(frozen/'source-layers.npz'),
            'sourceKnownSceneMaskSha256':sha(frozen/'source-scene-donor.png'),'rows':[]}
    for size in [128,256]:
        d=out/str(size);d.mkdir()
        layer=source[...,:3].astype('float64')/255;available=np.zeros((h,w),bool);tiles=[]
        for case in cases:
            box=case['prompt']['box'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            x0=int(np.clip(round(cx-size/2),0,w-size));y0=int(np.clip(round(cy-size/2),0,h-size));x1=x0+size;y1=y0+size
            patch=source[y0:y1,x0:x1].copy();missing=~known[y0:y1,x0:x1]
            image,meta=worker.inpaint(Image.fromarray(patch),missing.astype('uint8')*255,max_side=max(256,size),context_mode='full',mask_dilation=0,seam_blend='none')
            tile_name=f"component-{case['prompt']['component']}";image.save(d/(tile_name+'-scene.png'))
            patch[missing,:3]=0;Image.fromarray(patch).save(d/(tile_name+'-conditioning.png'));mask_save(d/(tile_name+'-unknown.png'),missing)
            tile_support=support[y0:y1,x0:x1]
            layer[y0:y1,x0:x1][tile_support]=np.asarray(image)[...,:3][tile_support]/255
            available[y0:y1,x0:x1]=True
            tiles.append({'component':case['prompt']['component'],'box':[x0,y0,x1,y1],'meta':meta})
        assert not (support&~available).any()
        Image.fromarray(np.floor(np.clip(layer,0,1)*255+.5).astype('uint8')).save(d/'background-layer.png')
        for amount in [100,-100]:
            material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';layers=material/'layers'
            st=json.loads((layers/'state.json').read_text());cw,ch=st['width'],st['height'];ox,oy=st['origin'];cs=np.s_[oy:oy+ch,ox:ox+cw]
            xy=np.fromfile(layers/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2)
            under=np.fromfile(layers/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype('float64');under[support[cs]]=1
            fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype='float64')/255;fg[support[cs]]=skin[cs][support[cs]]
            new_alpha=sample(under,xy)[...,0];new_rgb=sample(fg*under[...,None],xy)
            transported=sample(support[cs].astype('float64'),xy)[...,0]>1e-6
            yy_,xx_=np.indices((ch,cw));moved=np.hypot(xy[...,0]-xx_,xy[...,1]-yy_)>1e-4
            allowed=roi[cs]&(masks['ears'][cs]<.05)&(masks['hair'][cs]<.05)&(features[cs]<.05)&~opaque[cs]&moved
            added=(new_alpha>under+.001)&(masks['accessories'][cs]>=.05)&(alpha[cs]<=.05)&allowed
            labels,_=label(added,structure=np.ones((3,3)));touch=binary_dilation(support[cs],structure=np.ones((3,3)));ids=np.unique(labels[touch&added]);ids=ids[ids>0]
            connected=np.isin(labels,ids) if len(ids) else np.zeros_like(added);unknown=added&~connected
            writes=((support[cs]|transported|connected)&allowed)&~unknown
            assert not (writes&support[cs]&~available[cs]).any()
            fixed_alpha=np.where(masks['accessories'][cs]>=.05,alpha[cs],0)
            bg=np.asarray(Image.open(layers/'background.png').convert('RGB'),dtype='float64')/255;bg[support[cs]]=layer[cs][support[cs]]
            beneath=new_rgb+(1-new_alpha[...,None])*bg
            rendered=fixed_alpha[...,None]*metal[cs]+(1-fixed_alpha[...,None])*beneath
            base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'));output=base.copy();p=output[cs][:,:,:3].copy();p[writes]=np.floor(np.clip(rendered[writes],0,1)*255+.5).astype('uint8');output[cs][:,:,:3]=p
            full_write=np.zeros((h,w),bool);full_write[cs]=writes
            assert np.array_equal(output[~full_write],base[~full_write]) and np.array_equal(output[opaque],source[opaque])
            dest=d/str(amount);dest.mkdir();Image.fromarray(output).save(dest/'candidate.png')
            for n,m in [('allowed-write',writes),('connected-new-coverage',connected),('unknown-new-coverage',unknown),('newly-exposed-source-support',support[cs]&(new_alpha<.999))]:mask_save(dest/(n+'.png'),m)
            montage(dest/'face-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','V36','Neural backing hypothesis'],[287,197,730,709])
            for case in cases:
                b=case['prompt']['box'];pad=12;c=[b[0]-pad,b[1]-pad,b[2]+pad,b[3]+pad*4]
                montage(dest/f"component-{case['prompt']['component']}-1x.png",[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','V36','Candidate'],c)
            row={'size':size,'amount':amount,'tiles':tiles,'sourceSupportPixels':int(support.sum()),'writePixels':int(writes.sum()),'changedPixels':int(np.any(output!=base,axis=2).sum()),
                'unknownNewCoveragePixels':int(unknown.sum()),'opaqueMetalChanged':0,'outsideWritesChanged':0,'alphaRefined':False,
                'inverseSha256':sha(layers/'inverse-source-xy-f32le.bin'),'outputSha256':sha(dest/'candidate.png'),'directory':str(dest.relative_to(HERE))}
            report['rows'].append(row);print(json.dumps({'size':size,'amount':amount,'changedPixels':row['changedPixels']}),flush=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
