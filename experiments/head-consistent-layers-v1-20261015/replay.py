from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image,ImageDraw
from project import project

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/source-skin-color-consistency-v1-20261015'))
from compositor import sample_f32
byte=lambda a:np.floor(np.clip(a,0,255)+.5).astype('uint8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def panel(path,images,names,box):
    x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
    canvas=Image.new('RGB',(w*len(images),h+26),'#20242a');draw=ImageDraw.Draw(canvas)
    for i,(im,name) in enumerate(zip(images,names)):
        canvas.paste(Image.fromarray(im).convert('RGB').crop(box),(i*w,26));draw.text((i*w+2,6),name,fill='white')
    canvas.save(path)

def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    src=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    original=np.asarray(Image.open(src).convert('RGBA'))
    report={'accepted':False,'sourceSha256':sha(src),'scripts':{p.name:sha(p) for p in [Path(__file__),HERE/'project.py']},'rows':[]}
    for amount in [30,-30]:
        d=out/str(amount);d.mkdir();native=ROOT/f'experiments/whole-person-alpha-override-v1-20261015/outputs/raw-{amount}';trace=native/'debug'
        ox,oy,w,h=json.loads((trace/'layers.json').read_text())['crop'];roi=np.s_[oy:oy+h,ox:ox+w]
        rgb=lambda name:np.asarray(Image.open(trace/name).convert('RGB')).astype('float32')
        mask=lambda name:np.asarray(Image.open(trace/name).convert('L'))>0
        c,f,b=map(rgb,['crop-source.png','foreground.png','reconstructed-background.png'])
        a=np.fromfile(trace/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
        target=np.fromfile(trace/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
        xy=np.fromfile(trace/'source-xy-f32le.bin',dtype='<f4').reshape(h,w,2)-np.array([ox+.5,oy+.5],dtype='float32')
        changed=mask('changed.png');frozen=mask('frozen.png');write=changed&~frozen
        native_image=np.asarray(Image.open(native/'candidate.png').convert('RGBA'))
        fnew,bnew=project(c,a,f,b)
        reconstructed=byte(a[...,None]*fnew+(1-a[...,None])*bnew)
        assert np.array_equal(reconstructed,byte(c))
        sampled,newtarget=sample_f32(fnew,a,xy)
        assert np.array_equal(newtarget[write],target[write])
        rendered=byte(sampled+(1-target[...,None])*bnew)
        result=native_image.copy();patch=result[roi][...,:3].copy();patch[write]=rendered[write];result[roi][...,:3]=patch
        diff=np.any(result!=native_image,-1);allowed=np.zeros(diff.shape,bool);allowed[roi]=write
        assert not (diff&~allowed).any();assert np.array_equal(result[...,3],original[...,3])
        Image.fromarray(result).save(d/'candidate.png')
        np.savez_compressed(d/'projected-layers.npz',foreground=fnew,background=bnew,alpha=a,targetAlpha=target)
        low=write&(np.maximum(a,target)<.01)
        old_delta=np.abs(native_image[roi][...,:3].astype(int)-byte(c).astype(int)).max(-1)
        new_delta=np.abs(result[roi][...,:3].astype(int)-byte(c).astype(int)).max(-1)
        # Since both foreground colors and B are bounded and C=aF+(1-a)B,
        # the maximum per-channel change is bounded by 255*(a+target)+rounding.
        assert not np.any(new_delta[low]>255*(a[low]+target[low])+1)
        report['rows'].append({'amount':amount,'nativeSha256':sha(native/'candidate.png'),
            'traceSha256':{p.name:sha(p) for p in trace.iterdir() if p.is_file()},
            'sourceFitMaxByte':int(np.abs(reconstructed.astype(int)-byte(c).astype(int)).max()),
            'lowCoverageNativeMaxByte':int(old_delta[low].max()),'lowCoverageCandidateMaxByte':int(new_delta[low].max()),
            'lowCoverageNativeOver8':int((old_delta[low]>8).sum()),'lowCoverageCandidateOver8':int((new_delta[low]>8).sum()),
            'changedFromNative':int(diff.sum()),'outputSha256':sha(d/'candidate.png')})
        for name,box in [('face',[287,197,730,709]),('left',[305,285,430,565]),('right',[590,285,715,565]),('hair',[300,0,715,335])]:
            panel(d/f'{name}-1x.png',[original,native_image,result],['Source','Raw ZIM + old F/B','Consistent bounded F/B'],box)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k!='traceSha256'} for r in report['rows']]},indent=2))

if __name__=='__main__':main()
