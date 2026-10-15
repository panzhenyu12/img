"""Black-box output regression for existing opacity/protection/RGBA contracts.

Uses the old-core snapshot and candidate native public JSON graph interface.
This is a compositing permissions probe, not an aesthetic score or photoreal test.
"""
from pathlib import Path
import json, math, hashlib, subprocess
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=HERE/'public-options-v2';out.mkdir(exist_ok=False)
    binaries={'old':ROOT/'.run/face-source-residual-v39-independent-target/release/image-render','candidate':ROOT/'.run/face-source-residual-v39-target/release/image-render'}
    for p in binaries.values():assert p.is_file(),p
    w,h=160,320;y,x=np.mgrid[:h,:w]
    a=np.clip(.5+(1-np.hypot((x+.5-80)/38,(y+.5-112)/80))*38/3,0,1)
    n=((x>=69)&(x<91)&(y>=170)).astype(float);skin=np.maximum(a,n)
    foreground=np.array([220,155,100]);background=np.stack([20+x%16,45+y%16,np.full_like(x,130)],axis=-1)
    rgb=np.floor(skin[...,None]*foreground+(1-skin[...,None])*background+.5).astype('uint8')
    mask=lambda data:{'shape':'raster','width':w,'height':h,'data':np.asarray(data,dtype='uint8').ravel().tolist()}
    face=mask(np.floor(a*255+.5));joint=mask(np.floor(skin*255+.5));neck=mask(255*((n>0)&(a<.5)))
    outline={'shape':'polygon','points':[{'x':(80+38*math.cos(i*math.tau/64))/w,'y':(112+80*math.sin(i*math.tau/64))/h} for i in range(64)]}
    base={'region':{'x':0,'y':0,'width':1,'height':1},'semanticMasks':{'face':face,'faceOutline':outline,'faceContour':outline,'faceContourMask':face,'skin':joint,'neck':neck},'subjectMask':joint,'adjustments':{'faceWidth':100}}
    variants={f'{mode}-{opacity}':{'blendMode':mode,'opacity':opacity} for mode in ['normal','multiply'] for opacity in [0,50,100]}
    variants.update({f'{mode}-protect-{value}':{'blendMode':mode,'protectMask':mask(np.full((h,w),value))} for mode in ['normal','multiply'] for value in [128,254,255]})
    outputs={};rows=[]
    for kind,alpha in [('opaque',np.full((h,w),255,dtype='uint8')),('rgba',np.floor(skin*255+.5).astype('uint8'))]:
        source=np.dstack([rgb,alpha]);sp=out/f'{kind}-source.png';Image.fromarray(source).save(sp)
        for name,extra in variants.items():
            graph=out/f'{kind}-{name}.json';graph.write_text(json.dumps({'operations':[{'type':'face.adjust','params':{**base,**extra}}]}))
            for version,binary in binaries.items():
                dest=out/f'{version}-{kind}-{name}.png'
                p=subprocess.run([str(binary),'--input',str(sp),'--graph',str(graph),'--output',str(dest)],capture_output=True)
                if p.returncode:raise RuntimeError(p.stderr.decode())
                image=np.asarray(Image.open(dest).convert('RGBA'));outputs[version,kind,name]=image
                delta=np.abs(image.astype(int)-source.astype(int))
                row={'version':version,'sourceType':kind,'variant':name,'maxChange':int(delta.max()),'changedPixels':int(delta.max(2).astype(bool).sum()),'outputSha256':sha(dest)}
                if name.endswith('-0') or name.endswith('-255'):assert np.array_equal(image,source),row
                if name=='multiply-protect-254':assert delta.max()<=3,row
                if name=='multiply-protect-128' and kind=='opaque':assert delta.max()<=128,row
                rows.append(row)
        for mode in ['normal','multiply']:
            # Opacity is a public effect interpolation, independent of the
            # foreground estimator and new source-residual implementation.
            full=outputs['candidate',kind,f'{mode}-100'].astype(float)
            if kind=='opaque':
                expected=np.floor((source.astype(float)+full)*.5+.5).astype('uint8')
                half=outputs['candidate',kind,f'{mode}-50']
                assert np.max(np.abs(expected.astype(int)-half.astype(int)))<=1
        if kind=='opaque':
            # Half protection must still allow a real, smaller geometric edit.
            # Its behavior is not merely an RGB cross-fade: protection damps
            # the velocity before the public opacity/blend operation.
            span=lambda image:np.flatnonzero(image[112,:,0]>150)[[0,-1]]
            original_span=span(source)
            full_span=span(outputs['candidate',kind,'normal-100'])
            soft_span=span(outputs['candidate',kind,'normal-protect-128'])
            assert original_span[0]<soft_span[0]<full_span[0],(original_span,soft_span,full_span)
            assert full_span[1]<soft_span[1]<original_span[1],(original_span,soft_span,full_span)
        if kind=='rgba':
            for name in variants:assert np.array_equal(outputs['old',kind,name],outputs['candidate',kind,name]),'RGBA fallback changed '+name
    report={'binaries':{k:{'path':str(v.relative_to(ROOT)),'sha256':sha(v)} for k,v in binaries.items()},'passed':True,'rgbaBaselineParityCases':len(variants),'scope':__doc__,'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
