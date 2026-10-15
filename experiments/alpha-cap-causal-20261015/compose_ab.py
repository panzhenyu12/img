"""Replay exact frozen inverse maps and A/B code with only alpha rule changed.

Original source trusted/domain and pure-reclaim are fixed. A/B connected-coverage
gates are recomputed by the same frozen functions; changed domains are reported.
No geometry, learned alpha inference, layer-order choice, or production edits.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label,binary_dilation
from alpha import VARIANTS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'experiments/hair-skin-decomposition-v1-20261015'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen_functions(path,names):
    tree=ast.parse(path.read_text())
    selected=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),globals())
frozen_functions(HERE/'frozen/compose_real.py',['sample','montage'])
frozen_functions(HERE/'frozen/occlusion_hypotheses.py',['connected_coverage','write_domains'])
spec=importlib.util.spec_from_file_location('source_probe',ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/run.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
def save_mask(path,a):Image.fromarray(np.floor(np.clip(a,0,1)*255+.5).astype('uint8')).save(path)
def main():
    out=HERE/'renders';out.mkdir(exist_ok=False)
    source_report=HERE/'source/report.json';estimates=json.loads(source_report.read_text())
    report=dict(scope=__doc__,sourceReportSha256=sha(source_report),scriptSha256=sha(Path(__file__)),
        helpers={p.name:sha(p) for p in [HERE/'frozen/compose_real.py',HERE/'frozen/occlusion_hypotheses.py']},rows=[])
    for name in ['portrait-woman','user-face-contour','portrait-hair']:
      estimate=next(r for r in estimates['rows'] if r['name']==name)
      for amount in [100,-100]:
        material=ROOT/'experiments/face-material-fill-v1-20261015'/f"{'woman' if name=='portrait-woman' else name}-{amount}-full"
        layers=material/'layers';state=json.loads((layers/'state.json').read_text());cw,ch=state['width'],state['height'];ox,oy=state['origin']
        source=np.asarray(Image.open(ROOT/estimate['source']).convert('RGBA'));base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'));h,w=source.shape[:2]
        source_rgb=source[oy:oy+ch,ox:ox+cw,:3].astype('float64')/255
        source_alpha=np.fromfile(layers/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype('float64')
        xy=np.fromfile(layers/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2)
        person=np.fromfile(layers/'person-coverage-f32le.bin',dtype='<f4').reshape(ch,cw)
        foreground=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype='float64')/255
        params=json.loads((ROOT/estimate['graph']).read_text())['operations'][0]['params']
        masks={k:probe.zim.coverage(v,w,h)[oy:oy+ch,ox:ox+cw] for k,v in params['semanticMasks'].items()}
        protection=np.maximum(masks['ears'],masks['accessories'])>=.05
        yy,xx=np.indices((ch,cw));r=params['region']
        roi=((xx+ox+.5)/w>=r['x'])&((xx+ox+.5)/w<=r['x']+r['width'])&((yy+oy+.5)/h>=r['y'])&((yy+oy+.5)/h<=r['y']+r['height'])
        allowed=roi&~protection&(person>=.5);moved=np.hypot(xy[...,0]-xx,xy[...,1]-yy)>1e-4
        x0,y0,x1,y1=estimate['crop'];sl=np.s_[y0-oy:y1-oy,x0-ox:x1-ox]
        backings=[('v36-backing',np.asarray(Image.open(layers/'background.png').convert('RGB'),dtype='float64')/255,base)]
        if name=='portrait-woman' and amount==100:
            back=ROOT/'experiments/face-background-context-20261015/woman-100-local/256'
            backings.append(('lama-256',np.asarray(Image.open(back/'background-layer.png').convert('RGB'),dtype='float64')/255,np.asarray(Image.open(back/'candidate.png').convert('RGBA'))))
        outputs={};write_masks={}
        for variant in VARIANTS:
          row=next(v for v in estimates['rows'] if v['name']==name and v['variant']==variant)
          assert sha(ROOT/row['directory']/'layers.npz')==row['layersSha256']
          data=np.load(ROOT/row['directory']/'layers.npz')
          trusted=np.zeros((ch,cw),bool);trusted[sl]=data['trusted']
          pure=np.zeros((ch,cw),bool);pure[sl]=data['pureReclaim']
          hair_core=np.zeros((ch,cw),bool);hair_core[sl]=data['hairCore']
          hair_alpha=np.load(ROOT/estimate['selectedAlpha'],mmap_mode='r')[estimate['officialIndex'],oy:oy+ch,ox:ox+cw].astype('float64')
          hair_alpha[sl]=np.where(data['trusted'],data['alpha'],hair_alpha[sl])
          hair_color=source_rgb.copy();p=hair_color[sl].copy();p[data['trusted']|data['pureReclaim']]=data['hair'][data['trusted']|data['pureReclaim']];hair_color[sl]=p
          skin_color=foreground.copy();p=skin_color[sl].copy();p[data['trusted']]=data['skin'][data['trusted']];p[data['pureReclaim']]=source[y0:y1,x0:x1,:3][data['pureReclaim']]/255;skin_color[sl]=p
          physical=source_alpha.copy();p=physical[sl].copy();p[data['trusted']]=1-data['alpha'][data['trusted']];p[data['pureReclaim']]=1;physical[sl]=p
          under=physical.copy();under[trusted]=np.clip(physical[trusted]/np.maximum(1-hair_alpha[trusted],.001),0,1)
          alpha_b=sample(physical,xy)[...,0];rgb_b=sample(skin_color*physical[...,None],xy)
          alpha_a=sample(under,xy)[...,0];rgb_a=sample(skin_color*under[...,None],xy)
          transported=sample((trusted|pure).astype('float64'),xy)[...,0]>1e-6
          cover_b,unknown_b,_,_=connected_coverage(alpha_b,physical,masks['hair'],trusted|pure,allowed&moved)
          cover_a,unknown_a,_,_=connected_coverage(alpha_a,under,masks['hair'],trusted|pure,allowed&moved)
          edge=(trusted|pure|transported)&allowed&moved
          wa,wb=write_domains(edge,hair_core,cover_a,cover_b,unknown_a,unknown_b)
          fixed_hair=np.where(masks['hair']>=.05,hair_alpha,0);fixed_hair[hair_core]=1
          write_masks[variant]={'A':wa,'B':wb}
          for backing_name,back,initial in backings:
            dest=out/f'{name}-{amount}'/backing_name/variant;dest.mkdir(parents=True)
            backing=back.copy();backing[trusted|pure]=hair_color[trusted|pure]
            beneath=rgb_a+(1-alpha_a[...,None])*backing
            ra=fixed_hair[...,None]*hair_color+(1-fixed_hair[...,None])*beneath
            rb=rgb_b+(1-alpha_b[...,None])*backing
            variants={};results={}
            for order,render,writes in [('A',ra,wa),('B',rb,wb)]:
                output=initial.copy();patch=output[oy:oy+ch,ox:ox+cw,:3].copy();patch[writes]=np.floor(np.clip(render[writes],0,1)*255+.5).astype('uint8');output[oy:oy+ch,ox:ox+cw,:3]=patch
                full_write=np.zeros((h,w),bool);full_write[oy:oy+ch,ox:ox+cw]=writes
                assert np.array_equal(output[~full_write],initial[~full_write])
                assert np.array_equal(output[...,3],source[...,3])
                assert np.array_equal(output[oy:oy+ch,ox:ox+cw][protection],initial[oy:oy+ch,ox:ox+cw][protection])
                fixed_core=hair_core if order=='A' else hair_core&~cover_b
                assert np.array_equal(output[oy:oy+ch,ox:ox+cw][fixed_core],source[oy:oy+ch,ox:ox+cw][fixed_core])
                image_file=dest/f'{order}.png';Image.fromarray(output).save(image_file);save_mask(dest/f'{order}-write.png',writes)
                old_file=OLD/'occlusion-corrected-prompts'/f'{name}-{amount}'/backing_name/('A-hair-front.png' if order=='A' else 'B-skin-front.png')
                prior=np.asarray(Image.open(old_file).convert('RGBA'))
                difference=np.any(output!=prior,axis=-1)
                if variant=='bounded-015':assert not difference.any(),(name,amount,order)
                outputs[(backing_name,variant,order)]=output
                baseline_write=write_masks['bounded-015'][order]
                results[order]=dict(outputSha256=sha(image_file),changedFromBoundedPixels=int(difference.sum()),writes=int(writes.sum()),
                    writeChangesFromBounded=int(np.count_nonzero(writes!=baseline_write)),fixedTextureCoordinate=True,opaqueUncoveredCoreChanged=0,
                    commonWriteChangedPixels=int((difference[oy:oy+ch,ox:ox+cw]&writes&baseline_write).sum()))
            entry=dict(name=name,amount=amount,variant=variant,backing=backing_name,sourceSha256=sha(ROOT/estimate['source']),
                inverseSha256=sha(layers/'inverse-source-xy-f32le.bin'),sourceAlphaSha256=sha(layers/'source-alpha-f32le.bin'),
                layersSha256=row['layersSha256'],frozenTrustedPixels=int(trusted.sum()),orders=results,
                unknownCoveragePixels=int((unknown_a|unknown_b).sum()),newHairOcclusionPixels=int((cover_b&hair_core).sum()),
                directory=str(dest.relative_to(ROOT)),accepted=False)
            report['rows'].append(entry);print(json.dumps({k:entry[k] for k in ['name','amount','variant','backing','orders']}),flush=True)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for backing_name,_,initial in backings:
          for order in ['A','B']:
            dest=out/f'{name}-{amount}'/backing_name
            images=[Image.fromarray(source),Image.fromarray(base)]+[Image.fromarray(outputs[(backing_name,v,order)]) for v in VARIANTS]
            labels=['Source','V36','Bounded .15','Uncapped','Donor interval']
            for side,crop in [('left',[x0,y0,(x0+x1)//2,y1]),('right',[(x0+x1)//2,y0,x1,y1])]:montage(dest/f'{side}-{order}-comparison-1x.png',images,labels,crop)
    assert sha(source_report)==report['sourceReportSha256']
    for name,digest in report['helpers'].items():assert sha(HERE/'frozen'/name)==digest

if __name__=='__main__':main()
