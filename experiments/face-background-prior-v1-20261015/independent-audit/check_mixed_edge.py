"""Independent reconstruction/permission audit of frozen background factors."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=EXP.parents[1]
HAIR=ROOT/'experiments/hair-skin-decomposition-v1-20261015'
sys.path.insert(0,str(HAIR))
from run_real import probe
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rgba=lambda p:np.asarray(Image.open(p).convert('RGBA'))
mask=lambda p:np.asarray(Image.open(p))>0
case=sys.argv[1] if len(sys.argv)>1 else 'mixed-edge'
assert case in ['mixed-edge','mixed-edge-protected']
report=read(EXP/case/'report.json')
for p,digest in report['inputHashes'].items():assert sha(ROOT/p)==digest,(p,digest)
material=ROOT/'experiments/face-material-fill-v1-20261015/woman-100-full'
layers=material/'layers';state=read(layers/'state.json');ch,cw=state['height'],state['width'];ox,oy=state['origin']
source_path=Path(read(material/'provenance.json')['source']);source=rgba(source_path);h,w=source.shape[:2]
area=np.s_[oy:oy+ch,ox:ox+cw]
bdir=HAIR/'occlusion-corrected-prompts/portrait-woman-100/lama-256';old_b=rgba(bdir/'B-skin-front.png')
prior=ROOT/'experiments/face-background-context-20261015/woman-100'
old_target=mask(prior/'write-target.png');bwrite=mask(bdir/'B-write.png');target=mask(EXP/case/'write-target.png')
full_target=np.zeros((h,w),bool);full_target[area]=target
estimate=next(r for r in read(HAIR/'source-decomposition-corrected-prompts/report.json')['rows'] if r['name']=='portrait-woman' and r['variant']=='source-fit-srgb')
data=np.load(HAIR/estimate['directory']/'layers.npz');x0,y0,x1,y1=estimate['crop'];local=np.s_[y0-oy:y1-oy,x0-ox:x1-ox]
physical=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype(float)
p=physical[local].copy();p[data['trusted']]=1-data['alpha'][data['trusted']];p[data['pureReclaim']]=1;physical[local]=p
fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype=float)/255
p=fg[local].copy();p[data['trusted']]=data['skin'][data['trusted']];p[data['pureReclaim']]=source[y0:y1,x0:x1,:3][data['pureReclaim']]/255;fg[local]=p
xy=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2)
coordinates=[xy[...,1],xy[...,0]]
alpha=map_coordinates(physical,coordinates,order=1,mode='grid-constant',cval=0,prefilter=False)
rgb=np.stack([map_coordinates(fg[...,i]*physical,coordinates,order=1,mode='grid-constant',cval=0,prefilter=False) for i in range(3)],-1)
samples=np.fromfile(layers/'samples-f32le.bin','<f4').reshape(ch,cw,5)
alpha[~bwrite]=samples[...,4][~bwrite];rgb[~bwrite]=samples[...,:3][~bwrite]/255
params=read(ROOT/estimate['graph'])['operations'][0]['params']
protected=np.maximum(*[probe.zim.coverage(params['semanticMasks'][k],w,h)[area] for k in ['ears','accessories']])>=.05
expected=(old_target|bwrite)&~protected if case=='mixed-edge-protected' else old_target|bwrite
assert np.array_equal(target,expected)
core=np.zeros((ch,cw),bool);core[local]=data['hairCore']
rows=[];images=[source,old_b]
for row in report['rows']:
    d=EXP/case/row['name'];out=rgba(d/'candidate.png');assert sha(d/'candidate.png')==row['outputSha256']
    changed=np.any(out!=old_b,axis=2)
    assert not (changed&~full_target).any()
    assert np.array_equal(out[...,3],old_b[...,3])
    bg=np.asarray(Image.open(d/'inferred-background.png').convert('RGB'),dtype=float)/255
    replay=np.floor(np.clip(rgb+(1-alpha[...,None])*bg,0,1)*255+.5).astype('uint8')
    error=np.abs(replay[target].astype(int)-out[area][target,:3].astype(int))
    assert error.max()==0, 'Frozen foreground / alpha independent replay disagrees'
    forbidden_changes=changed[area]&protected
    Image.fromarray((forbidden_changes*255).astype('uint8')).save(HERE/(case+'-'+row['name']+'-ear-accessory-changes.png'))
    yy,xx=np.where(forbidden_changes)
    locations=[{'x':int(x+ox),'y':int(y+oy),'before':old_b[y+oy,x+ox,:3].tolist(),'after':out[y+oy,x+ox,:3].tolist()} for y,x in zip(yy,xx)]
    if case=='mixed-edge-protected':
        assert not forbidden_changes.any()
        assert not (changed[area]&core).any()
    rows.append({'name':row['name'],'changedVsPreviousBPixels':int(changed.sum()),'outsideUnionChanged':0,
                 'sourceAlphaAndForegroundReplayMaxByteError':int(error.max()),'opaqueHairCoreChanged':int((changed[area]&core).sum()),
                 'earAccessoryChanged':int(forbidden_changes.sum()),'earAccessoryCoordinates':locations,
                 'outputAlphaChanged':0,'outputSha256':sha(d/'candidate.png')})
    images.append(out)
# Right-side full native crop is deliberately independent of the author's
# left-side selection, so a localized improvement cannot hide another seam.
box=(590,480,915,1270);cw2=box[2]-box[0];ch2=box[3]-box[1]
canvas=Image.new('RGB',(cw2*4,ch2+28),'#20242a');draw=ImageDraw.Draw(canvas)
for i,(im,label) in enumerate(zip(images,['Source','Previous B','Observed only','Virtual context'])):
    canvas.paste(Image.fromarray(im).convert('RGB').crop(box),(i*cw2,28));draw.text((i*cw2+4,7),label,fill='white')
canvas.save(HERE/(case+'-right-comparison-1x.png'))
result={'scope':'One original woman +100; explicit B fixed; background experiment only.',
    'oldTargetPixels':int(old_target.sum()),'BWritePixels':int(bwrite.sum()),'unionTargetPixels':int(target.sum()),
    'excludedSoftProtectedPixels':int(((old_target|bwrite)&protected).sum()),
    'inheritedEarAccessoryVsSource':int((protected&np.any(old_b[area]!=source[area],axis=2)).sum()),
    'independentReconstruction':'SciPy bilinear sampling of frozen premultiplied source and alpha',
    'rows':rows,'interpretation':'Observed and virtual contexts also use different inference masks; joint conditioning experiment, not RGB-only factor.'}
(HERE/(case+'-report.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
