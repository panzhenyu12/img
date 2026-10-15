"""Source-only pixel ownership diagnostic. No synthesis, warp detection, or production edits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label,binary_dilation,distance_transform_edt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
graph_path=source_path.with_name('face-graph.json')
rgb=np.asarray(Image.open(source_path).convert('RGB'));h,w=rgb.shape[:2]
params=json.loads(graph_path.read_text())['operations'][0]['params'];masks={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
base=ROOT/'experiments/face-material-fill-v1-20261015/user-face-contour-100-full/layers';state=json.loads((base/'state.json').read_text());ox,oy=state['origin'];cw,ch=state['width'],state['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
physical=np.zeros((h,w));physical[cs]=np.fromfile(base/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw)
frozen=np.zeros((h,w),bool);frozen[cs]=np.fromfile(base/'frozen-u8.bin',dtype='uint8').reshape(ch,cw)>0
layer_path=ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz';layers=np.load(layer_path)
support=layers['pure']|layers['trusted'];raw=layers['alpha'];opaque=layers['opaque']
out=HERE/'outputs';out.mkdir(exist_ok=True)

def tint(ms,color=(0,255,80)):
    a=rgb.astype('float64').copy();a[ms]=a[ms]*.4+np.array(color)*.6
    return Image.fromarray(np.rint(a).clip(0,255).astype('uint8'))

def scalar(a):
    return Image.fromarray(np.rint(a*255).clip(0,255).astype('uint8')).convert('RGB')

def panel(name,items,crop):
    x0,y0,x1,y1=crop;ww,hh=x1-x0,y1-y0
    im=Image.new('RGB',(len(items)*ww,hh+32),'#20242a');d=ImageDraw.Draw(im)
    for i,(title,img) in enumerate(items):im.paste(img.crop(crop),(i*ww,32));d.text((i*ww+2,4),title,fill='white')
    im.save(out/(name+'-1x.png'));im.resize((im.width*4,im.height*4),Image.Resampling.NEAREST).save(out/(name+'-4x.png'))

def axes(name,crop,grid=False):
    x0,y0,x1,y1=crop;s=6 if grid else 4;ww,hh=x1-x0,y1-y0;im=Image.new('RGB',(ww*s+45,hh*s+30),'#20242a');im.paste(Image.fromarray(rgb).crop(crop).resize((ww*s,hh*s),Image.Resampling.NEAREST),(45,30));d=ImageDraw.Draw(im)
    for x in range(((x0+9)//10)*10,x1,10):
        xx=45+(x-x0)*s;d.text((xx-8,8),str(x),fill='white')
        if grid:d.line((xx,30,xx,30+hh*s),fill=(0,100,255),width=1)
    for y in range(((y0+9)//10)*10,y1,10):
        yy=30+(y-y0)*s;d.text((5,yy-4),str(y),fill='white')
        if grid:d.line((45,yy,45+ww*s,yy),fill=(0,100,255),width=1)
    im.save(out/(name+('-grid.png' if grid else '-coordinates.png')))

report={'scope':'Original source only; diagnostic labels are manual observations, not ground truth or new alpha. No rendered-output detection.','sourceSha256':sha(source_path),'graphSha256':sha(graph_path),'physicalAlphaSha256':sha(base/'source-alpha-f32le.bin'),'v2LayersSha256':sha(layer_path),'sourceSupportTotal':int(support.sum()),'pure':int(layers['pure'].sum()),'trusted':int(layers['trusted'].sum()),'components':[]}
for component,crop in [(1,[320,390,410,540]),(2,[610,390,700,540])]:
    x0,y0,x1,y1=crop;region=np.zeros((h,w),bool);region[y0:y1,x0:x1]=True
    panel(f'component-{component}-source-ownership',[('Source',Image.fromarray(rgb)),('Physical alpha',scalar(physical)),('Frozen cyan',tint(frozen,(0,255,255))),('v2 skin green',tint(support)),('ZIM alpha',scalar(raw))],crop)
    panel(f'component-{component}-semantic',[('Source',Image.fromarray(rgb)),('Skin',scalar(masks['skin'])),('Ear',scalar(masks['ears'])),('Old accessory',scalar(masks['accessories'])),('Hair',scalar(masks['hair']))],crop)
    axes(f'component-{component}',crop);axes(f'component-{component}',crop,True)
    original_support=np.asarray(Image.open(ROOT/f'experiments/zim-accessory-combined-20261015/outputs/component-{component}-audit/combined-point-source-supported-excluded.png'))>0
    rr={'component':component,'crop':crop,'v2Support':int((support&region).sum()),'pure':int((layers['pure']&region).sum()),'trusted':int((layers['trusted']&region).sum()),'oldSourceSupported':int(original_support.sum()),'v2SupportFrozen':int((support&region&frozen).sum()),'v2PhysicalZero':int((support&region&(physical<=1e-6)).sum()),'v2PhysicalLess099':int((support&region&(physical<.99)).sum()),'v2PhysicalAlphaSum':float(physical[support&region].sum())}
    for amount in [100,-100]:
        b=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full/layers';v2=ROOT/f'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/{amount}-source-local-scene';flags={}
        for key in ['allowed-write','unknown-new-coverage','insufficient-scene-support']:
            a=np.zeros((h,w),bool);a[cs]=np.asarray(Image.open(v2/(key+'.png')))>0;flags[key]=a
        rr[str(amount)]={key:int((value&region&support).sum()) for key,value in flags.items()}
    report['components'].append(rr)
np.savez_compressed(out/'source-diagnostics.npz',sourceRGB=rgb,physical=physical,frozen=frozen,v2Support=support,v2Pure=layers['pure'],v2Trusted=layers['trusted'],accessoryAlpha=raw,opaque=opaque,**{('semantic_'+k):v for k,v in masks.items()})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
