from pathlib import Path
import json,hashlib,sys
from collections import deque
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'));from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dilate(a,n):
    v=a.copy()
    for _ in range(n):
        p=np.pad(v,1);v=p[1:-1,1:-1]|p[:-2,1:-1]|p[2:,1:-1]|p[1:-1,:-2]|p[1:-1,2:]
    return v

def panels(path,items,box,scale=1):
    w,h=box[2]-box[0],box[3]-box[1];im=Image.new('RGB',(w*len(items)*scale,h*scale+24),'#191d22');d=ImageDraw.Draw(im)
    for i,(title,a) in enumerate(items):
        if isinstance(a,np.ndarray):a=Image.fromarray(a)
        im.paste(a.convert('RGB').crop(box).resize((w*scale,h*scale),Image.Resampling.NEAREST),(i*w*scale,24));d.text((i*w*scale+4,7),title,fill='white')
    im.save(path)
def cutout(rgb,alpha):return np.rint(rgb*alpha[...,None]+np.array([45,80,60])*(1-alpha[...,None])).astype(np.uint8)
def main():
    r=json.loads((EXP/'outputs/report.json').read_text());source=ROOT/r['source'];src=np.asarray(Image.open(source).convert('RGB'));h,w=src.shape[:2]
    gp=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json';g=json.loads(gp.read_text());sem={k:raster_coverage(v,w,h) for k,v in g['operations'][0]['params']['semanticMasks'].items() if v['shape']=='raster'}
    mat=ROOT/'experiments/face-material-fill-v1-20261015/user-face-contour-100-full/layers';st=json.loads((mat/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];old=np.zeros((h,w));old[oy:oy+ch,ox:ox+cw]=np.fromfile(mat/'source-alpha-f32le.bin','<f4').reshape(ch,cw)
    frozen=np.zeros((h,w),bool);frozen[oy:oy+ch,ox:ox+cw]=np.fromfile(mat/'frozen-u8.bin',np.uint8).reshape(ch,cw)>0
    v2=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');opaque=v2['opaque'];pure=v2['pure'];trusted=v2['trusted']
    alphas={f"{row['mode']}-{row['attention']}":np.load(EXP/'outputs'/f"{row['mode']}-{row['attention']}"/'alphas-full.npy')[row['officialScoreArgmax']] for row in r['rows']}
    af=alphas['full-point'];ac=alphas['crop-point'];lm=np.array([[p['x']*w,p['y']*h] for p in g['operations'][0]['params']['landmarks']]);fw=np.ptp(lm[:,0]);contact=dilate(sem['accessories']>1/255,round(float(fw*.08)))
    # Stable source-only hypothesis, without declaring model agreement truth.
    shared=(af>=.95)&(ac>=.95);new=shared&(old<.5)&contact;ambiguous=(np.abs(af-ac)>.2)&contact
    fixedcore=(sem['ears']>=.5)|(sem['hair']>=.5)|opaque
    local=new&~fixedcore
    report={'scope':'Source RGB and frozen source alpha audit only. No rendering, no support replacement. Full/crop official2 candidates, not hand-selected masks.',
        'sourceSha256':sha(source),'sourceHashMatchesAuthor':sha(source)==r['sourceSha256'],'sourceAlphaSha256':sha(mat/'source-alpha-f32le.bin'),
        'officialSelections':{f"{x['mode']}-{x['attention']}":x['officialScoreArgmax'] for x in r['rows']},
        'allCandidateAlphaHashesMatch':all(sha(EXP/'outputs'/f"{x['mode']}-{x['attention']}"/'alphas-full.npy')==x['alphaSha256'] for x in r['rows']),
        'acceptedAsWholeMatte':False,'rows':[],'sharedNewNearAccessory':int(new.sum()),'sharedNewOutsideFixedCore':int(local.sum()),
        'sharedNewOutsideFixedCoreAlreadyV2Pure':int((local&pure).sum()),'sharedNewOutsideFixedCoreAlreadyV2Trusted':int((local&trusted).sum()),
        'sharedNewOutsideFixedCoreNotV2Support':int((local&~(pure|trusted)).sum()),'sharedNewOutsideFixedCoreFrozenOutput':int((local&frozen).sum()),
        'fullVsCropDisagreementGreaterPointTwoNearAccessory':int(ambiguous.sum())}
    for name,a in alphas.items():
        report['rows'].append({'name':name,'opaqueMetalAlphaAboveHalf':int((opaque&(a>.5)).sum()),'opaqueMetalAlphaAbovePointOne':int((opaque&(a>.1)).sum()),'opaqueMetalMaxAlpha':float(a[opaque].max()),
             'earCoreMeanAlpha':float(a[sem['ears']>=.95].mean()),'earCoreAlphaAboveHalf':int(((sem['ears']>=.95)&(a>.5)).sum()),
             'sourceSkinCoreRetainedFraction':float((a[sem['skin']>=.95]>.5).mean()),
             'nearAccessoryMeanAbsDifferenceFromFullPoint':float(np.abs(a-af)[contact].mean())})
    for name,mask in [('new-shared-support',new),('new-outside-core',local),('new-not-v2-support',local&~(pure|trusted)),('full-crop-disagreement',ambiguous),('metal-alpha-contamination',opaque&((af>.1)|(ac>.1)))]:
        Image.fromarray((mask*255).astype(np.uint8)).save(HERE/(name+'.png'))
    overlay=src.copy();overlay[local]=np.rint(src[local]*.25+np.array([0,255,255])*.75).astype(np.uint8)
    overlay2=src.copy();overlay2[local&~(pure|trusted)]=np.rint(src[local&~(pure|trusted)]*.25+np.array([255,0,255])*.75).astype(np.uint8)
    for label,box in [('left',(320,395,420,550)),('right',(600,395,705,550))]:
        panels(HERE/(label+'-1x.png'),[('Source',src),('Frozen alpha',cutout(src,old)),('Full official2',cutout(src,af)),('Crop official2',cutout(src,ac)),('Shared new cyan',overlay),('Not v2 magenta',overlay2)],box)
        panels(HERE/(label+'-3x.png'),[('Source',src),('Frozen alpha',cutout(src,old)),('Full official2',cutout(src,af)),('Crop official2',cutout(src,ac))],box,3)
    report['scriptSha256']=sha(Path(__file__));(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
