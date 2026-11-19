"""Source-only causal control: prompted sleeve, frozen other manual identities."""
from pathlib import Path
import argparse
import json
import math
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, distance_transform_edt

ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT/'experiments/body-fresh-source-20261119'
sys.path.insert(0,str(SHARED))
from model_runtime import sha,predict_matte,inpaint,MEMATTE_WEIGHTS
import prepare_manual_layers as shared

SOURCE=ROOT/'test/results/body-material-candidates-20261119/side-fashion-source/source.png'
PROPOSAL=ROOT/'test/results/body-material-contrastive-20261119/side-fashion/sleeve-m1-mask.png'
CONTROL=ROOT/'test/results/body-fresh-source-20261119/fullbody-side-fashion/manual-role-control-v3'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def proposal_trimap(mask,band):
    """A binary proposal supplies a core; missed enclosed holes stay unknown."""
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or not mask.any() or band<1:
        raise ValueError('nonempty 2D mask and positive uncertainty width required')
    tri=np.full(mask.shape,128,np.uint8)
    tri[distance_transform_edt(mask)>band]=255
    tri[distance_transform_edt(~mask)>band]=0
    holes=binary_fill_holes(mask)&~mask
    tri[holes]=128
    return tri,holes


def prepare(base):
    base.mkdir(parents=True,exist_ok=False)
    original=json.loads((CONTROL/'source-materials.json').read_text())
    freeze=json.loads((CONTROL/'annotation-freeze.json').read_text())
    for name in ['source-materials.json','rear-identity.json']:
        assert sha(CONTROL/name)==freeze[name]
    assert original['sourceSHA256']==sha(SOURCE)
    assert original['originalGraphSHA256']==sha(SOURCE.parent/'source-graph.json')
    assert original['manualIntervention'] and not original['automaticGeneralizationClaim']
    inputs={'source':SOURCE,'graph':SOURCE.parent/'source-graph.json','sleeveProposal':PROPOSAL,
            'promptProtocol':PROPOSAL.parent/'protocol.json','promptRun':PROPOSAL.parent/'run.json',
            'manualControl':CONTROL/'source-materials.json','manualControlFreeze':CONTROL/'annotation-freeze.json',
            'rearIdentity':CONTROL/'rear-identity.json','rearSourceReview':CONTROL/'rear-trimap-source-review.json',
            'weights':MEMATTE_WEIGHTS,'script':Path(__file__),'runtime':SHARED/'model_runtime.py',
            'sharedPreparation':SHARED/'prepare_manual_layers.py'}
    source=Image.open(SOURCE).convert('RGB');rgb=np.array(source)
    sleeve=np.array(Image.open(PROPOSAL).convert('L'))>0
    # Two cells of the stored 256x256 SAM decoder grid, frozen before matting.
    sleeve_band=math.ceil(2*max(source.size)/256)
    sleeve_tri,holes=proposal_trimap(sleeve,sleeve_band)
    fixed_roles=[r for r in original['roles'] if r['role']=='stationary-front' and r['id']!='sweater-front-arm']
    assert len(fixed_roles)==6
    fixed_image=Image.new('L',source.size,0);draw=ImageDraw.Draw(fixed_image)
    for role in fixed_roles:
        draw.polygon([tuple(p) for p in role['polygon']],fill=255)
    fixed=np.array(fixed_image)>0
    fixed_tri=np.full(fixed.shape,128,np.uint8)
    fixed_tri[distance_transform_edt(fixed)>4]=255
    fixed_tri[distance_transform_edt(~fixed)>4]=0
    combined=np.maximum(sleeve_tri,fixed_tri)
    target=base/'source-front-matte-selected-domain';target.mkdir()
    for name,arr in [('sleeve-proposal',sleeve.astype(np.uint8)*255),('sleeve-trimap',sleeve_tri),
                     ('sleeve-enclosed-holes',holes.astype(np.uint8)*255),('fixed-manual-proposal',fixed.astype(np.uint8)*255),
                     ('fixed-manual-trimap',fixed_tri),('trimap',combined)]:
        Image.fromarray(arr).save(target/(name+'.png'))
    colors=np.zeros_like(rgb);colors[combined==255]=[0,255,80];colors[combined==128]=[255,210,0]
    overlay=np.where((combined>0)[...,None],np.rint(rgb*.55+colors*.45),np.rint(rgb*.3)).astype(np.uint8)
    Image.fromarray(overlay).save(target/'source-trimap.png')
    Image.fromarray(overlay).crop((480,200,700,440)).resize((660,720)).save(target/'source-trimap-sleeve.png')
    protocol={'version':'manual-contrastive-sleeve-source-control/1','source':str(SOURCE),
              'sourceSHA256':sha(SOURCE),'originalGraphSHA256':sha(SOURCE.parent/'source-graph.json'),
              'inputs':{key:{'path':str(path),'sha256':sha(path)} for key,path in inputs.items()},
              'manualIntervention':True,'automaticGeneralizationClaim':False,'sourceOnly':True,'motionRead':False,
              'sleeveIdentity':'Frozen SAM2 sleeve-m1 from four manual contrastive source points; no old sleeve polygon',
              'otherManualIdentityRoles':[r['id'] for r in fixed_roles]+['torso-and-skirt','coat-left','coat-right'],
              'otherManualRolesConstant':True,'oldAlphaRead':False,'oldPreparedLayersRead':False,
              'frontUncertainty':{'sleevePixels':sleeve_band,'rule':'ceil(2*max(source dimensions)/256); two decoder cells',
                                  'enclosedSleeveMisses':'unknown, not fixed transparent','enclosedMissPixels':int(holes.sum()),
                                  'otherManualPixels':4,'combinedRule':'union of component trimap support; known foreground of either component is front'},
              'rearUncertainty':'Same-source frozen manual identity and pre-inference 8px source review reused as control constants; new rear inference will use newly prepared underlay',
              'hiddenSupportAndColorsAreEstimates':True,'allImageLayerDependenciesRecomputed':True,
              'notAClaim':'Four points do not prepare the entire image: other manual identities remain explicit controls.',
              'trimapSHA256':sha(target/'trimap.png')}
    write(base/'protocol.json',protocol)
    write(base/'manual-controls.json',{'sourceSHA256':sha(SOURCE),'frontRoles':fixed_roles,
                                    'rearAndSceneRoles':[r for r in original['roles'] if r['role']!='stationary-front'],
                                    'sourceControl':str(CONTROL),'oldSleevePolygonUsed':False})
    write(target/'protocol.json',protocol)
    # Only source identity requests are copied. No alpha, underlay or edit is read.
    for name in ['rear-identity.json','rear-trimap-source-review.json']:
        shutil.copyfile(CONTROL/name,base/name)
    print(json.dumps({'stage':'prepared','sleeveBand':sleeve_band,'holesUnknown':int(holes.sum()),'protocolSHA256':sha(base/'protocol.json')}),flush=True)


def validate(base):
    p=json.loads((base/'protocol.json').read_text())
    for entry in p['inputs'].values():
        assert sha(entry['path'])==entry['sha256'],entry['path']
    assert sha(base/'source-front-matte-selected-domain/trimap.png')==p['trimapSHA256']
    return p


def front_matte(base):
    p=validate(base);target=base/'source-front-matte-selected-domain'
    assert not (target/'alpha.npy').exists()
    rgb=np.array(Image.open(SOURCE).convert('RGB'));tri=np.array(Image.open(target/'trimap.png'))
    raw,alpha,metrics=predict_matte(rgb,tri,MEMATTE_WEIGHTS)
    np.save(target/'alpha-raw.npy',raw);np.save(target/'alpha.npy',alpha)
    Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(target/'alpha.png')
    for name,bg in [('green',[40,170,90]),('dark',[30,30,30])]:
        preview=np.rint(rgb*alpha[...,None]+np.asarray(bg)*(1-alpha[...,None])).astype(np.uint8)
        Image.fromarray(preview).save(target/(name+'-preview.png'))
        Image.fromarray(preview).crop((480,200,700,440)).resize((660,720)).save(target/(name+'-sleeve.png'))
    report={**p,**metrics,'alphaSHA256':sha(target/'alpha.npy'),'alphaIsEstimate':True,'qualityAccepted':False}
    write(target/'report.json',report);print(json.dumps(metrics),flush=True)


def underlay(base):
    validate(base);target=base/'source-underlay';target.mkdir(exist_ok=False)
    path=base/'source-front-matte-selected-domain/alpha.npy'
    front_report=json.loads((path.parent/'report.json').read_text());assert front_report['alphaSHA256']==sha(path)
    rgb=np.array(Image.open(SOURCE).convert('RGB'));alpha=np.load(path);mask=alpha>0
    Image.fromarray(mask.astype(np.uint8)*255).save(target/'source-read-mask.png')
    completed,metrics=inpaint(Image.fromarray(rgb),mask);result=np.array(completed.convert('RGB'))
    assert np.array_equal(result[~mask],rgb[~mask])
    Image.fromarray(result).save(target/'underlay.png')
    write(target/'report.json',{'source':str(SOURCE),'sourceSHA256':sha(SOURCE),'frontAlphaSHA256':sha(path),
          'sourceOnly':True,'motionRead':False,'manualPrompts':True,'automaticGeneralizationClaim':False,
          'maskRule':'fresh frozen front alpha>0; no dilation','outsideMaskChanged':0,'hiddenRGBIsEstimate':True,
          'qualityAccepted':False,**metrics})
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','front-matte','underlay','rear-prepare','rear-matte','scene','export'])
    parser.add_argument('base',type=Path);args=parser.parse_args()
    if args.stage=='prepare':prepare(args.base)
    elif args.stage=='front-matte':front_matte(args.base)
    elif args.stage=='underlay':underlay(args.base)
    else:
        validate(args.base)
        getattr(shared,args.stage.replace('-','_'))(args.base,SOURCE)
