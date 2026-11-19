"""Prepare one explicitly manual, source-only body material control.

The role annotation is reviewed after an automatic failure. It is not automatic
segmentation or optical ground truth. Original body geometry is never replaced
by the matting result. Each stage refuses to overwrite prior evidence.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[2]
from model_runtime import sha, MEMATTE_WEIGHTS, predict_matte, inpaint

def write(path,content):Path(path).write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')

def validate(base):
    frozen=json.loads((base/'annotation-freeze.json').read_text())
    if any(sha(base/name)!=value for name,value in frozen.items()):raise ValueError('frozen manual source annotation changed')
    data=json.loads((base/'source-materials.json').read_text())
    source=Path(data['source'])
    if sha(source)!=data['sourceSHA256']:raise ValueError('source mismatch')
    if sha(source.parent/'source-graph.json')!=data['originalGraphSHA256']:raise ValueError('original geometry changed')
    if not data['manualIntervention'] or data['automaticGeneralizationClaim']:raise ValueError('manual correction must be explicit')
    return data,source

def front_input(base,source):
    request=base/'front-request.json'
    data=json.loads(request.read_text())
    if set(data) != {'version','source','sourceSHA256','role','polygons','uncertaintyPixels','provenance'}:
        raise ValueError('unexpected source-contour request fields')
    if data['version'] != 'explicit-source-contour/1' or data['role'] != 'stationary-front' or data['provenance'] != 'manual-source-image-contour':
        raise ValueError('manual source input must be explicitly declared')
    if Path(data['source']).resolve()!=source.resolve() or sha(source)!=data['sourceSHA256']:
        raise ValueError('source image mismatch')
    with Image.open(source) as opened:
        image=opened.convert('RGB')
    width,height=image.size
    band=data['uncertaintyPixels']
    if not isinstance(band,int) or band<1:
        raise ValueError('source contour requires an explicit uncertainty band')
    for polygon in data['polygons']:
        if len(polygon)<3 or any(not(0<=x<width and 0<=y<height) for x,y in polygon):
            raise ValueError('invalid source contour')
    return request,data,image

def front_prepare(base,source):
    request,data,image=front_input(base,source)
    target=base/'source-front-matte-selected-domain'
    target.mkdir(parents=True,exist_ok=False)
    mask=Image.new('L',image.size,0);draw=ImageDraw.Draw(mask)
    overlay=image.copy();lines=ImageDraw.Draw(overlay)
    for polygon in data['polygons']:
        vertices=[tuple(p) for p in polygon]
        draw.polygon(vertices,fill=255)
        lines.line(vertices+[vertices[0]],fill=(255,0,255),width=1)
    inside=np.array(mask)>0;band=data['uncertaintyPixels']
    tri=np.full(inside.shape,128,np.uint8)
    tri[distance_transform_edt(inside)>band]=255
    tri[distance_transform_edt(~inside)>band]=0
    mask.save(target/'manual-identity.png');overlay.save(target/'source-contour.png')
    Image.fromarray(tri).save(target/'trimap.png')
    preview=np.where(inside[...,None],np.array(image),np.array([40,170,90]))
    Image.fromarray(preview.astype('uint8')).save(target/'identity-preview.png')
    evidence={'source':str(source),'sourceSHA256':sha(source),
      'sourceRequestSHA256':sha(request),'manualPrompts':True,
      'manualContour':True,'sourceOnly':True,'motionRead':False,
      'neuralIdentityMaskModified':False,'automaticGeneralizationClaim':False,
      'uncertaintyPixels':band,'trimapSHA256':sha(target/'trimap.png')}
    (target/'protocol.json').write_text(json.dumps(evidence,indent=2))

def front_protocol(base,source):
    request,data,image=front_input(base,source)
    target=base/'source-front-matte-selected-domain'
    protocol=json.loads((target/'protocol.json').read_text())
    if protocol['sourceRequestSHA256']!=sha(request) or protocol['trimapSHA256']!=sha(target/'trimap.png'):
        raise ValueError('manual source input changed after freeze')
    return target,protocol,image

def front_matte(base,source):
    target,protocol,image=front_protocol(base,source)
    if (target/'alpha.npy').exists():
        raise FileExistsError('completed source matte must not be overwritten')
    weights=MEMATTE_WEIGHTS;rgb=np.array(image)
    tri=np.array(Image.open(target/'trimap.png').convert('L'))
    raw,alpha,metrics=predict_matte(rgb,tri,weights)
    np.save(target/'alpha-raw.npy',raw);np.save(target/'alpha.npy',alpha)
    Image.fromarray(np.rint(alpha*255).astype('uint8')).save(target/'alpha.png')
    preview=rgb*alpha[...,None]+np.array([40,170,90])*(1-alpha[...,None])
    Image.fromarray(np.rint(preview).astype('uint8')).save(target/'preview.png')
    evidence={**protocol,'weightsSHA256':sha(weights),'alphaIsEstimate':True,
      'qualityAccepted':False,**metrics}
    (target/'report.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))

def underlay(base,source):
    front,protocol,image=front_protocol(base,source)
    meta=json.loads((front/'report.json').read_text())
    if meta['sourceSHA256']!=sha(source) or not meta['sourceOnly']:
        raise ValueError('front layer is not bound to this source')
    rgb=np.array(image);alpha=np.load(front/'alpha.npy');mask=alpha>0
    target=base/'source-underlay';target.mkdir(exist_ok=False)
    Image.fromarray(np.uint8(mask)*255).save(target/'source-read-mask.png')
    completed,metrics=inpaint(Image.fromarray(rgb),mask)
    result=np.array(completed.convert('RGB'))
    if not np.array_equal(result[~mask],rgb[~mask]):
        raise ValueError('completion repainted observed source context')
    Image.fromarray(result).save(target/'underlay.png')
    evidence={'source':str(source),'sourceSHA256':sha(source),
      'frontAlphaSHA256':sha(front/'alpha.npy'),'sourceOnly':True,
      'manualPrompts':True,'outsideMaskChanged':0,
      'maskRule':'frozen source front alpha>0; no dilation',
      'hiddenRGBIsEstimate':True,'qualityAccepted':False,**metrics}
    (target/'report.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))

def rear_prepare(base,source):
    target=base/'rear-optical-source';target.mkdir(exist_ok=False)
    underlay=base/'source-underlay/underlay.png'
    under_meta=json.loads((base/'source-underlay/report.json').read_text())
    if under_meta['sourceSHA256']!=sha(source) or not under_meta['sourceOnly']:raise ValueError('unbound underlay')
    data=json.loads((base/'rear-identity.json').read_text())
    if data['sourceSHA256']!=sha(source):raise ValueError('unbound rear identity')
    image=Image.open(underlay).convert('RGB');rgb=np.asarray(image)
    mask=Image.new('L',image.size);draw=ImageDraw.Draw(mask)
    for polygon in data['polygons']:draw.polygon([tuple(p) for p in polygon],fill=255)
    selected=np.asarray(mask)>0;band=data['uncertaintyPixels']
    revision=base/'rear-trimap-source-review.json'
    if revision.exists():
        review=json.loads(revision.read_text())
        if review['sourceSHA256']!=sha(source) or review['identitySHA256']!=sha(base/'rear-identity.json') or not review['beforeRearInference']:
            raise ValueError('source trimap revision is not bound to original identity')
        band=review['uncertaintyPixels']
    tri=np.full(selected.shape,128,np.uint8)
    tri[distance_transform_edt(selected)>band]=255
    tri[distance_transform_edt(~selected)>band]=0
    mask.save(target/'manual-rear-support.png');Image.fromarray(tri).save(target/'trimap.png')
    weights=MEMATTE_WEIGHTS
    protocol={'sourceSHA256':sha(source),'underlaySHA256':sha(underlay),'identitySHA256':sha(base/'rear-identity.json'),
      'trimapSHA256':sha(target/'trimap.png'),'weightsSHA256':sha(weights),'sourceOnly':True,'manualIntervention':True,
      'inputIsOriginalWithFrontInpainted':True,'hiddenRearSupportIsEstimate':True,'opticalAlphaIsEstimate':True,
      'geometryModified':False,'motionRead':False,'automaticGeneralizationClaim':False,
      'uncertaintyPixels':band,'sourceReviewRevisionSHA256':sha(revision) if revision.exists() else None}
    write(target/'protocol.json',protocol)
    return

def rear_matte(base,source):
    target=base/'rear-optical-source'
    if (target/'alpha.npy').exists():raise FileExistsError('completed rear optical estimate exists')
    protocol=json.loads((target/'protocol.json').read_text())
    underlay=base/'source-underlay/underlay.png'
    weights=MEMATTE_WEIGHTS
    if any(protocol[key]!=sha(path) for key,path in [('sourceSHA256',source),('underlaySHA256',underlay),('identitySHA256',base/'rear-identity.json'),('trimapSHA256',target/'trimap.png'),('weightsSHA256',weights)]):
        raise ValueError('rear source inputs changed after freeze')
    if protocol['sourceReviewRevisionSHA256'] and sha(base/'rear-trimap-source-review.json')!=protocol['sourceReviewRevisionSHA256']:
        raise ValueError('rear source review changed after freeze')
    rgb=np.asarray(Image.open(underlay).convert('RGB'));tri=np.asarray(Image.open(target/'trimap.png').convert('L'))
    raw,alpha,metrics=predict_matte(rgb,tri,weights)
    np.save(target/'alpha-raw.npy',raw);np.save(target/'alpha.npy',alpha)
    Image.fromarray(np.floor(alpha*255+.5).astype('uint8')).save(target/'alpha.png')
    preview=rgb*alpha[...,None]+np.array([40,170,90])*(1-alpha[...,None])
    Image.fromarray(np.floor(preview+.5).astype('uint8')).save(target/'preview.png')
    write(target/'report.json',{**protocol,**metrics,'alphaSHA256':sha(target/'alpha.npy'),'visualAccepted':False})

def scene(base,source):
    target=base/'source-scene';target.mkdir(exist_ok=False)
    input_path=base/'source-underlay/underlay.png';alpha_path=base/'rear-optical-source/alpha.npy'
    rgb=np.array(Image.open(input_path).convert('RGB'));a=np.load(alpha_path);mask=a>0
    Image.fromarray(np.uint8(mask)*255).save(target/'source-read-mask.png')
    protocol={'sourceSHA256':sha(source),'underlaySHA256':sha(input_path),'rearAlphaSHA256':sha(alpha_path),
      'sourceOnly':True,'motionRead':False,'maskRule':'frozen rear optical alpha>0; no dilation',
      'hiddenSceneIsEstimate':True,'manualIntervention':True,'automaticGeneralizationClaim':False}
    write(target/'protocol.json',protocol)
    completed,metrics=inpaint(Image.fromarray(rgb),mask)
    result=np.array(completed.convert('RGB'))
    if not np.array_equal(result[~mask],rgb[~mask]):raise ValueError('known stationary scene repainted')
    Image.fromarray(result).save(target/'scene.png')
    write(target/'report.json',{**protocol,**metrics,'outsideMaskChanged':0,'sceneSHA256':sha(target/'scene.png'),'visualAccepted':False})

def export(base,source):
    target=base/'prepared';target.mkdir(exist_ok=False)
    sys.path[:0]=[str(ROOT/'experiments/body-sleeve-auto-20261117'),str(ROOT/'experiments/hair-skin-decomposition-v1-20261015')]
    from layers import prepare_layers
    from decompose import local_prior
    paths={'frontAlpha':base/'source-front-matte-selected-domain/alpha.npy','rearAlpha':base/'rear-optical-source/alpha.npy',
      'frontTrimap':base/'source-front-matte-selected-domain/trimap.png','underlay':base/'source-underlay/underlay.png','scene':base/'source-scene/scene.png'}
    under_meta=json.loads((base/'source-underlay/report.json').read_text())
    rear_meta=json.loads((base/'rear-optical-source/report.json').read_text())
    scene_meta=json.loads((base/'source-scene/report.json').read_text())
    bindings=[(under_meta['frontAlphaSHA256'],paths['frontAlpha']),
      (rear_meta['underlaySHA256'],paths['underlay']),(rear_meta['identitySHA256'],base/'rear-identity.json'),
      (rear_meta['trimapSHA256'],base/'rear-optical-source/trimap.png'),(rear_meta['alphaSHA256'],paths['rearAlpha']),
      (scene_meta['rearAlphaSHA256'],paths['rearAlpha']),(scene_meta['underlaySHA256'],paths['underlay']),
      (scene_meta['sceneSHA256'],paths['scene'])]
    if any(expected!=sha(path) for expected,path in bindings):raise ValueError('source preparation stage binding mismatch')
    protocol={'sourceSHA256':sha(source),'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
      'sourceOnly':True,'motionRead':False,'manualIntervention':True,'hiddenColorsAndRearShapeAreEstimates':True,
      'automaticGeneralizationClaim':False,'sourceReconstructionContract':'exact',
      'originalGeometryUnchanged':True,'sourceProjectionMethod':'existing prepare_layers; bounded exact front pair then bounded scene projection'}
    write(target/'protocol.json',protocol)
    src=np.array(Image.open(source).convert('RGB'));front=np.load(paths['frontAlpha']);rear=np.load(paths['rearAlpha'])
    underlay=np.array(Image.open(paths['underlay']).convert('RGB'));scene_rgb=np.array(Image.open(paths['scene']).convert('RGB'))
    tri=np.array(Image.open(paths['frontTrimap']).convert('L'));prior,_=local_prior(src.astype(float),tri==255)
    t=time.monotonic();pair=prepare_layers(src,front,rear,prior,underlay,scene_rgb)
    arrays={'front-alpha':pair['front_alpha'],'front-rgb':pair['foreground'],'rear-alpha':pair['rear_alpha'],
      'rear-source-rgb':pair['lower'],'scene-rgb':pair['scene']}
    for name,value in arrays.items():np.asarray(value,dtype='<f8').tofile(target/(name+'.f64'))
    error=float(np.max(np.abs(pair['source_reconstructed']-src)))
    if error>1e-9:raise ValueError('source mixture not exact')
    write(target/'report.json',{**protocol,'preparationSeconds':time.monotonic()-t,'sourceReconstructionMaxError':error,
      'floatingBoundCorrection':pair['floating_bound_correction'],'outputs':{name:sha(target/(name+'.f64')) for name in arrays},
      'sourceSceneProjectionChangedPixels':int(np.any(abs(pair['scene']-scene_rgb)>1e-8,axis=-1).sum()),'visualAccepted':False})

def replay(base,source,renderer):
    target=base/'ordinary';target.mkdir(exist_ok=False)
    prepared=base/'prepared';meta=json.loads((prepared/'report.json').read_text())
    if meta['sourceSHA256']!=sha(source):raise ValueError('prepared source mismatch')
    for name,digest in meta['outputs'].items():
        if sha(prepared/(name+'.f64'))!=digest:raise ValueError('frozen layer changed')
    graph_path=source.parent/'source-graph.json';graph=json.loads(graph_path.read_text());renderer=renderer.resolve()
    renderer_sha=sha(renderer)
    protocol={'sourceSHA256':sha(source),'originalGraphSHA256':sha(graph_path),'rendererSHA256':renderer_sha,'renderer':str(renderer),
      'layerReportSHA256':sha(prepared/'report.json'),'amounts':[0,-30,30],'onlyGraphChange':'params.adjustments.waist',
      'originalPartMaskPreserved':True,'originalProtectionPreserved':True,'originalAxisPreserved':True,
      'manualIntervention':True,'automaticGeneralizationClaim':False,'geometryAlphaSubstitution':False,
      'crops':{'waist':[410,280,770,680],'coat-right':[642,315,733,505],'foreground-sleeve':[565,310,681,414]},
      'expectedResponse':'left visible denim side only; right waist contour is occluded and not a measurement target'}
    write(target/'protocol.json',protocol)
    src=np.array(Image.open(source).convert('RGBA'));front=np.load(base/'source-front-matte-selected-domain/alpha.npy')
    sys.path.insert(0,str(Path(__file__).parent));from replay import sample_mask
    protection=sample_mask(graph['operations'][0]['params']['protectMask'],src.shape[1],src.shape[0])
    rows=[];images={}
    for amount in protocol['amounts']:
        label=f'{amount:+d}';current=json.loads(json.dumps(graph));current['operations'][0]['params']['adjustments']={'waist':amount}
        gp=target/(label+'.json');op=target/(label+'.png');fp=target/(label+'.f32');write(gp,current)
        if sha(renderer)!=renderer_sha:raise ValueError('renderer changed during replay')
        command=[str(renderer),'--source',str(source),'--prepared',str(prepared),'--graph',str(gp),'--output',str(op),'--save-field',str(fp)]
        t=time.monotonic();run=subprocess.run(command,text=True,capture_output=True)
        row={'amount':amount,'command':command,'processSeconds':time.monotonic()-t,'exitCode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'graphSHA256':sha(gp)}
        if run.returncode==0:
            actual=np.array(Image.open(op).convert('RGBA'));changed=np.any(actual!=src,axis=-1)
            field=np.fromfile(fp,'<f4').reshape(src.shape[:2]+(2,));fixed=np.all(field==0,axis=-1)
            row.update(outputSHA256=sha(op),fieldSHA256=sha(fp),changedPixels=int(changed.sum()),
              sourceIdentity=bool(np.array_equal(src,actual)),changedOpaqueFrontPixels=int((changed&(front==1)).sum()),
              changedProtectedPixels=int((changed&(protection>0)).sum()),changedZeroFieldPixels=int((changed&fixed).sum()),
              alphaUnchanged=bool(np.array_equal(actual[...,3],src[...,3])),maxDisplacement=float(np.linalg.norm(field,axis=-1).max()))
            images[amount]=Image.fromarray(actual[...,:3])
        rows.append(row);write(target/'report.json',{'protocol':protocol,'rows':rows,'visualAccepted':False})
        if run.returncode:raise RuntimeError(run.stderr)
    for name,box in protocol['crops'].items():
        scale=1 if name=='waist' else 3;cw=(box[2]-box[0])*scale;ch=(box[3]-box[1])*scale
        sheet=Image.new('RGB',(cw*3,ch+25),'#222');draw=ImageDraw.Draw(sheet)
        for col,amount in enumerate([0,-30,30]):
            crop=images[amount].crop(box).resize((cw,ch),Image.Resampling.NEAREST);sheet.paste(crop,(col*cw,25));draw.text((col*cw+5,5),f'{amount:+d}',fill='white')
        sheet.save(target/(name+'-triptych.png'))
    print(json.dumps({'output':str(target),'rows':[{k:v for k,v in r.items() if k not in ['command','stdout','stderr']} for r in rows]},indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['front-prepare','front-matte','underlay','rear-prepare','rear-matte','scene','export','replay']);ap.add_argument('directory',type=Path)
    ap.add_argument('--renderer',type=Path,default=ROOT/'core/image-core/target/release/examples/body-layered-replay')
    args=ap.parse_args();base=args.directory.resolve();data,source=validate(base)
    if args.stage=='front-prepare':front_prepare(base,source)
    elif args.stage=='front-matte':front_matte(base,source)
    elif args.stage=='underlay':underlay(base,source)
    elif args.stage=='rear-prepare':rear_prepare(base,source)
    elif args.stage=='rear-matte':rear_matte(base,source)
    elif args.stage=='scene':scene(base,source)
    elif args.stage=='export':export(base,source)
    else:replay(base,source,args.renderer)
if __name__=='__main__':main()
