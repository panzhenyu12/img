"""One frozen source estimator, one known-layer lash fixture, two fixed photos.

No detector, model, parameter sweep, target-conditioned source fit or core build.
Uses the existing layer compositor and existing -15 fixed-canthi inverse fields.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from PIL import Image,ImageDraw
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/eye-height-layers-20261117'
sys.path.insert(0,str(OLD))
from layers import compose,sample
from source_curves import frame,EYES
from source_estimator import estimate,prepare_sampling_support

BASE=ROOT/'test/results/eye-height-layers-20261117'
OUT=ROOT/'test/results/eye-closing-source-20261119'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def png(path,array):Image.fromarray(np.rint(array).astype(np.uint8)).save(path)


def save_layers(directory,source,layers):
    np.savez_compressed(directory/'candidate-source.npz',
        **{key:value for key,value in layers.items() if key!='report'})
    write(directory/'source-report.json',layers['report'])
    png(directory/'source.png',source)
    png(directory/'exposure.png',layers['exposure']*255)
    png(directory/'ocular.png',layers['ocular'])
    png(directory/'ambiguous.png',layers['ambiguous']*255)


def sheet(path,panels,scale=5):
    h,w=panels[0][1].shape[:2];sw,sh=w*scale,h*scale
    out=Image.new('RGB',(sw*len(panels),sh+24),'#222');d=ImageDraw.Draw(out)
    for i,(label,array) in enumerate(panels):
        image=Image.fromarray(np.rint(array).astype(np.uint8)).convert('RGB')
        out.paste(image.resize((sw,sh),Image.Resampling.NEAREST),(i*sw,24))
        d.text((i*sw+3,7),label,fill='white')
    out.save(path)


def synthetic(limbus=False):
    directory=OUT/('known-lash-limbus' if limbus else 'known-lash');directory.mkdir()
    yy,xx=np.mgrid[:41,:81];u=xx-10.;v=yy-20.;width=60.
    shape=lambda u:8*4*(u/width)*(1-u/width)
    upper=lambda u:-shape(u);lower=lambda u:shape(u)
    to_pixel=lambda u,v:np.stack([u+10,v+20],axis=-1)
    inside=(u>=0)&(u<=width);safe=np.clip(u,0,width)
    depth=np.minimum(v-upper(safe),lower(safe)-v)
    aperture=np.clip(depth+.5,0,1)*inside
    # A source-observed ocular texture that varies across, but not along, the
    # short normal donor stencil; one stationary iris/catchlight remains inside.
    ocular=np.stack([190+xx*.1,205+xx*.1,220+xx*.1],axis=-1)
    iris=((u-30)/10)**2+(v/9)**2<1
    ocular[iris]=np.stack([40+10*np.sin(u[iris]),65+8*np.cos(u[iris]),100+8*np.sin(u[iris])],axis=-1)
    if limbus:
        # Discriminating counterexample: normal texture and an iris/sclera
        # boundary cross the lash. Constant B along a vertical field would let
        # an incorrect E cancel algebraically and conceal source-estimator error.
        ocular+=v[...,None]*.7
        ocular[iris]+=12*np.sin(v[iris]*.8)[:,None]*np.array([1.,.5,.8])
    ocular[(abs(u-31)<1.5)&(abs(v+2)<1)]=[248,249,250]
    skin=np.broadcast_to([180.,120.,90.],ocular.shape)
    inward=v-upper(safe)
    lash_u=22 if limbus else 15
    lash=np.clip(1.1-abs(u-(lash_u+.35*inward)),0,1)*((inward>=-.5)&(inward<2.6))
    exposure=aperture*(1-lash)
    residual=(1-aperture[...,None])*skin+aperture[...,None]*lash[...,None]*[12.,10.,8.]
    source=residual+exposure[...,None]*ocular
    estimate_layers=estimate(source,u,v,upper,lower,to_pixel,width)
    save_layers(directory,source,estimate_layers)
    identity=np.stack([xx,yy],axis=-1).astype(float)
    window=np.sin(np.pi*safe/width)**2*inside
    scale=np.exp(.3*(-15)/100*window)
    outside_taper=np.clip((18-abs(v))/4,0,1)
    inverse=identity.copy();inverse[...,1]+=v*(1/scale-1)*outside_taper
    output,metrics=compose(source,estimate_layers['exposure'],estimate_layers['ocular'],
        estimate_layers['support'],inverse)
    truth=sample(residual,inverse)+sample(exposure,inverse)[...,None]*ocular
    target_exposure=sample(exposure,inverse)
    visible=(target_exposure==1)&(exposure==1)&estimate_layers['core']
    lash_pixels=(lash>.5)&(aperture==1)
    lash_error=float(np.max(abs(estimate_layers['exposure'][lash_pixels]-exposure[lash_pixels])))
    true_lash=sample(aperture*lash,inverse)
    near_lash=(lash>0)|(true_lash>0)
    output_error=float(np.max(abs(output[near_lash]-truth[near_lash])))
    visible_error=float(np.max(abs(output[visible]-source[visible])))
    zero,_=compose(source,estimate_layers['exposure'],estimate_layers['ocular'],estimate_layers['support'],identity)
    changed=estimate_layers['ocular'].copy();changed[~estimate_layers['support']]=[255,0,255]
    poisoned,_=compose(source,estimate_layers['exposure'],changed,estimate_layers['support'],inverse)
    tiny=identity+(inverse-identity)*1e-5
    almost,_=compose(source,estimate_layers['exposure'],estimate_layers['ocular'],estimate_layers['support'],tiny)
    tests={
        'knownLashExposureMaxErrorUnderPointOne':lash_error<.1,
        'movedLashRgbMaxErrorUnderTen':output_error<10,
        'visibleIrisCatchlightExact':visible_error<1e-9,
        'zeroExact':bool(np.array_equal(zero,source)),
        'unknownOutsideUnderlayCannotAffectOutput':bool(np.array_equal(poisoned,output)),
        'tinyClosureMaxChangeUnderPointZeroOne':float(abs(almost-source).max())<.01,
        'canthiFixed':bool(np.array_equal(inverse[20,[10,70]],identity[20,[10,70]])),
        'centerTargetHasRequestedScale':bool(abs(scale[20,40]-np.exp(-.045))<1e-12),
        'sourceExposureActuallyMoves':bool(sample(estimate_layers['exposure'],inverse).sum()<estimate_layers['exposure'].sum()),
    }
    report={'tests':tests,'accepted':all(tests.values()),'sourceLashExposureMaxError':lash_error,
        'targetLashRgbMaxError':output_error,'visibleIrisRgbMaxError':visible_error,
        'tinyChangeMaximum':float(abs(almost-source).max()),
        'targetUpperCenterPixelY':float(20-8*np.exp(-.045)),
        'sourceUpperCenterPixelY':12.,'composition':metrics,
        'scope':'known-layer iris-limbus and normal-texture counterexample' if limbus else
                'known-layer constant-normal sclera control; RGB accuracy alone cannot identify E'}
    png(directory/'candidate-minus15.png',output);png(directory/'oracle-minus15.png',truth)
    sheet(directory/'comparison.png',[('source',source),('known-layer target',truth),('source-estimated target',output)])
    write(directory/'report.json',report)
    print('synthetic',json.dumps(report))


def photos():
    for name in ['user','dark']:
        localization=BASE/'source-curve-localization'/name
        source_input=json.loads((localization/'leftEye/input.json').read_text())
        source_path=Path(source_input['source'])
        if sha(source_path)!=source_input['sourceSHA256']:raise ValueError('source changed')
        source=np.asarray(Image.open(source_path).convert('RGB'),float);h,w=source.shape[:2]
        raw=Path(source_input['rawObservations'])
        if sha(raw)!=source_input['rawSHA256']:raise ValueError('frozen detector evidence changed')
        mesh=np.array([[p['x']*w,p['y']*h] for p in json.loads(raw.read_text())['rows'][0]['sourceLandmarks'][0]])
        target=source.copy();directory=OUT/name;directory.mkdir()
        field_path=BASE/'canthi-field-candidate'/name/'field-15.json.inverse.f32'
        fixed_path=BASE/'canthi-field-candidate'/name/'field-15.json.stationary.u8'
        inverse=np.fromfile(field_path,'<f4').reshape(h,w,2).astype(float)
        stationary=np.fromfile(fixed_path,'u1').reshape(h,w)!=0
        reports=[]
        for eye in ['leftEye','rightEye']:
            eye_dir=directory/eye;eye_dir.mkdir()
            state=np.load(localization/eye/'source-states.npz');x0,y0,x1,y1=state['sourceCrop']
            width,to_local,to_image,_,_,_,_=frame(mesh,EYES[eye])
            upper=PchipInterpolator(state['grid'],state['upper']);lower=PchipInterpolator(state['grid'],state['lower'])
            yy,xx=np.mgrid[y0:y1,x0:x1]
            uv=to_local(np.stack([xx+.5,yy+.5],axis=-1))
            to_pixel=lambda u,v:to_image(np.stack([u,v],axis=-1))-[x0+.5,y0+.5]
            crop=source[y0:y1,x0:x1]
            candidate=estimate(crop,uv[...,0],uv[...,1],upper,lower,to_pixel,width)
            save_layers(eye_dir,crop,candidate)
            local_inverse=inverse[y0:y1,x0:x1]-[x0,y0]
            fixed=stationary[y0:y1,x0:x1]
            row={'eye':eye,'sourceCrop':[int(x0),int(y0),int(x1),int(y1)],'sourceEstimate':candidate['report']}
            try:
                output,metrics=compose(crop,candidate['exposure'],candidate['ocular'],candidate['support'],local_inverse,fixed)
                target[y0:y1,x0:x1]=output
                png(eye_dir/'candidate-minus15.png',output)
                old=Image.open(BASE/'canthi-field-candidate'/name/'native-minus15.png').convert('RGB')
                old=np.asarray(old.crop((x0,y0,x1,y1)),float)
                sheet(eye_dir/'comparison.png',[('source',crop),('old single-plane -15',old),('color-source layers -15',output)],6)
                row.update(status='rendered; independent visual review required',composition=metrics)
            except ValueError as error:
                row.update(status='rejected',error=str(error))
            reports.append(row)
        if all(r['status'].startswith('rendered') for r in reports):png(directory/'candidate-minus15.png',target)
        write(directory/'report.json',dict(source=str(source_path),sourceSHA256=sha(source_path),
            fieldSHA256=sha(field_path),frozenAmount=-15,rows=reports,
            status='source-only estimated layers; not an optical truth or usability claim'))
        print(name,[(r['eye'],r['status'],r.get('error')) for r in reports])


def replay_existing_full_source(amount=-15):
    """Correct crop coordinate setup without rerunning or tuning estimation."""
    for name in ['user','dark']:
        directory=OUT/name
        frozen=json.loads((directory/'report.json').read_text())
        source_path=Path(frozen['source'])
        if sha(source_path)!=frozen['sourceSHA256']:raise ValueError('frozen source changed')
        source=np.asarray(Image.open(source_path).convert('RGB'),float);h,w=source.shape[:2]
        exposure=np.zeros((h,w));ocular=np.zeros_like(source);support=np.zeros((h,w),bool)
        core=np.zeros((h,w),bool);hashes={};extra=np.zeros((h,w),bool)
        for eye in ['leftEye','rightEye']:
            row=next(r for r in frozen['rows'] if r['eye']==eye)
            x0,y0,x1,y1=row['sourceCrop'];path=directory/eye/'candidate-source.npz'
            hashes[eye]=sha(path)
            footprint_path=directory/eye/'candidate-source-footprint.npz'
            if footprint_path.exists():
                data=dict(np.load(footprint_path))
            else:
                data=prepare_sampling_support(source[y0:y1,x0:x1],dict(np.load(path)))
                np.savez_compressed(footprint_path,**data)
            hashes[eye+'SamplingFootprint']=sha(footprint_path)
            exposure[y0:y1,x0:x1]=data['exposure'];ocular[y0:y1,x0:x1]=data['ocular']
            support[y0:y1,x0:x1]=data['support'];core[y0:y1,x0:x1]=data['core']
            extra[y0:y1,x0:x1]=data['addedSamplingSupport']
        field_path=BASE/'canthi-field-candidate'/name/f'field{amount}.json.inverse.f32'
        if amount==-15 and sha(field_path)!=frozen['fieldSHA256']:raise ValueError('frozen inverse field changed')
        inverse=np.fromfile(field_path,'<f4').reshape(h,w,2).astype(float)
        stationary=np.fromfile(BASE/'canthi-field-candidate'/name/f'field{amount}.json.stationary.u8','u1').reshape(h,w)!=0
        report={'sameSourceAssets':hashes,'sourceSHA256':sha(source_path),'fieldSHA256':sha(field_path),
            'sourceExposureRefit':False,'sourceOcularPriorChanged':False,'fieldModified':False,'strength':amount,
            'sourceSamplingFootprintAddedPixels':int(extra.sum()),
            'sourceSamplingFootprintRule':'one fixed 3x3 bilinear neighbor support of source E>0; retain original same-column B proposal, no target consulted'}
        try:
            output,metrics=compose(source,exposure,ocular,support,inverse,stationary)
            target_exposure=sample(exposure,inverse)
            yy,xx=np.indices((h,w));moved=np.any(inverse!=np.stack([xx,yy],axis=-1),axis=-1)&~stationary
            visible=core&(target_exposure==1)
            report.update(status='rendered; independent visual review required',composition=metrics,
                retainedCoreRgbMaximum=float(abs(output[visible]-source[visible]).max()),
                sourceZeroExposureBecomesPositivePixels=int((moved&(exposure==0)&(target_exposure>1e-8)).sum()),
                exposureIncreasePixels=int((moved&(target_exposure>exposure+1e-8)).sum()),
                exposureIncreaseMaximum=float(np.max(target_exposure-exposure)),
                extraSamplingSupportUsedPixels=int((moved&extra&(target_exposure>1e-8)).sum()),
                extraSamplingSupportExposureMaximum=float(np.max(target_exposure[extra])) if extra.any() else 0.,
                interpretation='lashes can expose their former underlay even during lid closure; these samples are estimated, never observed truth')
            png(directory/f'candidate-minus{abs(amount)}.png',output)
            old=np.asarray(Image.open(BASE/'canthi-field-candidate'/name/f'native-minus{abs(amount)}.png').convert('RGB'),float)
            for row in frozen['rows']:
                eye=row['eye'];x0,y0,x1,y1=row['sourceCrop']
                crop=output[y0:y1,x0:x1]
                png(directory/eye/f'candidate-minus{abs(amount)}.png',crop)
                sheet(directory/eye/('comparison.png' if amount==-15 else f'comparison-minus{abs(amount)}.png'),
                    [('source',source[y0:y1,x0:x1]),(f'old single-plane {amount}',old[y0:y1,x0:x1]),
                     (f'color-source layers {amount}',crop)],6)
        except ValueError as error:
            report.update(status='rejected',error=str(error))
        write(directory/('full-source-report.json' if amount==-15 else f'full-source-report-minus{abs(amount)}.json'),report)
        print(name,json.dumps(report))


def main():
    if '--limbus-and-minus30' in sys.argv:
        synthetic(limbus=True);replay_existing_full_source(amount=-30);return
    if '--replay-existing' in sys.argv:
        replay_existing_full_source();return
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'protocol.json',dict(sourceOnly=True,newModels=False,newDetection=False,
        previousMEMatteRead=False,editedOutputUsedForEstimation=False,sourceBand=3.,
        fixedStrength=-15,parameterSweep=False,
        sourceMethod='same-normal interior median ocular prior, outside skin, local edge dark color; simplex RGB weights with source-noise weak geometric exposure prior',
        differenceFromOld='fit exposure from source appearance; do not fix geometric area and project foreground, do not project ocular RGB to preserve rejected MEMatte alpha',
        limitations=['same-normal local ocular continuity is an estimate','dark lash over dark iris is explicitly ambiguous','geometric core/support gates remain conservative engineering assumptions'],
        codeSHA256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('source_estimator.py')]}))
    synthetic();photos();replay_existing_full_source()


if __name__=='__main__':main()
