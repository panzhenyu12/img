"""Freeze V2 source admission once, then replay existing -15/-30 fields."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator

from probe import ROOT,BASE,sha,write,png,sheet,save_layers
from observed_ocular import estimate_observed_ocular
from source_estimator import prepare_sampling_support
sys.path.insert(0,str(ROOT/'experiments/eye-height-layers-20261117'))
from layers import compose,sample
from source_curves import frame,EYES

OUT=ROOT/'test/results/eye-closing-source-20261119/v2-observed-iris'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'protocol.json',dict(sourceOnly=True,sourceEstimator='observed_ocular.py',
        codeSHA256=sha(Path(__file__).with_name('observed_ocular.py')),
        fixedStrengths=[-15,-30],newModels=False,newDetections=False,sourcePreparedOnce=True,
        sourceThresholdsNotTunedToOutput=True,sourceImages=['user','dark']))
    for name in ['user','dark']:
        localization=BASE/'source-curve-localization'/name
        source_input=json.loads((localization/'leftEye/input.json').read_text())
        source_path=Path(source_input['source'])
        if sha(source_path)!=source_input['sourceSHA256']:raise ValueError('source changed')
        source=np.asarray(Image.open(source_path).convert('RGB'),float);h,w=source.shape[:2]
        raw=Path(source_input['rawObservations'])
        if sha(raw)!=source_input['rawSHA256']:raise ValueError('frozen observations changed')
        mesh=np.array([[p['x']*w,p['y']*h] for p in json.loads(raw.read_text())['rows'][0]['sourceLandmarks'][0]])
        directory=OUT/name;directory.mkdir()
        exposure=np.zeros((h,w));ocular=np.zeros_like(source);support=np.zeros((h,w),bool);core=np.zeros((h,w),bool)
        eyes=[]
        for eye in ['leftEye','rightEye']:
            eye_dir=directory/eye;eye_dir.mkdir()
            state=np.load(localization/eye/'source-states.npz');x0,y0,x1,y1=state['sourceCrop']
            width,to_local,to_image,_,_,iris_center,iris_radii=frame(mesh,EYES[eye])
            upper=PchipInterpolator(state['grid'],state['upper']);lower=PchipInterpolator(state['grid'],state['lower'])
            yy,xx=np.mgrid[y0:y1,x0:x1];uv=to_local(np.stack([xx+.5,yy+.5],axis=-1))
            to_pixel=lambda u,v:to_image(np.stack([u,v],axis=-1))-[x0+.5,y0+.5]
            crop=source[y0:y1,x0:x1]
            layers=estimate_observed_ocular(crop,uv[...,0],uv[...,1],upper,lower,to_pixel,
                width,iris_center,iris_radii)
            layers=prepare_sampling_support(crop,layers)
            save_layers(eye_dir,crop,layers)
            exposure[y0:y1,x0:x1]=layers['exposure'];ocular[y0:y1,x0:x1]=layers['ocular']
            support[y0:y1,x0:x1]=layers['support'];core[y0:y1,x0:x1]=layers['core']
            png(eye_dir/'iris-anchors.png',layers['observedIris']*255)
            png(eye_dir/'lash-veto.png',layers['lashVeto']*255)
            eyes.append(dict(eye=eye,crop=list(map(int,[x0,y0,x1,y1])),sourceReport=layers['report'],
                sourceLayersSHA256=sha(eye_dir/'candidate-source.npz')))
        yy,xx=np.indices((h,w));identity=np.stack([xx,yy],axis=-1)
        zero,_=compose(source,exposure,ocular,support,identity)
        rows=[]
        for amount in [-15,-30]:
            field_path=BASE/'canthi-field-candidate'/name/f'field{amount}.json.inverse.f32'
            inverse=np.fromfile(field_path,'<f4').reshape(h,w,2).astype(float)
            stationary=np.fromfile(BASE/'canthi-field-candidate'/name/f'field{amount}.json.stationary.u8','u1').reshape(h,w)!=0
            row=dict(amount=amount,fieldSHA256=sha(field_path),sourceRefitted=False,fieldModified=False,
                     zeroExact=bool(np.array_equal(source,zero)))
            try:
                output,metrics=compose(source,exposure,ocular,support,inverse,stationary)
                target_e=sample(exposure,inverse);retained=core&(target_e==1)
                tiny,_=compose(source,exposure,ocular,support,identity+(inverse-identity)*1e-5,stationary)
                row.update(status='rendered; independent visual review required',composition=metrics,
                    retainedOcularMaxRgbChange=float(abs(output[retained]-source[retained]).max()),
                    tinyClosureMaxRgbChange=float(abs(tiny-source).max()))
                png(directory/f'candidate-minus{abs(amount)}.png',output)
                previous=np.asarray(Image.open(OUT.parent/name/f'candidate-minus{abs(amount)}.png').convert('RGB'),float)
                for eye in eyes:
                    x0,y0,x1,y1=eye['crop'];crop=output[y0:y1,x0:x1]
                    png(directory/eye['eye']/f'candidate-minus{abs(amount)}.png',crop)
                    sheet(directory/eye['eye']/f'comparison-minus{abs(amount)}.png',
                        [('source',source[y0:y1,x0:x1]),(f'V1 {amount}',previous[y0:y1,x0:x1]),
                         (f'V2 observed iris {amount}',crop)],6)
                if name=='user':
                    row['fixedPreviouslyFailedPoints']=[dict(pixelIndex=[x,y],
                        sourceExposure=float(exposure[y,x]),targetExposure=float(target_e[y,x]),
                        sourceRGB=source[y,x].tolist(),outputRGB=output[y,x].tolist(),
                        maxChange=float(abs(output[y,x]-source[y,x]).max()))
                        for x,y in [(329,270),(434,271)]]
            except ValueError as error:
                row.update(status='rejected',error=str(error))
            rows.append(row)
        write(directory/'report.json',dict(source=str(source_path),sourceSHA256=sha(source_path),
            sourceEyes=eyes,rows=rows,claim='finite source material-admission experiment, not optical truth'))
        print(name,json.dumps(rows))


if __name__=='__main__':main()
