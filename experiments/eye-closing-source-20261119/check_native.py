"""Compare fresh native source preparation with the frozen Python reference.

The native source.png supplies the SAME decoded RGB to both implementations.
Source curves and layers are freshly prepared; saved fields are outputs of the
native call under review, never dependencies of the algorithm being delivered.
Image equality is a migration check, not a photographic quality rating.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/eye-height-layers-20261117'))
from layers import compose
from source_geometry import load_observation, prepare_geometry
from source_estimator import estimate, prepare_sampling_support
from source_material import load_materials, crop_materials, apply_materials, validate_material_transport


def save(path, array):
    Image.fromarray(np.rint(array).astype(np.uint8)).save(path)


def sheet(path, panels, scale=3):
    images = [(label, Image.fromarray(np.rint(a).astype(np.uint8))) for label,a in panels]
    width = sum(im.width*scale for _, im in images)
    height = max(im.height for _, im in images)*scale+24
    canvas = Image.new('RGB', (width,height), '#202124')
    draw = ImageDraw.Draw(canvas)
    x=0
    for label,im in images:
        draw.text((x+4,5),label,fill='white')
        canvas.paste(im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST),(x,24))
        x+=im.width*scale
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--materials',type=Path)
    parser.add_argument('--face-index',type=int)
    args=parser.parse_args()
    out=args.native/'python-reference'
    out.mkdir(exist_ok=False)
    native=json.loads((args.native/'report.json').read_text())
    source=np.asarray(Image.open(args.native/'source.png').convert('RGB'),float)
    h,w=source.shape[:2]
    mesh,_,_=load_observation(args.source,args.observations,(w,h),args.face_index)
    labels={}
    if args.materials:
        labels,_=load_materials(args.materials,hashlib.sha256(args.source.read_bytes()).hexdigest(),(w,h))
    exposure=np.zeros((h,w));ocular=np.zeros_like(source);support=np.zeros((h,w),bool)
    unknown=np.zeros((h,w),bool)
    automatic_e=exposure.copy();automatic_b=ocular.copy();automatic_s=support.copy()
    report={'decoder':'native source.png reused by frozen Python reference',
            'preparesSourceFromOriginalObservation':True,'readsHistoricalPreparedLayers':False,
            'sourceEyes':[],'rows':[]}
    for eye in ('leftEye','rightEye'):
        d=out/eye;d.mkdir()
        g=prepare_geometry(source,mesh,eye,d)
        p=estimate(g['crop'],g['u'],g['v'],g['upper'],g['lower'],g['to_pixel'],g['width'])
        p=prepare_sampling_support(g['crop'],p)
        x0,y0,x1,y1=g['box']
        active=p['support']
        automatic_e[y0:y1,x0:x1][active]=p['exposure'][active]
        automatic_b[y0:y1,x0:x1][active]=p['ocular'][active]
        automatic_s[y0:y1,x0:x1]|=active
        if eye in labels:
            u,v=g['u'],g['v'];safe=np.clip(u,0,g['width'])
            domain=(u>=0)&(u<=g['width'])&(v>g['upper'](safe))&(v<g['lower'](safe))
            p=apply_materials(g['crop'],p,crop_materials(labels[eye],g['box']),domain,noise=g['report']['sourceNoise'])
            unknown[y0:y1,x0:x1]|=p['materialUnknown']
        active=p['support']
        exposure[y0:y1,x0:x1][active]=p['exposure'][active]
        ocular[y0:y1,x0:x1][active]=p['ocular'][active]
        support[y0:y1,x0:x1]|=active
        ng=next(row for row in native['geometries'] if row['eye']==eye)
        grid=np.asarray(ng['knots'])
        record={'eye':eye,'crop':g['box'],'sourceReport':p['report'],
            'nativeCurveMaxPixelDifference':max(float(abs(g[side](grid)-ng[side]).max()) for side in ('upper','lower'))}
        report['sourceEyes'].append(record)
    for row in native['rows']:
        amount=row['amount'];tag='zero' if amount==0 else f'minus{abs(amount)}'
        field=np.fromfile(args.native/f'candidate-{tag}.inverse.f32','<f4').reshape(h,w,2).astype(float)
        stationary=np.fromfile(args.native/f'candidate-{tag}.stationary.u8','u1').reshape(h,w).astype(bool)
        validate_material_transport(unknown,field,stationary)
        reference,metrics=compose(source,exposure,ocular,support,field,stationary)
        automatic,_=compose(source,automatic_e,automatic_b,automatic_s,field,stationary)
        actual=np.asarray(Image.open(args.native/f'candidate-{tag}.png').convert('RGB'),float)
        save(out/f'reference-{tag}.png',reference);save(out/f'automatic-{tag}.png',automatic)
        error=abs(actual-np.rint(reference))
        record={'amount':amount,'maxRGBDifference':float(error.max()),
                'meanRGBDifference':float(error.mean()),'differingPixels':int(np.any(error,axis=-1).sum()),'referenceComposition':metrics}
        report['rows'].append(record)
        for eye in report['sourceEyes']:
            x0,y0,x1,y1=eye['crop']
            sheet(out/eye['eye']/f'comparison-{tag}.png',[
                ('original',source[y0:y1,x0:x1]),('V1 automatic',automatic[y0:y1,x0:x1]),
                ('Python explicit materials',reference[y0:y1,x0:x1]),('Rust explicit materials',actual[y0:y1,x0:x1])])
        if amount!=0:
            # Complete faces remain visible; the eye crop alone cannot accept a result.
            view=lambda a:np.asarray(Image.fromarray(np.rint(a).astype(np.uint8)).resize((round(w*min(1,640/w)),round(h*min(1,640/w)))))
            sheet(out/f'whole-{tag}.png',[('original',view(source)),('Rust',view(actual))],scale=1)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),'rows':report['rows'],
                     'curveDifferences':[{'eye':r['eye'],'max':r['nativeCurveMaxPixelDifference']} for r in report['sourceEyes']]},indent=2))


if __name__=='__main__':
    main()
