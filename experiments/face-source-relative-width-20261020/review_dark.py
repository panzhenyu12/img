"""Same-source dark portrait review; localized jewelry diagnostics are not solver inputs.

The source boxes only locate visible tiny earrings. A neutral-color subset is
an optical proxy, not a complete accessory mask or an independent truth label.
Regions were inspected after positive photos existed; this is diagnostic QA,
not a blinded evaluation, target generator, or threshold-based beauty gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/face-monotone-socp-20261020'))
from compare_controls import load,forward
from source_width_contract import read,sha,save

SOURCE_SHA='c1999306aa24f649f2aff3c078cbb5f33f060903f88b3326ab2365657336a548'
JEWELRY_BOXES={'left':(230,516,240,530),'right':(466,519,476,532)}


def samples(box):
    x0,y0,x1,y1=box;y,x=np.mgrid[y0:y1,x0:x1]
    return np.stack([x.ravel(),y.ravel()],-1).astype(float)


def material_stats(sample,p):
    q,closure=forward(sample,p);e=.01
    dx=(sample(q+[e,0])-sample(q-[e,0]))/(2*e)
    dy=(sample(q+[0,e])-sample(q-[0,e]))/(2*e)
    j=np.linalg.inv(np.stack([dx,dy],-1)+np.eye(2))
    singular=np.linalg.svd(j,compute_uv=False)
    theta=np.degrees(np.arctan2(j[:,1,0]-j[:,0,1],j[:,0,0]+j[:,1,1]))
    x=p-p.mean(0);y=q-q.mean(0)
    angle=np.arctan2(np.sum(x[:,0]*y[:,1]-x[:,1]*y[:,0]),np.sum(x*y))
    c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,s],[-s,c]])
    residual=np.linalg.norm(x@rotation-y,axis=1)
    return {'source_pixels':len(p),'source_pixel_set_sha256':__import__('hashlib').sha256(p.tobytes()).hexdigest(),
            'principal_stretch_min':float(singular.min()),'principal_stretch_max':float(singular.max()),
            'principal_stretch_medians':np.median(singular,axis=0).tolist(),
            'local_rotation_min_max_deg':[float(theta.min()),float(theta.max())],
            'rigid_fit_rotation_deg':float(np.degrees(angle)),
            'rigid_fit_rms_px':float(np.sqrt(np.mean(residual**2))),'rigid_fit_max_px':float(residual.max()),
            'mean_forward_translation_px':np.mean(q-p,axis=0).tolist(),'closure_max_px':closure}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['case30','case60','case-minus60']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cases=[a.case30.resolve(),a.case60.resolve(),a.case_minus60.resolve()];amounts=[30.,60.,-60.]
    provenance=[read(c/'provenance.json') for c in cases]
    source=Path(provenance[0]['feature_transport']['source']);assert sha(source)==SOURCE_SHA
    assert all(p['feature_transport']['source_sha256']==SOURCE_SHA for p in provenance)
    assert [p['source_relative_width']['amount'] for p in provenance]==amounts
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    views=[Image.open(f).convert('RGB') for f in [source,*[c/'photo.png' for c in cases]]]
    labels=['Source','Source-relative +30','Source-relative +60','Source-relative -60']
    crops=[('full',(0,0,683,1024),1),('face',(205,325,500,665),2),
           ('left-ear',(205,430,285,555),4),('right-ear',(430,430,515,555),4),
           ('nose',(290,415,420,535),4),('neck',(225,560,480,765),3),
           ('hairline',(195,280,515,410),3),('background-left',(160,390,250,675),3),
           ('left-earring',(215,505,265,545),10),('right-earring',(450,505,495,545),10)]
    for name,box,scale in crops:
        w,h=(box[2]-box[0])*scale,(box[3]-box[1])*scale
        panel=Image.new('RGB',(w*len(views),h+24),'white');draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(views,labels)):
            panel.paste(im.crop(box).resize((w,h),Image.Resampling.NEAREST),(i*w,24))
            draw.text((i*w+5,5),label,fill='black')
        panel.save(out/(name+'.png'))
    rgb=np.asarray(views[0],int);source_sets={};overlay=views[0].copy();d=ImageDraw.Draw(overlay)
    for name,box in JEWELRY_BOXES.items():
        p=samples(box);colors=rgb[p[:,1].astype(int),p[:,0].astype(int)]
        neutral=(np.ptp(colors,axis=1)<=25)&(colors.mean(axis=1)>=50)
        source_sets[name]={'all_localization_pixels':p,'neutral_visible_proxy':p[neutral]}
        for x,y in p[neutral].astype(int):overlay.putpixel((x,y),(0,255,255))
        d.rectangle([box[0]-1,box[1]-1,box[2],box[3]],outline='yellow')
    overlay.crop((215,500,490,540)).resize((1650,240),Image.Resampling.NEAREST).save(out/'source-earring-evidence.png')
    reports=[]
    for case,amount in zip(cases,amounts):
        sampler=load(case);audit=read(case/'feature-audit.json');eye=read(case/'image-material-audit.json')
        reports.append({'case':str(case),'amount':amount,
            'request':{k:v for k,v in audit['cases']['free-eyebrows'].items() if not isinstance(v,list)},
            'complete_support_missing':[g['missing_swept_support_coefficients'] for g in audit['groups']],
            'eye_rgb_max':[g['translation_rgb_oracle_max'] for g in eye['shape_translations']],
            'final_f32_geometry':read(case/'reconstruction-qc.json')['final_float32_pixel_reconstruction'],
            'localized_jewelry_diagnostics':{name:{kind:material_stats(sampler,points) for kind,points in groups.items()}
                                               for name,groups in source_sets.items()}})
    save(out/'review.json',{'source':{'path':str(source),'sha256':SOURCE_SHA},'crop_specs':crops,
        'inputs':[{n:{'path':str(c/n),'sha256':sha(c/n)} for n in ['controls.json','provenance.json','field.json',
                'photo-trace/inverse.f32','photo.png','feature-audit.json','image-material-audit.json','reconstruction-qc.json']}
                  for c in cases],'jewelry_source_boxes':JEWELRY_BOXES,
        'neutral_source_proxy':'Within source boxes: RGB range <=25 and mean >=50; no morphology or output-dependent tuning.',
        'limitations':['Localized QA boxes include nonmetal pixels; neutral-color proxy does not cover all earring material.',
            'Source regions chosen during diagnostic inspection after positive results; not blinded.',
            'No jewelry constraints or annotations passed to generation; measurements do not certify segmentation.',
            'No registered background lines on this source; no background-preservation guarantee.',
            'No aesthetic score or eight-point acceptance.'],'reports':reports})
    print(json.dumps([{k:v for k,v in r.items() if k!='final_f32_geometry'} for r in reports],indent=2))


if __name__=='__main__':main()
