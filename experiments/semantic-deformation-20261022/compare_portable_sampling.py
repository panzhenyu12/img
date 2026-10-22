"""Compare reconstruction of a frozen portable jaw field; never re-solve."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run_portable_jaw import read, save, sha
from resampling import render_frozen_field


def compare(case, output):
    if output.exists(): raise FileExistsError('refusing to overwrite sampling comparison')
    decision=read(case/'decision.json');protocol=read(case.parent/'protocol.json')
    source_path=Path(read(case.parent/'report.json')['source'])
    if decision.get('image_contract_pass') is not True or not decision['final_geometry']['passed_strict']:
        raise ValueError('a passing frozen image contract is required')
    paths=[source_path,case/'photo.png',case/'inverse-field.npy',case/'decision.json',
           case.parent/'protocol.json',case.parent/'fixed-source.png',Path(__file__).resolve(),
           Path(__file__).with_name('resampling.py').resolve()]
    hashes={str(p):sha(p) for p in paths}
    source=np.asarray(Image.open(source_path).convert('RGBA'))
    inverse=np.load(case/'inverse-field.npy');fixed=np.asarray(Image.open(case.parent/'fixed-source.png'))>0
    crop=protocol['crop']
    baseline,baseline_report=render_frozen_field(source,inverse,crop,method='bilinear')
    original_output=np.asarray(Image.open(case/'photo.png').convert('RGBA'))
    if not np.array_equal(baseline,original_output):
        raise ValueError('frozen bilinear result was not reproduced byte for byte')
    candidate,report=render_frozen_field(source,inverse,crop,method='bounded_lanczos3')
    if np.any(candidate[fixed]!=source[fixed]):raise ValueError('fixed source pixels changed')
    if not np.array_equal(candidate[:,:,3],baseline[:,:,3]):raise ValueError('alpha changed')
    output.mkdir(parents=True)
    Image.fromarray(candidate).save(output/'photo.png')
    p=np.asarray(protocol['source_geometry']['contour'])
    jaw=p[protocol['source_geometry']['selected_rows']]
    image_size=np.array([source.shape[1],source.shape[0]])
    images=[Image.fromarray(a).convert('RGB') for a in [source,baseline,candidate]]
    labels=['Original','Frozen field: bilinear','Frozen field: bounded Lanczos3']
    for name,points,pad,scale in [('face',p,25,1),('jaw',jaw,20,3)]:
        lo=np.maximum(np.floor(points.min(0)-pad).astype(int),0)
        hi=np.minimum(np.ceil(points.max(0)+pad).astype(int),image_size)
        box=(*lo,*hi);w,h=(hi-lo)*scale
        panel=Image.new('RGB',(int(3*w),int(h+28)),'white');draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(images,labels)):
            tile=im.crop(box)
            if scale!=1:tile=tile.resize((int(w),int(h)),Image.Resampling.NEAREST)
            panel.paste(tile,(int(i*w),28));draw.text((int(i*w+3),5),label,fill='black')
        panel.save(output/(name+'.png'))
    if any(sha(p)!=digest for p,digest in hashes.items()):raise ValueError('frozen sampling inputs changed')
    save(output/'report.json',{'inputs_sha256':hashes,'bilinear_exact_replay':True,
        'fixed_pixels_unchanged':int(fixed.sum()),'geometry_unchanged':True,'new_solve':False,
        'baseline_sampling':baseline_report,'candidate_sampling':report,'aesthetic_approval':False,
        'limits':'No high-resolution or anisotropic antialiasing claim; the 6x6 kernel has a different sampling footprint.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.case.resolve(),a.output.resolve())
