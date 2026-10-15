"""Layer-constrained LaMa ablation; never inpaints the transported face RGB."""
from pathlib import Path
import importlib.util
import json
import sys
import hashlib
import time
import argparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    options = parser.parse_args()
    base = options.evidence.resolve()
    if (base/'candidate-neural-neck.png').exists():
        raise SystemExit('Neural result already exists; preserve frozen evidence')
    debug = ROOT / 'test/results/face-material-v1-20261014/debug'
    graph = json.loads((debug / 'flow-geometry.json').read_text())
    ox, oy = graph['origin']
    layer = np.asarray(Image.open(base/'neck-layer.png').convert('RGB'))
    labels = np.asarray(Image.open(debug/'reconstruction-layer.png'))
    holes = np.asarray(Image.open(debug/'holes.png')) > 0
    frozen = np.asarray(Image.open(debug/'frozen.png')) > 0
    inferred = np.asarray(Image.open(base/'neck-target.png')) > 0
    alpha = np.asarray(Image.open(debug/'transported-physical-alpha.png')).astype(float)/255
    target = holes & ~frozen & ((labels == 100) | inferred)
    path = ROOT/'services/background-worker/worker.py'
    spec = importlib.util.spec_from_file_location('layer_refinement_worker',path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    worker = module.LamaWorker(threads=3)
    start=time.perf_counter()
    result, meta = worker.inpaint(Image.fromarray(layer).convert('RGBA'), target.astype('uint8')*255,
                                 max_side=512,context_mode='crop',context_padding=128,mask_dilation=2,seam_blend='none')
    filled=np.asarray(result)[:,:,:3].astype(float)
    result.save(base/'neural-neck-layer.png')
    initial=np.asarray(Image.open(base/'candidate-no-relight.png').convert('RGBA'))
    candidate=initial.copy();h,w=alpha.shape
    local=initial[oy:oy+h,ox:ox+w,:3].astype(float)
    prior=np.asarray(Image.open(debug/'background-before-lighting.png'))[:,:,:3].astype(float)
    # The previous amodal hypothesis replaces only inferred material pixels.
    prior[inferred]=layer[inferred]
    local[target]+=(filled-prior)[target]*(1-alpha[target,None])
    candidate[oy:oy+h,ox:ox+w,:3]=np.clip(np.rint(local),0,255).astype('uint8')
    Image.fromarray(candidate).save(base/'candidate-neural-neck.png')
    Image.fromarray((target*255).astype('uint8')).save(base/'neural-target.png')
    changed=np.any(candidate!=initial,axis=2);allowed=np.zeros(changed.shape,bool);allowed[oy:oy+h,ox:ox+w]=target
    fixed=np.zeros(changed.shape,bool);fixed[oy:oy+h,ox:ox+w]=frozen
    report={'scope':'Single existing rejected geometry; amodal layer conditioned inpaint, not proof of general quality.',
            'targetPixels':int(target.sum()),'changedPixels':int(changed.sum()),'outsideTargetChanges':int((changed&~allowed).sum()),
            'fixedChanges':int((changed&fixed).sum()),'alphaChanges':int(np.count_nonzero(candidate[:,:,3]!=initial[:,:,3])),
            'wallMs':(time.perf_counter()-start)*1000,'metadata':meta,
            'sourceSha256':hashlib.sha256((base/'candidate-no-relight.png').read_bytes()).hexdigest(),
            'outputSha256':hashlib.sha256((base/'candidate-neural-neck.png').read_bytes()).hexdigest()}
    (base/'neural-report.json').write_text(json.dumps(report,indent=2))
    box=(355,470,668,635);ims=[Image.fromarray(im).convert('RGB').crop(box) for im in [initial,candidate]];w,h=ims[0].size
    canvas=Image.new('RGB',(w*4,h*2+28),'#222');draw=ImageDraw.Draw(canvas)
    for j,(im,title) in enumerate(zip(ims,['Observed-neck hypothesis','Neck-layer inpaint'])):
        canvas.paste(im.resize((w*2,h*2),Image.Resampling.NEAREST),(j*w*2,28));draw.text((j*w*2+8,8),title,fill='white')
    canvas.save(base/'neural-comparison.png');print(json.dumps(report))


if __name__=='__main__':
    main()
