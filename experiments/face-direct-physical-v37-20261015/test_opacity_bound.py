"""A nearly absent moving layer cannot replace an opaque scene wholesale.

For opaque [0,255] layers, old/new foreground coverage each <= e bounds the
physical RGB edit by 2*255*e, plus at most one byte of quantization. This test
checks that behavioral bound on native PNGs, not a copied compositing formula.
Use --variant outputs to reproduce the pre-fix failure. Geometry and model
alpha are identical in both variants; the audited layer trace comes from delta.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCES={'user':'test/results/user-face-contour-20261014/1024/canonical-source.png','woman':'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png','hair':'test/results/face-core-review/portrait-hair-source.png'}

def audit(variant):
    rows=[]
    for name,source in SOURCES.items():
        src=np.asarray(Image.open(ROOT/source).convert('RGB')).astype(float)
        for amount in [100,-100]:
            d=HERE/'outputs-delta'/name/str(amount)/'layers'
            state=json.loads((d/'state.json').read_text());w,h=state['width'],state['height'];x,y=state['origin']
            a=np.fromfile(d/'source-alpha-f32le.bin',dtype='<f4').reshape(h,w)
            b=np.fromfile(d/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
            edited=np.fromfile(d/'changed.bin',dtype='uint8').reshape(h,w)>0
            domain=edited&(a<.002)&(b<.002)
            output=np.asarray(Image.open(HERE/variant/name/str(amount)/'candidate.png').convert('RGB')).astype(float)
            change=np.abs(output[y:y+h,x:x+w]-src[y:y+h,x:x+w]).max(2)
            bound=510*np.maximum(a,b)+1
            bad=domain&(change>bound)
            rows.append({'image':name,'amount':amount,'pixels':int(domain.sum()),'failures':int(bad.sum()),'maxRGBChange':float(change[domain].max()) if domain.any() else None})
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',default='outputs-delta');p.add_argument('--report',type=Path);a=p.parse_args();rows=audit(a.variant)
    result={'contract':__doc__,'variant':a.variant,'passed':all(r['failures']==0 for r in rows),'rows':rows}
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
