"""Measure accessory transport; affine fit is diagnostic, not a beauty acceptance."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
OUT=HERE/'outputs-scoped-owner/user-face-contour'
params=json.loads((OUT/'natural-30/graph.json').read_text())['operations'][0]['params']
accessory=probe.zim.coverage(params['semanticMasks']['accessories'],1024,1024)
report={'accepted':False,'method':'source-to-destination least-squares affine fit inside transported accessory alpha >0.8; singular values and non-affine residual are descriptive, not appearance scores','rows':[]}
for amount in [-30,30]:
 d=OUT/f'natural-{amount}/debug';left,top,w,h=json.loads((d/'layers.json').read_text())['crop']
 xy=np.fromfile(d/'source-xy-f32le.bin',dtype='<f4').reshape(h,w,2)
 changed=np.asarray(Image.open(d/'changed.png'))>0
 y,x=np.mgrid[:h,:w];dst=np.stack([x+left+.5,y+top+.5],axis=-1)
 transported=map_coordinates(accessory,[xy[...,1]-.5,xy[...,0]-.5],order=1,mode='constant')
 for name,select in [('left',xy[...,0]<512),('right',xy[...,0]>512)]:
  domain=changed&select&(transported>.8)
  source=xy[domain];target=dst[domain]
  design=np.column_stack([source,np.ones(len(source))]);coef=np.linalg.lstsq(design,target,rcond=None)[0]
  predicted=design@coef;residual=np.linalg.norm(predicted-target,axis=1)
  singular=np.linalg.svd(coef[:2].T,compute_uv=False)
  report['rows'].append({'amount':amount,'side':name,'pixels':len(source),'translationMedianPx':np.median(target-source,axis=0).tolist(),'affinePrincipalScales':singular.tolist(),'nonAffineResidualRmsPx':float(np.sqrt(np.mean(residual**2))),'nonAffineResidualP95Px':float(np.quantile(residual,.95))})
(HERE/'jewelry-transport-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
