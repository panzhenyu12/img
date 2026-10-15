from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'));from run_real import probe,sha,mask_save
from compose_real import montage

def q(a,d):
 v=a[d];return {'n':int(v.size),'mean':float(v.mean()) if v.size else None,'min':float(v.min()) if v.size else None,'p05':float(np.percentile(v,5)) if v.size else None,'p50':float(np.percentile(v,50)) if v.size else None,'p95':float(np.percentile(v,95)) if v.size else None,'lt05':int((v<.05).sum()),'lt5':int((v<.5).sum()),'ge9':int((v>=.9).sum())}
def main():
 d=HERE/'outputs-cpu';sam=np.load(d/'alpha-raw.npy');data=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/full-point/alphas-full.npy') # user for same ZIM source
 cases={'user-face-contour':(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png',data[2],ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/crop-point/alphas-full.npy',None)}
 # portrait source paths and graph semantic only for coverage-independent alpha comparison
 for name,source,fullp,cropp in [
 ('portrait-woman',ROOT/'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png',ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/portrait-woman/full-point/alphas-full.npy',ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/portrait-woman/crop-point/alphas-full.npy'),
 ('portrait-hair',ROOT/'test/results/face-core-review/portrait-hair-source.png',ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/portrait-hair/full-point/alphas-full.npy',ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/portrait-hair/crop-point/alphas-full.npy')]:
  f=np.load(fullp)[2];c=np.load(cropp)[2];report={'name':name,'sourceSha256':sha(source),'fullAlphaSha256':sha(fullp),'cropAlphaSha256':sha(cropp),'fullCropAbsDiff':{'mean':float(np.abs(f-c).mean()),'p95':float(np.percentile(np.abs(f-c),95)),'gt01':int((np.abs(f-c)>.1).sum()),'gt05':int((np.abs(f-c)>.5).sum())},'full':q(f,f>.5),'crop':q(c,c>.5),'allFull':q(f,np.ones_like(f,bool)),'allCrop':q(c,np.ones_like(c,bool))};print(json.dumps(report));(HERE/'outputs-cpu'/f'{name}-zim-seed-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 # save user full/crop difference
 c=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/crop-point/alphas-full.npy')[2];r={'name':'user-face-contour','sourceSha256':sha(cases['user-face-contour'][0]),'fullAlphaSha256':sha(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/full-point/alphas-full.npy'),'cropAlphaSha256':sha(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/crop-point/alphas-full.npy'),'fullCropAbsDiff':{'mean':float(np.abs(data[2]-c).mean()),'p95':float(np.percentile(np.abs(data[2]-c),95)),'gt01':int((np.abs(data[2]-c)>.1).sum()),'gt05':int((np.abs(data[2]-c)>.5).sum())},'fullAll':q(data[2],np.ones_like(data[2],bool)),'cropAll':q(c,np.ones_like(c,bool))};(d/'user-zim-seed-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
