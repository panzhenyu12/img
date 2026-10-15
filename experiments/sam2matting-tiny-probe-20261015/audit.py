from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha,mask_save
from compose_real import montage

def metric(a,domain):
 vals=a[domain]
 return {'pixels':int(len(vals)),'mean':float(vals.mean()) if len(vals) else None,'min':float(vals.min()) if len(vals) else None,'max':float(vals.max()) if len(vals) else None,'below05':int((vals<.5).sum()),'below09':int((vals<.9).sum()),'above005':int((vals>.05).sum()),'above05':int((vals>.5).sum())}
def main():
 d=HERE/'outputs-cpu';a=np.load(d/'alpha-raw.npy');zpath=ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/full-point/alphas-full.npy';z=np.load(zpath)[2];source=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGB'));p=json.loads((ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json').read_text())['operations'][0]['params'];m={k:probe.zim.coverage(v,1024,1024) for k,v in p['semanticMasks'].items()};layers=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');scene=np.load(ROOT/'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz')['addedScene']
 skin=binary_erosion((m['skin']>=.9)&(z>=.99)&(layers['alpha']<.01)&(m['hair']<.05)&(m['ears']<.05),iterations=2)
 forehead=np.zeros_like(skin);forehead[180:320,340:680]=True;forehead &= (z>=.99)&(m['skin']>=.5)&(m['hair']<.05)
 domains={'independentSkinCore':skin,'foreheadHighSupport':forehead,'addedObservedScene':scene,'metalCore':layers['opaque'],'ears':m['ears']>=.9}
 report={'scope':'Model output alpha audit only. No warped output, no foreground RGB estimate, no production changes. ZIM/semantic references are independent proxies, not ground truth.','sourceSha256':sha(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'),'rawAlphaSha256':sha(d/'alpha-raw.npy'),'accepted':False,'domains':{},'components':[]}
 for k,domain in domains.items():report['domains'][k]={'zim':metric(z,domain),'sam2matting':metric(a,domain)};mask_save(d/(k+'-mask.png'),domain)
 lost=skin&(a<.9);new_scene_skin=scene&(a>.05);overlay=source.copy();overlay[lost]=[255,0,0];overlay[new_scene_skin]=[0,255,255];Image.fromarray(overlay).save(d/'failure-overlay.png')
 for idx,crop in enumerate([[326,409,397,548],[620,414,692,551]],1):
  x0,y0,x1,y1=crop;cs=np.s_[y0:y1,x0:x1];band=((z>.01)&(z<.99))|((a>.01)&(a<.99));report['components'].append({'component':idx,'changedOver01':int((np.abs(z[cs]-a[cs])>.1).sum()),'mixedZim':int(((z[cs]>.01)&(z[cs]<.99)).sum()),'mixedSam2':int(((a[cs]>.01)&(a[cs]<.99)).sum()),'newSceneLeakOver005':int(new_scene_skin[cs].sum()),'metalAlphaMax':float(a[cs][layers['opaque'][cs]].max())})
  za=np.uint8(np.clip(z,0,1)*255);aa=np.uint8(np.clip(a,0,1)*255);montage(d/f'component-{idx}-alpha-1x.png',[Image.fromarray(source),Image.fromarray(za),Image.fromarray(aa),Image.fromarray(overlay)],['Source','ZIM alpha','SAM2 alpha','Red skin-loss cyan scene-leak'],crop)
 report['visualConclusion']='Rejected as a whole-domain physical skin alpha: forehead gains a broad translucent patch, cheek edge becomes softer/jagged, and known scene gains foreground alpha. No claim of improvement from model availability.'
 (d/'audit.json').write_text(json.dumps(report,indent=2)+'\n');r=json.loads((d/'report.json').read_text());r['visualAssessment']=report['visualConclusion'];(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
