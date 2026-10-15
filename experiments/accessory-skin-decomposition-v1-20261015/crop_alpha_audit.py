"""Input-level audit for high-resolution local accessory crops; no final compose."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe,sha

def main():
 p=ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs';r=json.loads((p/'report.json').read_text())
 source=np.asarray(Image.open(ROOT/r['source']).convert('RGB'));h,w=source.shape[:2]
 params=json.loads((ROOT/r['graph']).read_text())['operations'][0]['params'];m={k:probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
 old=m['accessories'];skin=m['skin'];hair=m['hair'];ears=m['ears'];
 report={'scope':'Source alpha only. Local crop resolution is compared to frozen full image alpha; no graph/warp/compose changes.','reportSha256':sha(p/'report.json'),'rows':[]}
 for row in r['rows']:
  for x in row['rows']:
   full=np.load(p/x['directory']/'alphas-full.npy')[x['officialScoreArgmax']]
   local=np.load(p/x['directory']/'alphas-local.npy')[x['officialScoreArgmax']]
   crop=np.array(row['crop']);x0,y0,x1,y1=crop
   a=np.zeros((h,w),float);a[y0:y1,x0:x1]=local
   core=old>=.5; skin_core=skin>=.9; hair_core=hair>=.5
   # detector measures only; semantic alpha is not ground truth.
   metrics={'fullForegroundPixels':int((full>.5).sum()),'localForegroundPixels':int((a>.5).sum()),'fullAccessoryCoreRetention':float((full[core]>.5).mean()),'localAccessoryCoreRetention':float((a[core]>.5).mean()),'fullSkinCoreLeak':float((full[skin_core]>.5).mean()),'localSkinCoreLeak':float((a[skin_core]>.5).mean()),'localHairCoreOverlap':float((a[hair_core]>.5).mean()),'changedAlphaPixelsInBox':int((np.abs(full-a)>.05)[y0:y1,x0:x1].sum()),'localVsFullMeanAbs':float(np.abs(full-a)[y0:y1,x0:x1].mean())}
   report['rows'].append({'component':row['component'],'scale':row['scale'],'mode':x['mode'],'directory':x['directory'],'officialScoreArgmax':x['officialScoreArgmax'],'metrics':metrics})
 (HERE/'crop-alpha-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'rows':len(report['rows']),'status':'passed'}))
if __name__=='__main__':main()
