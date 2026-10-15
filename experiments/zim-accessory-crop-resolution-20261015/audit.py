"""Native-frame audit for local crop alpha candidates against full-image baseline."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes,distance_transform_edt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=__import__('importlib').util.spec_from_file_location(name,path);m=__import__('importlib').util.module_from_spec(s);s.loader.exec_module(m);return m
probe=load('crop_audit_probe',ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/run.py');zim=probe.zim
cv2=probe.cv2
def main():
 out=HERE/'outputs';raw=json.loads((out/'report.json').read_text());source=ROOT/raw['source'];graph=ROOT/raw['graph'];rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2];params=json.loads(graph.read_text())['operations'][0]['params'];masks={k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()};skin,acc,hair,ears=(masks[k] for k in ['skin','accessories','hair','ears']);features=np.maximum.reduce([masks[k] for k in ['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']]);anchors=(skin>=.9)&(np.maximum.reduce([acc,hair,ears,features])<.05);distance,nearest=distance_transform_edt(~anchors,return_indices=True);raster=params['semanticMasks']['accessories'];cell=max(w/raster['width'],h/raster['height']);face_width=(max(p['x'] for p in params['landmarks'])-min(p['x'] for p in params['landmarks']))*w;radius=min(3*cell,.025*face_width);lab=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2LAB);kernel=2*int(np.ceil(cell))+1;count=cv2.boxFilter(anchors.astype('float32'),-1,(kernel,kernel),normalize=False);means=np.stack([cv2.boxFilter(lab[...,i]*anchors,-1,(kernel,kernel),normalize=False)/np.maximum(count,1) for i in range(3)],axis=-1);delta=lab-means[nearest[0],nearest[1]];support=((skin>=.5)|((np.linalg.norm(delta,axis=-1)<=8)&(np.abs(delta[...,0])<=6)&(np.linalg.norm(delta[...,1:],axis=-1)<=6)))&(distance<=radius)&(ears<.05)&(features<.05)
 base_report=json.loads((ROOT/'experiments/zim-accessory-combined-20261015/outputs/report.json').read_text());base={(r['prompt']['component'],r['mode']):r for r in base_report['rows']};report=dict(scope='Frozen native source ownership audit for local-resolution crop candidates; no alpha edits or semantic relabeling.',rawReportSha256=sha(out/'report.json'),scriptSha256=sha(Path(__file__)),sourceSha256=sha(source),graphSha256=sha(graph),thresholds=dict(parserCellPx=cell,maxSourceEvidenceDistancePx=radius),rows=[])
 for group in raw['rows']:
  comp=(acc>=.9);component=group['component'];comp=(probe.zim.coverage(params['semanticMasks']['accessories'],w,h)>=.9) # group component is selected below by connected label
  count,labels,stats,_=cv2.connectedComponentsWithStats((acc>=.5).astype('uint8'));comp=(labels==component);crop=group['crop'];x0,y0,x1,y1=crop;overlap=np.zeros((h,w),bool);overlap[y0:y1,x0:x1]=True;bright=comp&(rgb.astype('float32').mean(-1)>=np.quantile(rgb.astype('float32').mean(-1)[comp],.9));base_rows={m:base[(component,m)] for m in ['combined-point','combined-box']}
  for item in group['rows']:
   baseline=base_rows[item['mode']];b_alphas=np.load(ROOT/'experiments/zim-accessory-combined-20261015/outputs'/baseline['directory']/'alphas.npy',mmap_mode='r');a_full=np.load(out/item['directory']/'alphas-full.npy',mmap_mode='r');dest=out/item['directory']
   for i,a in enumerate(a_full):
    excluded=comp&(a<=.05);eligible=excluded&support;unknown=excluded&~support;holes=binary_fill_holes(a>=.5)&(a<.5)&comp;diff=np.abs(a.astype('float64')-b_alphas[i].astype('float64'));same=diff[overlap]
    crop_skin=(skin>=.9)&(hair<.05)&(acc<.05)&overlap
    report['rows'].append(dict(component=component,scale=group['scale'],mode=item['mode'],index=i,official=(i==item['officialScoreArgmax']),crop=group['crop'],sourcePixels=int(comp.sum()),excludedPixels=int(excluded.sum()),sourceSupportedExcludedPixels=int(eligible.sum()),unknownExcludedPixels=int(unknown.sum()),enclosedHoles=int(holes.sum()),enclosedHolesWithSkinEvidence=int((holes&support).sum()),brightMeanAlpha=float(a[bright].mean()),brightMinAlpha=float(a[bright].min()),componentCoreRetainedFraction=float((a[comp]>=.5).mean()),skinCoreRetainedFraction=float((a[(skin>=.9)&(hair<.05)&(acc<.05)]>=.5).mean()),cropSkinCorePixels=int(crop_skin.sum()),cropSkinCoreRetainedFraction=float((a[crop_skin]>=.5).mean()) if crop_skin.any() else None,sameCropMeanAbsAlphaDiff=float(same.mean()),sameCropMaxAbsAlphaDiff=float(same.max()),baselineOfficialIndex=baseline['officialScoreArgmax']))
 (out/'audit-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(rows=len(report['rows']),officialRows=sum(r['official'] for r in report['rows']),rawReportSha256=report['rawReportSha256'])))
if __name__=='__main__':main()
