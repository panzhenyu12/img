"""Raw whole-person physical alpha only; keep V2 targets, field, ownership, frozen domain."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import montage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
V2=ROOT/'experiments/whole-face-shape-v2-20261015/outputs-scoped-owner/user-face-contour'
SOURCE=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
RAW=HERE/'inputs/physical-alpha-f32le.bin'

def main():
 out=HERE/'outputs';out.mkdir(exist_ok=False)
 binary=ROOT/'.run/whole-person-alpha-override-v1-target/release/image-render';frozen=out/('image-render-'+sha(binary));shutil.copy2(binary,frozen)
 source=Image.open(SOURCE).convert('RGBA');sa=np.asarray(source);raw=np.fromfile(RAW,dtype='<f4').reshape(1024,1024)
 report={'accepted':False,'source':str(SOURCE.relative_to(ROOT)),'sourceSha256':sha(SOURCE),'binarySha256':sha(frozen),'rawAlphaSha256':sha(RAW),'geometryUsesModelAlpha':False,'knownColorBoundaryCorrection':False,'rows':[]}
 for mode,amount in [('control',30),('raw',0),('raw',30),('raw',-30)]:
  d=out/f'{mode}-{amount}';d.mkdir();ref=V2/f'natural-{amount}';shutil.copy2(ref/'graph.json',d/'graph.json')
  env={**os.environ,'WHOLE_FACE_DEBUG_DIR':str(d/'debug')};env.pop('WHOLE_PERSON_PHYSICAL_ALPHA_F32LE',None)
  if mode=='raw':env['WHOLE_PERSON_PHYSICAL_ALPHA_F32LE']=str(RAW)
  start=time.perf_counter();r=subprocess.run([str(frozen),'--input',str(SOURCE),'--graph',str(d/'graph.json'),'--output',str(d/'candidate.png')],env=env,capture_output=True)
  (d/'render.log').write_bytes(r.stdout+r.stderr);r.check_returncode();elapsed=(time.perf_counter()-start)*1000
  candidate=Image.open(d/'candidate.png').convert('RGBA');ca=np.asarray(candidate);v2=np.asarray(Image.open(ref/'candidate.png').convert('RGBA'))
  assert np.array_equal(sa[...,3],ca[...,3]);row={'mode':mode,'amount':amount,'elapsedMs':elapsed,'graphSha256':sha(d/'graph.json'),'outputSha256':sha(d/'candidate.png'),'outputAlphaChanged':0}
  if mode=='control':assert np.array_equal(ca,v2)
  if not amount:assert np.array_equal(ca,sa)
  else:
   debug=d/'debug';old=ref/'debug';left,top,w,h=json.loads((debug/'layers.json').read_text())['crop']
   row['geometryExactFiles']=[]
   for name in ['source-xy-f32le.bin','flow.json','requested.json','frozen.png','head-component.png','owner-support.png']:
    assert (debug/name).read_bytes()==(old/name).read_bytes(),f'geometry changed: {name}'
    row['geometryExactFiles'].append(name)
   alpha=np.fromfile(debug/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
   if mode=='raw':assert np.array_equal(alpha,raw[top:top+h,left:left+w])
   else:assert (debug/'moving-physical-f32le.bin').read_bytes()==(old/'moving-physical-f32le.bin').read_bytes()
   target=np.fromfile(debug/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
   changed=np.asarray(Image.open(debug/'changed.png'))>0;fixed=np.asarray(Image.open(debug/'frozen.png'))>0
   crop=sa[top:top+h,left:left+w];edited=ca[top:top+h,left:left+w]
   assert np.array_equal(edited[fixed],crop[fixed])
   assert np.array_equal(edited[~changed],crop[~changed])
   roi=np.zeros(sa.shape[:2],bool);roi[top:top+h,left:left+w]=True;assert np.array_equal(ca[~roi],sa[~roi])
   delta=np.max(np.abs(edited[...,:3].astype(int)-crop[...,:3].astype(int)),axis=-1)
   weak=changed&(np.maximum(alpha,target)<.01)
   row['changedPixelsVsSource']=int(np.any(sa!=ca,axis=2).sum());row['changedPixelsVsV2']=int(np.any(v2!=ca,axis=2).sum())
   row['weakCoverageChanged']={'pixels':int(weak.sum()),'maxRgbDelta':int(delta[weak].max()) if weak.any() else 0,'over8Bytes':int((weak&(delta>8)).sum())}
   f=np.asarray(Image.open(debug/'foreground.png'))[...,:3].astype(float);b=np.asarray(Image.open(debug/'estimated-background.png'))[...,:3].astype(float)
   resid=np.max(np.abs(crop[...,:3]-alpha[...,None]*f-(1-alpha[...,None])*b),axis=2)
   row['sourceReconstructionResidual']={'mean':float(resid.mean()),'max':float(resid.max()),'over8Bytes':int((resid>8).sum())}
   for label,box in [('face',[290,150,733,662]),('left',[300,270,450,570]),('right',[565,270,715,570]),('head',[287,0,730,709])]:
    montage(d/f'{label}-1x.png',[source,Image.open(ref/'candidate.png'),candidate],['Source','V2 parser physical alpha',f'ZIM raw whole person {amount}' if mode=='raw' else 'No override control'],box)
  report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)

if __name__=='__main__':main()
