"""Compare exact known-color boundaries with all V2 ownership/alpha/geometry frozen."""
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

def main():
 out=HERE/'outputs';out.mkdir(exist_ok=False)
 binary=ROOT/'.run/head-known-color-boundary-v1-target/release/image-render';frozen=out/('image-render-'+sha(binary));shutil.copy2(binary,frozen)
 source=Image.open(SOURCE).convert('RGBA');sa=np.asarray(source)
 report={'accepted':False,'source':str(SOURCE.relative_to(ROOT)),'sourceSha256':sha(SOURCE),'binarySha256':sha(frozen),'rows':[]}
 for mode in ['final','iterative']:
  for amount in [0,30,-30]:
   d=out/f'{mode}-{amount}';d.mkdir();ref=V2/f'natural-{amount}';graph=ref/'graph.json';shutil.copy2(graph,d/'graph.json')
   env={**os.environ,'WHOLE_FACE_DEBUG_DIR':str(d/'debug')}
   env.pop('KNOWN_COLOR_FINAL_ONLY',None)
   if mode=='final':env['KNOWN_COLOR_FINAL_ONLY']='1'
   start=time.perf_counter();r=subprocess.run([str(frozen),'--input',str(SOURCE),'--graph',str(d/'graph.json'),'--output',str(d/'candidate.png')],env=env,capture_output=True)
   (d/'render.log').write_bytes(r.stdout+r.stderr);r.check_returncode();elapsed=(time.perf_counter()-start)*1000
   candidate=Image.open(d/'candidate.png').convert('RGBA');ca=np.asarray(candidate)
   assert np.array_equal(sa[...,3],ca[...,3])
   row={'mode':mode,'amount':amount,'elapsedMs':elapsed,'graphSha256':sha(graph),'outputSha256':sha(d/'candidate.png'),'outputAlphaChanged':0}
   if not amount:assert np.array_equal(sa,ca)
   else:
    debug=d/'debug';old=ref/'debug';left,top,w,h=json.loads((debug/'layers.json').read_text())['crop']
    for name in ['moving-physical-f32le.bin','source-xy-f32le.bin','target-alpha-f32le.bin','flow.json','changed.png','frozen.png','holes.png']:
     assert (debug/name).read_bytes()==(old/name).read_bytes(),f'geometry/alpha changed: {name}'
    alpha=np.fromfile(debug/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
    crop=sa[top:top+h,left:left+w,:3]
    f=np.asarray(Image.open(debug/'foreground.png'))[...,:3];b=np.asarray(Image.open(debug/'estimated-background.png'))[...,:3]
    f0=np.asarray(Image.open(old/'foreground.png'))[...,:3];b0=np.asarray(Image.open(old/'estimated-background.png'))[...,:3]
    row['knownForegroundPixels']=int((alpha==1).sum());row['knownBackgroundPixels']=int((alpha==0).sum())
    for label,pixels,oldpixels,mask in [('foreground',f,f0,alpha==1),('background',b,b0,alpha==0)]:
     error=np.abs(pixels.astype(int)-crop.astype(int))[mask];old_error=np.abs(oldpixels.astype(int)-crop.astype(int))[mask]
     assert int(error.max())==0
     row[label]={'beforeMaxError':int(old_error.max()),'beforeMeanError':float(old_error.mean()),'afterMaxError':0}
    resid=np.max(np.abs(crop.astype(float)-alpha[...,None]*f-(1-alpha[...,None])*b),axis=2)
    oldresid=np.max(np.abs(crop.astype(float)-alpha[...,None]*f0-(1-alpha[...,None])*b0),axis=2)
    mixed=(alpha>0)&(alpha<1)
    row['mixedResidual']={'pixels':int(mixed.sum()),'beforeMean':float(oldresid[mixed].mean()),'afterMean':float(resid[mixed].mean()),'beforeMax':float(oldresid[mixed].max()),'afterMax':float(resid[mixed].max())}
    for label,box in [('face',[290,150,733,662]),('left',[300,270,450,570]),('right',[565,270,715,570]),('upper-ear',[310,275,400,410])]:
     montage(d/f'{label}-1x.png',[source,Image.open(ref/'candidate.png'),candidate],['Source','V2 scoped owner',f'Known colors {mode} {amount}'],box)
   report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
 for amount in [30,-30]:
  montage(out/f'final-vs-iterative-{amount}-left-1x.png',[source,Image.open(out/f'final-{amount}/candidate.png'),Image.open(out/f'iterative-{amount}/candidate.png')],['Source',f'Final known colors {amount}',f'Iterative known colors {amount}'],[300,270,450,570])

if __name__=='__main__':main()
