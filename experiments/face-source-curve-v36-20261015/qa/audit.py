from pathlib import Path
import hashlib,json,sys,os,subprocess,time,copy
import numpy as np
from PIL import Image,ImageDraw
Q=Path(__file__).resolve().parent;E=Q.parent;R=E.parents[1];sys.path.insert(0,str(R/'scripts'));from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();binary=next((E/'outputs').glob('image-render-*'));assert sha(binary)==binary.name.removeprefix('image-render-');v35=next((R/'experiments/face-source-curve-v35-20261015/bin').glob('image-render-*'));source_hashes={str(p.relative_to(E)):sha(p) for p in sorted((E/'src').rglob('*.rs'))};report={'binarySha256':sha(binary),'v35BinarySha256':sha(v35),'sourceHashes':source_hashes,'rows':[],'ablation':{}}
def verify():
 assert sha(binary)==report['binarySha256']
 for p,h in source_hashes.items():assert sha(E/p)==h

def analyze(name,amount,source,gp,output,old_dir):
 a=np.array(Image.open(source).convert('RGBA'));b=np.array(Image.open(output).convert('RGBA'));g=json.loads(gp.read_text());oldg=json.loads((old_dir/'graph.json').read_text());assert g==oldg;w,h=a.shape[1],a.shape[0];masks={k:raster_coverage(g['operations'][0]['params']['semanticMasks'][k],w,h)for k in ['hair','ears','accessories']};changed=np.any(a!=b,axis=2);old=np.array(Image.open(old_dir/'candidate.png').convert('RGBA'));delta=np.any(b!=old,axis=2);row={'name':name,'amount':amount,'sourceSha256':sha(source),'graphSha256':sha(gp),'graphDeepEqualV35':True,'outputSha256':sha(output),'changedPixelsVsV35':int(delta.sum()),'alphaChanges':int((a[:,:,3]!=b[:,:,3]).sum()),'zeroExact':bool(np.array_equal(a,b)),'fixed':{}}
 for k,m in masks.items():row['fixed'][k]={'confidentPixels':int((m>=.5).sum()),'confidentChanges':int(changed[m>=.5].sum()),'softEdgePixels':int(((m>1e-6)&(m<.5)).sum()),'softEdgeChanges':int(changed[(m>1e-6)&(m<.5)].sum())}
 D=output.parent
 old_debug=old_dir/('candidate-debug' if name=='portrait-hair' else 'debug')
 if amount:
  for key,file in [('sourcePhysicalAlpha','source-physical-alpha.png')]:
   p=D/'debug'/file;q=old_debug/file;row[key+'AvailableBoth']=p.exists() and q.exists()
   if p.exists() and q.exists():row[key+'ByteExactVsV35']=p.read_bytes()==q.read_bytes();row[key+'RGBAExactVsV35']=bool(np.array_equal(np.array(Image.open(p)),np.array(Image.open(q))))
  for label,directory in [('v35',old_debug),('v36',D/'debug')]:
   c=json.loads((directory/'source-width.json').read_text());x=json.loads((directory/'actual-source-width.json').read_text());row[label]={'request':c['requestedPeak'],'projectedPeaks':[max(abs(r['motion'][s])for r in c['rows']if r['valid'][s])for s in [0,1]],'actualPeaks':[max(abs(r['points'][s]['actualInward'])for r in x['rows']if r['points'][s]['valid'])for s in [0,1]]}
 report['rows'].append(row);(Q/'audit-report.json').write_text(json.dumps(report,indent=2));return row

for name in ['portrait-woman','portrait-hair','user-face-contour']:
 source=R/('experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png' if name=='portrait-woman' else 'test/results/face-core-review/portrait-hair-source.png' if name=='portrait-hair' else 'test/results/user-face-contour-20261014/1024/canonical-source.png')
 for amount in [100,-100,30,-30,0]:
  D=E/'outputs'/f'{name}-{amount}';old_dir=R/f'experiments/face-source-curve-v35-20261015/qa/base/{name}/parser/faceWidth_{amount}' if name!='portrait-hair' else R/f'experiments/face-source-curve-v35-20261015/extra-qa/portrait-hair-{amount}';analyze(name,amount,source,D/'graph.json',D/'candidate.png',old_dir)
 print('audited',name,flush=True)

# Roll back the single operative change by executing immutable V35 on the exact
# V36 woman graph and source. No candidate source is modified for this ablation.
A=Q/'ablation-woman100';A.mkdir(exist_ok=True);source=R/'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png';gp=E/'outputs/portrait-woman-100/graph.json';p=subprocess.run([str(v35),'--input',str(source),'--graph',str(gp),'--output',str(A/'v35-rollback.png')],env=dict(os.environ,FACE_V21_DEBUG_DIR=str(A/'debug'),FACE_V35_DEBUG_DIR=str(A/'debug')),capture_output=True);(A/'render.log').write_bytes(p.stdout+p.stderr);p.check_returncode();expected=R/'experiments/face-source-curve-v35-20261015/qa/base/portrait-woman/parser/faceWidth_100/candidate.png';report['ablation']={'sourceSha256':sha(source),'graphSha256':sha(gp),'v35RollbackOutputSha256':sha(A/'v35-rollback.png'),'v35PriorOutputSha256':sha(expected),'rollbackRGBAExact':bool(np.array_equal(np.array(Image.open(A/'v35-rollback.png')),np.array(Image.open(expected))))};(Q/'audit-report.json').write_text(json.dumps(report,indent=2));print('ablation',report['ablation'],flush=True)

source=R/'experiments/face-anatomical-v20-20261014/inputs/portrait-makeup-source.png';original=Image.open(source).convert('RGBA')
for amount in [30,100,-30,-100]:
 old_dir=R/f'experiments/face-source-curve-v35-20261015/qa/base/portrait-makeup/parser/faceWidth_{amount}';D=Q/f'makeup-{amount}';D.mkdir(exist_ok=True);gp=D/'graph.json';gp.write_bytes((old_dir/'graph.json').read_bytes());t=time.perf_counter();p=subprocess.run([str(binary),'--input',str(source),'--graph',str(gp),'--output',str(D/'candidate.png')],env=dict(os.environ,FACE_V21_DEBUG_DIR=str(D/'debug'),FACE_V36_DEBUG_DIR=str(D/'debug')),capture_output=True);(D/'render.log').write_bytes(p.stdout+p.stderr);p.check_returncode();verify();row=analyze('portrait-makeup',amount,source,gp,D/'candidate.png',old_dir);row['wallMs']=(time.perf_counter()-t)*1000
 old=Image.open(old_dir/'candidate.png').convert('RGBA');new=Image.open(D/'candidate.png').convert('RGBA');g=json.loads(gp.read_text());pts=g['operations'][0]['params']['semanticMasks']['faceContour']['points'];xs=[p['x']*new.width for p in pts];ys=[p['y']*new.height for p in pts];box=(int(min(xs))-50,int(min(ys))-30,int(max(xs))+51,min(new.height,int(max(ys))+120));w,h=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(3*w,h+24),'#111820');draw=ImageDraw.Draw(sheet)
 for i,(im,label)in enumerate([(original,'Original'),(old,'V35'),(new,'V36')]):sheet.paste(im.crop(box),(i*w,24));draw.text((i*w+4,6),label,fill='white')
 sheet.save(D/'face-comparison-1x.png');print('makeup',amount,row['v36'],flush=True)
verify();(Q/'audit-report.json').write_text(json.dumps(report,indent=2))
