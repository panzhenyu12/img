"""Isolated frozen-mask A/B, including mean-motion matched legacy controls."""
import os,sys,json,hashlib,subprocess,time,copy,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from review_body_refinement import raster

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trace_metrics(p):
 d=json.loads(p.read_text());out=[]
 for profile in d:
  for side in range(2):
   vals=[(s['t'],s['sides'][side]['motion']) for s in profile['sections'] if s['sides'][side]['source_radius']>1]
   active=[abs(v) for t,v in vals if .22<=t<=.875]
   out.append({'side':side,'mean_abs_shaft_motion':float(np.mean(active)) if active else 0,'max_abs_motion':max([abs(v) for t,v in vals],default=0),'motion_by_section':vals})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,default=ROOT/'test/results/body-parts-v14');ap.add_argument('--samples',default='fashion-standing,fullbody-skirt,fullbody-side-fashion,tilted-shirt,action-suit');ap.add_argument('--amounts',default='-100,-30,30,100');ap.add_argument('--label',default='frozen-five');args=ap.parse_args()
 base=EXP/'renders'/args.label;base.mkdir(parents=True,exist_ok=True)
 bins={k:EXP/'bin'/f'{k}-image-render' for k in ['baseline','candidate']};hashes={k:sha(p) for k,p in bins.items()};rows=[]
 def render(key,source,graph,out,trace):
  assert sha(bins[key])==hashes[key]
  start=time.perf_counter();subprocess.run([str(bins[key]),'--input',str(source),'--graph',str(graph),'--output',str(out)],env=dict(os.environ,BODY_CURVE_TRACE=str(trace)),check=True,capture_output=True)
  assert sha(bins[key])==hashes[key]
  return round((time.perf_counter()-start)*1000,2)
 for ident in args.samples.split(','):
  folder=args.cache/ident;source=folder/'source.png';original=Image.open(source).convert('RGBA');pix=np.asarray(original);w,h=original.size
  for gp in sorted(folder.glob('*.json')):
   g=json.loads(gp.read_text())
   if not isinstance(g,dict) or 'operations' not in g:continue
   params=g['operations'][0]['params'];part=params['bodyPart'];reg=params['region'];prot=raster(params['protectMask'],w,h)>1e-6;subject=raster(params['subjectMask'],w,h);part_mask=raster(params.get('partMask',params['subjectMask']),w,h)
   yy,xx=np.indices((h,w));outside=((xx+.5)/w<reg['x'])|((xx+.5)/w>reg['x']+reg['width'])|((yy+.5)/h<reg['y'])|((yy+.5)/h>reg['y']+reg['height'])
   box=(max(0,int(reg['x']*w)-20),max(0,int(reg['y']*h)-20),min(w,int((reg['x']+reg['width'])*w)+20),min(h,int((reg['y']+reg['height'])*h)+20))
   for amount in map(int,args.amounts.split(',')):
    stem=f'{ident}-{part}-{amount}';graph=copy.deepcopy(g);graph['operations'][0]['params']['adjustments']={k:amount for k in params['adjustments']};path=base/f'{stem}.json';path.write_text(json.dumps(graph,separators=(',',':')))
    row={'sample':ident,'part':part,'amount':amount,'source':str(source),'source_sha256':sha(source),'cache_graph_sha256':sha(gp),'graph_sha256':sha(path),'crop':box};images={'original':original}
    for key in bins:
     out=base/f'{stem}-{key}.png';trace=base/f'{stem}-{key}-curve.json';row[key+'_ms']=render(key,source,path,out,trace);im=Image.open(out).convert('RGBA');p=np.asarray(im);changed=np.any(p!=pix,axis=2)
     metrics={'protected_changes':int(changed[prot].sum()),'outside_region_changes':int(changed[outside].sum()),'known_background_changes':int(changed[subject<.01].sum()) if amount>=0 else None,'outside_part_changes':int(changed[part_mask<.01].sum()) if amount>=0 else None,'alpha_changes':int((p[:,:,3]!=pix[:,:,3]).sum()),'changed_pixels':int(changed.sum()),'output_sha256':sha(out),'curve':trace_metrics(trace)}
     for k in ['protected_changes','outside_region_changes','known_background_changes','outside_part_changes','alpha_changes']:
      assert metrics[k] in [0,None],(stem,key,k,metrics[k])
     row[key]=metrics;images[key]=im
    row['changed_vs_baseline']=int(np.any(np.asarray(images['baseline'])!=np.asarray(images['candidate']),axis=2).sum())
    if not part.endswith('thigh'):assert row['changed_vs_baseline']==0,(stem,'unrelated part changed')
    if part.endswith('thigh'):
     baseline_mean=float(np.mean([c['mean_abs_shaft_motion'] for c in row['baseline']['curve']]))
     candidate_mean=float(np.mean([c['mean_abs_shaft_motion'] for c in row['candidate']['curve']]))
     matched_amount=amount*candidate_mean/max(baseline_mean,1e-7)
     for step in range(3):
      mg=copy.deepcopy(graph);mg['operations'][0]['params']['adjustments']={k:matched_amount for k in params['adjustments']};mp=base/f'{stem}-matched.json';mp.write_text(json.dumps(mg,separators=(',',':')));mo=base/f'{stem}-matched.png';mt=base/f'{stem}-matched-curve.json';render('baseline',source,mp,mo,mt)
      mm=trace_metrics(mt);actual=float(np.mean([c['mean_abs_shaft_motion'] for c in mm]))
      if abs(actual-candidate_mean)<=.02:break
      matched_amount*=candidate_mean/max(actual,1e-7)
     row['matched_legacy']={'amount':matched_amount,'mean_abs_shaft_motion':actual,'candidate_mean_abs_shaft_motion':candidate_mean,'absolute_mismatch':abs(actual-candidate_mean),'curve':mm,'output_sha256':sha(mo)};images['matched']=Image.open(mo).convert('RGBA')
    keys=list(images);cw,ch=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(cw*len(keys),ch+28),'#151b23');draw=ImageDraw.Draw(sheet)
    for i,key in enumerate(keys):sheet.paste(images[key].convert('RGB').crop(box),(i*cw,28));draw.text((i*cw+6,7),f'{key} / {amount}',fill='white')
    sheet.save(base/f'{stem}-compare.png');row['comparison']=str(base/f'{stem}-compare.png');rows.append(row)
   print(ident,part,flush=True)
 report={'binary_sha256':hashes,'original_body_sha256':sha(EXP/'baseline-body.rs'),'candidate_body_sha256':sha(EXP/'src/body.rs'),'amounts':args.amounts,'rows':rows,'notes':['Original-size crops; no redetection.','Curve motion is inverse-map boundary solving after selection and protection.','Matched legacy minimizes mean absolute shaft motion at t=.22.. .875 across available sides; individual sections differ.','Photographic alpha=255 is checked; transparent inputs covered by Rust tests.','Other-person locality is checked only where original target masks exclude them; no multi-person claim without a multi-person sample.']}
 (base/'report.json').write_text(json.dumps(report,indent=2));print('DONE',len(rows),base,flush=True)
if __name__=='__main__':main()
