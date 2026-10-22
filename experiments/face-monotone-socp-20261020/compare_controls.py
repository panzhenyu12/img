#!/usr/bin/env python3
"""Same-intent original/final comparison using converged damped inverse solves."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from prepare_bounded_face import raw_controls
ALPHA=2/(.52+1.8)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
 meta=json.loads((path/'photo-trace/metadata.json').read_text());field=np.fromfile(path/'photo-trace/inverse.f32',dtype='<f4').reshape(meta['height'],meta['width'],2).astype(float);origin=np.array([meta['left'],meta['top']])
 def sample(points):
  uv=points-origin
  return np.stack([map_coordinates(field[:,:,c],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0.) for c in range(2)],axis=-1)
 return sample

def forward(sample,source):
 q=source.copy()
 for _ in range(160):q-=ALPHA*(q+sample(q)-source)
 residual=float(np.linalg.norm(q+sample(q)-source,axis=1).max())
 if residual>1e-7:raise ValueError(f'forward inverse closure did not converge: {residual}')
 return q,residual

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if sha(args.baseline/'controls.json')!=sha(args.candidate/'controls.json'):raise ValueError('candidate changed original controls')
 prov=json.loads((args.baseline/'provenance.json').read_text());trace=Path(prov['control_trace']['path']);src,req=raw_controls(trace);src,req=src[:128],req[:128];delta=req-src;magnitude=np.linalg.norm(delta,axis=1);active=magnitude>.5
 report={'controls_sha256':sha(args.baseline/'controls.json'),'fixed_target_requests_preserved':True,'forward_alpha':ALPHA,'forward_contraction_upper_bound':(1.8-.52+2*.24)/(1.8+.52),'source':src.tolist(),'requested':req.tolist(),'cases':{}}
 for name,path in [('baseline',args.baseline),('candidate',args.candidate)]:
  field=json.loads((path/'field.json').read_text());controls=json.loads((path/'controls.json').read_text());positions,closure=forward(load(path),src);d=positions-src
  ratio=np.sum(delta[active]*d[active],axis=1)/magnitude[active]**2
  fixed=np.array(controls['fixed_nodes'],int);dis=np.array(field['displacement']);node_error=float(np.linalg.norm(dis[fixed],axis=1).max()) if len(fixed) else 0.
  selected=[]
  for i in [44,45,82,83]:
   selected.append({'index':i,'source':src[i].tolist(),'requested':req[i].tolist(),'actual':positions[i].tolist(),'requested_magnitude':float(magnitude[i]),'actual_magnitude':float(np.linalg.norm(d[i])),'projection_ratio':float(np.dot(delta[i],d[i])/magnitude[i]**2)})
  report['cases'][name]={'path':str(path.resolve()),'field_sha256':sha(path/'field.json'),'final_f32_sha256':sha(path/'photo-trace/inverse.f32'),'positions':positions.tolist(),'inverse_closure_max':closure,'active_projection_ratio_mean':float(np.average(ratio,weights=magnitude[active])),'active_error_rms':float(np.sqrt(np.mean(np.sum((positions[active]-req[active])**2,axis=1)))),'actual_max':float(np.linalg.norm(d,axis=1).max()),'hard_fixed_nodes_max_displacement':node_error,'coarse_safety':{k:v for k,v in field['quality'].items() if k!='cells'},'selected_cheek_controls':selected}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({name:{k:v for k,v in case.items() if k not in ['positions','coarse_safety']} for name,case in report['cases'].items()},indent=2))
if __name__=='__main__':main()
