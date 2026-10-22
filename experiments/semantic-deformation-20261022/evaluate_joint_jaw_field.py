"""Inspect a frozen accepted source solution and render an offline comparison.

The final float32 inverse and numerical execution must pass independently.
Photographic quality and continuous final-curve constraints remain unapproved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from portable_field import source_geometry
from forward import inverse_points
from evaluate import inspect
from portable_result_gate import evaluate_result
from resampling import render_frozen_field
from run_portable_jaw import snapshot_local_code


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(report_path,source_path,fixed_path,baseline_path,output):
    report_path,source_path,fixed_path,baseline_path,output=[Path(p).resolve() for p in
        [report_path,source_path,fixed_path,baseline_path,output]]
    if output.exists():raise FileExistsError('refusing to overwrite joint photo evidence')
    reference=read(report_path)
    if reference.get('source_continuous_accepted') is not True:
        raise ValueError('original continuous source conditions must pass before photo evaluation')
    for path in [source_path,fixed_path]:
        if reference['inputs_sha256'].get(str(path))!=sha(path):
            raise ValueError('source/fixed mask must match the frozen solved problem')
    baseline_report=read(baseline_path.parent/'decision.json')
    if baseline_report.get('rendered') is not True or baseline_report.get('amount')!=reference['amount']:
        raise ValueError('baseline must be an earlier rendered case with the same signed strength')
    baseline_protocol=read(baseline_path.parent.parent/'protocol.json')
    if baseline_protocol['inputs_sha256'].get(str(source_path))!=sha(source_path):
        raise ValueError('baseline must bind the same original source pixels')
    state_path=report_path.parent/'diagnostic-source-state.npz'
    if sha(state_path)!=reference['diagnostic_source_state_sha256']:
        raise ValueError('frozen source coefficients changed')
    state=np.load(state_path);grid=reference['grid'];spacing=np.asarray(grid['spacing'],float)
    if spacing.shape!=(2,) or spacing[0]!=spacing[1] or spacing[0]<=0:
        raise ValueError('unchanged positive isotropic source grid required')
    z,mapping=state['scalar_solution'],state['mapping']
    scalar=np.zeros(len(mapping));active=mapping>=0;scalar[active]=z[mapping[active]]
    forward=(scalar[:,None]*np.asarray(reference['source_geometry']['frame']['across'])).reshape(
        grid['height'],grid['width'],2)
    source_check=source_geometry(forward/spacing)
    if source_check['strong_monotonicity_lower_bound']<=0 or not source_check['boundary_zero']:
        raise ValueError('frozen source field is not certified for inversion')
    spans=(np.array([grid['width'],grid['height']])-1)*spacing+1
    if np.any(spans!=np.round(spans)) or np.any(np.asarray(grid['origin'])!=np.round(grid['origin'])):
        raise ValueError('original integer crop required')
    width,height=spans.astype(int);left,top=map(int,grid['origin']);crop=[left,top,int(width),int(height)]
    rgba=np.asarray(Image.open(source_path).convert('RGBA'));fixed=np.asarray(Image.open(fixed_path).convert('L'))>0
    if fixed.shape!=rgba.shape[:2] or left<0 or top<0 or left+width>rgba.shape[1] or top+height>rgba.shape[0]:
        raise ValueError('source, fixed mask and original crop dimensions differ')
    inputs={str(path):sha(path) for path in [report_path,state_path,source_path,fixed_path,baseline_path,
        baseline_path.parent/'decision.json',baseline_path.parent.parent/'protocol.json']}
    output.mkdir(parents=True);manifest,code_hashes=snapshot_local_code(output)
    started=time.perf_counter();yy,xx=np.mgrid[:height,:width]
    q=np.column_stack([xx.ravel(),yy.ravel()])/spacing
    inverse,closure=inverse_points(forward/spacing,q)
    field=((inverse-q)*spacing).reshape(height,width,2).astype('<f4')
    np.save(output/'inverse-field.npy',field)
    final=inspect(field);report={'amount':reference['amount'],'source_report':str(report_path),
        'source_conditions_accepted':True,'source_field_geometry':source_check,
        'actual_f32_inverse_geometry':final,'inverse_closure_grid_units':closure,
        'inverse_closure_max_source_px':closure['closure_max_px']*spacing[0],
        'inputs_sha256':inputs,'executed_code_manifest':str(manifest),'rendered':False,
        'image_approval':False,'continuous_final_curve_approval':False,
        'limits':'Offline numerical execution and photo review only. Sampled curve response does not certify final continuous chord conditions or naturalness.'}
    if final['passed_strict']:
        edited,sampling=render_frozen_field(rgba,field,crop,method='bilinear')
        changed=np.any(edited!=rgba,axis=2);fixed_changed=int(np.count_nonzero(changed&fixed))
        fixed_inverse=float(np.linalg.norm(field[fixed[top:top+height,left:left+width]],axis=1).max(initial=0.))
        curve_results=[]
        for row in reference['diagnostic_curves']:
            path=report_path.parent/f"diagnostic-curve-{row['index']}.npz"
            if sha(path)!=row['sha256']:raise ValueError('frozen source/target curve changed')
            inputs[str(path)]=sha(path);curve=np.load(path)
            actual,curve_closure=inverse_points(field.astype(float),curve['source']-grid['origin'])
            actual+=grid['origin']
            execution=evaluate_result(curve['source'],curve['target'],actual,amount=reference['amount'],
                fixed_changed_pixels=fixed_changed,whole_image_changed_pixels=int(changed.sum()))
            np.savez_compressed(output/f"curve-{row['index']}.npz",source=curve['source'],
                target=curve['target'],actual=actual,source_arclength_px=curve['source_arclength_px'])
            curve_results.append({'index':row['index'],'execution':execution,'closure':curve_closure,
                'maximum_target_error_px':float(np.linalg.norm(actual-curve['target'],axis=1).max())})
        report.update(fixed_changed_pixels=fixed_changed,fixed_inverse_max_px=fixed_inverse,
            changed_pixels=int(changed.sum()),curve_execution=curve_results,sampling=sampling)
        if fixed_changed==0 and fixed_inverse==0. and all(c['execution']['passed'] for c in curve_results):
            Image.fromarray(edited).save(output/'photo.png');report['rendered']=True
            contour=np.asarray(reference['source_geometry']['contour'])
            lo=np.maximum(np.floor(contour.min(0)-20).astype(int),0)
            hi=np.minimum(np.ceil(contour.max(0)+60).astype(int),[rgba.shape[1],rgba.shape[0]])
            original=Image.fromarray(rgba).convert('RGB');baseline=Image.open(baseline_path).convert('RGB')
            if baseline.size!=original.size:raise ValueError('baseline dimensions differ')
            for name,box in [('full',(0,0,*original.size)),('face',(*lo,*hi)),
                ('jaw',(int(lo[0]),max(0,int(contour[:,1].max()-125)),int(hi[0]),int(hi[1])))]:
                w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(3*w,h+28),'white');draw=ImageDraw.Draw(panel)
                for i,(im,label) in enumerate([(original,'Source'),(baseline,'Retained-bend baseline'),
                    (Image.fromarray(edited).convert('RGB'),'Joint source field; unapproved')]):
                    panel.paste(im.crop(box),(i*w,28));draw.text((i*w+3,5),label,fill='black')
                panel.save(output/(name+'.png'))
    if any(sha(path)!=digest for path,digest in {**inputs,**code_hashes}.items()):
        raise ValueError('frozen inputs or evaluator code changed during evaluation')
    report['inverse_field_sha256']=sha(output/'inverse-field.npy')
    report['elapsed_evaluation_seconds']=time.perf_counter()-started
    save(output/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ['rendered','actual_f32_inverse_geometry','fixed_changed_pixels',
        'fixed_inverse_max_px','elapsed_evaluation_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['report','source','fixed','baseline','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.report,args.source,args.fixed,args.baseline,args.output)
