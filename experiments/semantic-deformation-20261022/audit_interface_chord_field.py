"""Independent dense sampling of source-relative chords in final f32 fields.

Measures the actual 2D transported geometry, not merely the requested curve.
Two source-arclength pitches expose sampling sensitivity; this is not a proof
of continuous extrema between all samples or an aesthetic acceptance test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.interpolate import BSpline,PchipInterpolator

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'face-source-forward-prototype-20261021'))
from forward import inverse_points


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def validate_frozen_profiles(protocols,artifact_profile_sha256):
    """Validate current contents against each protocol's historical digest."""
    for protocol in protocols.values():
        path=Path(protocol['frozen_target_profile']).resolve()
        historical=protocol.get('inputs_sha256',{}).get(str(path))
        if historical!=artifact_profile_sha256 or sha(path)!=historical:
            raise ValueError('same historically frozen jaw profile required')


def normalized_local_chord_change(source,reference,candidate,axes):
    """Return local numerator change / frozen source h0*h1 for 3xNx2 rows."""
    source,reference,candidate,axes=[np.asarray(v,float) for v in [source,reference,candidate,axes]]
    if (source.ndim!=3 or source.shape[0]!=3 or source.shape[2]!=2 or not source.shape[1]
        or reference.shape!=source.shape or candidate.shape!=source.shape
        or axes.shape!=(2,2) or not all(np.isfinite(v).all() for v in [source,reference,candidate,axes])
        or not np.allclose(axes@axes.T,np.eye(2),rtol=0.,atol=1e-10)):
        raise ValueError('finite matching 3xNx2 chord points and orthonormal axes required')
    a,b,c=source@axes.T
    h0,h1=b[:,1]-a[:,1],c[:,1]-b[:,1]
    if np.any(h0<=0.) or np.any(h1<=0.):
        raise ValueError('strictly increasing frozen source heights required')
    def numerator(points):
        a,b,c=points@axes.T
        return (c[:,0]-b[:,0])*(b[:,1]-a[:,1])-(b[:,0]-a[:,0])*(c[:,1]-b[:,1])
    # The original source numerator cancels between D_candidate and D_ref.
    # Actual 2D down movements are retained; only the denominator is frozen.
    return (numerator(candidate)-numerator(reference))/(h0*h1)


def run(baseline,candidate,target_path,output,*,actual_local_change_limit=None):
    if actual_local_change_limit is not None and (
        not np.isfinite(actual_local_change_limit) or actual_local_change_limit<=0.):
        raise ValueError('positive finite actual local-change observation limit required')
    baseline,candidate,target_path,output=[Path(p).resolve() for p in [baseline,candidate,target_path,output]]
    if output.exists():raise FileExistsError('refusing to overwrite an actual-field chord audit')
    artifact=read(target_path);interface_path=Path(artifact['report']['source_interface']);interface=read(interface_path)
    protocols={name:read(case.parent/'protocol.json') for name,case in [('baseline',baseline),('candidate',candidate)]}
    decisions={name:read(case/'decision.json') for name,case in [('baseline',baseline),('candidate',candidate)]}
    if protocols['baseline']['source_geometry']!=protocols['candidate']['source_geometry']:
        raise ValueError('same source geometry required')
    if any(d['amount']!=artifact['amount'] or not d['final_geometry']['passed_strict'] for d in decisions.values()):
        raise ValueError('same signed amount and strict final geometry pass required')
    validate_frozen_profiles(protocols,artifact['jaw_profile_sha256'])
    for p,digest in artifact['inputs_sha256'].items():
        if sha(p)!=digest:raise ValueError('frozen boundary target input changed')
    hashes={str(p):sha(p) for p in [target_path,interface_path,Path(__file__).resolve()]}
    for case in [baseline,candidate]:
        for p in [case/'inverse-field.npy',case/'decision.json',case.parent/'protocol.json',case.parent/'fixed-source.png']:
            hashes[str(p)]=sha(p)
    fields={name:np.load(case/'inverse-field.npy').astype(float) for name,case in [('baseline',baseline),('candidate',candidate)]}
    fixed={name:np.asarray(Image.open(case.parent/'fixed-source.png').convert('L')) for name,case in [('baseline',baseline),('candidate',candidate)]}
    if fixed['baseline'].shape!=fixed['candidate'].shape:raise ValueError('fixed mask dimensions differ')
    changed=int(np.count_nonzero(fixed['baseline']!=fixed['candidate']))
    if changed:raise ValueError('decoded fixed mask pixels differ')
    geometry=protocols['candidate']['source_geometry'];frame=geometry['frame']
    axes=np.asarray([frame['across'],frame['down']]);across=axes[0]
    output.mkdir(parents=True)
    records=[]
    for segment,curve in zip(interface['segments'],artifact['report']['curves']):
        if segment['id']!=curve['source_segment_id']:raise ValueError('curve source ordering differs')
        p=np.asarray(segment['source_points']);knots=np.r_[0.,np.linalg.norm(np.diff(p,axis=0),axis=1).cumsum()]
        profile=curve['profile'];length=float(knots[-1])
        if not np.allclose(knots,profile['source_arclength_px'],rtol=0.,atol=1e-10):raise ValueError('source arclength changed')
        def source_at(s):return np.column_stack([np.interp(s,knots,p[:,i]) for i in range(2)])
        pref=PchipInterpolator(knots,profile['preferred_across_px'])
        correction=BSpline(profile['spline_knots_px'],profile['correction_coefficients'],3,extrapolate=False)
        active_lo,active_hi=profile['spline_knots_px'][0],profile['spline_knots_px'][-1]
        def requested(s):
            u=pref(s);inside=(s>=active_lo)&(s<=active_hi);u[inside]+=correction(s[inside])
            return source_at(s)+u[:,None]*across
        def numerator(points,n):
            a,b,c=(points@axes.T).reshape(3,n,2)
            return (c[:,0]-b[:,0])*(b[:,1]-a[:,1])-(b[:,0]-a[:,0])*(c[:,1]-b[:,1])
        for budget in curve['chord_budgets']:
            r=float(budget['radius_px']);low,high=budget['preferred_signed_interval_px2']
            critical=np.unique(np.r_[knots,profile['spline_knots_px']])
            split=np.unique(np.r_[r,length-r,*[(critical+d)[(critical+d>r)&(critical+d<length-r)] for d in [-r,0.,r]]])
            for pitch in [.03125,.015625]:
                centers=np.unique(np.r_[np.linspace(r,length-r,int(np.ceil((length-2*r)/pitch))+1),split])
                sample_s=np.concatenate([centers-r,centers,centers+r]);source=source_at(sample_s);n=len(centers)
                source_n=numerator(source,n);request=requested(sample_s)
                row={'source_segment_id':segment['id'],'radius_source_arclength_px':r,
                    'max_center_pitch_px':pitch,'center_count':n,'budget_px2':[low,high],
                    'target_controller_reported_interval_px2':budget['actual_signed_interval_px2'],'cases':{}}
                arrays={'centers':centers,'source_numerator':source_n};case_points={}
                for name in ['requested_target','baseline_actual_f32','candidate_actual_f32']:
                    closure=None
                    if name=='requested_target':actual=request
                    else:
                        key='baseline' if name.startswith('baseline') else 'candidate'
                        offset=np.asarray(protocols[key]['crop'][:2])
                        actual,closure=inverse_points(fields[key],source-offset);actual+=offset
                    induced=numerator(actual,n)-source_n
                    lower_excess=np.maximum(low-induced,0.);upper_excess=np.maximum(induced-high,0.)
                    imn,imx=int(np.argmin(induced)),int(np.argmax(induced))
                    movement=(actual-source)@axes.T
                    # This second formula uses source heights exactly, like the
                    # target QP. Its discrepancy measures down-motion effects.
                    a,b,c=(source@axes.T).reshape(3,n,2)
                    u0,u1,u2=movement[:,0].reshape(3,n)
                    projected=(u2-u1)*(b[:,1]-a[:,1])-(u1-u0)*(c[:,1]-b[:,1])
                    stats={'sampled_interval_px2':[float(induced.min()),float(induced.max())],
                        'min_at_source_arclength_px':float(centers[imn]),'max_at_source_arclength_px':float(centers[imx]),
                        'min_center_source_xy':source_at(centers[[imn]])[0].tolist(),
                        'max_center_source_xy':source_at(centers[[imx]])[0].tolist(),
                        'lower_excess_max_px2':float(lower_excess.max()),'upper_excess_max_px2':float(upper_excess.max()),
                        'samples_beyond_budget_gt_1e_minus_7':int(np.count_nonzero((lower_excess>1e-7)|(upper_excess>1e-7))),
                        'within_budget_at_samples_1e_minus_7':bool(max(lower_excess.max(),upper_excess.max())<=1e-7),
                        'actual_vs_source_height_numerator_max_px2':float(abs(induced-projected).max()),
                        'down_displacement_max_px':float(abs(movement[:,1]).max()),'forward_closure':closure}
                    if name!='requested_target':stats['max_target_distance_px']=float(np.linalg.norm(actual-request,axis=1).max())
                    row['cases'][name]=stats;arrays[name+'_induced_numerator']=induced
                    if actual_local_change_limit is not None:case_points[name]=actual
                if actual_local_change_limit is not None:
                    frozen_preferred=source+pref(sample_s)[:,None]*across
                    comparisons={
                        'candidate_actual_vs_baseline_actual':(case_points['baseline_actual_f32'],case_points['candidate_actual_f32'],True),
                        'candidate_actual_vs_frozen_preferred_pchip':(frozen_preferred,case_points['candidate_actual_f32'],True),
                        'requested_target_vs_frozen_preferred_pchip':(frozen_preferred,request,False)}
                    local={'normalization':'frozen source h0*h1; signed delta of 2D chord numerator, dimensionless slope-difference change',
                        'actual_engineering_observation_limit':actual_local_change_limit,
                        'not_a_turn_angle_or_aesthetic_certificate':True,'comparisons':{}}
                    source_chords=source.reshape(3,n,2)
                    for key,(reference,points,is_actual) in comparisons.items():
                        values=normalized_local_chord_change(source_chords,reference.reshape(3,n,2),points.reshape(3,n,2),axes)
                        worst=int(np.argmax(abs(values)));magnitude=float(abs(values[worst]))
                        measured={'max_abs_normalized_change':magnitude,
                            'signed_normalized_change_at_worst':float(values[worst]),
                            'sampled_signed_interval':[float(values.min()),float(values.max())],
                            'worst_source_arclength_px':float(centers[worst]),
                            'worst_center_source_xy':source_at(centers[[worst]])[0].tolist()}
                        if is_actual:
                            excess=np.maximum(abs(values)-actual_local_change_limit,0.)
                            measured.update({'engineering_observation_limit':actual_local_change_limit,
                                'max_excess_over_engineering_limit':float(excess.max()),
                                'samples_above_limit_gt_1e_minus_10':int(np.count_nonzero(excess>1e-10)),
                                'within_engineering_limit_at_samples_1e_minus_10':bool(excess.max()<=1e-10)})
                        local['comparisons'][key]=measured;arrays[key+'_normalized_local_change']=values
                    row['local_change_observation']=local
                records.append(row)
                np.savez_compressed(output/(segment['id']+'-r'+str(int(r))+'-pitch'+str(pitch)+'.npz'),**arrays)
    comparison={'same_source_geometry':True,'same_signed_amount':True,'same_requested_profile':True,
        'decoded_fixed_mask_changed_pixels':changed,'same_fixed_png_bytes':sha(baseline.parent/'fixed-source.png')==sha(candidate.parent/'fixed-source.png'),
        'baseline_stride':protocols['baseline'].get('field_reduction_stride'),
        'candidate_stride':protocols['candidate'].get('field_reduction_stride')}
    report={'baseline_case':str(baseline),'candidate_case':str(candidate),'boundary_target':str(target_path),
        'inputs_sha256':hashes,'comparison':comparison,'measurements':records,
        'meaning':'D = actual 2D finite-chord numerator minus original source numerator; absolute pixel^2, not a turning angle. Each budget is a global signed interval per scale, not pointwise.',
        'sampling_limits':'Two dense source-arclength grids plus all shifted source/spline knots; sampled maxima are lower bounds on any continuous-domain violation. Not a continuous extrema certificate.',
        'beauty_approval':False}
    if actual_local_change_limit is not None:
        report['local_change_observation']={'actual_engineering_limit':actual_local_change_limit,
            'not_a_turn_angle_or_aesthetic_certificate':True,'global_budget_results_unmodified':True,
            'requested_target_has_no_actual_field_limit_applied':True,
            'all_sampled_within_engineering_limit':{key:all(
                row['local_change_observation']['comparisons'][key]['within_engineering_limit_at_samples_1e_minus_10']
                for row in records) for key in ['candidate_actual_vs_baseline_actual','candidate_actual_vs_frozen_preferred_pchip']}}
    if any(sha(p)!=digest for p,digest in hashes.items()):raise ValueError('audit inputs changed')
    save(output/'report.json',report);shutil.copy2(Path(__file__).resolve(),output/'executed-audit.py')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline','candidate','target','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--actual-local-change-limit',type=float)
    a=parser.parse_args();run(a.baseline,a.candidate,a.target,a.output,
        actual_local_change_limit=a.actual_local_change_limit)
