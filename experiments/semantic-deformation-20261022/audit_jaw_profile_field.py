"""Measure final f32 inverse-map jaw curves, not only requested control points.

This is a sampled geometric audit of a source proxy, not an anatomical or
beauty certificate. The profile's frozen known-convex intervals are interpreted
as open intervals. Every measured destination is obtained by inverting the
actual stored f32 inverse map.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from jaw_collocation import _prepare
from jaw_target_profile import load_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'face-source-forward-prototype-20261021'))
from forward import inverse_points


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')
def signed_area(points):
    q = points-points.mean(0)
    return float(.5*np.sum(q[:, 0]*np.roll(q[:, 1], -1)-q[:, 1]*np.roll(q[:, 0], -1)))


def run(baseline, candidate, profile_path, output, *, baseline_profile_path=None):
    baseline, candidate, profile_path, output = [Path(p).resolve() for p in [baseline, candidate, profile_path, output]]
    if output.exists(): raise FileExistsError('refusing to overwrite actual-field jaw audit')
    protocols = {name: read(case.parent/'protocol.json') for name,case in [('baseline',baseline),('candidate',candidate)]}
    geometry = protocols['candidate']['source_geometry']
    if geometry != protocols['baseline']['source_geometry']:
        raise ValueError('baseline and candidate must share the original source geometry')
    decisions = {name:read(case/'decision.json') for name,case in [('baseline',baseline),('candidate',candidate)]}
    amount = decisions['candidate']['amount']
    if decisions['baseline']['amount'] != amount: raise ValueError('signed amounts differ')
    if any(not decision['final_geometry']['passed_strict'] for decision in decisions.values()):
        raise ValueError('only final f32 geometry-passing fields can be audited')
    profile = load_profile(profile_path, geometry, amount)
    baseline_profile_path = Path(baseline_profile_path).resolve() if baseline_profile_path is not None else None
    baseline_profile = load_profile(baseline_profile_path, geometry, amount) if baseline_profile_path else None
    prepared = _prepare(geometry, amount, .13, 1.)
    p, edge, length, cumulative, origin, across, midline, arcs, _ = prepared
    down = np.asarray(geometry['frame']['down'])
    profiles = {row['arc_id']:row for row in profile['profiles']}
    baseline_profiles = {row['arc_id']:row for row in baseline_profile['profiles']} if baseline_profile else None
    fields = {name:np.load(case/'inverse-field.npy').astype(float) for name,case in [('baseline',baseline),('candidate',candidate)]}
    files = [profile_path, Path(__file__).resolve()]
    if baseline_profile_path: files.append(baseline_profile_path)
    for case in [baseline,candidate]:
        files += [case/'inverse-field.npy', case/'decision.json', case.parent/'protocol.json',case.parent/'fixed-source.png']
    hashes = {str(path):sha(path) for path in files}
    def source_at(position):
        position = np.asarray(position) % cumulative[-1]
        ids = np.minimum(np.searchsorted(cumulative, position, side='right')-1, len(p)-1)
        return p[ids]+((position-cumulative[ids])/length[ids])[:,None]*edge[ids]
    def arc_source(arc, s): return source_at(arc['start']+arc['direction']*np.asarray(s))
    def actual(name, source):
        offset = np.asarray(protocols[name]['crop'][:2])
        destination,closure = inverse_points(fields[name], source-offset)
        return destination+offset, closure
    def target(arc, s, mode):
        q = arc_source(arc,s)
        if mode == 'baseline' and baseline_profiles is None:
            t=s/arc['length'];du=-amount/100.*.13*((q-origin)@across-midline)*64.*(t*(1.-t))**3
        else:
            row=(baseline_profiles if mode == 'baseline' else profiles)[arc['id']]
            du=np.interp(s,row['arclength_px'],row['across_displacement_px'])
        return q+du[:,None]*across
    output.mkdir(parents=True)
    areas=[]
    # Retain every source vertex, profile knot, and endpoint in a dense closed
    # contour. Two pitches quantify polygonal area approximation sensitivity.
    for pitch in [.25,.125]:
        positions = np.concatenate([np.linspace(cumulative[i],cumulative[i+1],int(np.ceil(d/pitch))+1)[:-1]
                                    for i,d in enumerate(length)])
        positions = np.unique(np.r_[positions, *[(arc['start']+arc['direction']*np.asarray(profiles[arc['id']]['arclength_px'])) % cumulative[-1] for arc in arcs]])
        source = source_at(positions); source_area=signed_area(source)
        row={'source_pitch_max_px':pitch,'closed_polygon_point_count':len(source),'source_signed_area_px2':source_area,
             'source_area_difference_from_original_polygon_px2':source_area-signed_area(p),'cases':{}}
        for name in ['baseline','candidate']:
            request=source.copy()
            for arc in arcs:
                s=((positions-arc['start'])*arc['direction'])%cumulative[-1]
                inside=(s>=0.)&(s<=arc['length'])
                request[inside]=target(arc,s[inside],name)
            destination,closure=actual(name,source)
            request_delta=signed_area(request)-source_area
            actual_delta=signed_area(destination)-source_area
            row['cases'][name]={'requested_signed_area_delta_px2':request_delta,
                'actual_signed_area_delta_px2':actual_delta,'actual_minus_requested_area_px2':actual_delta-request_delta,
                'actual_to_requested_area_ratio':actual_delta/request_delta if request_delta else None,
                'forward_closure':closure}
        areas.append(row)
    windows=[]
    for arc in arcs:
        row=profiles[arc['id']]
        centers=np.arange(16.015625,arc['length']-16.,.03125)
        known=np.zeros(len(centers),bool)
        for lo,hi in row['known_source_convex_monotone_center_intervals_px']:
            known |= (centers>lo+1e-9)&(centers<hi-1e-9)
        s=np.concatenate([centers-16.,centers,centers+16.])
        source=arc_source(arc,s)
        side=-float(np.sign((arc['first']-origin)@across-midline))
        def measure(points):
            uv=(points-origin)@np.array([across,down]).T
            a,b,c=uv.reshape(3,len(centers),2)
            first,second=b-a,c-b;h0,h1=first[:,1],second[:,1]
            numerator=side*(second[:,0]*h0-first[:,0]*h1)
            monotone=(h0>0)&(h1>0)
            slopes=np.divide(numerator,h0*h1,out=np.full_like(numerator,np.nan),where=monotone)
            angles=side*(np.arctan2(second[:,0],second[:,1])-np.arctan2(first[:,0],first[:,1]))*180/np.pi
            return numerator,slopes,angles,monotone
        source_values=measure(source)
        if np.any(~source_values[3][known]) or np.any(source_values[0][known]<-1e-8):
            raise ValueError('frozen known interval conflicts with current source geometry')
        result={'arc_id':arc['id'],'window_half_span_source_px':16.,'sample_pitch_px':.03125,
                'known_open_interval_windows':int(known.sum()),'unknown_windows':int((~known).sum()),'cases':{}}
        arrays={'centers':centers,'known':known,'source':source,'source_numerator':source_values[0]}
        for name in ['baseline','candidate']:
            request=target(arc,s,name);destination,closure=actual(name,source)
            case={}
            for key,points in [('requested',request),('actual_f32_forward',destination)]:
                numerator,slopes,angles,monotone=measure(points)
                selected=known&monotone
                case[key]={'known_nonmonotone_windows':int(np.count_nonzero(known&~monotone)),
                    'known_reverse_numerator_below_minus_1e_minus_8_count':int(np.count_nonzero(selected&(numerator < -1e-8))),
                    'known_min_numerator_px2':float(numerator[known].min()),
                    'known_min_signed_slope_difference':float(slopes[selected].min()),
                    'known_min_signed_turn_degrees':float(angles[selected].min()),
                    'known_reverse_angle_below_minus_0_01deg_count':int(np.count_nonzero(selected&(angles < -.01))),
                    'known_reverse_angle_below_minus_0_1deg_count':int(np.count_nonzero(selected&(angles < -.1))),
                    'known_reverse_angle_below_minus_1deg_count':int(np.count_nonzero(selected&(angles < -1.))),
                    'unknown_reverse_count_not_used_as_anatomical_failure':int(np.count_nonzero(~known&monotone&(numerator < -1e-8)))}
                arrays[name+'_'+key+'_numerator']=numerator
                arrays[name+'_'+key+'_angle_deg']=angles
            case['actual_target_distance_max_px']=float(np.linalg.norm(destination-request,axis=1).max())
            case['forward_closure']=closure;result['cases'][name]=case
        windows.append(result);np.savez_compressed(output/(arc['id']+'-windows.npz'),**arrays)
    comparison={'same_signed_amount':True,'same_source_geometry':True,
        'same_fixed_mask_bytes':sha(baseline.parent/'fixed-source.png')==sha(candidate.parent/'fixed-source.png'),
        'baseline_stride':protocols['baseline'].get('field_reduction_stride',4),
        'candidate_stride':protocols['candidate'].get('field_reduction_stride',4),
        'baseline_collocation_spacing':protocols['baseline'].get('collocation_spacing'),
        'candidate_collocation_spacing':protocols['candidate'].get('collocation_spacing')}
    comparison['same_requested_profile'] = bool(baseline_profile_path and sha(baseline_profile_path) == sha(profile_path))
    report={'amount':amount,'baseline_case':str(baseline),'candidate_case':str(candidate),'inputs_sha256':hashes,
        'comparison':comparison,'closed_contour_area':areas,'known_source_convex_windows':windows,
        'sampling_limits':'Dense numerical samples and two area pitches, not continuous extrema proof. Open known intervals exclude unknown boundaries. Source proxy is not anatomical ground truth.',
        'beauty_approval':False}
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('audit inputs changed')
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline','candidate','profile','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--baseline-profile',type=Path)
    args=parser.parse_args();result=run(args.baseline,args.candidate,args.profile,args.output,baseline_profile_path=args.baseline_profile)
    print(json.dumps({'amount':result['amount'],'comparison':result['comparison'],
        'area':result['closed_contour_area'][-1], 'windows':result['known_source_convex_windows']},indent=2))
