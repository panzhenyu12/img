"""Frozen source-arclength quarters and motion peaks in actual f32 jaw fields.

The four intervals are fixed before reading a candidate: [0,.25,.5,.75,1]
of each reviewed original arc. They are not anatomical labels or reselected
around candidate peaks. Different preferred target shapes remain explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from audit_jaw_bilateral_field import inverse_points, _signed_area
from jaw_collocation import _prepare
from jaw_target_profile import load_profile


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def swept_signed_area(source, destination, direction):
    """Source-to-destination swept strip with the original contour direction."""
    return direction*_signed_area(np.r_[destination, source[::-1]])


def run(profile_audit_path, output):
    profile_audit_path, output = Path(profile_audit_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('refusing to overwrite fixed-quarter audit')
    prior = read(profile_audit_path)
    cases = {name: Path(prior[name+'_case']) for name in ['baseline', 'candidate']}
    protocols = {name: read(case.parent/'protocol.json') for name, case in cases.items()}
    geometry = protocols['baseline']['source_geometry']; amount = prior['amount']
    if any(p['source_geometry'] != geometry for p in protocols.values()): raise ValueError('same original source geometry required')
    if any(sha(path) != digest for path, digest in prior['inputs_sha256'].items() if Path(path).suffix.lower() != '.py'):
        raise ValueError('frozen prior field/profile input changed')
    hashes = {str(profile_audit_path): sha(profile_audit_path), str(Path(__file__).resolve()): sha(__file__)}
    for path in [Path(__file__).with_name(name) for name in ['audit_jaw_bilateral_field.py', 'jaw_collocation.py', 'jaw_target_profile.py']]:
        hashes[str(path)] = sha(path)
    profiles = {}; fields = {}
    for name, case in cases.items():
        protocol = protocols[name]; path = Path(protocol['frozen_target_profile']).resolve()
        if protocol['inputs_sha256'].get(str(path)) != sha(path): raise ValueError('historical profile differs')
        profiles[name] = load_profile(path, geometry, amount)
        hashes[str(path)] = sha(path); hashes[str(case/'inverse-field.npy')] = sha(case/'inverse-field.npy')
        fields[name] = np.load(case/'inverse-field.npy')
        if fields[name].dtype != np.float32: raise ValueError('stored actual f32 field required')
    prepared = _prepare(geometry, amount, .13, 1.)
    p, edge, lengths, cumulative, origin, across, midline, arcs, _ = prepared
    down = np.asarray(geometry['frame']['down'])
    rows = {name: {row['arc_id']: row for row in profile['profiles']} for name, profile in profiles.items()}
    def source_at(arc, s):
        position = (arc['start']+arc['direction']*s)%cumulative[-1]
        ids = np.minimum(np.searchsorted(cumulative, position, side='right')-1, len(p)-1)
        points = p[ids]+((position-cumulative[ids])/lengths[ids])[:, None]*edge[ids]
        points[s == 0.] = arc['first']; points[s == arc['length']] = arc['last']
        return points
    output.mkdir(parents=True); shutil.copyfile(__file__, output/'executed-audit.py')
    records = []
    for pitch in [.125, .0625]:
        record = {'max_source_arclength_pitch_px': pitch, 'arcs': []}
        for arc in arcs:
            original_s = ((cumulative[:-1]-arc['start'])*arc['direction'])%cumulative[-1]
            breaks = np.linspace(0., arc['length'], 5)
            s = np.unique(np.r_[np.linspace(0., arc['length'], int(np.ceil(arc['length']/pitch))+1),
                                original_s[(original_s > 0.) & (original_s < arc['length'])], breaks,
                                *[rows[name][arc['id']]['arclength_px'] for name in cases]])
            source = source_at(arc, s)
            item = {'arc_id': arc['id'], 'source_arc_length_px': arc['length'], 'cases': {}}
            arrays = {'source_arclength': s, 'source': source}
            for name in cases:
                row = rows[name][arc['id']]
                target = source+np.interp(s, row['arclength_px'], row['across_displacement_px'])[:, None]*across
                offset = np.asarray(protocols[name]['crop'][:2])
                actual, closure = inverse_points(fields[name], source-offset); actual += offset
                measured = {'forward_closure': closure}
                for mode, destination in [('requested', target), ('actual_f32_forward', actual)]:
                    du = (destination-source)@across; peak = int(np.argmax(abs(du))); quarters = []
                    for lo, hi in zip(breaks[:-1], breaks[1:]):
                        mask = (s >= lo) & (s <= hi)
                        values = du[mask]; distances = s[mask]
                        quarters.append({'normalized_arclength_interval': [lo/arc['length'], hi/arc['length']],
                            'signed_swept_area_px2': swept_signed_area(source[mask], destination[mask], arc['direction']),
                            'absolute_across_motion_integral_px2': float(np.sum(.5*(abs(values[:-1])+abs(values[1:]))*np.diff(distances)))})
                    total = swept_signed_area(source, destination, arc['direction'])
                    error = sum(q['signed_swept_area_px2'] for q in quarters)-total
                    if abs(error) > 1e-7: raise ValueError('fixed-quarter swept areas do not add to full original arc')
                    measured[mode] = {'quarters': quarters, 'whole_arc_signed_swept_area_px2': total,
                        'sum_of_quarters_minus_arc_px2': error,
                        'peak_across_displacement_px': float(du[peak]), 'peak_normalized_source_arclength': float(s[peak]/arc['length']),
                        'peak_source_down_px': float((source[peak]-origin)@down), 'peak_source_xy': source[peak].tolist(),
                        'source_down_motion_max_abs_px': float(abs((destination-source)@down).max())}
                    arrays[name+'_'+mode] = destination
                item['cases'][name] = measured
            record['arcs'].append(item); np.savez_compressed(output/(arc['id']+'-'+str(pitch)+'.npz'), **arrays)
        record['cases'] = {}
        for name in cases:
            record['cases'][name] = {}
            for mode in ['requested', 'actual_f32_forward']:
                per_arc = [arc['cases'][name][mode] for arc in record['arcs']]
                quarters = [sum(a['quarters'][i]['signed_swept_area_px2'] for a in per_arc) for i in range(4)]
                total = sum(quarters)
                field = 'requested_signed_area_delta_px2' if mode == 'requested' else 'actual_signed_area_delta_px2'
                closed = prior['closed_contour_area'][-1]['cases'][name][field]
                record['cases'][name][mode] = {'combined_quarter_signed_area_px2': quarters,
                    'combined_quarter_fraction_of_arc_swept_area': [q/total if total else None for q in quarters],
                    'whole_reviewed_arcs_signed_swept_area_px2': total, 'prior_full_closed_contour_delta_px2': closed,
                    'full_closed_minus_reviewed_arc_sweeps_px2': closed-total,
                    'peak_source_down_gap_abs_px': abs(per_arc[0]['peak_source_down_px']-per_arc[1]['peak_source_down_px'])}
        records.append(record)
    report = {'profile_field_audit': str(profile_audit_path), 'inputs_sha256': hashes, 'amount': amount,
              'baseline_case': str(cases['baseline']), 'candidate_case': str(cases['candidate']),
              'different_profile_is_explicit_new_shape_request': True, 'source_arclength_quarters': [0., .25, .5, .75, 1.],
              'samples': records, 'definition': 'signed polygon area of destination arc followed by reversed original arc, multiplied by original contour traversal direction',
              'limits': 'Finite field samples, not a continuous curvature/extrema proof. Source arcs/unknown tail are unchanged. Arc sweep excludes outside-arc/endpoint-connector effects measured separately from complete closed area.',
              'beauty_approval': False}
    if any(sha(path) != digest for path, digest in hashes.items()): raise ValueError('distribution audit input changed')
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-audit', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = run(args.profile_audit, args.output)
    print(json.dumps(report['samples'][-1]['cases'], indent=2))
