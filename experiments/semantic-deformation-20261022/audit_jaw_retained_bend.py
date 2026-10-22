"""Independent sampled retained-bend audit of frozen jaw field windows.

Source and requested chords are recomputed from the original geometry and
profiles. Actual f32 chord numerators/angles come from audit_jaw_profile_field's
frozen windows.npz, avoiding another field inversion. No target-QP/probe metric
helper is imported. Sampling is not a continuous certificate or beauty score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from jaw_collocation import _prepare
from jaw_target_profile import load_profile


RHO = .5
WINDOW_PX = 16.
CENTER_PITCH_PX = .03125
NUMERATOR_TOLERANCE_PX2 = 1e-8


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def finite_chord_measure(points, axes, side):
    """Measure oriented 3xNx2 point triplets using their actual 2D positions."""
    points, axes = np.asarray(points, float), np.asarray(axes, float)
    if (points.ndim != 3 or points.shape[0] != 3 or points.shape[2] != 2
            or not points.shape[1] or not np.isfinite(points).all()
            or axes.shape != (2, 2) or not np.isfinite(axes).all()
            or not np.allclose(axes@axes.T, np.eye(2), atol=1e-10, rtol=0.)
            or np.linalg.det(axes) <= 0. or side not in [-1., 1.]):
        raise ValueError('finite 3xNx2 chords, positively oriented orthonormal axes and side +/-1 required')
    first = (points[1]-points[0])@axes.T
    second = (points[2]-points[1])@axes.T
    return {'numerator': side*(second[:, 0]*first[:, 1]-first[:, 0]*second[:, 1]),
            'angle_degrees': side*np.degrees(np.arctan2(second[:, 0], second[:, 1])
                                           -np.arctan2(first[:, 0], first[:, 1])),
            'monotone': (first[:, 1] > 0.) & (second[:, 1] > 0.)}


def retained_bend_margin(source_numerator, target_numerator, rho=.5):
    source, target = np.asarray(source_numerator, float), np.asarray(target_numerator, float)
    if (source.shape != target.shape or source.ndim != 1 or not len(source)
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or not np.isfinite(rho) or not 0. <= rho <= 1.):
        raise ValueError('matching finite one-dimensional numerators and rho in [0,1] required')
    return target-rho*source


def _known_mask(centers, intervals):
    intervals = np.asarray(intervals, float).reshape(-1, 2)
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError('finite ordered frozen known source intervals required')
    mask = np.zeros(len(centers), bool)
    for lo, hi in intervals:
        # Match the earlier audit's open-interval convention exactly.
        mask |= (centers > lo+1e-9) & (centers < hi-1e-9)
    return mask


def run(baseline, candidate, profile, baseline_profile, output, *, field_audit):
    cases = {key: Path(value).resolve() for key, value in [('baseline', baseline), ('candidate', candidate)]}
    paths = {'baseline': Path(baseline_profile).resolve(), 'candidate': Path(profile).resolve()}
    output, field_audit = Path(output).resolve(), Path(field_audit).resolve()
    if field_audit.is_dir(): field_audit /= 'report.json'
    if output.exists(): raise FileExistsError('refusing to overwrite retained-bend audit')
    cache = read(field_audit)
    if any(Path(cache[name+'_case']).resolve() != path for name, path in cases.items()):
        raise ValueError('cached audit baseline/candidate cases differ')
    hashes = {str(field_audit): sha(field_audit)}
    # Old code may legitimately have changed; record its historical hash but
    # require unchanged fields, masks, profiles and all other data inputs.
    historical = cache['inputs_sha256']
    for path, digest in historical.items():
        if Path(path).suffix.lower() != '.py':
            if sha(path) != digest: raise ValueError('historically bound audit data changed: '+path)
            hashes[str(Path(path).resolve())] = digest
    protocols = {name: read(path.parent/'protocol.json') for name, path in cases.items()}
    geometry = protocols['candidate']['source_geometry']; amount = cache['amount']
    if protocols['baseline']['source_geometry'] != geometry: raise ValueError('same original source geometry required')
    profiles = {}
    for name, case in cases.items():
        protocol = protocols[name]; path = paths[name]
        frozen = Path(protocol['frozen_target_profile']).resolve()
        digest = sha(path)
        if (protocol.get('inputs_sha256', {}).get(str(frozen)) != digest
                or sha(frozen) != digest or digest not in historical.values()):
            raise ValueError('profile differs from field protocol or cached audit binding')
        decision = read(case/'decision.json')
        if decision['amount'] != amount or not decision['final_geometry']['passed_strict']:
            raise ValueError('same amount and strict final f32 field geometry pass required')
        field = np.load(case/'inverse-field.npy', mmap_mode='r', allow_pickle=False)
        if field.dtype != np.float32: raise ValueError('frozen actual f32 inverse field required')
        profiles[name] = load_profile(path, geometry, amount)
        if profiles[name]['parameters']['window_radius_px'] != WINDOW_PX:
            raise ValueError('original 16px source half-chord required')
        hashes[str(path)] = digest
        for required in [case/'inverse-field.npy', case/'decision.json', case.parent/'protocol.json', case.parent/'fixed-source.png']:
            if historical.get(str(required)) != sha(required):
                raise ValueError('required field input lacks unchanged historical binding: '+str(required))
        # Bind the source evidence that load_profile has just validated.
        source_protocol = Path(profiles[name]['source_protocol']).resolve()
        hashes[str(source_protocol)] = sha(source_protocol)
        for evidence, evidence_sha in read(source_protocol)['inputs_sha256'].items():
            if Path(evidence).suffix.lower() in ['.json', '.png', '.jpg', '.jpeg']:
                hashes[str(Path(evidence).resolve())] = evidence_sha
    source = _prepare(geometry, amount, .13, 1.)
    points, edges, lengths, cumulative, origin, across, midline, arcs, _ = source
    axes = np.asarray([across, geometry['frame']['down']], float)
    rows = {name: {row['arc_id']: row for row in value['profiles']} for name, value in profiles.items()}
    cache_rows = {row['arc_id']: row for row in cache['known_source_convex_windows']}
    measurements, retained_arrays = [], {}
    for arc in arcs:
        record = cache_rows[arc['id']]
        if record['window_half_span_source_px'] != WINDOW_PX or record['sample_pitch_px'] != CENTER_PITCH_PX:
            raise ValueError('cached window radius or center pitch differs')
        data_path = field_audit.parent/(arc['id']+'-windows.npz')
        hashes[str(data_path)] = sha(data_path)
        with np.load(data_path, allow_pickle=False) as cache_arrays:
            data = {key: cache_arrays[key] for key in cache_arrays.files}
        centers = np.arange(WINDOW_PX+CENTER_PITCH_PX/2., arc['length']-WINDOW_PX, CENTER_PITCH_PX)
        if not np.array_equal(data['centers'], centers): raise ValueError('cached center grid differs')
        intervals = rows['baseline'][arc['id']]['known_source_convex_monotone_center_intervals_px']
        if intervals != rows['candidate'][arc['id']]['known_source_convex_monotone_center_intervals_px']:
            raise ValueError('frozen known source intervals changed between profiles')
        known = _known_mask(centers, intervals)
        if not np.array_equal(known, data['known']) or not np.any(known):
            raise ValueError('cached known source mask differs or has no samples')
        if int(known.sum()) != record['known_open_interval_windows']:
            raise ValueError('cached report known count differs')
        arclength = np.concatenate([centers-WINDOW_PX, centers, centers+WINDOW_PX])
        position = (arc['start']+arc['direction']*arclength) % cumulative[-1]
        ids = np.minimum(np.searchsorted(cumulative, position, side='right')-1, len(points)-1)
        source_xy = points[ids]+((position-cumulative[ids])/lengths[ids])[:, None]*edges[ids]
        if not np.allclose(source_xy, data['source'], atol=1e-10, rtol=0.): raise ValueError('cached source points differ')
        source_xy = source_xy.reshape(3, len(centers), 2)
        side = -float(np.sign((arc['first']-origin)@across-midline))
        source_metrics = finite_chord_measure(source_xy, axes, side)
        source_n = source_metrics['numerator']
        if (not source_metrics['monotone'][known].all() or np.any(source_n[known] < -NUMERATOR_TOLERANCE_PX2)
                or not np.allclose(source_n, data['source_numerator'], atol=1e-8, rtol=0.)):
            raise ValueError('known source windows or independently recomputed source numerators differ')
        output_row = {'arc_id': arc['id'], 'window_half_span_source_px': WINDOW_PX,
            'sample_pitch_px': CENTER_PITCH_PX, 'known_open_interval_windows': int(known.sum()),
            'unknown_windows_excluded': int((~known).sum()), 'cases': {}}
        output_arrays = {'centers': centers, 'known': known, 'source': source_xy,
                         'source_numerator': source_n, 'source_angle_degrees': source_metrics['angle_degrees']}
        for name in ['baseline', 'candidate']:
            row = rows[name][arc['id']]
            displacement = np.interp(arclength, row['arclength_px'], row['across_displacement_px'])
            requested = source_xy+displacement.reshape(3, len(centers), 1)*across
            target_metrics = finite_chord_measure(requested, axes, side)
            if not target_metrics['monotone'][known].all(): raise ValueError('requested chord height changed')
            if (not np.allclose(target_metrics['numerator'], data[name+'_requested_numerator'], atol=1e-8, rtol=0.)
                    or not np.allclose(target_metrics['angle_degrees'], data[name+'_requested_angle_deg'], atol=1e-9, rtol=0.)):
                raise ValueError('cached requested chords differ from independently reconstructed targets')
            actual_n, actual_angles = [np.asarray(data[name+'_actual_f32_forward_'+key], float)
                                       for key in ['numerator', 'angle_deg']]
            if any(value.shape != centers.shape or not np.isfinite(value).all() for value in [actual_n, actual_angles]):
                raise ValueError('finite cached actual numerator/angle arrays required')
            target_n = target_metrics['numerator']; target_angles = target_metrics['angle_degrees']
            target_margin = retained_bend_margin(source_n, target_n, RHO)
            actual_margin = retained_bend_margin(source_n, actual_n, RHO)
            def witness(index):
                return {'source_arclength_px': float(centers[index]),
                    'source_triplet_xy_px': source_xy[:, index].tolist(),
                    'source_center_xy_px': source_xy[1, index].tolist(),
                    'source_numerator_px2': float(source_n[index]),
                    'requested_numerator_px2': float(target_n[index]), 'actual_numerator_px2': float(actual_n[index]),
                    'required_half_source_numerator_px2': float(RHO*source_n[index]),
                    'requested_margin_px2': float(target_margin[index]), 'actual_margin_px2': float(actual_margin[index]),
                    'source_signed_turn_degrees': float(source_metrics['angle_degrees'][index]),
                    'requested_signed_turn_degrees': float(target_angles[index]),
                    'actual_signed_turn_degrees': float(actual_angles[index])}
            case = {'forward_closure_from_cached_audit': record['cases'][name]['forward_closure']}
            for mode, numerator, margin in [('requested', target_n, target_margin), ('actual_f32_forward', actual_n, actual_margin)]:
                previous = record['cases'][name][mode]
                legacy = previous['known_reverse_numerator_below_minus_1e_minus_8_count']
                nonmonotone = previous['known_nonmonotone_windows']
                negative = int(np.count_nonzero(known & (numerator < -NUMERATOR_TOLERANCE_PX2)))
                if nonmonotone == 0 and legacy != negative:
                    raise ValueError('cached legacy negative numerator count differs')
                indices = np.flatnonzero(known); worst = int(indices[np.argmin(margin[known])])
                case[mode] = {'known_nonmonotone_windows_from_cached_audit': nonmonotone,
                    'legacy_monotone_known_reverse_numerator_below_minus_1e_minus_8_count': legacy,
                    'all_known_negative_numerator_below_minus_1e_minus_8_count': negative,
                    'known_min_numerator_px2': float(numerator[known].min()),
                    'known_retention_margin_below_minus_1e_minus_8_count': int(np.count_nonzero(known & (margin < -NUMERATOR_TOLERANCE_PX2))),
                    'known_min_retention_margin_px2': float(margin[worst]), 'worst_margin_witness': witness(worst)}
                output_arrays[name+'_'+mode+'_retention_margin_px2'] = margin
            output_row['cases'][name] = case
        measurements.append(output_row); retained_arrays[arc['id']] = output_arrays
    dependencies = [Path(__file__).resolve(), Path(__file__).with_name('jaw_collocation.py'), Path(__file__).with_name('jaw_target_profile.py')]
    hashes.update({str(path): sha(path) for path in dependencies})
    report = {'experiment': 'independent-fixed-rho-half-retained-bend-field-audit-v1',
        'rho': RHO, 'absolute_numerator_tolerance_px2': NUMERATOR_TOLERANCE_PX2, 'amount': amount,
        'baseline_case': str(cases['baseline']), 'candidate_case': str(cases['candidate']),
        'baseline_profile': str(paths['baseline']), 'candidate_profile': str(paths['candidate']),
        'frozen_field_audit': str(field_audit), 'inputs_sha256': hashes,
        'historical_field_audit_inputs_sha256': historical, 'measurements': measurements,
        'definition': 'N=side*(du_second*dh_first-du_first*dh_second); M=N-0.5*N_source, in absolute px^2; source frame and source-arclength triplets are frozen.',
        'actual_values_provenance': 'Cached actual f32 forward chord numerators and angles were computed by audit_jaw_profile_field independently of the retained-bend target probe. No actual point reconstruction is repeated here.',
        'sampling_limits': '16px source half-chord; 0.03125px center grid; original open source-convex monotone intervals only. Sampled minima do not prove continuous extrema. Actual nonmonotone windows are separately reported and remain in absolute margin counts.',
        'interpretation': 'A retained finite-chord numerator is not a curvature or angle ratio. No source-numerator division is used, including near-zero source bends.',
        'source_geometry_unmodified': True, 'beauty_approval': False, 'continuous_certificate': False}
    if any(sha(path) != digest for path, digest in hashes.items()): raise ValueError('retained-bend audit input changed')
    output.mkdir(parents=True)
    for arc_id, arrays in retained_arrays.items(): np.savez_compressed(output/(arc_id+'-margins.npz'), **arrays)
    shutil.copyfile(__file__, output/'executed-audit.py')
    save(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['baseline', 'candidate', 'profile', 'baseline-profile', 'field-audit', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.baseline, args.candidate, args.profile, args.baseline_profile, args.output, field_audit=args.field_audit)
    print(json.dumps({'amount': result['amount'], 'rho': result['rho'], 'measurements': result['measurements']}, indent=2))
