"""Independent same-source-height bilateral audit of stored f32 jaw fields.

Pairing is derived from the original source segments, without importing the
target optimizer's bilateral helper. Only open height intervals with exactly
one forward intersection on each side are paired. Unknown tails, reversals,
horizontal segments and multi-intersections are never extrapolated across.
This measures editing balance, not source facial symmetry or attractiveness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from jaw_collocation import _prepare
from jaw_target_profile import load_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'face-source-forward-prototype-20261021'))
from forward import inverse_points


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pair_source_height_segments(left, right):
    """Partition two source UV polylines into paired/excluded open intervals.

U is signed radius from the frozen midline; V is source down height. Counting
all intersections before checking orientation prevents accepting one forward
branch while silently discarding a second backward branch at the same height.
Shared vertex heights have measure zero and are not included in q samples.
"""
    sides = [np.asarray(value, float) for value in [left, right]]
    if any(v.ndim != 2 or v.shape[1] != 2 or len(v) < 2 or not np.isfinite(v).all() for v in sides):
        raise ValueError('two finite source UV polylines required')
    splits = np.unique(np.concatenate([v[:, 1] for v in sides]))
    # Split radius-zero crossings as well; a side that crosses the midline is
    # only eligible on the strictly signed part of its original source edge.
    for v in sides:
        ids = np.flatnonzero(v[:-1, 0]*v[1:, 0] < 0.)
        if len(ids):
            t = -v[ids, 0]/(v[ids+1, 0]-v[ids, 0])
            splits = np.unique(np.r_[splits, v[ids, 1]+t*(v[ids+1, 1]-v[ids, 1])])
    paired, excluded = [], []
    for lo, hi in zip(splits[:-1], splits[1:]):
        center = (lo+hi)/2.; segment_ids = []; reasons = []
        for side, v in zip(['left', 'right'], sides):
            ids = np.flatnonzero((np.minimum(v[:-1, 1], v[1:, 1]) < center)
                                 & (np.maximum(v[:-1, 1], v[1:, 1]) > center))
            if len(ids) != 1:
                reasons.append(side+('_missing_intersection' if len(ids) == 0 else '_multiple_intersections'))
                continue
            i = int(ids[0]); dy = v[i+1, 1]-v[i, 1]
            if dy <= 0.:
                reasons.append(side+'_reverse_segment'); continue
            u = v[i, 0]+(center-v[i, 1])/dy*(v[i+1, 0]-v[i, 0])
            if (side == 'left' and u >= 0.) or (side == 'right' and u <= 0.):
                reasons.append(side+'_wrong_signed_radius'); continue
            segment_ids.append(i)
        row = {'source_down_open_interval_px': [float(lo), float(hi)]}
        if reasons:
            row['reasons'] = reasons; excluded.append(row)
        else:
            row.update(left_segment=segment_ids[0], right_segment=segment_ids[1]); paired.append(row)
    return paired, excluded


def bilateral_statistics(signed_radius, across_displacement, weights, heights):
    """Integrate (q_left-q_right)^2 over shared source height, q=-du/r."""
    r, u, w, h = [np.asarray(value, float) for value in [signed_radius, across_displacement, weights, heights]]
    if (r.ndim != 2 or r.shape[1] != 2 or u.shape != r.shape or w.shape != r.shape[:1]
            or h.shape != w.shape or len(w) == 0 or not all(np.isfinite(v).all() for v in [r, u, w, h])
            or np.any(r[:, 0] >= 0.) or np.any(r[:, 1] <= 0.) or np.any(w <= 0.)):
        raise ValueError('finite left-negative/right-positive radii, displacement and positive height weights required')
    q = -u/r; difference = q[:, 0]-q[:, 1]; worst = int(np.argmax(abs(difference)))
    return {'relative_contraction_difference_rms': float(np.sqrt(np.sum(w*difference**2)/w.sum())),
            'relative_contraction_difference_max_abs': float(abs(difference[worst])),
            'signed_difference_at_max': float(difference[worst]),
            'worst_source_down_px': float(h[worst]),
            'q_left_right_at_max': q[worst].tolist(),
            'source_radius_left_right_at_max_px': r[worst].tolist(),
            'height_measure_px': float(w.sum()), 'sample_count': len(w)}, q


def _signed_area(points):
    p = points-points.mean(axis=0)
    return float(.5*np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))


def audit_radius_weighted_displacement(report_path, output):
    """Supplement frozen q samples with an explicit pixel-scale imbalance.

The harmonic mean of the two absolute source radii converts q difference
back to a common pixel radius and stays finite as one radius becomes small.
It does not require equal pixel movements on unequal original facial radii.
"""
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('refusing to overwrite radius-weighted supplement')
    report = read(report_path); records = []; hashes = {str(report_path): sha(report_path)}
    for sample in report['paired_height_samples']:
        path = report_path.parent/('height-'+str(sample['max_height_pitch_px'])+'.npz')
        hashes[str(path)] = sha(path); data = np.load(path)
        signed_radius = data['source_radius']; weights = data['height_weight']
        if (signed_radius.ndim != 2 or signed_radius.shape[1] != 2 or not len(signed_radius)
                or weights.shape != signed_radius.shape[:1] or not np.isfinite(signed_radius).all()
                or not np.isfinite(weights).all() or np.any(weights <= 0.)
                or np.any(signed_radius[:, 0] >= 0.) or np.any(signed_radius[:, 1] <= 0.)):
            raise ValueError('frozen signed radii and positive height weights required')
        r = abs(data['source_radius']); common_radius = 2*r[:, 0]*r[:, 1]/r.sum(axis=1)
        row = {'max_height_pitch_px': sample['max_height_pitch_px'], 'cases': {}}
        for name in ['baseline', 'candidate']:
            row['cases'][name] = {}
            for mode in ['requested', 'actual_f32_forward']:
                q = data[name+'_'+mode+'_q']
                if q.shape != r.shape or not np.isfinite(q).all(): raise ValueError('finite frozen q samples matching radii required')
                mismatch = common_radius*abs(q[:, 0]-q[:, 1]); j = int(np.argmax(mismatch))
                row['cases'][name][mode] = {'height_weighted_rms_px': float(np.sqrt(np.sum(weights*mismatch**2)/weights.sum())),
                    'height_weighted_mean_abs_px': float(np.sum(weights*mismatch)/weights.sum()),
                    'sampled_max_abs_px': float(mismatch[j]), 'worst_source_down_px': float(data['source_height'][j]),
                    'worst_source_radius_left_right_px': data['source_radius'][j].tolist()}
        records.append(row)
    result = {'source_audit': str(report_path), 'inputs_sha256': hashes, 'execution_sha256': sha(__file__),
              'definition': 'm = 2*abs(r_left*r_right)/(abs(r_left)+abs(r_right)) * abs(q_left-q_right); q=-du/r',
              'measure': 'RMS and mean absolute m integrated over paired source down-height dh',
              'samples': records, 'limitation': 'Supplementary sampled pixel-scale mismatch; no changed thresholds, continuous-maximum proof or aesthetic acceptance',
              'beauty_approval': False}
    if any(sha(p) != digest for p, digest in hashes.items()): raise ValueError('frozen weighted-audit inputs changed')
    output.mkdir(parents=True); shutil.copyfile(__file__, output/'executed-audit.py'); save(output/'report.json', result)
    return result


def audit_endpoint_limits(report_path, output):
    """Audit one-sided limits of the paired material bands in a frozen audit.

Open-interval midpoint maxima may miss a large ratio close to the chin,
where the source radius becomes small. These are one-sided limits along the
already eligible source segments, not extrapolation into an unknown tail.
Endpoint maxima supplement, and do not replace, the original sampled report.
"""
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('refusing to overwrite endpoint-limit supplement')
    report = read(report_path); sample = report['paired_height_samples'][-1]
    data_path = report_path.parent/('height-'+str(sample['max_height_pitch_px'])+'.npz')
    data = np.load(data_path); indices = []; cursor = 0
    for interval in sample['paired_source_open_intervals']:
        lo, hi = interval['source_down_open_interval_px']
        count = max(1, int(np.ceil((hi-lo)/sample['max_height_pitch_px'])))+2
        indices.extend([cursor, cursor+count-1]); cursor += count
    if cursor != len(data['band_source']): raise ValueError('paired-band sampling history differs')
    source, s = data['band_source'][indices], data['band_source_arclength'][indices]
    frame = report['frame']; origin = np.asarray(frame['origin']); across = np.asarray(frame['across']); down = np.asarray(frame['down'])
    radius = (source-origin)@across-float(frame['midline_u']); valid = (radius[:, 0] < 0.) & (radius[:, 1] > 0.)
    if not np.any(valid): raise ValueError('no finite signed-radius endpoint limit available')
    if any(sha(p) != digest for p, digest in report['inputs_sha256'].items() if Path(p).suffix.lower() != '.py'):
        raise ValueError('frozen audit image/geometry/profile/field inputs changed')
    result = {'source_audit': str(report_path), 'source_audit_sha256': sha(report_path),
              'frozen_sample_sha256': sha(data_path), 'execution_sha256': sha(__file__),
              'eligible_open_interval_endpoint_count': len(indices), 'non_signed_radius_endpoints_not_evaluated': int((~valid).sum()),
              'endpoint_role': 'one-sided continuous limits within each eligible original source segment; no extrapolation', 'cases': {}}
    for name in ['baseline', 'candidate']:
        case = Path(report[name+'_case']); protocol = read(case.parent/'protocol.json')
        profile = load_profile(protocol['frozen_target_profile'], report['source_geometry'], report['amount'])
        # The original two arcs are identified from their signed source start.
        prepared = _prepare(report['source_geometry'], report['amount'], .13, 1.)
        rows = {r['arc_id']: r for r in profile['profiles']}; by_side = {}
        for arc in prepared[7]:
            side = 0 if (arc['first']-origin)@across-frame['midline_u'] < 0. else 1
            by_side[side] = rows[arc['id']]
        du = np.stack([np.interp(s[:, i], by_side[i]['arclength_px'], by_side[i]['across_displacement_px']) for i in [0, 1]], axis=1)
        field = np.load(case/'inverse-field.npy'); offset = np.asarray(protocol['crop'][:2])
        actual, closure = inverse_points(field, source.reshape(-1, 2)-offset)
        actual = actual.reshape(source.shape)+offset
        metrics = {'forward_closure': closure}
        for mode, displacement in [('requested', du), ('actual_f32_forward', (actual-source)@across)]:
            q = -displacement[valid]/radius[valid]; delta = q[:, 0]-q[:, 1]; j = int(np.argmax(abs(delta)))
            metrics[mode] = {'relative_contraction_difference_max_abs': float(abs(delta[j])),
                             'signed_difference_at_max': float(delta[j]), 'q_left_right_at_max': q[j].tolist(),
                             'source_down_at_max_px': ((source[valid][j]-origin)@down).tolist(),
                             'source_radius_at_max_px': radius[valid][j].tolist(),
                             'source_xy_at_max': source[valid][j].tolist()}
        result['cases'][name] = metrics
    result['beauty_approval'] = False
    if sha(report_path) != result['source_audit_sha256'] or sha(data_path) != result['frozen_sample_sha256']:
        raise ValueError('frozen endpoint-audit samples changed')
    output.mkdir(parents=True); shutil.copyfile(__file__, output/'executed-audit.py'); save(output/'report.json', result)
    return result


def run(baseline, candidate, output):
    cases = {'baseline': Path(baseline).resolve(), 'candidate': Path(candidate).resolve()}
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('refusing to overwrite bilateral actual-field audit')
    protocols = {key: read(case.parent/'protocol.json') for key, case in cases.items()}
    decisions = {key: read(case/'decision.json') for key, case in cases.items()}
    geometry = protocols['baseline']['source_geometry']; amount = decisions['baseline']['amount']
    if any(p['source_geometry'] != geometry for p in protocols.values()):
        raise ValueError('same original source geometry required')
    if any(d['amount'] != amount or not d['final_geometry']['passed_strict'] for d in decisions.values()):
        raise ValueError('same signed amount and strict final f32 geometry pass required')
    if protocols['baseline']['crop'] != protocols['candidate']['crop']:
        raise ValueError('same source crop required for direct fixed-mask comparison')
    hashes = {}; historical = {}; profiles = {}; fields = {}; fixed = {}; source_images = {}
    for name, case in cases.items():
        protocol = protocols[name]; path = Path(protocol['frozen_target_profile']).resolve()
        history = protocol.get('inputs_sha256', {})
        if history.get(str(path)) != sha(path): raise ValueError('historically frozen profile differs')
        profiles[name] = load_profile(path, geometry, amount)
        # Preserve and recheck original image/evidence and profile history;
        # historical code is provenance, not required to equal current code.
        frozen = {p: digest for p, digest in history.items() if Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg', '.json']}
        if any(sha(p) != digest for p, digest in frozen.items()): raise ValueError('historical image/source/profile input changed')
        historical[name] = frozen; hashes.update(frozen)
        declared = {}
        for evidence, evidence_digest in frozen.items():
            if Path(evidence).suffix.lower() != '.json': continue
            record = read(evidence)
            if not isinstance(record, dict) or not isinstance(record.get('source_image'), str): continue
            image_path = str(Path(record['source_image']).resolve())
            if image_path not in frozen: raise ValueError('declared source image lacks historical binding')
            declared[image_path] = {'sha256': frozen[image_path], 'evidence': evidence,
                                    'evidence_sha256': evidence_digest}
        if not declared: raise ValueError('historically bound original source-image declaration required')
        source_images[name] = declared
        fields[name] = np.load(case/'inverse-field.npy')
        if fields[name].dtype != np.float32: raise ValueError('stored f32 inverse field required')
        fixed[name] = np.asarray(Image.open(case.parent/'fixed-source.png').convert('L'))
        for p in [case/'inverse-field.npy', case/'decision.json', case.parent/'protocol.json', case.parent/'fixed-source.png']:
            hashes[str(p)] = sha(p)
    common_images = {p: digest for p, digest in historical['baseline'].items()
                     if Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg']
                     and historical['candidate'].get(p) == digest}
    if not common_images: raise ValueError('shared frozen source-image evidence required')
    if ({value['sha256'] for value in source_images['baseline'].values()}
            != {value['sha256'] for value in source_images['candidate'].values()}):
        raise ValueError('original source-image bytes differ')
    if fixed['baseline'].shape != fixed['candidate'].shape or not np.array_equal(fixed['baseline'], fixed['candidate']):
        raise ValueError('decoded fixed-source pixels differ')
    dependencies = [Path(__file__).resolve(), Path(__file__).with_name('jaw_collocation.py'),
                    Path(__file__).with_name('jaw_target_profile.py'),
                    Path(__file__).resolve().parents[1]/'face-source-forward-prototype-20261021/forward.py']
    hashes.update({str(p): sha(p) for p in dependencies})
    prepared = _prepare(geometry, amount, .13, 1.)
    p, edge, length, cumulative, origin, across, midline, arcs, _ = prepared
    down = np.asarray(geometry['frame']['down'], float); axes = np.asarray([across, down])
    if not np.allclose(axes@axes.T, np.eye(2), atol=1e-10, rtol=0.): raise ValueError('orthonormal source frame required')
    def source_at(arc, distance):
        distance = np.asarray(distance); position = (arc['start']+arc['direction']*distance)%cumulative[-1]
        ids = np.minimum(np.searchsorted(cumulative, position, side='right')-1, len(p)-1)
        source = p[ids]+((position-cumulative[ids])/length[ids])[:, None]*edge[ids]
        source[distance == 0.] = arc['first']; source[distance == arc['length']] = arc['last']
        return source
    source_arcs = {}
    for arc in arcs:
        s = ((cumulative[:-1]-arc['start'])*arc['direction'])%cumulative[-1]
        s = np.r_[0., np.sort(s[(s > 0.) & (s < arc['length'])]), arc['length']]
        xy = source_at(arc, s); uv = (xy-origin)@axes.T; uv[:, 0] -= midline
        side = 'left' if np.median(uv[:, 0]) < 0. else 'right'
        if side in source_arcs: raise ValueError('one source arc per signed side required')
        source_arcs[side] = {'arc': arc, 's': s, 'xy': xy, 'uv': uv}
    if set(source_arcs) != {'left', 'right'}: raise ValueError('two opposite source arcs required')
    paired, excluded = pair_source_height_segments(*(source_arcs[side]['uv'] for side in ['left', 'right']))
    if not paired: raise ValueError('no unambiguous paired source-height support')
    rows = {name: {r['arc_id']: r for r in profile['profiles']} for name, profile in profiles.items()}
    def requested(name, side, s, source):
        row = rows[name][source_arcs[side]['arc']['id']]
        return source+np.interp(s, row['arclength_px'], row['across_displacement_px'])[:, None]*across
    def actual(name, source):
        offset = np.asarray(protocols[name]['crop'][:2])
        points, closure = inverse_points(fields[name], source-offset)
        return points+offset, closure
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'executed-audit.py')
    records = []
    for pitch in [.125, .0625]:
        heights, weights, distances, sources, band_groups = [], [], [], [], []
        band_sources = []; band_distances = []; band_cursor = 0
        for interval in paired:
            lo, hi = interval['source_down_open_interval_px']
            count = max(1, int(np.ceil((hi-lo)/pitch)))
            h = lo+(np.arange(count)+.5)*(hi-lo)/count
            bh = np.r_[lo, h, hi]  # one-sided band limits; not q samples
            pair_s, pair_xy, band_s, band_xy = [], [], [], []
            for side in ['left', 'right']:
                a = source_arcs[side]; i = interval[side+'_segment']
                v0, v1 = a['uv'][i:i+2, 1]
                for values, out_s, out_xy in [(h, pair_s, pair_xy), (bh, band_s, band_xy)]:
                    t = (values-v0)/(v1-v0)
                    s = a['s'][i]+t*(a['s'][i+1]-a['s'][i])
                    out_s.append(s); out_xy.append(source_at(a['arc'], s))
            heights.extend(h); weights.extend(np.full(count, (hi-lo)/count))
            distances.append(np.stack(pair_s, axis=1)); sources.append(np.stack(pair_xy, axis=1))
            band_distances.append(np.stack(band_s, axis=1)); band_sources.append(np.stack(band_xy, axis=1))
            band_groups.append(slice(band_cursor, band_cursor+len(bh))); band_cursor += len(bh)
        h = np.asarray(heights); w = np.asarray(weights); s = np.concatenate(distances); source = np.concatenate(sources)
        bs = np.concatenate(band_distances); bsource = np.concatenate(band_sources)
        radius = (source-origin)@across-midline
        record = {'max_height_pitch_px': pitch, 'paired_source_open_intervals': paired,
                  'excluded_source_open_intervals': excluded, 'cases': {}}
        arrays = {'source_height': h, 'height_weight': w, 'source': source, 'source_radius': radius,
                  'source_arclength': s, 'band_source': bsource, 'band_source_arclength': bs}
        def band_area(points):
            return float(sum(abs(_signed_area(np.r_[points[g, 0], points[g, 1][::-1]])) for g in band_groups))
        original_band_area = band_area(bsource)
        record['paired_original_band_area_px2'] = original_band_area
        for name in cases:
            target = np.stack([requested(name, side, s[:, i], source[:, i]) for i, side in enumerate(['left', 'right'])], axis=1)
            btarget = np.stack([requested(name, side, bs[:, i], bsource[:, i]) for i, side in enumerate(['left', 'right'])], axis=1)
            query = np.r_[source.reshape(-1, 2), bsource.reshape(-1, 2)]
            moved, closure = actual(name, query)
            destination = moved[:source.size//2].reshape(source.shape); bdest = moved[source.size//2:].reshape(bsource.shape)
            result = {'forward_closure': closure}
            for mode, points, band in [('requested', target, btarget), ('actual_f32_forward', destination, bdest)]:
                displacement = (points-source)@across
                metric, q = bilateral_statistics(radius, displacement, w, h)
                metric['paired_source_height_across_contraction_integral_px2'] = float(np.sum(w*(displacement[:, 0]-displacement[:, 1])))
                metric['paired_material_band_area_px2'] = band_area(band)
                metric['paired_material_band_area_reduction_px2'] = original_band_area-metric['paired_material_band_area_px2']
                metric['source_down_motion_max_abs_px'] = float(abs((points-source)@down).max())
                result[mode] = metric
                arrays[name+'_'+mode+'_q'] = q; arrays[name+'_'+mode+'_destination'] = points
            record['cases'][name] = result
        records.append(record); np.savez_compressed(output/('height-'+str(pitch)+'.npz'), **arrays)
    peaks = {}
    for name in cases:
        peaks[name] = {}
        for side, arc_data in source_arcs.items():
            arc = arc_data['arc']; row = rows[name][arc['id']]
            s = np.unique(np.r_[np.linspace(0., arc['length'], int(np.ceil(arc['length']/.0625))+1), arc_data['s'], row['arclength_px']])
            source = source_at(arc, s); target = requested(name, side, s, source); destination, closure = actual(name, source)
            item = {'forward_closure': closure, 'source_sample_pitch_max_px': .0625, 'sample_count': len(s)}
            for mode, points in [('requested', target), ('actual_f32_forward', destination)]:
                displacement = (points-source)@across; j = int(np.argmax(abs(displacement)))
                item[mode] = {'across_displacement_px': float(displacement[j]), 'source_arclength_px': float(s[j]),
                              'source_down_px': float((source[j]-origin)@down), 'destination_down_px': float((points[j]-origin)@down),
                              'source_xy': source[j].tolist(), 'destination_xy': points[j].tolist()}
            peaks[name][side] = item
        peaks[name]['peak_source_down_gap_abs_px'] = {mode: abs(peaks[name]['left'][mode]['source_down_px']-peaks[name]['right'][mode]['source_down_px'])
                                                    for mode in ['requested', 'actual_f32_forward']}
    report = {'amount': amount, 'baseline_case': str(cases['baseline']), 'candidate_case': str(cases['candidate']),
              'frame': geometry['frame'], 'source_geometry': geometry, 'inputs_sha256': hashes,
              'historical_source_profile_sha256': historical, 'declared_original_source_images': source_images,
              'shared_historical_image_sha256': common_images,
              'comparison': {'same_source_geometry': True, 'same_signed_amount': True, 'same_decoded_fixed_pixels': True,
                             'same_original_source_image_bytes': True, 'same_crop': True,
                             'same_profile_bytes': sha(protocols['baseline']['frozen_target_profile']) == sha(protocols['candidate']['frozen_target_profile']),
                             'same_fixed_png_bytes': sha(cases['baseline'].parent/'fixed-source.png') == sha(cases['candidate'].parent/'fixed-source.png')},
              'paired_height_samples': records, 'whole_reviewed_arc_motion_peaks': peaks,
              'definitions': {'q': '- across_displacement / original_signed_radius',
                              'rms_measure': 'uniform source down-height measure dh over eligible open intervals',
                              'paired_band_area': 'sum of closed material-band polygon areas over eligible intervals, with one-sided endpoints; actual 2D down motion retained',
                              'source_height_integral': 'integral(du_left-du_right) dh; equals area contraction only for down-preserving motion',
                              'peak': 'maximum absolute across displacement on each entire reviewed arc, including unpaired source tail'},
              'limitations': 'Dense sampled actual-field metrics, not continuous extrema proof. Original proxy source arcs and unknown semantics are unchanged. Equal relative edits do not imply anatomical or aesthetic symmetry.',
              'beauty_approval': False}
    if any(sha(path) != digest for path, digest in hashes.items()): raise ValueError('audit inputs changed')
    save(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['baseline', 'candidate', 'output']: parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args(); report = run(args.baseline, args.candidate, args.output)
    print(json.dumps({'amount': report['amount'], 'comparison': report['comparison'],
                      'paired': report['paired_height_samples'][-1]['cases'],
                      'peak_gaps': {k: v['peak_source_down_gap_abs_px'] for k, v in report['whole_reviewed_arc_motion_peaks'].items()}}, indent=2))
