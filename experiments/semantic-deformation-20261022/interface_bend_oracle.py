"""Sample source, requested and actual interface bending from coordinates.

This diagnostic never calls the solver's chord/constraint helpers. Actual
positions are found by inverting the saved final f32 map, not interpolating
saved actual curve samples. Dense samples are not a continuous certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sample_scalar(field, points):
    field = np.asarray(field, float); points = np.asarray(points, float)
    height, width = field.shape
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or np.any(points < 0.) or np.any(points > [width-1, height-1])):
        raise ValueError('query outside original scalar grid')
    cells = np.minimum(np.floor(points).astype(int), [width-2, height-2])
    x, y = cells.T; tx, ty = (points-cells).T
    return ((1-tx)*(1-ty)*field[y, x] + tx*(1-ty)*field[y, x+1]
            + (1-tx)*ty*field[y+1, x] + tx*ty*field[y+1, x+1])


def chord_metrics(triples, frame):
    """Reverse cross matches the source curve's across/down turn convention."""
    local = np.asarray(triples, float) @ np.asarray(frame, float).T
    first, second = local[:, 1]-local[:, 0], local[:, 2]-local[:, 1]
    numerator = second[:, 0]*first[:, 1]-first[:, 0]*second[:, 1]
    dot = np.sum(first*second, axis=1)
    heights = np.column_stack([first[:, 1], second[:, 1]])
    if np.any(heights <= 0.):
        raise ValueError('visible interface must remain height-monotone')
    return {'numerator_px2': numerator, 'dot_px2': dot,
            'turn_degrees': np.degrees(np.arctan2(numerator, dot)),
            'chord_lengths_px': np.column_stack([np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)]),
            'chord_heights_px': heights,
            'chord_slopes': np.column_stack([first[:, 0]/first[:, 1], second[:, 0]/second[:, 1]])}


def band_margins(source, changed, band, threshold=1e-8):
    ns, nt = source['numerator_px2'], changed['numerator_px2']
    sign = np.sign(ns)
    return sign*(nt-band[0]*ns), sign*(band[1]*ns-nt), abs(ns) > threshold


def interpolation(points, knots, positions):
    positions = np.asarray(positions, float)
    if np.any(positions < 0.) or np.any(positions > knots[-1]):
        raise ValueError('source arclength is outside its original curve')
    return np.column_stack([np.interp(positions, knots, points[:, axis]) for axis in range(2)])


def summary(source, changed, centers, source_centers, band):
    low, high, known = band_margins(source, changed, band)
    ids = np.flatnonzero(known)
    result = {'known_source_samples': int(known.sum()), 'unknown_source_samples': int((~known).sum()),
              'lower_band_violations_below_negative_1e_minus_8_px2': int(np.count_nonzero(low[known] < -1e-8)),
              'upper_band_violations_below_negative_1e_minus_8_px2': int(np.count_nonzero(high[known] < -1e-8)),
              'maximum_abs_turn_change_degrees': float(abs(changed['turn_degrees']-source['turn_degrees']).max()),
              'maximum_abs_mean_slope_change': float(abs(np.mean(changed['chord_slopes']-source['chord_slopes'], axis=1)).max()),
              'maximum_abs_chord_height_change_px': float(abs(changed['chord_heights_px']-source['chord_heights_px']).max()),
              'chord_length_ratio_range': [float((changed['chord_lengths_px']/source['chord_lengths_px']).min()),
                                           float((changed['chord_lengths_px']/source['chord_lengths_px']).max())]}
    for label, margins in [('lower', low), ('upper', high)]:
        if not len(ids):
            result[label+'_worst'] = None; continue
        i = ids[np.argmin(margins[ids])]
        result[label+'_worst'] = {'margin_px2': float(margins[i]), 'source_arclength_px': float(centers[i]),
            'source_xy': source_centers[i].tolist(), 'source_numerator_px2': float(source['numerator_px2'][i]),
            'changed_numerator_px2': float(changed['numerator_px2'][i]),
            'source_turn_degrees': float(source['turn_degrees'][i]), 'changed_turn_degrees': float(changed['turn_degrees'][i])}
    return result


def run(source_report_path, photo_report_path, output, pitch=.03125, band=(.5, 1.5)):
    source_report_path = Path(source_report_path).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite interface bend evidence')
    if not np.isfinite(pitch) or pitch <= 0.:
        raise ValueError('positive finite original-arclength sampling pitch required')
    reference = read(source_report_path); bound = {str(source_report_path): sha(source_report_path)}
    interface_paths = [Path(p) for p in reference['inputs_sha256'] if Path(p).name == 'source-interface.json']
    if len(interface_paths) != 1:
        raise ValueError('one frozen original interface source required')
    interface_path = interface_paths[0]; interface = read(interface_path)
    bound[str(interface_path)] = reference['inputs_sha256'][str(interface_path)]
    state_path = source_report_path.parent/'diagnostic-source-state.npz'
    bound[str(state_path)] = reference['diagnostic_source_state_sha256']
    if any(sha(p) != h for p, h in bound.items()):
        raise ValueError('frozen source evidence changed')
    state = np.load(state_path); grid = reference['grid']; origin = np.asarray(grid['origin'])
    spacing = np.asarray(grid['spacing']); mapping = state['mapping']; scalar = np.zeros(len(mapping))
    valid = mapping >= 0; scalar[valid] = state['scalar_solution'][mapping[valid]]
    scalar = scalar.reshape(grid['height'], grid['width'])
    frame = np.array([reference['source_geometry']['frame'][key] for key in ['across', 'down']])
    if not np.allclose(frame@frame.T, np.eye(2), atol=1e-10, rtol=0.):
        raise ValueError('original orthonormal frame required')
    final_field = None
    if photo_report_path is not None:
        photo_report_path = Path(photo_report_path).resolve(); photo = read(photo_report_path)
        if (Path(photo['source_report']).resolve() != source_report_path
                or photo['inputs_sha256'].get(str(source_report_path)) != sha(source_report_path)):
            raise ValueError('final map does not bind this original source solution')
        field_path = photo_report_path.parent/'inverse-field.npy'
        bound.update({str(photo_report_path): sha(photo_report_path), str(field_path): photo['inverse_field_sha256']})
        if sha(field_path) != bound[str(field_path)]:
            raise ValueError('frozen final inverse map changed')
        final_field = np.load(field_path)
        if final_field.dtype != np.dtype('<f4'):
            raise ValueError('actual oracle requires saved final float32 inverse field')
        from forward import inverse_points
    output.mkdir(parents=True)
    report = {'schema': 'independent-interface-bend-sampled-oracle-v1', 'source_report': str(source_report_path),
              'photo_report': str(photo_report_path) if photo_report_path else None,
              'source_arclength_pitch_px': pitch, 'diagnostic_numerator_band': list(band),
              'known_source_threshold_px2': 1e-8, 'violation_report_tolerance_px2': 1e-8,
              'source_continuous_accepted': reference['source_continuous_accepted'],
              'actual_positions': 'inverse of frozen final f32 map at each original source point; no saved-curve interpolation',
              'inputs_sha256': bound, 'curves': [], 'image_approval': False,
              'limits': 'Dense original-arclength samples only, not a continuous proof. Radii 6/24 are diagnostics, not new acceptance thresholds. Numerator ratio does not certify turn-angle ratio.'}
    for curve_index, segment in enumerate(interface['segments']):
        points = np.asarray(segment['source_points'], float)
        knots = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
        row = {'id': segment['id'], 'radii': []}
        if np.any(np.diff(points[:, 1]) <= 0.):
            raise ValueError('specific source-y witnesses require height-monotone original points')
        for radius in [6., 12., 24.]:
            length = knots[-1]
            if length <= 2*radius:
                continue
            witness_y = np.array([550., 560., 565., 570., 575., 580.])
            witness_y = witness_y[(witness_y >= points[0, 1]) & (witness_y <= points[-1, 1])]
            witness_s = np.interp(witness_y, points[:, 1], knots)
            fit = (witness_s >= radius) & (witness_s <= length-radius)
            witness_y, witness_s = witness_y[fit], witness_s[fit]
            centers = np.unique(np.r_[np.arange(radius, length-radius, pitch), length-radius, witness_s])
            positions = centers[:, None] + np.array([-radius, 0., radius])
            positions[centers == length-radius, 2] = length
            source = interpolation(points, knots, positions.ravel()).reshape(-1, 3, 2)
            motion = sample_scalar(scalar, (source.reshape(-1, 2)-origin)/spacing).reshape(-1, 3)
            target = source + motion[..., None]*frame[0]
            variants = {'source': source, 'target': target}; closure = None
            if final_field is not None:
                actual, closure = inverse_points(final_field, source.reshape(-1, 2)-origin)
                variants['actual'] = (actual+origin).reshape(-1, 3, 2)
            metrics = {key: chord_metrics(value, frame) for key, value in variants.items()}
            sample_path = output/f'interface-{curve_index}-radius-{int(radius)}.npz'
            np.savez_compressed(sample_path, source_arclength_px=centers,
                **{key+'_'+metric: value for key, values in metrics.items() for metric, value in values.items()})
            case = {'radius_px': radius, 'center_count': len(centers), 'actual_inverse_closure': closure,
                    'samples_sha256': sha(sample_path), 'samples_path': str(sample_path),
                    'target': summary(metrics['source'], metrics['target'], centers, source[:, 1], band),
                    'witnesses': []}
            if 'actual' in metrics:
                case['actual'] = summary(metrics['source'], metrics['actual'], centers, source[:, 1], band)
            for y, s in zip(witness_y, witness_s):
                i = int(np.searchsorted(centers, s))
                case['witnesses'].append({'source_y': float(y), 'source_arclength_px': float(s),
                    **{key: {name: np.asarray(value[i]).tolist() for name, value in values.items()}
                       for key, values in metrics.items()}})
            row['radii'].append(case)
        report['curves'].append(row)
    if any(sha(p) != h for p, h in bound.items()):
        raise ValueError('frozen inputs changed during the independent audit')
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(output/'report.json'), 'curves': len(report['curves']), 'image_approval': False}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--photo-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pitch', type=float, default=.03125)
    args = parser.parse_args()
    run(args.source_report, args.photo_report, args.output, args.pitch)
