"""Freeze a reproducible target-only jaw QP candidate; never render a photo."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import scipy

from jaw_collocation import _prepare, sample_jaw_targets, independent_jaw_probes
from jaw_curve_qp import solve_jaw_curve_profile, evaluate_jaw_curve_profile
from jaw_target_profile import _validate, load_profile, apply_profile_samples, profile_contour_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check_protocol(path):
    protocol = json.loads(path.read_text())
    inputs = protocol.get('inputs_sha256', {})
    evidence = {p: digest for p, digest in inputs.items()
                if Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg', '.json']}
    if (not any(Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg'] for p in evidence)
            or not any(Path(p).suffix.lower() == '.json' for p in evidence)):
        raise ValueError('source protocol must bind image and JSON evidence')
    if any(sha(p) != digest for p, digest in evidence.items()):
        raise ValueError('frozen source image or evidence changed')
    return protocol, evidence


def snapshot_code(output):
    folder = output/'executed-code'; folder.mkdir()
    original = Path(__file__).resolve().parent
    records = []
    for name in ['export_jaw_curve_profile.py', 'jaw_curve_qp.py', 'jaw_collocation.py', 'jaw_target_profile.py']:
        source, target = original/name, folder/name
        digest = sha(source); shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError('code changed during snapshot')
        records.append({'source': str(source), 'snapshot': str(target), 'sha256': digest})
    save(folder/'manifest.json', records)
    return folder, records


def verify_interfaces_and_area(geometry, profile):
    """Check the actual adapter and an independently assembled closed polygon."""
    prepared, _ = _validate(profile, geometry, profile['amount'])
    p, edges, lengths, cumulative, _, across, _, arcs, _ = prepared
    by_id = {row['arc_id']: row for row in profile['profiles']}
    nodes = np.unique(np.r_[cumulative[:-1], *[
        (arc['start']+arc['direction']*np.asarray(by_id[arc['id']]['arclength_px'])) % cumulative[-1]
        for arc in arcs]])
    ids = np.minimum(np.searchsorted(cumulative, nodes, side='right')-1, len(p)-1)
    source = p[ids]+((nodes-cumulative[ids])/lengths[ids])[:, None]*edges[ids]
    target = source.copy()
    for arc in arcs:
        s = ((nodes-arc['start'])*arc['direction']) % cumulative[-1]
        inside = s <= arc['length']; row = by_id[arc['id']]
        target[inside] += np.interp(s[inside], row['arclength_px'], row['across_displacement_px'])[:, None]*across
    area = lambda points: float(.5*np.sum(points[:, 0]*np.roll(points[:, 1], -1)-points[:, 1]*np.roll(points[:, 0], -1)))
    actual_area = area(target)-area(source)
    area_error = actual_area-profile['required_signed_area_delta_px2']
    if abs(area_error) > max(1e-8, abs(actual_area)*1e-10):
        raise ValueError('independent closed-polygon area differs from required continuous beta area')
    queries = 0; maximum = 0.
    for samples in [sample_jaw_targets(geometry, profile['amount'], spacing=2.),
                    independent_jaw_probes(geometry, profile['amount'], spacing=.5, collocation_spacing=2.)]:
        applied = apply_profile_samples(samples, profile, geometry)
        if (not np.array_equal(applied['source'], samples['source'])
                or not np.array_equal(applied['weights'], samples['weights'])
                or applied['controls'] != samples['controls']):
            raise ValueError('profile adapter altered source/quadrature/control metadata')
        for arc in arcs:
            selected = np.array([i for i, c in enumerate(samples['controls']) if c['arc_id'] == arc['id']], int)
            s = [samples['controls'][i]['arc_arclength_px'] for i in selected]
            expected = evaluate_jaw_curve_profile(geometry, profile, arc['id'], s)['target']
            maximum = max(maximum, float(np.abs(applied['target'][selected]-expected).max(initial=0.)))
            queries += len(selected)
    contour = profile_contour_targets(geometry, profile)
    if contour.shape != p.shape or not np.isfinite(contour).all() or maximum > 1e-10:
        raise ValueError('adapter does not reproduce exported profile')
    return {'closed_polygon_signed_area_delta_px2': actual_area,
            'closed_polygon_area_error_px2': area_error,
            'adapter_query_count': queries, 'adapter_max_target_error_px': maximum,
            'adapter_contour_shape': list(contour.shape), 'no_image_render': True}


def export(protocol_path, amount, smoothing, output, *, node_spacing=2., window_radius=16., radius_fraction=.13):
    output = Path(output).resolve(); protocol_path = Path(protocol_path).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite frozen candidate directory')
    protocol, evidence = check_protocol(protocol_path)
    protocol_hash = sha(protocol_path)
    output.mkdir(parents=True)
    code_folder, code_records = snapshot_code(output)
    frozen_protocol = output/'source-protocol.json'
    shutil.copy2(protocol_path, frozen_protocol)
    if sha(frozen_protocol) != protocol_hash:
        raise ValueError('source protocol changed during snapshot')
    parameters = {'amount': amount, 'max_radius_fraction': radius_fraction, 'node_spacing_px': node_spacing,
                  'smoothing_length_px': smoothing, 'window_radius_px': window_radius}
    provenance = {'source_protocol': str(frozen_protocol), 'source_protocol_sha256': protocol_hash,
        'original_source_protocol': str(protocol_path),
        'qp_source_path': str(code_folder/'jaw_curve_qp.py'), 'qp_source_sha256': sha(code_folder/'jaw_curve_qp.py'),
        'export_source_path': str(code_folder/'export_jaw_curve_profile.py'),
        'export_source_sha256': sha(code_folder/'export_jaw_curve_profile.py'),
        'executed_code_manifest': str(code_folder/'manifest.json'),
        'executed_code_manifest_sha256': sha(code_folder/'manifest.json'),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                    'platform': platform.platform()},
        'parameters': parameters, 'source_evidence_sha256': evidence}
    result = solve_jaw_curve_profile(protocol['source_geometry'], **parameters)
    result.update(provenance)
    if (any(sha(row['source']) != row['sha256'] for row in code_records)
            or sha(protocol_path) != protocol_hash or any(sha(path) != digest for path, digest in evidence.items())):
        raise ValueError('executed code or source evidence changed during solve')
    if not result['report']['accepted_numerically']:
        save(output/'qp-report.json', result)
        raise ValueError('QP did not pass all numerical checks; report saved without an executable profile')
    result['verification'] = verify_interfaces_and_area(protocol['source_geometry'], result)
    report_path = output/'qp-report.json'; save(report_path, result)
    profile = copy.deepcopy(result)
    profile.update(qp_report_path=str(report_path), qp_report_sha256=sha(report_path))
    profile_path = output/'profile.json'; save(profile_path, profile)
    load_profile(profile_path, protocol['source_geometry'], amount)
    summary = {'profile_path': str(profile_path), 'profile_sha256': sha(profile_path),
               'parameters': parameters, 'source_geometry_sha256': result['source_geometry_sha256'],
               'required_signed_area_delta_px2': result['required_signed_area_delta_px2'],
               'candidate_signed_area_delta_px2': result['candidate_signed_area_delta_px2'],
               'verification': result['verification'],
               'peak_displacement_px': {p['arc_id']: float(np.max(np.abs(p['across_displacement_px']))) for p in profile['profiles']},
               'continuous_check': result['report']['continuous_check'], 'aesthetic_approval': False}
    save(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--amount', type=float, required=True)
    parser.add_argument('--smoothing', type=float, default=4., help='physical smoothing length in source pixels')
    parser.add_argument('--node-spacing', type=float, default=2.)
    parser.add_argument('--window-radius', type=float, default=16.)
    parser.add_argument('--radius-fraction', type=float, default=.13)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.protocol, args.amount, args.smoothing, args.output,
        node_spacing=args.node_spacing, window_radius=args.window_radius,
        radius_fraction=args.radius_fraction), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
