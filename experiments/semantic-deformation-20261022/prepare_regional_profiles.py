"""Source-height profile plus an explicitly area-matched uniform control.

These are exploratory edit intents, not ideal anatomical proportions. No
photograph is rendered and no displacement solver is called here.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from reduced_field import ROOT
from adapt import read, save, sha


def invoke(binary, request):
    p = subprocess.run([str(binary)], input=json.dumps(request), text=True,
                       capture_output=True, check=True)
    return read_output(p.stdout)


def read_output(text):
    output = json.loads(text)
    return np.asarray([row['target'] for row in output['targets']], float)


def area(points):
    p = np.asarray(points, float); q = np.roll(p, -1, axis=0)
    return float(abs(np.sum(p[:, 0]*q[:, 1]-p[:, 1]*q[:, 0]))*.5)


def prepare(parent, audit_path, binary, output):
    if output.exists():
        raise FileExistsError('refusing to overwrite profile evidence')
    contract = read(parent/'contract.json'); audit = read(audit_path)
    provenance = read(Path(contract['baseline'])/'provenance.json')
    graph_path = Path(provenance['source_trace']).parent/'graph.json'
    if sha(graph_path) != provenance['feature_graph_sha256']:
        raise ValueError('source graph changed')
    if sha(Path(contract['source_photo'])) != contract['source_photo_sha256']:
        raise ValueError('source photograph changed')
    source = np.load(parent/'data.npz', allow_pickle=False)['target_sources'][:128]
    if not np.array_equal(source, audit['source_contour']):
        raise ValueError('source audit belongs to another contour')
    frame = audit['frame_raw']; top, bottom = frame['bounds'][1], frame['bounds'][3]
    size = provenance['control_trace']['source_evidence']['reference']['source_dimensions']
    masks = read(graph_path)['operations'][0]['params']['semanticMasks']
    levels = {}
    for name in ['leftEye', 'rightEye', 'lips']:
        poly = np.array([[p['x']*size[0], p['y']*size[1]] for p in masks[name]['points']])
        v = (poly-frame['origin']) @ np.asarray(frame['down'])
        levels[name] = float(((v.min()+v.max())*.5-top)/(bottom-top))
    eye = (levels['leftEye']+levels['rightEye'])*.5
    mouth = levels['lips']
    if not 0 < eye < mouth < .94:
        raise ValueError('source proxy heights do not support the declared trial')
    profile = {'knots': [{'height': h, 'gain': g} for h, g in
                        [(0., 1.), (eye, 1.), (mouth, .60), (.94, 1.), (1., 1.)]]}
    request = {'frame': {**frame, 'midline_u': 0.},
               'calibration': {'max_radius_fraction': audit['calibration_k_full'], 'chin_fade_start': .88},
               'amount': contract['requested_amount'],
               'contour': [{'id': str(i), 'source': p.tolist()} for i, p in enumerate(source)],
               'parts': [], 'relations': []}
    old = invoke(binary, request)
    regional = invoke(binary, {**request, 'regional_profile': profile})
    original_area, old_area, regional_area = area(source), area(old), area(regional)
    difference = original_area-old_area
    if abs(difference) < 1e-10:
        raise ValueError('zero edit has no area-matching contrast')
    gain = (original_area-regional_area)/difference
    if not .60 <= gain <= 1.:
        raise ValueError('area-matched control left the declared gain range')
    uniform = {'knots': [{'height': 0., 'gain': gain}, {'height': 1., 'gain': gain}]}
    matched = invoke(binary, {**request, 'regional_profile': uniform})
    if abs(area(matched)-regional_area) > 1e-7*original_area:
        raise ValueError('explicit uniform control did not match requested polygon area')
    output.mkdir(parents=True, exist_ok=False)
    save(output/'regional.json', profile); save(output/'uniform-area-matched.json', uniform)
    paths = [parent/'contract.json', parent/'data.npz', audit_path, graph_path, binary, Path(__file__)]
    report = {'inputs_sha256': {str(p.resolve()): sha(p) for p in paths},
              'source_proxy_heights': levels, 'amount': contract['requested_amount'],
              'uniform_control_gain': gain, 'original_contour_area': original_area,
              'whole_width_requested_area': old_area, 'regional_requested_area': regional_area,
              'uniform_requested_area': area(matched),
              'regional_area_change_relative_to_whole_width': gain,
              'uniform_control_is_explicitly_weaker_than_whole_width': True,
              'profile': profile, 'reference_request': request,
              'sampling_and_geometry_solver_not_changed': True,
              'anatomical_ground_truth_or_aesthetic_approval': False,
              'source_contour': source.tolist(), 'regional_targets': regional.tolist(),
              'uniform_targets': matched.tolist(), 'whole_width_targets': old.tolist()}
    save(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'audit', 'binary', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    r = prepare(a.parent.resolve(), a.audit.resolve(), a.binary.resolve(), a.output.resolve())
    print({k: r[k] for k in ['source_proxy_heights', 'uniform_control_gain', 'regional_requested_area', 'uniform_requested_area']})
