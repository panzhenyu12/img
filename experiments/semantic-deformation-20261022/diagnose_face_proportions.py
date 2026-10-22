"""Read-only source-relative proportions on frozen whole-face edits.

These are geometric measurements, not aesthetic targets or anatomical truth.
No image is decoded for detection, no target is modified, and no field is solved.
"""
import argparse
from pathlib import Path

import numpy as np

from reduced_field import ROOT
from adapt import read, save, sha
from evaluate import load, forward


CASES = {
    'user': 'user-plus60-rust-same-amount-focused-v2',
    'woman': 'woman-plus60-focused-v1',
    'hair': 'hair-plus60-focused-v2',
}


def local(points, frame):
    return (np.asarray(points, float)-frame['origin']) @ np.array([frame['across'], frame['down']]).T


def section(contour, height):
    """Outermost intersections with a closed polygon at one source-frame v."""
    a = np.asarray(contour); b = np.roll(a, -1, axis=0)
    crosses = ((a[:, 1] <= height) & (b[:, 1] > height)) | ((b[:, 1] <= height) & (a[:, 1] > height))
    if np.count_nonzero(crosses) < 2:
        raise ValueError('feature height has no two-sided contour section')
    t = (height-a[crosses, 1])/(b[crosses, 1]-a[crosses, 1])
    x = a[crosses, 0]+t*(b[crosses, 0]-a[crosses, 0])
    return [float(x.min()), float(x.max())], len(x)


def measure(contour, feature):
    lo, hi = feature.min(axis=0), feature.max(axis=0)
    level = (lo[1]+hi[1])*.5
    bounds, count = section(contour, level)
    clearances = []
    for point in feature:
        edge, _ = section(contour, point[1])
        clearances.append([point[0]-edge[0], edge[1]-point[0]])
    width = float(hi[0]-lo[0]); face_width = bounds[1]-bounds[0]
    return {'feature_width_px': width, 'feature_height_px': float(hi[1]-lo[1]),
            'source_frame_section_v': float(level), 'face_width_at_feature_height_px': face_width,
            'feature_to_face_width_ratio': width/face_width,
            'minimum_left_right_horizontal_margin_px': np.min(clearances, axis=0).tolist(),
            'section_intersection_count': count}


def diagnose(name, case):
    inp = Path(read(case/'protocol.json')['input'])
    c = read(inp/'contract.json'); source = read(inp/'source-contract.json')
    base = Path(c['baseline']); prov = read(base/'provenance.json')
    graph_path = Path(prov['source_trace']).parent/'graph.json'
    if sha(graph_path) != prov['feature_graph_sha256']:
        raise ValueError('source graph hash mismatch')
    if name == 'user':
        frame = source['controller_request']['frame']
    else:
        evidence = ROOT/'test/results/face-source-relative-width-20261020/whole-face-width-source-applicability-v2'/f'{name}-source-only.json'
        frame = read(evidence)['frame']
    graph = read(graph_path)['operations'][0]['params']
    size = prov['control_trace']['source_evidence']['reference']['source_dimensions']
    points = np.asarray([t['source'] for t in source['targets'][:128]])
    desired = np.asarray([t['destination'] for t in source['targets'][:128]])
    actual = np.asarray(read(case/'final-contour.json')['actual'])
    contours = {key: local(value, frame) for key, value in [('source', points), ('requested', desired), ('actual', actual)]}
    field = load(case); feature_rows = {}; feature_points = {}
    for feature in ['lips', 'nose', 'leftEye', 'rightEye', 'leftBrow', 'rightBrow']:
        poly = np.asarray([[p['x']*size[0], p['y']*size[1]] for p in graph['semanticMasks'][feature]['points']])
        mapped, _ = forward(field, poly)
        source_local, actual_local = local(poly, frame), local(mapped, frame)
        feature_points[feature] = (source_local, actual_local)
        feature_rows[feature] = {
            'source': measure(contours['source'], source_local),
            'requested_contour_with_fixed_source_feature': measure(contours['requested'], source_local),
            'actual': measure(contours['actual'], actual_local),
            'maximum_actual_feature_motion_px': float(np.linalg.norm(mapped-poly, axis=1).max()),
            'actual_feature_translation_px': np.mean(mapped-poly, axis=0).tolist(),
            'translation_residual_max_px': float(np.linalg.norm(mapped-poly-np.mean(mapped-poly, axis=0), axis=1).max()),
        }
    bands = []
    # Source-frame levels are geometric descriptors, not labelled cheekbones or jaw angles.
    for h in [.1, .3, .5, .7, .85, .9]:
        v = frame['bounds'][1]+h*(frame['bounds'][3]-frame['bounds'][1])
        row = {'normalized_source_frame_height': h}
        for key, curve in contours.items():
            sides, count = section(curve, v)
            row[key+'_width_px'] = sides[1]-sides[0]
            row[key+'_intersection_count'] = count
        bands.append(row)
    geometry = {}
    for key, curve in contours.items():
        span = np.ptp(curve, axis=0)
        geometry[key] = {'projected_width_px': float(span[0]), 'projected_height_px': float(span[1]),
                         'height_width_ratio': float(span[1]/span[0])}
    combined_eyes = {}
    for index, stage in enumerate(['source', 'actual']):
        pair = [feature_points[feature][index] for feature in ['leftEye', 'rightEye']]
        combined = np.concatenate(pair)
        row = measure(contours[stage], combined)
        row['eye_polygon_centroid_separation_px'] = float(np.linalg.norm(pair[1].mean(0)-pair[0].mean(0)))
        combined_eyes[stage] = row
    paths = [inp/'contract.json', inp/'source-contract.json', graph_path, case/'final-contour.json', case/'photo-trace/inverse.f32', Path(__file__)]
    return {'name': name, 'case': str(case), 'input': str(inp),
            'input_hashes': {str(path): sha(path) for path in paths}, 'frame': frame,
            'features': feature_rows, 'combined_eyes': combined_eyes,
            'width_bands': bands, 'contour_extent': geometry,
            'limitations': [
                'Source semantic polygons are cached detection-derived evidence, not reviewed anatomical ground truth.',
                'The 36 graph landmarks are an unlabelled oval; height bands cannot be called exact cheekbones or jaw angles.',
                'Margins measure horizontal polygon separation, not tissue thickness, safe clinical limits or beauty.',
                'Requested-contour fixed-feature ratios are a counterfactual descriptor; eyes actually translate under part rules.',
                'Graph polygons use normalized coordinates times original dimensions, as existing mask rasterization does.',
                'Measurements do not infer an ideal facial ratio, new target law or an aesthetic score.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite diagnostics')
    result = {name: diagnose(name, ROOT/'test/results/semantic-deformation-20261022'/case) for name, case in CASES.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    for name, row in result.items():
        print(name, 'mouth', row['features']['lips'], 'extent', row['contour_extent'])
