"""Source-only feature-plane associations; no anatomy or edit authorization.

The source128 path remains unchanged. This module records where source eye and
mouth planes intersect it, and the nearest source-path projection of mesh 152.
These are topology/geometry proxies, NOT visible-skin or bone annotations.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FEATURE_INDICES = {
    'leftEye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'rightEye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384],
    'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
}
CERTAINTY = {'mapping_status': 'source_geometric_proxy',
             'visibility_status': 'unknown', 'automatic_edit_authorized': False}


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def simple_path(points, name):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError(name+' must contain finite xy pairs')
    edges = np.roll(p, -1, axis=0)-p
    if np.any(np.linalg.norm(edges, axis=1) < 1e-8):
        raise ValueError(name+' has missing/duplicate rows')
    for i in range(len(p)):
        for j in range(i+2, len(p)):
            if i == 0 and j == len(p)-1:
                continue
            a, b, c, d = p[i], p[(i+1)%len(p)], p[j], p[(j+1)%len(p)]
            # Reject crossings and nonadjacent touching/overlap, including a
            # row-order error that happens to put a vertex on another edge.
            z = [cross(b-a, c-a), cross(b-a, d-a), cross(d-c, a-c), cross(d-c, b-c)]
            if z[0]*z[1] <= 1e-12 and z[2]*z[3] <= 1e-12:
                overlap = np.minimum(np.maximum(a, b), np.maximum(c, d))-np.maximum(np.minimum(a, b), np.minimum(c, d))
                if np.all(overlap >= -1e-9):
                    raise ValueError(name+' has crossing/touching or out-of-order rows')
    return p


def local(points, frame):
    return (np.asarray(points)-frame['origin']) @ np.array([frame['across'], frame['down']]).T


def validate_frame(oval, features, frame):
    vectors = np.array([frame['across'], frame['down']], float)
    if not np.isfinite(vectors).all() or not np.allclose(vectors@vectors.T, np.eye(2), atol=2e-6, rtol=0):
        raise ValueError('frame must be the source orthonormal frame')
    left, right = features['leftEye'].mean(0), features['rightEye'].mean(0)
    direction = right-left
    if direction[0] <= 0 or np.linalg.norm(direction) < 1e-8:
        raise ValueError('source eye identity/side is missing or reversed')
    direction /= np.linalg.norm(direction)
    down = np.array([-direction[1], direction[0]])
    mouth = features['lips'].mean(0)
    if np.dot(mouth-(left+right)*.5, down) < 0:
        down *= -1
    # Source frames are stored as f32; tolerance only covers that rounding.
    tol = 32*np.finfo(np.float32).eps*max(1., np.abs(oval).max())
    if (not np.allclose(frame['origin'], (left+right)*.5, atol=tol, rtol=0)
            or not np.allclose(vectors, [direction, down], atol=2e-6, rtol=0)):
        raise ValueError('frame does not match source feature identity')
    p = local(oval, frame)
    if not np.allclose(frame['bounds'], np.r_[p.min(0), p.max(0)], atol=tol, rtol=0):
        raise ValueError('frame bounds do not match the canonical source oval')


def associate_source(contour, oval, features, frame, canonical_indices):
    if list(canonical_indices) != FACE_OVAL:
        raise ValueError('canonical full36 ordering differs from the historical mapping')
    contour, oval = simple_path(contour, 'source128'), simple_path(oval, 'canonical oval')
    if len(contour) != 128 or len(oval) != 36:
        raise ValueError('source128 and full36 canonical landmarks are required')
    checked = {}
    for name, indices in FEATURE_INDICES.items():
        if name not in features or len(features[name]) != len(indices):
            raise ValueError('missing/full feature identity: '+name)
        checked[name] = simple_path(features[name], name)
    validate_frame(oval, checked, frame)
    p = local(contour, frame)
    edges = np.roll(contour, -1, axis=0)-contour
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    perimeter = float(cumulative[-1])

    def association(i, t):
        s = float(cumulative[i]+t*lengths[i]) % perimeter
        return {'row0': int(i), 'row1': int((i+1)%128), 't': float(t),
                'arclength_px': s, 'source': (contour[i]+t*edges[i]).tolist(), **CERTAINTY}

    def crossings(axis, level):
        q = np.roll(p, -1, axis=0)
        if np.any((np.abs(p[:, axis]-level)<1e-9)&(np.abs(q[:, axis]-level)<1e-9)):
            raise ValueError('ambiguous contour segment lies on a feature plane')
        indices = np.flatnonzero(((p[:, axis]<=level)&(q[:, axis]>level))|
                                 ((q[:, axis]<=level)&(p[:, axis]>level)))
        if len(indices) != 2:
            raise ValueError('ambiguous/missing two-sided contour intersections')
        return [association(i, (level-p[i, axis])/(q[i, axis]-p[i, axis])) for i in indices]

    def plane_sides(level):
        rows = sorted(crossings(1, level), key=lambda r: local([r['source']], frame)[0, 0])
        if not local([rows[0]['source']], frame)[0, 0] < 0 < local([rows[1]['source']], frame)[0, 0]:
            raise ValueError('feature plane does not have both source sides')
        return rows

    axis_rows = sorted(crossings(0, 0.), key=lambda r: local([r['source']], frame)[0, 1])
    top = axis_rows[0]
    chin = oval[FACE_OVAL.index(152)]
    ts = np.clip(np.sum((chin-contour)*edges, axis=1)/lengths**2, 0., 1.)
    candidates = contour+ts[:, None]*edges
    distances = np.linalg.norm(candidates-chin, axis=1)
    ids = np.flatnonzero(np.abs(distances-distances.min())<1e-7)
    if any(np.linalg.norm(candidates[i]-candidates[ids[0]])>1e-7 for i in ids[1:]):
        raise ValueError('ambiguous chin projection onto source path')
    chin_row = association(int(ids[0]), ts[ids[0]])
    chin_row.update({'canonical_id': 152, 'canonical_source': chin.tolist(),
                     'projection_distance_px': float(distances.min()), 'proxy_kind': 'nearest_source_path'})
    mouth = local(checked['lips'], frame)
    mouth_v = float((mouth[:, 1].min()+mouth[:, 1].max())*.5)
    mouth_sides = plane_sides(mouth_v)
    if local([chin_row['source']], frame)[0, 1] <= mouth_v:
        raise ValueError('chin proxy is not below mouth plane')
    # This is an inside-path test, not a detector of occlusion or visible skin.
    for feature_points in checked.values():
        for point in feature_points:
            point_uv = local([point], frame)[0]
            bounds = [local([r['source']], frame)[0, 0] for r in plane_sides(point_uv[1])]
            if not bounds[0] < point_uv[0] < bounds[1]:
                raise ValueError('source feature lies outside its contour sides')
    row_regions = [{'row': i, 'region': 'above_eye_or_unassigned', 'side': None,
                    'visibility_status': 'unknown', 'automatic_edit_authorized': False} for i in range(128)]
    sides = {}
    for side_index, (side, feature) in enumerate([('left', 'leftEye'), ('right', 'rightEye')]):
        eye = local(checked[feature], frame)
        eye_v = float((eye[:, 1].min()+eye[:, 1].max())*.5)
        if not eye_v < mouth_v:
            raise ValueError('eye/mouth source plane order is reversed')
        eye_row = plane_sides(eye_v)[side_index]
        mouth_row = mouth_sides[side_index]
        ordered = []
        for direction in [1, -1]:
            values = [((r['arclength_px']-top['arclength_px'])*direction)%perimeter
                      for r in [eye_row, mouth_row, chin_row]]
            if 0 < values[0] < values[1] < values[2] < perimeter:
                ordered.append((direction, values))
        if len(ordered) != 1:
            raise ValueError('ambiguous/reversed side arclength order')
        direction, values = ordered[0]
        eye_row = {**eye_row, 'feature': feature, 'feature_plane_v': eye_v,
                   'canonical_ids': FEATURE_INDICES[feature]}
        mouth_row = {**mouth_row, 'feature': 'lips', 'feature_plane_v': mouth_v,
                     'canonical_ids': FEATURE_INDICES['lips']}
        regions = []
        for i, label in enumerate(['eye_to_mouth_side_proxy', 'mouth_to_chin_side_proxy']):
            start, end = values[i], values[i+1]
            indices = []
            for row in range(128):
                distance = ((cumulative[row]-top['arclength_px'])*direction)%perimeter
                if start <= distance < end:
                    if row_regions[row]['side'] is not None:
                        raise ValueError('side arcs overlap or cross')
                    row_regions[row].update({'region': label, 'side': side})
                    indices.append(row)
            regions.append({'label': label, 'length_px': end-start,
                            'start_distance_from_top_px': start, 'end_distance_from_top_px': end,
                            'source_rows': indices, **CERTAINTY})
        sides[side] = {'direction_in_source_row_order': direction,
                       'boundaries': {'top_midline': top, 'eye_height': eye_row,
                                      'mouth_height': mouth_row, 'chin_proxy': chin_row},
                       'regions': regions, **CERTAINTY}
    if sides['left']['direction_in_source_row_order'] == sides['right']['direction_in_source_row_order']:
        raise ValueError('left and right use the same contour arc')
    return {'schema': 'source-region-associations-v1', 'source_contour': contour.tolist(),
            'source_frame_unchanged': frame, 'perimeter_px': perimeter,
            'canonical_landmarks': [{'canonical_id': i, 'source': point.tolist()}
                                    for i, point in zip(FACE_OVAL, oval)],
            'sides': sides, 'row_regions': row_regions,
            'anatomical_landmarks': {key: {'status': 'unknown', 'automatic_edit_authorized': False}
                                    for key in ['cheekbone', 'mandibular_angle', 'visible_skin_boundary']},
            'mapping_status': 'source_topology_associated', 'visibility_status': 'unknown',
            'automatic_edit_authorized': False,
            'limitations': ['Canonical indices establish detector topology, not bone or visible tissue.',
                            'Pose/depth and occult-side visibility cannot be recovered from cached xy polygons.',
                            'The unchanged source128 can include hair or attachments; associations do not grant edit ownership.']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_history(current, historical):
    names = {'FACE_OVAL': FACE_OVAL, 'LEFT_EYE': FEATURE_INDICES['leftEye'],
             'RIGHT_EYE': FEATURE_INDICES['rightEye'], 'LIP_OUTER': FEATURE_INDICES['lips']}
    for text in [historical, current]:
        for name, expected in names.items():
            match = re.search(r'const '+name+r'\s*=\s*(\[[^;]+\]);', text)
            if not match or ast.literal_eval(match[1]) != expected:
                raise ValueError('historical canonical ordering mismatch: '+name)
        if 'const points = semanticPolygon(mesh, FACE_OVAL)' not in text:
            raise ValueError('historical graph extraction binding is missing')
    return names


def verify_source_identity(params, reference, audit, size):
    # Compare actual arrays in order, not merely their length or claimed ids.
    if params['landmarks'] != reference['landmarks']:
        raise ValueError('source full36 coordinate identity/order differs from cached detection')
    convention = 'native absolute texel centres +.5 converted once to photograph indices'
    if audit.get('native_and_photo_coordinate_conversion', {}).get('coordinate_convention') != convention:
        raise ValueError('explicit source photograph-index coordinate contract is missing')
    pixels = lambda rows: np.array([[p['x']*size[0]-.5, p['y']*size[1]-.5] for p in rows])
    oval = pixels(params['landmarks'])
    if not np.array_equal(oval, audit['source_oval']):
        raise ValueError('source full36 differs from frozen source audit')
    features = {}
    for key in FEATURE_INDICES:
        value = params['semanticMasks'][key]
        if value != reference['semanticMasks'][key] or value.get('shape') != 'polygon':
            raise ValueError('source feature polygon identity mismatch: '+key)
        features[key] = pixels(value['points'])
    if any(not np.array_equal(features[key], audit['source_eyes'][i])
           for i, key in enumerate(['leftEye', 'rightEye'])):
        raise ValueError('source eyes differ from frozen source audit')
    return oval, features


def overlay(source, result, output):
    from PIL import Image, ImageDraw
    image = Image.open(source).convert('RGB'); draw = ImageDraw.Draw(image)
    contour = result['source_contour']
    draw.line([tuple(p) for p in contour]+[tuple(contour[0])], fill=(220, 220, 220), width=1)
    colors = {'left': (0, 235, 255), 'right': (255, 80, 215)}
    for side, data in result['sides'].items():
        color = colors[side]; direction = data['direction_in_source_row_order']
        for region in data['regions']:
            start_key = 'eye_height' if region['label'].startswith('eye') else 'mouth_height'
            end_key = 'mouth_height' if start_key == 'eye_height' else 'chin_proxy'
            rows = sorted(region['source_rows'], key=lambda row:
                          ((sum(np.linalg.norm(np.array(contour[(i+1)%128])-contour[i]) for i in range(row))
                            -data['boundaries']['top_midline']['arclength_px'])*direction)%result['perimeter_px'])
            line = [data['boundaries'][start_key]['source']]+[contour[row] for row in rows]+[data['boundaries'][end_key]['source']]
            draw.line([tuple(p) for p in line], fill=color, width=2)
        for number, key in enumerate(['eye_height', 'mouth_height', 'chin_proxy'], 1):
            x, y = data['boundaries'][key]['source']
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
            dx = -72 if side == 'left' else 7
            draw.text((x+dx, y+3), f'{side[0].upper()}{number}', fill=color, stroke_width=1, stroke_fill='black')
    for row in result['canonical_landmarks']:
        if row['canonical_id'] in [10, 152, 234, 454, 172, 397]:
            x, y = row['source']; draw.ellipse((x-2, y-2, x+2, y+2), fill='yellow')
            draw.text((x+4, y-12), str(row['canonical_id']), fill='yellow', stroke_width=1, stroke_fill='black')
    draw.rectangle((0, 0, image.width, 38), fill='black')
    draw.text((6, 5), 'MAPPING ONLY / VISIBILITY UNKNOWN', fill='white')
    draw.text((6, 20), '1 eye plane | 2 mouth plane | 3 chin projection (proxy)', fill='white')
    image.save(output)


def generate(output):
    if output.exists():
        raise FileExistsError('refusing to overwrite source-region evidence')
    # Artifacts contain photo-derived data and overlays and must stay ignored.
    if not output.resolve().is_relative_to(ROOT/'test/results'):
        raise ValueError('source-region artifacts must be under ignored test/results')
    definitions_path = ROOT/'apps/web/src/core/landmarks.ts'
    historical_commit = subprocess.check_output(['git', 'rev-parse', 'b7edfad'], cwd=ROOT, text=True).strip()
    historical = subprocess.check_output(['git', 'show', historical_commit+':apps/web/src/core/landmarks.ts'], cwd=ROOT, text=True)
    mapping = verify_history(definitions_path.read_text(), historical)
    base = ROOT/'test/results/semantic-deformation-20261022'
    diagnosis = read(base/'source-proportion-diagnosis-v2/report.json')
    output.mkdir(parents=True)
    rows = {}
    for name in ['user', 'woman', 'hair']:
        paths = [Path(__file__), definitions_path, base/'source-proportion-diagnosis-v2/report.json']
        try:
            audit_path = base/'controller-source-audit'/f'{name}-audit.json'
            audit = read(audit_path); paths.append(audit_path)
            for file, digest in audit['input_sha256'].items():
                if sha(file) != digest:
                    raise ValueError('frozen source input changed: '+file)
            parent = Path(diagnosis[name]['input']); contract_path = parent/'contract.json'
            contract = read(contract_path); prov_path = Path(contract['baseline'])/'provenance.json'
            provenance = read(prov_path)
            graph_path = Path(provenance['source_trace']).parent/'graph.json'
            if sha(graph_path) != provenance['feature_graph_sha256']:
                raise ValueError('frozen graph provenance hash mismatch')
            params = read(graph_path)['operations'][0]['params']
            if name == 'user':
                reference_path = ROOT/'test/results/user-face-contour-20261014/1024/detection.json'
                detection = read(reference_path)
                reference = detection['graph']['operations'][0]['params']
                if reference['landmarks'] != detection['people'][0]['points']:
                    raise ValueError('original graph differs from source detector points')
            else:
                reference_path = ROOT/'test/results/face-core-fresh'/f'portrait-{name}-faceWidth-60.json'
                reference = read(reference_path)['operations'][0]['params']
            size = audit['native_intent_source_proof']['source_dimensions']
            oval, features = verify_source_identity(params, reference, audit, size)
            data_path = parent/'data.npz'
            with np.load(data_path, allow_pickle=False) as data:
                if not np.array_equal(data['target_sources'][:128], audit['source_contour']):
                    raise ValueError('source128 differs from frozen solver input')
            result = associate_source(audit['source_contour'], oval, features, audit['frame_raw'], mapping['FACE_OVAL'])
            paths += [contract_path, prov_path, graph_path, reference_path, data_path, Path(audit['source_image'])]
            result['provenance'] = {'name': name, 'historical_commit': historical_commit,
                'historical_landmarks_source_sha256': hashlib.sha256(historical.encode()).hexdigest(),
                'canonical_mapping': mapping, 'full_dense_mesh_saved': False,
                'graph_to_source_index_conversion': 'normalized_xy * source_size - 0.5 exactly once; source128 and frame unchanged',
                'coordinate_spaces': {'graph': 'normalized image xy multiplied by original source dimensions',
                                      'source_audit': 'photograph pixel indices',
                                      'conversion_scope': 'graph-to-source associations only; never apply to source128 or source frame'},
                'actual36_and_features_match_cached_detection_exactly': True,
                'source128_matches_frozen_input_exactly': True,
                'input_sha256': {str(p): sha(p) for p in paths}}
            (output/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
            overlay(audit['source_image'], result, output/f'{name}-overlay.png')
            rows[name] = {'status': 'mapped_proxies_only', 'visibility_status': 'unknown',
                          'automatic_edit_authorized': False,
                          'sidecar': str(output/f'{name}.json'), 'overlay': str(output/f'{name}-overlay.png')}
        except (ValueError, KeyError, FileNotFoundError) as error:
            rows[name] = {'status': 'rejected', 'reason': str(error), 'automatic_edit_authorized': False,
                          'inputs_sha256': {str(p): sha(p) for p in paths if p.is_file()}}
    (output/'report.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output.resolve()), indent=2))
