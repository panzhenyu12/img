"""Freeze and compare the first v22 source-relative curve candidate.

Only the original user photo at +30/+100 is authorized at this stage.
No detection of edited outputs and no automatic aesthetic rating.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_STAGE = os.environ.get('FACE_V22_OUTPUT_STAGE', 'initial')
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_user_face_contour import raster_coverage
from review_face_refinement import polygon_interior, FEATURES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_freeze():
    frozen = json.loads((HERE / 'production-baseline-sha256.json').read_text())
    changed = {p: {'expected': expected, 'actual': sha(ROOT / p)}
               for p, expected in frozen.items() if sha(ROOT / p) != expected}
    if changed:
        raise RuntimeError(f'production source changed: {changed}')
    return frozen


def snapshot(path):
    digest = sha(path)
    output = HERE / 'binaries' / f'{path.name}-{digest}'
    output.parent.mkdir(exist_ok=True)
    if not output.exists():
        shutil.copy2(path, output)
    if sha(output) != digest:
        raise RuntimeError('binary snapshot differs')
    output.chmod(0o555)
    return output, digest


def montage(paths, box, output):
    panels = [Image.open(path).convert('RGB').crop(box) for path in paths]
    w, h = panels[0].size
    canvas = Image.new('RGB', (w * len(panels), h + 30), '#181c23')
    draw = ImageDraw.Draw(canvas)
    for n, (panel, label) in enumerate(zip(panels, ['Original', 'v19 formal', 'v22 isolated'])):
        canvas.paste(panel, (n*w, 30))
        draw.text((n*w+5, 8), label, fill='white')
    canvas.save(output)


def full_alpha(path, origin, size):
    result = np.zeros((size[1], size[0]), np.float32)
    crop = np.asarray(Image.open(path), np.float32) / 255
    x, y = origin
    result[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
    return result


def row_edge(alpha, y, center):
    # Quantized RGB/parser alpha is a review estimate, not an independent gold.
    values = alpha[y]
    c = int(round(center))
    if values[c] < .5:
        return None
    left = right = c
    while left > 0 and values[left-1] >= .5:
        left -= 1
    while right+1 < len(values) and values[right+1] >= .5:
        right += 1
    crossing_left = left - .5
    crossing_right = right + .5
    if left > 0:
        crossing_left = left - 1 + (.5-values[left-1])/(values[left]-values[left-1]) + .5
    if right+1 < len(values):
        crossing_right = right + (values[right]-.5)/(values[right]-values[right+1]) + .5
    return [float(crossing_left), float(crossing_right), float(crossing_right-crossing_left)]


def geometry_report(debug, old_debug, size):
    geometry = json.loads((debug / 'flow-geometry.json').read_text())
    source = np.asarray(geometry['controlPositions'], np.float64)
    actual = np.asarray(geometry['actualControls'], np.float64)
    requested = np.asarray(geometry['requested'], np.float64)
    feasible = np.asarray(geometry['feasible'], np.float64)
    before_qp = np.asarray(geometry['beforeQP'], np.float64)
    across = np.asarray(geometry['anatomicalAcross'])
    down = np.asarray(geometry['anatomicalDown'])
    origin = np.asarray(geometry['anatomicalOrigin'])
    origin_uv = (source-origin) @ np.stack([across, down], axis=1)
    a_delta = actual-source
    f_delta = feasible-source
    r_delta = requested-source
    face_n = len(before_qp)
    active = np.asarray(geometry['movableControls'], bool)
    face_active = active.copy(); face_active[face_n:] = False
    errors = np.linalg.norm(actual-feasible, axis=1)
    stages = []
    for i in range(face_n):
        stages.append({'index': i, 'source': source[i].tolist(), 'sourceUV': origin_uv[i].tolist(),
                       'requestLateral': float(r_delta[i]@across),
                       'preQPLateral': float((before_qp[i]-source[i])@across),
                       'feasibleLateral': float(f_delta[i]@across),
                       'actualLateral': float(a_delta[i]@across),
                       'actualErrorPixels': float(errors[i]), 'active': bool(active[i])})
    neck = []
    for p, q in geometry['neckRequested']:
        i = int(np.argmin(np.linalg.norm(source-np.asarray(p), axis=1)))
        neck.append({'index': i, 'source': p, 'requested': requested[i].tolist(), 'feasible': q,
                     'actual': actual[i].tolist(), 'sourceUV': origin_uv[i].tolist(),
                     'requestLateral': float(r_delta[i]@across),
                     'feasibleLateral': float(f_delta[i]@across),
                     'actualLateral': float(a_delta[i]@across),
                     'actualErrorPixels': float(errors[i]), 'active': bool(active[i])})
    for side in [-1, 1]:
        entries = [r for r in neck if r['sourceUV'][0]*side > 0]
        entries.sort(key=lambda r: r['sourceUV'][1])
        for n, entry in enumerate(entries):
            if n == 0:
                continue
            prev = entries[n-1]
            dv = entry['sourceUV'][1]-prev['sourceUV'][1]
            if dv <= .01:
                continue
            entry['sourceTangentLateralPerV'] = (entry['sourceUV'][0]-prev['sourceUV'][0])/dv
            entry['actualTangentLateralPerV'] = (
                entry['sourceUV'][0] + entry['actualLateral']
                - prev['sourceUV'][0]-prev['actualLateral'])/dv
    source_alpha = full_alpha(debug / 'source-physical-alpha.png', geometry['origin'], size)
    candidate_alpha = full_alpha(debug / 'transported-physical-alpha.png', geometry['origin'], size)
    old_geometry = json.loads((old_debug / 'flow-geometry.json').read_text())
    old_alpha_path = old_debug / 'full-transported-physical-alpha.png'
    old_alpha = full_alpha(old_alpha_path, old_geometry['origin'], size)
    old_source_alpha = full_alpha(old_debug / 'source-physical-alpha.png', old_geometry['origin'], size)
    source_alpha_same = np.array_equal(source_alpha, old_source_alpha)
    rows = []
    for y in [495, 510, 525, 535, 541, 545, 550, 560, 580, 600, 620, 640, 680, 710]:
        values = {'y': y}
        for label, a in [('source', source_alpha), ('v19', old_alpha), ('v22', candidate_alpha)]:
            center = origin[0] + (y-origin[1])*down[0]/down[1]
            edge = row_edge(a, y, center)
            values[label] = edge
            if edge is not None:
                before = row_edge(a, y-1, center)
                after = row_edge(a, y+1, center)
                if before is not None and after is not None:
                    values[label+'HorizontalEdgeSlope'] = [(after[0]-before[0])*.5, (after[1]-before[1])*.5]
        rows.append(values)
    row = {'folds': geometry['folds'], 'minimumJacobian': geometry['minJac'],
           'jacobianProbeCount': geometry['observed'],
           'maximumForwardInverseErrorPixels': geometry['maxForwardInverseErrorPixels'],
           'sourceWidthReferences': geometry['sourceWidthReferences'],
           'sourceBoundsUV': geometry['anatomicalBounds'],
           'faceMaxRequestedLateralPixels': float(np.max(np.abs(r_delta[:face_n]@across))),
           'faceMaxPreQPLateralPixels': float(np.max(np.abs((before_qp-source[:face_n])@across))),
           'faceMaxFeasibleLateralPixels': float(np.max(np.abs(f_delta[:face_n]@across))),
           'faceMaxActualLateralPixels': float(np.max(np.abs(a_delta[:face_n]@across))),
           'faceActiveMaxErrorPixels': float(errors[face_active].max(initial=0)),
           'allActiveMaxErrorPixels': float(errors[active].max(initial=0)),
           'shootingResidualHistory': geometry['residualHistory'],
           'neckControls': neck, 'faceControls': stages,
           'physicalAlphaHorizontalWidthsAndTangents': rows,
           'alphaCaution': 'source RGB/parser-derived quantized physical union; not independent gold or anatomical semantic boundary',
           'baselineGeometryCache': str(old_debug.relative_to(ROOT)),
           'sourcePhysicalAlphaMatchesV19': bool(source_alpha_same),
           'sourcePhysicalAlphaChangedPixelsFromV19': int(np.count_nonzero(source_alpha != old_source_alpha)),
           'sourcePhysicalAlphaMaximumDeltaFromV19': float(np.abs(source_alpha-old_source_alpha).max(initial=0)),
           'sourcePhysicalAlphaSha256': sha(debug / 'source-physical-alpha.png'),
           'baselineFullPhysicalAlphaSha256': sha(old_alpha_path),
           'candidateFullPhysicalAlphaSha256': sha(debug / 'transported-physical-alpha.png')}
    (debug.parent / f'{debug.name}-curves.json').write_text(json.dumps(row, indent=2)+'\n')
    return row


def main():
    production_source = source_freeze()
    case = json.loads((HERE / 'inputs/manifest.json').read_text())[0]
    source = ROOT / case['canonical']
    if sha(source) != case['canonicalSha256']:
        raise RuntimeError('source digest differs')
    base_graph_path = ROOT / case['sourceGraph']
    if sha(base_graph_path) != case['graphSha256']:
        raise RuntimeError('source-only detector graph differs')
    baseline, b_hash = snapshot(ROOT / '.run/face-anatomical-v20-baseline-image-render')
    candidate, c_hash = snapshot(ROOT / '.run/face-occlusion-v22-candidate-image-render')
    output = HERE / 'outputs' / OUTPUT_STAGE
    output.mkdir(parents=True, exist_ok=True)
    original = Image.open(source).convert('RGBA')
    before = np.asarray(original)
    w, h = original.size
    base_graph = json.loads(base_graph_path.read_text())
    params = base_graph['operations'][0]['params']
    masks = params['semanticMasks']
    coverages = {k: raster_coverage(m, w, h) for k, m in masks.items() if m.get('shape') == 'raster'}
    feature_interiors = {k: polygon_interior(m, (w,h)) for k, m in masks.items() if k in FEATURES and m.get('shape') == 'polygon'}
    subject = raster_coverage(params['subjectMask'], w, h) if params.get('subjectMask', {}).get('shape') == 'raster' else None
    region = params['region']
    xx, yy = np.meshgrid((np.arange(w)+.5)/w, (np.arange(h)+.5)/h)
    outside = ((xx < region['x']) | (xx > region['x']+region['width']) |
               (yy < region['y']) | (yy > region['y']+region['height']))
    report = {'scope': 'only original user photo +30/+100; native frozen comparison, production unchanged',
              'sourceSha256': sha(source), 'seedGraphSha256': sha(base_graph_path),
              'baselineBinarySha256': b_hash, 'candidateBinarySha256': c_hash,
              'baselineImmutableBinary': str(baseline), 'candidateImmutableBinary': str(candidate),
              'productionSourceFreeze': production_source,
              'candidateSourceFreeze': {str(p.relative_to(HERE)): sha(p) for p in (HERE/'src').rglob('*.rs')},
              'input': case, 'rating': 'requires independent original-size review; not automatically eight points',
              'timingCaution': 'debug timings include CLI, JSON/PNG and diagnostic numerical probes; not stable Web rendering',
              'cases': []}
    for amount in [30, 100]:
        graph = json.loads(json.dumps(base_graph))
        graph['operations'][0]['params']['adjustments'] = {'faceWidth': amount}
        graph_path = output / f'faceWidth{amount}.json'
        graph_path.write_text(json.dumps(graph, separators=(',', ':')))
        baseline_path = output / f'faceWidth{amount}-v19.png'
        candidate_path = output / f'faceWidth{amount}-v22.png'
        debug = output / f'debug{amount}'
        times = {}
        for name, binary, destination in [('v19', baseline, baseline_path), ('v22', candidate, candidate_path)]:
            if sha(binary) != (b_hash if name == 'v19' else c_hash):
                raise RuntimeError('binary changed')
            env = os.environ.copy()
            if name == 'v22':
                env['FACE_V22_DEBUG_DIR'] = str(debug)
            begin = time.perf_counter()
            result = subprocess.run([str(binary), '--input', str(source), '--graph', str(graph_path), '--output', str(destination)], env=env, capture_output=True, text=True)
            times[name] = (time.perf_counter()-begin)*1000
            if result.returncode:
                raise RuntimeError(result.stderr)
        old_cached = ROOT / f'test/results/user-face-contour-20261014/1024/face-joint-v19-candidate/faceWidth{amount}.png'
        baseline_matches = np.array_equal(np.asarray(Image.open(baseline_path).convert('RGBA')),
                                          np.asarray(Image.open(old_cached).convert('RGBA')))
        if not baseline_matches:
            raise RuntimeError('baseline does not match v19 diagnostic cache')
        after = np.asarray(Image.open(candidate_path).convert('RGBA'))
        changed = np.any(after != before, axis=2)
        fixed = {k: int(np.count_nonzero(changed & (coverages[k] >= .5))) for k in ['ears','accessories','hair']}
        feature_changes = {k: int(np.count_nonzero(changed & mask)) for k, mask in feature_interiors.items()}
        scene = (subject <= 1e-6) & (coverages['faceContourMask'] <= 1e-6) if subject is not None else None
        row = {'amount': amount, 'sourceSha256': sha(source), 'graphSha256': sha(graph_path),
               'baselineSha256': sha(baseline_path), 'candidateSha256': sha(candidate_path),
               'matchesFrozenV19AllRGBA': baseline_matches, 'nativeInstrumentedTotalMs': times,
               'changedPixels': int(changed.sum()), 'fixedClassChangedPixels': fixed,
               'featureInteriorChangedPixels': feature_changes, 'roiExteriorChangedPixels': int(changed[outside].sum()),
               'sourceAlphaChangedPixels': int(np.count_nonzero(before[:,:,3] != after[:,:,3])),
               'semanticSceneBackgroundChangedPixels': int(changed[scene].sum()) if scene is not None else None,
               'lowNeckYAbove740ChangedPixels': int(changed[740:].sum())}
        old_debug = ROOT / f'test/results/user-face-contour-20261014/1024/face-joint-v19-candidate/debug{amount}'
        curves = geometry_report(debug, old_debug, (w,h))
        row['curves'] = {k: v for k, v in curves.items() if k not in ['neckControls','faceControls']}
        row['hardChecksPassed'] = not (any(fixed.values()) or any(feature_changes.values()) or row['roiExteriorChangedPixels'] or row['sourceAlphaChangedPixels'] or row['semanticSceneBackgroundChangedPixels'] or row['lowNeckYAbove740ChangedPixels'])
        report['cases'].append(row)
        paths = [source, baseline_path, candidate_path]
        for location, box in {'face': (280, 160, 750, 665), 'jaw-neck': (340, 465, 670, 725),
                              'left': (290, 285, 445, 620), 'right': (570, 285, 738, 620)}.items():
            montage(paths, box, output / f'faceWidth{amount}-{location}-1x.png')
        (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: row[k] for k in ['amount','changedPixels','hardChecksPassed','nativeInstrumentedTotalMs']}), flush=True)
    source_freeze()


if __name__ == '__main__':
    main()
