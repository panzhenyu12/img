"""Fresh-source geometry for the experimental eye-closing replay.

Only original RGB and the explicitly supplied full-image observation are read.
The imported curve primitives have no historical sample IO. Curves describe
estimated anatomy; they are not an optical matte or hidden-material evidence.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/eye-height-layers-20261117'))
from source_curves import BAND, EYES, fit_curve, frame, light, read_rgb, redness

# Canonical perimeters from apps/web/src/core/landmarks.ts, never convex hulls.
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
             58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NATIVE_COORDINATE_ADAPTER = {
    'sourceObservation': 'MediaPipe normalized * dimensions is physical image XY',
    'sourceSampler': 'physical (x+.5,y+.5) pixel centers; read_rgb subtracts .5',
    'nativeSampler': 'integer pixel-index XY; graph geometry is normalized * dimensions',
    'graphConversion': '(MediaPipe normalized * dimensions - [.5,.5]) / dimensions',
    'region': 'original normalized FACE_OVAL bounds plus 20% padding; normalized boundary contract is unchanged',
    'applications': 1,
    'outsidePixelCenterDomain': 'reject; no point clamping',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_observation(source_path, observation_path, size, face_index=None):
    """Validate one complete observation; never silently choose among faces."""
    observation_path = Path(observation_path)
    obj = json.loads(observation_path.read_text())
    width, height = size
    expected = {'width': width, 'height': height}
    if obj.get('dimensions') != expected:
        raise ValueError('observation dimensions differ from the original source')
    full = [row for row in obj.get('rows', []) if row.get('id') == 'full']
    if len(full) != 1 or full[0].get('box') != [0, 0, width, height]:
        raise ValueError('exactly one full-image observation with matching box is required')
    faces = full[0].get('sourceLandmarks', [])
    if not faces or full[0].get('faceCount') != len(faces):
        raise ValueError('full-image face count is absent or inconsistent')
    if face_index is None:
        if len(faces) != 1:
            raise ValueError('multiple observed faces require an explicit --face-index')
        face_index = 0
    if not 0 <= face_index < len(faces):
        raise ValueError('selected face index does not exist in this observation')
    points = faces[face_index]
    if len(points) < 478:
        raise ValueError('complete MediaPipe face and iris observation requires at least 478 points')
    normalized = np.asarray([[point['x'], point['y']] for point in points], dtype=float)
    if normalized.shape != (len(points), 2) or not np.isfinite(normalized).all():
        raise ValueError('finite two-dimensional source landmarks are required')
    needed = set(LEFT_BROW + RIGHT_BROW + LEFT_EYE + RIGHT_EYE + FACE_OVAL)
    for spec in EYES.values():
        needed.update(spec['upper'] + spec['lower'] + spec['ring'] + [spec['iris']])
    selected = normalized[sorted(needed)]
    if np.any((selected < 0) | (selected > 1)):
        raise ValueError('required anatomical observation leaves the source canvas')
    binding_path = observation_path.with_name('report.json')
    binding = {'available': binding_path.is_file()}
    if binding_path.is_file():
        raw = json.loads(binding_path.read_text())
        for key, actual in [('sourceSHA256', sha(source_path)),
                            ('observationsSHA256', sha(observation_path))]:
            if raw.get(key) != actual:
                raise ValueError(f'observation binding report has a mismatched {key}')
        binding.update(path=str(binding_path.resolve()), sha256=sha(binding_path), verified=True)
    return normalized * [width, height], normalized, {
        'faceIndex': face_index, 'observedFaceCount': len(faces),
        'landmarkCount': len(points), 'binding': binding,
        'coordinateContract': 'normalized * image dimensions; physical pixel centers are x+.5,y+.5',
    }


def make_graph(normalized, size, amount):
    if amount not in (0, -15, -30):
        raise ValueError('this experiment supports only fixed closing strengths 0/-15/-30')
    dimensions = np.asarray(size, dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 1):
        raise ValueError('native graph requires the original width and height')
    # Source curves use physical centers (x+.5,y+.5), whereas the frozen native
    # field uses integer sampling indices. Convert graph points exactly once;
    # do not shift the original observation or the source curve frame.
    pixel_index = np.asarray(normalized, dtype=float)*dimensions-.5
    required = sorted(set(LEFT_BROW+RIGHT_BROW+LEFT_EYE+RIGHT_EYE+FACE_OVAL))
    if not np.isfinite(pixel_index[required]).all() or np.any(
            (pixel_index[required] < 0) | (pixel_index[required] > dimensions-1)):
        raise ValueError('adapted native graph point leaves pixel-center domain; no clamping')
    native_normalized = pixel_index/dimensions

    def polygon(ids):
        return [{'x': float(native_normalized[i, 0]), 'y': float(native_normalized[i, 1])} for i in ids]

    # Region is an outer normalized admission boundary, not a feature point.
    # Keep its physical-coordinate contract instead of applying the point shift.
    outline = np.asarray(normalized)[FACE_OVAL]
    low, high = outline.min(axis=0), outline.max(axis=0)
    padding = (high - low) * .2
    low, high = np.maximum(0, low - padding), np.minimum(1, high + padding)
    if np.any(high <= low):
        raise ValueError('source face region has no extent')
    masks = {name: {'shape': 'polygon', 'points': polygon(ids), 'feather': 2, 'invert': False}
             for name, ids in [('leftEye', LEFT_EYE), ('rightEye', RIGHT_EYE),
                               ('leftBrow', LEFT_BROW), ('rightBrow', RIGHT_BROW),
                               ('face', FACE_OVAL), ('faceOutline', FACE_OVAL)]}
    return {'version': '1.0', 'operations': [{
        'id': 'experimental-eye-closing', 'type': 'feature.adjust', 'enabled': True,
        'params': {'region': {'x': float(low[0]), 'y': float(low[1]),
                              'width': float(high[0]-low[0]), 'height': float(high[1]-low[1])},
                   'landmarks': polygon(FACE_OVAL), 'semanticMasks': masks,
                   'adjustments': {'eyeHeight': amount}},
    }]}


def prepare_geometry(rgb, mesh, eye, directory):
    """Localize one eye once from source-only evidence using fixed rules."""
    width, to_local, to_image, priors, grid, center, radii = frame(mesh, EYES[eye])
    if not np.isfinite(width) or width <= 2 or np.any(radii <= 0):
        raise ValueError('source eye frame has insufficient extent')
    points = mesh[EYES[eye]['upper'] + EYES[eye]['lower']]
    context = max(12, int(np.ceil(width*.25)))
    box = [max(0, int(np.floor(points[:, 0].min()))-context),
           max(0, int(np.floor(points[:, 1].min()))-context),
           min(rgb.shape[1], int(np.ceil(points[:, 0].max()))+context),
           min(rgb.shape[0], int(np.ceil(points[:, 1].max()))+context)]
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    lum, chroma = light(crop), redness(crop)
    noise = max(1., float(1.4826*np.median(abs(lum-gaussian_filter(lum, .6)))))
    chroma_noise = max(.5, float(1.4826*np.median(abs(chroma-gaussian_filter(chroma, .6)))))
    mid = (priors['upper'](grid)+priors['lower'](grid))/2
    external = np.median(np.stack([
        read_rgb(rgb, to_image(np.stack([grid, priors['upper'](grid)-4], axis=-1))),
        read_rgb(rgb, to_image(np.stack([grid, priors['lower'](grid)+4], axis=-1))),
    ]), axis=0)
    candidate = read_rgb(rgb, to_image(np.stack([grid, mid], axis=-1)))
    iris_rgb = read_rgb(rgb, to_image(center))
    anatomical_side = ((abs(grid-center[0]) > radii[0]+1) & (grid > width*.12)
                       & (grid < width*.88) & (priors['lower'](grid)-priors['upper'](grid) > 2*BAND))
    evidence = (anatomical_side & (light(candidate) > light(external)+3*noise)
                & (light(candidate) > light(iris_rgb)+3*noise)
                & (redness(candidate) < redness(external)-3*chroma_noise))
    curves, records, curve_reports = {}, {}, {}
    for side in ('upper', 'lower'):
        curves[side], records[side], curve_reports[side] = fit_curve(
            rgb, to_image, priors, grid, side, evidence, noise)
    dense = np.linspace(0, width, int(np.ceil(width*16))+1)
    gap = curves['lower'](dense)-curves['upper'](dense)
    if float(gap.min()) < -1e-7:
        raise ValueError('fitted source curves cross; reject without clipping')
    yy, xx = np.mgrid[y0:y1, x0:x1]
    uv = to_local(np.stack([xx+.5, yy+.5], axis=-1))
    to_pixel = lambda u, v: to_image(np.stack([u, v], axis=-1)) - [x0+.5, y0+.5]
    report = {'eye': eye, 'sourceCrop': box, 'canthiWidth': width,
              'irisCenter': center.tolist(), 'irisRadii': radii.tolist(),
              'sourceNoise': noise, 'sourceRednessNoise': chroma_noise,
              'confirmedSideColumns': int(evidence.sum()), 'curves': curve_reports,
              'edgeRecords': records, 'minimumCurveGap': float(gap.min()),
              'claim': 'source-only anatomical estimate; not optical alpha truth'}
    write_json(directory/'geometry-report.json', report)
    np.savez_compressed(directory/'source-curves.npz', grid=grid,
                        upper=curves['upper'](grid), lower=curves['lower'](grid),
                        sourceCrop=np.array(box), irisCenter=center, irisRadii=radii)
    image = Image.fromarray(crop.astype(np.uint8)).resize(
        ((x1-x0)*4, (y1-y0)*4), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for side in ('upper', 'lower'):
        for curve, color in [(priors[side], '#ff6e50'), (curves[side], '#3caaff')]:
            xy = (to_image(np.stack([dense, curve(dense)], axis=-1))-[x0, y0])*4
            draw.line([tuple(point) for point in xy], fill=color, width=1)
    image.save(directory/'source-curves.png')
    return {'crop': crop, 'box': box, 'width': width, 'u': uv[..., 0], 'v': uv[..., 1],
            'upper': curves['upper'], 'lower': curves['lower'], 'to_pixel': to_pixel,
            'iris_center': center, 'iris_radii': radii, 'report': report}
