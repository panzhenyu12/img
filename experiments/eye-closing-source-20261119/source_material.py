"""Source-bound categorical eye material corrections, separate from alpha.

An ocular anchor asserts visible ocular source RGB. Moving lash support names
thin foreground plus its mixed edge; it does not specify opacity or hidden B.
Unknown material is never automatically promoted to an observed donor.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from explicit_source_layers import infer_explicit_moving_source_layer
from source_estimator import estimate as estimate_legacy


LABELS = ('ocularAnchors', 'movingLashSupport', 'unknown')


def estimate_legacy_source(source, local_u, local_v, upper, lower, to_pixel, width,
                           iris_center, iris_radii, *, band=3.):
    """Explicit V1 interface adapter; iris arguments do not alter frozen V1."""
    return estimate_legacy(source, local_u, local_v, upper, lower, to_pixel, width, band=band)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{label} must be a finite source-pixel number')
    return float(value)


def rasterize_material_regions(regions, dimensions):
    """Hard categorical rect/polygon/brush labels in original pixel-index XY."""
    width, height = dimensions
    yy, xx = np.indices((height, width))
    mask = np.zeros((height, width), bool)
    if not isinstance(regions, list):
        raise ValueError('material regions must be a list of categorical shapes')
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError('material shape must be an object')
        if region.get('feather', 0) != 0 or region.get('invert', False) is not False:
            raise ValueError('source material labels do not accept feathered alpha or inverted roles')
        shape = region.get('shape')
        allowed = {'shape', 'feather', 'invert'}
        if shape == 'rect':
            allowed.add('region')
            box = region.get('region', {})
            if set(box) != {'x', 'y', 'width', 'height'}:
                raise ValueError('rect requires original pixel-index x/y/width/height')
            x, y, w, h = [_number(box[key], key) for key in ('x', 'y', 'width', 'height')]
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
                raise ValueError('material rectangle leaves source or has no extent')
            part = (xx >= x) & (xx < x+w) & (yy >= y) & (yy < y+h)
        elif shape == 'polygon':
            allowed.add('points')
            points = region.get('points', [])
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError('material polygon needs at least three source points')
            values = []
            for point in points:
                if not isinstance(point, dict) or set(point) != {'x', 'y'}:
                    raise ValueError('polygon material points contain only pixel-index x/y')
                values.append([_number(point['x'], 'x'), _number(point['y'], 'y')])
            points = np.asarray(values)
            if np.any((points < 0) | (points > np.array([width-1, height-1]))):
                raise ValueError('material polygon point leaves source')
            next_points = np.roll(points, -1, axis=0)
            if abs(np.sum(points[:, 0]*next_points[:, 1]-points[:, 1]*next_points[:, 0])) < 1e-8:
                raise ValueError('material polygon has no area')
            part = np.zeros((height, width), bool)
            boundary = np.zeros_like(part)
            for (ax, ay), (bx, by) in zip(points, next_points):
                if ay != by:
                    part ^= ((ay > yy) != (by > yy)) & (xx < (bx-ax)*(yy-ay)/(by-ay)+ax)
                dx, dy = bx-ax, by-ay
                denominator = dx*dx+dy*dy
                if denominator:
                    fraction = np.clip(((xx-ax)*dx+(yy-ay)*dy)/denominator, 0, 1)
                    boundary |= (xx-ax-fraction*dx)**2+(yy-ay-fraction*dy)**2 <= 1e-16
            part |= boundary
        elif shape == 'brush':
            allowed.add('strokes')
            strokes = region.get('strokes', [])
            if not isinstance(strokes, list) or not strokes:
                raise ValueError('material brush needs discrete source-pixel strokes')
            part = np.zeros((height, width), bool)
            for stroke in strokes:
                if not isinstance(stroke, dict) or set(stroke) != {'x', 'y', 'radius'}:
                    raise ValueError('material brush accepts only x/y/radius, not opacity or strength')
                x, y, radius = [_number(stroke[key], key) for key in ('x', 'y', 'radius')]
                if radius <= 0 or x-radius < 0 or y-radius < 0 or x+radius > width-1 or y+radius > height-1:
                    raise ValueError('material brush leaves source or has no radius')
                part |= (xx-x)**2+(yy-y)**2 <= radius*radius
        else:
            raise ValueError('material shape must be rect, polygon or brush; no alpha raster input')
        if set(region)-allowed:
            raise ValueError('material shape contains unsupported fields; labels are not alpha')
        if not part.any():
            raise ValueError('material region does not select a source pixel')
        mask |= part
    return mask


def load_materials(path, source_sha256, dimensions):
    """Bind annotation identities to one original, never to a filename."""
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj)-{'version', 'source', 'eyes', 'note'}:
        raise ValueError('invalid source material document fields')
    if obj.get('version') != '1.0':
        raise ValueError('source materials require version 1.0')
    source = obj.get('source', {})
    if set(source) != {'sha256', 'width', 'height', 'coordinates'}:
        raise ValueError('material source needs original SHA256/dimensions/coordinates')
    if source['sha256'] != source_sha256:
        raise ValueError('material source SHA256 does not match original')
    if [source['width'], source['height']] != list(dimensions):
        raise ValueError('material source dimensions do not match original')
    if source['coordinates'] != 'pixel-index':
        raise ValueError('material coordinates must be original pixel-index XY')
    eyes = obj.get('eyes', {})
    if not isinstance(eyes, dict) or not eyes or set(eyes)-{'leftEye', 'rightEye'}:
        raise ValueError('materials must explicitly name leftEye or rightEye')
    output = {}
    for eye, descriptions in eyes.items():
        if not isinstance(descriptions, dict) or set(descriptions)-set(LABELS):
            raise ValueError('eye material labels are ocularAnchors/movingLashSupport/unknown only')
        masks = {name: rasterize_material_regions(descriptions.get(name, []), dimensions) for name in LABELS}
        if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) > 1):
            raise ValueError(f'{eye}: source material labels conflict')
        if not any(mask.any() for mask in masks.values()):
            raise ValueError(f'{eye}: source material correction is empty')
        output[eye] = masks
    occupied = np.zeros((dimensions[1], dimensions[0]), bool)
    for masks in output.values():
        eye_pixels = np.logical_or.reduce(list(masks.values()))
        if np.any(occupied & eye_pixels):
            raise ValueError('source material labels conflict across eye ownership')
        occupied |= eye_pixels
    return output, {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'sourceSHA256': source_sha256, 'dimensions': list(dimensions),
                    'coordinates': 'original source pixel-index; integer XY are pixel centers',
                    'labelsAreOpticalAlpha': False, 'sourceOnly': True,
                    'role': 'manual material identity correction; inferred hidden colors remain estimated'}


def crop_materials(materials, box):
    x0, y0, x1, y1 = box
    cropped = {}
    for name, mask in materials.items():
        crop = mask[y0:y1, x0:x1]
        if int(crop.sum()) != int(mask.sum()):
            raise ValueError('eye material labels leave its prepared source crop; no silent dropping')
        cropped[name] = crop.copy()
    return cropped


def apply_materials(source, prepared, materials, ocular_domain, *, noise=1., radius=6.):
    """Apply source-only categorical corrections before any requested field."""
    source = np.asarray(source, dtype=float)
    if (source.ndim != 3 or source.shape[-1] != 3 or not np.isfinite(source).all()
            or np.any((source < 0) | (source > 255))):
        raise ValueError('finite source RGB in 0..255 required')
    ocular_domain = np.asarray(ocular_domain)
    if ocular_domain.dtype != np.bool_ or ocular_domain.shape != source.shape[:2]:
        raise ValueError('ocular domain must be boolean and match source dimensions')
    if set(materials) != set(LABELS):
        raise ValueError('exactly ocularAnchors, movingLashSupport and unknown labels are required')
    masks = {}
    for name in LABELS:
        mask = np.asarray(materials[name])
        if mask.shape != source.shape[:2] or mask.dtype != np.bool_:
            raise ValueError(f'{name} must be discrete boolean source material; it is not alpha')
        masks[name] = mask
    if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) > 1):
        raise ValueError('source material labels conflict; no last-writer overwrite')
    if np.any(masks['ocularAnchors'] & ~ocular_domain):
        raise ValueError('ocular anchors leave source domain; correct source geometry explicitly first')
    result = copy.deepcopy(prepared)
    ocular, moving, unknown = (masks[name] for name in LABELS)
    result['ocular'][ocular] = source[ocular]
    result['exposure'][ocular] = 1.
    result['support'][ocular] = True
    result['core'][ocular] = True
    for key in ('lashVeto', 'ambiguous', 'estimatedLashUnderlay', 'affineOcularUnderlay'):
        if key in result:
            result[key][ocular] = False
    result['manualOcularAnchors'] = ocular.copy()
    result['manualMovingLashSupport'] = moving.copy()
    result['materialUnknown'] = unknown.copy()
    inference_report = None
    if moving.any():
        inference = infer_explicit_moving_source_layer(
            source, moving, ocular, ocular_domain, noise,
            source_skin_prior=result.get('skinPrior'), radius=radius)
        estimated = inference['estimated']
        result['ocular'][estimated] = inference['ocular'][estimated]
        result['exposure'][estimated] = inference['exposure'][estimated]
        result['support'][estimated] = True
        result['core'][moving] = False
        if 'observedIris' in result:
            result['observedIris'][moving] = False
        if 'lashVeto' in result:
            result['lashVeto'][moving] = True
        result['manualEstimatedMoving'] = estimated
        result['manualModelUncertain'] = inference['modelUncertain']
        result['materialUnknown'] |= inference['unknown']
        inference_report = inference['report']
    unknown = result['materialUnknown']
    result['support'][unknown] = False
    result['core'][unknown] = False
    result['report']['manualMaterialCorrection'] = {
        'ocularAnchorPixels': int(ocular.sum()), 'movingSupportPixels': int(moving.sum()),
        'unknownPixels': int(unknown.sum()), 'labelsAreOpticalAlpha': False,
        'usesEditedOutput': False, 'movingInference': inference_report,
    }
    return result


def validate_material_transport(unknown, inverse, stationary=None):
    """Reject any requested transport that depends on unclassified material."""
    unknown = np.asarray(unknown)
    inverse = np.asarray(inverse, dtype=float)
    if unknown.dtype != np.bool_ or inverse.shape != unknown.shape+(2,):
        raise ValueError('boolean unknown material and matching pixel-index field required')
    height, width = unknown.shape
    yy, xx = np.indices(unknown.shape)
    frozen = np.zeros_like(unknown) if stationary is None else np.asarray(stationary, bool)
    if frozen.shape != unknown.shape or not np.isfinite(inverse).all():
        raise ValueError('finite field and matching stationary mask required')
    if np.any((inverse[..., 0] < 0) | (inverse[..., 0] > width-1)
              | (inverse[..., 1] < 0) | (inverse[..., 1] > height-1)):
        raise ValueError('material transport field leaves source')
    moved = ((inverse[..., 0] != xx) | (inverse[..., 1] != yy)) & ~frozen
    footprint = map_coordinates(unknown.astype(float), [inverse[..., 1], inverse[..., 0]],
                                order=1, mode='nearest', prefilter=False)
    invalid = moved & (unknown | (footprint > 0))
    if invalid.any():
        raise ValueError(f'edit depends on unknown source material at {int(invalid.sum())} destination pixels')
    return {'unknownDependentDestinations': 0}
