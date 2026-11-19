"""One posterior revision for face-width drivers and ear/accessory evidence.

This is a source-only experiment. The returned control mask is not optical
alpha and must not replace the runtime source-layer faceContourMask.
"""
import copy
import hashlib

import numpy as np
from scipy.ndimage import map_coordinates


CLASSES = {'faceWidthControlMask': (1, 2, 3, 4, 5, 10, 11, 12, 13),
           'earEvidenceMask': (7, 8), 'accessoryEvidenceMask': (6, 9)}
ROLE_CLASSES = {'hair': (17,), 'skin': (1, 10, 14), 'neck': (14,),
                'ears': (7, 8), 'accessories': (6, 9)}


def _project_probability(probability, crop, image_size, grid):
    width, height = image_size
    rw, rh = grid
    x, y, cw, ch = crop
    ph, pw = probability.shape
    yy, xx = np.mgrid[:rh, :rw]
    px, py = (xx + .5) * width / rw, (yy + .5) * height / rh
    query = [(py - y) / ch * ph - .5, (px - x) / cw * pw - .5]
    projected = map_coordinates(probability, query, order=1, mode='nearest')
    valid = (px >= x) & (px <= x + cw) & (py >= y) & (py <= y + ch)
    projected[~valid] = 0
    return projected, valid


def _coverage(projected):
    t = np.clip((projected - .4) / .2, 0, 1)
    return np.floor(t * t * (3 - 2 * t) * 255 + .5).astype(np.uint8)


def _project(probability, crop, image_size, grid):
    return _coverage(_project_probability(probability, crop, image_size, grid)[0])


def raw_role_proposal(selected, image_size, grid, validity_policy='confident'):
    """Return nullable evidence, never an implicitly zero-filled source mask.

    Confident preserves the v2 experimental .6 gate. Observed-support keeps
    ambiguous but observed source probabilities; only crop exterior is null.
    Neither class probability nor its coverage conversion is optical alpha.
    """
    probabilities, crop = selected['probabilities'], selected['crop']
    class_projections = [_project_probability(layer, crop, image_size, grid) for layer in probabilities]
    confidence = np.maximum.reduce([layer for layer, _ in class_projections])
    covered = class_projections[0][1]
    if validity_policy not in ('confident', 'observed-support'):
        raise ValueError('unknown source evidence validity policy')
    valid = covered & (confidence >= .6) if validity_policy == 'confident' else covered
    masks, status, role_posteriors = {}, {}, {}
    def posterior_raster(values):
        return {'shape': 'nullable-posterior-raster', 'width': grid[0], 'height': grid[1],
                'data': [float(v) if ok else None for v, ok in zip(values.ravel(), covered.ravel())]}
    for key, classes in ROLE_CLASSES.items():
        probability = probabilities[list(classes)].sum(0, dtype=np.float64)
        projected, _ = _project_probability(probability, crop, image_size, grid)
        role_posteriors[key] = posterior_raster(projected)
        coverage = _coverage(projected)
        clipped = bool(_clipped(probability, crop, image_size))
        masks[key] = {'shape': 'nullable-raster-evidence', 'width': grid[0], 'height': grid[1],
                      'data': [int(v) if ok else None for v, ok in zip(coverage.ravel(), valid.ravel())]}
        status[key] = {'internallyClipped': clipped, 'wholeImageReplacementValid': bool(valid.all() and not clipped),
                       'inferredProbabilityMax': float(probability.max()),
                       'validPresentNodes': int(np.count_nonzero(valid & (coverage >= 128))),
                       'unknownNodes': int(np.count_nonzero(~valid))}
    return {'masks': masks, 'status': status, 'classGroups': ROLE_CLASSES,
            'sourceValidityMask': {'shape': 'raster', 'width': grid[0], 'height': grid[1],
                                   'data': (valid.astype(np.uint8) * 255).ravel().tolist()},
            'coverageMask': {'shape': 'raster', 'width': grid[0], 'height': grid[1],
                              'data': (covered.astype(np.uint8) * 255).ravel().tolist()},
            'validityPolicy': validity_policy,
            'validityRule': ('inside selected source crop AND max(bilinear(each class posterior)) >= .6; null means unknown, never absence'
                             if validity_policy == 'confident' else
                             'inside selected source crop with validated complete 19-class posterior; class uncertainty remains soft evidence; null only outside observed crop'),
            'confidenceDiagnostics': {'maxClassPosterior': posterior_raster(confidence),
                                      'rolePosteriors': role_posteriors,
                                      'calibratedProbabilityClaim': False,
                                      'opticalAlphaClaim': False},
            'physicalAlpha': False, 'appliedToSource': False, 'manualEvidence': False}


def _clipped(layer, crop, image_size):
    x, y, width, height = crop
    iw, ih = image_size
    return ((x > 0 and np.any(layer[:, 0] > .4))
            or (x + width < iw and np.any(layer[:, -1] > .4))
            or (y > 0 and np.any(layer[0] > .4))
            or (y + height < ih and np.any(layer[-1] > .4)))


def derive_control_revision(base_control, passes, bounds, image_size, validity_policy='confident'):
    if (base_control.get('shape') != 'raster' or base_control['width'] < 1
            or base_control['height'] < 1
            or len(base_control['data']) != base_control['width'] * base_control['height']):
        raise ValueError('valid base control raster required')
    if (len(bounds) != 4 or len(image_size) != 2
            or not np.isfinite([*bounds, *image_size]).all()
            or min(image_size) <= 0 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
        raise ValueError('valid source bounds and image dimensions required')
    candidates, audit = [], []
    for inference in passes:
        probabilities = np.asarray(inference['probabilities'])
        crop = np.asarray(inference['crop'], dtype=float)
        if (probabilities.ndim != 3 or probabilities.shape[0] != 19
                or min(probabilities.shape[1:]) < 1
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1 + 1e-6))
                or np.max(abs(probabilities.sum(0) - 1)) > 1e-5
                or crop.shape != (4,) or not np.isfinite(crop).all()
                or min(crop[2:]) <= 0):
            raise ValueError('one complete normalized 19-class posterior is required')
        layers = {key: probabilities[list(classes)].sum(0, dtype=np.float64)
                  for key, classes in CLASSES.items()}
        x, y, width, height = crop
        covers = x <= bounds[0] and y <= bounds[1] and x + width >= bounds[2] and y + height >= bounds[3]
        clipped = [key for key, layer in layers.items() if _clipped(layer, crop, image_size)]
        pitch = max(width / probabilities.shape[2], height / probabilities.shape[1])
        row = {'id': inference['id'], 'crop': crop.tolist(), 'sourcePitch': float(pitch),
               'coversFaceBounds': bool(covers), 'internallyClipped': clipped}
        audit.append(row)
        if covers and not clipped:
            candidates.append((pitch, inference, layers, row))
    if not candidates:
        raise ValueError('no common unclipped face/ear/accessory pass; do not mix revisions')
    _, selected, layers, selected_report = min(candidates, key=lambda item: item[0])
    grid = (base_control['width'], base_control['height'])
    projected = {key: _project(layer, selected['crop'], image_size, grid)
                 for key, layer in layers.items()}
    # Groups are disjoint in a single normalized posterior. Separate byte
    # rounding can contribute at most one extra code value at an exact tie.
    total = sum(value.astype(np.int16) for value in projected.values())
    if np.max(total) > 256:
        raise AssertionError('inconsistent projected control/material evidence')
    output = {key: {'shape': 'raster', 'width': grid[0], 'height': grid[1],
                    'data': value.ravel().tolist()} for key, value in projected.items()}
    original = np.asarray(base_control['data']).reshape(grid[1], grid[0])
    output['provenance'] = {'selection': copy.deepcopy(selected_report), 'passAudit': audit,
                            'classGroups': CLASSES,
                            'posteriorSHA256': hashlib.sha256(np.ascontiguousarray(selected['probabilities']).tobytes()).hexdigest(),
                            'controlNodesChanged': int(np.count_nonzero(original != projected['faceWidthControlMask'])),
                            'sameRevisionCoverageSumMaxByte': int(total.max()),
                            'role': 'geometric face-width driver evidence only, not source optical alpha',
                            'sourceMaterialsMutated': False}
    output['rawSourceRoleProposal'] = raw_role_proposal(selected, image_size, grid, validity_policy)
    return output
