"""A source-relative jaw request for a new photo, independent of old traces.

Cached detector contours and coverage are explicit evidence, not visibility truth.
The selected source contour is never smoothed, reordered or resampled here.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt

from source_regions import ROOT, simple_path, local, FACE_OVAL

sys.path.insert(0, str(ROOT/'scripts'))
from classify_face_frozen_evidence import raster_sample


def pixels(rows, size):
    p = np.asarray([[r['x']*size[0]-.5, r['y']*size[1]-.5] for r in rows])
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('finite normalized detector points required')
    return p


def jaw_geometry(contour, oval, eyes, lips, *, start_plane='mouth-center'):
    if start_plane not in ['mouth-center','upper-lip']:
        raise ValueError('explicit mouth-center or upper-lip source start plane required')
    p = simple_path(contour, 'source detector contour')
    oval = simple_path(oval, 'source canonical oval')
    if len(oval) != len(FACE_OVAL):
        raise ValueError('full canonical36 source oval required')
    eyes = [simple_path(eye, 'source eye') for eye in eyes]
    lips = simple_path(lips, 'source lips')
    left, right = [eye.mean(0) for eye in eyes]
    across = right-left
    if across[0] <= 0 or np.linalg.norm(across) < 1e-8:
        raise ValueError('ordered separated source eyes required')
    across /= np.linalg.norm(across); down = np.array([-across[1], across[0]])
    origin = (left+right)*.5
    if (lips.mean(0)-origin)@down <= 0:
        raise ValueError('source mouth must lie below eyes')
    uv = (oval-origin)@np.array([across, down]).T
    frame = {'origin': origin.tolist(), 'across': across.tolist(), 'down': down.tolist(),
             'bounds': np.r_[uv.min(0), uv.max(0)].tolist(), 'midline_u': 0.}
    q = local(p, frame); mouth_height = float(local(lips, frame)[:, 1].mean())
    start_height = mouth_height if start_plane=='mouth-center' else float(local(lips,frame)[:,1].min())
    nxt = np.roll(q, -1, axis=0); edge = np.roll(p, -1, axis=0)-p
    lengths = np.linalg.norm(edge, axis=1); cumulative = np.r_[0., lengths.cumsum()]
    total = cumulative[-1]
    if np.any((abs(q[:, 1]-start_height)<1e-9)&(abs(nxt[:, 1]-start_height)<1e-9)):
        raise ValueError('source segment coincides with mouth plane')
    crossing = np.flatnonzero(((q[:, 1]<=start_height)&(nxt[:, 1]>start_height))|
                             ((nxt[:, 1]<=start_height)&(q[:, 1]>start_height)))
    if len(crossing) != 2:
        raise ValueError('exactly two unambiguous source mouth crossings required')
    boundaries = []
    for i in crossing:
        t = float((start_height-q[i, 1])/(nxt[i, 1]-q[i, 1]))
        boundaries.append({'edge': int(i), 't': t})
    at = lambda b: p[b['edge']]+b['t']*edge[b['edge']]
    s = lambda b: float((cumulative[b['edge']]+b['t']*lengths[b['edge']])%total)
    boundaries.sort(key=lambda b: float(local([at(b)], frame)[0, 0]))
    if not local([at(boundaries[0])], frame)[0, 0] < 0 < local([at(boundaries[1])], frame)[0, 0]:
        raise ValueError('mouth crossings do not straddle source midline')
    chin = oval[FACE_OVAL.index(152)]
    t = np.clip(((chin-p)*edge).sum(1)/(lengths**2), 0, 1)
    projections = p+t[:, None]*edge
    distance = np.linalg.norm(projections-chin, axis=1)
    i = int(np.argmin(distance)); end = {'edge': i, 't': float(t[i])}
    if local([at(end)], frame)[0, 1] <= mouth_height:
        raise ValueError('chin proxy must be below source mouth plane')
    arcs, selected = [], np.zeros(len(p), bool)
    for side, start in zip(['left', 'right'], boundaries):
        candidates = []
        for direction in [1, -1]:
            length = (direction*(s(end)-s(start)))%total
            d = (direction*(cumulative[:-1]-s(start)))%total
            inside = (d > 0)&(d < length)
            if length > 0 and inside.any() and np.all(q[inside, 1] >= start_height-1e-8):
                candidates.append((direction, inside))
        if len(candidates) != 1:
            raise ValueError('one lower source path to chin is required per side')
        direction, inside = candidates[0]
        if np.any(inside & selected):
            raise ValueError('left and right lower source paths overlap')
        selected |= inside
        arcs.append({'id': side+'-jaw', 'start': start, 'end': end, 'direction': direction})
    scope = ({'start_plane':start_plane,'start_height':start_height,
              'operation_scope':'diagnostic_lower_face_from_upper_lip'} if start_plane!='mouth-center' else {})
    return {**scope, 'frame': frame, 'arcs': arcs, 'contour': p.tolist(),
            'selected_rows': np.flatnonzero(selected).tolist(), 'mouth_height': mouth_height,
            'chin_projection_distance_px': float(distance[i]),
            'source_definition': 'unchanged cached detector contour, normalized image coordinates converted once to pixel indices',
            'visibility_status': 'unknown', 'automatic_edit_authorized': False}


def fixed_support(size, params, geometry, support_radius_fraction=.24, *, support_geometry=None):
    """Select a geometric neighborhood, then exactly constrain protected pixels.

    This mask constrains source nodes. It is never multiplied into a solved field
    or used to replace output pixels. Background inside the neighborhood is free.
    """
    support_geometry = geometry if support_geometry is None else support_geometry
    if (support_geometry['contour']!=geometry['contour'] or support_geometry['frame']!=geometry['frame']):
        raise ValueError('support geometry must use the exact same original contour and frame')
    w, h = size; p = np.asarray(geometry['contour']); frame = geometry['frame']
    if not np.isfinite(support_radius_fraction) or support_radius_fraction <= 0:
        raise ValueError('positive finite source support radius required')
    selected = support_geometry['selected_rows']; uv = local(p, frame)
    radius = support_radius_fraction*float(np.ptp(uv[:, 0]))
    band = Image.new('L', size); draw = ImageDraw.Draw(band)
    for arc in support_geometry['arcs']:
        a, b = arc['start'], arc['end']
        at = lambda x: p[x['edge']]+x['t']*(p[(x['edge']+1)%len(p)]-p[x['edge']])
        # The path order follows the original source; no shortcut across chin.
        lengths = np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1)
        cumulative = np.r_[0., lengths.cumsum()]
        pos = lambda x: (cumulative[x['edge']]+x['t']*lengths[x['edge']])%cumulative[-1]
        d = ((cumulative[:-1]-pos(a))*arc['direction'])%cumulative[-1]
        length = ((pos(b)-pos(a))*arc['direction'])%cumulative[-1]
        ids = np.flatnonzero((d>0)&(d<length)); ids = ids[np.argsort(d[ids])]
        path = np.vstack([at(a), p[ids], at(b)])
        draw.line([tuple(point) for point in path], fill=1, width=1)
    support = distance_transform_edt(np.asarray(band)==0) <= radius
    yy, xx = np.mgrid[:h, :w]
    v = (xx-frame['origin'][0])*frame['down'][0]+(yy-frame['origin'][1])*frame['down'][1]
    # The endpoint gain vanishes cubically; preserve the source upper face.
    upper = v < geometry.get('start_height',geometry['mouth_height'])-4.
    protect = upper.copy(); roles = {'upper_face': upper}
    for key in ['nose', 'lips']:
        mask = Image.new('L', size)
        ImageDraw.Draw(mask).polygon([tuple(point) for point in pixels(params['semanticMasks'][key]['points'], size)], fill=1)
        value = binary_dilation(np.asarray(mask)>0, iterations=2)
        protect |= value; roles[key] = value
    for key in ['ears', 'accessories']:
        if key not in params['semanticMasks']:
            raise ValueError('explicit cached '+key+' coverage required')
        value = raster_sample(params['semanticMasks'][key], size, [0, 0, w, h]) >= .5
        value = binary_dilation(value, iterations=1)
        protect |= value; roles[key] = value
    fixed = ~support | protect
    low = np.maximum(np.floor(p[selected].min(0)-radius-6).astype(int), 0)
    high = np.minimum(np.ceil(p[selected].max(0)+radius+6).astype(int), [w-1, h-1])
    # Include every unchanged source row in the problem too. Upper source rows
    # remain fixed, but must not be silently dropped from response measurement.
    low = np.minimum(low, np.maximum(np.floor(p.min(0)-4).astype(int), 0))
    high = np.maximum(high, np.minimum(np.ceil(p.max(0)+4).astype(int), [w-1, h-1]))
    for axis in range(2):
        if (high[axis]-low[axis])%2:
            if high[axis] < size[axis]-1: high[axis] += 1
            elif low[axis] > 0: low[axis] -= 1
            else: raise ValueError('cannot enclose jaw support with exact 2px grid')
    crop = [*low.tolist(),*(high-low+1).tolist()]
    roles['outside_geometric_neighborhood'] = ~support
    return fixed, crop, roles, radius


def request_targets(geometry, amount, binary, max_radius_fraction=.13):
    request = {k: geometry[k] for k in ['frame', 'contour', 'arcs']}
    request.update(amount=float(amount), max_radius_fraction=max_radius_fraction)
    run = subprocess.run([str(binary)], input=json.dumps(request), text=True, capture_output=True, check=True)
    output = json.loads(run.stdout)
    source = np.asarray(geometry['contour'])
    if output.get('controller') != 'jaw_arc_v1' or not np.array_equal([r['source'] for r in output['targets']], source):
        raise ValueError('Rust jaw source changed')
    return np.asarray([r['target'] for r in output['targets']]), request, output
