"""Independent double-precision field audit for local eyebrow inverse warps.

Checks Jacobian sign and eye-donor footprints; it is not a Rust-byte parity
test or aesthetic score. Root owns production geometry.rs and actual replays.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test/results/brow-eye-field-review-20261014'


def smooth(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def inside(points, x, y):
    result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), bool)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        delta = b[1] - a[1]
        crossing = (a[1] > y) != (b[1] > y)
        if abs(delta) > 1e-12:
            result ^= crossing & (x < (b[0] - a[0]) * (y - a[1]) / delta + a[0])
    return result


def frame(polygons, name):
    center = lambda p: (p.min(axis=0) + p.max(axis=0)) * .5
    direction = center(polygons['rightEye']) - center(polygons['leftEye'])
    across = direction / max(np.linalg.norm(direction), 1)
    down = np.array([-across[1], across[0]])
    basis = np.column_stack((across, down))
    uv = polygons[name] @ basis
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    return {'center': ((lo + hi) * .5) @ basis.T, 'across': across,
            'width': max(hi[0] - lo[0], 1), 'height': max(hi[1] - lo[1], 1)}


def eye_damping(eyes, x, y):
    weight = np.ones(np.broadcast_shapes(x.shape, y.shape))
    for eye in eyes:
        dx, dy = x - eye['center'][0], y - eye['center'][1]
        u = dx * eye['across'][0] + dy * eye['across'][1]
        v = -dx * eye['across'][1] + dy * eye['across'][0]
        du = np.maximum(abs(u) - eye['width'] * .5, 0)
        dv = np.maximum(abs(v) - eye['height'] * .5, 0)
        weight *= smooth(2, np.clip(eye['width'] * .28, 4, 24), np.hypot(du, dv))
    return weight


def mapping(polygons, key, amount, x, y, constrained):
    # cx,cy,rx,ry,sx,sy,mx,my; mirrors publicly exposed brow parameter intent.
    templates = {
        'browLength': (.28, .21, .10, -.25, 0, 0, 0),
        'browWidth': (.28, .21, .10, -.15, 0, 0, 0),
        'browThickness': (.28, .21, .09, 0, -.3, 0, 0),
        'browPeak': (.26, .13, .09, 0, 0, 0, .04),
        'browTail': (.29, .12, .08, 0, 0, None, .03),
        'browSpacing': (.28, .21, .09, 0, 0, None, 0),
    }
    cy, rx, ry, sx, sy, mx, my = templates[key]
    eyes = [frame(polygons, name) for name in ['leftEye', 'rightEye']]
    damping = eye_damping(eyes, x, y) if constrained else 1
    moved_x, moved_y = x.copy(), y.copy()
    for name, canonical_x, sign in [('leftBrow', .32, -1), ('rightBrow', .68, 1)]:
        brow = frame(polygons, name)
        scale_x, scale_y = brow['width'] / .28, brow['height'] / .06
        cx = canonical_x + (sign * .08 if key == 'browTail' else 0)
        du, dv = (cx - canonical_x) * scale_x, (cy - .28) * scale_y
        center = brow['center'] + np.array([du * brow['across'][0] - dv * brow['across'][1],
                                           du * brow['across'][1] + dv * brow['across'][0]])
        tx = (-sign * (.03 if key == 'browTail' else .07) if mx is None else mx) * scale_x * amount
        ty = my * scale_y * amount
        dx, dy = x - center[0], y - center[1]
        u = dx * brow['across'][0] + dy * brow['across'][1]
        v = -dx * brow['across'][1] + dy * brow['across'][0]
        d = (u / max(rx * scale_x, 1)) ** 2 + (v / max(ry * scale_y, 1)) ** 2
        weight = np.where(d < 1, (1 - d) ** 2, 0) * damping
        local_u = (u * sx * amount + tx) * weight
        local_v = (v * sy * amount + ty) * weight
        moved_x += local_u * brow['across'][0] - local_v * brow['across'][1]
        moved_y += local_u * brow['across'][1] + local_v * brow['across'][0]
    return moved_x, moved_y


def main():
    OUT.mkdir(exist_ok=True)
    graph_path = ROOT / 'test/results/user-face-contour-20261014/1024/face-graph.json'
    graph = json.loads(graph_path.read_text())
    polygons = {k: np.array([[p['x'] * 1024, p['y'] * 1024] for p in v['points']])
                for k, v in graph['operations'][0]['params']['semanticMasks'].items() if v['shape'] == 'polygon'}
    report = {'scope': 'Independent f64 field equation check on original source detector polygons and analytic derivatives. Evaluates brow changes before stationary-donor rejection; all evaluated support is interior to face region/selection. Derived roll and padded tall cases are not independent photos.',
              'geometrySourceSha256': hashlib.sha256((ROOT / 'core/image-core/src/geometry.rs').read_bytes()).hexdigest(),
              'graphSha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(), 'rows': []}
    for case in ['original', 'roll30', 'padded-tall']:
        points = {k: v.copy() for k, v in polygons.items()}
        if case == 'roll30':
            a = np.deg2rad(30); rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            points = {k: (v - 512) @ rotation.T + 700 for k, v in points.items()}
        elif case == 'padded-tall':
            points = {k: v + np.array([0, 256]) for k, v in points.items()}
        bounds = np.vstack([points[k] for k in ['leftEye', 'rightEye', 'leftBrow', 'rightBrow']])
        xmin, ymin = np.floor(bounds.min(axis=0) - 65).astype(int)
        xmax, ymax = np.ceil(bounds.max(axis=0) + 65).astype(int)
        y, x = np.mgrid[ymin:ymax, xmin:xmax].astype(float)
        dest_eye = inside(points['leftEye'], x + .5, y + .5) | inside(points['rightEye'], x + .5, y + .5)
        for key in ['browLength', 'browWidth', 'browThickness', 'browPeak', 'browTail', 'browSpacing']:
            for amount in [-1, 1]:
                for candidate in [False, True]:
                    mx, my = mapping(points, key, amount, x, y, candidate)
                    derivative = .05
                    xp, yp = mapping(points, key, amount, x + derivative, y, candidate)
                    xm, ym = mapping(points, key, amount, x - derivative, y, candidate)
                    xt, yt = mapping(points, key, amount, x, y + derivative, candidate)
                    xb, yb = mapping(points, key, amount, x, y - derivative, candidate)
                    det = ((xp - xm) * (yt - yb) - (yp - ym) * (xt - xb)) / (4 * derivative ** 2)
                    displacement = np.hypot(mx - x, my - y)
                    donor_eye = np.zeros(x.shape, bool)
                    bx, by = np.floor(mx), np.floor(my)
                    tx, ty = mx - bx, my - by
                    for dx, dy, weight in [(0, 0, (1-tx)*(1-ty)), (1, 0, tx*(1-ty)), (0, 1, (1-tx)*ty), (1, 1, tx*ty)]:
                        donor_eye |= ((weight > 1e-6) &
                                      (inside(points['leftEye'], bx + dx + .5, by + dy + .5) |
                                       inside(points['rightEye'], bx + dx + .5, by + dy + .5)))
                    imported_eye = donor_eye & ~dest_eye & (displacement > .1)
                    fold = det <= 0
                    minimum = np.unravel_index(det.argmin(), det.shape)
                    row = {'case': case, 'key': key, 'amount': amount * 100, 'candidate': candidate,
                           'minimumJacobian': float(det[minimum]), 'minimumJacobianXY': [int(x[minimum]), int(y[minimum])],
                           'foldPixels': int(fold.sum()), 'eyeDonorImportPixelsOutsideEyes': int(imported_eye.sum()),
                           'maxEyePixelDisplacement': float(displacement[dest_eye].max()),
                           'maxDisplacement': float(displacement.max())}
                    report['rows'].append(row)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    bad = [row for row in report['rows'] if row['candidate'] and (row['foldPixels'] or row['eyeDonorImportPixelsOutsideEyes'] or row['maxEyePixelDisplacement'] > 1e-8)]
    print(json.dumps({'rows': len(report['rows']), 'candidateIssues': bad}, indent=2))


if __name__ == '__main__':
    main()
