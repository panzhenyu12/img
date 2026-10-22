"""Frozen-map reconstruction comparison, not a new deformation algorithm.

Bounded Lanczos3 uses a normalized separable 6x6 kernel in encoded RGB, then
limits each color channel to the surrounding 2x2 source sample range. This
reduces overshoot at the cost of clipping legitimate extrema. Transparent
support and image boundaries retain bilinear RGBA; exact integer samples and
zero displacement retain original bytes. This fixed kernel does not solve
anisotropic minification or supply a full anti-aliasing guarantee.
"""
import time

import numpy as np


def _validate(source, points, method):
    source = np.asarray(source); points = np.asarray(points, float)
    if (source.ndim != 3 or source.shape[2] != 4 or min(source.shape[:2]) < 1
            or points.ndim != 2 or points.shape[1] != 2
            or not np.isfinite(source).all() or not np.isfinite(points).all()
            or np.any(source < 0) or np.any(source > 255)
            or method not in ['bilinear', 'bounded_lanczos3']):
        raise ValueError('finite RGBA samples in [0,255], Nx2 points and a known kernel required')
    return source, points


def _bilinear(source, points):
    h, w = source.shape[:2]
    p = np.clip(points, [0., 0.], [w-1., h-1.]); base = np.floor(p).astype(int)
    x, y = base.T; u, v = (p-base).T
    opacity = np.zeros(len(p)); premul = np.zeros((len(p), 3))
    for dx, dy, weight in [(0, 0, (1-u)*(1-v)), (1, 0, u*(1-v)),
                           (0, 1, (1-u)*v), (1, 1, u*v)]:
        rgba = source[np.minimum(y+dy, h-1), np.minimum(x+dx, w-1)].astype(float)
        a = rgba[:, 3]/255.*weight; opacity += a; premul += rgba[:, :3]*a[:, None]
    result = np.zeros((len(p), 4)); result[:, 3] = opacity*255.
    visible = np.floor(opacity*255.+.5) > 0
    result[visible, :3] = premul[visible]/opacity[visible, None]
    integer = np.all(p == base, axis=1)
    result[integer] = source[y[integer], x[integer]]
    return result, p, base, integer


def _sample(source, points, method):
    result, p, base, integer = _bilinear(source, points)
    report = {'sample_count': len(points), 'integer_sample_count': int(integer.sum()),
              'lanczos_sample_count': 0, 'bounded_sample_count': 0,
              'raw_local_overshoot_max': 0., 'final_local_overshoot_max': 0.}
    if method == 'bilinear' or not len(points):
        return result, report
    h, w = source.shape[:2]
    inside = np.all(base >= 2, axis=1) & np.all(base+3 < [w, h], axis=1) & ~integer
    ids = np.flatnonzero(inside)
    if not len(ids):
        return result, report
    b = base[ids]; t = p[ids]-b; offsets = np.arange(-2, 4)
    def weights(fraction):
        distance = fraction[:, None]-offsets
        value = np.sinc(distance)*np.sinc(distance/3.)
        value[abs(distance) >= 3.] = 0.
        return value/value.sum(axis=1, keepdims=True)
    wx, wy = weights(t[:, 0]), weights(t[:, 1])
    raw = np.zeros((len(ids), 3)); opaque = np.ones(len(ids), bool)
    for iy, dy in enumerate(offsets):
        for ix, dx in enumerate(offsets):
            rgba = source[b[:, 1]+dy, b[:, 0]+dx]
            opaque &= rgba[:, 3] == 255
            raw += rgba[:, :3]*(wx[:, ix]*wy[:, iy])[:, None]
    neighborhood = np.stack([source[b[:, 1]+dy, b[:, 0]+dx, :3]
                             for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]])
    lo, hi = neighborhood.min(axis=0), neighborhood.max(axis=0)
    bounded = np.clip(raw, lo, hi)
    selected = ids[opaque]
    result[selected, :3] = bounded[opaque]
    # Alpha is always the original bilinear alpha, including transparent
    # neighborhoods which never use the oscillatory reconstruction kernel.
    excess = np.maximum(np.maximum(lo-raw, raw-hi), 0.)
    report.update(lanczos_sample_count=len(selected),
                  bounded_sample_count=int(np.any(excess[opaque] > 1e-12, axis=1).sum()),
                  raw_local_overshoot_max=float(excess[opaque].max(initial=0.)),
                  final_local_overshoot_max=float(np.maximum(np.maximum(lo[opaque]-bounded[opaque],
                                                                          bounded[opaque]-hi[opaque]), 0.).max(initial=0.)))
    return result, report


def sample_rgba(source, points, method='bounded_lanczos3'):
    """Return floating RGBA samples and a kernel-specific diagnostic report."""
    source, points = _validate(source, points, method)
    return _sample(source, points, method)


def render_frozen_field(source, inverse, crop, method='bounded_lanczos3', *, fixed=None, block_size=8192):
    """Sample one immutable float32 inverse field; copy all untouched bytes."""
    started = time.perf_counter()
    source, _ = _validate(source, np.empty((0, 2)), method)
    field = np.asarray(inverse)
    if (source.dtype != np.uint8 or field.dtype != np.dtype('<f4') or len(crop) != 4
            or any(type(v) is not int for v in crop) or block_size < 1
            or field.shape != (crop[3], crop[2], 2) or not np.isfinite(field).all()
            or min(crop[:2]) < 0 or min(crop[2:]) < 1
            or crop[0]+crop[2] > source.shape[1] or crop[1]+crop[3] > source.shape[0]):
        raise ValueError('original uint8 image, float32 field and in-image integer crop required')
    frozen = np.zeros(field.shape[:2], bool) if fixed is None else np.asarray(fixed)
    if frozen.dtype != bool or frozen.shape != field.shape[:2]:
        raise ValueError('fixed mask must be bool on the exact field domain')
    moving = np.any(field != 0., axis=2) & ~frozen
    yy, xx = np.where(moving)
    output = source.copy()
    totals = {'sample_count': 0, 'integer_sample_count': 0, 'lanczos_sample_count': 0,
              'bounded_sample_count': 0, 'raw_local_overshoot_max': 0., 'final_local_overshoot_max': 0.}
    for begin in range(0, len(xx), block_size):
        x, y = xx[begin:begin+block_size], yy[begin:begin+block_size]
        points = np.column_stack([x+crop[0], y+crop[1]])+field[y, x].astype(float)
        sampled, report = _sample(source, points, method)
        output[y+crop[1], x+crop[0]] = np.floor(np.clip(sampled, 0., 255.)+.5).astype(np.uint8)
        for key, value in report.items():
            totals[key] = max(totals[key], value) if key.endswith('_max') else totals[key]+value
    totals.update(kernel=method, block_size=block_size, seconds=time.perf_counter()-started,
                  zero_displacement_bytes_preserved=True, explicit_fixed_pixels=int(frozen.sum()),
                  original_inverse_unmodified=True, bilinear_alpha_preserved=True,
                  color_space='original encoded RGB, same domain as baseline',
                  bounded_contract='per-channel min/max of surrounding source 2x2 samples',
                  transparent_or_boundary_support='original bilinear RGBA',
                  old_material_rgb_oracle_pass_claimed=False,
                  anisotropic_minification_antialiasing_claimed=False)
    return output, totals
