"""Frozen V37 f32 inverse sampling and source-residual composition contract.

The operation order is intentional: it reproduces the native trace exactly.
This module does not estimate masks, geometry, foreground, or background.
"""
import numpy as np


def sample_f32(foreground_byte, alpha, source_xy):
    h, w = alpha.shape
    xx, yy = source_xy[..., 0], source_xy[..., 1]
    x0, y0 = np.floor(xx).astype('int32'), np.floor(yy).astype('int32')
    tx, ty = xx-np.floor(xx), yy-np.floor(yy)
    output = np.zeros((*alpha.shape, 3), 'float32')
    membership = np.zeros(alpha.shape, 'float32')
    for dx, dy, weight in [(0, 0, (1-tx)*(1-ty)), (1, 0, tx*(1-ty)),
                           (0, 1, (1-tx)*ty), (1, 1, tx*ty)]:
        x, y = x0+dx, y0+dy
        valid = (x >= 0) & (y >= 0) & (x < w) & (y < h)
        wa = alpha[y[valid], x[valid]]*weight[valid]
        membership[valid] += wa
        output[valid] += foreground_byte[y[valid], x[valid]]*wa[:, None]
    return output, membership


def compose(source_byte, foreground_byte, background_byte, alpha, target_alpha, sampled):
    result = (source_byte+sampled)-foreground_byte*alpha[..., None]
    result = result+(alpha-target_alpha)[..., None]*background_byte
    return np.floor(np.clip(result, 0, 255)+.5).astype('uint8')
