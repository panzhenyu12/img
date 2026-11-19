"""Move eyelids in front of a stationary, source-observed ocular layer.

E is ocular exposure, not a landmark probability. The source decomposition is
C = R + E*B, where R is the premultiplied eyelid residual. Source preparation
must supply an admissible pair and explicit visibility support. This module
does not estimate hidden iris, alter gaze, or permit unobserved eye opening.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def sample(array, inverse):
    coordinates = [inverse[..., 1], inverse[..., 0]]
    if array.ndim == 2:
        return map_coordinates(array, coordinates, order=1, mode='nearest', prefilter=False)
    return np.stack([sample(array[..., c], inverse) for c in range(array.shape[-1])], axis=-1)


def compose(source, exposure, ocular, observed, inverse, stationary=None):
    source = np.asarray(source, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    ocular = np.asarray(ocular, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    inverse = np.asarray(inverse, dtype=np.float64)
    h, w, channels = source.shape
    if (channels != 3 or exposure.shape != (h, w) or ocular.shape != source.shape
            or observed.shape != (h, w) or inverse.shape != (h, w, 2)):
        raise ValueError('matching RGB source, exposure, ocular support and pixel-index inverse required')
    if any(not np.isfinite(v).all() for v in (source, exposure, ocular, inverse)):
        raise ValueError('finite source layers and field required')
    if np.any((exposure < 0) | (exposure > 1)) or np.any((source < 0) | (source > 255)):
        raise ValueError('source and exposure outside their domains')
    if np.any((ocular < 0) | (ocular > 255)):
        raise ValueError('ocular source outside RGB domain')
    if np.any((inverse[..., 0] < 0) | (inverse[..., 0] > w - 1)
              | (inverse[..., 1] < 0) | (inverse[..., 1] > h - 1)):
        raise ValueError('field leaves the source observation')
    frozen = np.zeros((h, w), bool) if stationary is None else np.asarray(stationary, dtype=bool)
    if frozen.shape != (h, w):
        raise ValueError('stationary source dimensions differ')
    yy, xx = np.indices((h, w))
    moved = ((inverse[..., 0] != xx) | (inverse[..., 1] != yy)) & ~frozen
    if not moved.any():
        return source.copy(), {'identityExact': True, 'changedPixels': 0, 'unknownOcularSamples': 0}
    # A successful numerical decomposition is only an admissibility check,
    # not proof that a photograph's estimated optical matte is physically true.
    residual = source - exposure[..., None] * ocular
    maximum = 255 * (1 - exposure[..., None])
    if np.any(residual < -1e-6) or np.any(residual > maximum + 1e-6):
        raise ValueError('source exposure/background cannot represent bounded eyelid colors')
    target_exposure = sample(exposure, inverse)
    needed = moved & (target_exposure > 1e-8)
    unknown = needed & ~observed
    if unknown.any():
        raise ValueError(f'transport requires unobserved ocular material at {int(unknown.sum())} pixels')
    target = sample(residual, inverse) + target_exposure[..., None] * ocular
    if np.any(target[moved] < -1e-6) or np.any(target[moved] > 255 + 1e-6):
        raise ValueError('layer transport produces out-of-range RGB; no clipping permitted')
    output = source.copy()
    output[moved] = target[moved]
    return output, {
        'identityExact': bool(np.array_equal(output, source)),
        'changedPixels': int(np.any(output != source, axis=2).sum()),
        'unknownOcularSamples': 0,
        'frozenChangedPixels': int(np.any(output[frozen] != source[frozen], axis=1).sum()),
        'fullyVisibleOcularPixels': int((needed & (target_exposure == 1)).sum()),
        'floatMinimum': float(output.min()), 'floatMaximum': float(output.max()),
        'sourceLayerClaim': 'bounded estimated decomposition; optical truth requires independent source evidence',
    }
