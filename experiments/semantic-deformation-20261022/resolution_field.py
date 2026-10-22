"""Transfer one frozen inverse map between aligned pixel-center resolutions.

This evaluates geometry at output pixel centers and scales its displacement;
it never resizes photograph pixels, changes edit strength, or solves a new
field. An output map still needs its own cell-geometry and image checks: with
nonaligned grids, bilinear reconstruction between output nodes need not equal
the reference map there, even before float32 rounding.
"""
from numbers import Integral

import numpy as np


def _integers(value, length, name):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in value)):
        raise ValueError(name+' must contain integer dimensions or coordinates')
    return [int(v) for v in value]


def _zero_collar(field):
    return (not np.any(field[:2]) and not np.any(field[-2:])
            and not np.any(field[:, :2]) and not np.any(field[:, -2:]))


def lift_inverse_field(inverse, crop, reference_size, target_size):
    """Return ``field``, ``crop``, ``scale`` and a non-approval ``report``.

    Sizes are (width, height), crop is [x, y, width, height], and inverse is the
    original finite float32 displacement in reference pixels. Only isotropic
    scale >= 1 is supported. Two exactly zero boundary rows/columns are required
    so the reference crop has a continuous identity exterior. The output crop
    encloses the transformed reference-node domain, clipped to the target image.

    For an output node q, q_ref = (q + .5)/scale - .5. Its source sample is
    q + scale * inverse(q_ref), with bilinear reference evaluation and zero
    displacement outside the reference crop. The returned float32 field is a
    new array; none of the arguments are mutated.
    """
    rw, rh = _integers(reference_size, 2, 'reference size')
    tw, th = _integers(target_size, 2, 'target size')
    left, top, width, height = _integers(crop, 4, 'reference crop')
    field = np.asarray(inverse)
    if (min(rw, rh, tw, th, width, height) < 5
            or min(left, top) < 0 or left+width > rw or top+height > rh
            or field.dtype != np.dtype('<f4') or field.shape != (height, width, 2)
            or not np.isfinite(field).all()):
        raise ValueError('finite float32 inverse and matching in-image crop required')
    if tw*rh != th*rw or tw < rw:
        raise ValueError('only exactly isotropic scale >= 1 is supported')
    if not _zero_collar(field):
        raise ValueError('two exactly zero boundary rows and columns are required')
    scale = tw/rw
    # Integer rational arithmetic avoids a floating ceil creating a spurious
    # extra column at a node that is exactly aligned in the output image.
    denominator = 2*rw
    def numerator(position):
        return (2*position+1)*tw-rw
    def floor_node(position):
        return numerator(position)//denominator
    def ceil_node(position):
        return -(-numerator(position)//denominator)
    x0, y0 = max(0, floor_node(left)), max(0, floor_node(top))
    x1 = min(tw-1, ceil_node(left+width-1))
    y1 = min(th-1, ceil_node(top+height-1))
    output_crop = [x0, y0, x1-x0+1, y1-y0+1]
    output = np.zeros((output_crop[3], output_crop[2], 2), dtype='<f4')
    max_rounding = 0.
    if scale == 1.:
        output[:] = field
    else:
        rx = (np.arange(x0, x1+1, dtype=float)+.5)/scale-.5-left
        inside_x = (rx >= 0.) & (rx <= width-1)
        sx = np.clip(rx, 0., width-1)
        ix = np.minimum(np.floor(sx).astype(int), width-2)
        fx = sx-ix
        for row in range(0, output.shape[0], 128):
            count = min(128, output.shape[0]-row)
            ry = (np.arange(y0+row, y0+row+count, dtype=float)+.5)/scale-.5-top
            inside_y = (ry >= 0.) & (ry <= height-1)
            sy = np.clip(ry, 0., height-1)
            iy = np.minimum(np.floor(sy).astype(int), height-2)
            fy = sy-iy
            sampled = np.zeros((count, len(rx), 2), float)
            for dx, wx in [(0, 1.-fx), (1, fx)]:
                for dy, wy in [(0, 1.-fy), (1, fy)]:
                    sampled += field[iy[:, None]+dy, ix[None, :]+dx] * (wy[:, None]*wx[None, :])[:, :, None]
            sampled *= scale
            sampled[~(inside_y[:, None] & inside_x[None, :])] = 0.
            rounded = sampled.astype('<f4')
            max_rounding = max(max_rounding, float(np.max(np.abs(rounded.astype(float)-sampled), initial=0.)))
            output[row:row+count] = rounded
    if not _zero_collar(output):
        raise RuntimeError('transferred inverse lost its exact zero boundary collar')
    return {'field': output, 'crop': output_crop, 'scale': scale,
            'report': {
                'reference_size': [rw, rh], 'target_size': [tw, th],
                'reference_crop': [left, top, width, height], 'target_crop': output_crop,
                'scale': scale, 'pixel_center_rule': 'q_ref = (q_target + 0.5) / scale - 0.5',
                'reference_interpolant': 'bilinear displacement, identity exterior',
                'max_float32_rounding_error_px': max_rounding,
                'input_unmodified': True, 'new_solve': False, 'photograph_resized': False,
                'zero_collar_exact': True, 'geometry_approval': False, 'aesthetic_approval': False,
                'correspondence_scope': 'Target pixel nodes, up to float32 rounding; target-cell reconstruction is checked separately.'}}


def shared_fixed_support(reference_fixed, native_fixed):
    """Union both resolutions' protection before solving the reference field.

    Masks are bool image arrays. Every native fixed pixel is mapped to the
    reference pixel-center domain; all in-bounds floor/ceil neighbors enter the
    reference mask, along with every original reference fixed pixel. Neighbors
    outside the image belong to the identity exterior and have no image index.

    This is a source constraint constructor, never output-pixel replacement or
    a post-solve displacement taper. A source solver must still enforce its
    complete fixed sampling supports. The resulting inverse, native field and
    actual fixed image bytes remain independently checked by the caller.
    """
    reference = np.asarray(reference_fixed)
    native = np.asarray(native_fixed)
    if (reference.ndim != 2 or native.ndim != 2
            or reference.dtype != np.dtype(bool) or native.dtype != np.dtype(bool)
            or min(*reference.shape, *native.shape) < 5):
        raise ValueError('two bool image masks with dimensions at least five are required')
    rh, rw = reference.shape; th, tw = native.shape
    if tw*rh != th*rw or tw < rw:
        raise ValueError('only exactly isotropic scale >= 1 is supported')
    scale = tw/rw
    required = np.zeros_like(reference)
    if scale == 1.:
        required[:] = native
    else:
        # Work in row blocks: a full-resolution protection mask is usually
        # dense, so a complete floating-point coordinate image is unnecessary.
        for begin in range(0, th, 128):
            yy, xx = np.where(native[begin:begin+128])
            if not len(xx):
                continue
            qx = (xx.astype(float)+.5)/scale-.5
            qy = (yy.astype(float)+begin+.5)/scale-.5
            for sy in (np.floor(qy).astype(int), np.ceil(qy).astype(int)):
                for sx in (np.floor(qx).astype(int), np.ceil(qx).astype(int)):
                    inside = (sx >= 0) & (sx < rw) & (sy >= 0) & (sy < rh)
                    required[sy[inside], sx[inside]] = True
    combined = reference | required
    return {'mask': combined, 'report': {
        'reference_size': [rw, rh], 'native_size': [tw, th], 'scale': scale,
        'reference_fixed_pixels': int(np.count_nonzero(reference)),
        'native_fixed_pixels': int(np.count_nonzero(native)),
        'native_required_reference_pixels': int(np.count_nonzero(required)),
        'combined_fixed_pixels': int(np.count_nonzero(combined)),
        'added_reference_pixels': int(np.count_nonzero(required & ~reference)),
        'original_reference_protection_preserved': True,
        'native_reference_support': 'All in-bounds floor/ceil reference pixels for each native fixed pixel center; exterior is identity.',
        'input_unmodified': True, 'post_solve_masking': False,
        'geometry_approval': False, 'aesthetic_approval': False}}
