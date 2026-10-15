"""Material-context ablation helpers; no geometry or production changes.

Known regions are confidence-based context hypotheses, not ground truth for the
occluded layer. The generated auxiliary layer must never replace the transported
foreground or write outside the independent exact disocclusion mask.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_user_face_contour import raster_coverage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context_masks(params, size, source_alpha, origin):
    """Return hair-only and hair+scene context using fixed source observations.

    Source alpha is a crop-local f32 physical matte; graph confidences are
    bilinearly sampled with the same pixel centers as Rust PreparedMask. Mask
    confidence only selects known inputs, never the final write target.
    """
    width, height = size
    semantics = params['semanticMasks']
    masks = {k: raster_coverage(v, width, height)
             for k, v in semantics.items() if v.get('shape') == 'raster'}
    person = raster_coverage(params['subjectMask'], width, height)
    foreground = np.maximum.reduce([masks[k] for k in
                                    ['faceContourMask', 'skin', 'neck', 'ears', 'accessories']])
    ox, oy = origin
    hh, ww = source_alpha.shape
    physical = np.zeros((height, width), bool)
    physical[oy:oy+hh, ox:ox+ww] = source_alpha > np.float32(1e-6)
    # A conservative exclusion ring keeps ambiguous mixed edge RGB out of
    # model conditioning. It does not erase or blur the final photograph.
    excluded = binary_dilation((foreground > 1/255) | physical, iterations=2)
    opaque = np.ones((height, width), bool)
    hair = (masks['hair'] >= .95) & ~excluded
    scene = (person <= 1/255) & ~excluded
    return {'hair-only': hair & opaque, 'hair-scene': (hair | scene) & opaque}, excluded


def byte(values):
    # Rust f32.round() for nonnegative clamped RGB.
    return np.floor(np.clip(values, 0, 255) + np.float32(.5)).astype('uint8')


def compose(background, transported, membership):
    """Mirror face.rs premultiplied f32 over, retaining output rounding."""
    bg = background.astype('float32')
    a_bg = bg[:, :, 3] / np.float32(255)
    bg_weight = a_bg * (np.float32(1) - membership)
    final_alpha = transported[:, :, 3] + bg_weight
    result = np.zeros(background.shape, dtype='uint8')
    for c in range(3):
        numerator = transported[:, :, c] + bg[:, :, c] * bg_weight
        np.divide(numerator, final_alpha, out=numerator, where=final_alpha > 1e-6)
        numerator[final_alpha <= 1e-6] = 0
        result[:, :, c] = byte(numerator)
    result[:, :, 3] = byte(final_alpha * np.float32(255))
    return result


def montage(images, labels, path, box=None, zoom=1):
    ims = [Image.fromarray(im).convert('RGB') if isinstance(im, np.ndarray)
           else im.convert('RGB') for im in images]
    if box:
        ims = [im.crop(box) for im in ims]
    w, h = ims[0].size
    canvas = Image.new('RGB', (w*zoom*len(ims), h*zoom+28), '#20242a')
    draw = ImageDraw.Draw(canvas)
    for index, (im, label) in enumerate(zip(ims, labels)):
        canvas.paste(im.resize((w*zoom, h*zoom), Image.Resampling.NEAREST), (index*w*zoom, 28))
        draw.text((index*w*zoom+5, 7), label, fill='white')
    canvas.save(path)
