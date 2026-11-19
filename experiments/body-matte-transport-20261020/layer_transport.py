"""Transport observed premultiplied subject evidence over a static background.

Isolated numerical prototype, in the caller's RGB encoding. A predicted matte
and background are estimates, not known physical layers. Do not silently clip
the recovered foreground term: that would destroy identity even for zero
motion. The caller evaluates excursions and decides where to use the result.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def merge_visible_material_coverages(coverages):
    """Area union for mutually exclusive *visible* subpixel materials.

    A pixel shared by white and black cloth still has full foreground
    coverage when their visible contributions sum to one. Taking the maximum
    incorrectly invents background between them. This rule is not valid for
    amodal layers, duplicate object masks or arbitrary overlapping selections.
    Independent model estimates may sum above one; the bounded estimate clips
    that inconsistent mass and the caller must report it, not claim truth.
    """
    values=np.asarray(coverages,dtype=float)
    if values.ndim!=3 or values.shape[0]<1 or not np.isfinite(values).all():
        raise ValueError('Expected one or more finite 2D visible-material coverages')
    if np.any((values<0)|(values>1)):
        raise ValueError('Visible-material coverage must be within [0,1]')
    return np.minimum(np.sum(values,axis=0),1.)


def project_background(source_rgb, source_coverage, estimated_background):
    """Closest RGB plate consistent with some bounded foreground at this alpha.

    Per channel, I = aF + (1-a)B and 0 <= F,B <= 255. This does not
    establish that model coverage is physically correct, or recover hidden
    detail. It prevents subtracting an impossible background contribution.
    """
    image=np.asarray(source_rgb,dtype=float)
    alpha=np.asarray(source_coverage,dtype=float)
    estimated=np.asarray(estimated_background,dtype=float)
    if image.ndim!=3 or image.shape[-1]!=3 or estimated.shape!=image.shape or alpha.shape!=image.shape[:2]:
        raise ValueError('RGB layers and coverage must match')
    if not all(np.isfinite(v).all() for v in [image,alpha,estimated]):
        raise ValueError('layer evidence must be finite')
    if np.any((image<0)|(image>255)) or np.any((alpha<0)|(alpha>1)):
        raise ValueError('source RGB must be in [0,255], coverage in [0,1]')
    transmission=1-alpha[...,None]
    lower=np.divide(image-alpha[...,None]*255,transmission,
                    out=np.zeros_like(image),where=transmission>0)
    upper=np.divide(image,transmission,out=np.full_like(image,255),where=transmission>0)
    return np.minimum(np.maximum(estimated,np.maximum(lower,0)),np.minimum(upper,255))


def sample(values, source_at_destination):
    values = np.asarray(values, dtype=float)
    xy = np.asarray(source_at_destination, dtype=float)
    if values.ndim == 2:
        return map_coordinates(values, [xy[..., 1], xy[..., 0]],
                               order=1, mode='nearest', prefilter=False)
    return np.stack([sample(values[..., c], xy) for c in range(values.shape[-1])], -1)


def prepare_fixed_plate(source_rgb, source_coverage, background_rgb):
    """Raise inconsistent estimated coverage, keeping the predicted plate fixed.

    For fixed I,B and bounded F, each channel requires
      a >= (I-B)/(255-B) when I>B, or (B-I)/B when I<B.
    Their maximum is the minimum feasible scalar coverage. Raising the supplied
    alpha only as necessary avoids encoding bright cloth in the hidden plate.
    This is an explicit alternative estimate, not recovered physical alpha:
    an incorrect B can still force false opacity. Callers must preserve raw
    evidence and report the changed coverage, including its transported edge.
    """
    image=np.asarray(source_rgb,dtype=float)
    alpha=np.asarray(source_coverage,dtype=float)
    background=np.asarray(background_rgb,dtype=float)
    if image.ndim!=3 or image.shape[-1]!=3 or background.shape!=image.shape or alpha.shape!=image.shape[:2]:
        raise ValueError('RGB layers and coverage must match')
    if not all(np.isfinite(v).all() for v in [image,alpha,background]):
        raise ValueError('layer evidence must be finite')
    if any(np.any((v<0)|(v>255)) for v in [image,background]) or np.any((alpha<0)|(alpha>1)):
        raise ValueError('RGB must be in [0,255], coverage in [0,1]')
    difference=image-background
    brighter=np.divide(difference,255-background,out=np.zeros_like(image),where=difference>0)
    darker=np.divide(-difference,background,out=np.zeros_like(image),where=difference<0)
    minimum=np.max(np.maximum(brighter,darker),axis=-1)
    return np.maximum(alpha,minimum), background.copy()


def transport_layer(source_rgb, source_coverage, background_rgb,
                    source_at_destination, destination_background_rgb=None):
    image = np.asarray(source_rgb, dtype=float)
    coverage = np.asarray(source_coverage, dtype=float)
    background = np.asarray(background_rgb, dtype=float)
    destination_background = (background if destination_background_rgb is None
                              else np.asarray(destination_background_rgb, dtype=float))
    xy = np.asarray(source_at_destination, dtype=float)
    if (image.ndim != 3 or image.shape[-1] != 3 or background.shape != image.shape
            or destination_background.shape != image.shape
            or coverage.shape != image.shape[:2] or xy.shape != (*coverage.shape, 2)):
        raise ValueError('RGB layers, coverage and sampling coordinates must match')
    if not all(np.isfinite(v).all() for v in [image, coverage, background, destination_background, xy]):
        raise ValueError('layer evidence must be finite')
    if np.any((coverage < 0) | (coverage > 1)):
        raise ValueError('coverage must be within [0,1]')
    destination_coverage = sample(coverage, xy)
    # The observed image already contains both layers. Subtract its estimated
    # source background contribution before transporting, without dividing by
    # a tiny alpha or multiplying the already mixed RGB by alpha a second time.
    premultiplied_foreground = image - (1 - coverage[..., None]) * background
    transported = sample(premultiplied_foreground, xy)
    # With independent layers, the lower layer in the destination may itself
    # move. Source background is still subtracted BEFORE sampling; only this
    # final over operation uses the destination lower layer.
    result = transported + (1 - destination_coverage[..., None]) * destination_background
    return result, destination_coverage
