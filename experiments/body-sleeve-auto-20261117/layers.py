"""Independent stationary sleeve over a moving complete rear shirt.

Source alpha and hidden-image priors are frozen before reading the motion.
The bounded exact source pair stays float64 through both layer operations;
there is no intermediate byte quantization or hidden RGB clipping.
"""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/body-independent-sleeve-layers-20261020'))
sys.path.insert(0,str(ROOT/'experiments/body-matte-transport-20261020'))
from source_joint_colors import fit_source_colors
from layer_transport import project_background,transport_layer


def prepare_layers(source,front_alpha,rear_alpha,front_prior,underlay_prior,scene_prior):
    source=np.asarray(source,float);front_alpha=np.asarray(front_alpha,float)
    if not np.array_equal(source[front_alpha==0],underlay_prior[front_alpha==0]):
        raise ValueError('underlay modifies known source context')
    pair=fit_source_colors(source,front_alpha,front_prior,underlay_prior)
    # The exact constrained solve can evaluate an active zero bound as
    # -1.4e-14 after subtraction. Canonicalize only IEEE roundoff here; any
    # substantive source excursion still rejects. Never clip rendered RGB.
    correction=0.
    for key in ('foreground','lower'):
        value=pair[key]
        error=max(0.,float(-value.min()),float(value.max()-255))
        if error>1e-12:raise ValueError('bounded source solver exceeded its box')
        bounded=np.minimum(np.maximum(value,0),255)
        correction=max(correction,float(abs(value-bounded).max()))
        pair[key]=bounded
    a=front_alpha[...,None]
    pair['source_reconstructed']=a*pair['foreground']+(1-a)*pair['lower']
    pair['floating_bound_correction']=correction
    scene=project_background(pair['lower'],rear_alpha,scene_prior)
    return {**pair,'source':source,'front_alpha':front_alpha,
            'rear_alpha':np.asarray(rear_alpha,float),'scene':scene}


def render_layers(pair,source_at_destination):
    rear,alpha=transport_layer(pair['lower'],pair['rear_alpha'],pair['scene'],source_at_destination)
    a=pair['front_alpha'][...,None]
    result=a*pair['foreground']+(1-a)*rear
    if not np.isfinite(result).all() or result.min() < -1e-8 or result.max()>255+1e-8:
        raise ValueError('unbounded independent-layer output; do not clip')
    return result,alpha
