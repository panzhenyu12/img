"""Bounded source F/lower colour fit in the nullspace of exact compositing.

The matte is held fixed. Both priors receive unit weight; there is no strength,
target geometry, output feedback or searched regularization parameter.
"""
import numpy as np


def fit_source_colors(image,alpha,foreground_prior,lower_prior):
    """Minimize ||F-Fp||²+||L-Lp||² s.t. I=aF+(1-a)L, 0<=F,L<=255.

    Parameterize F=I+(1-a)d, L=I-a*d. The source equality is exact for any d;
    each colour channel has an interval for d from the two RGB box constraints.
    Projecting the unconstrained scalar minimizer onto that interval gives the
    exact bounded least-squares solution. This is not clipping rendered RGB.
    """
    image=np.asarray(image,float);alpha=np.asarray(alpha,float);fp=np.asarray(foreground_prior,float);lp=np.asarray(lower_prior,float)
    if image.ndim!=3 or image.shape[-1]!=3 or fp.shape!=image.shape or lp.shape!=image.shape or alpha.shape!=image.shape[:2]:raise ValueError('Source RGB, priors and optical alpha must agree')
    if not all(np.isfinite(v).all() for v in [image,alpha,fp,lp]) or np.any((alpha<0)|(alpha>1)) or any(np.any((v<0)|(v>255)) for v in [image,fp,lp]):raise ValueError('Finite bounded inputs are required')
    a=alpha[...,None];b=1-a;unbounded=(b*(fp-image)-a*(lp-image))/(a*a+b*b)
    lower=np.maximum(np.divide(-image,b,out=np.full_like(image,-np.inf),where=b>0),np.divide(image-255,a,out=np.full_like(image,-np.inf),where=a>0))
    upper=np.minimum(np.divide(255-image,b,out=np.full_like(image,np.inf),where=b>0),np.divide(image,a,out=np.full_like(image,np.inf),where=a>0))
    d=np.minimum(np.maximum(unbounded,lower),upper)
    foreground=image+b*d;rear=image-a*d
    return {'foreground':foreground,'lower':rear,'nullspace_offset':d,'unbounded_offset':unbounded,'lower_bound':lower,'upper_bound':upper,'bound_active':(d!=unbounded),'source_reconstructed':a*foreground+b*rear}
