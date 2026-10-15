"""Conditional skin/scene color inversion; no material or background inference.

Only applies at caller-confirmed skin-over-known-scene pixels. Other pixels
remain unchanged. A lower bound ensures RGB gamut feasibility, not alpha truth.
"""
import numpy as np


def lower_bound(source, background):
    difference=source-background
    above=np.divide(difference,1-background,out=np.zeros_like(difference),where=background<1)
    below=np.divide(-difference,background,out=np.zeros_like(difference),where=background>0)
    return np.clip(np.maximum(above,below).max(axis=-1),0,1)


def solve(source,background,alpha_prior,foreground_prior,confirmed_skin_scene):
    c,b,a,f,d=map(np.asarray,(source,background,alpha_prior,foreground_prior,confirmed_skin_scene))
    if c.shape!=b.shape or c.shape!=f.shape or a.shape!=d.shape or c.shape[:-1]!=a.shape:raise ValueError('shape mismatch')
    if not all(np.isfinite(v).all() for v in (c,b,a,f)):raise ValueError('nonfinite inputs')
    if any(((v<0)|(v>1)).any() for v in (c,b,a,f)):raise ValueError('inputs must be normalized to [0,1]')
    bound=lower_bound(c,b);corrected=a.copy()
    # Minimum change of the prior within the physically feasible [lower, 1]
    # interval. Feasibility does not determine which alpha is the true one.
    corrected[d]=np.maximum(a[d],bound[d])
    color=f.copy();positive=d&(corrected>0)
    color[positive]=np.clip(b[positive]+(c[positive]-b[positive])/corrected[positive,None],0,1)
    return {'alpha':corrected,'foreground':color,'lowerBound':bound,'changed':d&(corrected!=a)}
