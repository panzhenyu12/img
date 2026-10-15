"""Source-only bounded F correction with fixed alpha and observed scene prior.

No inference at alpha<.1. The .1-.2 transition weights color correction only.
Channel bounds are the min/max of observed opaque source skin within a fixed
radius, plus one 8-bit code tolerance. This is a local RGB box, not a proof of
physical material or a convex-hull color model.
"""
import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter,uniform_filter


def solve(source,alpha,foreground,background,skin_core,allowed,radius=6,enabled=True):
    c,a,f,b,core,gate=map(np.asarray,(source,alpha,foreground,background,skin_core,allowed))
    size=radius*2+1
    lower=np.stack([minimum_filter(np.where(core,c[...,i],np.inf),size=size,mode='constant',cval=np.inf) for i in range(3)],-1)
    upper=np.stack([maximum_filter(np.where(core,c[...,i],-np.inf),size=size,mode='constant',cval=-np.inf) for i in range(3)],-1)
    available=np.isfinite(lower).all(axis=-1)&np.isfinite(upper).all(axis=-1)
    domain=gate&available&(a>=.1)&(a<.995)
    raw=f.copy();raw[domain]=(c[domain]-(1-a[domain,None])*b[domain])/a[domain,None]
    bounded=np.clip(raw,np.maximum(lower-1/255,0),np.minimum(upper+1/255,1))
    # Unavailable neighborhoods have infinite sentinel bounds; exclude them
    # before residual arithmetic so a refused pixel never creates 0*inf NaN.
    bounded=np.where(domain[...,None],bounded,f)
    t=np.clip((a-.1)/.1,0,1);weight=t*t*(3-2*t);weight=np.where(domain&enabled,weight,0)
    # An incompatible local color prior must not push an already better F
    # farther from its observed source fit. Refuse rather than widen bounds.
    old_residual=np.abs(c-a[...,None]*f-(1-a[...,None])*b)
    candidate_residual=np.abs(c-a[...,None]*bounded-(1-a[...,None])*b)
    nonworsening=np.all(candidate_residual<=old_residual+1e-12,axis=-1)
    weight=np.where(nonworsening,weight,0)
    output=f.copy();output[domain]=f[domain]*(1-weight[domain,None])+bounded[domain]*weight[domain,None]
    return {'foreground':output,'domain':domain,'weight':weight,'rawSolve':raw,'lower':lower,'upper':upper,'clipped':domain&np.any(np.abs(raw-bounded)>1e-10,axis=-1),'nonWorseningRefused':domain&~nonworsening,'changed':np.any(output!=f,axis=-1)}
