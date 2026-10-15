"""Closest bounded F/B colors that explain the observed source at fixed alpha.

Orthogonal projection in encoded RGB is a diagnostic baseline, not a physical
lighting model or a claim that the supplied alpha/background prior is correct.
"""
import numpy as np


def project(source, alpha, foreground, background):
    c=np.asarray(source,dtype='float64');a=np.asarray(alpha,dtype='float64')[...,None]
    f=np.asarray(foreground,dtype='float64');b=np.asarray(background,dtype='float64')
    beta=1-a;denom=a*a+beta*beta
    residual=c-a*f-beta*b
    unconstrained=f+a*residual/denom
    safe_a=np.where(a>0,a,1)
    lower=np.maximum(0,(c-beta*255)/safe_a)
    upper=np.minimum(255,c/safe_a)
    lower=np.where(a>0,lower,0);upper=np.where(a>0,upper,255)
    projected_f=np.minimum(np.maximum(unconstrained,lower),upper)
    projected_b=np.where(beta>0,(c-a*projected_f)/np.where(beta>0,beta,1),b)
    projected_f=np.where(a==1,c,projected_f)
    projected_b=np.where(a==0,c,projected_b)
    # Numerical roundoff alone can lie just beyond an endpoint.
    return np.clip(projected_f,0,255).astype('float32'),np.clip(projected_b,0,255).astype('float32')
