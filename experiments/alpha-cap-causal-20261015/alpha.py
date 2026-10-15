"""Frozen RGB solver with isolated alpha-rule alternatives; not production code."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.ndimage import uniform_filter,distance_transform_edt
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_decompose',HERE/'frozen/decompose.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
VARIANTS=['bounded-015','uncapped','donor-interval']

def donor_variance(colors,core,radius):
    """Same local donor support and nearest donor coordinate as frozen prior."""
    _,nearest=distance_transform_edt(~core,return_indices=True)
    size=2*radius+1
    mass=uniform_filter(core.astype('float64'),size=size,mode='nearest')
    mean=np.stack([uniform_filter(colors[...,c]*core,size=size,mode='nearest')/np.maximum(mass,1e-10) for c in range(3)],axis=-1)
    second=np.stack([uniform_filter(colors[...,c]**2*core,size=size,mode='nearest')/np.maximum(mass,1e-10) for c in range(3)],axis=-1)
    variance=np.maximum(0,second-mean**2)
    return variance[nearest[0],nearest[1]]

def prepare(rgb,raw,hair_core,skin_core,domain,prior_radius):
    hair,dh=frozen.local_prior(rgb,hair_core,prior_radius)
    skin,ds=frozen.local_prior(rgb,skin_core,prior_radius)
    d=hair-skin;lam=.0025
    fit=(((rgb-skin)*d).sum(axis=-1)+lam*raw)/((d*d).sum(axis=-1)+lam)
    # A conservative diagonal donor-scatter model: mixture covariance is no
    # larger than max(varF,varB), plus 8-bit quantization noise. This is a
    # compatibility interval, not a calibrated uncertainty guarantee.
    vh=donor_variance(rgb,hair_core,prior_radius);vs=donor_variance(rgb,skin_core,prior_radius)
    variance=np.maximum(vh,vs)+(1/255)**2/12
    info=(d*d/variance).sum(axis=-1)
    center=((rgb-skin)*d/variance).sum(axis=-1)/np.maximum(info,1e-12)
    residual=((rgb-skin-center[...,None]*d)**2/variance).sum(axis=-1)
    # Chi-square(3) 95% compatibility shell, assuming independent local color
    # scatter. Systematically wrong donors can still invalidate the interval.
    compatible=(residual<=7.814727903251179)&(info>1e-12)
    half=np.sqrt(np.maximum(0,7.814727903251179-residual)/np.maximum(info,1e-12))
    # Intersect intervals. Clamping BOTH endpoints into [0,1] would turn an
    # entirely nonphysical color solution into a false opaque/transparent fit.
    lower=np.maximum(center-half,.001);upper=np.minimum(center+half,.995)
    compatible &= lower<=upper
    interval=raw.copy();interval[compatible]=np.clip(raw[compatible],lower[compatible],upper[compatible])
    proposals={
        'bounded-015':np.clip(fit,np.maximum(.001,raw-.15),np.minimum(.995,raw+.15)),
        'uncapped':np.clip(fit,.001,.995),
        'donor-interval':interval,
    }
    return dict(raw=raw,domain=domain,hairPrior=hair,skinPrior=skin,dh=dh,ds=ds,fit=fit,
        variance=variance,intervalCenter=center,intervalHalfWidth=half,intervalCompatible=compatible,
        intervalResidual=residual,separability=info,proposals=proposals)

def solve(rgb,p,variant):
    raw,domain=p['raw'],p['domain'];a=raw.copy();a[domain]=p['proposals'][variant][domain]
    hair,skin,stats=frozen.solve_layers(rgb,a,domain,p['hairPrior'],p['skinPrior'])
    reconstructed=a[...,None]*hair+(1-a[...,None])*skin
    residual=np.max(np.abs(reconstructed-rgb),axis=-1)*255
    trusted=domain&(residual<=4)
    stats.update(variant=variant,sourceMeanAbsByte=float(np.abs(reconstructed[domain]-rgb[domain]).mean()*255) if domain.any() else None,
        sourceMaxAbsByte=float(residual[domain].max()) if domain.any() else None,intrinsicTrustedPixels=int(trusted.sum()),
        alphaMeanChange=float(np.abs(a[domain]-raw[domain]).mean()) if domain.any() else None,
        alphaMaxChange=float(np.abs(a-raw).max()),changedAlphaPixels=int((domain&(np.abs(a-raw)>1e-6)).sum()),
        capWouldBindPixels=int((domain&(np.abs(p['proposals']['uncapped']-raw)>.150001)).sum()),
        compatibleIntervalPixels=int((domain&p['intervalCompatible']).sum()))
    return dict(hair=hair,skin=skin,alpha=a,sourceReconstructed=reconstructed,residual=residual,
                intrinsicTrusted=trusted,stats=stats)
