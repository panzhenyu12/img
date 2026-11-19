"""One frozen RGB-guided eye/lid/lash source candidate, not optical truth.

Unlike the earlier geometric-area/donor experiment, exposure is a fitted RGB
mixture variable. The local ocular prior is transported nowhere and projected
nowhere. Its continuity along a source normal is an explicit assumption.
"""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates


def read(rgb, coordinates):
    return np.stack([map_coordinates(rgb[..., c],
        [coordinates[..., 1], coordinates[..., 0]], order=1,
        mode='nearest', prefilter=False) for c in range(3)], axis=-1)


def smoothstep(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def triangle_weights(color, ocular, skin, lash, prior, noise):
    """Closest point on a regularized color triangle, via its seven faces.

    This estimates material weights. The weak final coordinate records a
    geometric exposure prior only where RGB cannot identify a unique mixture.
    No fitted or projected ocular RGB is substituted into the supplied prior.
    """
    c=np.concatenate([color, (noise*prior)[...,None]],axis=-1)
    b=np.concatenate([ocular, np.full(prior.shape+(1,),noise)],axis=-1)
    f=np.concatenate([skin, np.zeros(prior.shape+(1,))],axis=-1)
    l=np.concatenate([lash, np.zeros(prior.shape+(1,))],axis=-1)
    vectors=[b,f,l]
    choices=[]
    for k in range(3):
        weights=np.zeros(prior.shape+(3,));weights[...,k]=1
        choices.append(weights)
    for i,j in [(0,1),(0,2),(1,2)]:
        delta=vectors[j]-vectors[i]
        fraction=np.clip(np.sum((c-vectors[i])*delta,axis=-1)/
                         np.maximum(np.sum(delta*delta,axis=-1),1e-20),0,1)
        weights=np.zeros(prior.shape+(3,));weights[...,i]=1-fraction;weights[...,j]=fraction
        choices.append(weights)
    a=f-b;d=l-b;z=c-b
    aa=np.sum(a*a,axis=-1);dd=np.sum(d*d,axis=-1);ad=np.sum(a*d,axis=-1)
    az=np.sum(a*z,axis=-1);dz=np.sum(d*z,axis=-1);det=aa*dd-ad*ad
    good=det>1e-12
    wf=np.divide(az*dd-dz*ad,det,out=np.zeros_like(det),where=good)
    wl=np.divide(dz*aa-az*ad,det,out=np.zeros_like(det),where=good)
    weights=np.stack([1-wf-wl,wf,wl],axis=-1)
    legal=good&np.all(weights>=0,axis=-1)
    # An illegal interior solution is excluded, not clipped into a new color.
    choices.append(weights)
    costs=[]
    for index,weights in enumerate(choices):
        reconstructed=sum(weights[...,k,None]*vectors[k] for k in range(3))
        cost=np.sum((reconstructed-c)**2,axis=-1)
        if index==len(choices)-1:cost=np.where(legal,cost,np.inf)
        costs.append(cost)
    best=np.argmin(np.stack(costs,axis=-1),axis=-1)
    weights=np.take_along_axis(np.stack(choices,axis=-2),best[...,None,None],axis=-2)[...,0,:]
    fit=weights[...,0,None]*ocular+weights[...,1,None]*skin+weights[...,2,None]*lash
    return weights,np.linalg.norm(fit-color,axis=-1)


def estimate(source, local_u, local_v, upper, lower, to_pixel, width, *, band=3.0):
    """Estimate only from one source crop and its frozen geometric frame.

    to_pixel(u,v) maps local canthus coordinates into this crop's pixel-index
    coordinates. Boundary-band material support is an estimate; it does not
    promote landmark geometry or dark iris/lash ambiguity to optical truth.
    """
    source=np.asarray(source,float);u=np.asarray(local_u,float);v=np.asarray(local_v,float)
    if source.shape!=u.shape+(3,) or v.shape!=u.shape:
        raise ValueError('source/frame dimensions differ')
    if not np.isfinite(source).all() or np.any((source<0)|(source>255)):
        raise ValueError('bounded finite source RGB required')
    safeu=np.clip(u,0,width);top=upper(safeu);bottom=lower(safeu)
    gap=np.maximum(0,bottom-top);inside=(u>=0)&(u<=width)
    upper_side=(v-top)<=(bottom-v)
    boundary=np.where(upper_side,top,bottom);inward=np.where(upper_side,1.,-1.)
    depth=np.minimum(v-top,bottom-v)
    # Donors remain in this source normal column. Interior depth cannot cross
    # the aperture middle. This preserves iris-vs-sclera variation across u,
    # unlike Euclidean nearest-core fill from another part of the eyeball.
    donor_depth=np.minimum(band+1.,gap*.45)
    ocular_samples=[];skin_samples=[];ocular_coordinates=[]
    for offset in [0.,.5,1.]:
        at=np.minimum(donor_depth+offset,gap*.48)
        coordinates=to_pixel(safeu,boundary+inward*at)
        ocular_coordinates.append(coordinates)
        ocular_samples.append(read(source,coordinates))
        skin_samples.append(read(source,to_pixel(safeu,boundary-inward*(band+offset))))
    ocular=np.median(np.stack(ocular_samples),axis=0)
    skin=np.median(np.stack(skin_samples),axis=0)
    # Dark color is a source appearance proposal, not a mask. A pixel enters
    # the lash mixture only if its own RGB supports that color, except where
    # the explicit weak prior resolves an underdetermined dark-on-dark case.
    dark_samples=[]
    for du in [-2.,0.,2.]:
        at_u=np.clip(safeu+du,0,width)
        at_boundary=np.where(upper_side,upper(at_u),lower(at_u))
        for dv in [-1.,0.,1.]:
            dark_samples.append(read(source,to_pixel(at_u,at_boundary+inward*dv)))
    dark_samples=np.stack(dark_samples,axis=-2)
    dark_luma=dark_samples@np.array([.2126,.7152,.0722])
    darkest=np.argmin(dark_luma,axis=-1)
    lash=np.take_along_axis(dark_samples,darkest[...,None,None],axis=-2)[...,0,:]
    luminance=source@np.array([.2126,.7152,.0722])
    noise=max(1.,float(1.4826*np.median(abs(luminance-gaussian_filter(luminance,.6)))))
    prior=smoothstep((depth+band)/(2*band))*inside
    weights,fit_error=triangle_weights(source,ocular,skin,lash,prior,noise)
    exposure=weights[...,0]
    support=inside&(depth>-band)
    core=inside&(depth>=band)
    exposure[~support]=0
    exposure[core]=1;ocular[core]=source[core]
    # Keep B fixed. Feasibility can only lower E; count/report this explicitly.
    # It is not a claim that the resulting alpha is physically correct.
    cap=np.ones_like(exposure)
    for c in range(3):
        b=ocular[...,c];observed=source[...,c]
        cap=np.minimum(cap,np.divide(observed,b,out=np.ones_like(b),where=b>0))
        cap=np.minimum(cap,np.divide(255-observed,255-b,out=np.ones_like(b),where=b<255))
    change=np.maximum(0,exposure-cap)
    exposure=np.minimum(exposure,cap)
    ocular[~support]=0
    residual=source-exposure[...,None]*ocular
    active=support&~core
    separation=np.linalg.norm(ocular-lash,axis=-1)
    report={
        'method':'source-normal-three-color-exposure-v1','bandSourcePixels':band,
        'noise':noise,'corePixels':int(core.sum()),'estimatedBandPixels':int(active.sum()),
        'lowOcularLashSeparationPixels':int((active&(separation<3*noise)).sum()),
        'medianBandColorFitError':float(np.median(fit_error[active])) if active.any() else 0.,
        'maxBandColorFitError':float(np.max(fit_error[active])) if active.any() else 0.,
        'exposureFeasibilityReducedPixels':int((change>1e-8).sum()),
        'exposureFeasibilityReductionMaximum':float(change.max()),
        'ocularProjectedPixels':0,'sourceReconstructionMaxError':float(np.max(abs(residual+exposure[...,None]*ocular-source))),
        'sourceResidualMinimum':float(residual.min()),
        'sourceResidualUpperViolation':float(np.max(residual-255*(1-exposure[...,None]))),
        'supportClaim':'source-fixed local underlay estimation domain; not directly observed hidden eye',
        'opticalTruth':False,'geometryPriorIsNotPhysicalAlpha':True,
    }
    return dict(exposure=exposure,ocular=ocular,support=support,core=core,
                lashWeight=weights[...,2],ambiguous=active&(separation<3*noise),
                fitError=fit_error,ocularDonorCoordinates=np.stack(ocular_coordinates),
                skinPrior=skin,lashPrior=lash,report=report)


def prepare_sampling_support(source, prepared):
    """Preserve the fixed B proposal through one bilinear source footprint.

    A zero at a pixel center is not a proof that its filter footprint has zero
    exposure. This source-only support rule neither changes E nor reads a
    target field. It explicitly adds estimated B, not observed hidden RGB.
    """
    result={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in prepared.items()}
    source_footprint=binary_dilation(prepared['exposure']>0,structure=np.ones((3,3),bool))
    added=source_footprint&~prepared['support']
    prior=np.median(np.stack([read(source,p) for p in prepared['ocularDonorCoordinates']]),axis=0)
    result['ocular'][added]=prior[added]
    result['support']|=source_footprint
    result['addedSamplingSupport']=added
    return result
