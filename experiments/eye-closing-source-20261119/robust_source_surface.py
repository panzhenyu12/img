"""Source-only ocular surface and explicit thin-lash layer inference.

Callers supply source material roles, never optical alpha or a hidden RGB plate.
The domain only bounds where estimation is allowed. Failed estimates remain
unknown; the caller must refuse edits that depend on them.
"""
import numpy as np
from scipy.ndimage import label


def robust_local_surface(source,observed,targets,*,radius,noise):
    """Eight fixed Huber IRLS iterations; source residual MAD, no output fit."""
    h,w=observed.shape;predicted=np.zeros_like(source)
    accepted=np.zeros_like(observed);scales=np.ones_like(observed,dtype=float)*noise
    counts=np.zeros_like(observed,dtype=int);records=[]
    for y,x in np.argwhere(targets):
        x0=max(0,int(np.floor(x-radius)));x1=min(w,int(np.ceil(x+radius))+1)
        y0=max(0,int(np.floor(y-radius)));y1=min(h,int(np.ceil(y+radius))+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];dx=(xx-x)/radius;dy=(yy-y)/radius;r2=dx*dx+dy*dy
        valid=observed[y0:y1,x0:x1]&(r2<1)
        count=int(valid.sum())
        if count<6:continue
        matrix=np.stack([np.ones(count),dx[valid],dy[valid]],axis=-1)
        colors=source[y0:y1,x0:x1][valid]
        spatial=(1-r2[valid])**2;weights=spatial.copy()
        sigma=noise
        for iteration in range(9):
            weighted=matrix*np.sqrt(weights[:,None]);singular=np.linalg.svd(weighted,compute_uv=False)
            if singular[-1]<=0 or singular[0]/singular[-1]>1e3:break
            beta=np.linalg.lstsq(weighted,colors*np.sqrt(weights[:,None]),rcond=None)[0]
            residual=colors-matrix@beta
            sigma=max(noise,float(1.4826*np.median(abs(residual-np.median(residual,axis=0)))))
            if iteration<8:
                rms=np.sqrt(np.mean(residual**2,axis=-1))
                weights=spatial*np.minimum(1,1.345*sigma/np.maximum(rms,1e-20))
        else:
            background=beta[0]
            if not np.isfinite(background).all() or np.any((background<0)|(background>255)):continue
            # A one-sided fit carries larger prediction uncertainty through
            # its source-only design leverage. This is an empirical model
            # scale, not a calibrated probability of a material identity.
            leverage=float(np.linalg.inv(matrix.T@(weights[:,None]*matrix))[0,0])
            prediction_scale=sigma*np.sqrt(1+leverage)
            predicted[y,x]=background;accepted[y,x]=True
            scales[y,x]=prediction_scale;counts[y,x]=count
            records.append(dict(pixel=[int(x),int(y)],donorCount=count,
                residualMADScale=sigma,predictionScale=prediction_scale,
                predictionLeverage=leverage,sourceBackground=background.tolist()))
    return predicted,accepted,scales,counts,records


def infer_moving_source_layer(source,moving_support,ocular_donors,ocular_domain,
                              noise,*,radius=6.0):
    """Estimate B/E/R only on explicitly supplied thin-lash support.

    This model is for dark thin lashes and their mixed edges inside the eye
    aperture. It is not a skin/eyelid or arbitrary garment material model.
    Ocular donors must be explicitly observed, source-bound material anchors.
    A larger ocular domain licenses estimation, not donor use or visibility.

    `estimated` is the only valid inferred-layer mask. `unknown` is the moving
    support without an adequate source model; it must not be made observable
    by a caller fallback. Arrays outside `estimated` are diagnostic proposals.
    """
    source=np.asarray(source,float)
    if source.ndim!=3 or source.shape[-1]!=3:
        raise ValueError('source must be an RGB image')
    if not np.isfinite(source).all() or np.any((source<0)|(source>255)):
        raise ValueError('source must have finite bounded RGB values')
    masks=[]
    for name,mask in [('moving_support',moving_support),('ocular_donors',ocular_donors),('ocular_domain',ocular_domain)]:
        mask=np.asarray(mask)
        if mask.shape!=source.shape[:2] or mask.dtype!=np.bool_:
            raise ValueError(f'{name} must be a boolean source-sized mask')
        masks.append(mask)
    moving,donors,domain=masks
    if np.any(moving&donors):raise ValueError('moving support and observed ocular donors conflict')
    if np.any(donors&~domain):raise ValueError('ocular donors lie outside the permitted source domain')
    if not np.isfinite(noise) or noise<=0:raise ValueError('positive finite source noise is required')
    if not np.isfinite(radius) or radius<=0:raise ValueError('positive finite source radius is required')
    targets=moving&domain
    background,fitted,scale,counts,records=robust_local_surface(source,donors,targets,
                                                              radius=radius,noise=noise)
    pigment=np.zeros_like(source);pigment_records=[]
    components,count=label(moving,np.ones((3,3),int))
    luma=source@np.array([.2126,.7152,.0722])
    for identifier in range(1,count+1):
        region=components==identifier;ys,xs=np.where(region)
        darkest=int(np.argmin(luma[region]));py,px=int(ys[darkest]),int(xs[darkest])
        pigment[region]=source[py,px]
        pigment_records.append(dict(component=identifier,pixels=int(region.sum()),
            sourcePigmentPixel=[px,py],sourcePigmentRGB=source[py,px].tolist()))
    direction=background-pigment
    exposure=np.clip(np.sum((source-pigment)*direction,axis=-1)/
        np.maximum(np.sum(direction*direction,axis=-1),1e-20),0,1)
    # Fit the mixing coefficient on the physically feasible segment. Any
    # adjustment must still explain the source with the SAME pigment model;
    # arbitrary bounded residuals are not promoted to known lash material.
    cap=np.ones(source.shape[:2])
    for channel in range(3):
        b=background[...,channel];c=source[...,channel]
        cap=np.minimum(cap,np.divide(c,b,out=np.ones_like(b),where=b>0))
        cap=np.minimum(cap,np.divide(255-c,255-b,out=np.ones_like(b),where=b<255))
    before_cap=exposure.copy();exposure=np.minimum(exposure,cap)
    model=exposure[...,None]*background+(1-exposure[...,None])*pigment
    error=np.sqrt(np.mean((source-model)**2,axis=-1))
    residual=source-exposure[...,None]*background
    physical=np.all(residual>=-1e-10,axis=-1)&np.all(residual<=255*(1-exposure[...,None])+1e-10,axis=-1)
    # One residual variance per channel is the fixed adequacy rule used by
    # the robust candidate. The scale includes source-only fit uncertainty;
    # it is an empirical model bound, not a calibrated material probability.
    model_consistent=fitted&(error<=scale)&physical
    estimated=targets&model_consistent
    unknown=moving&~estimated
    report=dict(method='explicit-source-lash-robust-affine-underlay-v1',
        radiusSourcePixels=radius,sourceNoise=float(noise),huberConstant=1.345,
        irlsIterations=8,sourceSurfaceFitRecords=records,sourcePigmentRecords=pigment_records,
        explicitMovingPixels=int(moving.sum()),explicitOcularDonorPixels=int(donors.sum()),
        fittedPixels=int(fitted.sum()),estimatedPixels=int(estimated.sum()),
        unknownPixels=int(unknown.sum()),outsideDomainPixels=int((moving&~domain).sum()),
        modelConflictPixels=int((targets&fitted&~model_consistent).sum()),
        feasibilityAdjustedPixels=int((targets&fitted&(before_cap>exposure+1e-12)).sum()),
        sourceReconstructionMaxError=float(np.max(abs(residual+exposure[...,None]*background-source))),
        candidateDependsOnEditedOutput=False,hiddenOcularIsObserved=False,
        restriction='explicit thin dark lash support only; model failure remains unknown; no skin/eyelid material claim')
    return dict(ocular=background,exposure=exposure,residual=residual,
        estimated=estimated,unknown=unknown,modelConsistent=model_consistent,
        fitError=error,predictionScale=scale,pigment=pigment,report=report)

