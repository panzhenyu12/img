"""Fixed robust surface and source model comparison, isolated candidate.

The V4 semantic candidate and its one-pixel sampling support are unchanged.
This estimates a robust local ocular surface from source observations, then
compares an ocular observation against a dark two-color occluder explanation.
An arbitrary physically feasible RGB residual is not evidence for a lash.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label

from observed_ocular_v4 import estimate_observed_ocular as estimate_v4


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


def estimate_observed_ocular(source,local_u,local_v,upper,lower,to_pixel,width,
                            iris_center,iris_radii,*,band=3.0):
    prepared=estimate_v4(source,local_u,local_v,upper,lower,to_pixel,width,
                         iris_center,iris_radii,band=band)
    source=np.asarray(source,float);u=np.asarray(local_u,float);v=np.asarray(local_v,float)
    centre=prepared['lashVeto'].copy()
    foreground=binary_dilation(centre,structure=np.ones((3,3),bool))
    safe=np.clip(u,0,width);depth=np.minimum(v-upper(safe),lower(safe)-v)
    targets=foreground&(u>=0)&(u<=width)&(depth>=1.)
    observed=prepared['core']&~foreground&(depth>=1.)
    noise=prepared['report']['noise']
    background,fitted,scale,counts,records=robust_local_surface(source,observed,targets,
                                                              radius=2*band,noise=noise)
    pigment=prepared['lashPrior'].copy();luma=source@np.array([.2126,.7152,.0722])
    components,count=label(centre,np.ones((3,3),int))
    for identifier in range(1,count+1):
        region=components==identifier;ys,xs=np.where(region)
        darkest=np.argmin(luma[region]);support=binary_dilation(region,structure=np.ones((3,3),bool))
        pigment[targets&support]=source[ys[darkest],xs[darkest]]
    delta=background-pigment
    exposure=np.clip(np.sum((source-pigment)*delta,axis=-1)/
        np.maximum(np.sum(delta*delta,axis=-1),1e-20),0,1)
    occluder_rgb=exposure[...,None]*background+(1-exposure[...,None])*pigment
    ocular_deviance=np.sum((source-background)**2,axis=-1)/scale**2
    occluder_deviance=np.sum((source-occluder_rgb)**2,axis=-1)/scale**2
    # One additional mixing parameter incurs a fixed information penalty over
    # the three-channel source observations. We also require an adequate
    # residual (<= one empirical variance per RGB channel), not merely a less
    # bad model. No threshold is fitted to edited images or counterexamples.
    penalty=np.log(np.maximum(3*(counts+1),1))
    negative=((background-source)@np.array([.2126,.7152,.0722]))>0
    dark_pigment=((background-pigment)@np.array([.2126,.7152,.0722]))>0
    accept=fitted&negative&dark_pigment&(exposure<1)
    accept&=(occluder_deviance<=3)&(ocular_deviance>occluder_deviance+penalty)
    residual=source-exposure[...,None]*background
    feasible=np.all(residual>=-1e-10,axis=-1)&np.all(residual<=255*(1-exposure[...,None])+1e-10,axis=-1)
    accept&=feasible
    # The source eye is directly visible throughout this interior aperture.
    # If the occluder explanation lacks evidence, retain observed source RGB;
    # never turn the rejected model's arbitrary residual into moving material.
    prepared['exposure'][targets]=1
    prepared['ocular'][targets]=source[targets]
    prepared['core'][targets]=True
    prepared['lashVeto'][targets]=False
    prepared['exposure'][accept]=exposure[accept]
    prepared['ocular'][accept]=background[accept]
    prepared['lashPrior'][accept]=pigment[accept]
    prepared['core'][accept]=False
    prepared['observedIris'][accept]=False
    prepared['lashVeto'][accept]=True
    prepared['estimatedLashUnderlay']=accept
    prepared['robustUnderlayCandidate']=background
    prepared['robustUnderlayScale']=scale
    prepared['occluderModelAccepted']=accept
    prepared['occluderModelDeviance']=occluder_deviance
    prepared['ocularModelDeviance']=ocular_deviance
    prepared['report'].update(method='source-robust-surface-two-explanation-v7',
        robustSurfaceIterations=8,robustHuberConstant=1.345,sourceSurfaceRadius=2*band,
        sourceOcularModel='retain source RGB unless a negative two-color occluder adequately fits and improves the fixed source information criterion',
        sourceRobustSurfaceRecords=records,
        sourceSemanticCandidatePixels=int(targets.sum()),
        sourceAcceptedOccluderPixels=int(accept.sum()),
        sourceFeasibilityCapsApplied=0,
        sourceOcularAnchors=int(prepared['core'].sum()),
        sourceLashVetoPixels=int(prepared['lashVeto'].sum()),
        caveat='same-colored ocular texture can remain indistinguishable from a lash; robust smooth-surface hypothesis is not an optical identity proof')
    return prepared
