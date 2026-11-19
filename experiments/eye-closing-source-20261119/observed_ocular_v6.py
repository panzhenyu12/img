"""V5 plus bounded, source-only affine ocular extrapolation at lash roots.

The material classifier, source support and two-sided estimates stay frozen.
Only admitted lash-support pixels without a two-sided ocular estimate use
this candidate. The hypothesis is local ocular RGB smoothness, not a new
claim that hidden pixels or physical alpha are observed.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label

from observed_ocular_v4 import estimate_observed_ocular as estimate_v4
from observed_ocular_v5 import estimate_observed_ocular as estimate_v5


def local_affine_surface(source, observed, targets, *, radius):
    """Weighted affine prediction with no donor outside observed source RGB.

    The fixed compact biweight uses radius 2*band (6 source pixels by default).
    A full-rank 3-term fit requires six samples. A normalized design condition
    limit of 1e3 excludes numerically unsupported extrapolation. Predictions
    outside the physical RGB cube are rejected, never clipped/projected.
    """
    h,w=observed.shape
    predicted=np.zeros_like(source);accepted=np.zeros_like(observed)
    records=[]
    for y,x in np.argwhere(targets):
        x0=max(0,int(np.floor(x-radius)));x1=min(w,int(np.ceil(x+radius))+1)
        y0=max(0,int(np.floor(y-radius)));y1=min(h,int(np.ceil(y+radius))+1)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        dx=(xx-x)/radius;dy=(yy-y)/radius;r2=dx*dx+dy*dy
        available=observed[y0:y1,x0:x1]&(r2<1)
        if int(available.sum())<6:continue
        matrix=np.stack([np.ones(int(available.sum())),dx[available],dy[available]],axis=-1)
        root_weight=1-r2[available]
        weighted=matrix*root_weight[:,None]
        singular=np.linalg.svd(weighted,compute_uv=False)
        if singular[-1]<=0 or singular[0]/singular[-1]>1e3:continue
        coefficients=np.linalg.lstsq(weighted,
            source[y0:y1,x0:x1][available]*root_weight[:,None],rcond=None)[0]
        background=coefficients[0]
        if not np.isfinite(background).all() or np.any((background<0)|(background>255)):continue
        predicted[y,x]=background;accepted[y,x]=True
        records.append(dict(pixel=[int(x),int(y)],donorCount=int(available.sum()),
            designCondition=float(singular[0]/singular[-1]),
            nearestObservedDistance=float(radius*np.sqrt(r2[available].min())),
            farthestObservedDistance=float(radius*np.sqrt(r2[available].max())),
            sourceFitRGBRMSE=float(np.sqrt(np.mean((matrix@coefficients-source[y0:y1,x0:x1][available])**2)))))
    return predicted,accepted,records


def estimate_observed_ocular(source,local_u,local_v,upper,lower,to_pixel,width,
                            iris_center,iris_radii,*,band=3.0):
    args=(source,local_u,local_v,upper,lower,to_pixel,width,iris_center,iris_radii)
    # V4 supplies only the unchanged material admission. V5 supplies the frozen
    # existing result. Neither sees an edited image, field or target strength.
    initial=estimate_v4(*args,band=band)
    prepared=estimate_v5(*args,band=band)
    source=np.asarray(source,float);u=np.asarray(local_u,float);v=np.asarray(local_v,float)
    centre=initial['lashVeto']
    foreground=binary_dilation(centre,structure=np.ones((3,3),bool))
    safe=np.clip(u,0,width);depth=np.minimum(v-upper(safe),lower(safe)-v)
    targets=foreground&(u>=0)&(u<=width)&(depth>=1.)&~prepared['estimatedLashUnderlay']
    observed=initial['core']&~foreground&(depth>=1.)
    ocular,accepted,records=local_affine_surface(source,observed,targets,radius=2*band)
    components,count=label(centre,np.ones((3,3),int))
    pigment=prepared['lashPrior'].copy();luma=source@np.array([.2126,.7152,.0722])
    for identifier in range(1,count+1):
        region=components==identifier
        ys,xs=np.where(region);darkest=np.argmin(luma[region])
        support=binary_dilation(region,structure=np.ones((3,3),bool))
        pigment[accepted&support]=source[ys[darkest],xs[darkest]]
    direction=ocular-pigment
    exposure=np.clip(np.sum((source-pigment)*direction,axis=-1)/
        np.maximum(np.sum(direction*direction,axis=-1),1e-20),0,1)
    cap=np.ones(u.shape)
    for channel in range(3):
        b=ocular[...,channel];observed_rgb=source[...,channel]
        cap=np.minimum(cap,np.divide(observed_rgb,b,out=np.ones_like(b),where=b>0))
        cap=np.minimum(cap,np.divide(255-observed_rgb,255-b,out=np.ones_like(b),where=b<255))
    prepared['exposure'][accepted]=np.minimum(exposure,cap)[accepted]
    prepared['ocular'][accepted]=ocular[accepted]
    prepared['lashPrior'][accepted]=pigment[accepted]
    prepared['core'][accepted]=False
    prepared['observedIris'][accepted]=False
    prepared['lashVeto']|=accepted
    prepared['estimatedLashUnderlay']|=accepted
    prepared['affineOcularUnderlay']=accepted
    prepared['report'].update(method='source-observed-iris-ridge-affine-root-underlay-v6',
        sourceAffineUnderlayPixels=int(accepted.sum()),
        sourceAffineUnderlayUnavailablePixels=int((targets&~accepted).sum()),
        sourceAffineUnderlayRadiusPixels=2*band,
        sourceAffineUnderlayRecords=records,
        sourceAffineUnderlayAssumption='ocular RGB locally affine across missing admitted thin foreground; bounded one-sided extrapolation can erase high-frequency underlay and is not optical truth',
        sourceEstimatedLashUnderlayPixels=int(prepared['estimatedLashUnderlay'].sum()),
        sourceOcularAnchors=int(prepared['core'].sum()),
        observedIrisPixels=int(prepared['observedIris'].sum()),
        sourceLashVetoPixels=int(prepared['lashVeto'].sum()))
    return prepared
