"""Source-only lash underlay: exclude the admitted foreground from donors.

V4's thin-ridge admission is unchanged. Its one-pixel source sampling support
is included because a ridge centre is not the entire antialiased foreground.
The ocular proposal below that thin support interpolates two observed sides;
it is an explicitly estimated underlay, never an observed hidden eye claim.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label, map_coordinates

from observed_ocular_v4 import estimate_observed_ocular as estimate_v4
from source_estimator import read


def estimate_observed_ocular(source,local_u,local_v,upper,lower,to_pixel,width,
                            iris_center,iris_radii,*,band=3.0):
    source=np.asarray(source,float);u=np.asarray(local_u,float);v=np.asarray(local_v,float)
    prepared=estimate_v4(source,u,v,upper,lower,to_pixel,width,
                         iris_center,iris_radii,band=band)
    centre=prepared['lashVeto'].copy()
    support=binary_dilation(centre,structure=np.ones((3,3),bool))
    yy,xx=np.indices(u.shape);xy=np.stack([xx,yy],axis=-1).astype(float)
    safe=np.clip(u,0,width)
    depth=np.minimum(v-upper(safe),lower(safe)-v)
    within=(u>=0)&(u<=width)
    components,count=label(centre,np.ones((3,3),int))
    estimated=np.zeros(u.shape,bool);donor_records=[]
    ocular=prepared['ocular'].copy();pigment=prepared['lashPrior'].copy()
    # A source-fixed bounded search in normal direction: no requested field or
    # edited RGB enters the estimate. Both donor footprints must clear the
    # whole admitted lash support and remain in the observed source aperture.
    for identifier in range(1,count+1):
        core=components==identifier
        points=xy[core]
        if len(points)<3:continue
        _,_,vectors=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
        tangent=vectors[0];normal=np.array([-tangent[1],tangent[0]])
        region=binary_dilation(core,structure=np.ones((3,3),bool))&within&(depth>=1.)
        directions=[]
        for sign in [-1.,1.]:
            found=np.zeros(u.shape,bool);distance=np.zeros(u.shape)
            coordinates=np.zeros_like(xy)
            for step in np.arange(.5,6.01,.5):
                p=xy+sign*step*normal
                inside=(p[...,0]>=0)&(p[...,0]<=source.shape[1]-1)&(p[...,1]>=0)&(p[...,1]<=source.shape[0]-1)
                contaminated=map_coordinates(support.astype(float),[p[...,1],p[...,0]],order=1,mode='constant',cval=1.,prefilter=False)>0
                qdepth=map_coordinates(depth,[p[...,1],p[...,0]],order=1,mode='constant',cval=-1.,prefilter=False)
                valid=region&~found&inside&~contaminated&(qdepth>=1.)
                coordinates[valid]=p[valid];distance[valid]=step;found|=valid
            directions.append((found,distance,coordinates))
        both=region&directions[0][0]&directions[1][0]
        if not both.any():continue
        _,d0,p0=directions[0];_,d1,p1=directions[1]
        colors0=read(source,p0);colors1=read(source,p1)
        denominator=np.maximum(d0+d1,1e-20)
        background=(colors0*d1[...,None]+colors1*d0[...,None])/denominator[...,None]
        ocular[both]=background[both]
        luma=source@np.array([.2126,.7152,.0722])
        ys,xs=np.where(core);darkest=int(np.argmin(luma[core]))
        pigment[both]=source[ys[darkest],xs[darkest]]
        estimated|=both
        donor_records.append(dict(component=identifier,estimatedPixels=int(both.sum()),
            normal=normal.tolist(),sourcePigmentPixel=[int(xs[darkest]),int(ys[darkest])],
            maxDonorDistance=float(max(d0[both].max(),d1[both].max()))))

    # Inside a fully observed aperture the admitted thin foreground is a
    # lash/ocular pair, not skin. Fit only their scalar mixing weight. This
    # changes the source decomposition; it does not project the background.
    direction=ocular-pigment
    exposure=np.clip(np.sum((source-pigment)*direction,axis=-1)/
        np.maximum(np.sum(direction*direction,axis=-1),1e-20),0,1)
    cap=np.ones(u.shape)
    for channel in range(3):
        b=ocular[...,channel];observed=source[...,channel]
        cap=np.minimum(cap,np.divide(observed,b,out=np.ones_like(b),where=b>0))
        cap=np.minimum(cap,np.divide(255-observed,255-b,out=np.ones_like(b),where=b<255))
    prepared['exposure'][estimated]=np.minimum(exposure,cap)[estimated]
    prepared['ocular'][estimated]=ocular[estimated]
    prepared['lashPrior'][estimated]=pigment[estimated]
    prepared['core'][estimated]=False
    prepared['observedIris'][estimated]=False
    prepared['lashVeto']|=estimated
    prepared['estimatedLashUnderlay']=estimated
    prepared['report'].update(method='source-observed-iris-ridge-clean-side-underlay-v5',
        sourceEstimatedLashUnderlayPixels=int(estimated.sum()),
        sourceLashUnderlayRule='V4 ridge plus fixed 1px sampling support; bounded normal search excludes all admitted foreground; interpolate two source-aperture donor sides',
        sourceLashUnderlayDonors=donor_records,
        sourceOcularAnchors=int(prepared['core'].sum()),
        observedIrisPixels=int(prepared['observedIris'].sum()),
        sourceLashVetoPixels=int(prepared['lashVeto'].sum()),
        hiddenOcularClaim='estimated only under admitted thin source foreground; not measured optical truth')
    return prepared
