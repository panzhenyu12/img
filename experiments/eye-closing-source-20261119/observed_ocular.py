"""V2: retain source-observed inner iris texture, except connected dark lashes.

The geometric pixel-footprint and ridge tests are explicit material-admission
assumptions, not physical alpha truth. No edited output or requested field is
used. V1 remains unchanged and its source assets are retained as a failure.
"""
import numpy as np
from scipy.ndimage import grey_closing,label

from source_estimator import estimate,read,smoothstep,triangle_weights


def estimate_observed_ocular(source,local_u,local_v,upper,lower,to_pixel,width,
                            iris_center,iris_radii,*,band=3.0):
    source=np.asarray(source,float);u=np.asarray(local_u,float);v=np.asarray(local_v,float)
    iris_center=np.asarray(iris_center,float);iris_radii=np.asarray(iris_radii,float)
    if iris_center.shape!=(2,) or iris_radii.shape!=(2,) or np.any(iris_radii<=0):
        raise ValueError('finite source iris center and positive radii required')
    if not np.isfinite(np.r_[iris_center,iris_radii]).all():
        raise ValueError('finite source iris geometry required')
    prepared=estimate(source,u,v,upper,lower,to_pixel,width,band=band)
    safe=np.clip(u,0,width);top=upper(safe);bottom=lower(safe)
    depth=np.minimum(v-top,bottom-v)
    inside=(u>=0)&(u<=width)

    # The local frame is affine. Test a fixed 3x3 pixel footprint against the
    # original source aperture and conservative inner-iris projection. This
    # admits bright and dark RGB alike; there is no highlight-specific patch.
    uy,ux=np.gradient(u);vy,vx=np.gradient(v)
    footprint=np.ones(u.shape,bool)
    for oy in [-.5,0.,.5]:
        for ox in [-.5,0.,.5]:
            qu=u+ox*ux+oy*uy;qv=v+ox*vx+oy*vy
            domain=(qu>=0)&(qu<=width);safequ=np.clip(qu,0,width)
            inside_aperture=domain&(qv>upper(safequ))&(qv<lower(safequ))
            radius=((qu-iris_center[0])/iris_radii[0])**2+((qv-iris_center[1])/iris_radii[1])**2
            footprint&=inside_aperture&(radius<.8**2)

    # A narrow dark feature connected to either source lid is a competing
    # foreground hypothesis. Broad dark iris/pupil regions are not rejected
    # merely for being black. This is still fallible for touching iris spokes.
    yy,xx=np.mgrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    luminance=source@np.array([.2126,.7152,.0722])
    dark_ridge=grey_closing(luminance,footprint=disk,mode='nearest')-luminance
    ridge=(dark_ridge>3*prepared['report']['noise'])&inside
    components,_=label(ridge,np.ones((3,3),int))
    at_lid=(abs(v-top)<=1.5)|(abs(v-bottom)<=1.5)
    attached_ids=np.unique(components[ridge&at_lid]);attached_ids=attached_ids[attached_ids!=0]
    lash_veto=np.isin(components,attached_ids)&(depth>=-band)
    iris=footprint&~lash_veto
    ocular_anchor=(prepared['core']|iris)&~lash_veto

    # Recover V1's appearance proposal before its deep-core replacement when
    # a connected lash enters that old core. Do not freeze that lash simply
    # because it lies inside the iris projection or far from the source lid.
    raw_ocular=np.median(np.stack([read(source,p) for p in prepared['ocularDonorCoordinates']]),axis=0)
    prior=smoothstep((depth+band)/(2*band))*inside
    weights,fit_error=triangle_weights(source,raw_ocular,prepared['skinPrior'],
        prepared['lashPrior'],prior,prepared['report']['noise'])
    cap=np.ones(u.shape)
    for c in range(3):
        b=raw_ocular[...,c];observed=source[...,c]
        cap=np.minimum(cap,np.divide(observed,b,out=np.ones_like(b),where=b>0))
        cap=np.minimum(cap,np.divide(255-observed,255-b,out=np.ones_like(b),where=b<255))
    prepared['exposure'][lash_veto]=np.minimum(weights[...,0],cap)[lash_veto]
    prepared['ocular'][lash_veto]=raw_ocular[lash_veto]
    prepared['exposure'][ocular_anchor]=1
    prepared['ocular'][ocular_anchor]=source[ocular_anchor]
    prepared['core']=ocular_anchor
    prepared['observedIris']=iris
    prepared['lashVeto']=lash_veto
    prepared['darkRidgeResponse']=dark_ridge
    prepared['fitError']=fit_error
    report=prepared['report']
    report.update(method='source-normal-three-color-plus-observed-inner-iris-v2',
        observedIrisPixels=int(iris.sum()),sourceOcularAnchors=int(ocular_anchor.sum()),
        sourceLashVetoPixels=int(lash_veto.sum()),
        lashVetoInsideIrisPixels=int((lash_veto&footprint).sum()),
        sourceOcularRule='fixed 3x3 pixel footprint inside source aperture and inner .8 iris radii; all RGB retained except lid-connected dark ridge veto',
        sourceLashRule='radius2 source-luma black-hat >3 source noise, 8-connected to source lid within1.5px; geometric/material assumption, not optical proof',
        sourceExposureRefitToOutput=False,sourceGlintSpecialCase=False,
        caveat='occluded dark iris/lash can remain ambiguous; an iris spoke attached to a lid may be a false veto')
    return prepared
