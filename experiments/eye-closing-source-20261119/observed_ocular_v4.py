"""V4: thin inward ridge tracks distinguish lashes from iris texture.

The geometric pixel-footprint and ridge tests are explicit material-admission
assumptions, not physical alpha truth. No edited output or requested field is
used. V1 remains unchanged and its source assets are retained as a failure.
"""
import numpy as np
from scipy.ndimage import gaussian_filter,label,map_coordinates

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

    # A black-hat basin is not necessarily a thin line. Use positive Hessian
    # curvature across a dark ridge, small curvature along it, and measured
    # brightness on BOTH sides. Its direction must extend into the aperture,
    # rather than run along the lid or around the known iris outer contour.
    luminance=source@np.array([.2126,.7152,.0722]);sigma=.8
    hxx=gaussian_filter(luminance,sigma,order=(0,2),mode='nearest')
    hyy=gaussian_filter(luminance,sigma,order=(2,0),mode='nearest')
    hxy=gaussian_filter(luminance,sigma,order=(1,1),mode='nearest')
    hessian=np.stack([np.stack([hxx,hxy],axis=-1),np.stack([hxy,hyy],axis=-1)],axis=-2)
    values,vectors=np.linalg.eigh(hessian)
    across=vectors[...,1];along=vectors[...,0]
    yy,xx=np.indices(u.shape);blur=gaussian_filter(luminance,.6,mode='nearest')
    sides=[]
    for sign in [-1.,1.]:
        sides.append(map_coordinates(blur,[yy+sign*1.2*across[...,1],xx+sign*1.2*across[...,0]],order=1,mode='nearest',prefilter=False))
    dark_ridge=np.minimum(*sides)-blur
    along_u=along[...,0]*ux+along[...,1]*uy
    along_v=along[...,0]*vx+along[...,1]*vy
    nearer_upper=(v-top)<=(bottom-v)
    delta_u=.25;ul=np.clip(safe-delta_u,0,width);ur=np.clip(safe+delta_u,0,width)
    tangent=np.where(nearer_upper,(upper(ur)-upper(ul))/(ur-ul+1e-20),
                    (lower(ur)-lower(ul))/(ur-ul+1e-20))
    inward_component=abs(along_v-tangent*along_u)/np.sqrt(1+tangent*tangent)
    du=(u-iris_center[0])/iris_radii[0]**2;dv=(v-iris_center[1])/iris_radii[1]**2
    iris_tangent_alignment=abs(-dv*along_u+du*along_v)/np.maximum(np.hypot(du,dv),1e-20)
    iris_r2=((u-iris_center[0])/iris_radii[0])**2+((v-iris_center[1])/iris_radii[1])**2
    follows_iris_rim=(abs(iris_r2-1)<.25)&(iris_tangent_alignment>np.cos(np.pi/9))
    noise=prepared['report']['noise']
    ridge=(values[...,1]>3*noise)&(abs(values[...,0])<.6*values[...,1])
    ridge&=(dark_ridge>3*noise)&inside&~follows_iris_rim
    components,_=label(ridge,np.ones((3,3),int))
    at_lid=(abs(v-top)<=1.5)|(abs(v-bottom)<=1.5)
    attached_ids=np.unique(components[ridge&at_lid]);attached_ids=attached_ids[attached_ids!=0]
    track_ids=[];track_masks=[]
    for identifier in attached_ids:
        region=components==identifier;positions=np.argwhere(region)
        if len(positions)<3:continue
        points=np.stack([u[region],v[region]],axis=-1)
        # Fit a line consensus, not an all-or-nothing whole component: a
        # junction at the eyelid can attach one off-track iris edge pixel.
        # The same blur-width bound defines inliers; it is never relaxed for
        # a failed sample. Deterministic pair hypotheses use only source data.
        candidates=np.linspace(0,len(points)-1,min(64,len(points))).astype(int)
        best=None;best_key=None
        for offset,i in enumerate(candidates):
            for j in candidates[offset+1:]:
                direction=points[j]-points[i];length=np.linalg.norm(direction)
                if length<3*sigma:continue
                direction/=length;normal=np.array([-direction[1],direction[0]])
                distances=abs((points-points[i])@normal)
                inliers=distances<=1.2*sigma
                if int(inliers.sum())<3:continue
                along_track=(points[inliers]-points[i])@direction
                span=np.ptp(along_track)
                if span<3*sigma:continue
                roots=positions[inliers&at_lid[region]]
                if len(roots)==0:continue
                ry,rx=roots[np.argmin(abs(depth[roots[:,0],roots[:,1]]))]
                slope=tangent[ry,rx]
                inward=abs(direction[1]-slope*direction[0])/np.sqrt(1+slope*slope)
                if inward<=np.sin(np.pi/9):continue
                key=(int(inliers.sum()),float(span),-float(np.sum(distances[inliers]**2)))
                if best_key is None or key>best_key:best_key=key;best=inliers
        if best is None:continue
        accepted=np.zeros(u.shape,bool);accepted[tuple(positions[best].T)]=True
        track_ids.append(identifier);track_masks.append(accepted)
    attached_ids=np.asarray(track_ids,int)
    lash_veto=np.zeros(u.shape,bool)
    for accepted in track_masks:lash_veto|=accepted
    lash_veto&=depth>=-band
    # The pigment proposal now comes from this independently admitted thin
    # source track, not the unrelated darkest iris pixel near a geometric lid.
    pigment=prepared['lashPrior'].copy();tracks=[]
    for identifier,region in zip(attached_ids,track_masks):
        positions=np.argwhere(region)
        y,x=positions[np.argmin(luminance[region])]
        pigment[region]=source[y,x]
        tracks.append(dict(component=int(identifier),pixels=int(region.sum()),
            sourcePigmentPixel=[int(x),int(y)],sourcePigmentRGB=source[y,x].tolist()))
    prepared['lashPrior']=pigment
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
    report.update(method='source-normal-observed-iris-inward-thin-ridge-v4',
        observedIrisPixels=int(iris.sum()),sourceOcularAnchors=int(ocular_anchor.sum()),
        sourceLashVetoPixels=int(lash_veto.sum()),
        lashVetoInsideIrisPixels=int((lash_veto&footprint).sum()),
        sourceOcularRule='fixed 3x3 pixel footprint inside source aperture and inner .8 iris radii; all RGB retained except lid-connected dark ridge veto',
        sourceLashRule='sigma.8 Hessian dark ridge with two-sided source contrast; deterministic source line-consensus spans >=3sigma with <=1.2sigma inlier residual and inward direction; excludes iris-rim tracks; connected to source lid within1.5px',
        sourceLashTracks=tracks,
        sourceExposureRefitToOutput=False,sourceGlintSpecialCase=False,
        caveat='occluded dark iris/lash can remain ambiguous; an iris spoke attached to a lid may be a false veto')
    return prepared
