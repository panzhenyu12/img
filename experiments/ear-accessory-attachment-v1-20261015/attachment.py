"""Isolated source-layer attachment transport, never imported by production."""
import numpy as np

def bilinear(values,xy):
    h,w=values.shape[:2];x0=np.floor(xy[...,0]).astype(int);y0=np.floor(xy[...,1]).astype(int);tx=xy[...,0]-x0;ty=xy[...,1]-y0
    if values.ndim==2:values=values[...,None]
    out=np.zeros((*xy.shape[:2],values.shape[2]),float)
    for dx,dy,weight in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]:
        x=x0+dx;y=y0+dy;valid=(x>=0)&(y>=0)&(x<w)&(y<h)
        out[valid]+=values[y[valid],x[valid]]*weight[valid,None]
    return out

def translate_layer(color,alpha,translation):
    h,w=alpha.shape;y,x=np.mgrid[:h,:w];xy=np.stack([x-translation[0],y-translation[1]],axis=-1)
    return bilinear(color*alpha[...,None],xy),bilinear(alpha,xy)[...,0]

def render_attachments(ear_color,ear_alpha,jewelry_color,jewelry_alpha,translation,link_jewelry=True):
    e,ea=translate_layer(ear_color,ear_alpha,translation)
    # An attached rigid component inherits its parent lobe translation.
    j,ja=translate_layer(jewelry_color,jewelry_alpha,translation if link_jewelry else (0,0))
    return j+(1-ja[...,None])*e,ja+(1-ja)*ea,ea,ja

def recompose_motion_delta(source,source_model,target_model):
    """Source-anchored layer recomposition seam."""
    return np.clip(source+(target_model-source_model),0,1)

def attachment_transforms(pivot,hook,desired_delta):
    """Return source-forward ear and hanging-component affine transforms."""
    p=np.asarray(pivot,float);h=np.asarray(hook,float);t=np.asarray(desired_delta,float)
    v=h-p;target=v+t
    if min(np.linalg.norm(v),np.linalg.norm(target))<1e-6:
        raise ValueError('unidentified pivot-to-hook direction')
    angle=np.arctan2(v[0]*target[1]-v[1]*target[0],np.dot(v,target))
    c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]])
    ear=np.column_stack([R,p-R@p])
    achieved=R@h+ear[:,2]
    jewelry=np.column_stack([np.eye(2),achieved-h])
    return ear,jewelry


def affine_layer(color,alpha,matrix):
    """One inverse bilinear sample for each rigid material layer."""
    h,w=alpha.shape;y,x=np.mgrid[:h,:w];points=np.stack([x,y],axis=-1)
    inverse=np.linalg.inv(matrix[:,:2]);xy=(points-matrix[:,2])@inverse.T
    return bilinear(color*alpha[...,None],xy),bilinear(alpha,xy)[...,0]


def partition_joint_alpha(physical, ear_ownership):
    """Separate ear-over-face opacity while retaining total source support.

    Ownership is a semantic hypothesis. It must not be confused with a second
    physical transparency that cuts a gap inside an opaque piece of skin.
    Face opacity behind a fully opaque ear is unobservable and set to zero.
    """
    physical=np.clip(np.asarray(physical,float),0,1)
    ear=physical*np.clip(np.asarray(ear_ownership,float),0,1)
    face=np.divide(physical-ear,1-ear,out=np.zeros_like(physical),where=(1-ear)>1e-12)
    return ear,np.clip(face,0,1)


def observed_lobe_contact(ear, skin, metal, hook, max_distance):
    """Require observed ear tissue near a hanging component before linking it."""
    y,x=np.mgrid[:ear.shape[0],:ear.shape[1]]
    supported=(ear>=.95)&(skin>=.95)&(metal<.05)
    distance=(x-hook[0])**2+(y-hook[1])**2
    cost=np.where(supported,distance,np.inf)
    iy,ix=np.unravel_index(np.argmin(cost),cost.shape)
    if not np.isfinite(cost[iy,ix]) or cost[iy,ix]>max_distance*max_distance:
        raise ValueError('ear material does not reach the observed attachment')
    return np.array([float(ix),float(iy)])
