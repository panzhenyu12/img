"""Explicit source-relative face-width intent; no output-image fitting.

Reuses the prior whole-face radius-scale idea with the existing faceWidth
height/lateral envelope. The reference height is the existing cheekbones
profile peak, not an image-specific searched landmark or beauty guarantee.
"""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from refine_face_width_intent import source_geometry, smoothstep
from prepare_bounded_face import retarget_attachment

REFERENCE_HEIGHT=.48


def intersections(poly,height):
    a=np.asarray(poly);b=np.roll(a,-1,axis=0)
    active=((a[:,1]<=height)&(b[:,1]>height))|((b[:,1]<=height)&(a[:,1]>height))
    if np.count_nonzero(active)!=2:raise ValueError('reference cut must have exactly two source intersections')
    x=a[active,0]+(height-a[active,1])*(b[active,0]-a[active,0])/(b[active,1]-a[active,1])
    return np.sort(x)


def derive(source,frame,amount):
    if not np.isfinite(amount) or not -100<=amount<=100:raise ValueError('faceWidth amount outside [-100,100]')
    p,across,down,y,side,base,_,width=source_geometry(source,frame)
    origin=np.asarray(frame['origin']);bounds=np.asarray(frame['bounds']);uv=(p-origin)@np.stack([across,down],1)
    v=bounds[1]+REFERENCE_HEIGHT*(bounds[3]-bounds[1]);radii=intersections(uv,v)
    if radii[0]>=0 or radii[1]<=0:raise ValueError('source eye-line origin must lie between reference face sides')
    lateral=smoothstep(.08,.28,abs((radii-(bounds[0]+bounds[2])*.5)/(width*.5)))
    envelope=smoothstep(.17,.34,REFERENCE_HEIGHT)*(1-smoothstep(.88,1.,REFERENCE_HEIGHT))*lateral
    original_reference=width*.065*amount/100*envelope
    # One source-derived normalization. No search over photos or output shape.
    k=float(original_reference.mean()/np.mean(abs(radii)*envelope))
    clean=p-(side*width*.065*base*amount/100)[:,None]*across
    requested=p-(k*uv[:,0]*base)[:,None]*across
    return {'source':p,'legacy':clean,'requested':requested,'envelope':base,'signed_radius':uv[:,0],
            'frame_y':y,'k':k,'reference':{'height':REFERENCE_HEIGHT,'origin':'existing native cheekbones profile peak (.25,.48,.69)',
            'signed_source_radii':radii.tolist(),'legacy_mean_inward_px':float(original_reference.mean()),
            'new_mean_inward_px':float(np.mean(k*abs(radii)*envelope)),'normalization_search':False}}


def neck_request(source,profile,frame,k):
    p=np.asarray(source,float);envelope=np.asarray(profile,float);a=np.asarray(frame['across']);u=(p-np.asarray(frame['origin']))@a
    if p.shape!=(len(envelope),2) or not np.isfinite(p).all() or not np.isfinite(envelope).all():raise ValueError('invalid source neck evidence')
    return p-(k*u*envelope)[:,None]*a


def attachment_request(evidence,source,requested):
    # The retained evidence uses photo indices. The mature helper accepts
    # native texel-centre coordinates and converts them exactly once.
    material=np.asarray(evidence['source_material']);contact=np.asarray(evidence['source_contact'])
    m,c,delta,record=retarget_attachment({'material':(material+.5).tolist(),'contact':(contact+.5).tolist()},source,requested)
    if not np.array_equal(m,material) or not np.array_equal(c,contact):raise ValueError('source attachment conversion lost precision')
    return delta,record
