"""Whole-face width removes the old upper ramp, retaining its lower envelope.

Source anatomy is measured, not used as an artificial splice in the width
law. This source-only target module does not warp pixels or change locks.
"""
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/face-source-relative-width-20261020'))
from intent import derive
from refine_face_width_intent import smoothstep


def _polygon(points,name):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError(f'finite nonempty {name} polygon is required')
    return points


def derive_whole_face_width(source,frame,source_oval,source_eyes,amount):
    """Use the same source-radius scaling everywhere above the old chin fade.

    The old source-relative law has height envelope
    smoothstep(.17,.34,y) * (1-smoothstep(.88,1,y)). Only its first factor is
    removed. Thus every original request at y>=.34 remains identical, the
    existing .48 cheek normalization is unchanged, and no measured eyelid
    plane creates a join. Original eye/brow material constraints remain the
    responsibility of the unchanged transport input.
    """
    original=derive(source,frame,amount)
    oval=_polygon(source_oval,'source oval')
    if len(source_eyes)!=2: raise ValueError('both observed source eye polygons are required')
    eyes=[_polygon(eye,'source eye') for eye in source_eyes]
    source=np.asarray(source,float); origin=np.asarray(frame['origin'],float)
    across=np.asarray(frame['across'],float); down=np.asarray(frame['down'],float)
    left,top,right,bottom=frame['bounds']; height=bottom-top
    radius=original['signed_radius']; y=original['frame_y']
    lateral=smoothstep(.08,.28,abs((radius-(left+right)*.5)/((right-left)*.5)))
    envelope=(1-smoothstep(.88,1.,y))*lateral
    requested=source-(original['k']*radius*envelope)[:,None]*across
    lower_eye_heights=[float((((eye-origin)@down).max()-top)/height) for eye in eyes]
    return {
        'name':'source-whole-face-width-retained-chin-envelope',
        'source':source,'original':original['requested'],'requested':requested,
        'changed_indices':np.flatnonzero(np.any(requested!=original['requested'],axis=1)).tolist(),
        'source_frame_y':y,'source_signed_radius':radius,'envelope':envelope,
        'k':original['k'],'reference':original['reference'],
        'source_detector_oval_top_height':float((((oval-origin)@down).min()-top)/height),
        'source_contour_top_height':float(y.min()),
        'source_lower_eye_heights':lower_eye_heights,
        'source_lower_eye_mean_height':float(np.mean(lower_eye_heights)),
        'original_unchanged_from_height':.34,
        'anatomy_controls_target_splice':False,
        'automatic_target_semantics_changed':True,
        'limitations':[
            'Explicit whole-face width intent, not a same-request numerical correction.',
            'Original source contour asymmetry, segmentation error and occluded evidence are retained.',
            'Eye geometry is recorded and checked for finite polygon coordinates; source shape protection is not implemented by this function.',
            'An off-axis hairline top can move. Lower chin fade and existing cheek amplitude are preserved.',
            'No solver feasibility, image quality, visibility, identity or beauty acceptance is implied.',
            'Opposite signed values reverse about the same source; they are not sequential inverse edits.'
        ],
    }
