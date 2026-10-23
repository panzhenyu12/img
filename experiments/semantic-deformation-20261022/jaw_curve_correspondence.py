"""Closed-form source-material correspondence on an unchanged target curve.

The baked profile defines q(s)=source(s)+across*u(s). Its complete polyline
contains every profile knot and original source vertex. This adapter traverses
that same curve at tau=(s/Lsource)*Ltarget; it changes material correspondence,
not target shape or the baked amount. It is not an anatomical correspondence
or a spatial-field invertibility guarantee.

Every query is evaluated on the complete target polyline. Joining a finite
set of returned samples need not preserve its corners: the report supplies
each corner's new source-parameter preimage. Source samples and quadrature
weights are deliberately not augmented by this adapter.
"""
import numpy as np

from jaw_target_profile import _validate, apply_profile_samples


def _curves(prepared, rows):
    p, _, _, cumulative, _, across, _, arcs, _ = prepared
    curves={}
    for arc in arcs:
        row=rows[arc['id']];length=arc['length']
        distance=((cumulative[:-1]-arc['start'])*arc['direction'])%cumulative[-1]
        ids=np.flatnonzero((distance>0.)&(distance<length))
        ids=ids[np.argsort(distance[ids])]
        source_s=np.r_[0.,distance[ids],length]
        source_vertices=np.vstack([arc['first'],p[ids],arc['last']])
        values=np.asarray(row['across_displacement_px'],float)
        identity=bool(np.all(values==0.))
        if identity:
            # Source cumulative-coordinate subtraction and freshly summed
            # subedge lengths can differ by ulps. The exact zero request must
            # keep supplied source coordinates bitwise, not reparameterize
            # that numerical discrepancy into artificial material motion.
            target_vertices=source_vertices.copy();target_s=source_s.copy()
            knots=source_s.copy();target_length=length
        else:
            # _validate already limits endpoint representation roundoff. Use
            # the source arc's authoritative endpoints, with exact hard zero
            # displacement there, as in the existing pointwise adapter.
            profile_s=np.clip(np.asarray(row['arclength_px'],float),0.,length)
            knots=np.unique(np.r_[source_s,profile_s])
            source=np.column_stack([np.interp(knots,source_s,source_vertices[:,i])for i in range(2)])
            displacement=np.interp(knots,row['arclength_px'],values)
            displacement[[0,-1]]=0.
            target_vertices=source+displacement[:,None]*across
            target_vertices[0]=arc['first'];target_vertices[-1]=arc['last']
            segment_length=np.linalg.norm(np.diff(target_vertices,axis=0),axis=1)
            if not np.isfinite(segment_length).all() or np.any(segment_length<=0.):
                raise ValueError('each target polyline segment must have positive finite length')
            target_s=np.r_[0.,segment_length.cumsum()]
            if not np.isfinite(target_s).all() or np.any(np.diff(target_s)<=0.):
                raise ValueError('target cumulative arclength must be strictly increasing')
            target_length=float(target_s[-1])
        curves[arc['id']]={'source_length':length,'target_length':target_length,
            'target_s':target_s,'target':target_vertices,'identity':identity,
            'original_parameter_knots':knots}
    return curves


def _evaluate(curve, source_s, original_source):
    if curve['identity']:
        return np.asarray(original_source).copy()
    s=np.clip(np.asarray(source_s,float),0.,curve['source_length'])
    tau=(s/curve['source_length'])*curve['target_length']
    target=np.column_stack([np.interp(tau,curve['target_s'],curve['target'][:,i])for i in range(2)])
    target[s==0.]=curve['target'][0]
    target[s==curve['source_length']]=curve['target'][-1]
    return target


def apply_arclength_samples(samples, profile, geometry):
    """Return the same source/weights/metadata with a new material pairing.

    Uses the original source and sample identity validation. No pointwise
    profile acceptance or beauty status is inferred from curve preservation.
    """
    prepared,rows=_validate(profile,geometry,samples['report']['amount'])
    # Reuse the established source/sample protocol checks and deep copy.
    output=apply_profile_samples(samples,profile,geometry)
    curves=_curves(prepared,rows);source=np.asarray(samples['source'],float)
    target=source.copy()
    for arc_id,curve in curves.items():
        ids=np.array([i for i,c in enumerate(samples['controls'])if c['arc_id']==arc_id],int)
        if not len(ids):continue
        s=np.array([samples['controls'][i]['arc_arclength_px']for i in ids],float)
        target[ids]=_evaluate(curve,s,source[ids])
    output['target']=target
    output['report'].update(new_correspondence=True,new_shape_request=False,
        original_pointwise_request_preserved=False,target_curve_set_preserved=True,
        correspondence_controller='uniform-target-arclength-v1',
        target_formula='Q(tau), tau=(source_arclength/Lsource)*Ltarget; Q traverses complete original profile target polyline',
        target_curve_preservation_scope='continuous query function only; joining finite output samples can cut corners',
        source_material_down_motion_allowed=True,anatomical_correspondence_claimed=False,
        input_profile_solver_status=profile.get('solver_status'),
        input_profile_target_optimization_accepted=profile.get('accepted_target_constraints') is True,
        aesthetic_approval=False,image_approval=False,
        curve_correspondence=[{'arc_id':name,'source_length_px':c['source_length'],
            'target_length_px':c['target_length'],'parameter_speed':c['target_length']/c['source_length'],
            'identity_shortcut':c['identity'],
            'original_target_parameter_knots_px':c['original_parameter_knots'].tolist(),
            'target_polyline_vertices':c['target'].tolist(),
            'target_cumulative_arclength_px':c['target_s'].tolist(),
            'target_corner_source_arclength_px':(c['target_s']/c['target_length']*c['source_length']).tolist()}
            for name,c in curves.items()])
    return output


def arclength_contour_targets(geometry, profile):
    """Evaluate original contour rows; this finite polygon is not the curve."""
    prepared,rows=_validate(profile,geometry,profile['amount'])
    p,_,_,cumulative,_,_,_,arcs,_=prepared
    curves=_curves(prepared,rows);target=p.copy()
    for arc in arcs:
        distance=((cumulative[:-1]-arc['start'])*arc['direction'])%cumulative[-1]
        inside=(distance>0.)&(distance<arc['length'])
        target[inside]=_evaluate(curves[arc['id']],distance[inside],p[inside])
    return target
