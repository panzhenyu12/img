"""Area-preserving scope transport of an existing baked jaw profile.

This creates a new target curve, not a new correspondence on the old curve.
The original contour rows, frame and chin endpoints stay unchanged. Only each
reviewed arc's beginning may move backwards to include more source material.
No anatomical boundary, optimization feasibility or photo quality is inferred.
"""
import copy
import hashlib
import json

import numpy as np

from jaw_collocation import _prepare
from jaw_target_profile import TOLERANCE_PX, _geometry_sha, _validate


def _source_polyline(prepared, arc):
    points, _, _, cumulative, *_ = prepared
    distances = ((cumulative[:-1]-arc['start'])*arc['direction']) % cumulative[-1]
    rows = np.flatnonzero((distances > 0.) & (distances < arc['length']))
    rows = rows[np.argsort(distances[rows])]
    return (np.r_[0., distances[rows], arc['length']],
            np.vstack([arc['first'], points[rows], arc['last']]))


def _area(source_s, source, profile_s, values, across, direction):
    # On the union, u and source are linear; cross(across, dsource)/ds is
    # constant. Endpoint averaging is therefore an exact integral, including
    # corners inside a profile segment and profile knots inside source edges.
    knots = np.unique(np.r_[source_s, profile_s])
    points = np.column_stack([np.interp(knots, source_s, source[:, i]) for i in range(2)])
    u = np.interp(knots, profile_s, values)
    edge = np.diff(points, axis=0)
    dy = across[0]*edge[:, 1]-across[1]*edge[:, 0]
    pieces = direction*.5*(u[:-1]+u[1:])*dy
    return float(pieces.sum()), float(abs(pieces).sum()), knots, u


def extend_profile_scope(profile, old_geometry, new_geometry):
    """Return a diagnostic new operation on strictly extended source arcs.

    u_new(s)=k*u_old(s*L_old/L_new). Each side's k is determined by its exact
    signed area integral, never guessed from the arclength ratio. Source/image
    protocol paths belong to the old operation and are retained as provenance
    only; a caller must bind a new protocol before exporting this new profile.
    """
    old_prepared, old_rows = _validate(profile, old_geometry, profile['amount'])
    if not np.array_equal(np.asarray(old_geometry['contour']), np.asarray(new_geometry['contour'])):
        raise ValueError('scope transport requires the unchanged source contour and row order')
    if old_geometry['frame'] != new_geometry['frame']:
        raise ValueError('scope transport requires the unchanged source frame')
    new_prepared = _prepare(new_geometry, profile['amount'], .13, 1.)
    old_arcs = {arc['id']: arc for arc in old_prepared[7]}
    new_arcs = {arc['id']: arc for arc in new_prepared[7]}
    if new_arcs.keys() != old_arcs.keys():
        raise ValueError('scope transport requires the same two arc identities')
    across = old_prepared[5]
    perimeter = old_prepared[3][-1]
    profiles, reports = [], []
    for name, old in old_arcs.items():
        new = new_arcs[name]
        if new['direction'] != old['direction'] or not np.array_equal(new['last'], old['last']):
            raise ValueError('scope transport requires unchanged traversal direction and chin endpoint')
        if new['length'] <= old['length']+TOLERANCE_PX:
            raise ValueError('each new source arc must strictly extend the original scope')
        extension = ((old['start']-new['start'])*new['direction']) % perimeter
        if abs(extension-(new['length']-old['length'])) > TOLERANCE_PX:
            raise ValueError('new source arc must contain the complete original arc')
        old_s, old_source = _source_polyline(old_prepared, old)
        new_s, new_source = _source_polyline(new_prepared, new)
        row = old_rows[name]
        # _validate allows only tiny endpoint representation roundoff. The
        # source arc endpoints and the profile's exact zero values are final.
        old_knots = np.clip(np.asarray(row['arclength_px'], float), 0., old['length'])
        values = np.asarray(row['across_displacement_px'], float)
        old_area, old_absolute, _, _ = _area(old_s, old_source, old_knots, values,
                                             across, old['direction'])
        mapped_knots = old_knots*(new['length']/old['length'])
        mapped_knots[[0, -1]] = [0., new['length']]
        unscaled_area, unscaled_absolute, knots, unscaled = _area(
            new_s, new_source, mapped_knots, values, across, new['direction'])
        if np.all(values == 0.):
            scale = 1.
        else:
            eps = 64*np.finfo(float).eps
            if abs(old_area) <= eps*old_absolute or abs(unscaled_area) <= eps*unscaled_absolute:
                raise ValueError('nonidentity profile has degenerate signed-area support')
            scale = old_area/unscaled_area
            if not np.isfinite(scale) or scale <= 0.:
                raise ValueError('scope extension cannot preserve signed area without reversing the edit')
        transported = scale*unscaled
        transported[[0, -1]] = 0.
        new_area, _, _, _ = _area(new_s, new_source, knots, transported, across, new['direction'])
        if not np.isfinite(transported).all() or not np.isclose(new_area, old_area, rtol=1e-12, atol=0.):
            raise ValueError('exact transported side-area verification failed')
        profiles.append({'arc_id': name, 'direction': new['direction'],
            'original_contour_start_arclength_px': new['start'], 'arc_length_px': new['length'],
            'original_source_vertex_arclength_px': new_s.tolist(), 'arclength_px': knots.tolist(),
            'across_displacement_px': transported.tolist()})
        reports.append({'arc_id': name, 'direction': new['direction'],
            'old_length_px': old['length'], 'new_length_px': new['length'],
            'added_source_arclength_px': float(extension), 'scale': float(scale),
            'old_signed_area_px2': old_area, 'new_unscaled_signed_area_px2': unscaled_area,
            'new_signed_area_px2': new_area, 'signed_area_error_px2': new_area-old_area,
            'new_original_vertex_arclength_px': new_s.tolist(),
            'mapped_old_profile_knots_px': mapped_knots.tolist(),
            'exact_integration_knots_px': knots.tolist(), 'identity': bool(np.all(values == 0.))})
    acceptance_keys = ['solver_status', 'accepted_target_constraints', 'all_actual_physical_constraints_pass',
                       'diagnostic_only', 'executable', 'image_approval', 'aesthetic_approval']
    result = {'schema': 'jaw-arclength-piecewise-linear-profile-v1',
        'controller': 'extended-lower-face-scope-area-transport-v1', 'amount': profile['amount'],
        'source_geometry': copy.deepcopy(new_geometry), 'source_geometry_sha256': _geometry_sha(new_geometry),
        'across_axis': across.tolist(), 'profiles': profiles,
        'amount_already_baked': True, 'solver_status': 'NotRun', 'accepted_target_constraints': False,
        'target_constraints_evaluated': False, 'diagnostic_only': True, 'executable': False,
        'image_approval': False, 'aesthetic_approval': False,
        'area_reference': 'per-side exact signed integral of the original baked piecewise-linear profile',
        'required_signed_area_delta_px2': float(sum(r['old_signed_area_px2'] for r in reports)),
        'candidate_signed_area_delta_px2': float(sum(r['new_signed_area_px2'] for r in reports)),
        'input_profile_provenance': {
            'profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'controller': profile['controller'], 'source_geometry_sha256': profile['source_geometry_sha256'],
            'acceptance': {key: copy.deepcopy(profile[key]) for key in acceptance_keys if key in profile},
            'source_protocol': profile.get('source_protocol'),
            'source_protocol_sha256': profile.get('source_protocol_sha256')},
        'transport_report': {'new_operation': True, 'new_shape_request': True,
            'target_curve_set_preserved': False, 'original_contour_unmodified': True,
            'source_frame_unmodified': True, 'chin_endpoints_unmodified': True,
            'anatomical_scope_inferred': False, 'shape_or_optimizer_acceptance_inherited': False,
            'new_source_protocol_binding_required': True,
            'formula': 'u_new(s)=k*u_old(s*L_old/L_new), k=A_old/A_new_unscaled, independently per side',
            'area_convention': 'direction * integral u * cross(across, dsource); original contour oriented area delta',
            'quadrature': 'exact piecewise-linear integral on union of original source vertices and mapped profile knots',
            'arcs': reports, 'aesthetic_approval': False, 'image_approval': False}}
    _validate(result, new_geometry, profile['amount'])
    return result
