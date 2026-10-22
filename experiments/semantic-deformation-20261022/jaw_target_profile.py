"""Source-bound arclength profiles are new shape requests, not the beta law.

Displacements already contain their signed slider amount. The adapter changes
only requested targets and their report; it never alters source geometry,
quadrature weights or sample metadata, and it does not certify the optimizer's
curvature/area claims or photographic naturalness.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from jaw_collocation import _prepare


TOLERANCE_PX = 1e-8


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry_sha(geometry):
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _validate(profile, geometry, amount):
    if profile.get('schema') != 'jaw-arclength-piecewise-linear-profile-v1':
        raise ValueError('source-bound piecewise-linear jaw profile required')
    if (not isinstance(profile.get('controller'), str) or not profile['controller']
            or not np.isfinite([amount, profile.get('amount', np.nan)]).all()
            or profile['amount'] != amount or abs(amount) > 100.):
        raise ValueError('profile controller and exactly matching signed amount required')
    digest = _geometry_sha(geometry)
    if (profile.get('source_geometry_sha256') != digest
            or _geometry_sha(profile.get('source_geometry')) != digest):
        raise ValueError('profile source geometry identity differs')
    prepared = _prepare(geometry, amount, .13, 1.)
    p, _, _, cumulative, _, across, _, arcs, _ = prepared
    if len(arcs) != 2:
        raise ValueError('exactly two reviewed jaw arcs required')
    axis = np.asarray(profile.get('across_axis'), float)
    if axis.shape != (2,) or not np.array_equal(axis, across):
        raise ValueError('profile across axis differs from source geometry')
    rows = profile.get('profiles')
    if (not isinstance(rows, list) or len(rows) != 2
            or len({r.get('arc_id') for r in rows}) != 2
            or {r.get('arc_id') for r in rows} != {arc['id'] for arc in arcs}):
        raise ValueError('one profile per original jaw arc required')
    by_id = {row['arc_id']: row for row in rows}
    for arc in arcs:
        row = by_id[arc['id']]
        knots = np.asarray(row['arclength_px'], float)
        values = np.asarray(row['across_displacement_px'], float)
        if (knots.ndim != 1 or len(knots) < 2 or values.shape != knots.shape
                or not np.isfinite(knots).all() or not np.isfinite(values).all()
                or np.any(np.diff(knots) <= 0.) or knots[0] != 0.
                or abs(knots[-1]-arc['length']) > TOLERANCE_PX
                or values[0] != 0. or values[-1] != 0.):
            raise ValueError('finite ascending source arclengths and exact zero displacement endpoints required')
        if (not np.isfinite([row['arc_length_px'], row['original_contour_start_arclength_px']]).all()
                or abs(row['arc_length_px']-arc['length']) > TOLERANCE_PX
                or abs(row['original_contour_start_arclength_px']-arc['start']) > TOLERANCE_PX
                or row['direction'] != arc['direction']):
            raise ValueError('profile source arc length, start or direction differs')
        distance = ((cumulative[:-1]-arc['start'])*arc['direction']) % cumulative[-1]
        expected = np.r_[0., np.sort(distance[(distance > 0.) & (distance < arc['length'])]), arc['length']]
        declared = np.asarray(row['original_source_vertex_arclength_px'], float)
        if (declared.shape != expected.shape or not np.isfinite(declared).all()
                or not np.allclose(declared, expected, rtol=0., atol=TOLERANCE_PX)
                or np.any(np.min(abs(knots[:, None]-expected[None, :]), axis=0) > TOLERANCE_PX)):
            raise ValueError('profile knots must contain every original source vertex position')
        if amount == 0. and np.any(values != 0.):
            raise ValueError('zero amount profile must preserve identity')
    return prepared, by_id


def load_profile(path, geometry, amount):
    """Validate geometry, baked amount and frozen source-image/evidence inputs.

    Historical .py hashes are not replay requirements: their immutable snapshot
    remains provenance. Image and JSON inputs from the bound protocol are checked
    as source evidence; callers still bind their current run to that same source.
    """
    profile = json.loads(Path(path).read_text())
    _validate(profile, geometry, amount)
    protocol_path = Path(profile['source_protocol'])
    if _sha(protocol_path) != profile['source_protocol_sha256']:
        raise ValueError('profile source protocol changed')
    protocol = json.loads(protocol_path.read_text())
    if _geometry_sha(protocol.get('source_geometry')) != _geometry_sha(geometry):
        raise ValueError('profile protocol source geometry differs')
    inputs = protocol.get('inputs_sha256', {})
    source_inputs = {p: digest for p, digest in inputs.items()
                     if Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg', '.json']}
    if (not any(Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg'] for p in source_inputs)
            or not any(Path(p).suffix.lower() == '.json' for p in source_inputs)):
        raise ValueError('profile protocol must bind image and source evidence')
    if any(_sha(p) != digest for p, digest in source_inputs.items()):
        raise ValueError('profile source image or evidence changed')
    return profile


def apply_profile_samples(samples, profile, geometry):
    """Replace targets on collocation/probe samples with the new baked request."""
    prepared, by_id = _validate(profile, geometry, samples['report']['amount'])
    p, edge, length, cumulative, _, across, _, arcs, selected = prepared
    source = np.asarray(samples['source'], float)
    if source.ndim != 2 or source.shape[1] != 2 or not np.isfinite(source).all() or len(source) != len(samples['controls']):
        raise ValueError('finite source samples with matching control metadata required')
    target = source.copy()
    for index, control in enumerate(samples['controls']):
        if control['arc_id'] is None:
            row = control['original_index']
            if (type(row) is not int or not 0 <= row < len(p) or selected[row]
                    or not np.array_equal(source[index], p[row])):
                raise ValueError('outside control differs from original source row')
        elif control['arc_id'] not in by_id:
            raise ValueError('sample belongs to an undeclared source arc')
    for arc in arcs:
        ids = np.array([i for i, control in enumerate(samples['controls']) if control['arc_id'] == arc['id']], int)
        if not len(ids):
            continue
        distance = np.array([samples['controls'][i]['arc_arclength_px'] for i in ids], float)
        if (not np.isfinite(distance).all() or np.any(distance < -TOLERANCE_PX)
                or np.any(distance > arc['length']+TOLERANCE_PX)):
            raise ValueError('sample arclength lies outside its reviewed source arc')
        distance = np.clip(distance, 0., arc['length'])
        position = (arc['start']+arc['direction']*distance) % cumulative[-1]
        indices = np.minimum(np.searchsorted(cumulative, position, side='right')-1, len(p)-1)
        fraction = (position-cumulative[indices])/length[indices]
        expected = p[indices]+fraction[:, None]*edge[indices]
        expected[distance == 0.] = arc['first']; expected[distance == arc['length']] = arc['last']
        if not np.allclose(source[ids], expected, rtol=0., atol=TOLERANCE_PX):
            raise ValueError('sample source positions differ from their original arc')
        row = by_id[arc['id']]
        displacement = np.interp(distance, row['arclength_px'], row['across_displacement_px'])
        target[ids] += displacement[:, None]*across
    output = copy.deepcopy(samples)
    output['target'] = target
    output['report'].update(controller=profile['controller'], amount_already_baked=True,
        source_geometry_sha256=profile['source_geometry_sha256'], new_shape_request=True,
        target_formula='original source + across * piecewise-linear baked signed profile(arclength); identity outside arcs',
        original_point_gain_metadata_used_for_target=False, profile_curvature_or_area_claim_verified=False,
        aesthetic_approval=False)
    return output


def profile_contour_targets(geometry, profile):
    """Return new Nx2 targets in the original contour's unchanged row order."""
    prepared, by_id = _validate(profile, geometry, profile['amount'])
    p, _, _, cumulative, _, across, _, arcs, _ = prepared
    target = p.copy()
    for arc in arcs:
        distance = ((cumulative[:-1]-arc['start'])*arc['direction']) % cumulative[-1]
        inside = (distance > 0.) & (distance < arc['length'])
        row = by_id[arc['id']]
        displacement = np.interp(distance[inside], row['arclength_px'], row['across_displacement_px'])
        target[inside] += displacement[:, None]*across
    return target
