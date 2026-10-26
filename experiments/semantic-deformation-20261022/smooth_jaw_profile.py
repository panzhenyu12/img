"""Construct an analytic C2 jaw target from a reviewed source profile.

This module changes only the requested contour target.  It does not solve the
field, infer an anatomical boundary, or approve a photograph.  The candidate
uses a sin^2 bump along each reviewed source arc, which is zero with zero
first derivative at both arc endpoints.  Each side is scaled independently so
its signed polygon-area change matches the corresponding side of the frozen
reference profile.
"""
import copy

import numpy as np

from jaw_target_profile import _validate


def polygon_area(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('a finite polygon with at least three points is required')
    if not np.isfinite(points).all():
        raise ValueError('polygon points must be finite')
    return float(.5*np.sum(points[:, 0]*np.roll(points[:, 1], -1)
                              - points[:, 1]*np.roll(points[:, 0], -1)))


def sin2_bump(arclength, total_length):
    """Return a unit bump with zero value and slope at both endpoints."""
    s = np.asarray(arclength, dtype=float)
    if (not np.isfinite(s).all() or not np.isfinite(total_length)
            or total_length <= 0. or np.any(s < 0.) or np.any(s > total_length)):
        raise ValueError('finite arclength inside a positive source arc is required')
    return np.sin(np.pi*s/total_length)**2


def _targets_for_side(prepared, arc, values):
    p, edge, lengths, cumulative, _, across, _, arcs, _ = prepared
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('finite source-arc values are required')
    row = next((r for r in arcs if r['id'] == arc['id']), None)
    if row is None:
        raise ValueError('arc is not part of the reviewed source geometry')
    distance = ((cumulative[:-1]-arc['start'])*arc['direction']) % cumulative[-1]
    inside = (distance > 0.) & (distance < arc['length'])
    if len(values) < 2:
        raise ValueError('at least two source-arc values are required')
    knots = np.linspace(0., arc['length'], len(values))
    target = p.copy()
    target[inside] += np.interp(distance[inside], knots, values)[:, None]*across
    return target


def build_profile(reference_profile, geometry, amount):
    """Replace the reference displacement with per-side area-matched sin² bumps."""
    prepared, rows = _validate(reference_profile, geometry, amount)
    p, _, _, _, _, _, _, arcs, _ = prepared
    candidate = copy.deepcopy(reference_profile)
    candidate['controller'] = 'analytic-c2-sine2-jaw-v1'
    candidate['solver_status'] = 'AnalyticConstructed'
    candidate['diagnostic_only'] = True
    candidate['executable'] = False
    candidate['accepted_target_constraints'] = False
    report = {
        'method': 'per-side sin(pi*s/L)^2 across displacement',
        'reference_controller': reference_profile.get('controller'),
        'reference_solver_status': reference_profile.get('solver_status'),
        'source_area_px2': polygon_area(p),
        'aesthetic_approval': False,
    }
    side_reports = []
    for arc in arcs:
        row = rows[arc['id']]
        knots = np.asarray(row['arclength_px'], dtype=float)
        old_values = np.asarray(row['across_displacement_px'], dtype=float)
        old_target = _targets_for_side(prepared, arc, old_values)
        old_delta = polygon_area(old_target)-report['source_area_px2']
        unit_values = sin2_bump(knots, arc['length'])
        unit_target = _targets_for_side(prepared, arc, unit_values)
        unit_delta = polygon_area(unit_target)-report['source_area_px2']
        if abs(unit_delta) <= 1e-10:
            raise ValueError('unit sin2 side has zero area response')
        amplitude = old_delta/unit_delta
        values = amplitude*unit_values
        row['across_displacement_px'] = values.tolist()
        side_reports.append({'arc_id': arc['id'], 'arc_length_px': float(arc['length']),
            'reference_side_area_delta_px2': old_delta,
            'unit_side_area_delta_px2': unit_delta,
            'amplitude_px': float(amplitude),
            'endpoint_value_px': [float(values[0]), float(values[-1])],
            'endpoint_slope_limit': 'analytic zero; sampled knots retain the C2 bump',
        })
    target = np.asarray(p, dtype=float).copy()
    for arc, row in zip(arcs, candidate['profiles']):
        # The profile rows are validated as a complete two-arc set; find by id
        # rather than relying on list order.
        if row['arc_id'] != arc['id']:
            row = next(r for r in candidate['profiles'] if r['arc_id'] == arc['id'])
        target = _targets_for_side(prepared, arc, row['across_displacement_px'])
        # Add the side displacement to the common source polygon.
        side_target = target
        distance = ((prepared[3][:-1]-arc['start'])*arc['direction']) % prepared[3][-1]
        inside = (distance > 0.) & (distance < arc['length'])
        if 'combined_target' not in locals():
            combined_target = np.asarray(p, dtype=float).copy()
        combined_target[inside] = side_target[inside]
    report['sides'] = side_reports
    report['target_area_px2'] = polygon_area(combined_target)
    report['target_area_delta_px2'] = report['target_area_px2']-report['source_area_px2']
    report['reference_target_area_delta_px2'] = sum(r['reference_side_area_delta_px2'] for r in side_reports)
    report['area_difference_to_reference_px2'] = (report['target_area_delta_px2']
        - report['reference_target_area_delta_px2'])
    report['area_match_tolerance_px2'] = 1e-7
    report['area_match_passed'] = abs(report['area_difference_to_reference_px2']) <= 1e-7
    report['target_constraints_reverified'] = False
    candidate['report'] = report
    return candidate
