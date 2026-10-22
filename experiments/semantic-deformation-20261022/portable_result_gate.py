"""Portable jaw numerical execution checks in source pixel units.

These bounds describe target execution, not anatomy, identity or beauty. Every
active point (requested motion >0.5px) must have projection gain in [0.90,1.10];
every contour point must be within 0.25px of its requested position. Zero edits
must retain exact source coordinates and image bytes. No mean hides a bad row.
These checks do not validate high-resolution detection or photographic beauty.
"""
import numpy as np


def evaluate_result(source, requested, actual, *, amount, fixed_changed_pixels,
                    whole_image_changed_pixels):
    """Return independent execution evidence, never an aesthetic assessment."""
    p, q, r = (np.asarray(value, float) for value in [source, requested, actual])
    if (p.ndim != 2 or p.shape[1] != 2 or not len(p) or q.shape != p.shape or r.shape != p.shape
            or not all(np.isfinite(value).all() for value in [p, q, r]) or not np.isfinite(amount)):
        raise ValueError('matching finite source/requested/actual xy arrays and amount required')
    for count in [fixed_changed_pixels, whole_image_changed_pixels]:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError('nonnegative integer changed-pixel counts required')
    if fixed_changed_pixels > whole_image_changed_pixels:
        raise ValueError('fixed changed pixels cannot exceed complete changed pixels')
    delta, motion = q-p, r-p
    length = np.linalg.norm(delta, axis=1)
    active = length > .5
    gain = np.sum(delta[active]*motion[active], axis=1)/length[active]**2
    errors = np.linalg.norm(r-q, axis=1)
    bad_gain = (gain < .90)|(gain > 1.10)
    checks = {'fixed_pixels_unchanged': fixed_changed_pixels == 0,
              'contour_error_within_limit': bool(np.all(errors <= .25)),
              'each_active_gain_within_limits': bool(not np.any(bad_gain)),
              'zero_request_identity': bool(amount != 0 or (
                  np.array_equal(q, p) and np.array_equal(r, p) and whole_image_changed_pixels == 0))}
    return {'passed': bool(all(checks.values())), 'checks': checks,
            'failed_checks': [key for key, passed in checks.items() if not passed],
            'limits': {'active_requested_motion_gt_px': .5, 'active_projection_gain_min': .90,
                       'active_projection_gain_max': 1.10, 'all_contour_error_max_px': .25,
                       'zero_coordinates_and_image_bytes': 'exact'},
            'active_count': int(active.sum()),
            'active_gain_min': float(gain.min()) if len(gain) else None,
            'active_gain_max': float(gain.max()) if len(gain) else None,
            'gain_failed_rows': np.flatnonzero(active)[bad_gain].tolist(),
            'contour_error_max_px': float(errors.max()),
            'error_failed_rows': np.flatnonzero(errors > .25).tolist(),
            'aesthetic_approval': False,
            'scope': 'Portable jaw numerical execution in source pixel units; sub-0.5px requests have no relative-gain check. Not high-resolution detection or beauty validation.'}
