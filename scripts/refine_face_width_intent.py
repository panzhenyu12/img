"""Offline faceWidth target experiment. Does not warp images or change production.

Coordinates remain absolute source-image coordinates, with no half-pixel shift.
The clean baseline re-evaluates native faceWidth semantics at supplied controls;
it is not a byte-exact replay of target-before-resampling in the Rust pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def smoothstep(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def source_geometry(source, frame):
    p = np.asarray(source, dtype=float)
    origin = np.asarray(frame['origin'], dtype=float)
    across = np.asarray(frame['across'], dtype=float)
    down = np.asarray(frame['down'], dtype=float)
    bounds = np.asarray(frame['bounds'], dtype=float)
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 12
            or origin.shape != (2,) or across.shape != (2,)
            or down.shape != (2,) or bounds.shape != (4,)):
        raise ValueError('Expected closed ordered source contour and 2D face frame')
    if not all(np.isfinite(v).all() for v in [p, origin, across, down, bounds]):
        raise ValueError('Contour and frame must be finite')
    basis = np.stack([across, down])
    if not np.allclose(basis @ basis.T, np.eye(2), atol=1e-5, rtol=0):
        raise ValueError('Face frame must be orthonormal')
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    if min(width, height) < 4:
        raise ValueError('Face frame is degenerate')
    edge = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    if np.min(edge) <= 1e-6:
        raise ValueError('Contour contains duplicated consecutive vertices')
    u, v = (p - origin) @ across, (p - origin) @ down
    y = (v - top) / height
    side = np.sign(u - (left + right) * .5)
    lateral = smoothstep(.08, .28, abs((u - (left + right) * .5) / (width * .5)))
    base = smoothstep(.17, .34, y) * (1 - smoothstep(.88, 1., y)) * lateral
    arc_weight = (edge + np.roll(edge, 1)) * .5
    return p, across, down, y, side, base, arc_weight, width


def turns(points):
    edge = np.roll(points, -1, axis=0) - points
    prior = np.roll(edge, 1, axis=0)
    return np.arctan2(prior[:, 0] * edge[:, 1] - prior[:, 1] * edge[:, 0],
                      np.sum(prior * edge, axis=1))


def turn_change(source, target):
    delta = turns(target) - turns(source)
    return np.degrees(abs((delta + np.pi) % (2 * np.pi) - np.pi))


def jaw_path(active, y):
    """Traverse existing cyclic order; never sort/reconnect contour points."""
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    if len(starts) != 1:
        raise ValueError('Jaw must be one contiguous source-curve branch per side')
    start, count = int(starts[0]), int(active.sum())
    if count < 4:
        raise ValueError('Too few jaw samples for an arclength falloff')
    path = (np.arange(start - 1, start + count + 1) % len(y)).astype(int)
    if y[path[0]] > y[path[-1]]:
        path = path[::-1]
    if np.any(np.diff(y[path]) < -1e-6):
        raise ValueError('Jaw branch reverses vertical order; refinement is ambiguous')
    return path


def refine(source, frame, amount, control='faceWidth'):
    if control != 'faceWidth':
        raise ValueError('This candidate supports faceWidth only')
    if not np.isfinite(amount) or not -100 <= amount <= 100:
        raise ValueError('faceWidth amount must be finite and within [-100, 100]')
    p, across, down, y, side, base, arc_weight, width = source_geometry(source, frame)
    unit = base.copy()
    side_reports = []
    for sign in [-1, 1]:
        active = (side == sign) & (y >= .78) & (y < 1.) & (base > 0)
        path = jaw_path(active, y)
        ids = path[1:-1]
        if (abs(base[path[0]] - 1) > 1e-8 or base[path[-1]] > 1e-8
                or np.any(base[ids] <= 0)):
            raise ValueError('Candidate requires full cheek-to-zero-chin support')
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[path], axis=0), axis=1))]
        length = s[-1]
        wanted_mass = float(base[ids] @ arc_weight[ids])

        def profile(onset):
            t = np.clip((s[1:-1] - onset) / (length - onset), 0, 1)
            return 1 - t ** 3 * (10 + t * (-15 + 6 * t))

        def mass_error(onset):
            return float(profile(onset) @ arc_weight[ids] - wanted_mass)

        lower, upper = -length, .99 * length
        if mass_error(lower) > 0 or mass_error(upper) < 0:
            raise ValueError('Requested jaw mass is outside the supported profile family')
        onset = brentq(mass_error, lower, upper, xtol=1e-12)
        unit[ids] = profile(onset)
        side_reports.append({
            'side': sign, 'path_indices': path.tolist(),
            'onset_fraction': float(onset / length),
            'baseline_arclength_integral': wanted_mass,
            'candidate_arclength_integral': float(unit[ids] @ arc_weight[ids]),
            'integral_residual': mass_error(onset),
        })
    clean = p - (side * width * .065 * base * amount / 100)[:, None] * across
    candidate = p - (side * width * .065 * unit * amount / 100)[:, None] * across
    untouched = (y < .78) | (y >= 1.)
    report = {
        'control': control, 'amount': float(amount), 'sides': side_reports,
        'clean_max_turn_increment_degrees': float(turn_change(p, clean).max()),
        'candidate_max_turn_increment_degrees': float(turn_change(p, candidate).max()),
        'clean_peak_displacement_px': float(np.linalg.norm(clean - p, axis=1).max()),
        'candidate_peak_displacement_px': float(np.linalg.norm(candidate - p, axis=1).max()),
        'core_coordinate_change_px': float(np.linalg.norm(candidate[untouched] - clean[untouched], axis=1).max()),
        'vertical_displacement_max_px': float(abs((candidate - p) @ down).max()),
        'candidate_vs_clean_coordinate_rms_px': float(np.sqrt(np.mean((candidate - clean) ** 2))),
        'coordinate_convention': 'absolute_source_image_no_half_pixel_conversion',
        'limitations': [
            'Offline target-intent candidate only; no transport, image-render or visual-quality pass.',
            'Clean semantics are evaluated at supplied controls, not replayed before native resampling.',
            'Source curve is preserved, including segmentation error; no claim of corrected anatomy.',
            'Requires monotone contiguous jaw branches and full cheek-to-zero-chin support.',
            'The .78 jaw onset region is a candidate design choice, not a universal anatomical boundary.',
        ],
    }
    return {'source': p, 'clean': clean, 'candidate': candidate,
            'baseline_unit': base, 'refined_unit': unit, 'report': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--frame', type=Path, required=True,
                        help='JSON containing origin, across, down, bounds [l,t,r,b]')
    parser.add_argument('--amount', type=float, required=True)
    parser.add_argument('--control', default='faceWidth')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.controls.read_text())
    frame = json.loads(args.frame.read_text())
    r = refine([c['source'] for c in raw], frame, args.amount, args.control)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ['clean', 'candidate']:
        pairs = [{'source': p.tolist(), 'requested': q.tolist()}
                 for p, q in zip(r['source'], r[key])]
        (args.output / f'{key}-controls.json').write_text(json.dumps(pairs, indent=2) + '\n')
    (args.output / 'raw-controls.json').write_text(json.dumps(raw, indent=2) + '\n')
    (args.output / 'unit-profiles.json').write_text(json.dumps({
        'baseline': r['baseline_unit'].tolist(), 'refined': r['refined_unit'].tolist(),
    }, indent=2) + '\n')
    r['report'].update({'raw_controls_path': str(args.controls.resolve()),
                        'frame_path': str(args.frame.resolve())})
    (args.output / 'report.json').write_text(json.dumps(r['report'], indent=2) + '\n')
    print(json.dumps(r['report']))


if __name__ == '__main__':
    main()
