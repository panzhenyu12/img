"""Tight linear chord intervals from independent original scanline x errors.

Each original observation is (x_i +/- epsilon_i, y_i), with y_i fixed. The
three queried points share the SAME original observations through frozen
piecewise-linear interpolation weights. Combine their original-node
coefficients before taking the box support; treating the three interpolated
points as independent can invent uncertainty even on one straight source edge.

No source evidence, optimization gate, request, or target is modified here.
Caller-supplied uncertainty is not a calibrated truth unless independently
established by that caller. These intervals do not cover y errors, correlated
original measurements, or reparameterization with uncertain arclength.
"""
from fractions import Fraction

import numpy as np


def _fraction(value):
    return Fraction.from_float(float(value))


def _float(value, direction=0):
    """Round exact rational bounds outwards, without a sign tolerance."""
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('finite representable chord statistics required') from error
    if not np.isfinite(result):
        raise ValueError('finite representable chord statistics required')
    represented = _fraction(result)
    if (direction < 0 and represented > value) or (direction > 0 and represented < value):
        result = float(np.nextafter(result, -np.inf if direction < 0 else np.inf))
    if not np.isfinite(result):
        raise ValueError('finite representable chord interval required')
    return result


def _validate(source_points, epsilon, calibrated):
    points = np.asarray(source_points, float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all() or np.any(np.diff(points[:, 1]) <= 0.)):
        raise ValueError('finite scanline observations with strictly increasing fixed y required')
    uncertainty = np.asarray(epsilon, float)
    if (uncertainty.ndim > 1 or (uncertainty.ndim == 1 and uncertainty.shape != (len(points),))
            or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0.)):
        raise ValueError('finite nonnegative scalar or per-original-node epsilon required')
    if not isinstance(calibrated, (bool, np.bool_)):
        raise ValueError('explicit boolean calibration declaration required')
    uncertainty = np.broadcast_to(uncertainty, (len(points),)).copy()
    return points, uncertainty


def _bound(points, weights, epsilon, calibrated, sampling):
    exact_points = [list(map(_fraction, p)) for p in points]
    sample_y = [sum((w*exact_points[i][1] for i, w in row.items()), Fraction(0)) for row in weights]
    h0 = sample_y[1]-sample_y[0]; h1 = sample_y[2]-sample_y[1]
    factors = [h1, -(h0+h1), h0]
    coefficients = [Fraction(0)]*len(points)
    for factor, row in zip(factors, weights):
        for index, weight in row.items():
            coefficients[index] += factor*weight
    nominal = sum((c*p[0] for c, p in zip(coefficients, exact_points)), Fraction(0))
    exact_epsilon = list(map(_fraction, epsilon))
    radius = sum((abs(c)*e for c, e in zip(coefficients, exact_epsilon)), Fraction(0))
    lower, upper = nominal-radius, nominal+radius
    gain = sum(map(abs, coefficients), Fraction(0))
    # Sign is decided in exact arithmetic, including zero-touch cases. The
    # displayed f64 interval is rounded outwards; no near-zero gate is added.
    sign = 1 if lower > 0 else (-1 if upper < 0 else 0)
    uniform_flip = abs(nominal)/gain if gain else (Fraction(0) if nominal == 0 else None)
    epsilon_scale = abs(nominal)/radius if radius else (Fraction(0) if nominal == 0 else None)
    coefficient_values = [_float(c) for c in coefficients]
    return {'nominal_N': _float(nominal), 'radius': _float(radius, 1),
        'interval': [_float(lower, -1), _float(upper, 1)], 'reliable_sign': sign,
        'flipping_epsilon': _float(uniform_flip) if uniform_flip is not None else None,
        'current_epsilon_scale_to_touch_zero': _float(epsilon_scale) if epsilon_scale is not None else None,
        'uniform_epsilon_gain': _float(gain, 1),
        'source_coefficients': coefficient_values,
        'active_source_node_indices': [i for i, c in enumerate(coefficients) if c],
        'sample_weights': [[_float(row.get(i, Fraction(0))) for i in range(len(points))] for row in weights],
        'sample_y': [_float(y) for y in sample_y], 'epsilon_per_original_node': epsilon.tolist(),
        'maximizing_x_perturbation': [float(e)*(1 if c > 0 else -1 if c < 0 else 0)
                                    for e, c in zip(epsilon, coefficients)],
        'exact': {'nominal_N': str(nominal), 'radius': str(radius),
                  'interval': [str(lower), str(upper)],
                  'source_coefficients': [str(c) for c in coefficients]},
        'units': {'nominal_N': 'px^2', 'radius': 'px^2', 'epsilon': 'px'},
        'formula': 'N=(x2-x1)*(y1-y0)-(x1-x0)*(y2-y1); radius=sum_i abs(merged_coefficient_i)*epsilon_i',
        'sampling': sampling, 'uncertainty_calibrated': bool(calibrated),
        'uncertainty_status': 'caller declares calibrated' if calibrated else 'caller-supplied uncalibrated sensitivity; not ground truth',
        'flipping_epsilon_definition': 'smallest uniform original-node x half-width whose tight interval touches zero; zero means nominal N already zero',
        'assumptions': 'independent original x intervals, original y and interpolation weights fixed; shared sampled nodes merged exactly',
        'arithmetic': 'exact rational evaluation of supplied binary64 values; displayed interval rounded outwards',
        'solver_gate_changed': False}


def localize_chord_uncertainty(source_points, sample_y, epsilon, *, calibrated=False):
    """Bound any three fixed image-y samples of the original source polyline.

    Queries may be in any order, repeated, or exactly on original observations.
    No query is snapped or extrapolated. For every query, exact barycentric
    weights come from its two adjacent original y coordinates. The source
    measurements and caller's epsilon remain unchanged.
    """
    points, uncertainty = _validate(source_points, epsilon, calibrated)
    samples = np.asarray(sample_y, float)
    if (samples.shape != (3,) or not np.isfinite(samples).all()
            or np.any(samples < points[0, 1]) or np.any(samples > points[-1, 1])):
        raise ValueError('three finite fixed sample y positions inside the original interval required')
    weights = []
    for value in samples:
        edge = min(int(np.searchsorted(points[:, 1], value, side='right'))-1, len(points)-2)
        first, last = map(_fraction, points[edge:edge+2, 1])
        t = (_fraction(value)-first)/(last-first)
        weights.append({edge: 1-t, edge+1: t})
    return _bound(points, weights, uncertainty, calibrated, 'fixed image-y queries; exact adjacent source-node interpolation')


def chord_uncertainty_from_edge_samples(source_points, sample_edges, sample_fractions, epsilon, *, calibrated=False):
    """Bound three already-frozen source samples (edge index, fraction).

    This entry point can reuse nominal arclength sampling, but does NOT redo
    the arclength parameterization under uncertain x. Each fraction is a
    caller-specified constant; its complementary weight is exactly 1-t.
    """
    points, uncertainty = _validate(source_points, epsilon, calibrated)
    edges = np.asarray(sample_edges); fractions = np.asarray(sample_fractions, float)
    if (edges.shape != (3,) or edges.dtype.kind not in 'iu'
            or np.any(edges < 0) or np.any(edges >= len(points)-1)
            or fractions.shape != (3,) or not np.isfinite(fractions).all()
            or np.any(fractions < 0.) or np.any(fractions > 1.)):
        raise ValueError('three original edge indices and fixed fractions in [0,1] required')
    weights = [{int(edge): 1-_fraction(t), int(edge)+1: _fraction(t)} for edge, t in zip(edges, fractions)]
    return _bound(points, weights, uncertainty, calibrated, 'fixed original edge/fraction queries; no uncertain arclength reparameterization')
