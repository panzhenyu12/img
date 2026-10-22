"""Soft source-boundary quadrature; no semantic inference or output taper.

The requested position is the original source point. Its squared displacement
is penalized with a source-arclength measure and a smooth endpoint confidence
fade. This is an input to a new solve, not a fixed-pixel promise, a replacement
source curve, or a modification of any solved displacement/image.
"""
from numbers import Real

import numpy as np


def sample_boundary_soft_targets(segments, *, uncertainty_px=2., spacing=2., density=.5, fade_length=16.):
    """Return source/target Nx2 arrays, weights N and a source-bound report.

    Each independent source polyline retains its exact original vertices.
    Regular source-arclength samples and the two fade transition positions are
    added only as quadrature locations; unknown gaps are never connected.
    Trapezoidal line weights are multiplied by density and
    smoothstep(s/fade_length) * smoothstep((length-s)/fade_length).

    uncertainty_px accepts a nonnegative scalar or one value per polyline. It
    records the source review's uncertainty; this first experiment samples only
    the centerline, without claiming protection across that uncertainty band.
    """
    if not isinstance(segments, (list, tuple, np.ndarray)) or len(segments) == 0:
        raise ValueError('at least one independent reviewed source polyline required')
    if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
           or not np.isfinite(value) or value <= 0 for value in [spacing, density, fade_length]):
        raise ValueError('positive finite spacing, density and fade length required')
    uncertainty = np.asarray(uncertainty_px, float)
    if (uncertainty.ndim > 1 or (uncertainty.ndim == 1 and uncertainty.shape != (len(segments),))
            or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0)):
        raise ValueError('nonnegative finite source uncertainty scalar or one value per polyline required')
    uncertainties = np.broadcast_to(uncertainty, (len(segments),))
    curves = []
    for segment in segments:
        points = np.asarray(segment, float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError('at least two finite source points per reviewed polyline required')
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError('consecutive source points must have positive finite edge lengths')
        curves.append((points, lengths))
    sources, weights, reports = [], [], []
    row_start = 0
    def smoothstep(value):
        t = np.clip(value, 0., 1.)
        return t*t*(3.-2.*t)
    for segment_index, (points, lengths) in enumerate(curves):
        cumulative = np.r_[0., lengths.cumsum()]
        total = float(cumulative[-1])
        if not np.isfinite(total):
            raise ValueError('source polyline total length must be finite')
        # At least two intervals retain an interior sample even for a source
        # segment shorter than the requested pitch; endpoint-only samples would
        # otherwise silently erase the complete soft constraint.
        intervals = max(2, int(np.ceil(total/spacing)))
        positions = np.unique(np.r_[np.linspace(0., total, intervals+1), cumulative,
                                    np.clip([fade_length, total-fade_length], 0., total)])
        edge_index = np.minimum(np.searchsorted(cumulative, positions, side='right')-1, len(lengths)-1)
        fraction = (positions-cumulative[edge_index])/lengths[edge_index]
        source = points[edge_index]+fraction[:,None]*(points[edge_index+1]-points[edge_index])
        source[np.searchsorted(positions, cumulative)] = points
        line_weight = np.zeros(len(positions))
        steps = np.diff(positions)
        line_weight[:-1] += steps*.5; line_weight[1:] += steps*.5
        confidence = smoothstep(positions/fade_length)*smoothstep((total-positions)/fade_length)
        weight = density*line_weight*confidence
        sources.append(source); weights.append(weight)
        row_end = row_start+len(source)
        reports.append({'source_segment_index': segment_index, 'original_source_points': points.tolist(),
            'original_vertex_arclength_px': cumulative.tolist(), 'length_px': total,
            'uncertainty_px': float(uncertainties[segment_index]), 'row_start': row_start, 'row_end': row_end,
            'sample_arclength_px': positions.tolist(), 'source_edge': edge_index.tolist(),
            'source_edge_fraction': fraction.tolist(), 'sample_count': len(source),
            'actual_spacing_max_px': float(steps.max()), 'unfaded_line_measure_px': float(line_weight.sum()),
            'total_soft_weight': float(weight.sum()), 'endpoint_weight_exact_zero': bool(weight[0] == weight[-1] == 0.)})
        row_start = row_end
    source = np.vstack(sources); weight = np.concatenate(weights)
    return {'source': source, 'target': source.copy(), 'weights': weight, 'report': {
        'operation': 'source_reviewed_boundary_soft_line_measure', 'segments': reports,
        'source_segment_count': len(curves), 'sample_count': len(source),
        'uncertainty_px': float(uncertainty) if uncertainty.ndim == 0 else uncertainty.tolist(),
        'spacing_px': float(spacing), 'density_per_source_px': float(density), 'fade_length_px': float(fade_length),
        'total_soft_weight': float(weight.sum()), 'sample_support': 'reviewed polyline centerline only',
        'quadrature': 'irregular trapezoid over source arclength, original vertices and fade transitions retained',
        'confidence': 'cubic smoothstep(s/fade_length) * smoothstep((length-s)/fade_length)',
        'requested_target': 'source position (soft zero displacement)',
        'objective': 'density * integral_source_curve confidence(s) * squared_displacement(s) ds',
        'uncertainty_band_protection_claimed': False, 'hard_fixed_boundary_claimed': False,
        'original_source_vertices_preserved': True, 'new_detection_points': False,
        'unknown_gaps_connected': False, 'input_unmodified': True, 'post_solve_field_taper': False,
        'output_pixel_replacement': False, 'aesthetic_approval': False}}
