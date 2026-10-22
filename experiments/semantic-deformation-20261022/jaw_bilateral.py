"""Soft equal-height jaw-motion coupling on the unchanged source polylines.

Equal source height is a geometric correspondence, not an anatomical one.
Every original edge participates in the intersection count, including reverse
edges; ambiguous heights and nonpositive height directions are never paired.
The radius-weighted fractional-motion residual has no division by an
individual radius. Its Hessian is a sum of positive weighted outer products.
"""
import copy

import numpy as np


def _merge_intervals(intervals):
    merged = []
    for lo, hi in intervals:
        if merged and lo == merged[-1][1]:
            merged[-1][1] = float(hi)
        else:
            merged.append([float(lo), float(hi)])
    return merged


class BilateralCoupling:
    def __init__(self, problem, basis, intervals, excluded, domain, arcs, offsets):
        self.problem, self.basis = problem, basis
        self.intervals, self.excluded = intervals, excluded
        self.domain, self.arcs, self.offsets = domain, arcs, offsets
        self.size = sum(len(a['nodes']) for a in problem.arcs)
        previous = self._quadrature(8)
        # Rational radius coefficients are not polynomial-exact quadrature.
        # Verify convergence of the actual Hessian before using it in the QP.
        for order in [16, 32, 64, 128]:
            current = self._quadrature(order)
            difference = float(np.linalg.norm(current['hessian']-previous['hessian']))
            scale = float(np.linalg.norm(current['hessian']))
            relative = difference/max(scale, np.finfo(float).tiny)
            if difference <= 1e-12+1e-10*scale:
                break
            previous = current
        else:
            raise ValueError('bilateral rational quadrature failed Hessian convergence check')
        self.order, self.error, self.relative_error = order, difference, relative
        self.heights, self.source_arclength = current['heights'], current['source_arclength']
        self.weights, self.rows = current['weights'], current['rows']
        self.local_basis, self.radii = current['basis'], current['radii']
        self.hessian = current['hessian']

    def _quadrature(self, order):
        xi, w = np.polynomial.legendre.leggauss(order)
        heights, weights, arclength = [], [], []
        for interval in self.intervals:
            lo, hi = interval['height_interval_px']
            h = (lo+hi)/2.+(hi-lo)/2.*xi
            heights.append(h); weights.append((hi-lo)/2.*w)
            mapped = []
            for side in ['left-jaw', 'right-jaw']:
                s0, s1 = interval['source_arclength_intervals_px'][side]
                mapped.append(s0+(h-lo)/(hi-lo)*(s1-s0))
            arclength.append(np.column_stack(mapped))
        h, weights, s = np.concatenate(heights), np.concatenate(weights), np.vstack(arclength)
        bases, radii = [], []
        for j, arc in enumerate(self.arcs):
            bases.append(self.basis(arc['nodes'], s[:, j]))
            radii.append((self.problem.source(arc, s[:, j])-self.problem.origin)
                         @self.problem.across-self.problem.midline)
        a, b = -radii[0], radii[1]
        if not np.all((a > 0.) & (b > 0.)):
            raise ValueError('bilateral quadrature lost opposite-side source radius support')
        rows = np.zeros((len(h), self.size))
        for arc, local, coefficient in zip(self.arcs, bases, [2.*b/(a+b), 2.*a/(a+b)]):
            start, stop = self.offsets[arc['id']]
            rows[:, start:stop] = coefficient[:, None]*local
        hessian = rows.T@(weights[:, None]*rows)
        if not np.isfinite(hessian).all():
            raise ValueError('bilateral quadrature contains nonfinite coefficients')
        return {'heights': h, 'weights': weights, 'source_arclength': s,
                'basis': bases, 'radii': radii, 'rows': rows, 'hessian': hessian}

    def energy(self, values):
        residual = self.rows@np.asarray(values, float)
        return float(self.weights@(residual*residual))

    def mismatch(self, values):
        values = np.asarray(values, float)
        fraction = []
        for arc, basis, radius in zip(self.arcs, self.local_basis, self.radii):
            start, stop = self.offsets[arc['id']]
            fraction.append(-(basis@values[start:stop])/radius)
        delta = fraction[0]-fraction[1]
        return {'fractional_motion_rms_quadrature_estimate': float(np.sqrt(self.weights@(delta*delta)/self.weights.sum())),
                'fractional_motion_max_abs_at_quadrature': float(np.max(np.abs(delta))),
                'radius_weighted_motion_rms_px_quadrature_estimate': float(np.sqrt(self.energy(values)/self.weights.sum())),
                'radius_weighted_energy_px3_quadrature_estimate': self.energy(values)}

    def describe(self):
        return {'definition': 'integral rho^2 * (q_left - q_right)^2 dh; q=-u/r; rho=2*abs(r_left)*r_right/(abs(r_left)+r_right)',
                'anatomical_correspondence_claimed': False,
                'intersection_rule': 'count every original source edge before requiring exactly one positive-height intersection on each side',
                'source_height_overlap_px': list(self.domain),
                'common_paired_height_intervals_px': _merge_intervals([r['height_interval_px'] for r in self.intervals]),
                'paired_source_intervals': copy.deepcopy(self.intervals),
                'excluded_height_intervals': copy.deepcopy(self.excluded),
                'paired_height_measure_px': float(self.weights.sum()),
                'quadrature': {'method': 'Gauss-Legendre on source/control/radius-zero partitions; rational, not polynomial-exact',
                    'order_per_interval': self.order, 'previous_order': self.order//2,
                    'node_count': len(self.heights), 'hessian_frobenius_difference': self.error,
                    'hessian_relative_frobenius_difference': self.relative_error,
                    'acceptance': 'difference <= 1e-12 + 1e-10 * current Hessian Frobenius norm',
                    'diagnostic_limits': 'fraction RMS is a quadrature estimate; maximum is only at quadrature nodes, not a continuous maximum'},
                'limitations': ['Equal source heights do not identify corresponding anatomy.',
                    'Lower fractional mismatch does not establish better appearance or aligned absolute-displacement peaks.',
                    'Unknown and nonmonotone source tails retain their existing constraints and are not frozen or relabeled.']}


def build_bilateral_coupling(problem, basis):
    """Build only a soft PSD term, without changing geometry or hard gates."""
    by_id = {a['id']: a for a in problem.arcs}
    arcs = [by_id['left-jaw'], by_id['right-jaw']]
    offsets = {}; start = 0
    for arc in problem.arcs:
        offsets[arc['id']] = (start, start+len(arc['nodes']))
        start += len(arc['nodes'])
    data, cuts = [], []
    for arc in arcs:
        source = problem.source(arc, arc['knots'])
        h = (source-problem.origin)@problem.down
        radius = (source-problem.origin)@problem.across-problem.midline
        node_h = (problem.source(arc, arc['nodes'])-problem.origin)@problem.down
        cuts.extend(h); cuts.extend(node_h)
        for j in np.flatnonzero(radius[:-1]*radius[1:] < 0.):
            t = -radius[j]/(radius[j+1]-radius[j])
            cuts.append(h[j]+t*(h[j+1]-h[j]))
        data.append({'height': h, 'radius': radius, 'arc': arc})
    domain = [max(float(d['height'].min()) for d in data),
              min(float(d['height'].max()) for d in data)]
    cuts = np.unique(np.r_[domain, cuts])
    cuts = cuts[(cuts >= domain[0]) & (cuts <= domain[1])]
    accepted, excluded = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        middle = (lo+hi)/2.
        intersections = [np.flatnonzero((np.minimum(d['height'][:-1], d['height'][1:]) < middle)
                                      & (middle < np.maximum(d['height'][:-1], d['height'][1:]))) for d in data]
        reason = None
        if any(len(ids) > 1 for ids in intersections):
            reason = 'multiple_source_intersections'
        elif any(len(ids) == 0 for ids in intersections):
            reason = 'missing_source_intersection'
        else:
            directions = [d['height'][ids[0]+1]-d['height'][ids[0]] for d, ids in zip(data, intersections)]
            if any(dh <= 0. for dh in directions):
                reason = 'nonpositive_source_height_direction'
        mapping = {}
        if reason is None:
            radii = []
            for d, ids in zip(data, intersections):
                j = ids[0]; h0, h1 = d['height'][j:j+2]
                knots = d['arc']['knots']
                mapping[d['arc']['id']] = [float(knots[j]+(h-h0)/(h1-h0)*(knots[j+1]-knots[j])) for h in [lo, hi]]
                radii.append(float(d['radius'][j]+(middle-h0)/(h1-h0)*(d['radius'][j+1]-d['radius'][j])))
            if not radii[0] < 0. < radii[1]:
                reason = 'source_radius_not_on_expected_side'
        if reason is None:
            accepted.append({'height_interval_px': [float(lo), float(hi)], 'source_arclength_intervals_px': mapping})
        elif excluded and excluded[-1]['reason'] == reason and excluded[-1]['height_interval_px'][1] == lo:
            excluded[-1]['height_interval_px'][1] = float(hi)
        else:
            excluded.append({'height_interval_px': [float(lo), float(hi)], 'reason': reason})
    if not accepted:
        raise ValueError('bilateral coupling has no common unique positive-height source support')
    return BilateralCoupling(problem, basis, accepted, excluded, domain, arcs, offsets)
