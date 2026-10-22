"""Target-only, fixed-plan equal-area plateau preference ablation.

This does not modify the production experimental controller or render photos.
Only its preferred nodal motion is replaced inside this process. The original
continuous beta-cubed area and every original hard gate remain in force.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
from unittest.mock import patch

import numpy as np
import scipy

import jaw_curve_qp as core
from export_jaw_curve_profile import verify_interfaces_and_area
from jaw_target_profile import load_profile


OriginalProblem = core._Problem


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def plateau(t):
    def step(x):
        z = np.clip(x, 0., 1.)
        return z**3*(10.+z*(-15.+6.*z))
    return step(np.asarray(t)/.30)*step((1.-np.asarray(t))/.30)


class PlateauProblem(OriginalProblem):
    def __init__(self, geometry, spacing, window):
        super().__init__(geometry, spacing, window)
        self.original = OriginalProblem(geometry, spacing, window)
        self.scale = self.original.beta_area(100., 1.)/self.plateau_area(100., 1.)
        if not np.isfinite(self.scale) or self.scale <= 0.:
            raise ValueError('plateau normalization must preserve edit direction with one positive scale')

    def raw_preference(self, arc, s, amount, fraction):
        radius = (self.source(arc, s)-self.origin)@self.across-self.midline
        return -amount/100.*fraction*radius*plateau(np.asarray(s)/arc['length'])

    def plateau_area(self, amount, fraction):
        # radius times a quintic ramp is degree six on each original edge
        # split at .3L/.7L; four Gauss nodes integrate that polynomial exactly.
        xi, weights = np.polynomial.legendre.leggauss(4)
        area = 0.
        for arc in self.arcs:
            cuts = np.unique(np.r_[arc['knots'], [.3*arc['length'], .7*arc['length']]])
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                s = (lo+hi)/2.+(hi-lo)/2.*xi
                edge = (self.source(arc, np.array([hi]))-self.source(arc, np.array([lo])))[0]
                dv = self.across[0]*edge[1]-self.across[1]*edge[0]
                area += arc['direction']*dv/2.*float(weights@self.raw_preference(arc, s, amount, fraction))
        return float(area)

    def beta(self, arc, s, amount, fraction):
        return self.scale*self.raw_preference(arc, s, amount, fraction)

    def beta_area(self, amount, fraction):
        # Deliberately use a separate ORIGINAL problem: its beta is never
        # overridden, so the hard equality cannot follow the new preference.
        return self.original.beta_area(amount, fraction)


def motion_statistics(problem, arc, row, regions=None):
    s = np.unique(np.r_[row['arclength_px'], arc['knots']])
    u = np.interp(s, row['arclength_px'], row['across_displacement_px'])
    if regions is not None:
        pieces = []
        for lo, hi in regions:
            query = np.unique(np.r_[lo, s[(s > lo) & (s < hi)], hi])
            pieces.append((query, np.interp(query, s, u)))
        if not pieces:
            return {'arclength_measure_px': 0., 'motion_integral_px2': 0., 'max_abs_motion_px': 0., 'signed_area_px2': 0.}
        mass = area = measure = peak = 0.
        for query, values in pieces:
            measure += query[-1]-query[0]
            mass += float(np.sum(.5*(np.abs(values[:-1])+np.abs(values[1:]))*np.diff(query)))
            h = (problem.source(arc, query)-problem.origin)@problem.down
            area += arc['direction']*float(np.sum(.5*(values[:-1]+values[1:])*np.diff(h)))
            peak = max(peak, float(np.max(np.abs(values))))
        return {'arclength_measure_px': float(measure), 'motion_integral_px2': mass,
                'max_abs_motion_px': peak, 'signed_area_px2': area}
    absolute = np.abs(u); ds = np.diff(s)
    segment_mass = .5*(absolute[:-1]+absolute[1:])*ds
    cumulative = np.r_[0., np.cumsum(segment_mass)]
    quantiles = {}
    for q in [.1, .25, .5, .75, .9]:
        mass = q*cumulative[-1]
        i = min(int(np.searchsorted(cumulative, mass, side='right')-1), len(s)-2)
        remaining = mass-cumulative[i]; slope = (absolute[i+1]-absolute[i])/ds[i]
        root = np.sqrt(max(0., absolute[i]**2+2.*slope*remaining))
        delta = 2.*remaining/(absolute[i]+root) if remaining > 0. else 0.
        quantiles[str(q)] = float((s[i]+delta)/arc['length'])
    points = problem.source(arc, s); heights = (points-problem.origin)@problem.down
    peak = int(np.argmax(absolute))
    def high_width():
        cutoff = .5*absolute[peak]; total = 0.
        for lo, hi, a, b in zip(s[:-1], s[1:], absolute[:-1], absolute[1:]):
            if min(a, b) >= cutoff: total += hi-lo
            elif max(a, b) >= cutoff: total += (hi-lo)*(max(a, b)-cutoff)/abs(b-a)
        return float(total/arc['length'])
    samples = np.arange(.1, 1., .1)
    return {'signed_area_px2': arc['direction']*float(np.sum(.5*(u[:-1]+u[1:])*np.diff(heights))),
        'motion_integral_px2': float(cumulative[-1]),
        'motion_quantile_normalized_arclength': quantiles,
        'motion_q90_minus_q10_normalized_width': quantiles['0.9']-quantiles['0.1'],
        'half_peak_normalized_width': high_width(),
        'peak_across_motion_px': float(u[peak]), 'peak_normalized_arclength': float(s[peak]/arc['length']),
        'peak_source_down_height_px': float(heights[peak]),
        'decile_normalized_arclength': samples.tolist(),
        'decile_across_motion_px': np.interp(samples*arc['length'], s, u).tolist(),
        'proximal_first15percent': motion_statistics(problem, arc, row, [[0., .15*arc['length']]]),
        'source_nonmonotone_tail': motion_statistics(problem, arc, row, arc['nonmonotone']),
        'unknown_centers': motion_statistics(problem, arc, row, row.get('unknown_center_intervals_px', []))}


def run(baselines, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('refusing to overwrite frozen plateau ablation')
    output.mkdir(parents=True); code = output/'executed-code'; code.mkdir()
    records = []
    for name in ['probe_jaw_plateau.py', 'jaw_curve_qp.py', 'jaw_bilateral.py', 'jaw_collocation.py',
                 'jaw_target_profile.py', 'export_jaw_curve_profile.py']:
        source = Path(__file__).resolve().with_name(name); target = code/name
        digest = sha(source); shutil.copy2(source, target)
        if sha(target) != digest: raise ValueError('code changed while taking snapshot')
        records.append({'source': str(source), 'snapshot': str(target), 'sha256': digest})
    save(code/'manifest.json', records)
    report = {'experiment': 'one predeclared plateau[.3,.7] C2 preference; lambda1; no parameter scan',
        'hypothesis': 'At fixed original signed beta-cubed area, a broader preference may reduce mid-arc concentration. This is falsified if solved motion does not broaden.',
        'preference': 'f(t)=smoothstep5(clamp(t/.30))*smoothstep5(clamp((1-t)/.30)); du=-amount/100*.13*r*f; one shared positive scale restores original continuous beta-cubed total signed area',
        'continuity_scope': 'The scalar envelope f is C2; source radius remains the unchanged piecewise-linear geometry, and executable preferred/solved profiles remain piecewise linear.',
        'reference': 'frozen bilateral-weight=1 beta-cubed QP profiles; no baseline re-solve',
        'motion_distribution': 'exact integral of absolute piecewise-linear across motion with respect to original source arclength; quantiles use that mass',
        'proximal_definition': 'first 15% of each reviewed source arc; location near arc start, not an anatomical classification',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'executed_code_manifest': str(code/'manifest.json'), 'executed_code_manifest_sha256': sha(code/'manifest.json'),
        'no_photo_render': True, 'aesthetic_approval': False, 'cases': []}
    for path in baselines:
        path = Path(path).resolve(); baseline = json.loads(path.read_text())
        geometry = baseline['source_geometry']; parameters = baseline['parameters'].copy()
        if parameters['bilateral_weight'] != 1. or parameters['max_radius_fraction'] != .13:
            raise ValueError('the predeclared probe requires lambda1/radius .13')
        load_profile(path, geometry, parameters['amount'])
        case = output/path.parent.name; case.mkdir()
        frozen_baseline = case/'baseline-profile.json'; shutil.copy2(path, frozen_baseline)
        original = OriginalProblem(geometry, parameters['node_spacing_px'], parameters['window_radius_px'])
        preferred = PlateauProblem(geometry, parameters['node_spacing_px'], parameters['window_radius_px'])
        with patch.object(core, '_Problem', PlateauProblem):
            candidate = core.solve_jaw_curve_profile(geometry, **parameters)
        if candidate['source_geometry'] != geometry:
            raise ValueError('source geometry changed')
        if candidate['required_signed_area_delta_px2'] != baseline['required_signed_area_delta_px2']:
            raise ValueError('original continuous beta-cubed hard area changed')
        candidate['controller'] = 'probe-equal-original-beta-area-c2-plateau-bilateral1-v1'
        candidate['report']['bilateral_coupling']['reference_kind'] = 'piecewise-linear scaled C2 plateau control preference; not a lambda-zero solution'
        candidate['report']['area_quadrature'] += '; unchanged original beta-cubed, not plateau preference'
        candidate['source_protocol'] = baseline['source_protocol']
        candidate['source_protocol_sha256'] = baseline['source_protocol_sha256']
        candidate['source_evidence_sha256'] = baseline['source_evidence_sha256']
        candidate['parameters'] = parameters
        candidate['preference_global_scale'] = preferred.scale
        candidate['baseline_profile'] = str(frozen_baseline); candidate['baseline_profile_sha256'] = sha(frozen_baseline)
        candidate['probe_code_manifest'] = str(code/'manifest.json'); candidate['probe_code_manifest_sha256'] = sha(code/'manifest.json')
        metrics = {'baseline_original_path': str(path), 'baseline_frozen_path': str(frozen_baseline),
            'baseline_sha256': sha(path), 'parameters': parameters,
            'source_evidence_sha256': baseline['source_evidence_sha256'],
            'preference_global_scale': preferred.scale,
            'scaled_continuous_preference_area_px2': preferred.scale*preferred.plateau_area(parameters['amount'], .13),
            'original_beta_area_px2': candidate['required_signed_area_delta_px2'],
            'accepted_numerically': candidate['report']['accepted_numerically'], 'sides': []}
        if candidate['report']['accepted_numerically']:
            candidate['verification'] = verify_interfaces_and_area(geometry, candidate)
            for arc in original.arcs:
                before = next(p for p in baseline['profiles'] if p['arc_id'] == arc['id'])
                after = next(p for p in candidate['profiles'] if p['arc_id'] == arc['id'])
                a, b = motion_statistics(original, arc, before), motion_statistics(original, arc, after)
                preference = copy.deepcopy(after)
                preference['arclength_px'] = arc['nodes'].tolist()
                preference['across_displacement_px'] = preferred.beta(arc, arc['nodes'], parameters['amount'], .13).tolist()
                preferred_stats = motion_statistics(original, arc, preference)
                preference['across_displacement_px'] = original.beta(arc, arc['nodes'], parameters['amount'], .13).tolist()
                original_preference = motion_statistics(original, arc, preference)
                metrics['sides'].append({'arc_id': arc['id'], 'baseline': a, 'candidate': b,
                    'baseline_preference_PL': original_preference, 'candidate_preference_PL': preferred_stats,
                    'motion_q10_q90_width_change': b['motion_q90_minus_q10_normalized_width']-a['motion_q90_minus_q10_normalized_width'],
                    'half_peak_width_change': b['half_peak_normalized_width']-a['half_peak_normalized_width'],
                    'signed_area_change_px2': b['signed_area_px2']-a['signed_area_px2'],
                    'peak_down_change_px': b['peak_source_down_height_px']-a['peak_source_down_height_px']})
        save(case/'candidate-profile.json', candidate); save(case/'comparison.json', metrics)
        metrics['candidate_profile_path'] = str(case/'candidate-profile.json')
        metrics['candidate_profile_sha256'] = sha(case/'candidate-profile.json')
        report['cases'].append(metrics)
        print(json.dumps({'case': path.parent.name, 'accepted': metrics['accepted_numerically'],
            'scale': preferred.scale, 'width_changes': [s['motion_q10_q90_width_change'] for s in metrics['sides']]}), flush=True)
    if any(sha(row['source']) != row['sha256'] for row in records):
        raise ValueError('code changed during target-only probe')
    save(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.output)
