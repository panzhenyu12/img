"""Diagnose the finite mouth-height transition at each frozen jaw start.

Uses the original closed source contour on both sides of its existing start,
without assigning anatomical roles or joining separately detected interfaces.
Actual coordinates are obtained by inverting the saved final float32 map.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from interface_bend_oracle import chord_metrics, sample_scalar


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class OriginalContourArc:
    def __init__(self, geometry, arc):
        self.points = np.asarray(geometry['contour'], float)
        self.edges = np.roll(self.points, -1, axis=0)-self.points
        self.lengths = np.linalg.norm(self.edges, axis=1)
        if np.any(self.lengths <= 0.):
            raise ValueError('original closed contour requires positive edge lengths')
        self.knots = np.r_[0., self.lengths.cumsum()]
        start = arc['start']; self.direction = arc['direction']
        if self.direction not in [-1, 1]:
            raise ValueError('original arc direction must be ±1')
        self.start = self.knots[start['edge']]+start['t']*self.lengths[start['edge']]

    def source(self, s):
        t = (self.start+self.direction*np.asarray(s, float)) % self.knots[-1]
        ids = np.minimum(np.searchsorted(self.knots, t, side='right')-1, len(self.points)-1)
        return self.points[ids]+((t-self.knots[ids])/self.lengths[ids])[:, None]*self.edges[ids]

    def tangent(self, s):
        t = (self.start+self.direction*np.asarray(s, float)) % self.knots[-1]
        # At an original vertex select the edge reached as s increases.
        side = 'right' if self.direction > 0 else 'left'
        ids = (np.searchsorted(self.knots, t, side=side)-1) % len(self.points)
        return self.direction*self.edges[ids]/self.lengths[ids, None]

    def partition(self, length, grid):
        positions = ((self.knots[:-1]-self.start)*self.direction) % self.knots[-1]
        cuts = np.r_[0., np.sort(positions[(positions > 0.) & (positions < length)]), length]
        result = list(cuts); origin = np.asarray(grid['origin']); spacing = np.asarray(grid['spacing'])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            a, b = self.source([lo, hi])
            for axis, count in enumerate([grid['width'], grid['height']]):
                if a[axis] == b[axis]:
                    continue
                lines = origin[axis]+spacing[axis]*np.arange(count)
                t = (lines-a[axis])/(b[axis]-a[axis]); t = t[(t > 0.) & (t < 1.)]
                result.extend(lo+(hi-lo)*t)
        return np.unique(result)


def directional_derivative(field, points, tangent, grid):
    """Analytic bilinear derivative, using the forward-in-s side on grid seams."""
    uv = (np.asarray(points)-grid['origin'])/grid['spacing']
    direction = np.asarray(tangent)/grid['spacing']
    height, width = field.shape
    if np.any(uv < 0.) or np.any(uv > [width-1, height-1]):
        raise ValueError('original source outside scalar grid')
    cells = np.floor(uv).astype(int)
    cells -= ((uv == cells) & (direction < 0.)).astype(int)
    cells = np.minimum(np.maximum(cells, 0), [width-2, height-2])
    x, y = cells.T; tx, ty = (uv-cells).T
    dx = (1.-ty)*(field[y, x+1]-field[y, x])+ty*(field[y+1, x+1]-field[y+1, x])
    dy = (1.-tx)*(field[y+1, x]-field[y, x])+tx*(field[y+1, x+1]-field[y, x+1])
    return dx*direction[:, 0]+dy*direction[:, 1]


def onset_energy(arc, field, grid, length=12.):
    """Independent 5-point Gauss evaluation on original edge/grid pieces."""
    cuts = arc.partition(length, grid); lo, hi = cuts[:-1], cuts[1:]
    nodes, weights = np.polynomial.legendre.leggauss(5)
    s = ((lo+hi)[:, None]/2.+(hi-lo)[:, None]*nodes/2.).ravel()
    slope = directional_derivative(field, arc.source(s), arc.tangent(s), grid)
    t = s/length; confidence = 1.-10.*t**3+15.*t**4-6.*t**5
    measure = ((hi-lo)[:, None]*weights/2.).ravel()
    return float(length**2*np.sum(measure*confidence*slope**2))


def unweighted_prefix_energy(arc, field, grid, end, length_scale=12.):
    """Keep the physical length scale fixed when changing the prefix endpoint."""
    if not np.isfinite([end, length_scale]).all() or end <= 0. or length_scale <= 0.:
        raise ValueError('positive finite prefix and length scale required')
    cuts = arc.partition(end, grid); lo, hi = cuts[:-1], cuts[1:]
    nodes, weights = np.polynomial.legendre.leggauss(5)
    s = ((lo+hi)[:, None]/2.+(hi-lo)[:, None]*nodes/2.).ravel()
    slope = directional_derivative(field, arc.source(s), arc.tangent(s), grid)
    measure = ((hi-lo)[:, None]*weights/2.).ravel()
    return float(length_scale**2*np.sum(measure*slope**2))


def run(source_report_path, photo_report_path, output):
    source_report_path = Path(source_report_path).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite proximal jaw evidence')
    reference = read(source_report_path); geometry = reference['source_geometry']; grid = reference['grid']
    origin = np.asarray(grid['origin']); spacing = np.asarray(grid['spacing'])
    frame = np.asarray([geometry['frame'][key] for key in ['across', 'down']])
    state_path = source_report_path.parent/'diagnostic-source-state.npz'
    code_paths = [Path(__file__).resolve(), Path(__file__).resolve().with_name('interface_bend_oracle.py')]
    bound = {str(source_report_path): sha(source_report_path), str(state_path): reference['diagnostic_source_state_sha256'],
             **{str(path): sha(path) for path in code_paths}}
    if sha(state_path) != bound[str(state_path)]:
        raise ValueError('frozen scalar state changed')
    state = np.load(state_path); mapping = state['mapping']; scalar = np.zeros(len(mapping)); valid = mapping >= 0
    scalar[valid] = state['scalar_solution'][mapping[valid]]; scalar = scalar.reshape(grid['height'], grid['width'])
    field = None
    if photo_report_path is not None:
        photo_report_path = Path(photo_report_path).resolve(); photo = read(photo_report_path)
        if (Path(photo['source_report']).resolve() != source_report_path
                or photo['inputs_sha256'].get(str(source_report_path)) != sha(source_report_path)):
            raise ValueError('actual field does not bind this frozen source solution')
        field_path = photo_report_path.parent/'inverse-field.npy'
        bound.update({str(photo_report_path): sha(photo_report_path), str(field_path): photo['inverse_field_sha256']})
        if sha(field_path) != bound[str(field_path)]:
            raise ValueError('frozen actual inverse field changed')
        field = np.load(field_path)
        if field.dtype != np.dtype('<f4'):
            raise ValueError('actual map must be the saved float32 inverse')
        from forward import inverse_points
    output.mkdir(parents=True)
    result = {'schema': 'joint-jaw-start-independent-v1', 'source_report': str(source_report_path),
        'photo_report': str(photo_report_path) if photo_report_path else None, 'amount': reference['amount'],
        'inputs_sha256': bound, 'source_continuous_accepted': reference['source_continuous_accepted'],
        'requested_jaw_and_full_contour_area_px2': reference['requested_jaw_and_full_contour_area_px2'],
        'jaw_area_px2': reference['jaw_area_px2'], 'full_contour_area_px2': reference['full_contour_area_px2'],
        'source_geometry': 'original closed contour, parameterized by distance from each existing jaw start; upstream samples extend the same contour, no separate curves joined',
        'actual_coordinates': 'direct inverse of saved final float32 inverse map; no interpolation of saved actual samples',
        'limits': 'Finite-radius/sample diagnostics only. No new acceptance thresholds, continuous final-map proof or image approval.',
        'image_approval': False, 'arcs': []}
    for definition in geometry['arcs']:
        arc = OriginalContourArc(geometry, definition)
        def evaluate(s):
            source = arc.source(s); u = sample_scalar(scalar, (source-origin)/spacing)
            values = {'source': source, 'target': source+u[:, None]*frame[0]}; closure = None
            if field is not None:
                actual, closure = inverse_points(field, source-origin); values['actual'] = actual+origin
            return values, closure
        s = np.unique(np.r_[np.arange(-12., 52.001, .125), [-.01, 0., .01, .1, .25, .5, 1., 2., 4., 6., 8., 10., 12., 16., 20., 24., 28., 32., 36., 40.]])
        variants, closure = evaluate(s)
        motion = {key: (value-variants['source'])@frame[0] for key, value in variants.items() if key != 'source'}
        slope = directional_derivative(scalar, variants['source'], arc.tangent(s), grid)
        row = {'id': definition['id'], 'start_source_xy': arc.source([0.])[0].tolist(),
            'original_start_arclength_px': arc.start, 'actual_inverse_closure': closure,
            'source_target_onset_energy_px3': onset_energy(arc, scalar, grid),
            'onset_energy_definition': '12^2 integral_0^12 (1-10t^3+15t^4-6t^5) |du/ds|^2 ds; independent 5-point Gauss per original edge/grid piece',
            'source_target_unweighted_prefix_energies': [
                {'prefix_end_px': end, 'length_scale_px': 12.,
                 'energy_px3': unweighted_prefix_energy(arc, scalar, grid, end, 12.)}
                for end in [2., 6., 12.]],
            'motion_witnesses': [], 'turns': []}
        for t in [-4., -1., -.01, 0., .01, .1, .25, .5, 1., 2., 4., 6., 8., 10., 12., 16., 20., 24., 28., 32., 36., 40.]:
            i = int(np.searchsorted(s, t)); row['motion_witnesses'].append({'s': t,
                'source_xy': variants['source'][i].tolist(), 'target_forward_s_derivative': float(slope[i]),
                **{key+'_u_px': float(value[i]) for key, value in motion.items()}})
        centers = np.arange(-2., 40.001, .125)
        for radius in [2., 6., 12.]:
            queries = (centers[:, None]+[-radius, 0., radius]).ravel(); v, _ = evaluate(queries)
            metrics = {key: chord_metrics(value.reshape(-1, 3, 2), frame) for key, value in v.items()}
            case = {'radius_px': radius, 'witnesses': []}
            for key in metrics:
                if key == 'source':
                    continue
                delta = metrics[key]['turn_degrees']-metrics['source']['turn_degrees']; i = int(np.argmax(abs(delta)))
                case[key+'_peak_change'] = {'s': float(centers[i]), 'source_xy': arc.source([centers[i]])[0].tolist(),
                    'delta_angle_degrees': float(delta[i]), 'source_angle_degrees': float(metrics['source']['turn_degrees'][i]),
                    'changed_angle_degrees': float(metrics[key]['turn_degrees'][i])}
            for t in [0., 2., 4., 6., 8., 10., 12., 16., 20., 24., 28., 32., 36., 40.]:
                i = int(np.searchsorted(centers, t)); case['witnesses'].append({'s': t,
                    'source_xy': arc.source([t])[0].tolist(),
                    **{key+'_angle_degrees': float(value['turn_degrees'][i]) for key, value in metrics.items()}})
            row['turns'].append(case)
        result['arcs'].append(row)
        np.savez_compressed(output/(definition['id']+'.npz'), source_s=s, **variants,
            **{key+'_u': value for key, value in motion.items()}, target_forward_s_derivative=slope)
    result['total_source_target_onset_energy_px3'] = sum(row['source_target_onset_energy_px3'] for row in result['arcs'])
    if any(sha(path) != value for path, value in bound.items()):
        raise ValueError('frozen inputs changed during start audit')
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(output/'report.json'), 'arcs': len(result['arcs']),
                     'onset_energy_px3': result['total_source_target_onset_energy_px3']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--photo-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_report, args.photo_report, args.output)
