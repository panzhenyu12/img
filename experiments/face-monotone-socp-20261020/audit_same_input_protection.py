#!/usr/bin/env python3
"""Audit unchanged controls, rigid source support, and whole-line normal error."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from prepare_bounded_face import swept_attachment_nodes


def sampler(values, origin, spacing):
    def sample(points):
        uv = (np.asarray(points)-origin)/spacing
        return np.stack([map_coordinates(values[:, :, c], [uv[:, 1], uv[:, 0]], order=1, mode='nearest') for c in range(2)], axis=-1)
    return sample


def line_maximum(sample, start, end, origin, spacing):
    """Quadratic on each line/cell interval: endpoints + analytic extremum."""
    start, end, origin, spacing = map(lambda x: np.asarray(x, float), [start, end, origin, spacing])
    direction = end-start
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    low, high = np.minimum(start, end), np.maximum(start, end)
    breaks = [0., 1.]
    for c in range(2):
        if abs(direction[c]) < 1e-14:
            continue
        indices = np.arange(np.ceil((low[c]-origin[c])/spacing[c]), np.floor((high[c]-origin[c])/spacing[c])+1)
        fractions = (origin[c]+indices*spacing[c]-start[c])/direction[c]
        breaks.extend(fractions[(fractions > 0) & (fractions < 1)])
    breaks = np.unique(breaks)
    lo, hi = breaks[:-1], breaks[1:]
    ts = np.stack([lo, (lo+hi)*.5, hi], axis=-1)
    values = (sample(start+ts.reshape(-1, 1)*direction) @ normal).reshape(-1, 3)
    c = values[:, 0]
    a = 2*(values[:, 2]+c-2*values[:, 1])
    b = values[:, 2]-c-a
    extrema = np.divide(-b, 2*a, out=np.full_like(a, -1.), where=np.abs(a)>1e-30)
    valid = (extrema > 0) & (extrema < 1)
    interior = a[valid]*extrema[valid]**2+b[valid]*extrema[valid]+c[valid]
    return float(np.max(np.abs(np.r_[values.reshape(-1), interior])))


def audit(path):
    problem = json.loads((path/'controls.json').read_text())
    field = json.loads((path/'field.json').read_text())
    provenance = json.loads((path/'provenance.json').read_text())
    grid = problem['grid']
    flat = np.array(field['displacement'])
    coarse = sampler(flat.reshape(grid['height'], grid['width'], 2), np.array(grid['origin']), np.array(grid['spacing']))
    meta = json.loads((path/'photo-trace/metadata.json').read_text())
    pixels = np.fromfile(path/'photo-trace/inverse.f32', dtype='<f4').reshape(meta['height'], meta['width'], 2).astype(float)
    pixel_origin = np.array([meta['left'], meta['top']])
    final = sampler(pixels, pixel_origin, np.ones(2))
    groups = []
    for i, group in enumerate(problem['rigid_groups']):
        delta = np.array(field['rigid_translations'][i]['actual'])
        check = {'group': i, 'node_count': len(group['nodes']),
                 'coarse_node_equality_max': float(np.linalg.norm(flat[group['nodes']]-delta, axis=-1).max())}
        evidence = provenance.get('rigid_evidence', [])
        if i < len(evidence) and all(k in evidence[i] for k in ['source_material', 'source_contact']):
            points = np.concatenate([evidence[i]['source_material'], evidence[i]['source_contact']])
            required = swept_attachment_nodes(points, delta, np.array(grid['origin']), np.array(grid['spacing']), grid['width'], grid['height'], False)
            check['actual_geometry_fringe_missing_nodes'] = sorted(required-set(group['nodes']))
            destination = points-delta
            check['source_geometry_coarse_inverse_closure_max'] = float(np.linalg.norm(destination+coarse(destination)-points, axis=-1).max())
            check['source_geometry_final_f32_inverse_closure_max'] = float(np.linalg.norm(destination+final(destination)-points, axis=-1).max())
        groups.append(check)
    lines = []
    for i, line in enumerate(problem.get('background_lines', [])):
        lines.append({'line': i, 'hard': line.get('weight') is None,
                      'coarse_whole_line_normal_max': line_maximum(coarse, line['start'], line['end'], grid['origin'], grid['spacing']),
                      'final_f32_whole_line_normal_max': line_maximum(final, line['start'], line['end'], pixel_origin, [1, 1])})
    return {'fixed_nodes_max': float(np.linalg.norm(flat[problem['fixed_nodes']], axis=-1).max()),
            'rigid_groups': groups, 'background_lines': lines,
            'limitation': 'No visual/aesthetic score. Coarse equality and final reconstruction are reported separately; pixel reconstruction can change exact coarse line constraints.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.baseline/'controls.json').read_bytes() != (args.candidate/'controls.json').read_bytes():
        raise ValueError('inputs differ')
    report = {'baseline': audit(args.baseline), 'candidate': audit(args.candidate)}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
