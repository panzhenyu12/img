"""Freeze one local refinement request from actual rejected pixel cells.

Uses original geometry thresholds and adds one coarse-cell-radius of source
degrees of freedom. It changes neither targets nor masks. No solver runs here.
"""
import argparse
from pathlib import Path

import numpy as np

from reduced_field import ROOT
from adapt import read, save, sha
from evaluate import inverse_points, cell_jacobians


def prepare(case, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = read(case / 'protocol.json'); inp = Path(protocol['input'])
    c = read(inp / 'contract.json'); grid = c['grid']; u = np.load(case / 'source-forward.npy')
    field_file = case / 'inverse-field.npy'
    if field_file.exists():
        field = np.load(field_file)
    else:
        _, _, width, height = c['crop']; yy, xx = np.mgrid[:height, :width]
        q = np.column_stack([xx.ravel(), yy.ravel()]) / 2.
        inverse, _ = inverse_points(u / 2., q)
        field = ((inverse-q)*2.).reshape(height, width, 2).astype('<f4')
    j = cell_jacobians(field.astype(float)); sym = (j+j.swapaxes(-1, -2))*.5
    eigen = np.linalg.eigvalsh(sym); skew = abs(j[..., 1, 0]-j[..., 0, 1])*.5
    bad = np.any((eigen[..., 0] < .52) | (eigen[..., 1] > 1.8)
                 | (skew > .24) | (np.linalg.det(j) <= 0), axis=0)
    yy, xx = np.where(bad)
    footprint = []
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        footprint.append((np.column_stack([xx+dx, yy+dy])+field[yy+dy, xx+dx])/2.)
    points = np.concatenate(footprint) if footprint else np.empty((0, 2))
    radius = protocol['stride_grid_cells']
    selected = np.zeros((grid['height'], grid['width']), bool)
    for x, y in points:
        lo = np.maximum(np.floor([x, y]).astype(int)-radius, 0)
        hi = np.minimum(np.ceil([x, y]).astype(int)+radius, [grid['width']-1, grid['height']-1])
        selected[lo[1]:hi[1]+1, lo[0]:hi[0]+1] = True
    nodes = np.flatnonzero(selected.ravel()); np.save(output / 'extra-nodes.npy', nodes)
    save(output / 'report.json', {'source_case': str(case), 'source_solution_sha256': sha(case/'solution.npy'),
         'source_input': str(inp), 'input_data_sha256': sha(inp/'data.npz'), 'bad_pixel_cells': len(xx),
         'source_footprints': points.tolist(), 'radius_source_grid_cells': radius,
         'extra_nodes': len(nodes), 'single_local_refinement': True,
         'thresholds_and_request_unchanged': True, 'geometry_pass_not_implied': True})
    print({'bad_pixel_cells': len(xx), 'extra_nodes': len(nodes), 'source_footprints': points.tolist()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); prepare(args.case.resolve(), args.output.resolve())
