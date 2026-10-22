"""Inspect frozen local jaw edits, with the original whole-width image as context.

Same signed amount is not equal area or equal local displacement: the old
operation includes a chin envelope. Report actual midpoint motion explicitly.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from diagnose_face_proportions import diagnose
from adapt import read, save, sha
from compare_regional_results import panel, polygon_area
from evaluate import load, forward


def turns(points):
    p = np.asarray(points, float)
    incoming = p-np.roll(p, 1, axis=0)
    outgoing = np.roll(p, -1, axis=0)-p
    return np.arctan2(incoming[:, 0]*outgoing[:, 1]-incoming[:, 1]*outgoing[:, 0],
                      (incoming*outgoing).sum(axis=1))


def source_midpoints(sidecar):
    p = np.asarray(sidecar['source_contour'], float)
    lengths = np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1)
    cumulative = np.r_[0., lengths.cumsum()]
    points, positions = [], []
    for side in ['left', 'right']:
        d = sidecar['sides'][side]; b = d['boundaries']
        start = b['mouth_height']['arclength_px']; end = b['chin_proxy']['arclength_px']
        sign = d['direction_in_source_row_order']
        span = ((end-start)*sign)%cumulative[-1]
        s = (start+sign*span*.5)%cumulative[-1]
        i = int(np.searchsorted(cumulative, s, side='right')-1)
        t = float((s-cumulative[i])/lengths[i])
        points.append(p[i]+t*(p[(i+1)%len(p)]-p[i]))
        positions.append({'side': side, 'edge': i, 't': t, 'span_px': span})
    return np.asarray(points), positions


def compare(sidecar_path, cases, output):
    if output.exists():
        raise FileExistsError('refusing to overwrite jaw comparison')
    sidecar = read(sidecar_path)
    if sidecar.get('provenance', {}).get('name') != 'woman':
        raise ValueError('this comparison currently supports the audited woman source only')
    source = np.asarray(sidecar['source_contour'])
    midpoints, positions = source_midpoints(sidecar)
    arc_rows = sorted({i for side in sidecar['sides'].values()
                       for i in side['regions'][1]['source_rows']})
    outside = np.setdiff1d(np.arange(len(source)), arc_rows)
    hashes = {str(sidecar_path): sha(sidecar_path), str(Path(__file__).resolve()): sha(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    images, labels, records = [], ['Original source'], []
    first_photo = None
    for label, case in cases:
        protocol = read(case/'protocol.json'); inp = Path(protocol['input'])
        contract = read(inp/'contract.json'); sc = read(inp/'source-contract.json')
        decision = read(case/'decision.json')
        if decision.get('final_geometry_pass') is not True or not (case/'photo.png').is_file():
            raise ValueError('a geometry-passing existing photograph is required')
        p = np.asarray([r['source'] for r in sc['targets'][:128]])
        if not np.array_equal(source, p):
            raise ValueError('source128 changed between comparison arms')
        photo = Path(contract['source_photo'])
        if sha(photo) != contract['source_photo_sha256']:
            raise ValueError('source photo changed')
        if first_photo is None:
            first_photo = sha(photo); images.append(Image.open(photo).convert('RGB'))
        elif sha(photo) != first_photo:
            raise ValueError('comparison source photos differ')
        desired = np.asarray([r['destination'] for r in sc['targets'][:128]])
        actual = np.asarray(read(case/'final-contour.json')['actual'])
        mids_actual, _ = forward(load(case), midpoints)
        mids_desired = np.array([(1-r['t'])*desired[r['edge']]+r['t']*desired[(r['edge']+1)%128]
                                for r in positions])
        before_turn, desired_turn, actual_turn = turns(source), turns(desired), turns(actual)
        change = np.arctan2(np.sin(actual_turn-before_turn), np.cos(actual_turn-before_turn))
        requested_change = np.arctan2(np.sin(desired_turn-before_turn), np.cos(desired_turn-before_turn))
        record = diagnose('woman', case)
        record.update(label=label, decision=decision,
            midpoint_motion=[{**r, 'source': pt.tolist(),
                'requested_measurement': 'linear interpolation of adjacent target vertices at the fixed source-arc midpoint',
                'requested_delta': (target-pt).tolist(), 'actual_delta': (reached-pt).tolist(),
                'requested_distance_px': float(np.linalg.norm(target-pt)),
                'actual_distance_px': float(np.linalg.norm(reached-pt))}
                for r, pt, target, reached in zip(positions, midpoints, mids_desired, mids_actual)],
            selected_arc_mean_motion_px=float(np.linalg.norm(actual[arc_rows]-source[arc_rows], axis=1).mean()),
            selected_arc_max_motion_px=float(np.linalg.norm(actual[arc_rows]-source[arc_rows], axis=1).max()),
            outside_arc_requested_motion_max_px=float(np.linalg.norm(desired[outside]-source[outside], axis=1).max()),
            outside_arc_actual_motion_max_px=float(np.linalg.norm(actual[outside]-source[outside], axis=1).max()),
            requested_area_px2=polygon_area(desired), actual_area_px2=polygon_area(actual),
            arc_turn_change_degrees_max=float(np.rad2deg(abs(change[arc_rows])).max()),
            arc_requested_turn_change_degrees_max=float(np.rad2deg(abs(requested_change[arc_rows])).max()),
            arc_turn_sign_changes_gt_one_degree=[i for i in arc_rows if before_turn[i]*actual_turn[i] < 0
                and abs(before_turn[i]) > np.deg2rad(1) and abs(actual_turn[i]) > np.deg2rad(1)],
            curve_measurement_limit='Discrete polygon turning angles are diagnostics, not anatomical curvature or an aesthetic pass.')
        paths = [photo, inp/'data.npz', case/'photo.png', case/'decision.json',
                 case/'source-forward.npy', case/'protocol.json']
        hashes.update(record['input_hashes'])
        hashes.update({str(p): sha(p) for p in paths})
        records.append(record)
        images.append(Image.open(case/'photo.png').convert('RGB')); labels.append(label)
        overlay = images[0].copy(); draw = ImageDraw.Draw(overlay)
        for curve, color in [(source, 'yellow'), (desired, 'magenta'), (actual, 'cyan')]:
            draw.line([tuple(p) for p in np.vstack([curve, curve[0]])], fill=color, width=1)
        for pt in midpoints:
            x,y=pt; draw.ellipse((x-2,y-2,x+2,y+2), fill='white')
        overlay.save(output/f'curve-{len(records)}.png')
    lo = np.maximum(np.floor(source.min(0)-35), [0, 0]).astype(int)
    hi = np.minimum(np.ceil(source.max(0)+45), images[0].size).astype(int)
    panel(images, labels, output/'face.png', (*lo, *hi))
    panel(images, labels, output/'full.png')
    if any(sha(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError('frozen comparison evidence changed during reading')
    report = {'arms': records, 'input_hashes': hashes, 'source_area_px2': polygon_area(source),
              'aesthetic_approval': False, 'new_solve_or_sampling': False,
              'comparison_limit': 'Different edit intents. Same nominal amount does not imply equal area or equal midpoint response.',
              'source_visibility': 'Model-reviewed offline selection; automatic visibility remains unknown.'}
    save(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--case', nargs=2, action='append', required=True, metavar=('LABEL', 'PATH'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    compare(a.sidecar.resolve(), [(label, Path(p).resolve()) for label,p in a.case], a.output.resolve())
