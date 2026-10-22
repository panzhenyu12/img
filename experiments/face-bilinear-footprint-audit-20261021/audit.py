"""Read-only final-photo oracle over every nonzero source-material footprint.

The historical Euclidean <= 1px oracle remains a separate result. Bilinear
source contributors instead have a square, strictly positive tent footprint.
Rebuild it using this candidate's translation, never an earlier destination.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stencils(points, shape):
    h, w = shape[:2]
    p = np.clip(np.asarray(points, float), [0., 0.], [w-1., h-1.])
    xy = np.floor(p).astype(int)
    t = p-xy
    x0, y0 = xy[:, 0], xy[:, 1]
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    u, v = t[:, 0], t[:, 1]
    return [(x0, y0, (1-u)*(1-v)), (x1, y0, u*(1-v)),
            (x0, y1, (1-u)*v), (x1, y1, u*v)]


def contributes(mask, points):
    # Boolean membership avoids an arbitrary optical-alpha threshold. A source
    # material pixel contributes whenever its actual bilinear weight is > 0.
    result = np.zeros(len(points), bool)
    for x, y, weight in stencils(points, mask.shape):
        result |= mask[y, x] & (weight > 0.)
    return result


def legacy_rgb(source, points):
    """Preserve the old separable opaque-RGB oracle, including its order."""
    h, w = source.shape[:2]
    p = np.clip(np.asarray(points, float), [0., 0.], [w-1., h-1.])
    xy = np.floor(p).astype(int); t = p-xy
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.minimum(x+1, w-1), np.minimum(y+1, h-1)
    u, v = t[:, 0, None], t[:, 1, None]
    return ((source[y, x, :3].astype(float)*(1-u)+source[y, x1, :3].astype(float)*u)*(1-v)
            +(source[y1, x, :3].astype(float)*(1-u)+source[y1, x1, :3].astype(float)*u)*v)


def renderer_rgb(source, points):
    """Independent CPU oracle matching the published Rust arithmetic order."""
    premul = np.zeros((len(points), 3)); opacity = np.zeros(len(points))
    for x, y, weight in stencils(points, source.shape):
        rgba = source[y, x].astype(float)
        alpha = rgba[:, 3]/255.*weight
        opacity += alpha
        premul += rgba[:, :3]*alpha[:, None]
    out = np.zeros_like(premul)
    visible = np.floor(opacity*255.+.5) > 0
    out[visible] = premul[visible]/opacity[visible, None]
    return out


def rgb_bytes(values):
    return np.floor(np.clip(values, 0, 255)+.5).astype(int)


def error_summary(actual, expected):
    err = np.abs(actual.astype(int)-expected.astype(int))
    return {'max_byte_error': int(err.max(initial=0)),
            'nonzero_channels': int(np.count_nonzero(err)),
            'changed_pixels': int(np.count_nonzero(np.any(err != 0, axis=1))),
            'strict_pass': not bool(np.any(err))}


def audit(case):
    case = case.resolve(); p = read(case/'provenance.json'); f = read(case/'field.json')
    source_path = Path(p['feature_transport']['source'])
    if sha(source_path) != p['feature_transport']['source_sha256']:
        raise ValueError('source provenance digest mismatch')
    source = np.asarray(ImageOps.exif_transpose(Image.open(source_path)).convert('RGBA'))
    photo = np.asarray(Image.open(case/'photo.png').convert('RGB'))
    if photo.shape[:2] != source.shape[:2]:
        raise ValueError('source/photo shape mismatch')
    h, w = source.shape[:2]
    meta = read(case/'photo-trace/metadata.json')
    l, t, cw, ch = [meta[k] for k in ['left', 'top', 'width', 'height']]
    raw = np.fromfile(case/'photo-trace/inverse.f32', dtype='<f4').reshape(ch, cw, 2)
    color = np.fromfile(case/'photo-trace/color.f32', dtype='<f4')
    frozen = np.asarray(Image.open(case/'photo-trace/frozen.png'))
    if not np.isfinite(raw).all() or not np.all(color == 1.) or np.any(frozen):
        raise ValueError('expected untapered all-one color and no frozen trace')
    field = np.zeros((h, w, 2)); field[t:t+ch, l:l+cw] = raw
    yy, xx = np.mgrid[:h, :w]; allpoints = np.stack([xx.ravel(), yy.ravel()], axis=1)
    allflow = field.reshape(-1, 2); allactual = allpoints+allflow
    materials = [{'features': ['earring'+str(i)],
                  'source_material': e['source_material']+e['source_contact']}
                 for i, e in enumerate(p.get('rigid_evidence', []))]+p['feature_transport']['groups']
    groups = []
    for i, e in enumerate(materials):
        material = np.asarray(e['source_material'], float)
        if not np.array_equal(material, np.round(material)):
            raise ValueError('source-material footprint requires original integer source pixels')
        if np.any(material < 0) or np.any(material >= [w, h]):
            raise ValueError('source material out of image bounds')
        mask = np.zeros((h, w), bool)
        xy = material.astype(int); mask[xy[:, 1], xy[:, 0]] = True
        delta = np.asarray(f['rigid_translations'][i]['actual'], float)
        rounded = delta.astype('f4').astype(float)
        ideal_domain = contributes(mask, allpoints+delta)
        f32_domain = contributes(mask, allpoints+rounded)
        actual_domain = contributes(mask, allactual)
        # Preserve old domain verbatim within its original bounding rectangle.
        dst = material-delta; lo = np.floor(dst.min(0)-2).astype(int); hi = np.ceil(dst.max(0)+2).astype(int)
        in_box = np.all((allpoints >= lo) & (allpoints <= hi), axis=1)
        circle = np.zeros(len(allpoints), bool)
        circle[in_box] = cKDTree(dst).query(allpoints[in_box])[0] <= 1.
        union = ideal_domain | f32_domain | actual_domain | circle
        points = allpoints[union]; flow = allflow[union]
        rendered = photo.reshape(-1, 3)[union].astype(int)
        ideal_legacy_float = legacy_rgb(source, points+delta)
        ideal_renderer_float = renderer_rgb(source, points+delta)
        actual_float = renderer_rgb(source, points+flow)
        actual_bytes = rgb_bytes(actual_float)
        # Rust preserves source bytes without resampling when displacement=0.
        zero = np.all(flow == 0., axis=1)
        actual_bytes[zero] = source[points[zero, 1], points[zero, 0], :3]
        legacy_bytes, ideal_bytes = rgb_bytes(ideal_legacy_float), rgb_bytes(ideal_renderer_float)
        domains = {'historical_circle': circle, 'full_f64_bilinear_footprint': ideal_domain,
                   'nominal_f32_bilinear_footprint': f32_domain,
                   'actual_rendered_source_footprint': actual_domain,
                   'circle_omitted_footprint': ideal_domain & ~circle, 'union': union}
        summaries = {}
        for name, domain in domains.items():
            select = domain[union]
            err = flow[select]-delta
            summaries[name] = {'pixels': int(domain.sum()),
                'strict_f64_legacy_rgb': error_summary(rendered[select], legacy_bytes[select]),
                'strict_f64_renderer_order_rgb': error_summary(rendered[select], ideal_bytes[select]),
                'actual_f32_renderer_order_rgb': error_summary(rendered[select], actual_bytes[select]),
                'translation_map_max_error_px': float(np.linalg.norm(err, axis=1).max(initial=0.)),
                'uniform_nominal_f32_translation': bool(np.all(flow[select] == rounded))}
        failures = []
        wrong = (legacy_bytes != rendered) | (ideal_bytes != rendered) | (actual_bytes != rendered)
        for row, channel in np.argwhere(wrong):
            failures.append({'destination': points[row].tolist(), 'channel': int(channel),
                'legacy_f64_unrounded': float(ideal_legacy_float[row, channel]),
                'renderer_order_f64_unrounded': float(ideal_renderer_float[row, channel]),
                'actual_f32_map_unrounded': float(actual_float[row, channel]),
                'legacy_expected': int(legacy_bytes[row, channel]),
                'renderer_order_expected': int(ideal_bytes[row, channel]),
                'actual_map_expected': int(actual_bytes[row, channel]), 'png': int(rendered[row, channel]),
                'actual_flow': flow[row].tolist(), 'in_full_footprint': bool(ideal_domain[union][row])})
        groups.append({'group': i, 'features': e['features'], 'source_pixels_including_duplicates': len(material),
            'source_unique_pixels': int(mask.sum()), 'actual_inverse_translation': delta.tolist(),
            'nominal_f32_inverse_translation': rounded.tolist(), 'domains': summaries,
            'circle_omitted_xy': allpoints[ideal_domain & ~circle].tolist(),
            'actual_footprint_outside_both_translation_domains_xy': allpoints[actual_domain & ~(ideal_domain | f32_domain)].tolist(),
            'f64_footprint_missing_from_actual_xy': allpoints[ideal_domain & ~actual_domain].tolist(),
            'failures': failures})
    return {'case': str(case), 'inputs_sha256': {str(v): sha(v) for v in
            [case/'provenance.json', case/'field.json', case/'photo.png', case/'photo-trace/inverse.f32', source_path]},
        'groups': groups,
        'all_actual_f32_renderer_order_oracles_exact': all(z['domains']['union']['actual_f32_renderer_order_rgb']['strict_pass'] for z in groups),
        'all_full_footprint_strict_f64_legacy_oracles_exact': all(z['domains']['full_f64_bilinear_footprint']['strict_f64_legacy_rgb']['strict_pass'] for z in groups),
        'limitations': ['Read-only diagnostic; no coefficient, target, source or RGB byte changed.',
            'Historical circular domains retained. Full domains independently re-enumerated at this candidate translation.',
            'Actual source footprint scans the entire rendered image to expose leaked or duplicated material.',
            'Strict f64 oracle failures remain failures even when independent actual-f32 resampling matches PNG.',
            'Pixel conservation does not establish anatomical plausibility or an aesthetic score.']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', required=True, type=Path); ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); report = audit(args.case)
    if args.output.exists():
        raise FileExistsError('retain previous report; choose a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output.resolve()),
        'actual_f32_exact': report['all_actual_f32_renderer_order_oracles_exact'],
        'strict_f64_legacy_exact': report['all_full_footprint_strict_f64_legacy_oracles_exact'],
        'groups': [{'features': z['features'], 'full_pixels': z['domains']['full_f64_bilinear_footprint']['pixels'],
                    'circle_omitted': len(z['circle_omitted_xy']), 'different_channels': len(z['failures'])} for z in report['groups']]}, indent=2))


if __name__ == '__main__':
    main()
