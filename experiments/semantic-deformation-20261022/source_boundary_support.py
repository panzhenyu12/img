"""Source-reviewed boundary constraints; no boundary detection or output repair.

Each explicitly reviewed polyline remains separate. The uncertainty band has
round endpoint caps and never connects unknown gaps between reviewed segments.
Only source coordinates contribute to it; edit destinations are not used.
"""
import argparse
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from PIL import Image


def boundary_support(image_shape, segments, radius=2.):
    """Return source pixels within ``radius`` of any reviewed polyline segment.

    image_shape is (height, width); coordinates are pixel indices/centers. The
    support includes Euclidean endpoint caps. Only consecutive points within
    each individual polyline are connected. No skin/neck coverage is inferred.
    """
    if (not isinstance(image_shape, (list, tuple, np.ndarray)) or len(image_shape) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1 for v in image_shape)):
        raise ValueError('positive integer image shape (height, width) required')
    if (not isinstance(radius, Real) or isinstance(radius, (bool, np.bool_))
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError('positive finite uncertainty radius required')
    if not isinstance(segments, (list, tuple, np.ndarray)) or len(segments) == 0:
        raise ValueError('at least one independently reviewed polyline required')
    curves = []
    for segment in segments:
        points = np.asarray(segment, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError('each polyline needs at least two finite xy source points')
        curves.append(points)
    height, width = map(int, image_shape)
    mask = np.zeros((height, width), bool)
    for points in curves:
        for start, end in zip(points[:-1], points[1:]):
            low = np.minimum(start, end)-radius
            high = np.maximum(start, end)+radius
            x0, y0 = max(0, math.floor(low[0])), max(0, math.floor(low[1]))
            x1, y1 = min(width-1, math.ceil(high[0])), min(height-1, math.ceil(high[1]))
            if x1 < x0 or y1 < y0:
                continue
            direction = end-start
            length2 = float(direction@direction)
            if not np.isfinite(length2):
                raise ValueError('source segment length exceeds finite numerical range')
            yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
            vx, vy = xx-start[0], yy-start[1]
            t = np.zeros_like(vx) if length2 == 0 else np.clip((vx*direction[0]+vy*direction[1])/length2, 0., 1.)
            distance = np.hypot(vx-t*direction[0], vy-t*direction[1])
            mask[y0:y1+1, x0:x1+1] |= distance <= radius
    return mask


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def prepare(reference, review_path, output, radius=2.):
    """Freeze source-reviewed additions for a new solve; preserve prior pixels."""
    reference, review_path, output = [Path(p).resolve() for p in [reference, review_path, output]]
    if output.exists():
        raise FileExistsError('refusing to overwrite source boundary protection')
    report_path = reference/'report.json'; fixed_path = reference/'fixed-source.png'
    report, review = _read(report_path), _read(review_path)
    source_path = Path(report['source']).resolve(); source_hash = _sha(source_path)
    review_hashes = review.get('hashes', {})
    def binds_source(hashes):
        return any(Path(path).resolve() == source_path and digest == source_hash
                   for path, digest in hashes.items())
    if (not binds_source(report.get('inputs_sha256', {}))
            or not binds_source(review_hashes)):
        raise ValueError('reference source identity must be bound by both run and source review')
    if review.get('schema') != 'source-visible-neck-border-review-v1':
        raise ValueError('explicit source-visible-neck-border-review-v1 evidence required')
    if not isinstance(review.get('segments'), list) or not review['segments']:
        raise ValueError('source review must retain separate nonempty segments')
    segments = []
    for segment in review['segments']:
        if segment.get('automatic_protection_authorized') is not False:
            raise ValueError('offline source review must not declare automatic protection')
        segments.append([point['source'] for point in segment['points']])
    hashes = dict(review_hashes)
    for path in [source_path, report_path, fixed_path, review_path, Path(__file__).resolve()]:
        hashes[str(path)] = _sha(path)
    if any(_sha(path) != digest for path, digest in review_hashes.items()):
        raise ValueError('frozen source review evidence changed')
    with Image.open(source_path) as source:
        size = source.size
    with Image.open(fixed_path) as image:
        fixed = np.asarray(image).copy()
    if fixed.ndim != 2 or fixed.shape != (size[1], size[0]):
        raise ValueError('reference fixed mask dimensions differ from source photograph')
    if not np.all((fixed == 0) | (fixed == 255)):
        raise ValueError('reference fixed mask must use exact binary 0/255 pixels')
    fixed = fixed > 0
    support = boundary_support(fixed.shape, segments, radius=radius)
    for curve in segments:
        points = np.asarray(curve, float)
        if np.any(points < 0) or np.any(points > [size[0]-1, size[1]-1]):
            raise ValueError('reviewed source points must lie inside the bound source photograph')
    combined = fixed | support
    summary = {
        'source_size': list(size), 'radius_px': float(radius),
        'radius_definition': 'Explicit source-point uncertainty band, not neck-mask dilation.',
        'independent_polyline_count': len(segments),
        'line_segment_count': sum(len(points)-1 for points in segments),
        'boundary_support_pixels': int(np.count_nonzero(support)),
        'original_fixed_pixels': int(np.count_nonzero(fixed)),
        'added_fixed_pixels': int(np.count_nonzero(support & ~fixed)),
        'combined_fixed_pixels': int(np.count_nonzero(combined)),
        'original_fixed_support_preserved': True, 'unknown_gaps_connected': False,
        'automatic_protection_authorized': False, 'source_detection_performed': False,
        'post_solve_masking': False, 'aesthetic_approval': False}
    if any(_sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('frozen source protection inputs changed during preparation')
    output.mkdir(parents=True)
    mask_path = output/'fixed-reference.png'; band_path = output/'source-boundary-band.png'
    Image.fromarray(combined.astype('uint8')*255).save(mask_path)
    Image.fromarray(support.astype('uint8')*255).save(band_path)
    protocol = {
        'operation': 'source_reviewed_neck_boundary_protection',
        'reference_source_sha256': source_hash, 'mask': str(mask_path),
        'mask_sha256': _sha(mask_path), 'inputs_sha256': hashes, 'support': summary,
        'boundary_band': str(band_path), 'boundary_band_sha256': _sha(band_path),
        'scope': 'New source constraints before solving. Independent source polylines plus explicit Euclidean uncertainty band; no unknown-gap extension, output replacement, or solved-field masking.'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if any(_sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('frozen source protection inputs changed while writing protocol')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['reference', 'review', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--radius', type=float, default=2.)
    args = parser.parse_args()
    print(prepare(args.reference, args.review, args.output, radius=args.radius))
