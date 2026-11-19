"""One source-only challenge of fixed-distance hard-background seeds.

Demote the four-pixel outside band to unknown, then infer one new optical
matte. Never copy raw alpha into the result or grant any material motion.
The candidate stays isolated until paired-scene, edit and shadow review.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demote_outer_background(trimap):
    tri = np.asarray(trimap)
    if tri.ndim != 2 or not np.isin(tri, [0, 128, 255]).all():
        raise ValueError('requires a tri-state source prompt')
    # Same four source pixels and cross connectivity as the context guard.
    # This is prompt uncertainty, not an assertion that all these pixels move.
    demoted = (tri == 0) & binary_dilation(tri > 0, iterations=4)
    candidate = tri.copy()
    candidate[demoted] = 128
    return candidate, demoted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', required=True,
                        choices=['portrait-614810', 'portrait-762020'])
    args = parser.parse_args()
    base = ROOT/'test/results/face-generalization-20261118'/args.source_case
    proposal = base/'source-preparation/head-proposal'
    original = proposal/'component-1'
    out = base/'optical-boundary-review-20261119'
    # Allow restarting a dependency-load failure, never overwriting inference.
    out.mkdir(exist_ok=True)
    if (out/'alpha-official.npy').exists():
        raise ValueError('this frozen source was already inferred')
    rgb = np.asarray(Image.open(original/'source-crop.png').convert('RGB'))
    tri = np.asarray(Image.open(original/'trimap.png').convert('L'))
    candidate, demoted = demote_outer_background(tri)
    protocol = json.loads((proposal/'protocol.json').read_text())
    report = {
        'status': 'prepared source-only candidate', 'accepted': False,
        'sourceCase': args.source_case, 'sourceCropBox': protocol['components'][0]['cropBox'],
        'sourceSHA256': sha(original/'source-crop.png'),
        'originalTrimapSHA256': sha(original/'trimap.png'),
        'promptPolicy': 'original non-BG support dilated four source pixels with cross connectivity; hard BG in that band becomes unknown; all hard FG and all other labels retained',
        'demotedBackgroundPixels': int(demoted.sum()),
        'rawAlphaReadForPrompt': False, 'editedImageReadForPrompt': False,
        'motionPermissionChanged': False, 'sourcePixelsChanged': False,
        'knownShadowRisk': 'New inferred alpha may include a stationary shadow; nonzero alpha alone does not justify moving it.',
        'qualityClaim': 'none; this is an isolated source hypothesis',
    }
    Image.fromarray(candidate).save(out/'trimap.png')
    Image.fromarray(np.uint8(demoted)*255).save(out/'demoted-background.png')
    report['candidateTrimapSHA256'] = sha(out/'trimap.png')
    (out/'protocol.json').write_text(json.dumps(report, indent=2)+'\n')
    sys.path[:0] = [str(ROOT/'.run/mematte-20261117-deps'),
                   str(ROOT/'.run/mematte-20261020/detectron2-vendor'),
                   str(ROOT/'experiments/body-matting-model-review-20261020')]
    from run_mematte import official_prediction
    weights = ROOT/'.models/mematte-20261020/MEMatte-AIM500-ViTS.pth'
    expected = '538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09'
    if sha(weights) != expected:
        raise ValueError('checkpoint changed')
    raw, official, evidence = official_prediction(rgb, candidate, weights)
    report.update(evidence)
    report['modelSHA256'] = expected
    for name, alpha in [('raw', raw), ('official', official)]:
        np.save(out/f'alpha-{name}.npy', alpha)
        Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(out/f'alpha-{name}.png')
    old = np.load(original/'inference/alpha-official.npy')
    report.update(status='one inference complete; not accepted',
                  inferredCoverageInDemotedBand=float(official[demoted].sum()),
                  inferredCoverageAreaChange=float((official-old).sum()),
                  oldUnknownAlphaMaxChange=float(np.max(abs(official[tri == 128]-old[tri == 128]))),
                  remainingHardBGNonzero=int(np.count_nonzero(official[candidate == 0])),
                  hardFGChanged=int(np.count_nonzero(official[tri == 255] != 1)),
                  probes=[])
    x0, y0, _, _ = report['sourceCropBox']
    points = [(300,571), (302,571), (304,571), (302,580), (318,580)] if args.source_case == 'portrait-614810' else []
    for x, y in points:
        cx, cy = x-x0, y-y0
        report['probes'].append(dict(xy=[x,y], rgb=rgb[cy,cx].tolist(),
            oldTrimap=int(tri[cy,cx]), newTrimap=int(candidate[cy,cx]),
            oldAlpha=float(old[cy,cx]), newAlpha=float(official[cy,cx])))
    report['outputsSHA256'] = {name: sha(out/name) for name in ['alpha-raw.npy','alpha-official.npy']}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
