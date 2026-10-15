"""Isolated occluded-neck hypothesis; never modifies production algorithms.

Infer an upper-neck envelope from observed lateral neck boundaries in the eye
coordinate frame. Reconstruct only revealed neck texels, then recompose the
unchanged transported face. This is a falsifiable task-specific experiment,
not an amodal ground-truth claim or a learned model.
"""
from pathlib import Path
import hashlib
import json
import sys
import argparse

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_user_face_contour import raster_coverage


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    evidence = ROOT / 'test/results/face-material-v1-20261014'
    debug = evidence / 'debug'
    out = options.output.resolve()
    if out.exists():
        raise SystemExit('Use a fresh output directory; frozen evidence must not be overwritten')
    out.mkdir(parents=True)
    graph_path = ROOT / 'test/results/user-face-contour-20261014/1024/face-graph.json'
    graph = json.loads(graph_path.read_text())
    masks = graph['operations'][0]['params']['semanticMasks']
    geometry = json.loads((debug / 'flow-geometry.json').read_text())
    ox, oy = geometry['origin']
    crop = np.asarray(Image.open(debug / 'source-crop.png').convert('RGBA'))
    background = np.asarray(Image.open(debug / 'background-after-lighting.png').convert('RGBA'))
    original = np.asarray(Image.open(ROOT / 'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'))
    rendered = np.asarray(Image.open(evidence / 'output.png').convert('RGBA'))
    known = np.asarray(Image.open(debug / 'known-neck.png')) > 0
    holes = np.asarray(Image.open(debug / 'holes.png')) > 0
    frozen = np.asarray(Image.open(debug / 'frozen.png')) > 0
    alpha = np.asarray(Image.open(debug / 'transported-physical-alpha.png')).astype(float) / 255
    height, width = known.shape
    yy, xx = np.mgrid[:height, :width]
    centers = []
    for key in ['leftEye', 'rightEye']:
        points = np.array([[p['x'] * original.shape[1], p['y'] * original.shape[0]] for p in masks[key]['points']])
        centers.append(points.mean(axis=0))
    across = centers[1] - centers[0]
    across /= np.linalg.norm(across)
    if across[0] < 0:
        across *= -1
    down = np.array([-across[1], across[0]])
    center = np.mean(centers, axis=0)
    dx, dy = xx + ox + .5 - center[0], yy + oy + .5 - center[1]
    u = dx * across[0] + dy * across[1]
    v = dx * down[0] + dy * down[1]
    source_curve = np.array(geometry['source'])
    face_width = np.ptp((source_curve - center) @ across)
    reach = face_width * .20
    envelope = []
    fits = []
    for sign in [-1, 1]:
        samples = []
        side = known & (u * sign > face_width * .08)
        for row in np.unique(np.floor(v[side] / 2).astype(int)):
            section = side & (np.floor(v / 2).astype(int) == row)
            # Only the observed outer skin boundary, not the occluded jaw arc.
            if section.sum() >= 3:
                samples.append([row * 2 + 1, np.quantile(u[section] * sign, .99)])
        samples = np.asarray(samples)
        start = samples[:, 0].min()
        samples = samples[samples[:, 0] <= start + face_width * .16]
        if len(samples) < 8:
            raise RuntimeError('insufficient observed neck boundary; no completion')
        fit = least_squares(lambda coeff: coeff[0] + coeff[1] * (samples[:, 0] - start) - samples[:, 1],
                            [samples[0, 1], .2], loss='soft_l1', f_scale=1.5,
                            bounds=([0, -.5], [face_width, .5]))
        predicted = fit.x[0] + fit.x[1] * (v - start)
        envelope.append((u * sign <= predicted, start))
        fits.append({'side': sign, 'startV': float(start), 'intercept': float(fit.x[0]),
                     'slope': float(fit.x[1]), 'observedPoints': samples.tolist(),
                     'residualRms': float(np.sqrt(np.mean(fit.fun ** 2))), 'maxExtrapolationPixels': float(reach)})
    # The source face occludes this layer; only the newly revealed strip can
    # become visible. Outside the inferred envelope remains the original layer.
    support = envelope[0][0] & envelope[1][0]
    confidence = np.minimum(smoothstep(envelope[0][1] - reach, envelope[0][1] - reach + 4, v),
                            smoothstep(envelope[1][1] - reach, envelope[1][1] - reach + 4, v))
    candidate_domain = holes & ~frozen & support & (confidence > 0)
    # Extrapolate *observed neck* colors, excluding fixed hair/ear/accessories.
    # No generated foreground texels can act as a neck donor.
    _, nearest = distance_transform_edt(~known, return_indices=True)
    donor_y, donor_x = nearest
    patch = crop[donor_y, donor_x, :3].astype(float)
    # Preserve the observed boundary at distance=0. A directional neck texture
    # continuation is only a baseline hypothesis; it cannot recover hidden pores.
    replacement = background[:, :, :3].astype(float)
    replacement[candidate_domain] = patch[candidate_domain]
    local = rendered[oy:oy+height, ox:ox+width, :3].astype(float)
    delta = (replacement - background[:, :, :3]) * (1 - alpha[:, :, None])
    local[candidate_domain] += delta[candidate_domain] * confidence[candidate_domain, None]
    candidate = rendered.copy()
    candidate[oy:oy+height, ox:ox+width, :3] = np.clip(np.rint(local), 0, 255).astype(np.uint8)
    Image.fromarray(candidate).save(out / 'candidate.png')
    # Separate the old illumination algorithm from the material hypothesis.
    # Record this as a second ablation, not a hidden additional correction.
    before_lighting = np.asarray(Image.open(debug/'background-before-lighting.png')).astype(float)
    correction = (background[:,:,:3].astype(float)-before_lighting[:,:,:3])*(1-alpha[:,:,None])
    correction[frozen] = 0
    no_relight = candidate.copy()
    rgb = no_relight[oy:oy+height,ox:ox+width,:3].astype(float)-correction
    no_relight[oy:oy+height,ox:ox+width,:3] = np.clip(np.rint(rgb),0,255).astype('uint8')
    Image.fromarray(no_relight).save(out/'candidate-no-relight.png')
    Image.fromarray((candidate_domain * 255).astype(np.uint8)).save(out / 'neck-target.png')
    Image.fromarray(np.clip(replacement, 0, 255).astype(np.uint8)).save(out / 'neck-layer.png')
    changed = np.any(candidate != rendered, axis=2)
    target_full = np.zeros(original.shape[:2], bool)
    target_full[oy:oy+height, ox:ox+width] = candidate_domain
    fixed_full = np.maximum.reduce([raster_coverage(masks[key], original.shape[1], original.shape[0]) for key in ['ears', 'hair', 'accessories']]) >= .5
    report = {'scope': 'One same-photo isolated material hypothesis; no independent alpha truth; not accepted until visual QA.',
              'inputs': {str(path.relative_to(ROOT)): digest(path) for path in [graph_path, evidence/'output.png', debug/'source-crop.png', debug/'background-after-lighting.png', debug/'known-neck.png', debug/'holes.png']},
              'fits': fits, 'writtenPixels': int(changed.sum()), 'outsideTargetChanges': int((changed & ~target_full).sum()),
              'fixedOutputChanges': int((changed & fixed_full).sum()), 'alphaChanges': int(np.count_nonzero(candidate[:,:,3] != rendered[:,:,3])),
              'outputSha256': digest(out/'candidate.png')}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    boxes = {'face': (300,280,720,650), 'jaw': (355,470,668,635)}
    for label, box in boxes.items():
        a, b, c = [Image.fromarray(im).convert('RGB').crop(box) for im in [original, rendered, candidate]]
        w,h = a.size
        canvas = Image.new('RGB', (w*6,h*2+28), '#22252b'); draw = ImageDraw.Draw(canvas)
        for j, (im, title) in enumerate(zip([a,b,c], ['Original','v15','Neck hypothesis'])):
            canvas.paste(im.resize((w*2,h*2),Image.Resampling.NEAREST),(j*w*2,28));draw.text((j*w*2+8,8),title,fill='white')
        canvas.save(out/f'{label}-comparison.png')
    box = boxes['jaw']
    images = [Image.fromarray(im).convert('RGB').crop(box) for im in [candidate,no_relight]]
    w,h=images[0].size
    canvas=Image.new('RGB',(w*4,h*2+28),'#222');draw=ImageDraw.Draw(canvas)
    for j,(im,title) in enumerate(zip(images,['Amodal + old relighting','Amodal; no old relighting'])):
        canvas.paste(im.resize((w*2,h*2),Image.Resampling.NEAREST),(j*w*2,28));draw.text((j*w*2+8,8),title,fill='white')
    canvas.save(out/'relighting-ablation.png')
    print(json.dumps({key: report[key] for key in ['writtenPixels','outsideTargetChanges','fixedOutputChanges','alphaChanges','outputSha256']}))


if __name__ == '__main__':
    main()
