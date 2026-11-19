"""Source-evidence assignment over a frozen segmentation candidate bank.

The first proposal separates sleeves from the remaining visible garments. It
does not infer separate coat instances, optical alpha or hidden layer order.
Those limitations are explicit rather than assigned a guessed depth label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, label


def select_candidate(masks, positive, negative, groups=None):
    """Choose a proposal supported by each visible bone segment.

    Model confidence is deliberately absent. A silhouette cannot acquire an
    anatomical identity just by containing an arm. Witnesses are proposal
    evidence, not ground truth; final material boundaries need source review.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError('candidate masks must be K,H,W')
    positive, negative = [np.asarray(p, dtype=int).reshape(-1, 2) for p in (positive, negative)]
    if len(positive) < 3 or len(negative) < 1:
        return {'status': 'insufficient-source-evidence', 'index': None}
    for points in (positive, negative):
        if (points < 0).any() or (points[:, 0] >= masks.shape[2]).any() or (points[:, 1] >= masks.shape[1]).any():
            raise ValueError('witness outside source')
    positive_hits = masks[:, positive[:, 1], positive[:, 0]]
    positive_count = positive_hits.sum(axis=1)
    negative_count = masks[:, negative[:, 1], negative[:, 0]].sum(axis=1)
    if groups is None:
        supported = positive_count == len(positive)
    else:
        if not groups or any(sum(i is not None for i in group) < 2 for group in groups):
            return {'status': 'insufficient-source-segment-evidence', 'index': None}
        supported = np.ones(len(masks), dtype=bool)
        for group in groups:
            hits = np.zeros((len(masks), len(group)), dtype=bool)
            for slot, index in enumerate(group):
                if index is not None:
                    hits[:, slot] = positive_hits[:, index]
            # Require an observed interval in every bone, not an arbitrary
            # global vote threshold. An uncertain endpoint remains missing;
            # it never causes us to paint or extrapolate the candidate mask.
            supported &= np.any(hits[:, :-1] & hits[:, 1:], axis=1)
    admissible = np.flatnonzero(supported & (negative_count == 0))
    if not len(admissible):
        return {'status': 'no-consistent-source-candidate', 'index': None,
                'positiveCovered': positive_count.tolist(), 'negativeCovered': negative_count.tolist()}
    admissible = admissible[positive_count[admissible] == positive_count[admissible].max()]
    sizes = masks[admissible].sum(axis=(1, 2))
    index = int(admissible[np.argmin(sizes)])
    return {'status': 'selected-source-proposal', 'index': index,
            'consistentCandidates': admissible.tolist(), 'pixels': int(masks[index].sum()),
            'unresolvedWitnesses': np.flatnonzero(~positive_hits[index]).tolist()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_witnesses(points, classes, width, height):
    """Central bone samples, filtered by observed foreground class confidence.

    Samples are source-only proposals. A clothed/occluded skeleton can be
    wrong despite visibility, so witness consistency is not role acceptance.
    """
    roles, role_groups = {}, {}
    good = lambda p: all(np.isfinite(p.get(k, float('nan'))) for k in ('x', 'y')) and p.get('visibility', 1) >= .5
    def samples(bones, allowed):
        out, groups = [], []
        for a, b in bones:
            # Preserve physical sample positions. Filtering a missing middle
            # observation must not make the two endpoints adjacent evidence.
            group = [None, None, None]
            if not good(a) or not good(b):
                groups.append(group)
                continue
            for slot, t in enumerate((.25, .5, .75)):
                x, y = ((1-t)*a[k]+t*b[k] for k in ('x', 'y'))
                if not (0 <= x < 1 and 0 <= y < 1):
                    continue
                px, py = int(x*classes.shape[2]), int(y*classes.shape[1])
                posterior = classes[:, py, px]
                if int(posterior.argmax()) in allowed and posterior.max() >= 128:
                    group[slot] = len(out)
                    out.append([min(width-1, int(x*width)), min(height-1, int(y*height))])
            groups.append(group)
        return out, groups
    for name, ids in [('left-arm', (11, 13, 15)), ('right-arm', (12, 14, 16))]:
        a, b, c = [points[i] for i in ids]
        roles[name], role_groups[name] = samples([(a, b), (b, c)], {3, 4, 12, 16})
    shoulders = {k: (points[11][k]+points[12][k])/2 for k in ('x', 'y')}
    hips = {k: (points[23][k]+points[24][k])/2 for k in ('x', 'y')}
    shoulders['visibility'] = min(points[i].get('visibility', 1) for i in (11, 12))
    hips['visibility'] = min(points[i].get('visibility', 1) for i in (23, 24))
    roles['torso'], role_groups['torso'] = samples([(shoulders, hips)], {3, 4, 5, 6, 7, 16})
    return roles, role_groups


def select_person_candidate(masks, foreground, witnesses):
    """Match a complete observed instance, without filling any candidate.

    Foreground is an independent semantic observation, not a reference edit.
    All body witnesses must belong to the proposed instance. Soft overlap
    ranks the remaining whole-person proposals; scores from SAM are unused.
    """
    points = np.asarray(witnesses, dtype=int).reshape(-1, 2)
    if len(points) < 3:
        return {'status': 'insufficient-person-evidence', 'index': None}
    admissible = np.flatnonzero(masks[:, points[:, 1], points[:, 0]].all(axis=1))
    if not len(admissible):
        return {'status': 'no-complete-person-proposal', 'index': None}
    foreground = np.asarray(foreground, dtype=np.float64)
    if foreground.shape != masks.shape[1:] or not np.isfinite(foreground).all() or ((foreground < 0) | (foreground > 1)).any():
        raise ValueError('finite source-sized foreground probabilities required')
    total = float(foreground.sum())
    scores = []
    for index in admissible:
        intersection = float(foreground[masks[index]].sum())
        union = total+int(masks[index].sum())-intersection
        scores.append(intersection/union if union > 0 else 0)
    chosen = int(np.argmax(scores))
    return {'status': 'selected-source-proposal', 'index': int(admissible[chosen]),
            'semanticSoftIOU': scores[chosen], 'consistentCandidates': admissible.tolist()}


def connected_torso_proposal(proposal, witnesses):
    """Keep only observed material connected to a source torso witness.

    Subtraction is not identity evidence: isolated pixels left outside a
    sleeve proposal do not become chest material. Removed components remain
    unassigned/fixed, not optical background or a guessed sleeve layer.
    """
    proposal = np.asarray(proposal, dtype=bool)
    points = np.asarray(witnesses, dtype=int).reshape(-1, 2)
    if proposal.ndim != 2 or (points < 0).any() or (points[:, 0] >= proposal.shape[1]).any() or (points[:, 1] >= proposal.shape[0]).any():
        raise ValueError('source-sized torso mask and in-bounds witnesses required')
    components, count = label(proposal, structure=np.ones((3, 3)))
    observed = components[points[:, 1], points[:, 0]]
    selected = np.unique(observed[observed != 0])
    kept = np.isin(components, selected)
    return kept, {
        'status': 'source-proposal-review-required' if len(selected) else 'no-observed-torso-component',
        'componentCount': int(count), 'observedComponentCount': len(selected),
        'unassignedPixels': int((proposal & ~kept).sum()),
        'unresolvedWitnesses': np.flatnonzero(observed == 0).tolist(),
        'unassignedMeaning': 'unresolved fixed material, not optical background',
    }


def run(bank_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((bank_dir/'protocol.json').read_text())
    bank_meta = json.loads((bank_dir/'run.json').read_text())
    source_dir = Path(protocol['sourceDirectory'])
    source_path = Path(protocol['files']['source']['path'])
    for entry in protocol['files'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('candidate source input changed')
    if sha(bank_dir/'candidate-bank.npz') != bank_meta['bankSha256']:
        raise ValueError('candidate bank changed')
    evidence = json.loads((source_dir/'class-evidence.json').read_text())
    classes = np.fromfile(source_dir/'fashn-classes.u8', np.uint8).reshape(evidence['decodedDimensions'])
    observation = json.loads((source_dir/'observation.json').read_text())
    source = Image.open(source_path).convert('RGB')
    width, height = source.size
    bank = np.load(bank_dir/'candidate-bank.npz')
    if tuple(bank['imageShape']) != (height, width):
        raise ValueError('source dimensions differ')
    masks = np.unpackbits(bank['masksPacked'], axis=1)[:, :height*width].reshape(-1, height, width).astype(bool)
    witnesses, groups = body_witnesses(observation['points'], classes, width, height)
    ph, pw = classes.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    coordinates = [np.clip((yy+.5)*ph/height-.5, 0, ph-1),
                   np.clip((xx+.5)*pw/width-.5, 0, pw-1)]
    foreground = map_coordinates(1-classes[0].astype(float)/255, coordinates, order=1, mode='nearest')
    whole_person = select_person_candidate(masks, foreground, [p for pts in witnesses.values() for p in pts])
    if whole_person['index'] is not None:
        whole_person['candidateID'] = bank_meta['candidates'][whole_person['index']]['id']
    report = {'sourceSHA256': sha(source_path), 'bankSHA256': bank_meta['bankSha256'],
              'implementationSHA256': sha(__file__), 'sourceOnly': True, 'manualPrompts': False,
              'editedOutputsRead': False, 'witnesses': witnesses, 'witnessGroups': groups,
              'wholePerson': whole_person, 'roles': {},
              'limitations': ['Pose witnesses can be occluded or mislocated.',
                  'Remaining clothing is a torso proposal, not proof of garment instance or depth.',
                  'No optical alpha or hidden anatomy is inferred.',
                  'Source proposal needs independent acceptance before any deformation.']}
    preview = source.copy()
    draw = ImageDraw.Draw(preview)
    for name, color in [('left-arm', 'lime'), ('right-arm', 'cyan'), ('torso', 'magenta')]:
        for x, y in witnesses[name]:
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color, outline='black')
    preview.save(out/'source-witnesses.png')
    sleeves = []
    for name in ('left-arm', 'right-arm'):
        negative = [p for other, pts in witnesses.items() if other != name for p in pts]
        selected = select_candidate(masks, witnesses[name], negative, groups[name])
        if selected['index'] is not None:
            selected['candidateID'] = bank_meta['candidates'][selected['index']]['id']
            mask = masks[selected['index']]
            sleeves.append(mask)
            Image.fromarray(mask.astype(np.uint8)*255).save(out/(name+'.png'))
        report['roles'][name] = selected
    if len(sleeves) == 2 and whole_person['index'] is not None:
        # Anatomical ownership, deliberately separate from the original
        # subject alpha which the ordinary renderer continues to use.
        apparel_probability = classes[[3, 4, 5, 6, 7, 16]].astype(float).sum(axis=0)/255
        apparel = map_coordinates(apparel_probability, coordinates, order=1, mode='nearest') > .25
        sleeve = sleeves[0] | sleeves[1]
        torso = apparel & masks[whole_person['index']] & ~sleeve
        torso, component_evidence = connected_torso_proposal(torso, witnesses['torso'])
        report['torsoComponents'] = component_evidence
        if not torso.any():
            report['status'] = 'rejected-no-observed-torso-component'
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'status': report['status'], 'torsoComponents': component_evidence}))
            return
        Image.fromarray(torso.astype(np.uint8)*255).save(out/'torso.png')
        rgb = np.array(source)
        rgb[torso] = np.rint(rgb[torso]*.6+np.array([0, 255, 80])*.4).astype(np.uint8)
        rgb[sleeve] = np.rint(rgb[sleeve]*.6+np.array([255, 0, 180])*.4).astype(np.uint8)
        Image.fromarray(rgb).save(out/'source-roles.png')
        payload = {'sourceSHA256': sha(source_path), 'mask': {'shape': 'raster', 'width': width,
                   'height': height, 'data': (torso.astype(np.uint8)*255).reshape(-1).tolist()}}
        (out/'torso-mask.json').write_text(json.dumps(payload, separators=(',', ':')))
        report.update(status='source-proposal-review-required', torsoPixels=int(torso.sum()),
                      overlappingSleevePixels=int((sleeves[0] & sleeves[1]).sum()),
                      torsoMaskSHA256=sha(out/'torso-mask.json'))
    else:
        report['status'] = 'rejected-insufficient-consistent-source-roles'
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'witnesses', 'roles']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.bank, args.output)
