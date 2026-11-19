"""Derive visible motion identity from explicit source roles, never matte alpha.

This is a manual correction adapter. Hidden rear support is an estimate and
cannot become a visible contour through a foreground occluder. It does not
infer anatomy, choose prompts, modify optical layers or read edited images.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, map_coordinates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visible_motion_identity(roles, size, observed, proposals):
    width, height = size
    if observed.shape != (height, width):
        raise ValueError('source-sized coverage required')
    if not np.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
        raise ValueError('finite source coverage in [0,1] required')
    masks = {key: Image.new('L', size, 0) for key in ('moving-rear', 'stationary-front', 'stationary-scene')}
    proposed_front = np.zeros((height,width),bool)
    seen = set()
    for role in roles:
        if role['role'] not in masks or role['id'] in seen:
            raise ValueError('unique source roles with an explicit movement class required')
        seen.add(role['id'])
        if role['id'] in proposals:
            proposal = np.asarray(proposals[role['id']])
            if role['role'] != 'stationary-front' or proposal.shape != observed.shape or proposal.dtype != bool:
                raise ValueError('replacement must be a source-sized categorical stationary-front identity')
            proposed_front |= binary_fill_holes(proposal)
            continue
        points = role['polygon']
        if len(points) < 3 or any(len(p) != 2 or not all(np.isfinite(p)) or not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
            raise ValueError('source-coordinate polygon required')
        ImageDraw.Draw(masks[role['role']]).polygon([tuple(p) for p in points], fill=255)
    if set(proposals) - seen:
        raise ValueError('replacement proposal has no explicit source role')
    moving, fixed, scene = (np.asarray(masks[key]) > 0 for key in masks)
    # A candidate's enclosed omissions are unknown identity, not proof of
    # visible rear material. Exclude them from geometric target measurement;
    # this does not fill physical sleeve opacity or change source RGB.
    exclusion = fixed | scene | proposed_front
    return moving & (observed >= .5) & ~exclusion, exclusion


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source_directory', type=Path)
    ap.add_argument('manual_roles_directory', type=Path)
    ap.add_argument('prompted_source_layers_directory', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--replacement-role', required=True, help='Explicit stationary-front role replaced by the source-bound prompted proposal')
    args = ap.parse_args()
    source = args.source_directory/'source.png'
    graph_path = args.source_directory/'source-graph.json'
    roles_path = args.manual_roles_directory/'source-materials.json'
    freeze = json.loads((args.manual_roles_directory/'annotation-freeze.json').read_text())
    roles = json.loads(roles_path.read_text())
    if sha(roles_path) != freeze['source-materials.json'] or roles['sourceSHA256'] != sha(source) or roles['originalGraphSHA256'] != sha(graph_path):
        raise ValueError('frozen manual source roles do not match this original image/graph')
    if not roles['manualIntervention'] or roles['automaticGeneralizationClaim']:
        raise ValueError('manual intervention must remain explicit')
    front_dir = args.prompted_source_layers_directory/'source-front-matte-selected-domain'
    front_protocol = json.loads((front_dir/'protocol.json').read_text())
    if front_protocol['sourceSHA256'] != sha(source):
        raise ValueError('prompted sleeve belongs to another source')
    proposal_path = front_dir/'sleeve-proposal.png'
    # Bind the proposal to the frozen source-only prompt input, not an alpha file.
    original_proposal = front_protocol['inputs']['sleeveProposal']
    if sha(original_proposal['path']) != original_proposal['sha256']:
        raise ValueError('original prompted proposal changed')
    proposal = np.asarray(Image.open(proposal_path).convert('L')) > 0
    original = np.asarray(Image.open(original_proposal['path']).convert('L')) > 0
    if not np.array_equal(proposal, original):
        raise ValueError('source proposal was edited after the prompt run')
    with Image.open(source) as im:
        rgb = np.asarray(im.convert('RGB'))
    height, width = rgb.shape[:2]
    graph = json.loads(graph_path.read_text())
    mask = graph['operations'][0]['params']['subjectMask']
    values = np.asarray(mask['data'], float).reshape(mask['height'], mask['width']) / 255
    yy, xx = np.mgrid[:height, :width]
    observed = map_coordinates(values, [(yy+.5)*mask['height']/height-.5, (xx+.5)*mask['width']/width-.5], order=1, mode='nearest')
    evidence_path = args.source_directory/'class-evidence.json'
    evidence = json.loads(evidence_path.read_text())
    classes_path = args.source_directory/'fashn-classes.u8'
    if sha(classes_path) != evidence['classesSHA256']:
        raise ValueError('same-source class evidence changed')
    classes = np.fromfile(classes_path, np.uint8).reshape(evidence['decodedDimensions'])
    apparel = classes[[3,4,5,6,7,16]].astype(float).sum(axis=0)/255
    ah, aw = apparel.shape
    compatible = map_coordinates(apparel, [(yy+.5)*ah/height-.5, (xx+.5)*aw/width-.5], order=1, mode='nearest')
    moving, exclusion = visible_motion_identity(roles['roles'], (width,height), observed, {args.replacement_role:proposal})
    # Class logits can disagree with explicitly reviewed material roles (e.g.
    # shaded pink knit classified as skin). Keep that conflict visible for
    # review; do not punch holes in the user's material identity with it.
    class_conflicts = moving & (compatible < .5)
    if not moving.any():
        raise ValueError('no observed moving source material')
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {'sourceSHA256': sha(source), 'mask': {'shape':'raster','width':width,'height':height,
        'data':(moving.astype(np.uint8)*255).ravel().tolist()}}
    (args.output/'torso-mask.json').write_text(json.dumps(payload, separators=(',',':')))
    Image.fromarray(moving.astype(np.uint8)*255).save(args.output/'moving-identity.png')
    Image.fromarray(np.where(moving[...,None], np.rint(rgb*.55+[0,110,35]), rgb*.35).clip(0,255).astype(np.uint8)).save(args.output/'source-geometry.png')
    report = {'sourceSHA256':sha(source), 'sourceRolesSHA256':sha(roles_path),
        'promptProtocolSHA256':sha(front_dir/'protocol.json'), 'sleeveProposalSHA256':sha(proposal_path),
        'sourceClassEvidenceSHA256':sha(evidence_path),
        'implementationSHA256':sha(__file__), 'manualCorrection':True, 'sourceOnly':True,
        'replacementRole':args.replacement_role,
        'alphaRead':False, 'editedImagesRead':False, 'automaticGeneralizationClaim':False,
        'rule':'observed source foreground within explicit moving identity, excluding fixed source identities and enclosed unknown sleeve omissions; coarse apparel classes are diagnostic only',
        'coarseClassConflictPixels':int(class_conflicts.sum()),
        'coarseClassMayOverrideExplicitRoles':False,
        'pixels':int(moving.sum()), 'excludedPixels':int(exclusion.sum()),
        'maskSHA256':sha(args.output/'torso-mask.json'), 'requiresIndependentSourceReview':True}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    Image.fromarray(class_conflicts.astype(np.uint8)*255).save(args.output/'coarse-class-conflicts.png')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
