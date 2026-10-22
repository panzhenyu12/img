"""Prepare an explicitly reviewed jaw-arc intent; never solve or render."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.sparse import load_npz

from reduced_field import ROOT
from adapt import read, save, sha
from prepare_regional_input import input_artifacts

VARIANT = 'jaw-arc-source-v1'
ORIGINAL_DEFINITION = 'source-whole-face-width-retained-chin-envelope'
NO_NECK = 'none declared in original baseline; none borrowed from another photograph'
SELECTED = [{'side': side, 'start_boundary': 'mouth_height', 'end_boundary': 'chin_proxy'}
            for side in ['left', 'right']]


def validate_review(review, audit, sidecar, sidecar_sha256):
    if (review.get('mode') != 'model_reviewed_offline_selection' or review.get('name') != 'woman'
            or audit['name'] != 'woman' or review.get('selected_arcs') != SELECTED):
        raise ValueError('jaw requires the explicit woman two-arc model review')
    if (review.get('source_sha256') != audit['input_sha256'][audit['source_image']]
            or review.get('sidecar_sha256') != sidecar_sha256):
        raise ValueError('jaw review source/sidecar identity differs')
    if (not isinstance(review.get('notes'), list) or not review['notes']
            or any(not isinstance(note, str) or not note.strip() for note in review['notes'])
            or not isinstance(review.get('evidence_images_sha256'), dict) or not review['evidence_images_sha256']):
        raise ValueError('jaw review requires source evidence images and explanatory notes')
    if (sidecar.get('automatic_edit_authorized') is not False or sidecar.get('visibility_status') != 'unknown'
            or review.get('automatic_edit_authorized', False) is not False):
        raise ValueError('offline review cannot upgrade unknown sidecar automatic authorization')
    if (sidecar.get('source_frame_unchanged') != audit['frame_raw']
            or sidecar.get('source_contour') != audit['source_contour']):
        raise ValueError('jaw sidecar source order/frame differs from source audit')
    return [{'id': side+'-mouth-to-chin',
             'start': {'edge': sidecar['sides'][side]['boundaries']['mouth_height']['row0'],
                       't': sidecar['sides'][side]['boundaries']['mouth_height']['t']},
             'end': {'edge': sidecar['sides'][side]['boundaries']['chin_proxy']['row0'],
                     't': sidecar['sides'][side]['boundaries']['chin_proxy']['t']},
             'direction': sidecar['sides'][side]['direction_in_source_row_order']}
            for side in ['left', 'right']]


def verify_sidecar(audit, sidecar, graph_path):
    """Reconstruct associations from the unchanged source evidence, not edits."""
    from source_regions import associate_source, FACE_OVAL, FEATURE_INDICES
    if sidecar.get('schema') != 'source-region-associations-v1':
        raise ValueError('unsupported source sidecar schema')
    proof = sidecar['provenance']['input_sha256']
    if proof.get(str(graph_path)) != sha(graph_path):
        raise ValueError('source sidecar graph identity differs')
    for path, digest in proof.items():
        if sha(Path(path)) != digest: raise ValueError('source sidecar evidence changed: '+path)
    size = audit['native_intent_source_proof']['source_dimensions']
    params = read(graph_path)['operations'][0]['params']
    pixels = lambda rows: [[p['x']*size[0]-.5, p['y']*size[1]-.5] for p in rows]
    if pixels(params['landmarks']) != audit['source_oval']:
        raise ValueError('sidecar canonical source coordinates differ')
    features = {key: pixels(params['semanticMasks'][key]['points']) for key in FEATURE_INDICES}
    if [features['leftEye'], features['rightEye']] != audit['source_eyes']:
        raise ValueError('sidecar source eye identity differs')
    expected = associate_source(audit['source_contour'], audit['source_oval'], features,
                                audit['frame_raw'], FACE_OVAL)
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError('sidecar source boundary/order differs: '+key)


def controller_request(data, audit, arcs, amount):
    return {'frame': {**audit['frame_raw'], 'midline_u': audit['midline_u']},
            'amount': float(amount), 'max_radius_fraction': audit['calibration_k_full'],
            'contour': np.asarray(data['target_sources'][:128]).tolist(), 'arcs': arcs}


def expected_targets(data, source, provenance, audit, arcs, amount):
    """Independent source-arclength beta polynomial, with all other rows frozen."""
    if audit['name'] != 'woman': raise ValueError('first jaw adapter supports only woman')
    if provenance.get('source_neck_support') or source.get('source_neck_targets') not in (None, NO_NECK):
        raise ValueError('jaw adapter does not inherit neck support/attachments')
    allowed = {'leftEye', 'leftBrow', 'rightEye', 'rightBrow'}
    if any(not g.get('features') or not set(g['features']).issubset(allowed) for g in source['groups']):
        raise ValueError('jaw adapter rejects non eye/brow material roles')
    if (np.any(np.asarray(data['group_priors']) != 0) or np.any(np.asarray(data['group_prior_weights']) != 0)
            or any(g['translation_prior_weight'] != 0 or np.any(np.asarray(g['forward_translation_prior']) != 0)
                   for g in source['groups'])):
        raise ValueError('jaw adapter rejects nonzero group priors')
    p = np.asarray(data['target_sources'], float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 128 or not np.array_equal(p[:128], audit['source_contour']):
        raise ValueError('jaw source contour differs from source audit')
    k = float(audit['calibration_k_full'])
    if not np.isfinite([amount, k]).all() or not -100 <= amount <= 100 or not 0 <= k <= .45:
        raise ValueError('invalid signed jaw amount or source calibration')
    if audit['midline_u'] != 0.: raise ValueError('jaw must retain the audited midline')
    lengths = np.linalg.norm(np.roll(p[:128], -1, axis=0)-p[:128], axis=1)
    if np.any(lengths <= 0): raise ValueError('jaw source path has degenerate edges')
    cumulative = np.r_[0., np.cumsum(lengths)]; perimeter = cumulative[-1]
    gains = np.zeros(128); ids = [None]*128
    for arc in arcs:
        if arc['direction'] not in [-1, 1]: raise ValueError('jaw arc direction invalid')
        positions = []
        for boundary in [arc['start'], arc['end']]:
            edge, t = boundary['edge'], boundary['t']
            if type(edge) is not int or not 0 <= edge < 128 or not np.isfinite(t) or not 0 <= t <= 1:
                raise ValueError('jaw arc boundary invalid')
            positions.append((cumulative[edge]+t*lengths[edge])%perimeter)
        start, end = positions; length = ((end-start)*arc['direction'])%perimeter
        if length <= 0: raise ValueError('jaw arc endpoints must differ')
        distance = ((cumulative[:-1]-start)*arc['direction'])%perimeter
        selected = (distance > 0)&(distance < length)
        if any(ids[i] is not None for i in np.flatnonzero(selected)):
            raise ValueError('jaw arcs overlap source rows')
        t = distance[selected]/length
        gains[selected] = 64*t**3*(1-t)**3
        for i in np.flatnonzero(selected): ids[i] = arc['id']
    frame = audit['frame_raw']; across = np.asarray(frame['across'])
    u = (p[:128]-np.asarray(frame['origin']))@across
    delta = -amount/100*k*u*gains
    result = np.asarray(data['target_destinations']).copy()
    result[:128] = p[:128]+delta[:, None]*across
    return result, gains, ids


def check_controller_output(output, data, expected, gains, ids):
    if output.get('controller') != 'jaw_arc_v1' or len(output.get('targets', [])) != 128:
        raise ValueError('Rust did not execute the declared jaw controller')
    rows = output['targets']
    if ([r['index'] for r in rows] != list(range(128))
            or not np.array_equal([r['source'] for r in rows], data['target_sources'][:128])):
        raise ValueError('jaw Rust source rows missing or reordered')
    if [r['arc_id'] for r in rows] != ids:
        raise ValueError('jaw Rust arc ownership differs from reviewed source arcs')
    for actual, wanted, name in [([r['target'] for r in rows], expected[:128], 'targets'),
                                  ([r['point_gain'] for r in rows], gains, 'beta gains')]:
        actual = np.asarray(actual)
        if actual.shape != wanted.shape or not np.isfinite(actual).all() or np.max(abs(actual-wanted), initial=0.) > 1e-10:
            raise ValueError('jaw Rust '+name+' differ from independent polynomial reference')


def prepare(parent, audit_path, sidecar_path, review_path, output, binary):
    from input_contract import validate_input, require_same_signed_amount
    parent, audit_path, sidecar_path, review_path, output, binary = [Path(p).resolve() for p in
        (parent, audit_path, sidecar_path, review_path, output, binary)]
    if output.exists(): raise FileExistsError('refusing to overwrite jaw inputs: '+str(output))
    validated = validate_input(parent); pc, ps = read(parent/'contract.json'), read(parent/'source-contract.json')
    if validated['input_kind'] != 'source_input' or pc.get('target_definition') != ORIGINAL_DEFINITION:
        raise ValueError('jaw adapter requires the original woman source_input')
    audit, sidecar, review = read(audit_path), read(sidecar_path), read(review_path)
    if sidecar['provenance']['input_sha256'].get(str(audit_path)) != sha(audit_path):
        raise ValueError('jaw calibration/frame audit is not the sidecar frozen audit')
    if (audit['name'] != 'woman' or audit['input_sha256'][audit['source_image']] != pc['source_photo_sha256']
            or sha(Path(pc['source_photo'])) != pc['source_photo_sha256']):
        raise ValueError('jaw audited woman photograph identity differs')
    hashes = dict(validated['verified_input_hashes'])
    for path, digest in {**audit['input_sha256'], **review.get('evidence_images_sha256', {})}.items():
        if sha(Path(path)) != digest: raise ValueError('jaw source review evidence changed: '+path)
        hashes[path] = digest
    provenance = read(Path(pc['baseline'])/'provenance.json')
    graph = Path(provenance['source_trace']).parent/'graph.json'
    verify_sidecar(audit, sidecar, graph)
    hashes.update(sidecar['provenance']['input_sha256'])
    arcs = validate_review(review, audit, sidecar, sha(sidecar_path))
    data = dict(np.load(parent/'data.npz', allow_pickle=False)); amount = pc['requested_amount']
    require_same_signed_amount(amount, ps['requested_amount'])
    expected, gains, ids = expected_targets(data, ps, provenance, audit, arcs, amount)
    request = controller_request(data, audit, arcs, amount)
    completed = subprocess.run([str(binary)], input=json.dumps(request), text=True, capture_output=True, check=False)
    if completed.returncode: raise ValueError('Rust jaw controller failed: '+completed.stderr)
    rust = json.loads(completed.stdout); check_controller_output(rust, data, expected, gains, ids)
    new_dest = expected.copy(); new_dest[:128] = [r['target'] for r in rust['targets']]
    copied = input_artifacts(parent)
    for path in [parent/name for name in copied]+[parent/'report.json', audit_path, sidecar_path, review_path, binary, Path(__file__)]:
        hashes[str(path)] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    for name in copied:
        (output/name).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(parent/name, output/name)
    new_delta = data['target_deltas'].copy(); new_delta[:128] = new_dest[:128]-data['target_sources'][:128]
    new_data = {**data, 'target_destinations': new_dest, 'target_deltas': new_delta}
    np.savez_compressed(output/'data.npz', **new_data); np.save(output/'target_deltas.npy', new_data['target_deltas'])
    descriptor = {'variant': VARIANT, 'parent_input': str(parent), 'parent_contract_sha256': sha(parent/'contract.json'),
        'source_audit': str(audit_path), 'source_audit_sha256': sha(audit_path),
        'source_sidecar': str(sidecar_path), 'source_sidecar_sha256': sha(sidecar_path),
        'source_review': str(review_path), 'source_review_sha256': sha(review_path),
        'controller_request': request, 'controller_output': rust,
        'controller_binary': str(binary), 'controller_binary_sha256': sha(binary),
        'new_editing_intent_not_same_request': True, 'automatic_edit_authorized': False}
    sc = copy.deepcopy(ps)
    for i, target in enumerate(sc['targets']):
        target.update(destination=new_dest[i].tolist(), forward_delta=new_data['target_deltas'][i].tolist())
    sc.update(target_definition=VARIANT, jaw_variant=descriptor,
        source_target_rule='reviewed mouth-to-chin source arcs use compact beta gain; all non-face targets and original source roles retained')
    sc.pop('solving_authorized', None); save(output/'source-contract.json', sc)
    save(output/'target-intent.json', {'name': VARIANT, 'amount': amount, 'source': data['target_sources'][:128].tolist(),
        'requested': new_dest[:128].tolist(), 'jaw_variant': descriptor, 'no_solve_no_render': True})
    rhs = np.zeros_like(np.load(parent/'quadratic/rhs.npy', allow_pickle=False)); assembly = read(output/'assembly.json')
    for block in assembly['blocks']:
        matrix = load_npz(output/block['matrix_file'])
        weights = np.load(output/block['weights_file'], allow_pickle=False) if block.get('weights_file') else block.get('weights', 1.)
        targets = np.load(output/block['rhs_file'], allow_pickle=False) if block.get('rhs_file') else np.zeros((matrix.shape[0], 2))
        rhs += matrix.T @ (np.broadcast_to(weights, (matrix.shape[0],))[:, None]*targets)
    np.save(output/'quadratic/rhs.npy', rhs)
    c = copy.deepcopy(pc); c.pop('solving_authorized', None)
    c.update(source_contract=str(output/'source-contract.json'), source_contract_sha256=sha(output/'source-contract.json'),
        target_intent_sha256=sha(output/'target-intent.json'), target_definition=VARIANT, jaw_variant=descriptor,
        execution_status='inputs_prepared_no_solve_no_render')
    save(output/'contract.json', c)
    mutable = {'contract.json', 'source-contract.json', 'target-intent.json', 'data.npz', 'target_deltas.npy', 'quadratic/rhs.npy'}
    report = {'status': 'PreparedReviewedJawSourceVariant', 'inputs_sha256': hashes,
        'constructed_outputs_sha256': {name: sha(output/name) for name in copied},
        'static_parent_files': {name: sha(parent/name) for name in copied if name not in mutable},
        'jaw_variant': descriptor, 'no_solve_no_render': True, 'aesthetic_approval': False}
    if any(sha(Path(path)) != digest for path, digest in hashes.items()): raise ValueError('source changed during jaw adaptation')
    save(output/'report.json', report)
    result = validate_input(output); save(output/'validation.json', result)
    print({'output': str(output), 'amount': amount, 'status': result['status']})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['parent', 'audit', 'sidecar', 'review', 'output', 'binary']:
        parser.add_argument('--'+option, type=Path, required=True)
    a = parser.parse_args(); prepare(a.parent, a.audit, a.sidecar, a.review, a.output, a.binary)
