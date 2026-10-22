"""Freeze explicit regional targets without solving; this is a new editing intent."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import load_npz
from reduced_field import ROOT
from adapt import read, save, sha

VARIANT = 'regional-contour-source-v1'
DEFAULT_BINARY = ROOT/'experiments/semantic-deformation-20261022/rust-controls/target/debug/face-width-targets'


def profile_digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def profile_function(profile, k):
    if not isinstance(profile, dict) or set(profile) != {'knots'}:
        raise ValueError('regional profile requires only explicit knots')
    knots = profile['knots']
    if not isinstance(knots, list) or len(knots) < 2 or any(set(v) != {'height', 'gain'} for v in knots):
        raise ValueError('regional profile requires at least two height/gain knots')
    values = np.asarray([[v['height'], v['gain']] for v in knots], float); x, y = values.T
    if (not np.isfinite(values).all() or x[0] != 0. or x[-1] != 1. or np.any(np.diff(x) <= 0)
            or np.any(y < 0) or not np.isfinite(k) or k < 0 or k*float(y.max()) > .45):
        raise ValueError('regional profile requires increasing [0,1] heights and bounded nonnegative gain')
    interpolate = PchipInterpolator(x, y, extrapolate=False)
    return lambda height: interpolate(np.clip(height, 0., 1.))


def controller_request(data, audit, profile, amount):
    return {'frame': {**audit['frame_raw'], 'midline_u': audit['midline_u']},
            'calibration': {'max_radius_fraction': audit['calibration_k_full'], 'chin_fade_start': .88},
            'amount': float(amount), 'contour': [{'id': f'source-{i}', 'source': p.tolist()}
                for i, p in enumerate(data['target_sources'][:128])],
            'parts': [], 'relations': [], 'regional_profile': profile}


def expected_targets(data, source, provenance, audit, profile, amount):
    """Independent SciPy PCHIP face and source-entry neck attachment reference."""
    if not np.isfinite(amount) or not -100. <= amount <= 100.:
        raise ValueError('finite signed amount in [-100,100] required')
    if (np.any(np.asarray(data['group_prior_weights']) != 0)
            or any(g['translation_prior_weight'] != 0 for g in source['groups'])):
        raise ValueError('regional adapter cannot reuse nonzero group position prior')
    allowed = {'leftEye', 'leftBrow', 'rightEye', 'rightBrow'}
    if any(not g.get('features') or not set(g['features']).issubset(allowed) for g in source['groups']):
        raise ValueError('regional adapter has an unsupported attachment/material role')
    p = np.asarray(data['target_sources'], float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 128 or not np.array_equal(p[:128], audit['source_contour']):
        raise ValueError('source contour differs from independent audit')
    frame = audit['frame_raw']; k = float(audit['calibration_k_full'])
    if audit['midline_u'] != 0.: raise ValueError('explicit original midline_u=0 required')
    gain = profile_function(profile, k)
    local = (p-np.asarray(frame['origin'])) @ np.asarray([frame['across'], frame['down']]).T
    height = (local[:, 1]-frame['bounds'][1])/(frame['bounds'][3]-frame['bounds'][1])
    fade = np.clip((height[:128]-.88)/.12, 0., 1.)
    radial = -amount/100.*k*local[:128, 0]*(1.-fade*fade*(3.-2.*fade))*gain(height[:128])
    result = np.asarray(data['target_destinations'], float).copy()
    result[:128] = p[:128]+radial[:, None]*frame['across']
    neck = provenance.get('source_neck_support'); reports = []
    if neck:
        start, count = neck['base_target_count'], neck['added_target_count']
        if (neck.get('profile_kind') != 'neck-arclength' or type(start) is not int or type(count) is not int
                or start < 128 or count < 1 or start+count != len(p)
                or not np.array_equal(p[start:], neck['source'])
                or not np.array_equal(data['target_destinations'][start:], neck['requested'])):
            raise ValueError('neck source/targets do not match original arc evidence')
        covered = []
        for arc in neck['continuation']:
            ids = np.asarray(arc['source_indices'])
            if (ids.dtype.kind not in 'iu' or ids.ndim != 1 or not len(ids)
                    or np.any(ids < 0) or np.any(ids >= count)):
                raise ValueError('neck arc source indices invalid')
            rows = ids+start; entry = int(rows[0]); factor = float(gain(height[entry]))
            result[rows] = p[rows]+(data['target_destinations'][rows]-p[rows])*factor
            reports.append({'source_rows': rows.tolist(), 'entry_row': entry,
                'entry_source': p[entry].tolist(), 'entry_height': float(height[entry]),
                'profile_entry_gain': factor, 'same_profile_sha256': profile_digest(profile),
                'rule': 'same profile at original neck arc entry; retain signed radius and arclength fade'})
            covered.extend(ids.tolist())
        if sorted(covered) != list(range(count)):
            raise ValueError('neck arcs must cover each original neck row exactly once')
    return result, reports


def check_target_variant(actual, expected, neck_report, recorded_neck):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or np.max(abs(actual-expected), initial=0.) > 1e-10:
        raise ValueError('regional target differs from independent profile and neck source evidence')
    if neck_report != recorded_neck:
        raise ValueError('regional neck entry metadata differs from independent profile')


def input_artifacts(parent):
    names = {'contract.json', 'source-contract.json', 'data.npz', 'source-evidence.npz', 'assembly.json',
        'target-intent.json', 'original-controls.json', 'original-provenance.json', 'prolongation.npz',
        'target.npz', 'target_deltas.npy', 'target_weights.npy', 'group_priors.npy', 'group_prior_weights.npy',
        'quadratic/hessian.npz', 'quadratic/rhs.npy'}
    assembly = read(parent/'assembly.json')
    for block in [*assembly['blocks'], *assembly.get('hard_equalities', [])]:
        for key in ['matrix_file', 'rhs_file', 'weights_file']:
            if block.get(key): names.add(block[key])
    for name in ['Dx.npz', 'Dy.npz', 'background-line-contract.json']:
        if (parent/name).is_file(): names.add(name)
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix not in ('.json', '.npy', '.npz') or not (parent/name).is_file():
            raise ValueError('invalid declared input artifact: '+name)
    return sorted(names)


def prepare(parent, audit_path, profile_path, output, binary=DEFAULT_BINARY):
    from input_contract import validate_input
    parent, audit_path, profile_path, output, binary = [Path(p).resolve() for p in (parent, audit_path, profile_path, output, binary)]
    if output.exists(): raise FileExistsError('refusing to overwrite regional input: '+str(output))
    validated = validate_input(parent); pc, ps = read(parent/'contract.json'), read(parent/'source-contract.json')
    if validated['input_kind'] != 'source_input' or pc.get('target_definition') == VARIANT:
        raise ValueError('regional adapter requires original woman/hair source_input, not signed-user or regional input')
    data = dict(np.load(parent/'data.npz', allow_pickle=False)); audit = read(audit_path); profile = read(profile_path)
    if (audit['name'] not in ['hair', 'woman'] or sha(Path(pc['source_photo'])) != pc['source_photo_sha256']
            or audit['input_sha256'][audit['source_image']] != pc['source_photo_sha256']):
        raise ValueError('audited source photograph identity differs')
    hashes = dict(validated['verified_input_hashes'])
    for name, digest in audit['input_sha256'].items():
        if sha(Path(name)) != digest: raise ValueError('audited source evidence changed: '+name)
        hashes[name] = digest
    provenance = read(Path(pc['baseline'])/'provenance.json'); amount = float(pc['requested_amount'])
    expected, neck_report = expected_targets(data, ps, provenance, audit, profile, amount)
    request = controller_request(data, audit, profile, amount)
    completed = subprocess.run([str(binary)], input=json.dumps(request), text=True, capture_output=True, check=False)
    if completed.returncode: raise ValueError('Rust regional controller failed: '+completed.stderr)
    rust = json.loads(completed.stdout)
    if rust['controller'] != 'regional_contour_v1' or len(rust['targets']) != 128:
        raise ValueError('Rust binary did not execute the declared regional controller')
    if not np.array_equal([r['source'] for r in rust['targets']], data['target_sources'][:128]):
        raise ValueError('Rust controller changed source coordinates')
    new_dest = expected.copy(); new_dest[:128] = [r['target'] for r in rust['targets']]
    check_target_variant(new_dest, expected, neck_report, neck_report)
    copied = input_artifacts(parent)
    for path in [parent/name for name in copied]+[parent/'report.json', audit_path, profile_path, binary, Path(__file__)]:
        hashes[str(path)] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    for name in copied:
        (output/name).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(parent/name, output/name)
    new_data = {**data, 'target_destinations': new_dest, 'target_deltas': new_dest-data['target_sources']}
    np.savez_compressed(output/'data.npz', **new_data); np.save(output/'target_deltas.npy', new_data['target_deltas'])
    descriptor = {'variant': VARIANT, 'parent_input': str(parent), 'parent_contract_sha256': sha(parent/'contract.json'),
        'source_audit': str(audit_path), 'source_audit_sha256': sha(audit_path), 'profile': profile,
        'profile_sha256': profile_digest(profile), 'controller_request': request, 'controller_output': rust,
        'controller_binary': str(binary), 'controller_binary_sha256': sha(binary), 'neck_arcs': neck_report,
        'new_editing_intent_not_same_request': True}
    sc = copy.deepcopy(ps)
    for index, target in enumerate(sc['targets']):
        target.update(destination=new_dest[index].tolist(), forward_delta=new_data['target_deltas'][index].tolist())
    sc.update(target_definition=VARIANT, regional_variant=descriptor,
        source_target_rule='regional profile replaces 128 face targets and scales neck arcs by source-entry gain; other targets and source roles unchanged')
    sc.pop('solving_authorized', None); save(output/'source-contract.json', sc)
    intent = {'name': VARIANT, 'amount': amount, 'source': data['target_sources'][:128].tolist(),
        'requested': new_dest[:128].tolist(), 'regional_variant': descriptor, 'no_solve_no_render': True}
    save(output/'target-intent.json', intent)
    rhs = np.zeros_like(np.load(parent/'quadratic/rhs.npy', allow_pickle=False)); assembly = read(output/'assembly.json')
    for block in assembly['blocks']:
        matrix = load_npz(output/block['matrix_file'])
        weights = np.load(output/block['weights_file']) if block.get('weights_file') else block.get('weights', 1.)
        targets = np.load(output/block['rhs_file']) if block.get('rhs_file') else np.zeros((matrix.shape[0], 2))
        rhs += matrix.T @ (np.broadcast_to(weights, (matrix.shape[0],))[:, None]*targets)
    np.save(output/'quadratic/rhs.npy', rhs)
    c = copy.deepcopy(pc); c.pop('solving_authorized', None)
    c.update(source_contract=str(output/'source-contract.json'), source_contract_sha256=sha(output/'source-contract.json'),
        target_intent_sha256=sha(output/'target-intent.json'), target_definition=VARIANT,
        regional_variant=descriptor, execution_status='inputs_prepared_no_solve_no_render')
    save(output/'contract.json', c)
    mutable = {'contract.json', 'source-contract.json', 'target-intent.json', 'data.npz', 'target_deltas.npy', 'quadratic/rhs.npy'}
    report = {'status': 'PreparedRegionalSourceVariant', 'inputs_sha256': hashes,
        'constructed_outputs_sha256': {name: sha(output/name) for name in copied},
        'static_parent_files': {name: sha(parent/name) for name in copied if name not in mutable},
        'regional_variant': descriptor, 'no_solve_no_render': True, 'aesthetic_approval': False}
    if any(sha(Path(name)) != digest for name, digest in hashes.items()): raise ValueError('source changed during adaptation')
    save(output/'report.json', report)
    validation = validate_input(output); save(output/'validation.json', validation)
    print({'output': str(output), 'amount': amount, 'neck_arcs': len(neck_report), 'status': validation['status']})
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['parent', 'audit', 'profile', 'output']: parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args(); prepare(args.parent, args.audit, args.profile, args.output, args.binary)
