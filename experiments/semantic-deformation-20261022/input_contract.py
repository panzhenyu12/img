"""Verify frozen source evidence before changing only the solve subspace.

This is an input audit, not a solver or an image-quality certificate. It reads
existing masks; it never rasterizes a new mask or changes a protection role.
"""
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags, load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/face-source-forward-photo-20261021'))
from adapt import read, sha, square_nodes, point_operators
from source_forward_input import line_constraints


RUST_CONTROLLER_VARIANT = 'rust-whole-face-width-controller-v1'
REGIONAL_CONTROLLER_VARIANT = 'regional-contour-source-v1'
JAW_CONTROLLER_VARIANT = 'jaw-arc-source-v1'


def require_same_signed_amount(parent_amount, requested_amount):
    """A controller substitution retains every other operation's drive."""
    if (not np.isfinite([parent_amount, requested_amount]).all()
            or not -100. <= requested_amount <= 100.
            or parent_amount != requested_amount):
        raise ValueError('controller replacement requires the same signed amount as its frozen parent')


def _same(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise ValueError('source input differs: ' + name)


def _matrix_same(actual, expected, name, tolerance=0.):
    if actual.shape != expected.shape:
        raise ValueError('matrix shape differs: ' + name)
    delta = actual - expected
    error = float(abs(delta.data).max(initial=0.))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError('matrix coefficients differ: ' + name)
    return error


def _hashes(record, base, checked):
    if not isinstance(record, dict) or not record:
        raise ValueError('nonempty frozen hash evidence required')
    for name, digest in record.items():
        path = Path(name)
        path = path if path.is_absolute() else base / path
        if not path.is_file() or sha(path) != digest:
            raise ValueError('frozen input hash differs: ' + str(path))
        checked[str(path.resolve())] = digest


def _validate_regional(inp, c, source, intent, data, checked):
    """Check the new editing intent against its untouched same-amount parent."""
    from prepare_regional_input import (expected_targets, check_target_variant, controller_request,
                                       profile_digest, input_artifacts)
    descriptor = c['regional_variant']
    if source.get('regional_variant') != descriptor or intent.get('regional_variant') != descriptor:
        raise ValueError('regional profile/parent/controller descriptors differ')
    parent = Path(descriptor['parent_input']).resolve()
    if parent == inp or read(parent/'contract.json').get('target_definition') == REGIONAL_CONTROLLER_VARIANT:
        raise ValueError('regional parent must be original source input, not recursive variant')
    parent_validation = validate_input(parent)
    if parent_validation['input_kind'] != 'source_input':
        raise ValueError('regional parent must be original source_input')
    checked.update(parent_validation['verified_input_hashes'])
    if sha(parent/'contract.json') != descriptor['parent_contract_sha256']:
        raise ValueError('regional parent contract differs')
    pc, ps = read(parent/'contract.json'), read(parent/'source-contract.json')
    require_same_signed_amount(pc['requested_amount'], c['requested_amount'])
    for key in ('source_photo', 'source_photo_sha256', 'baseline', 'grid', 'crop', 'target_count',
                'scalar_variables', 'inverse_geometry_gate', 'hard_line_rows', 'background_line_count'):
        if c[key] != pc[key]: raise ValueError('regional original contract changed: '+key)
    for key in ('groups', 'background_lines', 'source_fixed_masks'):
        if source[key] != ps[key]: raise ValueError('regional source role changed: '+key)
    pd = dict(np.load(parent/'data.npz', allow_pickle=False))
    if set(pd) != set(data): raise ValueError('regional original data fields changed')
    for key in pd:
        if key not in ('target_destinations', 'target_deltas'):
            _same(data[key], pd[key], 'regional original '+key)
    mutable = {'contract.json', 'source-contract.json', 'target-intent.json', 'data.npz',
               'target_deltas.npy', 'quadratic/rhs.npy'}
    for name in input_artifacts(parent):
        if name not in mutable and sha(inp/name) != sha(parent/name):
            raise ValueError('regional static source artifact differs: '+name)
    audit_path = Path(descriptor['source_audit']); audit = read(audit_path)
    if (sha(audit_path) != descriptor['source_audit_sha256'] or audit['name'] not in ['hair', 'woman']
            or audit['input_sha256'][audit['source_image']] != pc['source_photo_sha256']):
        raise ValueError('regional source audit identity differs')
    _hashes(audit['input_sha256'], inp, checked)
    checked[str(audit_path)] = sha(audit_path)
    profile = descriptor['profile']
    if profile_digest(profile) != descriptor['profile_sha256']:
        raise ValueError('regional profile digest differs')
    request = controller_request(pd, audit, profile, c['requested_amount'])
    if request != descriptor['controller_request']:
        raise ValueError('regional controller request differs from source/frame/profile')
    binary = Path(descriptor['controller_binary'])
    if sha(binary) != descriptor['controller_binary_sha256']:
        raise ValueError('regional controller binary changed')
    checked[str(binary)] = sha(binary)
    expected, neck = expected_targets(pd, ps, read(Path(pc['baseline'])/'provenance.json'),
                                      audit, profile, c['requested_amount'])
    check_target_variant(data['target_destinations'], expected, neck, descriptor['neck_arcs'])
    output = descriptor['controller_output']
    if output['controller'] != 'regional_contour_v1' or len(output['targets']) != 128:
        raise ValueError('regional controller output identity differs')
    _same(np.asarray([t['source'] for t in output['targets']]), pd['target_sources'][:128], 'regional output sources')
    _same(np.asarray([t['target'] for t in output['targets']]), data['target_destinations'][:128], 'regional output destinations')
    _same(np.asarray(intent['requested']), data['target_destinations'][:128], 'regional intent face targets')
    allowed = set(range(128)) | {i for arc in neck for i in arc['source_rows']}
    untouched = [i for i in range(len(expected)) if i not in allowed]
    _same(data['target_destinations'][untouched], pd['target_destinations'][untouched], 'regional nonedited target rows')


def _validate_jaw(inp, c, source, intent, data, checked):
    """Independently validate an explicitly reviewed source-arc editing intent."""
    from prepare_jaw_input import (ORIGINAL_DEFINITION, validate_review, verify_sidecar,
        expected_targets, check_controller_output, controller_request, input_artifacts)
    descriptor = c['jaw_variant']
    if (descriptor.get('variant') != JAW_CONTROLLER_VARIANT or source.get('jaw_variant') != descriptor
            or intent.get('jaw_variant') != descriptor or descriptor.get('automatic_edit_authorized') is not False):
        raise ValueError('jaw source/review/controller descriptors differ')
    parent = Path(descriptor['parent_input']).resolve()
    if parent == inp or read(parent/'contract.json').get('target_definition') != ORIGINAL_DEFINITION:
        raise ValueError('jaw parent must be original woman source_input')
    parent_validation = validate_input(parent)
    if parent_validation['input_kind'] != 'source_input': raise ValueError('jaw parent is not source_input')
    checked.update(parent_validation['verified_input_hashes'])
    if sha(parent/'contract.json') != descriptor['parent_contract_sha256']:
        raise ValueError('jaw parent contract changed')
    pc, ps = read(parent/'contract.json'), read(parent/'source-contract.json')
    require_same_signed_amount(pc['requested_amount'], c['requested_amount'])
    mutable_contract = {'source_contract', 'source_contract_sha256', 'target_intent_sha256',
        'target_definition', 'jaw_variant', 'execution_status', 'solving_authorized'}
    if {k: v for k, v in c.items() if k not in mutable_contract} != {k: v for k, v in pc.items() if k not in mutable_contract}:
        raise ValueError('jaw original geometry/material contract changed')
    mutable_source = {'targets', 'target_definition', 'jaw_variant', 'source_target_rule', 'solving_authorized'}
    if {k: v for k, v in source.items() if k not in mutable_source} != {k: v for k, v in ps.items() if k not in mutable_source}:
        raise ValueError('jaw original source roles/weights/priors changed')
    pd = dict(np.load(parent/'data.npz', allow_pickle=False))
    if set(data) != set(pd): raise ValueError('jaw original data fields changed')
    for key in pd:
        if key not in ('target_destinations', 'target_deltas'):
            _same(data[key], pd[key], 'jaw original '+key)
        else:
            _same(data[key][128:], pd[key][128:], 'jaw non-face '+key)
    mutable_files = {'contract.json', 'source-contract.json', 'target-intent.json', 'data.npz',
                     'target_deltas.npy', 'quadratic/rhs.npy'}
    for name in input_artifacts(parent):
        if name not in mutable_files and sha(inp/name) != sha(parent/name):
            raise ValueError('jaw static source artifact changed: '+name)
    audit_path, sidecar_path, review_path = [Path(descriptor[key]) for key in
        ['source_audit', 'source_sidecar', 'source_review']]
    for key, path in [('source_audit', audit_path), ('source_sidecar', sidecar_path), ('source_review', review_path)]:
        if sha(path) != descriptor[key+'_sha256']: raise ValueError('jaw frozen '+key+' differs')
        checked[str(path)] = sha(path)
    audit, sidecar, review = read(audit_path), read(sidecar_path), read(review_path)
    if (audit['name'] != 'woman' or audit['input_sha256'][audit['source_image']] != pc['source_photo_sha256']
            or sidecar['provenance']['input_sha256'].get(str(audit_path)) != sha(audit_path)):
        raise ValueError('jaw frozen source audit/calibration differs')
    _hashes(audit['input_sha256'], inp, checked)
    _hashes(sidecar['provenance']['input_sha256'], inp, checked)
    _hashes(review['evidence_images_sha256'], inp, checked)
    provenance = read(Path(pc['baseline'])/'provenance.json')
    graph = Path(provenance['source_trace']).parent/'graph.json'
    verify_sidecar(audit, sidecar, graph)
    arcs = validate_review(review, audit, sidecar, sha(sidecar_path))
    expected, gains, ids = expected_targets(pd, ps, provenance, audit, arcs, c['requested_amount'])
    if (not np.isfinite(data['target_destinations']).all()
            or np.max(abs(data['target_destinations']-expected), initial=0.) > 1e-10):
        raise ValueError('jaw targets differ from independent source-arc polynomial')
    request = controller_request(pd, audit, arcs, c['requested_amount'])
    if descriptor['controller_request'] != request:
        raise ValueError('jaw request differs from reviewed arcs/source calibration/signed amount')
    binary = Path(descriptor['controller_binary'])
    if sha(binary) != descriptor['controller_binary_sha256']: raise ValueError('jaw controller binary changed')
    checked[str(binary)] = sha(binary)
    output = descriptor['controller_output']
    check_controller_output(output, pd, expected, gains, ids)
    _same(np.asarray([row['target'] for row in output['targets']]), data['target_destinations'][:128], 'jaw Rust output targets')
    _same(np.asarray(intent['source']), pd['target_sources'][:128], 'jaw intent source')
    _same(np.asarray(intent['requested']), data['target_destinations'][:128], 'jaw intent targets')


def _validate_input(inp):
    started = time.perf_counter()
    inp = Path(inp).resolve()
    checked = {}
    if ((inp / 'report.json').is_file()
            and all(key in read(inp / 'report.json') for key in
                    ('constructed_outputs_sha256', 'inputs_sha256'))):
        evidence = read(inp / 'report.json')
        _hashes(evidence['constructed_outputs_sha256'], inp, checked)
        _hashes(evidence['inputs_sha256'], inp, checked)
        kind = 'source_input'
    elif (inp / 'signed-input-audit.json').is_file():
        evidence = read(inp / 'signed-input-audit.json')
        _hashes(evidence['output_sha256'], inp, checked)
        _hashes(evidence['parent_inputs_sha256'], inp, checked)
        kind = 'signed_user_input'
    else:
        raise ValueError('frozen source-input report or signed-input audit required')
    c = read(inp / 'contract.json')
    source = read(inp / 'source-contract.json')
    manifest = read(inp / 'assembly.json')
    data = dict(np.load(inp / 'data.npz', allow_pickle=False))
    if sha(inp / 'source-contract.json') != c['source_contract_sha256']:
        raise ValueError('source contract digest differs')
    photo = Path(c['source_photo'])
    if not photo.is_file() or sha(photo) != c['source_photo_sha256']:
        raise ValueError('source photograph digest differs')
    checked[str(photo.resolve())] = c['source_photo_sha256']
    base = Path(c['baseline'])
    if str((base / 'provenance.json').resolve()) not in checked:
        raise ValueError('original material provenance is not hash-bound')

    if kind == 'signed_user_input':
        parent = Path(c['parent_input'])
        if sha(parent / 'contract.json') != c['parent_input_contract_sha256']:
            raise ValueError('signed parent contract differs')
        original_data = dict(np.load(parent / 'data.npz', allow_pickle=False))
        if set(data) != set(original_data):
            raise ValueError('signed source data fields changed')
        for name in data:
            if name not in ['target_destinations', 'target_deltas', 'group_priors']:
                _same(data[name], original_data[name], 'signed original ' + name)
        for name, identical in evidence['static_files_byte_identical'].items():
            if identical is not True or sha(inp / name) != sha(parent / name):
                raise ValueError('signed original static file differs: ' + name)
        if sha(inp / 'proposal/hessian.npz') != sha(parent / 'proposal/hessian.npz'):
            raise ValueError('signed original Hessian differs')
        amount = source['explicit_target_redefinition']['amount']
        old_priors = source['old_inverse_translation_priors']
        priors = -np.asarray([g['delta'] for g in old_priors], float)
        prior_weights = np.asarray([g['target_weight'] for g in old_priors], float)
    else:
        frozen = dict(np.load(inp / 'source-evidence.npz', allow_pickle=False))
        _same(data['source_fixed_pixels'], frozen['source_fixed_pixels'], 'frozen fixed pixels')
        for i in range(len(source['groups'])):
            _same(data['group_nodes_' + str(i)], frozen['group_nodes_' + str(i)], 'frozen material support')
        original_provenance = read(base / 'provenance.json')
        if read(inp / 'original-provenance.json') != original_provenance:
            raise ValueError('copied original material provenance differs')
        amount = source['requested_amount']
        priors = np.asarray([g['forward_translation_prior'] for g in source['groups']], float).reshape(-1, 2)
        prior_weights = np.asarray([g['translation_prior_weight'] for g in source['groups']], float)
    if amount != c['requested_amount'] or not np.isfinite(amount):
        raise ValueError('declared signed amount differs')
    intent = read(inp / 'target-intent.json')
    if intent['amount'] != amount or sha(inp / 'target-intent.json') != c['target_intent_sha256']:
        raise ValueError('target intent differs')
    variant_names = (c.get('target_definition'), source.get('target_definition'), intent.get('name'))
    if JAW_CONTROLLER_VARIANT in variant_names:
        if kind != 'source_input' or any(name != JAW_CONTROLLER_VARIANT for name in variant_names):
            raise ValueError('jaw variant declarations differ')
        _validate_jaw(inp, c, source, intent, data, checked)
    if REGIONAL_CONTROLLER_VARIANT in variant_names:
        if kind != 'source_input' or any(name != REGIONAL_CONTROLLER_VARIANT for name in variant_names):
            raise ValueError('regional controller variant declarations differ')
        _validate_regional(inp, c, source, intent, data, checked)
    if RUST_CONTROLLER_VARIANT in variant_names:
        if kind != 'signed_user_input' or any(name != RUST_CONTROLLER_VARIANT for name in variant_names):
            raise ValueError('controller variant declarations differ')
        require_same_signed_amount(read(parent / 'contract.json')['requested_amount'], amount)
        for name in ('target_destinations', 'target_deltas'):
            _same(data[name][128:], original_data[name][128:], 'controller non-face ' + name)
        _same(data['group_priors'], original_data['group_priors'], 'controller attachment priors')
        for ledger in (source, intent):
            if ledger.get('controller_request', {}).get('amount') != amount:
                raise ValueError('controller request amount differs')

    targets = source['targets']
    if [v['original_index'] for v in targets] != list(range(len(targets))):
        raise ValueError('original target rows missing or reordered')
    for key, name in [('target_sources', 'source'), ('target_destinations', 'destination'),
                      ('target_deltas', 'forward_delta'), ('target_weights', 'weight')]:
        _same(data[key], np.asarray([v[name] for v in targets]), name + ' target ledger')
        if not np.isfinite(data[key]).all():
            raise ValueError('nonfinite target data')
    if len(targets) != c['target_count'] or np.any(data['target_weights'] < 0):
        raise ValueError('target count or weights invalid')
    if not np.allclose(data['target_sources'] + data['target_deltas'], data['target_destinations'], rtol=0, atol=1e-10):
        raise ValueError('source displacement ledger inconsistent')
    _same(data['group_priors'], priors, 'translation priors')
    _same(data['group_prior_weights'], prior_weights, 'translation prior weights')
    for name in ['target_deltas', 'target_weights', 'group_priors', 'group_prior_weights']:
        _same(np.load(inp / (name + '.npy'), allow_pickle=False), data[name], name + ' residual array')

    grid = c['grid']; width, height = grid['width'], grid['height']
    mapping = data['node_to_variable']; nvars = manifest['nvars']
    if (mapping.dtype.kind not in 'iu' or mapping.shape != (width * height,)
            or np.any(mapping < -1) or np.any(mapping >= nvars)
            or nvars != c['scalar_variables']):
        raise ValueError('original source variable mapping invalid')
    counts = np.bincount(mapping[mapping >= 0], minlength=nvars)
    group_ids = data['group_variable_ids']
    if (group_ids.dtype.kind not in 'iu' or len(group_ids) != len(source['groups'])
            or len(np.unique(group_ids)) != len(group_ids) or np.any(group_ids < 0)
            or np.any(group_ids >= nvars) or np.any(counts == 0)):
        raise ValueError('original material variable identities invalid')
    ordinary = np.ones(nvars, bool); ordinary[group_ids] = False
    if np.any(counts[ordinary] != 1):
        raise ValueError('undeclared source variables share material motion')
    fixed = data['source_fixed_pixels']
    if not np.isfinite(fixed).all() or not np.array_equal(fixed, np.round(fixed)):
        raise ValueError('complete integer source fixed pixels required')
    fixed_nodes = square_nodes(fixed, grid, identity_extension=True)
    if np.any(mapping[fixed_nodes] != -1) or np.any(mapping[data['preserved_context_nodes']] != -1):
        raise ValueError('original fixed source/context support unlocked')
    border = mapping.reshape(height, width)
    if np.any(border[:2] != -1) or np.any(border[-2:] != -1) or np.any(border[:, :2] != -1) or np.any(border[:, -2:] != -1):
        raise ValueError('original two-node identity collar changed')
    if kind == 'source_input' and np.any(mapping[frozen['locked_nodes']] != -1):
        raise ValueError('frozen source locks changed')
    for i, (group, variable) in enumerate(zip(source['groups'], group_ids)):
        expected = square_nodes(np.asarray(group['source_material'], float), grid)
        _same(data['group_nodes_' + str(i)], expected, 'complete material source support')
        _same(np.flatnonzero(mapping == variable), expected, 'material shared translation mapping')
    active = np.flatnonzero(mapping >= 0)
    prolong = coo_matrix((np.ones(len(active)), (active, mapping[active])), shape=(len(mapping), nvars)).tocsc()
    _matrix_same(load_npz(inp / 'prolongation.npz'), prolong, 'source prolongation')
    target_matrix, _, _ = point_operators(data['target_sources'], grid, mapping, nvars)
    _matrix_same(load_npz(inp / 'target.npz'), target_matrix, 'source targets')

    controls_path = Path(source.get('old_controls', str(base / 'controls.json')))
    if str(controls_path.resolve()) not in checked:
        raise ValueError('original controls are not hash-bound')
    original_lines = read(controls_path).get('background_lines', [])
    if source.get('background_lines', original_lines) != original_lines:
        raise ValueError('original background lines changed')
    declared = manifest.get('hard_equalities', [])
    has_lines = bool(original_lines) or bool(c.get('hard_line_rows', 0)) or bool(declared)
    line_rows = 0
    if has_lines:
        lines = load_npz(inp / 'hard_line_equalities.npz')
        line_rhs = np.load(inp / 'hard_line_rhs.npy', allow_pickle=False)
        expected, _ = line_constraints(original_lines, grid, mapping, nvars)
        _matrix_same(lines, expected, 'all original background lines')
        if (len(declared) != 1 or declared[0]['matrix_file'] != 'hard_line_equalities.npz'
                or declared[0]['rhs_file'] != 'hard_line_rhs.npy'
                or declared[0]['rows'] != lines.shape[0]
                or c.get('hard_line_rows') != lines.shape[0]
                or c.get('background_line_count') != len(original_lines)
                or line_rhs.shape != (lines.shape[0],) or np.any(line_rhs != 0)):
            raise ValueError('complete homogeneous background equality declaration required')
        line_rows = lines.shape[0]
    elif (inp / 'hard_line_equalities.npz').exists() or (inp / 'hard_line_rhs.npy').exists():
        raise ValueError('undeclared background equality artifact')

    folder = inp / ('quadratic' if kind == 'source_input' else 'proposal')
    h = load_npz(folder / 'hessian.npz').tocsc()
    rhs = np.load(folder / 'rhs.npy', allow_pickle=False)
    if h.shape != (nvars, nvars) or rhs.shape != (nvars, 2):
        raise ValueError('source objective dimensions differ')
    rebuilt_h = csc_matrix(h.shape); rebuilt_rhs = np.zeros_like(rhs)
    names = set(); positive_floor = None
    for block in manifest['blocks']:
        if block['name'] in names:
            raise ValueError('duplicate source energy block')
        names.add(block['name'])
        matrix = load_npz(inp / block['matrix_file']).tocsc()
        weights = (np.load(inp / block['weights_file'], allow_pickle=False) if block.get('weights_file')
                   else block.get('weights', 1.))
        weights = np.broadcast_to(np.asarray(weights, float), (matrix.shape[0],))
        wanted = (np.load(inp / block['rhs_file'], allow_pickle=False) if block.get('rhs_file')
                  else np.zeros((matrix.shape[0], 2)))
        if (matrix.shape[1] != nvars or wanted.shape != (matrix.shape[0], 2)
                or not all(np.isfinite(v).all() for v in [matrix.data, weights, wanted]) or np.any(weights < 0)):
            raise ValueError('invalid original residual block')
        gram = matrix.T @ diags(weights) @ matrix
        rebuilt_h += gram; rebuilt_rhs += matrix.T @ (weights[:, None] * wanted)
        if block['name'] == 'source_displacement':
            _matrix_same(matrix, prolong, 'positive displacement source support')
            off_diagonal = gram - diags(gram.diagonal())
            if off_diagonal.nnz:
                raise ValueError('source displacement Gram must be diagonal')
            positive_floor = float(gram.diagonal().min())
    if positive_floor is None or positive_floor <= 0:
        raise ValueError('positive source displacement Gram required')
    error_h = _matrix_same(h, rebuilt_h, 'complete source Hessian', 1e-12 * max(1., float(abs(h.data).max(initial=0))))
    error_b = float(abs(rhs - rebuilt_rhs).max(initial=0.))
    if not np.isfinite(error_b) or error_b > 1e-12 * max(1., float(abs(rhs).max(initial=0.))):
        raise ValueError('complete source RHS differs from residual blocks')
    return {'status': 'FrozenInputValidated', 'input_kind': kind, 'input': str(inp),
            'verified_input_hashes': checked, 'source_target_rows': len(targets),
            'fixed_source_pixels': len(fixed), 'translation_groups': len(group_ids),
            'original_background_lines': len(original_lines), 'original_hard_line_rows': line_rows,
            'full_scalar_variables': nvars, 'source_amount': amount,
            'complete_H_reassembly_max_error': error_h, 'complete_rhs_reassembly_max_error': error_b,
            'positive_displacement_Gram_lower_bound': positive_floor,
            'seconds': time.perf_counter() - started, 'no_mask_rerasterization': True,
            'no_solve_or_image_quality_claim': True}


def validate_input(inp):
    """Return audited source provenance; reject missing or inconsistent inputs."""
    try:
        return _validate_input(inp)
    except (OSError, KeyError, TypeError, IndexError) as error:
        raise ValueError('invalid or incomplete frozen source input: ' + str(error)) from error
