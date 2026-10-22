"""Freeze the third photograph's own signed source inputs, without solving.

The historical builder is reused unchanged. Each amount selects its own old
567-row baseline, so its 18 signed neck targets are never borrowed from the
opposite amount. Missing historical oracle files are explicitly recorded;
new masks are frozen from this photograph and its existing later RGB audit.
"""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version
from scipy.ndimage import distance_transform_edt

from input_contract import validate_input
from reduced_field import ROOT
from adapt import read, save, sha
from source_forward_input import prepare, resolve

RELATIVE = ROOT/'test/results/face-source-relative-width-20261020'
APPLICABILITY = RELATIVE/'whole-face-width-source-applicability-v2/hair-source-only.json'
POSITIVE_BASE = RELATIVE/'hair-+60-contract'
NEGATIVE_BASE = RELATIVE/'hair--60-offline/contract'
POSITIVE_AUDIT = RELATIVE/'hair-+60-support2-solve/image-material-audit.json'
NEGATIVE_AUDIT = RELATIVE/'hair--60-offline/solve-2/image-material-audit.json'


def _target_arrays(controls):
    rows = controls['targets']
    destinations = np.asarray([row['position'] for row in rows], float)
    source = destinations+np.asarray([row['delta'] for row in rows], float)
    weights = np.asarray([row['weight'] for row in rows], float)
    if not all(np.isfinite(v).all() for v in (source, destinations, weights)):
        raise ValueError('finite original target evidence required')
    return source, destinations, weights


def check_signed_target_evidence(positive_controls, positive_provenance,
                                 negative_controls, negative_provenance, amount):
    """Reject the face-negative / neck-positive mix before input construction."""
    pairs = [(positive_controls, positive_provenance), (negative_controls, negative_provenance)]
    amounts = [float(p['source_relative_width']['amount']) for _, p in pairs]
    if amounts[0] <= 0 or amounts[1] != -amounts[0] or amount not in amounts:
        raise ValueError('one exact signed baseline is required for the requested amount')
    arrays = [_target_arrays(controls) for controls, _ in pairs]
    if not np.array_equal(arrays[0][0], arrays[1][0]):
        raise ValueError('signed baselines have different original source target coordinates')
    if not np.array_equal(arrays[0][2], arrays[1][2]):
        raise ValueError('signed baselines changed original target weights')
    necks = [p['source_neck_support'] for _, p in pairs]
    start = necks[0]['base_target_count']; count = necks[0]['added_target_count']
    if type(start) is not int or type(count) is not int or start < 0 or count < 1:
        raise ValueError('explicit positive neck row range required')
    for index, neck in enumerate(necks):
        source, destinations, _ = arrays[index]
        if (neck['base_target_count'] != start or neck['added_target_count'] != count
                or len(source) != start+count
                or not np.array_equal(source[start:], np.asarray(neck['source'], float))):
            raise ValueError('signed neck source rows do not match their own recorded range')
        if not np.array_equal(destinations[start:], np.asarray(neck['requested'], float)):
            raise ValueError('signed neck destinations differ from their own recorded evidence')
    for key in ['source_photo_sha256', 'graph_sha256']:
        if necks[0][key] != necks[1][key]:
            raise ValueError('neck evidence belongs to different photographs or source graphs')
    residual = (arrays[0][1][start:]-arrays[0][0][start:]
                + arrays[1][1][start:]-arrays[1][0][start:])
    if np.any(residual != 0.):
        raise ValueError('opposite signed neck displacement evidence is inconsistent')
    return {'selected_amount': float(amount), 'available_amounts': amounts,
            'original_target_rows': start+count, 'neck_row_start': start,
            'neck_rows': count, 'all_original_sources_identical': True,
            'all_original_weights_identical': True,
            'signed_neck_displacement_max_error': float(abs(residual).max()),
            'neck_target_policy': 'select unchanged neck rows from the exact signed original baseline'}


def _freeze_oracle(base, audit_path, output):
    if pillow_version != '11.1.0':
        raise ValueError('this evidence requires the recorded Pillow 11.1.0 polygon runtime')
    p = read(base/'provenance.json'); audit = read(audit_path)
    source, image, graph_path, hard = resolve(base, p)
    if sha(source) != audit['source_sha256'] or sha(hard) != audit['classifier_hard_sha256']:
        raise ValueError('later RGB audit does not belong to the same source and classifier')
    graph = read(graph_path)['operations'][0]['params']; iw, ih = image.size
    left, top, width, height = p['crop']; arrays = {}; checks = {}
    for name in ['nose', 'lips']:
        points = [(v['x']*iw, v['y']*ih) for v in graph['semanticMasks'][name]['points']]
        mask = Image.new('L', (iw, ih)); ImageDraw.Draw(mask).polygon(points, fill=1)
        full = np.asarray(mask) > 0
        mask = Image.new('L', (width, height))
        ImageDraw.Draw(mask).polygon([(x-left, y-top) for x, y in points], fill=255)
        core = distance_transform_edt(np.asarray(mask) > 0) >= 2.
        expected_full = audit['fixed'][name]['pixels']
        expected_core = p['feature_cores'][name]['core_pixels']
        if int(full.sum()) != expected_full or int(core.sum()) != expected_core:
            raise ValueError(name+': source polygon/core does not reproduce recorded same-source counts')
        arrays[name+'_full_polygon'] = full
        arrays[name+'_reconstructed_core'] = np.zeros((ih, iw), bool)
        arrays[name+'_reconstructed_core'][top:top+height, left:left+width] = core
        checks[name] = {'later_audit_full_pixels': expected_full, 'new_frozen_full_pixels': int(full.sum()),
                        'baseline_core_pixels': expected_core, 'new_frozen_core_pixels': int(core.sum())}
    missing = Path(p['feature_transport']['comparison'])/'image-material-audit.json'
    if missing.exists():
        raise ValueError('recorded historical audit now exists; re-review oracle provenance before adaptation')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'masks.npz', **arrays)
    report = {'source': str(source), 'source_sha256': sha(source),
              'graph': str(graph_path), 'graph_sha256': sha(graph_path),
              # These two field names are the existing builder's input schema.
              # They point to the explicit later audit, not the missing path.
              'original_audit': str(audit_path), 'original_audit_sha256': sha(audit_path),
              'audit_evidence_kind': 'existing_later_same_photo_rgb_audit',
              'missing_recorded_historical_audit': str(missing), 'historical_audit_present': False,
              'missing_historical_audit_recreated': False, 'historical_frozen_masks_claimed': False,
              'base_provenance_sha256': sha(base/'provenance.json'), 'source_dimensions': [iw, ih],
              'crop': p['crop'], 'python_executable': sys.executable, 'pillow_version': pillow_version,
              'script_sha256': sha(Path(__file__)), 'masks_sha256': sha(output/'masks.npz'),
              'checks': checks,
              'generation': 'newly frozen original source polygons with Pillow 11.1.0; checked against own later RGB audit and baseline core counts',
              'no_pixel_added_removed_or_count_matching_heuristic': True,
              'no_solver_no_model_no_photo_edit': True}
    save(output/'report.json', report)
    return report


def prepare_signed_hair(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite source-input evidence: '+str(output))
    applicability = read(APPLICABILITY)
    if Path(applicability['base']).resolve() != POSITIVE_BASE.resolve():
        raise ValueError('source applicability no longer references the reviewed positive baseline')
    hashes = dict(applicability['inputs_sha256'])
    for path, digest in hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError('frozen hair applicability evidence changed: '+path)
    pairs = [(read(base/'controls.json'), read(base/'provenance.json'))
             for base in (POSITIVE_BASE, NEGATIVE_BASE)]
    for amount in [-60., 60.]:
        check_signed_target_evidence(*pairs[0], *pairs[1], amount)
    # Verify each baseline's own earlier source references before freezing the
    # current positive/negative controls as explicit adapter inputs.
    for base, (_, provenance) in zip((POSITIVE_BASE, NEGATIVE_BASE), pairs):
        width = provenance['source_relative_width']; ancestor = Path(width['baseline'])
        for path, expected in [(ancestor/'controls.json', width['baseline_controls_sha256']),
                               (ancestor/'provenance.json', width['baseline_provenance_sha256']),
                               (Path(width['contract']), width['contract_sha256'])]:
            if sha(path) != expected:
                raise ValueError('signed original source evidence changed: '+str(path))
            hashes[str(path)] = expected
        for path in [base/'controls.json', base/'provenance.json', base/'target-comparison.json']:
            hashes[str(path)] = sha(path)
    for path in [APPLICABILITY, POSITIVE_AUDIT, NEGATIVE_AUDIT, Path(__file__)]:
        hashes[str(path)] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for amount, base, audit in [(60., POSITIVE_BASE, POSITIVE_AUDIT), (-60., NEGATIVE_BASE, NEGATIVE_AUDIT)]:
        folder = output/('plus60' if amount > 0 else 'minus60'); folder.mkdir()
        signed = check_signed_target_evidence(*pairs[0], *pairs[1], amount)
        oracle = _freeze_oracle(base, audit, folder/'oracle')
        own_applicability = copy.deepcopy(applicability)
        own_applicability.update(base=str(base), inputs_sha256=hashes,
                                signed_neck_evidence=signed, parent_applicability=str(APPLICABILITY))
        app_path = folder/'applicability.json'; save(app_path, own_applicability)
        destination = folder/'input'
        # Selecting the same-amount baseline keeps every non-face row intact.
        prepare(app_path, destination, folder/'oracle', amount_override=None)
        source = read(destination/'source-contract.json'); contract = read(destination/'contract.json')
        report = read(destination/'report.json'); intent = read(destination/'target-intent.json')
        selected_controls = pairs[0][0] if amount > 0 else pairs[1][0]
        original_source, original_dest, original_weights = _target_arrays(selected_controls)
        data = dict(np.load(destination/'data.npz', allow_pickle=False))
        if (not np.array_equal(data['target_sources'], original_source)
                or not np.array_equal(data['target_weights'], original_weights)
                or not np.array_equal(data['target_destinations'][128:], original_dest[128:])):
            raise ValueError('hair adapter changed original non-face targets, source coordinates or weights')
        source.update(source_neck_targets={'evidence': str(base/'provenance.json'), **signed},
                      original_rgb_oracle_raster=oracle,
                      source_target_rule='first 128 face targets use the audited whole-face intent; all other 439 rows including 18 neck targets retain their exact same-amount baseline destination and weight')
        intent['signed_neck_evidence'] = signed
        save(destination/'source-contract.json', source); save(destination/'target-intent.json', intent)
        contract.update(source_contract_sha256=sha(destination/'source-contract.json'),
                        target_intent_sha256=sha(destination/'target-intent.json'), signed_neck_evidence=signed)
        save(destination/'contract.json', contract)
        report.update(adapter_script_sha256=sha(Path(__file__)), signed_neck_evidence=signed,
                      newly_frozen_oracle=oracle,
                      source_limits=[
                          'This photograph retains its own 18 signed neck targets and 2 complete eye/brow material groups; no ear material is invented.',
                          'The recorded historical RGB audit path is missing. New source masks use Pillow 11.1.0 and the explicitly bound later audit; this does not recreate the missing audit.',
                          'No source segmentation truth, solved-field validity or aesthetic approval is inferred from input validation.'])
        report['inputs_sha256'].update(hashes)
        report['constructed_outputs_sha256']['target-intent.json'] = sha(destination/'target-intent.json')
        for name in report['constructed_outputs_sha256']:
            report['constructed_outputs_sha256'][name] = sha(destination/name)
        save(destination/'report.json', report)
        validation = validate_input(destination)
        save(destination/'validation.json', validation)
        summaries.append({'amount': amount, 'input': str(destination), 'oracle': str(folder/'oracle'),
                          'status': validation['status'], 'source_target_rows': validation['source_target_rows'],
                          'fixed_source_pixels': validation['fixed_source_pixels'],
                          'translation_groups': validation['translation_groups'],
                          'scalar_variables': validation['full_scalar_variables'],
                          'original_background_lines': validation['original_background_lines'],
                          'signed_neck_evidence': signed})
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('original evidence changed during hair input construction')
    result = {'cases': summaries, 'original_inputs_sha256': hashes,
              'original_evidence_unchanged': True, 'no_solve_no_render': True,
              'newly_frozen_source_oracles_not_historical_masks': True, 'aesthetic_approval': False}
    save(output/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare_signed_hair(args.output)
    print({'output': str(args.output.resolve()), 'cases': result['cases'], 'no_solve_no_render': True})
