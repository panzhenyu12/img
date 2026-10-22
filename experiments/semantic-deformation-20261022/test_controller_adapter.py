"""Source-only adapter regressions; no image decode, deformation solve or build.

The optional local frozen corpus supplies real neck/ear attachment constraints.
Controller invocation is replaced by an exact signed translation to isolate
adapter behavior from Rust target generation, which has its own behavior tests.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from input_contract import ROOT, validate_input


SPEC = importlib.util.spec_from_file_location(
    'controller_adapter', Path(__file__).parent / 'rust-controls/prepare_source_forward_input.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
CASES = ROOT / 'test/results/face-source-forward-photo-20261021/whole-width-signed-direct-v1'


def controller_fixture(binary, request):
    shift = np.asarray(request['frame']['across']) * (-request['amount'] * .01)
    return {'targets': [dict(row, target=(np.asarray(row['source']) + shift).tolist())
                        for row in request['contour']]}


class ControllerAdapterTests(unittest.TestCase):
    def setUp(self):
        if not (CASES / '+60/input').is_dir() or not adapter.DEFAULT_AUDIT.is_file():
            self.skipTest('local ignored source-contract evidence is unavailable')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / 'controller-test-fixture'
        self.binary.write_text('fixture: invocation is patched; no executable is run')
        self.invocation = patch.object(adapter, 'invoke', side_effect=controller_fixture).start()
        self.addCleanup(patch.stopall)

    def prepare(self, parent=None, amount=60., output='prepared'):
        return adapter.prepare(parent or CASES / '+60/input', self.root / output,
                               adapter.DEFAULT_AUDIT, self.binary, amount)

    def overlay(self, source):
        destination = self.root / 'parent-overlay'
        for original in source.rglob('*'):
            target = destination / original.relative_to(source)
            if original.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(original)
        return destination

    def refresh_hashes(self, folder):
        # Keep all bookkeeping internally consistent: semantics must reject
        # the mutation independently of accidental stale digest detection.
        audit = adapter.read(folder / 'signed-input-audit.json')
        for path in audit['output_sha256']:
            audit['output_sha256'][path] = adapter.sha(Path(path))
        adapter.save(folder / 'signed-input-audit.json', audit)
        contract = adapter.read(folder / 'contract.json')
        for name, field in [('source-contract.json', 'source_contract_sha256'),
                            ('target-intent.json', 'target_intent_sha256'),
                            ('signed-input-audit.json', 'signed_input_audit_sha256')]:
            contract[field] = adapter.sha(folder / name)
        adapter.save(folder / 'contract.json', contract)

    def test_different_signed_amount_rejected_before_output_or_controller(self):
        for amount in (60., 0., -60.):
            with self.subTest(amount=amount):
                output = f'wrong-{amount}'
                with self.assertRaisesRegex(ValueError, 'same signed amount'):
                    self.prepare(CASES / '+30/input', amount, output)
                self.assertFalse((self.root / output).exists())
        self.invocation.assert_not_called()

    def test_invalid_parent_rejected_before_output_or_controller(self):
        parent = self.overlay(CASES / '+60/input')
        target = parent / 'target_deltas.npy'
        values = np.load(target).copy(); values[0, 0] += .5
        target.unlink(); np.save(target, values)
        with self.assertRaisesRegex(ValueError, 'frozen input hash differs'):
            self.prepare(parent)
        self.assertFalse((self.root / 'prepared').exists())
        self.invocation.assert_not_called()

    def test_same_signed_parent_preserves_neck_attachments_and_copies_only_inputs(self):
        parent = self.overlay(CASES / '+60/input')
        (parent / 'candidate').mkdir()
        (parent / 'candidate/renderer').write_bytes(b'not an input')
        np.save(parent / 'proposal/solution.npy', np.ones((2, 2)))
        (parent / 'proposal/report.json').write_text('{"status":"Solved"}')
        (parent / 'source-renderer').write_bytes(b'not an input')
        report = self.prepare(parent)
        out = self.root / 'prepared'
        before = dict(np.load(parent / 'data.npz')); after = dict(np.load(out / 'data.npz'))
        for name in ['target_destinations', 'target_deltas']:
            np.testing.assert_array_equal(after[name][128:], before[name][128:])
        for name in ['group_priors', 'group_prior_weights', 'target_weights', 'source_fixed_pixels']:
            np.testing.assert_array_equal(after[name], before[name])
        self.assertEqual(validate_input(out)['source_amount'], 60.)
        for name in ['candidate', 'proposal/solution.npy', 'proposal/report.json', 'source-renderer',
                     'pre-metadata-correction']:
            self.assertFalse((out / name).exists(), name)
        self.assertNotIn('solving_authorized', adapter.read(out / 'contract.json'))
        self.assertNotIn('solving_authorized', report)
        self.assertEqual(report['execution_status'], 'inputs_prepared_no_solve_no_render')
        signed = adapter.read(out / 'signed-input-audit.json')
        self.assertEqual(set(signed['cases']), {'-60', '0', '60'})

    def test_rehashed_uniform_labels_cannot_hide_mixed_parent_strength(self):
        self.prepare(amount=60.)
        out = self.root / 'prepared'
        contract = adapter.read(out / 'contract.json'); contract['requested_amount'] = 30.
        adapter.save(out / 'contract.json', contract)
        source = adapter.read(out / 'source-contract.json')
        source['explicit_target_redefinition']['amount'] = 30.
        source['controller_request']['amount'] = 30.
        adapter.save(out / 'source-contract.json', source)
        intent = adapter.read(out / 'target-intent.json'); intent['amount'] = 30.
        intent['controller_request']['amount'] = 30.
        adapter.save(out / 'target-intent.json', intent)
        self.refresh_hashes(out)
        with self.assertRaisesRegex(ValueError, 'same signed amount'):
            validate_input(out)

    def test_rehashed_nonface_drive_must_match_same_signed_parent(self):
        for key in ('target_destinations', 'group_priors'):
            with self.subTest(key=key):
                output = 'changed-' + key
                self.prepare(output=output)
                out = self.root / output
                data = dict(np.load(out / 'data.npz'))
                source = adapter.read(out / 'source-contract.json')
                if key == 'target_destinations':
                    index = len(data['target_sources']) - 1
                    data[key][index, 0] += .25
                    data['target_deltas'][index] = data[key][index] - data['target_sources'][index]
                    source['targets'][index]['destination'] = data[key][index].tolist()
                    source['targets'][index]['forward_delta'] = data['target_deltas'][index].tolist()
                    np.save(out / 'target_deltas.npy', data['target_deltas'])
                else:
                    data[key][0, 0] += .25
                    source['old_inverse_translation_priors'][0]['delta'] = (-data[key][0]).tolist()
                    source['explicit_target_redefinition']['signed_translation_priors'] = data[key].tolist()
                    np.save(out / 'group_priors.npy', data[key])
                np.savez_compressed(out / 'data.npz', **data)
                adapter.save(out / 'source-contract.json', source)
                adapter.rebuild_quadratic(out, data['target_deltas'])
                self.refresh_hashes(out)
                with self.assertRaisesRegex(ValueError, 'controller (non-face|attachment)'):
                    validate_input(out)


if __name__ == '__main__':
    unittest.main()
