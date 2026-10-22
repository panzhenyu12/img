"""Tamper regressions using local frozen inputs, without copying photographs.

The ignored evidence corpus is optional in a clean checkout. Each mutation
uses a private file overlay; original inputs are read-only symlink targets.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from input_contract import ROOT, validate_input


CASES = ROOT / 'test/results/face-source-forward-photo-20261021'
USER = CASES / 'whole-width-signed-direct-v1/+30/input'
WOMAN = CASES / 'woman-whole-width-source-input-v3'


class InputContractTests(unittest.TestCase):
    def setUp(self):
        if not USER.is_dir() or not WOMAN.is_dir():
            self.skipTest('local ignored source-contract evidence is unavailable')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def overlay(self, source):
        destination = Path(self.temporary.name) / 'input'
        for original in source.rglob('*'):
            target = destination / original.relative_to(source)
            if original.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(original)
        return destination

    def replace_array(self, folder, name, values):
        target = folder / name
        target.unlink()
        np.save(target, values)

    def replace_data(self, folder, data):
        target = folder / 'data.npz'
        target.unlink()
        np.savez_compressed(target, **data)

    def refresh_manifest_hash(self, folder, name):
        # Explicitly update only the overlay's builder record to test semantic
        # checks independently of the initial stale-hash rejection.
        manifest = folder / ('report.json' if (folder / 'report.json').exists()
                             else 'signed-input-audit.json')
        contents = json.loads(manifest.read_text())
        key = 'constructed_outputs_sha256' if manifest.name == 'report.json' else 'output_sha256'
        contents[key][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        manifest.unlink()
        manifest.write_text(json.dumps(contents))

    def test_both_real_frozen_inputs_pass_without_loading_photos(self):
        user = validate_input(USER)
        woman = validate_input(WOMAN)
        self.assertEqual((user['source_target_rows'], user['translation_groups']), (6353, 4))
        self.assertEqual((woman['original_background_lines'], woman['original_hard_line_rows']), (8, 723))
        self.assertGreater(user['positive_displacement_Gram_lower_bound'], 0)
        self.assertGreater(woman['positive_displacement_Gram_lower_bound'], 0)

    def test_changed_request_file_is_rejected_against_frozen_hash(self):
        folder = self.overlay(USER)
        values = np.load(folder / 'target_deltas.npy').copy(); values[0, 0] += .25
        self.replace_array(folder, 'target_deltas.npy', values)
        with self.assertRaisesRegex(ValueError, 'frozen input hash differs'):
            validate_input(folder)

    def test_rehashed_target_weight_still_must_match_original_source(self):
        folder = self.overlay(USER)
        data = dict(np.load(folder / 'data.npz')); data['target_weights'][0] += 1.
        self.replace_data(folder, data); self.refresh_manifest_hash(folder, 'data.npz')
        with self.assertRaisesRegex(ValueError, 'signed original target_weights'):
            validate_input(folder)

    def test_rehashed_removed_fixed_pixel_still_fails_frozen_mask(self):
        folder = self.overlay(WOMAN)
        data = dict(np.load(folder / 'data.npz'))
        data['source_fixed_pixels'] = data['source_fixed_pixels'][1:]
        self.replace_data(folder, data); self.refresh_manifest_hash(folder, 'data.npz')
        with self.assertRaisesRegex(ValueError, 'frozen fixed pixels'):
            validate_input(folder)

    def test_rehashed_moving_group_node_cannot_become_free(self):
        folder = self.overlay(WOMAN)
        data = dict(np.load(folder / 'data.npz'))
        node = int(data['group_nodes_0'][0]); data['node_to_variable'][node] = -1
        self.replace_data(folder, data); self.refresh_manifest_hash(folder, 'data.npz')
        with self.assertRaisesRegex(ValueError, 'material shared translation mapping'):
            validate_input(folder)

    def test_missing_required_line_matrix_is_rejected(self):
        folder = self.overlay(WOMAN); (folder / 'hard_line_equalities.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'frozen input hash differs'):
            validate_input(folder)

    def test_rehashed_nonzero_line_rhs_is_rejected(self):
        folder = self.overlay(WOMAN)
        rhs = np.load(folder / 'hard_line_rhs.npy').copy(); rhs[0] = 1.
        self.replace_array(folder, 'hard_line_rhs.npy', rhs)
        self.refresh_manifest_hash(folder, 'hard_line_rhs.npy')
        with self.assertRaisesRegex(ValueError, 'homogeneous background equality'):
            validate_input(folder)

    def test_rehashed_quadratic_rhs_must_match_all_original_blocks(self):
        folder = self.overlay(WOMAN)
        rhs = np.load(folder / 'quadratic/rhs.npy').copy(); rhs[0, 0] += 1.
        self.replace_array(folder, 'quadratic/rhs.npy', rhs)
        self.refresh_manifest_hash(folder, 'quadratic/rhs.npy')
        with self.assertRaisesRegex(ValueError, 'complete source RHS differs'):
            validate_input(folder)


if __name__ == '__main__':
    unittest.main()
