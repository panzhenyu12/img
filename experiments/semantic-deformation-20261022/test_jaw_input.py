"""Focused source-arc behavior; no photographs, solver or system build."""
import copy
import unittest

import numpy as np

from prepare_jaw_input import expected_targets, validate_review, check_controller_output
from input_contract import require_same_signed_amount


def fixture():
    # 32 equal edges on each side of a square. Arc edge 32 -> 64 is the
    # right side; row 48 is its midpoint and must have unit beta gain.
    corners = np.array([[-20., -20.], [20., -20.], [20., 20.], [-20., 20.]])
    contour = np.concatenate([a+np.arange(32)[:, None]/32*(b-a)
                              for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    p = np.vstack([contour, [[2., 3.], [4., 5.]]])
    destination = p.copy(); destination[-2:] += [[.25, -.5], [1., 2.]]
    data = {'target_sources': p, 'target_destinations': destination,
            'target_deltas': destination-p, 'group_priors': np.zeros((1, 2)),
            'group_prior_weights': np.zeros(1)}
    source = {'groups': [{'features': ['leftEye', 'leftBrow'], 'forward_translation_prior': [0., 0.],
                         'translation_prior_weight': 0.}],
              'source_neck_targets': 'none declared in original baseline; none borrowed from another photograph'}
    audit = {'name': 'woman', 'source_contour': contour.tolist(), 'midline_u': 0., 'calibration_k_full': .2,
             'frame_raw': {'origin': [0., 0.], 'across': [1., 0.], 'down': [0., 1.],
                           'bounds': [-20., -20., 20., 20.]}}
    arcs = [{'id': 'right', 'start': {'edge': 32, 't': 0.}, 'end': {'edge': 64, 't': 0.}, 'direction': 1}]
    return data, source, {}, audit, arcs


class JawInputTests(unittest.TestCase):
    def test_known_midpoint_gain_endpoints_and_nonface_rows(self):
        args = fixture(); original = copy.deepcopy(args[0])
        target, gains, ids = expected_targets(*args, 50.)
        np.testing.assert_array_equal(target[48], [18., 0.])
        self.assertEqual(gains[48], 1.)
        self.assertEqual(ids[48], 'right')
        np.testing.assert_array_equal(target[[32, 64]], args[0]['target_sources'][[32, 64]])
        np.testing.assert_array_equal(target[128:], original['target_destinations'][128:])
        for key in original:
            np.testing.assert_array_equal(args[0][key], original[key])

    def test_signs_and_zero_use_original_source_without_touching_other_rows(self):
        args = fixture(); positive, _, _ = expected_targets(*args, 50.)
        negative, _, _ = expected_targets(*args, -50.)
        zero, _, _ = expected_targets(*args, 0.)
        np.testing.assert_array_equal(positive[:128]+negative[:128], 2*args[0]['target_sources'][:128])
        np.testing.assert_array_equal(zero[:128], args[0]['target_sources'][:128])
        np.testing.assert_array_equal(zero[128:], args[0]['target_destinations'][128:])

    def test_other_source_neck_prior_and_material_are_rejected(self):
        for kind in ['source', 'neck', 'prior', 'material']:
            args = list(fixture())
            if kind == 'source': args[3]['name'] = 'hair'
            if kind == 'neck': args[2]['source_neck_support'] = {'added_target_count': 4}
            if kind == 'prior': args[0]['group_priors'][0, 0] = .01
            if kind == 'material': args[1]['groups'][0]['features'] = ['ears']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                expected_targets(*args, 50.)

    def test_unknown_review_reordered_contour_and_upgraded_visibility_rejected(self):
        audit = fixture()[3]; audit.update(source_image='source', input_sha256={'source': 'photo-hash'})
        sidecar = {'source_frame_unchanged': audit['frame_raw'], 'source_contour': audit['source_contour'],
            'automatic_edit_authorized': False, 'visibility_status': 'unknown',
            'sides': {side: {'direction_in_source_row_order': direction, 'boundaries': {
                'mouth_height': {'row0': 32 if side == 'right' else 96, 't': .25},
                'chin_proxy': {'row0': 64, 't': .5}}} for side, direction in [('left', -1), ('right', 1)]}}
        review = {'mode': 'model_reviewed_offline_selection', 'name': 'woman', 'source_sha256': 'photo-hash',
            'sidecar_sha256': 'sidecar-hash', 'notes': ['Synthetic source test, no automatic ownership.'],
            'evidence_images_sha256': {'source-overlay.png': 'image-hash'},
            'selected_arcs': [{'side': side, 'start_boundary': 'mouth_height', 'end_boundary': 'chin_proxy'}
                              for side in ['left', 'right']]}
        arcs = validate_review(review, audit, sidecar, 'sidecar-hash')
        self.assertEqual(arcs[0]['start'], {'edge': 96, 't': .25})
        for kind in ['unknown_review', 'source_order', 'auto_authorized', 'wrong_hash', 'wrong_arc']:
            r, s = copy.deepcopy(review), copy.deepcopy(sidecar)
            if kind == 'unknown_review': r['mode'] = 'inferred_from_nearest_landmark'
            if kind == 'source_order': s['source_contour'][0], s['source_contour'][1] = s['source_contour'][1], s['source_contour'][0]
            if kind == 'auto_authorized': s['automatic_edit_authorized'] = True
            if kind == 'wrong_hash': r['sidecar_sha256'] = 'different-sidecar'
            if kind == 'wrong_arc': r['selected_arcs'][0]['start_boundary'] = 'eye_height'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_review(r, audit, s, 'sidecar-hash')

    def test_mixed_strength_or_signed_parent_is_rejected(self):
        args = fixture(); target, gains, ids = expected_targets(*args, 50.)
        output = {'controller': 'jaw_arc_v1', 'targets': [{'index': i, 'source': p.tolist(),
                  'target': q.tolist(), 'point_gain': float(g), 'arc_id': ids[i]}
                  for i, (p, q, g) in enumerate(zip(args[0]['target_sources'][:128], target[:128], gains))]}
        check_controller_output(output, args[0], target, gains, ids)
        mixed = copy.deepcopy(output); mixed['targets'][48]['target'][0] = 19.
        with self.assertRaisesRegex(ValueError, 'independent polynomial'):
            check_controller_output(mixed, args[0], target, gains, ids)
        for amount in [-50., 25.]:
            with self.assertRaisesRegex(ValueError, 'same signed amount'):
                require_same_signed_amount(50., amount)

if __name__ == '__main__':
    unittest.main()
