"""Regional targets and attached neck must share the declared profile."""
import copy
import unittest

import numpy as np

from prepare_regional_input import expected_targets, check_target_variant


class RegionalInputTests(unittest.TestCase):
    def fixture(self):
        t = np.linspace(0, 2*np.pi, 128, endpoint=False)
        contour = np.column_stack([50+30*np.cos(t), 50+50*np.sin(t)])
        neck = np.array([[75., 70.], [78., 90.], [80., 110.]])
        points = np.vstack([contour, [[40., 45.]], neck])
        delta = np.zeros_like(points); delta[-3:, 0] = [-3., -1.5, 0.]
        data = {'target_sources': points, 'target_destinations': points+delta,
                'target_deltas': delta, 'group_prior_weights': np.array([0.])}
        source = {'groups': [{'features': ['leftEye', 'leftBrow'], 'translation_prior_weight': 0.}]}
        provenance = {'source_neck_support': {'profile_kind': 'neck-arclength',
            'base_target_count': 129, 'added_target_count': 3, 'source': neck.tolist(),
            'requested': (points+delta)[-3:].tolist(),
            'continuation': [{'source_indices': [0, 1, 2], 'side': 1}]}}
        audit = {'frame_raw': {'origin': [50., 0.], 'across': [1., 0.],
            'down': [0., 1.], 'bounds': [-30., 0., 30., 100.]},
            'midline_u': 0., 'calibration_k_full': .2, 'source_contour': contour.tolist()}
        profile = {'knots': [{'height': 0., 'gain': 1.}, {'height': .7, 'gain': .5}, {'height': 1., 'gain': 1.}]}
        return data, source, provenance, audit, profile

    def test_entire_neck_arc_uses_same_profile_entry_gain(self):
        data, source, provenance, audit, profile = self.fixture()
        targets, report = expected_targets(data, source, provenance, audit, profile, 60.)
        np.testing.assert_array_equal(targets[-3:]-data['target_sources'][-3:], [[-1.5, 0.], [-.75, 0.], [0., 0.]])
        np.testing.assert_array_equal(targets[128], data['target_destinations'][128])
        self.assertEqual(report[0]['profile_entry_gain'], .5)

    def test_mixed_old_neck_is_rejected_even_if_face_is_current(self):
        data, source, provenance, audit, profile = self.fixture()
        targets, report = expected_targets(data, source, provenance, audit, profile, 60.)
        targets[-3:] = data['target_destinations'][-3:]
        with self.assertRaisesRegex(ValueError, 'target'):
            expected, _ = expected_targets(data, source, provenance, audit, profile, 60.)
            check_target_variant(targets, expected, report, report)

    def test_wrong_neck_gain_metadata_is_rejected(self):
        data, source, provenance, audit, profile = self.fixture()
        targets, report = expected_targets(data, source, provenance, audit, profile, 60.)
        wrong = copy.deepcopy(report); wrong[0]['profile_entry_gain'] = 1.
        with self.assertRaisesRegex(ValueError, 'neck'):
            check_target_variant(targets, targets, report, wrong)

    def test_nonzero_position_prior_is_not_silently_reused(self):
        data, source, provenance, audit, profile = self.fixture()
        data['group_prior_weights'][0] = 2.
        with self.assertRaisesRegex(ValueError, 'prior'):
            expected_targets(data, source, provenance, audit, profile, 60.)

    def test_unknown_accessory_role_is_not_silently_retained(self):
        data, source, provenance, audit, profile = self.fixture()
        source['groups'][0] = {'name': 'accessory_0', 'translation_prior_weight': 0.}
        with self.assertRaisesRegex(ValueError, 'attachment'):
            expected_targets(data, source, provenance, audit, profile, 60.)

    def test_unity_profile_preserves_neck_and_invalid_profile_rejected(self):
        data, source, provenance, audit, _ = self.fixture()
        profile = {'knots': [{'height': 0., 'gain': 1.}, {'height': 1., 'gain': 1.}]}
        targets, _ = expected_targets(data, source, provenance, audit, profile, 60.)
        np.testing.assert_array_equal(targets[128:], data['target_destinations'][128:])
        profile['knots'][1]['height'] = 0.
        with self.assertRaisesRegex(ValueError, 'profile'):
            expected_targets(data, source, provenance, audit, profile, 60.)


if __name__ == '__main__':
    unittest.main()
