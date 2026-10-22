"""Reject no-op/incorrect edits even when all protected image bytes survive."""
import unittest

import numpy as np

from portable_result_gate import evaluate_result


class PortableResultGateTests(unittest.TestCase):
    def check(self, actual, *, amount=60., fixed=0, changed=40):
        source = np.array([[10., 10.], [20., 10.], [30., 10.]])
        requested = source+[[2., 0.], [0., 2.], [0., 0.]]
        return evaluate_result(source, requested, np.asarray(actual), amount=amount,
            fixed_changed_pixels=fixed, whole_image_changed_pixels=changed)

    def test_zero_response_to_nonzero_request_is_rejected(self):
        result = self.check([[10., 10.], [20., 10.], [30., 10.]])
        self.assertFalse(result['passed'])

    def test_large_orthogonal_error_is_rejected_despite_unit_projection_gain(self):
        result = self.check([[12., 10.3], [20., 12.], [30., 10.]])
        self.assertFalse(result['passed'])

    def test_each_active_point_has_bounds_not_only_the_average(self):
        result = self.check([[12.22, 10.], [20., 11.78], [30., 10.]])
        self.assertFalse(result['passed'])
        self.assertEqual(result['gain_failed_rows'], [0, 1])
        self.assertTrue(result['checks']['contour_error_within_limit'])

    def test_unchanged_contour_rows_cannot_drift_outside_error_limit(self):
        result = self.check([[12., 10.], [20., 12.], [30.3, 10.]])
        self.assertFalse(result['passed'])
        self.assertEqual(result['error_failed_rows'], [2])

    def test_actual_success_and_protected_pixel_failure_are_distinct(self):
        actual = [[12., 10.], [20., 12.], [30., 10.]]
        passed = self.check(actual)
        self.assertTrue(passed['passed'])
        self.assertFalse(passed['aesthetic_approval'])
        failed = self.check(actual, fixed=1)
        self.assertFalse(failed['passed'])
        self.assertEqual(failed['failed_checks'], ['fixed_pixels_unchanged'])

    def test_zero_needs_exact_coordinates_and_bytes(self):
        source = np.array([[10., 10.], [20., 10.]])
        def check(q, r, changed):
            return evaluate_result(source, q, r, amount=0., fixed_changed_pixels=0,
                                   whole_image_changed_pixels=changed)
        self.assertTrue(check(source, source, 0)['passed'])
        self.assertFalse(check(source, source+1e-12, 0)['passed'])
        self.assertFalse(check(source+1e-12, source, 0)['passed'])
        self.assertFalse(check(source, source, 1)['passed'])

    def test_nonfinite_or_mismatched_evidence_is_not_accepted(self):
        for actual in [[[np.nan, 10.], [20., 12.], [30., 10.]], [[12., 10.]]]:
            with self.assertRaises(ValueError):
                self.check(actual)

    def test_source_review_must_cover_both_mouth_boundaries(self):
        from run_portable_jaw import validate_review_boundaries
        starts=[{'edge':2,'t':.2},{'edge':8,'t':.4}]
        end={'edge':5,'t':.5}
        geometry={'arcs':[{'start':value,'end':end} for value in starts]}
        validate_review_boundaries(geometry,{'mouth_crossings':starts,'chin152_projection':end})
        for incomplete in [[],starts[:1],starts+starts[:1]]:
            with self.assertRaisesRegex(ValueError,'exactly two'):
                validate_review_boundaries(geometry,{'mouth_crossings':incomplete,'chin152_projection':end})


if __name__ == '__main__':
    unittest.main()
