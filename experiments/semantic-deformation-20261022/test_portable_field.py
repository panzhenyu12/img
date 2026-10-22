"""New-photo source-field behavior; synthetic pixels only, no renderer/build."""
import unittest

import numpy as np

from portable_field import build_problem, solve_problem, evaluate_motion


class PortableFieldTests(unittest.TestCase):
    def make(self, delta=(0., 0.), **kwargs):
        source = np.array([[20., 16.], [20., 20.], [24., 16.], [24., 20.]])
        return build_problem((41, 41), source, source+delta, [[8., 18.]],
                             crop=[0, 0, 41, 41], target_weights=1000., **kwargs)

    def test_zero_keeps_the_complete_source_field_exactly_zero(self):
        result = solve_problem(self.make())
        np.testing.assert_array_equal(result['source_displacement'], 0.)
        self.assertEqual(result['geometry']['determinant_min'], 1.)
        self.assertTrue(result['geometry']['boundary_zero'])
        self.assertEqual(result['target_max_error_px'], 0.)

    def test_local_translation_reaches_targets_and_fixed_pixel_neighborhood_stays_exact(self):
        problem = self.make(delta=(.8, -.2))
        result = solve_problem(problem)
        self.assertLess(result['target_max_error_px'], .001)
        self.assertTrue(result['inverse_evaluation_allowed'])
        # Closed +/-1px footprint is protected, not merely the central sample.
        yy, xx = np.meshgrid(np.linspace(17, 19, 9), np.linspace(7, 9, 9))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        np.testing.assert_array_equal(evaluate_motion(problem, result['solution'], points), 0.)
        field = result['source_displacement']
        np.testing.assert_array_equal(field[[0, 1, -2, -1]], 0.)
        np.testing.assert_array_equal(field[:, [0, 1, -2, -1]], 0.)
        np.testing.assert_array_equal(evaluate_motion(problem, result['solution'], [[-1., 8.], [45., 8.]]), 0.)

    def test_opposite_requests_share_a_continuous_field_basis(self):
        positive_problem = self.make(delta=(.8, -.2))
        negative_problem = self.make(delta=(-.8, .2))
        positive = solve_problem(positive_problem); negative = solve_problem(negative_problem)
        np.testing.assert_allclose(positive['source_displacement'], -negative['source_displacement'],
                                   rtol=0, atol=1e-12)
        points = [[18.-1e-7, 22.3], [18.+1e-7, 22.3]]
        motion = evaluate_motion(positive_problem, positive['solution'], points)
        self.assertLess(float(np.linalg.norm(motion[0]-motion[1])), 1e-6)
        self.assertFalse(positive['aesthetic_approval'])

    def test_hard_diagonal_line_preserves_its_complete_normal_motion(self):
        problem = self.make(delta=(.8, -.2), lines=[{'start': [4., 14.], 'end': [26., 36.], 'samples': 2}])
        result = solve_problem(problem)
        t = np.linspace(4, 26, 101)
        motion = evaluate_motion(problem, result['solution'], np.column_stack([t, t+10]))
        np.testing.assert_allclose(motion[:, 0]-motion[:, 1], 0., rtol=0, atol=1e-9)
        self.assertLess(result['report']['full_line_unit_max_residual'], 1e-9)

    def test_nonzero_request_cannot_be_silently_locked_by_fixed_pixels(self):
        with self.assertRaisesRegex(ValueError, 'fully fixed source support'):
            build_problem((41, 41), [[20., 20.]], [[21., 20.]], [[20., 20.]], crop=[0, 0, 41, 41])
        with self.assertRaisesRegex(ValueError, 'exactly covered'):
            build_problem((41, 41), [[20., 20.]], [[21., 20.]], [], crop=[0, 0, 40, 41])

    def test_source_on_hard_line_rejects_a_conflicting_normal_target(self):
        with self.assertRaisesRegex(ValueError, 'hard background line'):
            self.make(delta=(.8, -.2), lines=[{'start': [4., 4.], 'end': [36., 36.], 'samples': 2}])

    def test_duplicate_source_points_cannot_average_opposing_intents(self):
        with self.assertRaisesRegex(ValueError, 'conflicting repeated source'):
            build_problem((41, 41), [[20., 20.], [20., 20.]],
                          [[21., 20.], [19., 20.]], [], crop=[0, 0, 41, 41])

    def test_target_cannot_cross_the_fixed_roi_boundary(self):
        with self.assertRaisesRegex(ValueError, 'requested target outside'):
            build_problem((61, 61), [[20., 20.]], [[42., 20.]], [], crop=[0, 0, 41, 41])


if __name__ == '__main__':
    unittest.main()
