"""Small mathematical behavior checks; no photograph or full-system runs."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, eye

from reduced_constrained import project_source_cones, solve_constrained_reduced
from constrained_solve import assemble_cones, inspect_corners


class ReducedConstrainedTests(unittest.TestCase):
    def test_projected_cones_have_identical_slack_on_lifted_original_field(self):
        grid = {'width': 3, 'height': 3, 'spacing': [2., 2.]}
        mapping = np.arange(9)
        # A genuine proper subspace with shared coarse control contributions.
        r = csc_matrix([[1, 0, 0], [.5, .5, 0], [0, 1, 0],
                        [.5, 0, .5], [.25, .25, .5], [0, .5, .5],
                        [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        active = [0, 3, 5, 8, 15]
        full_a, full_b, full_cones = assemble_cones(grid, mapping, 9, active)
        reduced_a, reduced_b, reduced_cones = project_source_cones(grid, mapping, r, active)
        rng = np.random.default_rng(20261022)
        for _ in range(12):
            coefficients = rng.normal(0, .2, (3, 2)); auxiliaries = rng.normal(0, .2, len(active))
            full = r @ coefficients
            original_slack = full_b - full_a @ np.r_[full[:, 0], full[:, 1], auxiliaries]
            projected_slack = reduced_b - reduced_a @ np.r_[coefficients[:, 0], coefficients[:, 1], auxiliaries]
            np.testing.assert_allclose(projected_slack, original_slack, rtol=0, atol=3e-16)
        np.testing.assert_array_equal(full_b, reduced_b)
        self.assertEqual([str(c) for c in full_cones], [str(c) for c in reduced_cones])

    def test_constrained_motion_keeps_coupled_original_background_rows(self):
        grid = {'width': 3, 'height': 3, 'spacing': [2., 2.]}
        mapping = np.full(9, -1, dtype=int); mapping[4] = 0
        rhs = np.array([[3., 1.]])
        lines = csc_matrix([[1., 1.], [2., 2.], [0., 0.]])
        result = solve_constrained_reduced(eye(1, format='csc'), rhs, eye(1, format='csc'), grid, mapping, lines)
        z, report = result['solution'], result['report']
        np.testing.assert_array_equal(rhs, [[3., 1.]])
        np.testing.assert_allclose(lines @ z.T.ravel(), 0., rtol=0, atol=1e-8)
        self.assertGreater(z[0, 0], .1)
        self.assertLess(z[0, 0], 1.)  # Unconstrained equality solution was 1.
        self.assertFalse(report['full_space_optimality_claimed'])
        self.assertEqual(report['all_original_background']['original_rows'], 3)
        self.assertTrue(report['all_original_background']['passed'])
        self.assertGreaterEqual(len(report['rounds']), 1)
        self.assertLessEqual(len(report['rounds']), 2)
        for row in report['rounds']:
            self.assertLessEqual(row['selected_seed_count'], 256)
            self.assertEqual(row['solver_time_limit_seconds'], 12.)
            for corner in row['all_active_corner_ids']:
                self.assertTrue(all(k*4+corner % 4 in row['all_active_corner_ids'] for k in range(4)))
        _, actual_scan = inspect_corners(z, grid, mapping)
        self.assertEqual(report['source_proxy_pass'], actual_scan['corners_violating_1e_8'] == 0)
        self.assertTrue(report['final_actual_f32_geometry_and_RGB_pending'])

    def test_infeasibility_certificate_is_not_returned_as_a_photo_candidate(self):
        grid = {'width': 3, 'height': 3, 'spacing': [2., 2.]}
        mapping = np.full(9, -1, dtype=int); mapping[4] = 0
        rhs = np.array([[3., 1.]])
        rejected = {'status': 'PrimalInfeasible', 'primal_candidate_available': False}
        with patch('reduced_constrained._solve_round', return_value=(None, rejected)):
            result = solve_constrained_reduced(eye(1), rhs, eye(1), grid, mapping)
        np.testing.assert_array_equal(result['solution'], rhs)
        self.assertEqual(result['report']['status'], 'PrimalInfeasible')
        self.assertEqual(result['report']['selected_stage'], 'initial_equality_quadratic')
        self.assertTrue(result['report']['diagnostic_only'])
        self.assertFalse(result['report']['subspace_optimization_converged'])

    def test_passing_proxy_does_not_erase_solver_time_limit(self):
        grid = {'width': 3, 'height': 3, 'spacing': [2., 2.]}
        mapping = np.full(9, -1, dtype=int); mapping[4] = 0
        interrupted = {'status': 'MaxTime', 'primal_candidate_available': True, 'active_cones_pass': True}
        with patch('reduced_constrained._solve_round', return_value=(np.zeros((1, 2)), interrupted)):
            result = solve_constrained_reduced(eye(1), np.array([[3., 1.]]), eye(1), grid, mapping)
        self.assertEqual(result['report']['status'], 'MaxTime')
        self.assertTrue(result['report']['source_proxy_pass'])
        self.assertTrue(result['report']['diagnostic_only'])
        self.assertFalse(result['report']['subspace_optimization_converged'])


if __name__ == '__main__':
    unittest.main()
