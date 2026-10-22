"""Behavior checks for a coarse field with explicit protected motion groups."""
import unittest

import numpy as np
from scipy.sparse import diags, csc_matrix

from reduced_field import build_reduction, solve_reduced


class ReducedFieldTests(unittest.TestCase):
    def test_reduction_rejects_silent_zero_variable_rows(self):
        # The only movable source node is enclosed by fixed coarse nodes. A
        # reduction must reject this instead of quietly removing its motion.
        mapping = np.full(9, -1, dtype=int)
        mapping[4] = 0
        with self.assertRaises(ValueError):
            build_reduction(mapping, 3, 3, [], stride=2, fine_band=0)

    def test_unconstrained_bilinear_motion_reproduces_between_coarse_anchors(self):
        width, height = 18, 14  # Last cell is deliberately shorter.
        mapping = np.arange(width * height)
        r, report = build_reduction(mapping, width, height, [], stride=4, fine_band=1)
        yy, xx = np.mgrid[:height, :width]
        expected = (2 + .3 * xx - .2 * yy + .01 * xx * yy).ravel()
        anchors = np.asarray(report['anchor_variables'])
        np.testing.assert_allclose(r @ expected[anchors], expected, rtol=0, atol=2e-14)
        self.assertLess(r.shape[1], r.shape[0] / 4)
        np.testing.assert_array_equal(r[anchors].toarray(), np.eye(r.shape[1]))

    def test_fixed_pixels_and_shape_preserving_group_keep_different_roles(self):
        width = height = 17
        mapping = np.full((height, width), -1, dtype=int)
        free = np.ones_like(mapping, dtype=bool)
        free[:, 0] = False
        free[6:10, 6:10] = False
        mapping[free] = np.arange(free.sum())
        group = int(free.sum())
        mapping[6:10, 6:10] = group
        r, report = build_reduction(mapping.ravel(), width, height, [group], stride=4, fine_band=2)
        coarse = np.zeros((r.shape[1], 2))
        coarse[report['anchor_variables'].index(group)] = [3.25, -.5]
        z = r @ coarse
        field = np.zeros((height, width, 2))
        active = mapping >= 0
        field[active] = z[mapping[active]]
        np.testing.assert_array_equal(field[:, 0], 0)
        np.testing.assert_array_equal(field[6:10, 6:10], np.broadcast_to([3.25, -.5], (4, 4, 2)))
        np.testing.assert_array_equal(r[report['anchor_variables']].toarray(), np.eye(r.shape[1]))

    def test_reduced_solution_respects_vector_coupled_background_constraint(self):
        # Independent worked example: min (x-2)^2+(y-1)^2 s.t. x+y=0.
        r = csc_matrix([[1.], [1.]])
        h = diags([.5, .5]).tocsc()
        rhs = np.array([[1., .5], [1., .5]])
        line = csc_matrix([[1., 0., 0., 1.]])
        result = solve_reduced(h, rhs, r, line)
        np.testing.assert_allclose(result['solution'], [[.5, -.5], [.5, -.5]], atol=1e-12)
        self.assertLess(result['report']['full_line_max_residual'], 1e-12)
        self.assertFalse(result['report']['full_space_optimality_claimed'])
        self.assertLess(result['report']['projected_backward_error'], 1e-10)

    def test_local_refinement_contains_previous_motion_without_changing_groups(self):
        width = height = 17
        mapping = np.arange(width * height)
        old, before = build_reduction(mapping, width, height, [], stride=4, fine_band=1)
        new, after = build_reduction(mapping, width, height, [], stride=4, fine_band=1,
                                     extra_nodes=np.array([6*width+6, 6*width+7]))
        old_values = np.random.default_rng(42).normal(size=(old.shape[1], 2))
        old_motion = old @ old_values
        np.testing.assert_allclose(new @ old_motion[after['anchor_variables']], old_motion, atol=1e-14)
        self.assertEqual(new.shape[1], old.shape[1] + 2)


if __name__ == '__main__':
    unittest.main()
