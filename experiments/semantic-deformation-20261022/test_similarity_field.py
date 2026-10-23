"""Closed-form physical energy checks; no photo, renderer or field solve."""
import unittest

import numpy as np

from similarity_field import similarity_residuals


class SimilarityEnergyTests(unittest.TestCase):
    def test_checkerboard_cannot_be_invisible_at_the_cell_center(self):
        # u=(1-2x)(1-2y), v=0 on the unit square: .5 int|grad u|²=4/3.
        operator = similarity_residuals(2, 2, (1., 1.))
        displacement = np.r_[[1., -1., -1., 1.], np.zeros(4)]
        self.assertAlmostEqual(float(np.linalg.norm(operator @ displacement)**2), 4./3., places=13)

    def test_free_domain_finite_rotation_uniform_scale_and_translation_are_zero(self):
        xx, yy = np.meshgrid(np.arange(4)*2., np.arange(3)*3.)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        displacement = points @ (1.3*rotation-np.eye(2)).T + [3., -1.]
        operator = similarity_residuals(4, 3, (2., 3.))
        np.testing.assert_allclose(operator @ displacement.T.ravel(), 0., atol=2e-15, rtol=0)

    def test_affine_anisotropy_and_shear_have_the_analytic_physical_energy(self):
        points = np.array([[0., 0.], [2., 0.], [0., 3.], [2., 3.]])
        operator = similarity_residuals(2, 2, (2., 3.))
        for gradient, expected in [(np.diag([.4, -.2]), 1.08),
                                   (np.array([[0., .7], [0., 0.]]), 1.47)]:
            displacement = points @ gradient.T
            self.assertAlmostEqual(float(np.linalg.norm(operator @ displacement.T.ravel())**2),
                                   expected, places=13)

    def test_locked_nodes_are_eliminated_without_removing_surrounding_cells(self):
        mapping = np.arange(30)
        mapping[[0, 9, 12, 29]] = -1
        free = np.flatnonzero(mapping >= 0)
        mapping[free] = np.arange(len(free))
        full = similarity_residuals(6, 5, (2., 3.))
        fixed = similarity_residuals(6, 5, (2., 3.), mapping)
        np.testing.assert_array_equal(fixed.toarray(), full[:, np.r_[free, free+30]].toarray())

    def test_zero_boundary_cancels_coupling_to_half_of_exact_q1_membrane(self):
        # Independently integrated scalar Q1 stiffness, corner order 00,10,01,11.
        # int N_i,x N_j,x uses the 1D linear mass matrix [[2,1],[1,2]]/6.
        width, height, hx, hy = 7, 6, 2., 3.
        kx = np.array([[2, -2, 1, -1], [-2, 2, -1, 1],
                       [1, -1, 2, -2], [-1, 1, -2, 2]]) * (hy/hx/6.)
        ky = np.array([[2, 1, -2, -1], [1, 2, -1, -2],
                       [-2, -1, 2, 1], [-1, -2, 1, 2]]) * (hx/hy/6.)
        scalar = np.zeros((width*height, width*height))
        for y in range(height-1):
            for x in range(width-1):
                corners = np.array([0, 1, width, width+1])+y*width+x
                scalar[np.ix_(corners, corners)] += kx+ky
        for internal_fixed in [False, True]:
            locked = np.zeros((height, width), bool)
            locked[[0, -1], :] = True
            locked[:, [0, -1]] = True
            if internal_fixed:
                locked[3, 3] = True
            free = np.flatnonzero(~locked.ravel())
            mapping = np.full(width*height, -1)
            mapping[free] = np.arange(len(free))
            operator = similarity_residuals(width, height, (hx, hy), mapping)
            gram = (operator.T @ operator).toarray()
            n = len(free)
            expected = .5*scalar[np.ix_(free, free)]
            np.testing.assert_allclose(gram[:n, :n], expected, atol=1e-15, rtol=0)
            np.testing.assert_allclose(gram[n:, n:], expected, atol=1e-15, rtol=0)
            np.testing.assert_allclose(gram[:n, n:], 0., atol=1e-15, rtol=0)
            np.testing.assert_allclose(gram[n:, :n], 0., atol=1e-15, rtol=0)


if __name__ == '__main__':
    unittest.main()
