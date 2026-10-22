"""Analytic oracle checks for the frozen jaw-start transition diagnostic."""
import unittest

import numpy as np

from audit_joint_jaw_start import OriginalContourArc, directional_derivative, onset_energy, unweighted_prefix_energy


class JawStartOracleTests(unittest.TestCase):
    def setUp(self):
        self.grid = {'origin': [-10., -10.], 'spacing': [1., 1.], 'width': 51, 'height': 51}
        self.arc = OriginalContourArc({'contour': [[0., 0.], [20., 20.], [30., 20.], [30., 0.]]},
            {'start': {'edge': 0, 't': 0.}, 'direction': 1})
        self.y, self.x = np.mgrid[:51, :51].astype(float)-10.

    def test_constant_linear_quadratic_displacement_have_literal_energy(self):
        length = 12.; a = .03; b = .4
        for field, expected in [
            (np.full_like(self.x, 3.), 0.),
            (b*(self.x+self.y)/np.sqrt(2.), .5*b*b*length**3),
            (2.*a*self.x*self.y, 5./21.*a*a*length**5),
        ]:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(onset_energy(self.arc, field, self.grid, length), expected, places=10)

    def test_bilinear_directional_derivative_uses_original_physical_units(self):
        s = np.array([.3, 2.7, 7.8]); a = .03; b = .4
        field = 2.*a*self.x*self.y+b*(self.x+self.y)/np.sqrt(2.)+7.
        actual = directional_derivative(field, self.arc.source(s), self.arc.tangent(s), self.grid)
        np.testing.assert_allclose(actual, 2.*a*s+b, atol=2e-14, rtol=0.)

    def test_prefix_endpoint_does_not_change_the_external_length_scale(self):
        scale = 12.; a = .03; b = .4
        for end in [2., 6., 12.]:
            with self.subTest(end=end):
                self.assertAlmostEqual(unweighted_prefix_energy(self.arc,
                    b*(self.x+self.y)/np.sqrt(2.), self.grid, end, scale), scale**2*b*b*end, places=10)
                self.assertAlmostEqual(unweighted_prefix_energy(self.arc,
                    2.*a*self.x*self.y, self.grid, end, scale), 4./3.*scale**2*a*a*end**3, places=10)

    def test_grid_seam_selects_the_forward_direction_in_source_arclength(self):
        field = abs(self.x-2.); points = np.array([[2., 0.], [2., 0.]])
        directions = np.array([[1., 0.], [-1., 0.]])
        np.testing.assert_array_equal(directional_derivative(field, points, directions, self.grid), [1., 1.])

    def test_negative_direction_retains_original_closed_contour_and_upstream(self):
        arc = OriginalContourArc({'contour': [[0., 0.], [2., 0.], [2., 2.], [0., 2.]]},
            {'start': {'edge': 0, 't': 0.}, 'direction': -1})
        np.testing.assert_array_equal(arc.source([-1., 0., 1.]), [[1., 0.], [0., 0.], [0., 1.]])
        np.testing.assert_array_equal(arc.tangent([0.]), [[0., 1.]])


if __name__ == '__main__':
    unittest.main()
