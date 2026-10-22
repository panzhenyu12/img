"""Analytic variance cases without a photograph or an optimization solve."""
import unittest

import numpy as np
from scipy.sparse import eye

from curve_curvature_variance import curvature_variance
from joint_jaw_boundary_field import GridCurve


def make_curve(points, *, width=3, height=3, origin=(0., 0.), spacing=(1., 1.)):
    problem = {'grid': {'origin': list(origin), 'spacing': list(spacing),
                        'width': width, 'height': height},
               'mapping': np.arange(width*height), 'hessian': eye(width*height, format='csc')}
    return GridCurve(points, problem, [1., 0.], [0., 1.])


def residual_energy(block, z):
    residual = block['operator']@np.asarray(z).ravel()
    return float(residual@residual)


class CurvatureVarianceTests(unittest.TestCase):
    def test_constant_curvature_gets_no_extra_penalty(self):
        curve = make_curve([[0., 0.], [2., 2.]])
        y, x = np.mgrid[:3, :3]
        block = curvature_variance(curve, 1., np.sqrt(2.))
        self.assertEqual(block['report']['window_count'], 1)
        self.assertLess(residual_energy(block, x*y), 1e-27)
        self.assertLess(block['report']['maximum_kernel_mass_error'], 1e-14)

    def test_opposite_curvatures_cancel_in_d2_but_have_literal_positive_variance(self):
        curve = make_curve([[0., 0.], [2., 2.]])
        # Along x=y, C=+1 up to x=1 and C=-1 thereafter. u and u' are
        # continuous; samples [0,1,2] have D2=0. Each half has kernel mass .5.
        z = np.array([[0., 0., 0.], [0., 1., 2.], [0., 2., 2.]])
        block = curvature_variance(curve, 1., np.sqrt(2.))
        sample = curve.op(np.linspace(0., curve.length, 3))@z.ravel()
        np.testing.assert_array_equal(sample, [0., 1., 2.])
        self.assertEqual(float(np.diff(sample, n=2)[0]), 0.)
        self.assertAlmostEqual(residual_energy(block, z), np.sqrt(2.), places=14)
        self.assertAlmostEqual(float(z.ravel()@(block['hessian']@z.ravel())), np.sqrt(2.), places=14)

    def test_spatial_affine_field_has_zero_curvature_on_each_original_edge(self):
        curve = make_curve([[.25, .25], [1.25, 2.5], [2.75, .5]], width=4, height=4)
        y, x = np.mgrid[:4, :4]
        block = curvature_variance(curve, 12., 2.)
        self.assertLess(residual_energy(block, 2.*x-3.*y+7.), 1e-23)
        self.assertTrue(block['report']['original_vertex_derivative_jumps_excluded'])

    def test_original_vertex_kink_has_zero_supplement_while_old_d2_still_penalizes_it(self):
        curve = make_curve([[0., 0.], [1., 0.], [1., 1.]])
        _, x = np.mgrid[:3, :3]
        block = curvature_variance(curve, 1., .5)
        self.assertEqual(residual_energy(block, x), 0.)
        values = curve.op(np.linspace(0., 2., 5))@x.ravel()
        # u(s)=min(s,1). The original h=.5 discrete energy remains 2.
        self.assertEqual(float(.5*np.sum((np.diff(values, n=2)/.5**2)**2)), 2.)
        self.assertEqual(block['report']['replaces_original_d2'], False)

    def test_physical_similarity_scales_energy_as_cubic_pixels(self):
        z = np.array([[0., 0., 0.], [0., 1., 2.], [0., 2., 2.]])
        base = curvature_variance(make_curve([[0., 0.], [2., 2.]]), 1., np.sqrt(2.))
        k = 3.; shift = np.array([-3., 5.])
        curve = make_curve((shift+np.array([[0., 0.], [2., 2.]])*k).tolist(),
                           origin=shift, spacing=(k, k))
        scaled = curvature_variance(curve, k, k*np.sqrt(2.))
        self.assertAlmostEqual(residual_energy(scaled, k*z), k**3*residual_energy(base, z), places=12)

    def test_fixed_zero_nodes_use_only_existing_scalar_columns(self):
        curve = make_curve([[0., 0.], [2., 2.]])
        full = curvature_variance(curve, 1., np.sqrt(2.))
        free = np.array([1, 3, 4, 5, 6, 7, 8]); mapping = np.full(9, -1)
        mapping[free] = np.arange(len(free))
        curve.problem['mapping'] = mapping
        curve.problem['hessian'] = eye(len(free), format='csc'); curve.size = len(free)
        reduced = curvature_variance(curve, 1., np.sqrt(2.))
        np.testing.assert_allclose(reduced['operator'].toarray(), full['operator'].toarray()[:, free], atol=0., rtol=0.)
        self.assertEqual(reduced['hessian'].shape, (len(free), len(free)))

    def test_constant_curvature_can_still_cancel_against_an_original_vertex_jump(self):
        vertex = np.array([2., 2.])
        before = np.array([1., 2.])/np.sqrt(5.)
        after = np.array([2., 1.])/np.sqrt(5.)
        curve = make_curve([vertex-before, vertex, vertex+after], width=5, height=5)
        y, x = np.mgrid[:5, :5]
        z = (1.25*x*y-np.sqrt(5.)*x).ravel()
        block = curvature_variance(curve, 1., 1.)
        nodes = np.linspace(0., curve.length, 3); h = curve.length/2
        values = curve.op(nodes)@z
        old_energy = h*float(np.sum((np.diff(values, n=2)/h**2)**2))
        # Both unit-length edges have C=1, while the source-vertex slope
        # jump is -1. Therefore D2=mu+J=0 and Var(C)=0 despite nonzero C.
        self.assertLess(old_energy, 1e-26)
        self.assertLess(residual_energy(block, z), 1e-26)
        self.assertFalse(block['report']['source_vertex_mean_cancellation_controlled'])


if __name__ == '__main__':
    unittest.main()
