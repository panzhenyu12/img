"""Analytic trace cases independent of any solved photograph or optimizer."""
import unittest

import numpy as np
from scipy.sparse import eye

from curve_grid_continuity import curve_grid_continuity
from joint_jaw_boundary_field import GridCurve


def make_curve(points, width=4, height=3):
    problem = {'grid': {'origin': [0., 0.], 'spacing': [1., 1.],
                        'width': width, 'height': height},
               'mapping': np.arange(width*height), 'hessian': eye(width*height, format='csc')}
    return GridCurve(points, problem, [1., 0.], [0., 1.])


class CurveGridContinuityTests(unittest.TestCase):
    def test_global_bilinear_is_continuous_but_grid_cusp_has_literal_jump(self):
        curve = make_curve([[.25, .375], [2.75, 1.625]])
        block = curve_grid_continuity(curve)
        y, x = np.mgrid[:3, :4].astype(float)
        self.assertEqual(block['A'].shape, (3, 12))
        np.testing.assert_array_equal(block['b'], np.zeros(3))
        np.testing.assert_allclose(block['A']@(3.+2.*x-4.*y+x*y).ravel(), 0., atol=2e-15, rtol=0.)
        np.testing.assert_allclose(block['A']@abs(x-1.).ravel(),
                                   [4./np.sqrt(5.), 0., 0.], atol=2e-15, rtol=0.)
        self.assertEqual([r['source_edge_index'] for r in block['report']['crossings']], [0, 0, 0])

    def test_oblique_c1_trace_can_have_different_quadratics_on_each_side(self):
        curve = make_curve([[.25, .3125], [1.75, .6875]], width=3, height=2)
        block = curve_grid_continuity(curve)
        y, x = np.mgrid[:2, :3].astype(float)
        # Left u=0; right u=(x-1)(y-.5). Along y=x/4+1/4 this is
        # 0 then (x-1)^2/4, with equal first but unequal second derivatives.
        z = (np.maximum(x-1., 0.)*(y-.5)).ravel()
        self.assertEqual(block['A'].shape, (1, 6))
        np.testing.assert_allclose(block['A']@z, 0., atol=1e-15, rtol=0.)
        samples = curve.op(np.linspace(0., curve.length, 4))@z
        np.testing.assert_allclose(samples, [0., 0., .015625, .140625], atol=1e-15, rtol=0.)
        self.assertAlmostEqual(float(np.diff(samples, n=3)[0]), .09375)

    def test_axis_aligned_trace_exposes_affine_locking_risk(self):
        curve = make_curve([[.25, .5], [2.75, .5]], height=2)
        block = curve_grid_continuity(curve)
        # A field constant in y has four nodal x values; C1 imposes exactly
        # these two second differences, leaving only affine x traces.
        repeat_x = np.tile(np.eye(4), (2, 1))
        np.testing.assert_array_equal(block['A']@repeat_x,
                                      [[1., -2., 1., 0.], [0., 1., -2., 1.]])
        self.assertEqual(block['report']['axis_aligned_source_edge_count'], 1)
        self.assertEqual(block['report']['source_edges'][0]['abs_quadratic_trace_factor'], 0.)

    def test_original_vertex_on_grid_is_excluded_without_flattening_it(self):
        points = [[.25, .3125], [1., .5], [1.75, .3125]]
        curve = make_curve(points, width=3, height=2)
        block = curve_grid_continuity(curve)
        self.assertEqual(block['A'].shape, (0, 6))
        np.testing.assert_array_equal(curve.points, points)
        excluded = block['report']['excluded_source_vertices_on_grid']
        self.assertEqual([r['source_vertex_index'] for r in excluded], [1])
        self.assertEqual(excluded[0]['grid_axes'], ['x'])

    def test_grid_corner_is_one_crossing_and_reversing_the_arc_keeps_the_jump(self):
        points = [[.25, .25], [2.75, 2.75]]
        forward = curve_grid_continuity(make_curve(points, height=4))
        backward = curve_grid_continuity(make_curve(points[::-1], height=4))
        self.assertEqual(forward['A'].shape, (2, 16))
        self.assertEqual([r['grid_axes'] for r in forward['report']['crossings']], [['x', 'y'], ['x', 'y']])
        np.testing.assert_array_equal(forward['A'].toarray(), backward['A'].toarray()[::-1])
        y, x = np.mgrid[:4, :4].astype(float)
        np.testing.assert_allclose(forward['A']@(abs(x-1.)+abs(y-1.)).ravel(),
                                   [2.*np.sqrt(2.), 0.], atol=1e-15, rtol=0.)

    def test_anisotropic_grid_derivatives_keep_physical_source_pixel_units(self):
        problem = {'grid': {'origin': [-5., 7.], 'spacing': [2., 3.], 'width': 4, 'height': 3},
                   'mapping': np.arange(12), 'hessian': eye(12, format='csc')}
        curve = GridCurve([[-4.5, 8.125], [.5, 11.875]], problem, [1., 0.], [0., 1.])
        block = curve_grid_continuity(curve)
        y, x = np.mgrid[:3, :4].astype(float)
        x = -5.+2.*x; y = 7.+3.*y
        np.testing.assert_allclose(block['A']@abs(x+3.).ravel(), [1.6, 0., 0.], atol=1e-15, rtol=0.)
        np.testing.assert_allclose(block['A']@(x*y-3.*y+2.*x).ravel(), 0., atol=1e-14, rtol=0.)

    def test_nearby_vertex_and_nearby_grid_corner_are_not_snapped(self):
        offset = 2.**-42
        near_vertex = curve_grid_continuity(make_curve(
            [[.25, .375], [1.+offset, .5], [1.75, .625]], width=3, height=2))
        self.assertEqual(near_vertex['A'].shape, (1, 6))
        self.assertEqual(near_vertex['report']['crossings'][0]['source_edge_index'], 0)
        self.assertEqual(near_vertex['report']['excluded_source_vertices_on_grid'], [])
        near_corner = curve_grid_continuity(make_curve(
            [[.25, .25+offset], [1.75, 1.75+offset]], width=3, height=3))
        self.assertEqual(near_corner['A'].shape, (2, 9))
        self.assertEqual([r['grid_axes'] for r in near_corner['report']['crossings']], [['y'], ['x']])

    def test_fixed_nodes_keep_scalar_mapping_and_zero_rows_keep_provenance(self):
        curve = make_curve([[.25, .3125], [1.75, .6875]], width=3, height=2)
        full = curve_grid_continuity(curve)
        curve.problem['mapping'] = np.array([-1, 0, 1, 2, 3, 4])
        curve.problem['hessian'] = eye(5, format='csc'); curve.size = 5
        reduced = curve_grid_continuity(curve)
        np.testing.assert_array_equal(reduced['A'].toarray(), full['A'].toarray()[:, 1:])
        curve.problem['mapping'] = np.full(6, -1)
        curve.problem['hessian'] = eye(0, format='csc'); curve.size = 0
        fixed = curve_grid_continuity(curve)
        self.assertEqual(fixed['A'].shape, (1, 0))
        self.assertEqual(fixed['report']['zero_support_equations'], 1)
        self.assertEqual(len(fixed['report']['crossings']), 1)


if __name__ == '__main__':
    unittest.main()
