"""Small geometry regression for success-with-infeasible SLSQP stagnation."""
import unittest
from unittest.mock import patch

import numpy as np

import jaw_curve_qp as core
from test_jaw_curve_qp import curved


def rounded_stagnation_fixture():
    # Lower arcs from the observed failure, projected to its local frame and
    # rounded to 0.1 px. No photo, cache, model, or result files are required.
    left = np.array([[-101.2,112.2],[-101.1,113.8],[-99.2,117.8],[-99.,119.8],
        [-94.9,131.7],[-92.9,135.7],[-80.6,151.6],[-72.8,159.5],[-66.6,167.4],
        [-59.7,172.7],[-51.,181.3],[-39.,189.2],[-35.5,191.8],[-21.5,195.6],
        [-7.5,196.4],[-.4,196.1]])
    right = np.array([[97.6,112.2],[96.,115.8],[90.6,123.9],[90.4,125.9],
        [89.7,126.6],[88.7,127.9],[88.1,129.9],[83.8,135.6],[75.3,144.],
        [70.1,152.1],[58.1,167.6],[44.2,180.4],[35.8,186.4],[22.4,193.4],
        [8.5,195.8],[-.4,196.1]])
    geometry = curved(left=left, right=right)
    geometry['frame']['origin'] = [0.,0.]
    return geometry


class JawQPPrecisionTests(unittest.TestCase):
    def test_stalled_continuous_cut_is_solved_without_relaxing_request(self):
        geometry = rounded_stagnation_fixture()
        result = core.solve_jaw_curve_profile(geometry, 100., node_spacing_px=2.,
            window_radius_px=16., smoothing_length_px=4., bilateral_weight=1.)
        report = result['report']
        self.assertTrue(report['accepted_numerically'], report['continuous_check'])
        fallback = report['precision_fallback']
        self.assertEqual(fallback['method'], 'clarabel_cholesky_whitened_same_qp')
        self.assertTrue(fallback['same_objective_and_constraints'])
        self.assertTrue(all(r['continuous_violating_subintervals']==0 for r in report['continuous_check']))
        self.assertLess(abs(report['signed_area_error_px2']), 1e-8)
        self.assertEqual(report['endpoint_displacement_max_px'], 0.)
        self.assertLessEqual(report['sign_violation_px'], 1e-10)
        self.assertEqual(result['source_geometry'], geometry)
        self.assertEqual(result['required_signed_area_delta_px2'],
            core._Problem(geometry,2.,16.).beta_area(100.,.13))
        # The source tail remains unknown; no repair by flattening or freezing.
        self.assertTrue(result['profiles'][0]['source_nonmonotone_height_intervals_px'])

    def test_successful_existing_path_does_not_invoke_fallback(self):
        with patch.object(core, '_solve_precise_qp', side_effect=AssertionError('unexpected fallback')):
            result = core.solve_jaw_curve_profile(curved(),60.,node_spacing_px=5.,window_radius_px=8.)
        self.assertTrue(result['report']['accepted_numerically'])
        self.assertNotIn('precision_fallback',result['report'])

    def test_nonzero_equality_bounds_and_fixed_coordinate_keep_analytic_optimum(self):
        result = core._solve_precise_qp(
            np.array([[4.,1.,.7],[1.,3.,0.],[.7,0.,2.]]), np.array([-8.,-6.,1.]),
            np.array([[1.,0.,0.]]), np.array([1.4]),
            np.array([[1.,1.,0.]]), np.array([2.]),
            np.array([0.,0.,.25]), np.array([np.inf,np.inf,.25]),17.)
        self.assertTrue(result.success,result.message)
        np.testing.assert_allclose(result.x,[1.4,.6,.25],rtol=0.,atol=1e-8)
        self.assertEqual(result.x[2],.25)
        for wanted, expected in [(2.,True),(3.,False)]:
            fixed = core._solve_precise_qp(np.eye(1),np.zeros(1),np.empty((0,1)),np.array([]),
                np.ones((1,1)),np.array([wanted]),np.array([2.]),np.array([2.]),1.)
            self.assertEqual(fixed.success,expected)
            self.assertEqual(fixed.x[0],2.)


if __name__=='__main__':unittest.main()
