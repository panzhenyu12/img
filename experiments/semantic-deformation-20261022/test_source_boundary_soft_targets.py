"""Physical line-measure and source-path checks; no photo or field solve."""
import copy
import unittest

import numpy as np

from source_boundary_soft_targets import sample_boundary_soft_targets


class SourceBoundarySoftTargetTests(unittest.TestCase):
    def test_soft_targets_are_source_positions_with_smooth_endpoint_weight_fades(self):
        segments = [[[3., 4.], [7., 4.], [13., 4.]]]; before = copy.deepcopy(segments)
        result = sample_boundary_soft_targets(segments, uncertainty_px=2., spacing=1., density=2., fade_length=2.)
        source = result['source']; weights = result['weights']
        np.testing.assert_array_equal(result['target'], source)
        np.testing.assert_array_equal(source[[0,-1]], [[3.,4.],[13.,4.]])
        np.testing.assert_array_equal(weights[[0,-1]], [0.,0.])
        self.assertAlmostEqual(float(weights[np.flatnonzero(source[:,0] == 4.)[0]]), 1.)
        self.assertAlmostEqual(float(weights[np.flatnonzero(source[:,0] == 8.)[0]]), 2.)
        self.assertEqual(segments, before)
        self.assertEqual(result['report']['segments'][0]['original_source_points'], before[0])
        self.assertFalse(result['report']['hard_fixed_boundary_claimed'])
        self.assertEqual(result['report']['uncertainty_px'], 2.)
        self.assertEqual(result['report']['sample_support'], 'reviewed polyline centerline only')

    def test_corners_and_separate_polylines_are_not_replaced_by_shortcuts(self):
        segments = [[[0.,0.],[3.,0.],[3.,4.]], [[30.,30.],[34.,30.]]]
        result = sample_boundary_soft_targets(segments, uncertainty_px=[2.,3.], spacing=2., density=1., fade_length=1.)
        a, b = result['report']['segments']
        first = result['source'][a['row_start']:a['row_end']]
        second = result['source'][b['row_start']:b['row_end']]
        self.assertTrue(np.any(np.all(first == [3.,0.], axis=1)))
        self.assertTrue(np.all(((first[:,1] == 0.) & (first[:,0] >= 0.) & (first[:,0] <= 3.)) |
                               ((first[:,0] == 3.) & (first[:,1] >= 0.) & (first[:,1] <= 4.))))
        self.assertTrue(np.all(second[:,1] == 30.))
        self.assertTrue(np.all((second[:,0] >= 30.) & (second[:,0] <= 34.)))
        self.assertEqual(a['length_px'], 7.); self.assertEqual(b['length_px'], 4.)
        self.assertEqual([a['uncertainty_px'], b['uncertainty_px']], [2.,3.])
        self.assertFalse(result['report']['unknown_gaps_connected'])

    def test_density_is_per_arc_length_and_does_not_grow_with_sample_count(self):
        for spacing in [2.,5.]:
            result = sample_boundary_soft_targets([[[0.,0.],[100.,0.]]], spacing=spacing, density=.5, fade_length=10.)
            # Integral of two nonoverlapping cubic smoothstep fades:
            # 0.5 * (100 - 10) = 45, because each 10px fade averages 1/2.
            self.assertAlmostEqual(float(result['weights'].sum()),45.,places=12)
            doubled = sample_boundary_soft_targets([[[0.,0.],[100.,0.]]], spacing=spacing, density=1., fade_length=10.)
            np.testing.assert_array_equal(doubled['weights'],2.*result['weights'])
        short = sample_boundary_soft_targets([[[0.,0.],[1.,0.]]],spacing=20.,fade_length=16.)
        self.assertGreater(float(short['weights'].sum()),0.)
        np.testing.assert_array_equal(short['weights'][[0,-1]],0.)

    def test_rejects_unknown_empty_or_nonfinite_geometry_and_invalid_weights(self):
        for segments in [[], [[[1.,2.]]], [[[1.,2.],[1.,2.]]], [[[1.,2.],[float('nan'),3.]]]]:
            with self.assertRaises(ValueError): sample_boundary_soft_targets(segments)
        source = [[[0.,0.],[10.,0.]]]
        for option,value in [('uncertainty_px',-1.),('spacing',0.),('density',-1.),('fade_length',0.),('density',float('inf'))]:
            with self.assertRaises(ValueError): sample_boundary_soft_targets(source,**{option:value})
        with self.assertRaises(ValueError): sample_boundary_soft_targets(source,uncertainty_px=[2.,3.])


if __name__ == '__main__': unittest.main()
