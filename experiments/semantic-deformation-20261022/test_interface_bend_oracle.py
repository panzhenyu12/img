"""Coordinate-level analytic checks, independent of the solver chord helper."""
import unittest

import numpy as np

from interface_bend_oracle import chord_metrics, band_margins, sample_scalar


class InterfaceBendOracleTests(unittest.TestCase):
    def test_band_detects_flattening_and_overbending_for_both_source_signs(self):
        for sign in [-1., 1.]:
            source = np.array([[[0., 0.], [0., 1.], [sign * .1, 2.]]])
            original = chord_metrics(source, np.eye(2))
            self.assertAlmostEqual(original['numerator_px2'][0], sign * .1)
            for scale, accepted in [(.1, False), (.75, True), (1.25, True), (2., False)]:
                target = source.copy(); target[..., 0] *= scale
                changed = chord_metrics(target, np.eye(2))
                low, high, known = band_margins(original, changed, (.5, 1.5))
                self.assertTrue(known[0])
                self.assertEqual(bool(low[0] >= 0. and high[0] >= 0.), accepted)

    def test_equal_height_numerator_band_does_not_certify_turn_angle_ratio(self):
        source = np.array([[[0., 0.], [.45, 1.], [1., 2.]]])
        target = np.array([[[0., 0.], [.075, 1.], [.30, 2.]]])
        old = chord_metrics(source, np.eye(2)); new = chord_metrics(target, np.eye(2))
        np.testing.assert_allclose(new['numerator_px2'] / old['numerator_px2'], 1.5)
        low, high, _ = band_margins(old, new, (.5, 1.5))
        self.assertGreaterEqual(low[0], 0.); self.assertGreaterEqual(high[0], -1e-15)
        self.assertGreater(new['turn_degrees'][0] / old['turn_degrees'][0], 1.8)
        # u(v)=-.4*v+.025*v*v gives b in [-.4,-.3], D=1. Its
        # inverse eigenvalues are [1-|b|/2,1+|b|/2], skew=|b|/2:
        # this counterexample survives the existing geometry contract.
        self.assertGreaterEqual(1. - .4 / 2., .52)
        self.assertLessEqual(1. + .4 / 2., 1.8)
        self.assertLessEqual(.4 / 2., .24)

    def test_near_zero_source_is_unassigned_not_a_division_by_zero(self):
        original = chord_metrics(np.array([[[0., 0.], [0., 1.], [0., 2.]]]), np.eye(2))
        _, _, known = band_margins(original, original, (.5, 1.5))
        self.assertFalse(known[0])

    def test_bilinear_scalar_reads_original_cell_and_rejects_outside(self):
        yy, xx = np.mgrid[:4, :5]; field = 2.*xx + 3.*yy + xx*yy
        p = np.array([[.2, .7], [2.1, 1.4], [4., 3.]])
        np.testing.assert_allclose(sample_scalar(field, p), 2.*p[:, 0]+3.*p[:, 1]+p[:, 0]*p[:, 1])
        with self.assertRaisesRegex(ValueError, 'outside'):
            sample_scalar(field, np.array([[-.1, 0.]]))


if __name__ == '__main__':
    unittest.main()
