import unittest
import numpy as np

from audit_jaw_retained_bend import finite_chord_measure, retained_bend_margin


class RetainedBendAnalyticTest(unittest.TestCase):
    def test_parabola_three_points_absolute_margin_in_rotated_mirrored_frame(self):
        # u=h^2/8 at h=-2,0,2 has N=2 px² and angle 2*atan(1/4).
        # Halving its u retains N=1; quartering retains N=.5 and fails
        # rho=.5 by an absolute .5 px², while still having positive turn.
        axes = np.array([[.6, .8], [-.8, .6]])
        origin = np.array([125., 74.])
        for side in [1., -1.]:
            measurements = []
            for scale in [1., .5, .25]:
                uv = np.array([[side*.5*scale, -2.], [0., 0.], [side*.5*scale, 2.]])
                points = (uv@axes+origin)[:, None, :]
                measurements.append(finite_chord_measure(points, axes, side))
            source, target, actual = measurements
            np.testing.assert_allclose(source['numerator'], [2.], atol=1e-12, rtol=0.)
            np.testing.assert_allclose(target['numerator'], [1.], atol=1e-12, rtol=0.)
            np.testing.assert_allclose(actual['numerator'], [.5], atol=1e-12, rtol=0.)
            self.assertAlmostEqual(source['angle_degrees'][0], 28.07248693585296, places=10)
            self.assertTrue(source['monotone'][0])
            np.testing.assert_allclose(retained_bend_margin(source['numerator'], target['numerator']), [0.], atol=1e-12, rtol=0.)
            np.testing.assert_allclose(retained_bend_margin(source['numerator'], actual['numerator']), [-.5], atol=1e-12, rtol=0.)


if __name__ == '__main__': unittest.main()
