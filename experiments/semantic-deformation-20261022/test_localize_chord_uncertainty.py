"""Analytic observation bounds; no photographs or optimization solves."""
import itertools
import unittest

import numpy as np

from localize_chord_uncertainty import localize_chord_uncertainty, chord_uncertainty_from_edge_samples


class ChordUncertaintyTests(unittest.TestCase):
    def test_tight_interval_matches_all_original_measurement_box_vertices(self):
        points = np.array([[0., 0.], [1., 1.], [1., 3.], [-1., 4.]])
        sample_y = np.array([.5, 2., 3.5]); epsilon = np.array([.25, .5, .125, .375])
        result = localize_chord_uncertainty(points, sample_y, epsilon)
        enumerated = []
        for signs in itertools.product([-1., 1.], repeat=4):
            x = np.interp(sample_y, points[:, 1], points[:, 0]+epsilon*np.array(signs))
            numerator = (x[2]-x[1])*1.5-(x[1]-x[0])*1.5
            enumerated.append(numerator)
        self.assertEqual(result['nominal_N'], -2.25)
        self.assertEqual(result['radius'], .9375)
        self.assertEqual(result['interval'], [min(enumerated), max(enumerated)])
        self.assertEqual(result['reliable_sign'], -1)
        self.assertEqual(result['source_coefficients'], [.75, -.75, -.75, .75])
        self.assertEqual(result['flipping_epsilon'], .75)

    def test_shared_original_nodes_cancel_on_one_straight_edge(self):
        points = [[2., 0.], [-3., 2.]]
        result = localize_chord_uncertainty(points, [.25, 1., 1.75], [100., 20.])
        # All three locations remain on the same perturbed original edge.
        # Independent errors at the three interpolated samples would invent
        # a positive radius and fail this exact cancellation.
        self.assertEqual(result['source_coefficients'], [0., 0.])
        self.assertEqual(result['radius'], 0.)
        self.assertEqual(result['interval'], [0., 0.])
        self.assertEqual(result['reliable_sign'], 0)
        self.assertEqual(result['flipping_epsilon'], 0.)

    def test_translation_and_physical_scale_preserve_declared_units(self):
        points = np.array([[0., 0.], [1., 1.], [1., 3.], [-1., 4.]])
        samples = np.array([.5, 2., 3.5]); epsilon = np.array([.25, .5, .125, .375])
        base = localize_chord_uncertainty(points, samples, epsilon)
        translated = localize_chord_uncertainty(points+[128., -256.], samples-256., epsilon)
        for key in ['nominal_N', 'radius', 'interval', 'flipping_epsilon', 'reliable_sign']:
            self.assertEqual(translated[key], base[key])
        scaled = localize_chord_uncertainty(4.*points, 4.*samples, 4.*epsilon)
        self.assertEqual(scaled['nominal_N'], 16.*base['nominal_N'])
        self.assertEqual(scaled['radius'], 16.*base['radius'])
        np.testing.assert_array_equal(scaled['interval'], 16.*np.asarray(base['interval']))
        self.assertEqual(scaled['flipping_epsilon'], 4.*base['flipping_epsilon'])
        self.assertEqual(scaled['reliable_sign'], base['reliable_sign'])

    def test_zero_uncertainty_restores_nominal_measurement_without_certifying_truth(self):
        points = np.array([[0., 0.], [1., 1.], [0., 2.]])
        original = points.copy()
        result = localize_chord_uncertainty(points, [0., 1., 2.], 0.)
        self.assertEqual(result['nominal_N'], -2.)
        self.assertEqual(result['interval'], [-2., -2.])
        self.assertEqual(result['radius'], 0.)
        self.assertEqual(result['reliable_sign'], -1)
        self.assertFalse(result['uncertainty_calibrated'])
        self.assertIn('not ground truth', result['uncertainty_status'])
        self.assertFalse(result['solver_gate_changed'])
        np.testing.assert_array_equal(points, original)
        calibrated = localize_chord_uncertainty(points, [0., 1., 2.], 0., calibrated=True)
        self.assertEqual(calibrated['interval'], result['interval'])
        self.assertEqual(calibrated['uncertainty_status'], 'caller declares calibrated')

    def test_exact_touch_and_immediately_adjacent_uncertainty_intervals(self):
        points = [[0., 0.], [1., 1.], [0., 2.]]
        below = localize_chord_uncertainty(points, [0., 1., 2.], np.nextafter(.5, 0.))
        touch = localize_chord_uncertainty(points, [0., 1., 2.], .5)
        above = localize_chord_uncertainty(points, [0., 1., 2.], np.nextafter(.5, 1.))
        self.assertEqual(touch['interval'], [-4., 0.])
        self.assertEqual(touch['reliable_sign'], 0)
        self.assertEqual(touch['flipping_epsilon'], .5)
        self.assertEqual(touch['current_epsilon_scale_to_touch_zero'], 1.)
        self.assertLess(below['interval'][1], 0.)
        self.assertEqual(below['reliable_sign'], -1)
        self.assertGreater(above['interval'][1], 0.)
        self.assertEqual(above['reliable_sign'], 0)

    def test_original_node_and_adjacent_query_select_exact_source_intervals(self):
        points = [[0., 0.], [1., 1.], [0., 2.]]
        exact = localize_chord_uncertainty(points, [0., 1., 2.], .25)
        np.testing.assert_array_equal(exact['sample_weights'], np.eye(3))
        below = localize_chord_uncertainty(points, [0., np.nextafter(1., 0.), 2.], .25)
        above = localize_chord_uncertainty(points, [0., np.nextafter(1., 2.), 2.], .25)
        self.assertGreater(below['sample_weights'][1][0], 0.)
        self.assertEqual(below['sample_weights'][1][2], 0.)
        self.assertEqual(above['sample_weights'][1][0], 0.)
        self.assertGreater(above['sample_weights'][1][2], 0.)
        for sample in [np.nextafter(0., -1.), np.nextafter(2., 3.)]:
            with self.assertRaisesRegex(ValueError, 'inside'):
                localize_chord_uncertainty(points, [sample, 1., 1.], 0.)

    def test_frozen_edge_weights_agree_and_do_not_reparameterize_uncertain_arclength(self):
        points = [[0., 0.], [1., 1.], [1., 3.], [-1., 4.]]
        direct = localize_chord_uncertainty(points, [.5, 2., 3.5], .25)
        frozen = chord_uncertainty_from_edge_samples(points, [0, 1, 2], [.5, .5, .5], .25)
        for key in ['nominal_N', 'radius', 'interval', 'source_coefficients', 'reliable_sign']:
            self.assertEqual(frozen[key], direct[key])
        self.assertIn('no uncertain arclength reparameterization', frozen['sampling'])
        reverse = localize_chord_uncertainty(points, [3.5, 2., .5], .25)
        self.assertEqual(reverse['nominal_N'], -direct['nominal_N'])
        self.assertEqual(reverse['interval'], [-direct['interval'][1], -direct['interval'][0]])

    def test_invalid_measurement_domains_are_rejected(self):
        for points in [[[0., 0.], [1., 0.]], [[0., 1.], [1., 0.]], [[np.nan, 0.], [1., 1.]]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                localize_chord_uncertainty(points, [0., .5, 1.], .25)
        for epsilon in [-1., np.inf, [1.], [0., np.nan]]:
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                localize_chord_uncertainty([[0., 0.], [1., 1.]], [0., .5, 1.], epsilon)
        with self.assertRaises(ValueError):
            chord_uncertainty_from_edge_samples([[0., 0.], [1., 1.]], [0, 0, 1], [0., .5, 1.], 0.)


if __name__ == '__main__':
    unittest.main()
