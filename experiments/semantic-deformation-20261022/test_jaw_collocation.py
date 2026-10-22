"""Continuous jaw intent tests, without photographs or a deformation solve."""
import copy
import unittest

import numpy as np

from jaw_collocation import independent_jaw_probes, sample_jaw_targets


def fixture(points=None):
    p = np.array(points if points is not None else
                 [[-4, 0], [-2, 2], [0, 4], [2, 2], [4, 0], [4, -3], [-4, -3]], float)
    index = lambda xy: int(np.flatnonzero(np.all(p == xy, axis=1))[0])
    chin = index([0, 4])
    return {'contour': p.tolist(),
            'frame': {'origin': [0., -2.], 'across': [1., 0.], 'midline_u': 0.},
            'arcs': [{'id': name, 'start': {'edge': index(xy), 't': 0.},
                      'end': {'edge': chin, 't': 0.}, 'direction': direction}
                     for name, xy, direction in [('left-jaw', [-4, 0], 1), ('right-jaw', [4, 0], -1)]],
            'selected_rows': np.flatnonzero((p[:, 1] > 0) & (np.arange(len(p)) != chin)).tolist()}


def arc_rows(result, name):
    return np.array([i for i, row in enumerate(result['controls']) if row['arc_id'] == name])


class JawCollocationTests(unittest.TestCase):
    def test_trapezoid_weights_preserve_arc_measure_at_different_densities(self):
        geometry = fixture(); before = copy.deepcopy(geometry)
        length = 4*np.sqrt(2.)
        for spacing in [4., 2., .5]:
            result = sample_jaw_targets(geometry, 100., spacing=spacing)
            for name in ['left-jaw', 'right-jaw']:
                ids = arc_rows(result, name); weights = result['weights'][ids]
                n = int(np.ceil(length/spacing))
                self.assertEqual(len(ids), n+1)
                self.assertAlmostEqual(weights.sum(), .5*length, places=13)
                self.assertAlmostEqual(weights[0], .5*length/n/2, places=13)
                self.assertAlmostEqual(weights[-1], weights[0], places=13)
                # A constant squared residual integrates identically, regardless
                # of how many collocation samples approximate the same arc.
                self.assertAlmostEqual(float(weights @ np.full(n+1, 9.)), 4.5*length, places=12)
        self.assertEqual(geometry, before)

    def test_continuous_midpoint_has_the_original_gain_and_both_signed_directions(self):
        geometry = fixture(); spacing = 2*np.sqrt(2.)
        for amount in [100., -100.]:
            result = sample_jaw_targets(geometry, amount, spacing=spacing)
            for name, source, sign in [('left-jaw', [-2., 2.], 1), ('right-jaw', [2., 2.], -1)]:
                ids = arc_rows(result, name)
                np.testing.assert_allclose(result['source'][ids[1]], source, atol=1e-14, rtol=0)
                np.testing.assert_allclose(result['target'][ids[1]],
                                           np.array(source)+[sign*.26*amount/100., 0.], atol=1e-14, rtol=0)
                self.assertEqual(result['controls'][ids[1]]['point_gain'], 1.)
                np.testing.assert_array_equal(result['target'][ids[[0, -1]]], result['source'][ids[[0, -1]]])

    def test_zero_request_and_original_outside_rows_are_exact(self):
        geometry = fixture(); original = np.asarray(geometry['contour'])
        for amount in [0., 100.]:
            result = sample_jaw_targets(geometry, amount)
            for i, control in enumerate(result['controls']):
                if control['original_index'] is not None:
                    np.testing.assert_array_equal(result['source'][i], original[control['original_index']])
                    np.testing.assert_array_equal(result['target'][i], result['source'][i])
                    self.assertEqual(result['weights'][i], 4.)
            if amount == 0.:
                np.testing.assert_array_equal(result['target'], result['source'])
                probes = independent_jaw_probes(geometry, amount)
                np.testing.assert_array_equal(probes['target'], probes['source'])

    def test_adding_detector_rows_on_an_existing_edge_does_not_change_arc_intent(self):
        sparse = fixture()
        refined = fixture([[-4, 0], [-3, 1], [-2, 2], [-.5, 3.5], [0, 4],
                           [.5, 3.5], [1, 3], [2, 2], [4, 0], [4, -3], [-4, -3]])
        a, b = [sample_jaw_targets(g, 100., spacing=.7) for g in [sparse, refined]]
        for name in ['left-jaw', 'right-jaw']:
            ia, ib = arc_rows(a, name), arc_rows(b, name)
            for key in ['source', 'target', 'weights']:
                np.testing.assert_allclose(a[key][ia], b[key][ib], atol=2e-14, rtol=0)

    def test_rigid_rotation_and_translation_preserve_the_same_edit(self):
        geometry = fixture(); angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        translation = np.array([71., 40.]); transformed = copy.deepcopy(geometry)
        transformed['contour'] = (np.asarray(geometry['contour'])@rotation.T+translation).tolist()
        transformed['frame']['origin'] = (np.asarray(geometry['frame']['origin'])@rotation.T+translation).tolist()
        transformed['frame']['across'] = (np.asarray(geometry['frame']['across'])@rotation.T).tolist()
        for method in [sample_jaw_targets, independent_jaw_probes]:
            a, b = method(geometry, 60.), method(transformed, 60.)
            for key in ['source', 'target']:
                np.testing.assert_allclose(b[key], a[key]@rotation.T+translation, atol=5e-13, rtol=0)
            np.testing.assert_allclose(b['weights'], a['weights'], atol=2e-13, rtol=0)

    def test_quadratic_residual_energy_converges_to_the_known_line_integral(self):
        # Residual r(s)=s/L gives integral_0^L r(s)^2 ds = L/3.
        # Composite trapezoid's exact quadratic excess is L/(6*n^2).
        geometry = fixture(); length = 4*np.sqrt(2.)
        previous_error = np.inf
        for spacing in [4., 2., 1., .5]:
            result = sample_jaw_targets(geometry, 100., spacing=spacing)
            ids = arc_rows(result, 'left-jaw')
            t = np.array([result['controls'][i]['arc_t'] for i in ids])
            energy = float(result['weights'][ids] @ t**2); n = len(ids)-1
            exact = .5*length/3.; excess = .5*length/(6*n*n)
            self.assertAlmostEqual(energy, exact+excess, places=13)
            self.assertLess(energy-exact, previous_error)
            previous_error = energy-exact

    def test_probe_midpoints_do_not_reuse_training_nodes_even_for_commensurate_lengths(self):
        geometry = fixture()
        # Naively choosing ceil(L/probe_spacing) midpoint bins can hit a
        # training node (e.g. 72 training bins against 286 probe bins).
        length = 4*np.sqrt(2.); train_spacing = length/72; probe_spacing = length/286
        training = sample_jaw_targets(geometry, 100., spacing=train_spacing)
        probes = independent_jaw_probes(geometry, 100., spacing=probe_spacing,
                                        collocation_spacing=train_spacing)
        for name in ['left-jaw', 'right-jaw']:
            a, b = arc_rows(training, name), arc_rows(probes, name)
            sa = np.array([training['controls'][i]['arc_arclength_px'] for i in a])
            sb = np.array([probes['controls'][i]['arc_arclength_px'] for i in b])
            self.assertTrue(np.all((sb > 0) & (sb < length)))
            self.assertGreater(float(np.min(abs(sa[:, None]-sb))), 1e-10)
            self.assertAlmostEqual(probes['weights'][b].sum(), .5*length, places=12)
        self.assertTrue(all(c['original_index'] is None for c in probes['controls']))
        self.assertTrue(all(a['actual_spacing_px'] <= probe_spacing for a in probes['report']['arcs']))

    def test_invalid_phases_cannot_turn_fit_nodes_into_evaluation_probes(self):
        with self.assertRaisesRegex(ValueError, 'phase 0'):
            sample_jaw_targets(fixture(), 100., phase=.5)
        for phase in [0., 1., np.nan]:
            with self.assertRaises(ValueError):
                independent_jaw_probes(fixture(), 100., phase=phase)
        for spacing in [0., -1., np.inf]:
            with self.assertRaises(ValueError):
                sample_jaw_targets(fixture(), 100., spacing=spacing)


if __name__ == '__main__':
    unittest.main()
