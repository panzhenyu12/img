"""Synthetic behavior tests: no photographs, detector outputs, or result files."""
import copy
import unittest

import numpy as np

from jaw_curve_qp import (evaluate_jaw_curve_profile, evaluate_profile_constraints,
                          solve_jaw_curve_profile)
from jaw_collocation import independent_jaw_probes, sample_jaw_targets
from jaw_target_profile import _validate, apply_profile_samples, profile_contour_targets


def triangle():
    return {
        'contour': [[-40., 0.], [0., 60.], [40., 0.], [40., -20.], [-40., -20.]],
        'frame': {'origin': [0., -10.], 'across': [1., 0.],
                  'down': [0., 1.], 'midline_u': 0.},
        'arcs': [
            {'id': 'left-jaw', 'start': {'edge': 0, 't': 0.},
             'end': {'edge': 1, 't': 0.}, 'direction': 1},
            {'id': 'right-jaw', 'start': {'edge': 2, 't': 0.},
             'end': {'edge': 1, 't': 0.}, 'direction': -1}],
        'selected_rows': [],
    }


def curved(left=None, right=None):
    y = np.arange(0., 61., 10.)
    if left is None:
        left = np.column_stack([-40.+40.*(y/60.)**2, y])
    if right is None:
        right = np.column_stack([40.-40.*(y/60.)**2, y])
    points = np.vstack([left, right[-2::-1], [[40., -20.], [-40., -20.]]])
    chin = len(left)-1; right_mouth = len(left)+len(right)-2
    return {'contour': points.tolist(),
        'frame': {'origin': [0., -10.], 'across': [1., 0.], 'down': [0., 1.], 'midline_u': 0.},
        'arcs': [{'id': 'left-jaw', 'start': {'edge': 0, 't': 0.}, 'end': {'edge': chin, 't': 0.}, 'direction': 1},
                 {'id': 'right-jaw', 'start': {'edge': right_mouth, 't': 0.}, 'end': {'edge': chin, 't': 0.}, 'direction': -1}],
        'selected_rows': [i for i in range(1, right_mouth) if i != chin]}


def source_arc(geometry, name):
    # Independent synthetic fixture walker; tests only use vertex endpoints.
    p = np.asarray(geometry['contour'])
    arc = next(a for a in geometry['arcs'] if a['id'] == name)
    index, end = arc['start']['edge'], arc['end']['edge']; vertices = [p[index]]
    while index != end:
        index = (index+arc['direction']) % len(p); vertices.append(p[index])
    vertices = np.asarray(vertices)
    return vertices, np.r_[0., np.linalg.norm(np.diff(vertices, axis=0), axis=1).cumsum()]


def identity_profiles(geometry):
    return [{'arc_id': name, 'arclength_px': source_arc(geometry, name)[1].tolist(),
             'across_displacement_px': np.zeros(len(source_arc(geometry, name)[1])).tolist()}
            for name in ['left-jaw', 'right-jaw']]


class JawCurveTests(unittest.TestCase):
    def test_generated_profile_runs_through_real_collocation_probe_and_contour_adapter(self):
        geometry = curved()
        result = solve_jaw_curve_profile(geometry, 60., node_spacing_px=5., window_radius_px=8.)
        _validate(result, geometry, 60.)
        for sample in [sample_jaw_targets(geometry, 60., spacing=3.),
                       independent_jaw_probes(geometry, 60., spacing=.7, collocation_spacing=3.)]:
            before = copy.deepcopy(sample)
            applied = apply_profile_samples(sample, result, geometry)
            np.testing.assert_array_equal(applied['source'], sample['source'])
            np.testing.assert_array_equal(applied['weights'], sample['weights'])
            self.assertEqual(applied['controls'], sample['controls'])
            for index, control in enumerate(sample['controls']):
                if control['arc_id'] is None:
                    np.testing.assert_array_equal(applied['target'][index], sample['source'][index])
                else:
                    expected = evaluate_jaw_curve_profile(geometry, result, control['arc_id'],
                                                         [control['arc_arclength_px']])['target'][0]
                    np.testing.assert_allclose(applied['target'][index], expected, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(sample['target'], before['target'])
        actual = profile_contour_targets(geometry, result)
        original = np.asarray(geometry['contour']); expected = original.copy()
        for name in ['left-jaw', 'right-jaw']:
            points, s = source_arc(geometry, name)
            edited = evaluate_jaw_curve_profile(geometry, result, name, s)['target']
            for point, target in zip(points, edited):
                row = np.flatnonzero(np.all(original == point, axis=1))[0]
                expected[row] = target
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_signed_continuous_beta_area_uses_the_true_arc_integral(self):
        # On either straight side, r=40(1-t), dv=60dt. Hence total
        # integral = 2*.13*40*60*64*Beta(4,5) = 16*.13*40*60/35.
        exact = 16*.13*40*60/35
        for amount in [100., -100.]:
            result = solve_jaw_curve_profile(triangle(), amount, node_spacing_px=8.)
            self.assertTrue(result['report']['accepted_numerically'])
            self.assertAlmostEqual(result['required_signed_area_delta_px2'], exact*amount/100., places=10)
            self.assertAlmostEqual(result['candidate_signed_area_delta_px2'], exact*amount/100., places=8)
            for profile in result['profiles']:
                self.assertEqual(profile['across_displacement_px'][0], 0.)
                self.assertEqual(profile['across_displacement_px'][-1], 0.)

    def test_continuous_checker_finds_a_narrow_reverse_window_between_coarse_nodes(self):
        geometry = curved(); profiles = identity_profiles(geometry)
        # A 0.02px notch centered away from a 0.5px grid is a deliberate
        # adversarial piecewise-linear target, not a sampling of the solver.
        left = profiles[0]
        knots = np.unique(np.r_[left['arclength_px'], [31.123, 31.133, 31.143]])
        left['arclength_px'] = knots.tolist()
        left['across_displacement_px'] = np.where(knots == 31.133, 3., 0.).tolist()
        checked = evaluate_profile_constraints(geometry, profiles, window_radius_px=8.)
        self.assertGreater(checked[0]['continuous_violating_subintervals'], 0)
        self.assertEqual(checked[1]['continuous_violating_subintervals'], 0)
        # A conventional 0.5px grid entirely misses the 0.02px notch and its
        # two shifted windows, so this guards the original v1/v2 failure mode.
        original, source_knots = source_arc(geometry, 'left-jaw')
        centers = np.arange(8., source_knots[-1]-8., .5)
        q = []
        for shift in [-8., 0., 8.]:
            s = centers+shift
            xy = np.column_stack([np.interp(s, source_knots, original[:, i]) for i in [0, 1]])
            xy[:, 0] += np.interp(s, knots, left['across_displacement_px'])
            q.append(xy)
        a, b, c = q
        coarse = (c[:, 0]-b[:, 0])*(b[:, 1]-a[:, 1])-(b[:, 0]-a[:, 0])*(c[:, 1]-b[:, 1])
        self.assertGreaterEqual(coarse.min(), -1e-10)

    def test_unknown_flat_and_nonmonotone_regions_are_never_called_convex(self):
        flat = evaluate_profile_constraints(triangle(), identity_profiles(triangle()), window_radius_px=8.)
        for arc in flat:
            self.assertEqual(arc['continuous_convex_subintervals'], 0)
            self.assertIsNone(arc['continuous_min_numerator'])
        geometry = curved(left=np.array([[-40., 0.], [-38., 12.], [-33., 24.],
                                         [-24., 36.], [-13., 48.], [-1., 62.], [0., 60.]]))
        checked = evaluate_profile_constraints(geometry, identity_profiles(geometry), window_radius_px=8.)[0]
        self.assertEqual(len(checked['source_nonmonotone_height_intervals_px']), 1)
        lo, hi = checked['source_nonmonotone_height_intervals_px'][0]
        for a, b in checked['known_source_convex_monotone_center_intervals_px']:
            self.assertTrue(b+8. <= lo+1e-9 or a-8. >= hi-1e-9)

    def test_both_signs_preserve_area_corners_and_independent_dense_convex_windows(self):
        geometry = curved(); before = copy.deepcopy(geometry)
        for amount in [100., -100.]:
            result = solve_jaw_curve_profile(geometry, amount, node_spacing_px=5., window_radius_px=8.)
            self.assertTrue(result['report']['accepted_numerically'], result['report'])
            for profile in result['profiles']:
                original, knots = source_arc(geometry, profile['arc_id'])
                exported = np.asarray(profile['arclength_px'])
                for value in knots:
                    self.assertLess(np.min(abs(exported-value)), 1e-10)
                s = np.arange(8.01237, knots[-1]-8., .01731)
                source = [np.column_stack([np.interp(s+shift, knots, original[:, i]) for i in [0, 1]])
                          for shift in [-8., 0., 8.]]
                target = [evaluate_jaw_curve_profile(geometry, result, profile['arc_id'], s+shift)['target']
                          for shift in [-8., 0., 8.]]
                side = 1. if profile['arc_id'] == 'left-jaw' else -1.
                # Independent cross-multiplied chord slope, using explicit
                # original vertices rather than the controller's checker.
                def turn(q):
                    a, b, c = q
                    return side*((c[:, 0]-b[:, 0])*(b[:, 1]-a[:, 1])
                                 -(b[:, 0]-a[:, 0])*(c[:, 1]-b[:, 1]))
                known = turn(source) > 1e-7
                self.assertGreater(known.sum(), 100)
                self.assertGreaterEqual(turn(target)[known].min(), -1e-7)
        self.assertEqual(geometry, before)

    def test_zero_is_exact_identity_and_hash_rejects_other_geometry(self):
        geometry = curved()
        result = solve_jaw_curve_profile(geometry, 0., node_spacing_px=5., window_radius_px=8.)
        self.assertTrue(result['report']['accepted_numerically'])
        for profile in result['profiles']:
            s = np.linspace(0., profile['arc_length_px'], 111)
            points = evaluate_jaw_curve_profile(geometry, result, profile['arc_id'], s)
            np.testing.assert_array_equal(points['source'], points['target'])
        other = copy.deepcopy(geometry); other['contour'][0][0] -= .001
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluate_jaw_curve_profile(other, result, 'left-jaw', [0.])

    def test_collinear_source_subdivision_and_rigid_frame_changes_preserve_intent(self):
        geometry = curved(); left, _ = source_arc(geometry, 'left-jaw'); right, _ = source_arc(geometry, 'right-jaw')
        subdivided = curved(left=np.insert(left, 2, .37*left[1]+.63*left[2], axis=0),
                             right=np.insert(right, 3, .61*right[2]+.39*right[3], axis=0))
        angle = .41
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        translation = np.array([91., -14.]); moved = copy.deepcopy(geometry)
        moved['contour'] = (np.asarray(moved['contour'])@rotation.T+translation).tolist()
        moved['frame']['origin'] = (np.asarray(moved['frame']['origin'])@rotation.T+translation).tolist()
        for axis in ['across', 'down']:
            moved['frame'][axis] = (np.asarray(moved['frame'][axis])@rotation.T).tolist()
        results = [solve_jaw_curve_profile(g, 60., node_spacing_px=5., window_radius_px=8.)
                   for g in [geometry, subdivided, moved]]
        self.assertTrue(all(r['report']['accepted_numerically'] for r in results))
        for result in results[1:]:
            self.assertAlmostEqual(result['required_signed_area_delta_px2'], results[0]['required_signed_area_delta_px2'], places=10)
        for name in ['left-jaw', 'right-jaw']:
            length = source_arc(geometry, name)[1][-1]
            s = np.linspace(0., length, 301)[1:-1]
            samples = [evaluate_jaw_curve_profile(g, r, name, s) for g, r in zip([geometry, subdivided, moved], results)]
            np.testing.assert_allclose(samples[1]['target'], samples[0]['target'], atol=2e-6, rtol=0)
            np.testing.assert_allclose(samples[2]['target'], samples[0]['target']@rotation.T+translation, atol=2e-6, rtol=0)

    def test_smoothing_redistributes_the_same_area_and_scale_is_physical(self):
        geometry = triangle()
        low, high = [solve_jaw_curve_profile(geometry, 100., node_spacing_px=5.,
                                            smoothing_length_px=smooth, window_radius_px=8.)
                     for smooth in [1., 12.]]
        self.assertTrue(low['report']['accepted_numerically'] and high['report']['accepted_numerically'])
        self.assertAlmostEqual(low['candidate_signed_area_delta_px2'], high['candidate_signed_area_delta_px2'], places=8)
        bending = lambda r: sum(np.sum(np.diff(p['across_displacement_px'], n=2)**2) for p in r['profiles'])
        self.assertLess(bending(high), bending(low))
        enlarged = copy.deepcopy(geometry)
        enlarged['contour'] = (2*np.asarray(geometry['contour'])).tolist()
        enlarged['frame']['origin'] = (2*np.asarray(geometry['frame']['origin'])).tolist()
        scaled = solve_jaw_curve_profile(enlarged, 100., node_spacing_px=10.,
                                         smoothing_length_px=24., window_radius_px=16.)
        self.assertTrue(scaled['report']['accepted_numerically'])
        self.assertAlmostEqual(scaled['candidate_signed_area_delta_px2'], 4*high['candidate_signed_area_delta_px2'], places=7)
        for a, b in zip(high['profiles'], scaled['profiles']):
            np.testing.assert_allclose(b['arclength_px'], 2*np.asarray(a['arclength_px']), atol=1e-12, rtol=0)
            np.testing.assert_allclose(b['across_displacement_px'], 2*np.asarray(a['across_displacement_px']), atol=2e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
