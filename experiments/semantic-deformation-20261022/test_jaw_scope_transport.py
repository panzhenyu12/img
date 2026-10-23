"""Analytical scope transport; no image, optimization or field evaluation."""
import copy
import hashlib
import json
import unittest

import numpy as np

from jaw_scope_transport import extend_profile_scope
from jaw_target_profile import profile_contour_targets


def profile_for(geometry, lengths, starts, knots, magnitudes, amount=100.):
    result = {'schema': 'jaw-arclength-piecewise-linear-profile-v1',
        'controller': 'synthetic-rejected-prefix', 'amount': amount,
        'source_geometry': copy.deepcopy(geometry),
        'source_geometry_sha256': hashlib.sha256(json.dumps(geometry, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        'across_axis': [1., 0.], 'profiles': [], 'solver_status': 'AlmostSolved',
        'accepted_target_constraints': False, 'diagnostic_only': True,
        'executable': False, 'image_approval': False}
    for i, (name, direction, sign) in enumerate([('left-jaw', 1, 1.), ('right-jaw', -1, -1.)]):
        result['profiles'].append({'arc_id': name, 'direction': direction,
            'original_contour_start_arclength_px': starts[i], 'arc_length_px': lengths[i],
            'original_source_vertex_arclength_px': knots,
            'arclength_px': sorted(set(knots+[lengths[i]/2.])),
            'across_displacement_px': (sign*np.asarray(magnitudes)).tolist()})
    return result


def straight_fixture():
    old = {'contour': [[-6., -2.], [-3., 2.], [0., 6.], [3., 2.], [6., -2.], [6., -6.], [-6., -6.]],
        'frame': {'origin': [0., 0.], 'across': [1., 0.], 'down': [0., 1.], 'midline_u': 0.},
        'arcs': [{'id': 'left-jaw', 'start': {'edge': 1, 't': 0.}, 'end': {'edge': 2, 't': 0.}, 'direction': 1},
                 {'id': 'right-jaw', 'start': {'edge': 3, 't': 0.}, 'end': {'edge': 2, 't': 0.}, 'direction': -1}],
        'selected_rows': []}
    new = copy.deepcopy(old)
    new['arcs'][0]['start']['edge'] = 0
    new['arcs'][1]['start']['edge'] = 4
    new['selected_rows'] = [1, 3]
    return old, new, profile_for(old, [5., 5.], [5., 15.], [0., 5.], [0., 1., 0.])


def corner_fixture():
    old = {'contour': [[-4., -4.], [-4., 0.], [-4., 4.], [0., 4.], [4., 4.], [4., 0.], [4., -4.]],
        'frame': {'origin': [0., 0.], 'across': [1., 0.], 'down': [0., 1.], 'midline_u': 0.},
        'arcs': [{'id': 'left-jaw', 'start': {'edge': 1, 't': 0.}, 'end': {'edge': 3, 't': 0.}, 'direction': 1},
                 {'id': 'right-jaw', 'start': {'edge': 5, 't': 0.}, 'end': {'edge': 3, 't': 0.}, 'direction': -1}],
        'selected_rows': [2, 4]}
    new = copy.deepcopy(old)
    new['arcs'][0]['start']['edge'] = 0
    new['arcs'][1]['start']['edge'] = 6
    new['selected_rows'] = [1, 2, 4, 5]
    return old, new, profile_for(old, [8., 8.], [4., 20.], [0., 4., 8.], [0., 1., 0.])


class ScopeTransportTests(unittest.TestCase):
    def test_straight_source_arc_has_closed_form_scale_and_preserved_side_area(self):
        old, new, profile = straight_fixture()
        result = extend_profile_scope(profile, old, new)
        for row, report, sign in zip(result['profiles'], result['transport_report']['arcs'], [1., -1.]):
            np.testing.assert_array_equal(row['arclength_px'], [0., 5., 10.])
            np.testing.assert_allclose(row['across_displacement_px'], [0., sign*.5, 0.], atol=1e-14, rtol=0)
            self.assertAlmostEqual(report['scale'], .5, places=14)
            self.assertAlmostEqual(report['old_signed_area_px2'], 2., places=14)
            self.assertAlmostEqual(report['new_signed_area_px2'], 2., places=14)

    def test_source_corner_and_mapped_profile_peak_are_both_integrated_exactly(self):
        old, new, profile = corner_fixture()
        result = extend_profile_scope(profile, old, new)
        for row, report, sign in zip(result['profiles'], result['transport_report']['arcs'], [1., -1.]):
            # New peak at s=6 falls inside the vertical edge; the source turns
            # horizontal at s=8 inside a descending profile interval.
            np.testing.assert_array_equal(row['arclength_px'], [0., 4., 6., 8., 12.])
            np.testing.assert_allclose(row['across_displacement_px'],
                sign*np.array([0., 2./7., 3./7., 2./7., 0.]), atol=1e-14, rtol=0)
            self.assertAlmostEqual(report['old_signed_area_px2'], 2., places=14)
            # Integral on [0,6] is 3; on [6,8] it is 5/3; horizontal rest adds 0.
            self.assertAlmostEqual(report['new_unscaled_signed_area_px2'], 14./3., places=14)
            self.assertAlmostEqual(report['scale'], 3./7., places=14)
            self.assertNotAlmostEqual(report['scale'], 8./12., places=6)
            self.assertAlmostEqual(report['new_signed_area_px2'], 2., places=14)

    def test_negative_amount_and_mirrored_sides_preserve_sign_without_reapplying_amount(self):
        old, new, profile = corner_fixture()
        profile['amount'] = -100.
        for row in profile['profiles']:
            row['across_displacement_px'] = (-np.array(row['across_displacement_px'])).tolist()
        result = extend_profile_scope(profile, old, new)
        left, right = result['profiles']
        np.testing.assert_array_equal(left['across_displacement_px'], -np.array(right['across_displacement_px']))
        self.assertEqual(result['amount'], -100.)
        self.assertLess(left['across_displacement_px'][2], 0.)
        for report in result['transport_report']['arcs']:
            self.assertAlmostEqual(report['scale'], 3./7., places=14)
            self.assertAlmostEqual(report['old_signed_area_px2'], -2., places=14)
            self.assertAlmostEqual(report['new_signed_area_px2'], -2., places=14)

    def test_zero_profile_remains_exact_identity(self):
        old, new, profile = corner_fixture()
        profile['amount'] = 0.
        for row in profile['profiles']:
            row['across_displacement_px'] = [0.]*len(row['arclength_px'])
        result = extend_profile_scope(profile, old, new)
        for row, report in zip(result['profiles'], result['transport_report']['arcs']):
            np.testing.assert_array_equal(row['across_displacement_px'], 0.)
            self.assertEqual(report['old_signed_area_px2'], 0.)
            self.assertEqual(report['new_signed_area_px2'], 0.)
            self.assertTrue(report['identity'])
        np.testing.assert_array_equal(profile_contour_targets(new, result), new['contour'])

    def test_signed_area_uses_the_reviewed_across_axis_on_a_rotated_source(self):
        old, new, profile = straight_fixture()
        rotation = np.array([[.6, -.8], [.8, .6]])
        for geometry in [old, new]:
            geometry['contour'] = (np.array(geometry['contour']) @ rotation.T).tolist()
            geometry['frame']['across'] = [.6, .8]
            geometry['frame']['down'] = [-.8, .6]
        profile['source_geometry'] = copy.deepcopy(old)
        profile['source_geometry_sha256'] = hashlib.sha256(json.dumps(old, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        profile['across_axis'] = [.6, .8]
        result = extend_profile_scope(profile, old, new)
        for report in result['transport_report']['arcs']:
            self.assertAlmostEqual(report['old_signed_area_px2'], 2., places=12)
            self.assertAlmostEqual(report['new_signed_area_px2'], 2., places=12)
            self.assertAlmostEqual(report['scale'], .5, places=12)

    def test_reversed_contour_traversal_keeps_geometry_and_flips_oriented_area(self):
        old, new, profile = straight_fixture()
        for geometry in [old, new]:
            count = len(geometry['contour'])
            geometry['contour'] = geometry['contour'][::-1]
            geometry['selected_rows'] = sorted(count-1-i for i in geometry['selected_rows'])
            for arc in geometry['arcs']:
                arc['direction'] *= -1
                for boundary in ['start', 'end']:
                    arc[boundary]['edge'] = count-1-arc[boundary]['edge']
        profile['source_geometry'] = copy.deepcopy(old)
        profile['source_geometry_sha256'] = hashlib.sha256(json.dumps(old, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        points = np.asarray(old['contour'])
        cumulative = np.r_[0., np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).cumsum()]
        for row, arc in zip(profile['profiles'], old['arcs']):
            row['direction'] = arc['direction']
            row['original_contour_start_arclength_px'] = cumulative[arc['start']['edge']]
        result = extend_profile_scope(profile, old, new)
        for row, report, sign in zip(result['profiles'], result['transport_report']['arcs'], [1., -1.]):
            np.testing.assert_allclose(row['across_displacement_px'], [0., sign*.5, 0.], atol=1e-14, rtol=0)
            self.assertAlmostEqual(report['old_signed_area_px2'], -2., places=14)
            self.assertAlmostEqual(report['new_signed_area_px2'], -2., places=14)

    def test_nonextended_scope_and_changed_source_rows_are_rejected(self):
        old, new, profile = straight_fixture()
        with self.assertRaisesRegex(ValueError, 'strictly extend'):
            extend_profile_scope(profile, old, old)
        shorter = copy.deepcopy(old)
        shorter['arcs'][0]['start']['t'] = .5
        with self.assertRaisesRegex(ValueError, 'strictly extend'):
            extend_profile_scope(profile, old, shorter)
        changed = copy.deepcopy(new)
        changed['contour'][0][0] -= .1
        with self.assertRaisesRegex(ValueError, 'unchanged source contour'):
            extend_profile_scope(profile, old, changed)

    def test_original_inputs_and_failed_optimizer_status_cannot_be_promoted(self):
        old, new, profile = corner_fixture()
        before = copy.deepcopy((old, new, profile))
        result = extend_profile_scope(profile, old, new)
        self.assertEqual((old, new, profile), before)
        np.testing.assert_array_equal(result['source_geometry']['contour'], old['contour'])
        self.assertFalse(result['accepted_target_constraints'])
        self.assertFalse(result['executable'])
        self.assertFalse(result['image_approval'])
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual(result['solver_status'], 'NotRun')
        provenance = result['input_profile_provenance']['acceptance']
        self.assertEqual(provenance['solver_status'], 'AlmostSolved')
        self.assertFalse(provenance['accepted_target_constraints'])
        self.assertFalse(result['transport_report']['target_curve_set_preserved'])
        self.assertNotIn('source_protocol', result)


if __name__ == '__main__':
    unittest.main()
