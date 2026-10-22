"""Source-only region associations must not invent anatomy or move pixels."""
import copy
import unittest

import numpy as np

from source_regions import associate_source, verify_source_identity, verify_history, ROOT


OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def fixture():
    corners = np.array([[-100., -150.], [100., -150.], [100., 220.], [-100., 220.]])
    contour = np.concatenate([a + np.arange(32)[:, None]/32*(b-a)
                              for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    t = -np.pi/2 + np.arange(36)*np.pi/18
    oval = np.column_stack([80*np.cos(t), 50+150*np.sin(t)])
    def ellipse(cx, cy, rx, ry, count):
        t = np.arange(count)*2*np.pi/count
        return np.column_stack([cx+rx*np.cos(t), cy+ry*np.sin(t)])
    features = {'leftEye': ellipse(-30, 0, 10, 5, 16),
                'rightEye': ellipse(30, 0, 10, 5, 15),
                'lips': ellipse(0, 100, 20, 8, 20)}
    # Make the known frame exact even though one polygon has an odd point count.
    frame = {'origin': [0., 0.], 'across': [1., 0.], 'down': [0., 1.],
             'bounds': [-80., -100., 80., 200.]}
    return contour, oval, features, frame, OVAL.copy()


class SourceRegionTests(unittest.TestCase):
    def test_continuous_arcs_and_original_rows_are_preserved(self):
        args = fixture(); before = copy.deepcopy(args)
        result = associate_source(*args)
        for a, b in zip(args[:2], before[:2]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(result['source_contour'], args[0])
        self.assertEqual(len(result['row_regions']), 128)
        for side in ['left', 'right']:
            boundaries = result['sides'][side]['boundaries']
            np.testing.assert_allclose(boundaries['eye_height']['source'], [-100 if side == 'left' else 100, 0], atol=1e-10)
            np.testing.assert_allclose(boundaries['mouth_height']['source'], [-100 if side == 'left' else 100, 100], atol=1e-10)
            self.assertAlmostEqual(result['sides'][side]['regions'][0]['length_px'], 100.)
            self.assertAlmostEqual(result['sides'][side]['regions'][1]['length_px'], 220.)
            self.assertFalse(result['sides'][side]['automatic_edit_authorized'])
            self.assertEqual(result['sides'][side]['visibility_status'], 'unknown')
        self.assertEqual(result['anatomical_landmarks']['cheekbone']['status'], 'unknown')

    def test_mouth_outside_source_side_is_rejected(self):
        args = list(fixture())
        args[2]['lips'][:, 0] *= 6  # Keep its center/frame valid, cross both sides.
        with self.assertRaisesRegex(ValueError, 'feature.*outside'):
            associate_source(*args)

    def test_bad_indices_missing_features_and_changed_frame_are_rejected(self):
        for corruption, expected in [('indices', 'canonical'), ('missing', 'missing'),
                                     ('side', 'identity/side'), ('frame', 'frame'), ('order', 'out-of-order')]:
            args = list(fixture())
            if corruption == 'indices': args[4][0], args[4][1] = args[4][1], args[4][0]
            if corruption == 'missing': del args[2]['rightEye']
            if corruption == 'side':
                args[2]['leftEye'][:, 0] += 60; args[2]['rightEye'][:, 0] -= 60
            if corruption == 'frame': args[3]['origin'][0] = 1.
            if corruption == 'order': args[0][[10, 80]] = args[0][[80, 10]]
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, expected):
                associate_source(*args)

    def test_ambiguous_multiple_source_sections_are_rejected(self):
        args = list(fixture())
        corners = np.array([[-100., -150.], [100., -150.], [100., 220.], [20., 220.],
                            [20., -20.], [-20., -20.], [-20., 220.], [-100., 220.]])
        args[0] = np.concatenate([a+np.arange(16)[:, None]/16*(b-a)
                                  for a, b in zip(corners, np.roll(corners, -1, axis=0))])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            associate_source(*args)

    def test_rigid_photo_rotation_preserves_arclength_and_original_frame(self):
        args = list(fixture()); baseline = associate_source(*args)
        angle = .31; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift = np.array([300., 250.])
        args[0] = args[0]@rotation.T+shift; args[1] = args[1]@rotation.T+shift
        args[2] = {key: value@rotation.T+shift for key, value in args[2].items()}
        args[3] = {'origin': shift.tolist(), 'across': rotation[:, 0].tolist(),
                   'down': rotation[:, 1].tolist(), 'bounds': args[3]['bounds']}
        result = associate_source(*args)
        self.assertEqual(result['source_frame_unchanged'], args[3])
        for side in ['left', 'right']:
            for actual, expected in zip(result['sides'][side]['regions'], baseline['sides'][side]['regions']):
                self.assertAlmostEqual(actual['length_px'], expected['length_px'])

    def test_actual_full36_reordering_is_rejected_even_with_correct_id_list(self):
        _, oval, features, _, _ = fixture()
        # Quarter-pixel source positions test the half-pixel convention without
        # introducing irrelevant decimal round-trip noise into exact identity.
        oval = np.round(oval*4)/4
        features = {key: np.round(value*4)/4 for key, value in features.items()}
        # Graph uses texel-center coordinates; audit stores photograph indices.
        points = lambda rows: [{'x': float(p[0]+.5), 'y': float(p[1]+.5)} for p in rows]
        reference = {'landmarks': points(oval), 'semanticMasks':
                     {key: {'shape': 'polygon', 'points': points(value)} for key, value in features.items()}}
        audit = {'source_oval': oval.tolist(), 'source_eyes': [features['leftEye'].tolist(), features['rightEye'].tolist()],
                 'native_and_photo_coordinate_conversion': {'coordinate_convention':
                   'native absolute texel centres +.5 converted once to photograph indices'}}
        actual, _ = verify_source_identity(reference, reference, audit, [1, 1])
        np.testing.assert_array_equal(actual, oval)
        twice = copy.deepcopy(audit); twice['source_oval'] = (oval-.5).tolist()
        with self.assertRaisesRegex(ValueError, 'frozen source audit'):
            verify_source_identity(reference, reference, twice, [1, 1])
        corrupt = copy.deepcopy(reference)
        corrupt['landmarks'][4], corrupt['landmarks'][5] = corrupt['landmarks'][5], corrupt['landmarks'][4]
        with self.assertRaisesRegex(ValueError, 'identity/order'):
            verify_source_identity(corrupt, reference, audit, [1, 1])
        corrupt = copy.deepcopy(reference); corrupt['semanticMasks']['lips']['points'].reverse()
        with self.assertRaisesRegex(ValueError, 'polygon identity'):
            verify_source_identity(corrupt, reference, audit, [1, 1])

    def test_historical_definition_must_match_all_indices(self):
        text = (ROOT/'apps/web/src/core/landmarks.ts').read_text()
        verify_history(text, text)
        with self.assertRaisesRegex(ValueError, 'historical canonical ordering'):
            verify_history(text, text.replace('10, 338, 297', '10, 297, 338'))


if __name__ == '__main__':
    unittest.main()
