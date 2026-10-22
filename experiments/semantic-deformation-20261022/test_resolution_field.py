"""Pixel-center geometry transfer oracles; no photographs or new field solve."""
import unittest

import numpy as np

from resolution_field import lift_inverse_field, shared_fixed_support


class ResolutionFieldTests(unittest.TestCase):
    def fixture(self, nonlinear=False):
        y, x = np.mgrid[:15, :13]
        if nonlinear:
            field = np.stack([.004*x*y+.01*x, -.003*x*y+.02*y], -1)
        else:
            field = np.stack([.08*x-.03*y+.12, -.02*x+.06*y-.07], -1)
        field[:2] = field[-2:] = 0.
        field[:, :2] = field[:, -2:] = 0.
        return field.astype('<f4')

    def at(self, result, x, y):
        return result['field'][y-result['crop'][1], x-result['crop'][0]].astype(float)

    def test_affine_reference_map_has_correct_pixel_center_phase(self):
        field = self.fixture()
        result = lift_inverse_field(field, [3, 4, 13, 15], (20, 24), (50, 60))
        self.assertEqual(result['scale'], 2.5)
        self.assertEqual(result['crop'], [8, 10, 32, 37])
        for x, y in [(20, 25), (26, 30), (31, 35)]:
            # Analytic displacement evaluated in the reference crop frame.
            u, v = (np.array([x, y], float)+.5)/2.5-.5-[3., 4.]
            expected = 2.5*np.array([.08*u-.03*v+.12, -.02*u+.06*v-.07])
            np.testing.assert_allclose(self.at(result, x, y), expected, rtol=0, atol=2e-7)

    def test_bilinear_nonlinear_map_preserves_source_correspondence_at_output_nodes(self):
        result = lift_inverse_field(self.fixture(True), [3, 4, 13, 15], (20, 24), (50, 60))
        for x, y in [(20, 25), (26, 30), (31, 35)]:
            reference_q = (np.array([x, y], float)+.5)/2.5-.5
            u, v = reference_q-[3., 4.]
            reference_sample = reference_q+[.004*u*v+.01*u, -.003*u*v+.02*v]
            expected_native_sample = (reference_sample+.5)*2.5-.5
            actual_native_sample = np.array([x, y])+self.at(result, x, y)
            np.testing.assert_allclose(actual_native_sample, expected_native_sample, rtol=0, atol=2e-7)

    def test_identity_and_zero_collar_stay_exact_without_changing_input(self):
        field = self.fixture(); before = field.copy()
        result = lift_inverse_field(field, [3, 4, 13, 15], (20, 24), (50, 60))
        output = result['field']
        np.testing.assert_array_equal(output[:2], 0.)
        np.testing.assert_array_equal(output[-2:], 0.)
        np.testing.assert_array_equal(output[:, :2], 0.)
        np.testing.assert_array_equal(output[:, -2:], 0.)
        np.testing.assert_array_equal(field, before)
        identity = lift_inverse_field(np.zeros_like(field), [3, 4, 13, 15], (20, 24), (50, 60))
        np.testing.assert_array_equal(identity['field'], 0.)
        self.assertEqual(output.dtype, np.dtype('<f4'))
        self.assertFalse(result['report']['geometry_approval'])
        self.assertFalse(result['report']['aesthetic_approval'])

    def test_scale_one_is_an_independent_byte_identical_copy(self):
        field = self.fixture()
        result = lift_inverse_field(field, [3, 4, 13, 15], (20, 24), (20, 24))
        self.assertEqual(result['crop'], [3, 4, 13, 15])
        np.testing.assert_array_equal(result['field'], field)
        self.assertFalse(np.shares_memory(result['field'], field))

    def test_crop_at_image_edge_uses_identity_exterior(self):
        field = self.fixture()
        result = lift_inverse_field(field, [0, 0, 13, 15], (20, 24), (50, 60))
        self.assertEqual(result['crop'][:2], [0, 0])
        np.testing.assert_array_equal(result['field'][0], 0.)
        np.testing.assert_array_equal(result['field'][:, 0], 0.)
        # The first output pixel samples q_ref=-0.3, outside the reference image.
        np.testing.assert_array_equal(self.at(result, 0, 0), [0., 0.])

    def test_rejects_unsupported_geometry_or_malformed_fields(self):
        field = self.fixture()
        for target in [(40, 47), (10, 12), (0, 0), (50., 60)]:
            with self.assertRaises(ValueError):
                lift_inverse_field(field, [3, 4, 13, 15], (20, 24), target)
        for crop in [[-1, 4, 13, 15], [3, 4, 12, 15], [10, 4, 13, 15], [3., 4, 13, 15]]:
            with self.assertRaises(ValueError):
                lift_inverse_field(field, crop, (20, 24), (50, 60))
        for candidate in [field.astype(float), np.full_like(field, np.nan), field[:-1]]:
            with self.assertRaises(ValueError):
                lift_inverse_field(candidate, [3, 4, 13, 15], (20, 24), (50, 60))
        field[1, 6, 0] = .01
        with self.assertRaisesRegex(ValueError, 'two.*zero'):
            lift_inverse_field(field, [3, 4, 13, 15], (20, 24), (50, 60))


class SharedFixedSupportTests(unittest.TestCase):
    def test_fractional_scale_locks_all_reference_neighbors_and_keeps_old_support(self):
        reference = np.zeros((12, 16), bool); reference[8, 11] = True
        native = np.zeros((30, 40), bool); native[13, 18] = True
        before_reference, before_native = reference.copy(), native.copy()
        result = shared_fixed_support(reference, native)
        expected = reference.copy()
        # (18.5/2.5-.5, 13.5/2.5-.5) = (6.9, 4.9).
        expected[4:6, 6:8] = True
        np.testing.assert_array_equal(result['mask'], expected)
        self.assertEqual(result['report']['added_reference_pixels'], 4)
        np.testing.assert_array_equal(reference, before_reference)
        np.testing.assert_array_equal(native, before_native)
        self.assertFalse(np.shares_memory(result['mask'], reference))

    def test_every_native_fixed_sample_has_zero_interpolated_reference_motion(self):
        reference = np.zeros((12, 16), bool); reference[7, 3] = True
        native = np.zeros((30, 40), bool)
        native[::3, ::4] = True; native[0, 0] = True; native[-1, -1] = True
        result = shared_fixed_support(reference, native)
        mask = result['mask']
        self.assertTrue(np.all(mask[reference]))
        for y, x in zip(*np.where(native)):
            qx, qy = (np.array([x, y], float)+.5)/2.5-.5
            # Independently enumerate the mathematical interpolation support.
            for sy in {int(np.floor(qy)), int(np.ceil(qy))}:
                for sx in {int(np.floor(qx)), int(np.ceil(qx))}:
                    if 0 <= sy < 12 and 0 <= sx < 16:
                        self.assertTrue(mask[sy, sx], (x, y, sx, sy))
        self.assertTrue(mask[0, 0]); self.assertTrue(mask[-1, -1])

    def test_equal_resolution_is_exact_union_and_empty_native_keeps_reference(self):
        reference = np.zeros((12, 16), bool); reference[3, 4] = True
        native = np.zeros_like(reference); native[7, 8] = True
        np.testing.assert_array_equal(shared_fixed_support(reference, native)['mask'], reference | native)
        native[:] = False
        np.testing.assert_array_equal(shared_fixed_support(reference, native)['mask'], reference)

    def test_invalid_masks_or_scale_are_rejected(self):
        reference = np.zeros((12, 16), bool)
        for native in [np.zeros((6, 8), bool), np.zeros((30, 41), bool),
                       np.zeros((30, 40), np.uint8), np.zeros((30, 40, 1), bool)]:
            with self.assertRaises(ValueError):
                shared_fixed_support(reference, native)
        with self.assertRaises(ValueError):
            shared_fixed_support(reference.astype(np.uint8), np.zeros((30, 40), bool))


if __name__ == '__main__':
    unittest.main()
