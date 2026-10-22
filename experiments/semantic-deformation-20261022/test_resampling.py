"""Independent reconstruction oracles; no portrait data or full-system run."""
import unittest

import numpy as np

from resampling import render_frozen_field, sample_rgba


class ResamplingTests(unittest.TestCase):
    def test_subpixel_sinusoid_has_less_error_than_bilinear(self):
        yy, xx = np.mgrid[:80, :96]
        value = 128.+60.*np.sin(2*np.pi*(.14*xx+.09*yy))
        source = np.dstack([value, value, value, np.full_like(value, 255.)])
        py, px = np.mgrid[5:74, 5:90]
        points = np.column_stack([px.ravel()+.37, py.ravel()-.23])
        truth = 128.+60.*np.sin(2*np.pi*(.14*points[:, 0]+.09*points[:, 1]))
        baseline, _ = sample_rgba(source, points, 'bilinear')
        candidate, _ = sample_rgba(source, points, 'bounded_lanczos3')
        mse_base = np.mean((baseline[:, 0]-truth)**2)
        mse_candidate = np.mean((candidate[:, 0]-truth)**2)
        self.assertLess(mse_candidate, mse_base*.45)

    def test_constant_and_integer_coordinates_remain_exact(self):
        source = np.empty((15, 17, 4), dtype=np.uint8); source[:] = [41, 87, 143, 255]
        points = np.array([[4.1, 4.7], [7.99, 8.02], [11.3, 9.1]])
        samples, _ = sample_rgba(source, points, 'bounded_lanczos3')
        np.testing.assert_allclose(samples, np.tile(source[0, 0], (len(points), 1)), rtol=0, atol=1e-12)
        rng = np.random.default_rng(19); source = rng.integers(0, 256, source.shape, dtype=np.uint8)
        points = np.array([[0, 0], [16, 14], [7, 9], [4, 5]])
        samples, _ = sample_rgba(source, points, 'bounded_lanczos3')
        np.testing.assert_array_equal(samples, source[points[:, 1], points[:, 0]])

    def test_image_boundary_and_transparency_keep_bilinear_rgba(self):
        rng = np.random.default_rng(31); source = rng.integers(0, 256, (15, 17, 4), dtype=np.uint8)
        source[..., 3] = 255; source[7, 8] = [0, 255, 0, 0]
        points = np.array([[-1., 3.2], [.4, .7], [16.4, 9.2], [7.3, 7.8], [8.2, 6.9]])
        candidate, report = sample_rgba(source, points, 'bounded_lanczos3')
        baseline, _ = sample_rgba(source, points, 'bilinear')
        np.testing.assert_array_equal(candidate, baseline)
        self.assertEqual(report['lanczos_sample_count'], 0)

    def test_step_edge_cannot_overshoot_local_two_by_two_sample_range(self):
        source = np.zeros((20, 32, 4), dtype=np.uint8); source[..., 3] = 255
        source[:, 16:, :3] = 255
        points = np.column_stack([np.arange(10.03, 21., .031), np.full(354, 9.4)])
        samples, report = sample_rgba(source, points, 'bounded_lanczos3')
        left = np.floor(points[:, 0]).astype(int)
        lo = source[9, left, 0]; hi = source[9, left+1, 0]
        self.assertTrue(np.all(samples[:, 0] >= lo))
        self.assertTrue(np.all(samples[:, 0] <= hi))
        self.assertGreater(report['bounded_sample_count'], 0)

    def test_zero_displacement_fixed_pixels_and_outside_crop_are_byte_exact(self):
        rng = np.random.default_rng(45); source = rng.integers(0, 256, (20, 20, 4), dtype=np.uint8)
        field = np.zeros((10, 10, 2), dtype=np.float32); field[2:8, 2:8] = [.31, -.22]
        fixed = np.zeros((10, 10), bool); fixed[4, 4] = True
        original = field.copy()
        rendered, _ = render_frozen_field(source, field, [5, 5, 10, 10], 'bounded_lanczos3', fixed=fixed)
        mask = np.ones((20, 20), bool); mask[7:13, 7:13] = False; mask[9, 9] = True
        np.testing.assert_array_equal(rendered[mask], source[mask])
        np.testing.assert_array_equal(rendered[..., 3], render_frozen_field(source, field, [5, 5, 10, 10], 'bilinear', fixed=fixed)[0][..., 3])
        np.testing.assert_array_equal(field, original)


if __name__ == '__main__':
    unittest.main()
