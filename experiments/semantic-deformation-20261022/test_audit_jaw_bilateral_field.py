import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from audit_jaw_bilateral_field import bilateral_statistics, pair_source_height_segments, audit_radius_weighted_displacement


class BilateralFieldMetricTests(unittest.TestCase):
    def test_radius_weighted_supplement_uses_height_weights_and_rejects_invalid_radii(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report_path = root/'report.json'
            report_path.write_text(json.dumps({'paired_height_samples': [{'max_height_pitch_px': .125}]}))
            data = {'source_radius': np.array([[-10., 20.], [-12., 24.]]),
                    'height_weight': np.array([1., 3.]), 'source_height': np.array([1., 2.])}
            for name in ['baseline', 'candidate']:
                for mode in ['requested', 'actual_f32_forward']:
                    data[name+'_'+mode+'_q'] = np.array([[.1, .05], [.1, .05]])
            np.savez(root/'height-0.125.npz', **data)
            result = audit_radius_weighted_displacement(report_path, root/'valid')
            actual = result['samples'][0]['cases']['candidate']['actual_f32_forward']
            self.assertAlmostEqual(actual['height_weighted_rms_px'], np.sqrt(((2./3.)**2+3.*.8**2)/4.))
            self.assertAlmostEqual(actual['sampled_max_abs_px'], .8)
            data['source_radius'][0, 0] = 0.
            np.savez(root/'height-0.125.npz', **data)
            with self.assertRaisesRegex(ValueError, 'signed radii'):
                audit_radius_weighted_displacement(report_path, root/'invalid')

    def test_known_unequal_radii_measure_relative_contraction_not_pixels(self):
        radius = np.array([[-10., 20.], [-12., 24.]])
        # Twofold right radius needs twice the pixel displacement for the
        # same 10% contraction. Same pixel edits would wrongly look balanced.
        result, q = bilateral_statistics(radius, -radius*.1, [1., 3.], [1., 2.])
        np.testing.assert_allclose(q, .1, atol=1e-15, rtol=0.)
        self.assertAlmostEqual(result['relative_contraction_difference_rms'], 0.)
        result, _ = bilateral_statistics(radius, [[1., -1.], [1.2, -1.2]], [1., 3.], [1., 2.])
        self.assertAlmostEqual(result['relative_contraction_difference_rms'], .05)
        self.assertAlmostEqual(result['relative_contraction_difference_max_abs'], .05)

    def test_multiple_and_reversed_source_branches_cannot_be_paired_or_extended(self):
        left = np.array([[-10., 0.], [-9., 4.], [-8., 3.], [-7., 6.], [-6., 5.]])
        right = np.array([[20., 1.], [18., 6.], [17., 7.]])
        paired, excluded = pair_source_height_segments(left, right)
        self.assertEqual([r['source_down_open_interval_px'] for r in paired], [[1., 3.], [4., 5.]])
        self.assertTrue(any(r['source_down_open_interval_px'] == [3., 4.]
                            and 'left_multiple_intersections' in r['reasons'] for r in excluded))
        self.assertTrue(any(r['source_down_open_interval_px'] == [6., 7.]
                            and 'left_missing_intersection' in r['reasons'] for r in excluded))
        paired, excluded = pair_source_height_segments([[-10., 3.], [-8., 1.]], [[10., 1.], [12., 3.]])
        self.assertEqual(paired, [])
        self.assertEqual(excluded[0]['reasons'], ['left_reverse_segment'])


if __name__ == '__main__': unittest.main()
