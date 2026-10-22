"""Known-distance and source-binding tests without portrait data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from source_boundary_support import boundary_support, prepare


class SourceBoundarySupportTests(unittest.TestCase):
    def test_euclidean_band_has_round_caps_and_diagonal_distances(self):
        mask = boundary_support((15, 15), [[[4., 5.], [8., 5.]]], radius=2.)
        for x, y in [(2, 5), (10, 5), (4, 3), (8, 7), (6, 7)]:
            self.assertTrue(mask[y, x], (x, y))
        for x, y in [(2, 4), (10, 6), (6, 8)]:
            self.assertFalse(mask[y, x], (x, y))
        diagonal = boundary_support((15, 15), [[[3., 3.], [9., 9.]]], radius=1.)
        self.assertTrue(diagonal[6, 5])  # Distance 1/sqrt(2).
        self.assertFalse(diagonal[7, 5])  # Distance sqrt(2).

    def test_independent_reviewed_segments_never_bridge_the_unknown_gap(self):
        segments = [np.array([[4., 3.], [4., 6.]]), np.array([[16., 3.], [16., 6.]])]
        before = [p.copy() for p in segments]
        mask = boundary_support((20, 24), segments, radius=2.)
        self.assertTrue(mask[5, 4]); self.assertTrue(mask[5, 16])
        self.assertFalse(mask[4, 10]); self.assertFalse(mask[8, 10])
        for a, b in zip(segments, before):
            np.testing.assert_array_equal(a, b)
        for invalid in [[], [[[1., 1.]]], [[[1., 1.], [float('nan'), 3.]]]]:
            with self.assertRaises(ValueError):
                boundary_support((20, 24), invalid)
        for radius in [-1., 0., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                boundary_support((20, 24), segments, radius=radius)

    def test_prepare_preserves_source_and_original_protection_and_binds_review(self):
        def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); reference = base/'reference'; reference.mkdir()
            source = base/'source.png'; Image.new('RGB', (24, 20), (50, 80, 110)).save(source)
            fixed = np.zeros((20, 24), np.uint8); fixed[18, 22] = 255
            mask_path = reference/'fixed-source.png'; Image.fromarray(fixed).save(mask_path)
            (reference/'report.json').write_text(json.dumps({'source': str(source), 'inputs_sha256': {str(source): sha(source)}}))
            review_path = base/'review.json'
            review = {'schema': 'source-visible-neck-border-review-v1',
                      'hashes': {str(source): sha(source)},
                      'segments': [{'points': [{'source': [4., 3.], 'actual_final_map_destination': [19., 17.]},
                                                {'source': [4., 6.], 'actual_final_map_destination': [19., 19.]}],
                                    'automatic_protection_authorized': False}]}
            review_path.write_text(json.dumps(review))
            original = {p: p.read_bytes() for p in [source, mask_path, review_path]}
            output = base/'candidate'
            prepare(reference, review_path, output)
            protocol = json.loads((output/'protocol.json').read_text())
            combined = np.asarray(Image.open(protocol['mask'])) > 0
            self.assertTrue(combined[18, 22]); self.assertTrue(combined[5, 4])
            self.assertFalse(combined[17, 19])  # Never use output-derived destinations.
            self.assertEqual(protocol['operation'], 'source_reviewed_neck_boundary_protection')
            self.assertEqual(protocol['reference_source_sha256'], sha(source))
            self.assertEqual(protocol['mask_sha256'], sha(Path(protocol['mask'])))
            self.assertEqual(protocol['inputs_sha256'][str(review_path)], sha(review_path))
            self.assertEqual(protocol['inputs_sha256'][str(mask_path)], sha(mask_path))
            for path, data in original.items(): self.assertEqual(path.read_bytes(), data)
            with self.assertRaises(FileExistsError): prepare(reference, review_path, output)
            review['hashes'][str(source)] = '0'*64; review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, 'source'):
                prepare(reference, review_path, base/'bad-candidate')
            self.assertFalse((base/'bad-candidate').exists())


if __name__ == '__main__':
    unittest.main()
