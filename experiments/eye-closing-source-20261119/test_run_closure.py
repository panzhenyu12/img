"""Input/field provenance regressions for the fresh-source experimental runner."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from run_closure import guards, read_stationary
from source_geometry import FACE_OVAL, load_observation, make_graph, sha


class FreshSourceContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source.png'
        Image.new('RGB', (80, 60), (110, 120, 130)).save(self.source)
        self.observation = self.root/'observations.json'
        self.face = [{'x': .4, 'y': .3} for _ in range(478)]

    def write_observation(self, faces, dimensions=None):
        self.observation.write_text(json.dumps({
            'dimensions': dimensions or {'width': 80, 'height': 60},
            'rows': [{'id': 'full', 'box': [0, 0, 80, 60],
                      'faceCount': len(faces), 'sourceLandmarks': faces}],
        }))

    def test_physical_coordinates_use_original_dimensions(self):
        self.write_observation([self.face])
        mesh, normalized, _ = load_observation(self.source, self.observation, (80, 60))
        np.testing.assert_array_equal(mesh[0], [32, 18])
        np.testing.assert_array_equal(normalized[0], [.4, .3])

    def test_multiface_input_is_not_silently_selected(self):
        self.write_observation([self.face, self.face])
        with self.assertRaisesRegex(ValueError, 'multiple observed faces'):
            load_observation(self.source, self.observation, (80, 60))
        _, _, report = load_observation(self.source, self.observation, (80, 60), 1)
        self.assertEqual(report['faceIndex'], 1)

    def test_resized_observation_is_rejected(self):
        self.write_observation([self.face], {'width': 40, 'height': 30})
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            load_observation(self.source, self.observation, (80, 60))

    def test_observation_cannot_be_rebound_to_a_different_original(self):
        self.write_observation([self.face])
        (self.root/'report.json').write_text(json.dumps({
            'sourceSHA256': '0'*64, 'observationsSHA256': sha(self.observation),
        }))
        with self.assertRaisesRegex(ValueError, 'mismatched sourceSHA256'):
            load_observation(self.source, self.observation, (80, 60))

    def test_experiment_cannot_silently_switch_to_eye_opening(self):
        with self.assertRaisesRegex(ValueError, 'fixed closing strengths'):
            make_graph(np.full((478, 2), .5), (80, 60), 15)

    def test_native_graph_converts_physical_to_pixel_index_exactly_once(self):
        points = np.full((478, 2), .5)
        points[FACE_OVAL[0]] = [.4, .3]
        original = points.copy()
        graph = make_graph(points, (80, 60), -15)
        first = graph['operations'][0]['params']['landmarks'][0]
        self.assertEqual(first['x']*80, 31.5)
        self.assertEqual(first['y']*60, 17.5)
        self.assertAlmostEqual(graph['operations'][0]['params']['region']['x'], .38)
        self.assertAlmostEqual(graph['operations'][0]['params']['region']['y'], .26)
        np.testing.assert_array_equal(points, original)

    def test_native_graph_does_not_clip_half_pixel_outside_canvas(self):
        points = np.full((478, 2), .5)
        points[FACE_OVAL[0]] = [0, .3]
        with self.assertRaisesRegex(ValueError, 'leaves pixel-center domain'):
            make_graph(points, (80, 60), -15)

    def test_field_requires_positive_evidence_of_geometry_safety(self):
        with self.assertRaisesRegex(ValueError, 'fixedDonorImports'):
            guards({}, -15)
        complete = {'fixedDonorImports': 0, 'foldPixels': 0,
                    'maxInverseFlowReverseErrorPx': .001,
                    'maxEyeEnvelopeDeviationFromOtherEditsPx': 0}
        guards(complete, -15)
        with self.assertRaisesRegex(ValueError, 'foldPixels'):
            guards({**complete, 'foldPixels': 1}, -15)
        with self.assertRaisesRegex(ValueError, 'RGBA identity'):
            guards({}, 0)
        guards({'identityRGBAExact': True}, 0)

    def test_native_empty_stationary_vector_means_no_frozen_semantic_masks(self):
        path = self.root/'stationary.u8'
        path.write_bytes(b'')
        raster, _ = read_stationary(path, 60, 80)
        self.assertEqual(raster.shape, (60, 80))
        self.assertFalse(raster.any())
        path.write_bytes(bytes([0])*100)
        with self.assertRaisesRegex(ValueError, 'incorrect byte count'):
            read_stationary(path, 60, 80)


if __name__ == '__main__':
    unittest.main()
