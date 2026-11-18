"""Source adapter contracts; no model, renderer or application build."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import prepare_face_source_replay as adapter


class SourceReplayTests(unittest.TestCase):
    def test_alpha_raster_survives_the_renderer_json_decoder_without_u8_quantization(self):
        source = np.array([[0., 1., .1234567], [.5, .9991, .00321]], dtype=np.float32)
        raster = json.loads(json.dumps(adapter.alpha_raster(source, (0, 0, 3, 2))))
        decoded = np.asarray(raster['data'], dtype=np.float32).reshape(source.shape) / np.float32(255)
        self.assertEqual(decoded[0, 1], 1., 'opaque source must remain opaque after PreparedMask decoding')
        np.testing.assert_allclose(decoded, source, rtol=0, atol=1e-7)
        self.assertNotEqual(raster['data'][2], round(raster['data'][2]), 'partial alpha must not be quantized to u8')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.graph = self.root / 'source.png', self.root / 'graph.json'
        self.cache_dir = self.root / 'pair/replay-cache'
        self.complete = self.root / 'pair/complete-optical-domain'
        self.output = self.root / 'prepared'
        self.rgba = np.zeros((5, 6, 4), np.uint8)
        self.rgba[..., :3] = [100, 140, 180]
        self.rgba[..., 3] = 255
        self.rgba[3, 4, :3] = [50, 90, 120]
        Image.fromarray(self.rgba).save(self.source)
        self.public = {'source': {'width': 6, 'height': 5},
                       'operations': [{'type': 'face.adjust', 'params': {'adjustments': {'faceWidth': 0}}}]}
        self.graph.write_text(json.dumps(self.public))
        self.alpha = np.array([[0, 1, .1234567, .5], [.01, .25, .75, .99], [1, .4, .6, 0]], '<f4')
        self.background = self.rgba[1:4, 1:5].copy()
        self.background[0, 1, :3] = [30, 40, 50]
        for directory in (self.cache_dir, self.complete):
            (directory / 'input').mkdir(parents=True)
            (directory / 'layers').mkdir()
        self.rgba[1:4, 1:5].tofile(self.complete / 'input/source.rgba')
        self.alpha.tofile(self.complete / 'layers/final-alpha.f32')
        self.background.tofile(self.complete / 'layers/source-background.rgba')
        self.rgba[1:4, 2:4].tofile(self.cache_dir / 'input/source.rgba')
        np.zeros((3, 2), np.uint8).tofile(self.cache_dir / 'input/frozen.u8')
        self.alpha[:, 1:3].tofile(self.cache_dir / 'layers/final-alpha.f32')
        self.background[:, 1:3].tofile(self.cache_dir / 'layers/source-background.rgba')
        self.cache_path = self.cache_dir / 'source-cache.json'
        self.cache = {'sourceSHA256': adapter.sha(self.source), 'graphSHA256': adapter.sha(self.graph),
                      'sourceObservationBox': [1, 1, 5, 4],
                      'replayView': {'origin': [2, 1], 'width': 2, 'height': 3}}
        for key, path in [('sourceInputSHA256', 'input/source.rgba'), ('sourceFrozenSHA256', 'input/frozen.u8'),
                          ('finalAlphaSHA256', 'layers/final-alpha.f32'), ('sourceBackgroundSHA256', 'layers/source-background.rgba')]:
            self.cache[key] = adapter.sha(self.cache_dir / path)
        self.save_cache()

    def save_cache(self):
        self.cache_path.write_text(json.dumps(self.cache))

    def prepare(self, **kwargs):
        return adapter.prepare(self.source, self.graph, self.cache_path, self.output, 'test-source', **kwargs)

    def bind_permissions(self, permissions):
        cached = json.loads(json.dumps(self.public))
        cached['operations'][0]['params'].update(permissions)
        cache_graph = self.root / 'cache-graph.json'
        cache_graph.write_text(json.dumps(cached))
        self.cache['graphSHA256'] = adapter.sha(cache_graph)
        self.save_cache()
        return cache_graph

    def test_complete_source_domain_embeds_and_decodes_at_the_original_coordinates(self):
        case, provenance = self.prepare()
        raster = json.loads(Path(case['physicalAlpha']).read_text())
        decoded = np.array(raster['data'], np.float32).reshape(5, 6) / np.float32(255)
        self.assertEqual(decoded[1, 2], 1.)
        np.testing.assert_allclose(decoded[1:4, 1:5], self.alpha, rtol=0, atol=1e-7)
        self.assertEqual(np.count_nonzero(decoded[0]), 0)
        descriptor = json.loads(Path(case['sourcePlatePrior']).read_text())
        actual_background = np.asarray(Image.open(descriptor['image']))
        np.testing.assert_array_equal(actual_background[1:4, 1:5], self.background)
        np.testing.assert_array_equal(actual_background[0], self.rgba[0])
        weights = np.fromfile(descriptor['weights'], '<f4').reshape(5, 6)
        self.assertEqual(float(weights.sum()), 12.)
        self.assertEqual(raster['sourceRegion'], {'x': 1 / 6, 'y': 1 / 5, 'width': 4 / 6, 'height': 3 / 5})
        self.assertLessEqual(provenance['physicalAlphaRendererDecodeAudit']['cacheReplayViewMaxAbsError'], 1e-7)
        self.assertFalse(provenance['renderOrModelRun'])

    def test_wrong_encoded_source_and_same_hash_wrong_rgba_are_rejected(self):
        wrong = self.rgba.copy()
        wrong[1, 2, 0] += 1
        Image.fromarray(wrong).save(self.source)
        with self.assertRaisesRegex(ValueError, 'source PNG SHA256'):
            self.prepare()
        self.cache['sourceSHA256'] = adapter.sha(self.source)
        self.save_cache()
        with self.assertRaisesRegex(ValueError, 'source RGBA'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_out_of_bounds_or_malformed_roi_is_rejected_before_output(self):
        for box in ([-1, 1, 5, 4], [1, 1, 7, 4], [1, 1, 1, 4], [1.5, 1, 5, 4]):
            with self.subTest(box=box):
                self.cache['sourceObservationBox'] = box
                self.save_cache()
                with self.assertRaisesRegex(ValueError, 'sourceObservationBox'):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_nonfinite_or_out_of_range_alpha_is_rejected(self):
        for invalid in (np.nan, np.inf, -0.01, 1.01):
            with self.subTest(alpha=invalid):
                alpha = self.alpha.copy()
                alpha[0, 0] = invalid
                alpha.tofile(self.complete / 'layers/final-alpha.f32')
                with self.assertRaisesRegex(ValueError, 'alpha must be finite'):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_background_cannot_change_an_observed_zero_alpha_pixel(self):
        background = self.background.copy()
        background[0, 0, 0] -= 10
        background.tofile(self.complete / 'layers/source-background.rgba')
        with self.assertRaisesRegex(ValueError, 'zero-alpha paired background'):
            self.prepare()

    def test_private_graph_is_not_silently_sanitized_and_hash_difference_is_explicit(self):
        self.public['operations'][0]['params']['__sourceMotionInterface'] = {'example': True}
        private_graph = self.root / 'cache-graph.json'
        private_graph.write_text(json.dumps(self.public))
        self.graph.write_text(private_graph.read_text())
        self.cache['graphSHA256'] = adapter.sha(private_graph)
        self.save_cache()
        with self.assertRaisesRegex(ValueError, 'must already be public'):
            self.prepare()
        del self.public['operations'][0]['params']['__sourceMotionInterface']
        self.graph.write_text(json.dumps(self.public))
        with self.assertRaisesRegex(ValueError, 'supply --cache-graph'):
            self.prepare()
        before = adapter.sha(self.graph)
        _, provenance = self.prepare(cache_graph=private_graph)
        self.assertEqual(adapter.sha(self.graph), before)
        self.assertEqual(provenance['graphBinding']['privateFieldsAbsentFromPublicGraph'],
                         ['/operations/0/params/__sourceMotionInterface'])

    def test_cache_source_permissions_are_forwarded_unchanged_in_separate_files(self):
        permissions = {
            '__sourceMotionInterface': {'shape': 'raster', 'width': 3, 'height': 2,
                                        'data': [0, 1, .125, 255, .007, 12.75]},
            '__sourceMovingAttachmentSupport': {'sourceRegion': {'x': .25, 'y': .125},
                                                'anchors': [[.1, .4], [.3, .8]],
                                                'weights': [0, .123456789, 1]},
        }
        cache_graph = self.bind_permissions(permissions)
        before = self.graph.read_bytes()
        case, provenance = self.prepare(cache_graph=cache_graph)
        saved_case = json.loads((self.output / 'replay-case.json').read_text())
        self.assertEqual(case, saved_case)
        for private, value in permissions.items():
            field = private[2:]
            exported = Path(saved_case[field])
            self.assertEqual(exported.parent, self.output)
            self.assertEqual(json.loads(exported.read_text()), value)
            self.assertEqual(provenance['outputSHA256'][exported.name], adapter.sha(exported))
        self.assertEqual(self.graph.read_bytes(), before)
        fields = ['/operations/0/params/' + key for key in permissions]
        self.assertEqual(provenance['graphBinding']['privateFieldsAbsentFromPublicGraph'], fields)
        self.assertEqual(provenance['sourcePermissions']['applied'], fields)
        self.assertEqual(provenance['sourcePermissions']['omitted'], [])
        self.assertFalse(provenance['sourcePermissions']['omissionRequested'])

    def test_source_permissions_are_omitted_only_by_explicit_ablation(self):
        permissions = {'__sourceMotionInterface': {'weight': .7},
                       '__sourceMovingAttachmentSupport': {'weight': .4}}
        cache_graph = self.bind_permissions(permissions)
        case, provenance = self.prepare(cache_graph=cache_graph, omit_source_permissions=True)
        for key in permissions:
            self.assertNotIn(key[2:], case)
        fields = ['/operations/0/params/' + key for key in permissions]
        self.assertEqual(provenance['graphBinding']['privateFieldsAbsentFromPublicGraph'], fields)
        self.assertEqual(provenance['sourcePermissions']['applied'], [])
        self.assertEqual(provenance['sourcePermissions']['omitted'], fields)
        self.assertTrue(provenance['sourcePermissions']['omissionRequested'])
        self.assertEqual(len(list(self.output.iterdir())), 6)

    def test_unsupported_private_fields_are_rejected_even_during_permission_ablation(self):
        for key in ('__unreviewedMotion', 'experimentalSeamRepair'):
            for omit in (False, True):
                with self.subTest(key=key, omit=omit):
                    cache_graph = self.bind_permissions({key: {'weight': .5}})
                    with self.assertRaisesRegex(ValueError, 'unsupported private fields'):
                        self.prepare(cache_graph=cache_graph, omit_source_permissions=omit)
                    self.assertFalse(self.output.exists())

    def test_permission_name_is_not_allowed_at_a_different_graph_location(self):
        cached = json.loads(json.dumps(self.public))
        cached['__sourceMotionInterface'] = {'weight': .5}
        cache_graph = self.root / 'cache-graph.json'
        cache_graph.write_text(json.dumps(cached))
        self.cache['graphSHA256'] = adapter.sha(cache_graph)
        self.save_cache()
        with self.assertRaisesRegex(ValueError, 'unsupported private fields'):
            self.prepare(cache_graph=cache_graph)
        self.assertFalse(self.output.exists())

    def test_cache_graph_cannot_authorize_other_semantic_or_target_changes(self):
        cache_graph = self.root / 'cache-graph.json'
        cache_graph.write_text(self.graph.read_text())
        self.public['operations'][0]['params']['adjustments']['faceWidth'] = 99
        self.graph.write_text(json.dumps(self.public))
        with self.assertRaisesRegex(ValueError, 'nonprivate fields'):
            self.prepare(cache_graph=cache_graph)

    def test_existing_output_is_preserved_even_if_new_inputs_are_valid(self):
        self.prepare()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            self.prepare()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_replay_only_cache_does_not_claim_unobserved_complete_head(self):
        self.complete.rename(self.root / 'unused-complete-domain')
        case, provenance = self.prepare()
        self.assertEqual(provenance['sourceDomainKind'], 'cache-replay-view-only')
        self.assertEqual(provenance['sourceObservationBox'], [2, 1, 4, 4])
        raster = json.loads(Path(case['physicalAlpha']).read_text())
        decoded = np.array(raster['data'], np.float32).reshape(5, 6) / np.float32(255)
        self.assertEqual(decoded[1, 1], 0.)
        np.testing.assert_allclose(decoded[1:4, 2:4], self.alpha[:, 1:3], rtol=0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
