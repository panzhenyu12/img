"""Material corrections receive discrete source labels, never oracle alpha/B."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'eye-height-layers-20261117'))
from layers import compose
from observed_ocular_v6 import estimate_observed_ocular as estimate_v6
from observed_ocular_v5 import estimate_observed_ocular as estimate_v5
from source_estimator import prepare_sampling_support
from source_material import apply_materials, load_materials, validate_material_transport
from test_observed_ocular import known_eye, oracle, bilinear


def prepare(fixture, estimator):
    f = fixture
    result = estimator(f['source'], f['u'], f['v'], f['upper'], f['lower'],
                       f['to_pixel'], f['width'], f['iris_center'], f['iris_radii'])
    return prepare_sampling_support(f['source'], result)


def render(f, prepared):
    if 'materialUnknown' in prepared:
        validate_material_transport(prepared['materialUnknown'], f['inverse'])
    return compose(f['source'], prepared['exposure'], prepared['ocular'],
                   prepared['support'], f['inverse'])[0]


def domain(f):
    return ((f['u'] >= 0) & (f['u'] <= f['width'])
            & (f['v'] > f['upper'](f['u'])) & (f['v'] < f['lower'](f['u'])))


def manual_lash_labels(shape):
    """Fixed coarse source brush and four visible-ocular rectangles.

    Literal image coordinates are drawn from the source view. This routine
    does not receive the fixture's true opacity, underlay, or generated roles.
    """
    yy, xx = np.indices(shape)
    start = np.array([31., 16.]); end = np.array([38., 20.5])
    points = np.stack([xx, yy], axis=-1)
    direction = end-start
    fraction = np.clip(np.sum((points-start)*direction, axis=-1)/np.dot(direction, direction), 0, 1)
    moving = (np.linalg.norm(points-(start+fraction[..., None]*direction), axis=-1) <= 1.4) & (yy >= 16)
    ocular = np.zeros(shape, bool)
    ocular[18:27, 27:31] = True
    ocular[23:28, 31:40] = True
    ocular[17:26, 40:45] = True
    ocular[16:17, 35:41] = True
    return {'ocularAnchors': ocular, 'movingLashSupport': moving,
            'unknown': np.zeros(shape, bool)}


class SourceMaterialImageTests(unittest.TestCase):
    def test_explicit_ocular_region_corrects_false_iris_track_and_keeps_texture(self):
        f = known_eye()
        automatic = prepare(f, estimate_v6)
        expected = oracle(f)
        original_error = float(abs(render(f, automatic)[f['feature']]-expected[f['feature']]).max())
        self.assertGreater(original_error, 100., 'fixture must expose the existing false-track failure')
        # A coarse rectangle drawn on the source iris covers texture AND its
        # nearby ocular samples. No oracle aperture, alpha or hidden RGB is fed.
        ocular = np.zeros(f['u'].shape, bool)
        ocular[16:19, 30:42] = True
        labels = {'ocularAnchors': ocular, 'movingLashSupport': np.zeros_like(ocular),
                  'unknown': np.zeros_like(ocular)}
        corrected = apply_materials(f['source'], automatic, labels, domain(f))
        output = render(f, corrected)
        self.assertLess(float(abs(output[f['feature']]-expected[f['feature']]).max()), 1.)
        np.testing.assert_array_equal(corrected['ocular'][ocular], f['source'][ocular])
        np.testing.assert_array_equal(corrected['exposure'][ocular], 1.)

    def test_manual_lash_support_repairs_automatic_miss_in_final_rgb(self):
        f = known_eye(attached_lash=True)
        labels = manual_lash_labels(f['u'].shape)
        automatic = prepare(f, estimate_v5)
        moving = labels['movingLashSupport']
        # Reproduce automatic missed foreground in the user-marked region.
        # The correction must actually estimate layers, not rename this mask.
        automatic['exposure'][moving] = 1
        automatic['ocular'][moving] = f['source'][moving]
        automatic['core'][moving] = True
        automatic['support'][moving] = True
        corrected = apply_materials(f['source'], automatic, labels, domain(f))
        output = render(f, corrected)
        expected = oracle(f)
        twice_field = bilinear(f['inverse'], f['inverse'])
        twice = oracle(f, twice_field)
        moved_lash = bilinear(f['lash_coverage'], f['inverse'])
        twice_lash = bilinear(f['lash_coverage'], twice_field)
        near_lash = binary_dilation((f['lash_coverage'] > 0) | (moved_lash > 0) | (twice_lash > 0))
        radius = (((f['u']-f['iris_center'][0])/f['iris_radii'][0])**2
                  + ((f['v']-f['iris_center'][1])/f['iris_radii'][1])**2)
        focus = (near_lash & (radius < 1) & (f['aperture'] == 1)
                 & (bilinear(f['aperture'], f['inverse']) == 1)
                 & (bilinear(f['aperture'], twice_field) == 1))
        rms = lambda image: float(np.sqrt(np.mean((image[focus]-expected[focus])**2)))
        reference = min(rms(f['source']), rms(twice))
        self.assertGreater(reference, 5.)
        self.assertGreater(rms(render(f, automatic)), .4*reference,
                           'the simulated missed foreground must fail the existing final-RGB criterion')
        self.assertLess(rms(output), .4*reference,
                        'manual material support must restore once-only motion using inferred source layers')

        def centroid(image):
            contrast = f['ocular']-f['lash_rgb']
            darkness = np.sum((f['ocular']-image)*contrast, axis=-1)/np.sum(contrast**2, axis=-1)
            column = (f['u'] == 25) & (f['v'] >= -5) & (f['v'] <= 0)
            weights = np.clip(darkness, 0, 1)*column
            self.assertGreater(float(weights.sum()), 1.)
            return float(np.sum(weights*f['v'])/weights.sum())

        self.assertLess(abs(centroid(output)-centroid(expected)), .25)

    def test_unknown_blocks_dependent_motion_but_zero_remains_exact(self):
        f = known_eye()
        prepared = prepare(f, estimate_v5)
        unknown = np.zeros(f['u'].shape, bool)
        unknown[17, 35] = True
        labels = {'ocularAnchors': np.zeros_like(unknown),
                  'movingLashSupport': np.zeros_like(unknown), 'unknown': unknown}
        corrected = apply_materials(f['source'], prepared, labels, domain(f))
        self.assertFalse(corrected['support'][17, 35])
        validate_material_transport(corrected['materialUnknown'], f['identity'])
        zero, _ = compose(f['source'], corrected['exposure'], corrected['ocular'],
                          corrected['support'], f['identity'])
        np.testing.assert_array_equal(zero, f['source'])
        with self.assertRaisesRegex(ValueError, 'unknown source material'):
            render(f, corrected)
        # A moving label alone is not enough to invent a hidden eye. Missing
        # observed donors must also remain structurally unknown.
        labels = {'ocularAnchors': np.zeros_like(unknown),
                  'movingLashSupport': unknown.copy(), 'unknown': np.zeros_like(unknown)}
        missing_donors = apply_materials(f['source'], prepared, labels, domain(f))
        self.assertTrue(missing_donors['materialUnknown'][17, 35])
        with self.assertRaisesRegex(ValueError, 'unknown source material'):
            render(f, missing_donors)

    def test_source_bound_categorical_document_rejects_wrong_source_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'materials.json'
            digest = hashlib.sha256(b'original encoded source').hexdigest()
            region = {'shape': 'rect', 'region': {'x': 30, 'y': 16, 'width': 12, 'height': 3}}
            document = {'version': '1.0',
                        'source': {'sha256': digest, 'width': 73, 'height': 45, 'coordinates': 'pixel-index'},
                        'eyes': {'leftEye': {'ocularAnchors': [region]}}}
            path.write_text(json.dumps(document))
            materials, _ = load_materials(path, digest, (73, 45))
            self.assertEqual(int(materials['leftEye']['ocularAnchors'].sum()), 36)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_materials(path, '0'*64, (73, 45))
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                load_materials(path, digest, (74, 45))
            document['eyes']['leftEye']['movingLashSupport'] = [region]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'conflict'):
                load_materials(path, digest, (73, 45))
            del document['eyes']['leftEye']['movingLashSupport']
            document['eyes']['rightEye'] = {'movingLashSupport': [region]}
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'across eye ownership'):
                load_materials(path, digest, (73, 45))
            del document['eyes']['rightEye']
            region['feather'] = .5
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'alpha'):
                load_materials(path, digest, (73, 45))

    def test_no_moving_branch_still_validates_source_and_domain(self):
        f = known_eye()
        prepared = prepare(f, estimate_v5)
        ocular = np.zeros(f['u'].shape, bool)
        ocular[0, 0] = True
        labels = {'ocularAnchors': ocular, 'movingLashSupport': np.zeros_like(ocular),
                  'unknown': np.zeros_like(ocular)}
        with self.assertRaisesRegex(ValueError, 'anchors leave source domain'):
            apply_materials(f['source'], prepared, labels, domain(f))
        with self.assertRaisesRegex(ValueError, 'domain must be boolean'):
            apply_materials(f['source'], prepared, labels, domain(f).astype(float))
        invalid = f['source'].copy(); invalid[0, 0, 0] = 256
        with self.assertRaisesRegex(ValueError, '0..255'):
            apply_materials(invalid, prepared, labels, domain(f))


if __name__ == '__main__':
    unittest.main()
