"""Independent image-output counterexamples for ocular/lash source ownership.

Known source layers generate these images. They are test oracles, not a claim
that arbitrary photographic alpha can be identified from a single RGB image.
The source eye texture varies along the closing direction, so incorrect
exposure cannot disappear through a constant-underlay algebraic cancellation.
"""
from pathlib import Path
import importlib
import os
import sys
import unittest

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eye-height-layers-20261117"))
from layers import compose
from source_estimator import estimate, prepare_sampling_support

# Keep the original V2 default explicit. Changing the selected candidate must
# not change these independent fixtures, output measurements or thresholds.
ESTIMATOR_SPEC = os.environ.get('EYE_SOURCE_ESTIMATOR', 'observed_ocular:estimate_observed_ocular')
_module_name, _function_name = ESTIMATOR_SPEC.split(':')
estimate_observed_ocular = getattr(importlib.import_module(_module_name), _function_name)


def bilinear(array, coordinates):
    """Independent four-corner oracle for the explicitly supplied field."""
    h, w = array.shape[:2]
    x, y = coordinates[..., 0], coordinates[..., 1]
    if np.any((x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)):
        raise ValueError("test field leaves the canvas")
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = x - x0, y - y0
    if array.ndim == 3:
        tx, ty = tx[..., None], ty[..., None]
    return ((1 - tx) * (1 - ty) * array[y0, x0] + tx * (1 - ty) * array[y0, x1]
            + (1 - tx) * ty * array[y1, x0] + tx * ty * array[y1, x1])


def known_eye(*, attached_lash=False):
    """Area-sampled source aperture and an independently drawn lash capsule."""
    yy, xx = np.mgrid[:45, :73]
    u, v, width = xx.astype(float) - 10, yy.astype(float) - 22, 52.
    upper = lambda q: -7.25 * np.sin(np.pi * np.clip(q, 0, width) / width)
    lower = lambda q: 6.75 * np.sin(np.pi * np.clip(q, 0, width) / width)
    to_pixel = lambda q, r: np.stack([q + 10, r + 22], axis=-1)
    iris_center, iris_radii = np.array([26., 0.]), np.array([11., 9.5])
    iris = ((u - iris_center[0]) / iris_radii[0]) ** 2 + (v / iris_radii[1]) ** 2 < 1
    ocular = np.broadcast_to([205., 218., 223.], u.shape + (3,)).copy() + .35 * v[..., None]
    textured = (np.array([65., 85., 105.]) + v[..., None] * [1.3, .7, .8]
                + np.sin(.7 * v + .15 * u)[..., None] * [10., 6., 8.])
    ocular[iris] = textured[iris]
    feature = np.zeros(u.shape, bool)
    if not attached_lash:
        # These are ocular paint/highlight samples, not eyelid coverage. Their
        # positions are fixed by this generator, independently of admission code.
        for column, color in [(25, [242., 246., 249.]), (26, [242., 246., 249.]),
                              (28, [15., 25., 37.]), (29, [15., 25., 37.])]:
            at = (u == column) & (v == -5)
            ocular[at] = color
            feature |= at
    skin = np.broadcast_to([175., 124., 102.], ocular.shape).copy()
    skin += u[..., None] * [.03, .01, .02]
    lash_rgb = np.array([9., 10., 13.])
    aperture = np.zeros(u.shape)
    lash_coverage = np.zeros(u.shape)
    a = np.array([19., float(upper(19.)) - .6])
    b = np.array([27.8, -1.7])
    segment = b - a
    # Exact known material roles are integrated over 64 subpixel rays. No
    # estimator mask, geometric core, color triangle or ridge detector is used.
    for dy in (np.arange(8) + .5) / 8 - .5:
        for dx in (np.arange(8) + .5) / 8 - .5:
            qu, qv = u + dx, v + dy
            open_ray = (qu > 0) & (qu < width) & (qv > upper(qu)) & (qv < lower(qu))
            point = np.stack([qu, qv], axis=-1)
            t = np.clip(np.sum((point - a) * segment, axis=-1) / np.dot(segment, segment), 0, 1)
            distance = np.linalg.norm(point - (a + t[..., None] * segment), axis=-1)
            on_lash = (distance < .65) & attached_lash
            aperture += open_ray / 64
            lash_coverage += (open_ray & on_lash) / 64
    exposure = aperture - lash_coverage
    residual = (1 - aperture[..., None]) * skin + lash_coverage[..., None] * lash_rgb
    source = residual + exposure[..., None] * ocular
    identity = np.stack([xx, yy], axis=-1).astype(float)
    window = np.sin(np.pi * np.clip(u, 0, width) / width) ** 2 * ((u > 0) & (u < width))
    taper = np.clip((18 - abs(v)) / 6, 0, 1)
    inverse = identity.copy()
    inverse[..., 1] += .18 * window * taper * v
    return dict(source=source, ocular=ocular, exposure=exposure, residual=residual,
                aperture=aperture, lash_coverage=lash_coverage, lash_rgb=lash_rgb,
                feature=feature, u=u, v=v, upper=upper, lower=lower, to_pixel=to_pixel,
                width=width, iris_center=iris_center, iris_radii=iris_radii,
                inverse=inverse, identity=identity)


def oracle(fixture, inverse=None):
    field = fixture["inverse"] if inverse is None else inverse
    return (bilinear(fixture["residual"], field)
            + bilinear(fixture["exposure"], field)[..., None] * fixture["ocular"])


def estimated_output(fixture, *, legacy=False):
    common = (fixture["source"], fixture["u"], fixture["v"], fixture["upper"],
              fixture["lower"], fixture["to_pixel"], fixture["width"])
    prepared = (estimate(*common) if legacy else estimate_observed_ocular(
        *common, fixture["iris_center"], fixture["iris_radii"]))
    prepared = prepare_sampling_support(fixture["source"], prepared)
    output, _ = compose(fixture["source"], prepared["exposure"], prepared["ocular"],
                        prepared["support"], fixture["inverse"])
    return output, prepared


class ObservedOcularImageTests(unittest.TestCase):
    def test_visible_inner_iris_bright_and_dark_texture_remains_stationary(self):
        f = known_eye()
        expected = oracle(f)
        visible = f["feature"] & (f["exposure"] == 1) & (bilinear(f["exposure"], f["inverse"]) == 1)
        self.assertEqual(int(visible.sum()), 4, "all independently drawn texture samples must remain visible")
        depth = np.minimum(f["v"] - f["upper"](f["u"]), f["lower"](f["u"]) - f["v"])
        self.assertGreater(float(depth[visible].min()), 1., "fixture texture must not be an actual lid sample")
        np.testing.assert_allclose(expected[visible], f["source"][visible], rtol=0, atol=1e-10)
        legacy, _ = estimated_output(f, legacy=True)
        legacy_error = float(abs(legacy[visible] - expected[visible]).max())
        self.assertGreater(legacy_error, 5., "counterexample must expose the old same-column estimator in final RGB")
        candidate, _ = estimated_output(f)
        candidate_error = float(abs(candidate[visible] - expected[visible]).max())
        self.assertLess(candidate_error, 1.,
                        f"visible iris texture became moving foreground: RGB error {candidate_error:.6f}; old {legacy_error:.6f}")

    def test_lid_connected_lash_inside_iris_moves_once_in_final_rgb(self):
        f = known_eye(attached_lash=True)
        expected = oracle(f)
        twice_field = bilinear(f["inverse"], f["inverse"])
        twice = oracle(f, twice_field)
        moved_lash = bilinear(f["lash_coverage"], f["inverse"])
        twice_lash = bilinear(f["lash_coverage"], twice_field)
        near_lash = binary_dilation((f["lash_coverage"] > 0) | (moved_lash > 0) | (twice_lash > 0))
        radius = (((f["u"] - f["iris_center"][0]) / f["iris_radii"][0]) ** 2
                  + ((f["v"] - f["iris_center"][1]) / f["iris_radii"][1]) ** 2)
        # Limit the output comparison to actually visible iris, away from
        # skin/aperture mixtures; this does not consult an estimated core mask.
        focus = (near_lash & (radius < 1) & (f["aperture"] == 1)
                 & (bilinear(f["aperture"], f["inverse"]) == 1)
                 & (bilinear(f["aperture"], twice_field) == 1))
        self.assertGreater(int(focus.sum()), 8)
        candidate, _ = estimated_output(f)

        def rms(image):
            return float(np.sqrt(np.mean((image[focus] - expected[focus]) ** 2)))

        # Frozen iris and a twice-moved lash are independent wrong outcomes.
        # A slightly altered wrong image is insufficient: the visible error
        # must be well below both wrong controls, without demanding exact E.
        frozen_error, twice_error, error = rms(f["source"]), rms(twice), rms(candidate)
        reference = min(frozen_error, twice_error)
        self.assertGreater(reference, 5., "fixture must discriminate lash motion through final RGB")
        self.assertLess(error, .4 * reference,
                        f"lash RGB disagrees with once-only oracle: candidate {error:.6f}, frozen {frozen_error:.6f}, twice {twice_error:.6f}")

        def visible_dark_centroid(image):
            contrast = f["ocular"] - f["lash_rgb"]
            darkness = np.sum((f["ocular"] - image) * contrast, axis=-1) / np.sum(contrast ** 2, axis=-1)
            # A source-fixed interior column captures the entire local lash
            # cross-section. An all-ROI centroid would crop different shaft
            # portions at the eyelid and obscure real motion through that ROI.
            column = (f["u"] == 25) & (f["v"] >= -5) & (f["v"] <= 0)
            weights = np.clip(darkness, 0, 1) * column
            mass = float(weights.sum())
            self.assertGreater(mass, 1., "test must contain visible lash darkness")
            return float(np.sum(weights * f["v"]) / mass)

        expected_y = visible_dark_centroid(expected)
        frozen_y = visible_dark_centroid(f["source"])
        twice_y = visible_dark_centroid(twice)
        candidate_y = visible_dark_centroid(candidate)
        self.assertGreater(min(abs(expected_y - frozen_y), abs(expected_y - twice_y)), .35,
                           "the visible lash must distinguish zero, one and two motions")
        self.assertLess(abs(candidate_y - expected_y), .25,
                        f"lash darkness is at the wrong position: source {frozen_y:.6f}, once {expected_y:.6f}, twice {twice_y:.6f}, candidate {candidate_y:.6f}")


if __name__ == "__main__":
    print(f'Automatic estimator under test: {ESTIMATOR_SPEC}', flush=True)
    unittest.main()
