"""Focused provenance regressions; no production image pipeline or build."""
from pathlib import Path
import unittest

import numpy as np

from classify_face_frozen_evidence import Evidence, classify, independent_masks, load_case, raster_sample


def synthetic():
    shape = (72, 96)
    z = np.zeros(shape, dtype=float)
    head = z.copy(); head[8:27, 35:60] = 1
    y, x = np.mgrid[:shape[0], :shape[1]]
    neck = z.copy(); neck[30:, 40:56] = 1
    rgb = np.zeros((*shape, 3), dtype=np.uint8); rgb[:] = [210, 176, 150]
    frozen = np.zeros(shape, dtype=bool); frozen[32:35, 32:35] = True; frozen[58:62, 25:29] = True
    return Evidence(rgb, frozen, head > .5, head > .5, head, np.ones(shape), neck,
                    z.copy(), z.copy(), z.copy(), z.copy(), y.astype(float), 27, 20, 1)


class FrozenEvidenceTests(unittest.TestCase):
    def test_absent_head_is_unknown_not_other_person(self):
        e = synthetic(); e.neck[:] = 0
        r = classify(e)
        self.assertEqual(np.count_nonzero(r['classes'] == 2), 0)
        self.assertTrue(np.all(r['unknown'][e.original_frozen]))
        self.assertFalse(r['hard'].any())

    def test_explicit_protection_overrides_near_head_and_rgb(self):
        e = synthetic(); e.user_protection[32:35, 32:35] = 1
        r = classify(e)
        self.assertTrue(np.all(r['classes'][32:35, 32:35] == 1))
        self.assertTrue(np.all(r['hard'][32:35, 32:35]))

    def test_protection_outside_old_frozen_is_not_dropped(self):
        e = synthetic(); e.user_protection[10:13, 45:48] = 1
        r = classify(e)
        self.assertTrue(np.all(r['hard'][10:13, 45:48]))

    def test_partial_protection_remains_explicit(self):
        e = synthetic(); e.user_protection[32:35, 32:35] = .4
        r = classify(e)
        self.assertTrue(np.all(r['classes'][32:35, 32:35] == 7))
        self.assertTrue(np.all(r['partial_user_strength'][32:35, 32:35] == .4))

    def test_known_other_instance_requires_independent_mask(self):
        e = synthetic(); e.other_instance[32:35, 32:35] = 1
        r = classify(e)
        self.assertTrue(np.all(r['classes'][32:35, 32:35] == 2))
        self.assertTrue(np.all(r['hard'][32:35, 32:35]))

    def test_observed_clothing_preserved_even_near_head(self):
        e = synthetic(); e.clothing[32:35, 32:35] = 1
        r = classify(e)
        self.assertTrue(np.all(r['classes'][32:35, 32:35] == 3))
        self.assertTrue(np.all(r['hard'][32:35, 32:35]))

    def test_lower_context_needs_contact_not_only_position(self):
        e = synthetic(); e.rgb[:, :40] = [1, 240, 1]
        e.person[:, 35:40] = 0
        r = classify(e)
        self.assertTrue(np.all(r['unknown'][58:62, 25:29]))
        self.assertFalse(np.any(r['hard'][58:62, 25:29]))

    def test_long_hair_class_is_not_promoted_to_fixed_torso(self):
        e = synthetic(); e.hair[58:62, 25:29] = 1
        r = classify(e)
        self.assertTrue(np.all(r['unknown'][58:62, 25:29]))

    def test_selected_owner_support_overrides_lower_context_hypothesis(self):
        e = synthetic(); e.owner_support[58:62, 25:29] = True
        r = classify(e)
        self.assertTrue(np.all(r['unknown'][58:62, 25:29]))

    def test_nonfinite_scale_fails_instead_of_silently_classifying(self):
        e = synthetic(); e.face_height = float('nan')
        with self.assertRaises(ValueError):
            classify(e)

    def test_missing_selected_head_does_not_prove_ownership(self):
        e = synthetic(); e.selected_head[:] = False; e.neck[:] = 0
        r = classify(e)
        self.assertTrue(np.all(r['unknown'][e.original_frozen]))

    def test_no_explicit_evidence_without_provenance_or_distinct_identity(self):
        mask = {'shape': 'raster', 'width': 1, 'height': 1, 'data': [255]}
        for record in [{'mask': mask}, {'mask': mask, 'evidence': 'synthetic', 'instance_id': 'selected'}]:
            with self.assertRaises(ValueError):
                independent_masks({'selected_instance_id': 'selected', 'other_instance_masks': [record]},
                                  'other_instance_masks', (1, 1), (0, 0, 1, 1))

    def test_unsupported_protection_never_silently_ignored(self):
        with self.assertRaises(ValueError):
            raster_sample({'shape': 'polygon', 'points': []}, (10, 10), (0, 0, 10, 10))

    def test_actual_makeup_strap_kept_hair_tip_unknown_and_user_override(self):
        case = Path('test/results/face-root-cause-20261019/makeup-rigid-60')
        source = Path('experiments/face-anatomical-v22-20261015/inputs/portrait-makeup-source.png')
        if not case.exists() or not source.exists():
            self.skipTest('ignored local photo fixtures unavailable')
        e, crop = load_case(case, source)
        r = classify(e)
        # Coordinates are only the independently reviewed regression labels;
        # the classifier itself has no photo identity or coordinate constants.
        strap = np.zeros_like(e.original_frozen); strap[852:902, 275-crop[0]:294-crop[0]] = True
        strap &= e.original_frozen
        tip = np.zeros_like(e.original_frozen); tip[671:677, 749-crop[0]:754-crop[0]] = True
        tip &= e.original_frozen
        self.assertEqual(int(strap.sum()), 420)
        self.assertEqual(int(tip.sum()), 15)
        self.assertTrue(np.all(r['classes'][strap] == 4))
        self.assertTrue(np.all(r['hard'][strap]))
        self.assertTrue(np.all(r['unknown'][tip]))
        e.user_protection[tip] = 1
        protected = classify(e)
        self.assertTrue(np.all(protected['classes'][tip] == 1))
        self.assertTrue(np.all(protected['hard'][tip]))


if __name__ == '__main__':
    unittest.main()
