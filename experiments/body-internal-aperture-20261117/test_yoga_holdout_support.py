"""Known crop material must cover the entire unchanged ordinary field."""
import unittest

import numpy as np

from yoga_visible_rear_holdout import support_audit


class FrozenCropSupportTests(unittest.TestCase):
    def setUp(self):
        self.field = np.zeros((10, 10, 2), dtype=np.float32)
        self.view = {"x": 2, "y": 2, "width": 6, "height": 6}

    def test_identity_needs_no_unknown_material(self):
        self.assertTrue(support_audit(self.field, self.view)["allMovingDestinationsAndFourNeighboursKnown"])

    def test_fractional_read_inside_known_material_is_admissible(self):
        self.field[4, 4] = [2.25, .5]
        self.assertTrue(support_audit(self.field, self.view)["allMovingDestinationsAndFourNeighboursKnown"])

    def test_fractional_source_read_cannot_cross_crop_border(self):
        self.field[4, 4] = [3.25, .5]
        result = support_audit(self.field, self.view)
        self.assertEqual(result["unknownFourNeighbourPixels"], 1)
        self.assertEqual(result["unknownNonzeroWeightNeighbourPixels"], 1)
        self.assertFalse(result["allMovingDestinationsAndFourNeighboursKnown"])

    def test_known_source_read_does_not_authorize_unknown_destination_scene(self):
        self.field[1, 4] = [0, 3.25]
        result = support_audit(self.field, self.view)
        self.assertEqual(result["unknownDestinationPixels"], 1)
        self.assertEqual(result["unknownFourNeighbourPixels"], 0)
        self.assertFalse(result["allMovingDestinationsAndFourNeighboursKnown"])

    def test_conservative_four_neighbour_rejection_is_separate_from_weighted_support(self):
        self.field[4, 4] = [3, 0]
        result = support_audit(self.field, self.view)
        self.assertEqual(result["unknownFourNeighbourPixels"], 1)
        self.assertEqual(result["unknownNonzeroWeightNeighbourPixels"], 0)
        self.assertFalse(result["allMovingDestinationsAndFourNeighboursKnown"])

    def test_only_the_full_canvas_boundary_may_clamp_source_reads(self):
        self.field[9, 9] = [3.25, 2.5]
        canvas = {"x": 0, "y": 0, "width": 10, "height": 10}
        self.assertTrue(support_audit(self.field, canvas)["allMovingDestinationsAndFourNeighboursKnown"])


if __name__ == "__main__":
    unittest.main()
