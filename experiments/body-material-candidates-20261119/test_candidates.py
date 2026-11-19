import unittest
import numpy as np
from candidates import spatial_queries


class SpatialQueriesTest(unittest.TestCase):
    def test_empty_support_has_no_invented_prompt(self):
        posterior = np.zeros((18, 8, 4), dtype=np.uint8)
        posterior[0] = 255
        self.assertEqual(spatial_queries(posterior, (40, 80)), ([], None))

    def test_grid_covers_support_once_with_exact_limit_and_coordinate_mapping(self):
        posterior = np.zeros((18, 8, 4), dtype=np.uint8)
        posterior[3] = 255
        queries, bbox = spatial_queries(posterior, (40, 80))
        self.assertEqual(bbox, [0, 0, 4, 8])
        self.assertEqual(len(queries), 32)
        self.assertEqual(len({tuple(q["posteriorXY"]) for q in queries}), 32)
        self.assertEqual(queries[0]["xy"], [5., 5.])
        self.assertEqual(queries[-1]["xy"], [35., 75.])

    def test_empty_cells_are_skipped_without_duplicates(self):
        posterior = np.zeros((18, 16, 16), dtype=np.uint8)
        posterior[0] = 255
        for y, x in [(0, 0), (15, 15)]:
            posterior[0, y, x] = 0
            posterior[3, y, x] = 255
        queries, _ = spatial_queries(posterior, (160, 160))
        self.assertEqual([q["id"] for q in queries], ["r0c0", "r7c3"])

    def test_uncertain_nonbackground_is_not_a_prompt(self):
        posterior = np.full((18, 8, 4), 5, dtype=np.uint8)
        posterior[3] = 100
        self.assertEqual(spatial_queries(posterior, (40, 80)), ([], None))

    def test_center_tie_is_stable_row_major(self):
        posterior = np.zeros((18, 16, 8), dtype=np.uint8)
        posterior[3] = 255
        queries, _ = spatial_queries(posterior, (80, 160))
        self.assertEqual(queries[0]["posteriorXY"], [0, 0])


if __name__ == "__main__":
    unittest.main()
