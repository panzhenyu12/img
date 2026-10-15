"""Physical support must be independent of the uncertain material ownership."""
import unittest
import numpy as np
from attachment import partition_joint_alpha


class JointSupportContract(unittest.TestCase):
    def test_material_partition_cannot_erase_opaque_joint_skin(self):
        physical = np.ones(11)
        ownership = np.linspace(0, 1, 11)
        ear, face = partition_joint_alpha(physical, ownership)
        np.testing.assert_allclose(ear + (1-ear)*face, physical, atol=1e-12)
        self.assertTrue(np.all(face[ownership < 1] == 1))

    def test_partly_covered_boundary_keeps_measured_total_opacity(self):
        physical = np.array([0, .05, .3, .8, .999, 1.])
        ownership = np.array([.7, .9, .1, .6, .999, 1.])
        ear, face = partition_joint_alpha(physical, ownership)
        np.testing.assert_allclose(ear + (1-ear)*face, physical, atol=1e-12)
        self.assertTrue(np.all((face >= 0) & (face <= 1)))
        self.assertEqual(ear[0], 0)
        self.assertEqual(face[0], 0)


if __name__ == '__main__':
    unittest.main()
