"""A geometric anchor cannot stand in for missing visible ear material."""
import unittest
import numpy as np
from attachment import observed_lobe_contact


class ObservedContact(unittest.TestCase):
    def test_truncated_ear_cannot_claim_attachment_to_distant_metal(self):
        ear=np.zeros((60,40));ear[5:30,10:16]=1
        skin=np.zeros_like(ear);skin[5:42,10:25]=1
        metal=np.zeros_like(ear);metal[43:55,12:15]=1
        with self.assertRaisesRegex(ValueError,'material'):
            observed_lobe_contact(ear,skin,metal,[13,43],max_distance=4)

    def test_real_lobe_contact_excludes_metal_and_uses_visible_skin(self):
        ear=np.zeros((60,40));ear[5:42,10:16]=1
        skin=ear.copy();metal=np.zeros_like(ear);metal[43:55,12:15]=1
        point=observed_lobe_contact(ear,skin,metal,[13,43],max_distance=4)
        np.testing.assert_array_equal(point,[13,41])
        self.assertEqual(ear[int(point[1]),int(point[0])],1)
        self.assertEqual(metal[int(point[1]),int(point[0])],0)


if __name__=='__main__':unittest.main()
