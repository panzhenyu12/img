import unittest
import numpy as np
from source_layers import proposal_trimap


class SourceTrimapTest(unittest.TestCase):
    def test_enclosed_miss_is_never_locked_transparent(self):
        mask=np.zeros((51,51),bool);mask[5:46,5:46]=True;mask[15:36,15:36]=False
        tri,holes=proposal_trimap(mask,2)
        self.assertTrue(holes[25,25])
        self.assertEqual(tri[25,25],128)
        self.assertEqual(tri[0,0],0)
        self.assertEqual(tri[9,9],255)

    def test_boundary_and_open_notch_keep_resolution_uncertainty(self):
        mask=np.zeros((51,51),bool);mask[10:41,10:41]=True;mask[10:15,23:27]=False
        tri,_=proposal_trimap(mask,5)
        self.assertEqual(tri[12,25],128)
        self.assertEqual(tri[8,25],128)
        self.assertEqual(tri[1,25],0)
        self.assertEqual(tri[25,25],255)


if __name__=='__main__':unittest.main()
