import unittest
import numpy as np
from project import project


class KnownColors(unittest.TestCase):
    def test_known_foreground_and_background_are_observations(self):
        c=np.array([[[225,30,77],[17,99,230]]],dtype='float32')
        a=np.array([[1,0]],dtype='float32')
        f,b=project(c,a,np.zeros_like(c),np.zeros_like(c))
        np.testing.assert_array_equal(f[0,0],c[0,0])
        np.testing.assert_array_equal(b[0,1],c[0,1])

    def test_ambiguous_mixture_is_nearest_projection_not_material_truth(self):
        # .5F+.5B=110 is F+B=220. The closest point to (170,20)
        # on this line is (185,35), even if the real scene was (200,20).
        c=np.full((1,1,3),110);a=np.array([[.5]])
        f,b=project(c,a,np.full_like(c,170),np.full_like(c,20))
        np.testing.assert_array_equal(f,np.full_like(c,185))
        np.testing.assert_array_equal(b,np.full_like(c,35))

    def test_bad_matte_cannot_make_superopaque_foreground(self):
        c=np.array([[[220,0,255]]]);a=np.array([[.1]])
        f,b=project(c,a,np.full_like(c,180),np.full_like(c,20))
        self.assertTrue(np.all((f>=0)&(f<=255)&(b>=0)&(b<=255)))
        np.testing.assert_allclose(a[...,None]*f+(1-a[...,None])*b,c,atol=1e-5)
        self.assertGreater(float(b[0,0,0]),200) # bad alpha forces old color into B


if __name__=='__main__':unittest.main()
