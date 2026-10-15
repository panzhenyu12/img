"""Behavioral known-layer tests. These do not promise recovery from unknown B."""
import unittest
import numpy as np
from solver import solve


class FeasibleSkin(unittest.TestCase):
    def test_low_alpha_highlight_does_not_leave_old_skin_line(self):
        # Exact known skin/scene; after skin moves away this pixel must be B.
        a=np.array([[.38]]);f=np.array([[[1,.56,.37]]]);b=np.array([[[.035,.045,.055]]]);c=a[...,None]*f+(1-a[...,None])*b
        r=solve(c,b,a-.02,f,np.ones_like(a,dtype=bool))
        target=c-r['alpha'][...,None]*r['foreground']+r['alpha'][...,None]*b
        self.assertLess(np.max(abs(target-b))*255,1e-8)
        self.assertLess(np.max(abs(r['alpha']-a)),1e-12)
    def test_feasible_nonsaturated_prior_is_unchanged(self):
        a=np.array([[.5]]);f=np.array([[[.5,.3,.2]]]);b=np.array([[[.04,.06,.08]]]);c=a[...,None]*f+(1-a[...,None])*b;prior=a-.02
        r=solve(c,b,prior,f,np.ones_like(a,dtype=bool))
        self.assertTrue(np.array_equal(r['alpha'],prior));self.assertFalse(r['changed'].any())
        reconstructed=r['alpha'][...,None]*r['foreground']+(1-r['alpha'][...,None])*b
        self.assertLess(np.max(abs(c-reconstructed)),1e-12)
    def test_black_white_scene_channels_and_zero_opacity(self):
        b=np.array([[[0,1,.4],[0,1,.4]]]);c=np.array([[[.5,.5,.5],[0,1,.4]]]);a=np.array([[.1,0.]]);f=np.zeros_like(b)
        r=solve(c,b,a,f,np.ones_like(a,dtype=bool));self.assertTrue(np.isfinite(r['foreground']).all())
        self.assertGreaterEqual(r['alpha'][0,0],.5);self.assertEqual(r['alpha'][0,1],0)
    def test_unconfirmed_hair_metal_unknown_background_stay_exact(self):
        c=np.array([[[1,.9,.8],[.8,.5,.3],[.4,.3,.2]]]);b=np.zeros_like(c);a=np.array([[.02,.1,.01]]);f=c.copy()
        r=solve(c,b,a,f,np.zeros_like(a,dtype=bool))
        self.assertTrue(np.array_equal(r['alpha'],a));self.assertTrue(np.array_equal(r['foreground'],f))
    def test_projection_keeps_all_confirmed_rgb_in_gamut(self):
        rng=np.random.default_rng(73);a=rng.uniform(0,1,(32,32));f=rng.uniform(0,1,(32,32,3));b=rng.uniform(0,1,(32,32,3));c=a[...,None]*f+(1-a[...,None])*b
        r=solve(c,b,a*.3,f,np.ones_like(a,dtype=bool));reconstructed=r['alpha'][...,None]*r['foreground']+(1-r['alpha'][...,None])*b
        self.assertLess(np.max(abs(c-reconstructed)),1e-12)
        self.assertTrue((r['foreground']>=0).all());self.assertTrue((r['foreground']<=1).all())

if __name__=='__main__':unittest.main(verbosity=2)
