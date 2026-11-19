import unittest
import numpy as np
from source_joint_colors import fit_source_colors


class SourceJointColorsTest(unittest.TestCase):
    def test_true_priors_are_recovered_across_transparent_mixed_and_opaque_source(self):
        a=np.array([[0.,.01,.25,.5,.9,1.]]);f=np.broadcast_to([210.,95.,40.],(1,6,3));b=np.broadcast_to([20.,45.,180.],(1,6,3));image=a[...,None]*f+(1-a[...,None])*b;r=fit_source_colors(image,a,f,b)
        np.testing.assert_allclose(r['foreground'],f,atol=1e-12);np.testing.assert_allclose(r['lower'],b,atol=1e-12);np.testing.assert_allclose(r['source_reconstructed'],image,atol=1e-12)

    def test_visible_layer_is_fixed_when_other_layer_is_fully_hidden(self):
        image=np.array([[[30.,90.,200.],[180.,40.,30.]]]);a=np.array([[0.,1.]]);fp=np.full_like(image,230.);bp=np.full_like(image,10.);r=fit_source_colors(image,a,fp,bp)
        np.testing.assert_array_equal(r['lower'][0,0],image[0,0]);np.testing.assert_array_equal(r['foreground'][0,1],image[0,1]);np.testing.assert_array_equal(r['foreground'][0,0],fp[0,0]);np.testing.assert_array_equal(r['lower'][0,1],bp[0,1])

    def test_incompatible_priors_hit_exact_layer_bound_without_clipping_source_or_output(self):
        image=np.full((1,1,3),250.);a=np.full((1,1),.2);r=fit_source_colors(image,a,np.zeros_like(image),np.zeros_like(image))
        np.testing.assert_allclose(r['foreground'],230.,atol=1e-12);np.testing.assert_allclose(r['lower'],255.,atol=1e-12);np.testing.assert_array_equal(r['source_reconstructed'],image);self.assertTrue(r['bound_active'].all())
        # Feasible alternative F=255,L=248.75 has larger prior error.
        self.assertLess(float((r['foreground']**2+r['lower']**2).sum()),3*(255**2+248.75**2))

    def test_layer_bounds_source_equality_and_convex_optimality(self):
        rng=np.random.default_rng(4);image=rng.uniform(0,255,(4,6,3));a=rng.uniform(0,1,(4,6));fp=rng.uniform(0,255,image.shape);bp=rng.uniform(0,255,image.shape);r=fit_source_colors(image,a,fp,bp)
        for value in [r['foreground'],r['lower']]:self.assertGreaterEqual(value.min(),-1e-12);self.assertLessEqual(value.max(),255+1e-12)
        np.testing.assert_allclose(r['source_reconstructed'],image,atol=1e-12)
        # Every feasible endpoint/interior offset has at least the chosen
        # per-channel energy; uses the actual constrained objective as oracle.
        objective=(r['foreground']-fp)**2+(r['lower']-bp)**2
        for fraction in np.linspace(0,1,13):
            d=r['lower_bound']*(1-fraction)+r['upper_bound']*fraction;f=image+(1-a[...,None])*d;b=image-a[...,None]*d
            self.assertTrue(np.all(objective<=(f-fp)**2+(b-bp)**2+1e-8))


if __name__=='__main__':unittest.main()
