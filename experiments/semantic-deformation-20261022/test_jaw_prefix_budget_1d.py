import unittest
import numpy as np
from jaw_prefix_budget_1d import prefix_energy_operator, reduce_fixed_area


class PrefixBudget1DTests(unittest.TestCase):
    def test_partial_last_edge_matches_independent_gauss_integration(self):
        nodes=np.array([0.,1.5,4.,7.]);u=np.array([0.,.4,-.2,.5]);end=3.2;scale=12.
        R=prefix_energy_operator(nodes,end,scale)
        # Independent 3-point Gauss on both original intervals intersecting [0,3.2].
        g,w=np.polynomial.legendre.leggauss(3);energy=0.
        for lo,hi in [(0.,1.5),(1.5,3.2)]:
            samples=(lo+hi)/2+(hi-lo)*g/2
            ids=np.searchsorted(nodes,samples,side='right')-1
            slopes=(u[ids+1]-u[ids])/(nodes[ids+1]-nodes[ids])
            energy+=scale**2*(hi-lo)/2*float(w@slopes**2)
        self.assertAlmostEqual(float(np.linalg.norm(R@u)**2),energy,places=12)
        np.testing.assert_array_equal(R@np.ones(4),np.zeros(2))

    def test_each_prefix_uses_fixed_outer_scale_and_keeps_early_energy_visible(self):
        nodes=np.arange(0.,14.,2.);u=.1*np.minimum(nodes,2.);scale=12.
        energies=[];passes=[]
        for end in [2.,6.,12.]:
            R=prefix_energy_operator(nodes,end,scale);e=float(np.linalg.norm(R@u)**2)
            energies.append(e)
            # Independent intent u=(s/12)^3 has integral ell^2*9*r^5/(5*12^6).
            cap=scale**2*9*end**5/(5*12.**6);passes.append(e<=cap)
        np.testing.assert_allclose(energies,np.full(3,2.88),atol=1e-14,rtol=0.)
        self.assertFalse(passes[0]);self.assertTrue(passes[-1])

    def test_original_fixed_zero_and_area_survive_whitening_for_every_coordinate(self):
        H=np.diag([1.,2.,3.,4.,5.]);l=np.array([.1,-.3,.5,.7,-.2]);area=np.array([1.,2.,3.,4.,5.])
        lower=np.full(5,-np.inf);upper=np.full(5,np.inf);lower[[0,-1]]=upper[[0,-1]]=0.
        item=reduce_fixed_area(H,l,area,.4,lower,upper,7.)
        R=prefix_energy_operator(np.arange(5.),2.5,12.)
        for t in [np.zeros(2),np.array([.1,-.3]),np.array([-2.,5.])]:
            x=item['base']+item['transform']@t
            np.testing.assert_array_equal(x[[0,-1]],[0.,0.])
            self.assertAlmostEqual(float(area@x),.4,places=12)
            energy=lambda y:float((y@H@y+2*l@y)/7.)
            self.assertAlmostEqual(energy(x)-energy(item['base']),float(t@t+item['q']@t),places=12)
            # Reassociation changes floating-point summation at ~1e-15 relative.
            np.testing.assert_allclose(R@x,R@item['base']+(R@item['transform'])@t,atol=2e-14,rtol=2e-15)

    def test_bad_domain_inputs_are_rejected_without_clamping(self):
        for nodes,end,scale in [([1.,2.,3.],2.,12.),([0.,2.,2.],1.,12.),([0.,1.],2.,12.),([0.,1.],0.,12.),([0.,1.],1.,0.)]:
            with self.subTest(nodes=nodes,end=end,scale=scale),self.assertRaises(ValueError):
                prefix_energy_operator(nodes,end,scale)


if __name__=='__main__':unittest.main()
