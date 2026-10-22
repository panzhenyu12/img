"""Compare local inverse derivatives with independently re-inverted fields."""
import unittest
import numpy as np

from inverse_field_linearization import linearize_inverse_samples


def fixture():
    grid={'origin':[0.,0.],'spacing':[2.,2.],'width':11,'height':11}
    mapping=np.full((11,11),-1,dtype=int);mapping[2:-2,2:-2]=np.arange(49).reshape(7,7)
    y,x=np.mgrid[2:9,2:9]
    z=(.6*np.sin(np.pi*x/10.)**2*np.sin(np.pi*y/10.)**2).ravel()
    return grid,mapping.ravel(),z,np.array([np.cos(.27),np.sin(.27)])


class InverseFieldLinearizationTests(unittest.TestCase):
    def test_derivative_matches_reinversion_instead_of_fixed_source_sampling(self):
        grid,mapping,z,across=fixture()
        points=np.array([[8.3,7.6],[12.4,10.7],[13.2,14.4]])
        tangent=linearize_inverse_samples(grid,mapping,z,across,points)
        direction=np.random.default_rng(398).normal(size=len(z))*.2
        eps=1e-4
        plus=linearize_inverse_samples(grid,mapping,z+eps*direction,across,points)
        minus=linearize_inverse_samples(grid,mapping,z-eps*direction,across,points)
        finite_difference=(plus['inverse_scalar']-minus['inverse_scalar'])/(2*eps)
        self.assertGreater(np.linalg.norm(finite_difference),.01)
        np.testing.assert_allclose(tangent['rows']@direction,finite_difference,atol=2e-8,rtol=1e-7)

    def test_folded_source_cannot_produce_an_inverse_derivative(self):
        grid,mapping,z,across=fixture();z[:]=0.;z[24]=20.
        with self.assertRaisesRegex(ValueError,'strongly monotone'):
            linearize_inverse_samples(grid,mapping,z,across,[[10.,10.]])


if __name__=='__main__':unittest.main()
