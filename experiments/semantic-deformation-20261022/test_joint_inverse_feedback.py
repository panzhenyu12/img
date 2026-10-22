"""Independent matrix and sampled-inverse checks for final-map feedback.

Expected acceptance uses the actual 2x2 final map Jacobian, without taking
another inverse and without copying the source-inverse conic formulas.
"""
import unittest

import numpy as np
from scipy.sparse import csr_matrix


def direct_final_metrics(e,c,angle=0.):
    e,c=np.broadcast_arrays(np.asarray(e,float),np.asarray(c,float))
    across=np.array([np.cos(angle),np.sin(angle)])
    down=np.array([-np.sin(angle),np.cos(angle)])
    gradient=(e-1.)[...,None]*across+c[...,None]*down
    matrix=np.eye(2)+across[:,None]*gradient[...,None,:]
    eigen=np.linalg.eigvalsh((matrix+matrix.swapaxes(-1,-2))*.5)
    skew=abs(matrix[...,0,1]-matrix[...,1,0])*.5
    determinant=np.linalg.det(matrix)
    return {'minimum':eigen[...,0],'maximum':eigen[...,1],'skew':skew,
            'determinant':determinant,
            'passed':(eigen[...,0]>=.52)&(eigen[...,1]<=1.8)&(skew<=.24)&(determinant>0.)}


def block_margins(block,vector):
    slack=np.asarray(block['b']-block['A']@vector);offset=0;values=[]
    for cone in block['cones']:
        part=slack[offset:offset+cone.dim];offset+=cone.dim
        if type(cone).__name__=='SecondOrderConeT':values.append(part[0]-np.linalg.norm(part[1:]))
        elif type(cone).__name__=='NonnegativeConeT':values.append(float(part.min()))
        else:raise AssertionError('unexpected final-Jacobian cone type')
    if offset!=len(slack):raise AssertionError('cone dimensions must exhaust rows')
    return np.asarray(values).reshape(-1,3)


class JointInverseFeedbackTests(unittest.TestCase):
    def test_direct_final_matrix_matches_cones_across_rotated_frames(self):
        from probe_joint_inverse_feedback import final_jacobian_soc_block
        rng=np.random.default_rng(871);n=4096
        e=rng.uniform(-1.,3.,n);c=rng.uniform(-1.,1.,n)
        block=final_jacobian_soc_block(csr_matrix((n,2)),e,csr_matrix((n,2)),c,5)
        cone=np.all(block_margins(block,np.zeros(5))>=0.,axis=1)
        self.assertTrue(cone.any());self.assertTrue((~cone).any())
        for angle in [0.,.37,1.7,-2.2]:
            np.testing.assert_array_equal(cone,direct_final_metrics(e,c,angle)['passed'])

    def test_final_map_boundaries_and_wrong_inverse_interpretation(self):
        from probe_joint_inverse_feedback import final_jacobian_soc_block
        # E is already the final Jacobian; E=.53 is legal whereas interpreting
        # it as a source Jacobian and testing its inverse would reject 1/.53.
        e=np.array([.52-1e-6,.52+1e-6,1.8-1e-6,1.8+1e-6,1.,1.,1.,.53,0.,-1.])
        c=np.array([0.,0.,0.,0.,.48-1e-6,.48+1e-6,3.,0.,0.,0.])
        expected=np.array([False,True,True,False,True,False,False,True,False,False])
        block=final_jacobian_soc_block(csr_matrix((len(e),1)),e,csr_matrix((len(e),1)),c,1)
        np.testing.assert_array_equal(np.all(block_margins(block,np.zeros(1))>=0.,axis=1),expected)
        np.testing.assert_array_equal(direct_final_metrics(e,c)['passed'],expected)
        self.assertEqual(float(direct_final_metrics(1.,3.)['determinant']),1.)
        for e0,c0,active in [(.52,0.,0),(1.8,0.,1),(1.,.48,2)]:
            edge=final_jacobian_soc_block(csr_matrix((1,1)),[e0],csr_matrix((1,1)),[c0],1)
            self.assertAlmostEqual(block_margins(edge,np.zeros(1))[0,active],0.,places=13)

    def test_affine_rows_constants_and_padding_use_original_variables_only(self):
        from probe_joint_inverse_feedback import final_jacobian_soc_block
        rng=np.random.default_rng(431);n=17;source_size=7;extra=6
        er=csr_matrix(rng.normal(size=(n,source_size))*.2)
        cr=csr_matrix(rng.normal(size=(n,source_size))*.2)
        ec=rng.uniform(.7,1.3,n);cc=rng.uniform(-.3,.3,n)
        block=final_jacobian_soc_block(er,ec,cr,cc,source_size+extra)
        self.assertEqual(block['A'].shape,(8*n,source_size+extra))
        self.assertEqual(block['A'][:,source_size:].nnz,0)
        for _ in range(20):
            z=rng.normal(size=source_size+extra)
            actual=np.all(block_margins(block,z)>=0.,axis=1)
            expected=direct_final_metrics(ec+er@z[:source_size],cc+cr@z[:source_size],.83)['passed']
            np.testing.assert_array_equal(actual,expected)

    def test_pixel_edge_gradient_chain_matches_fresh_inverse_finite_difference(self):
        from inverse_field_linearization import linearize_inverse_samples
        from probe_joint_inverse_feedback import final_corner_linearization
        from test_inverse_field_linearization import fixture
        grid,mapping,z,across=fixture();down=np.array([-across[1],across[0]])
        grid['origin']=[31.,47.]
        points=np.array([[8.,8.],[9.,8.],[8.,9.],[9.,9.]])+grid['origin']
        tangent=linearize_inverse_samples(grid,mapping,z,across,points)
        # Actual pixel-cell TL/TR/BL/BR edge differences, at spacing 1px.
        dx=csr_matrix([[-1.,1.,0.,0.],[-1.,1.,0.,0.],[0.,0.,-1.,1.],[0.,0.,-1.,1.]])
        dy=csr_matrix([[-1.,0.,1.,0.],[0.,-1.,0.,1.],[-1.,0.,1.,0.],[0.,-1.,0.,1.]])
        er=(across[0]*dx+across[1]*dy)@tangent['rows']
        cr=(down[0]*dx+down[1]*dy)@tangent['rows']
        assembled=final_corner_linearization(grid,mapping,z,across,down,
            np.array([[corner,8,8] for corner in range(4)]))
        np.testing.assert_allclose(assembled['e_rows'].toarray(),er.toarray(),atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(assembled['c_rows'].toarray(),cr.toarray(),atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(assembled['e_const']+assembled['e_rows']@z,
            1.+(across[0]*dx+across[1]*dy)@tangent['inverse_scalar'],atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(assembled['c_const']+assembled['c_rows']@z,
            (down[0]*dx+down[1]*dy)@tangent['inverse_scalar'],atol=1e-13,rtol=1e-12)
        direction=np.random.default_rng(967).normal(size=len(z))*.2;eps=1e-4
        plus=linearize_inverse_samples(grid,mapping,z+eps*direction,across,points)['inverse_scalar']
        minus=linearize_inverse_samples(grid,mapping,z-eps*direction,across,points)['inverse_scalar']
        scalar_derivative=(plus-minus)/(2*eps)
        expected_e=(across[0]*dx+across[1]*dy)@scalar_derivative
        expected_c=(down[0]*dx+down[1]*dy)@scalar_derivative
        np.testing.assert_allclose(er@direction,expected_e,atol=2e-8,rtol=1e-7)
        np.testing.assert_allclose(cr@direction,expected_c,atol=2e-8,rtol=1e-7)


if __name__=='__main__':unittest.main()
