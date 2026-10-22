"""Analytic counterexamples and certificates for interval polynomial cones."""
import unittest

import clarabel
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, vstack

from interval_polynomial_cones import nonnegative_cubic_block


def feasibility(coefficients):
    # Include an actual original decision variable fixed to one. This tests
    # the c+Cz coefficient/sign contract, not only constant polynomials.
    coefficients = np.asarray(coefficients, float)
    c = np.array([.31, -.27, .19, -.11])
    block = nonnegative_cubic_block(csr_matrix((coefficients-c)[:, None]), c,
                                   auxiliary_start=1, total_variables=7)
    fix = csr_matrix(([1.], ([0], [0])), shape=(1, 7))
    a = vstack([fix, block['A']], format='csc')
    b = np.r_[1., block['b']]
    settings = clarabel.DefaultSettings(); settings.verbose = False
    settings.max_iter = 100; settings.tol_feas = 1e-10
    settings.tol_gap_abs = settings.tol_gap_rel = 1e-10
    result = clarabel.DefaultSolver(csc_matrix((7, 7)), np.zeros(7), a, b,
                                   [clarabel.ZeroConeT(1), *block['cones']], settings).solve()
    return result, block


class IntervalPolynomialConesTests(unittest.TestCase):
    def test_four_positive_nodes_do_not_hide_negative_cubic_interior(self):
        # p=(t-.19)^2*(1+t)-.004; p(.19)=-.004, but all four old
        # interpolation nodes t=0,1/3,2/3,1 are strictly positive.
        coefficients = np.array([.0321, -.3439, .62, 1.])
        self.assertTrue(np.all(np.polynomial.polynomial.polyval(np.linspace(0., 1., 4), coefficients) > 0.))
        result, _ = feasibility(coefficients)
        self.assertEqual(result.status, clarabel.SolverStatus.PrimalInfeasible)

    def test_nonnegative_cubic_with_internal_double_root_has_psd_certificate(self):
        # p=(t-.19)^2*(1+t). An explicit certificate is Q=2vv^T,
        # R=vv^T with v=(-.19,1), independent of the assembly formulas.
        coefficients = np.array([.0361, -.3439, .62, 1.])
        result, block = feasibility(coefficients)
        self.assertEqual(result.status, clarabel.SolverStatus.Solved)
        vector = np.array([1., .0722, -.38, 2., .0361, -.19, 1.])
        slack = block['b']-block['A']@vector
        np.testing.assert_allclose(slack[:4], 0., atol=1e-14, rtol=0.)
        for start in [4, 7]:
            self.assertAlmostEqual(float(slack[start]-np.linalg.norm(slack[start+1:start+3])), 0., places=13)

    def test_boundary_zero_quadratic_is_allowed_and_its_negative_rejected(self):
        # t(1-t) vanishes at both endpoints and is positive inside.
        result, _ = feasibility([0., 1., -1., 0.])
        self.assertEqual(result.status, clarabel.SolverStatus.Solved)
        result, _ = feasibility([0., -1., 1., 0.])
        self.assertEqual(result.status, clarabel.SolverStatus.PrimalInfeasible)


if __name__ == '__main__': unittest.main()
