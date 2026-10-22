"""Direct matrix oracle for the original inverse geometry thresholds.

The oracle deliberately forms, rotates and inverts the full 2x2 Jacobian;
it does not derive its expected result from the conic helper's formulas.
"""
import unittest

import numpy as np


def direct_inverse_metrics(d, b, rotation=0.):
    d, b = np.broadcast_arrays(np.asarray(d, float), np.asarray(b, float))
    across = np.array([np.cos(rotation), np.sin(rotation)])
    down = np.array([-np.sin(rotation), np.cos(rotation)])
    gradient = (d-1.)[..., None]*across+b[..., None]*down
    forward = np.eye(2)+across[:, None]*gradient[..., None, :]
    inverse = np.linalg.inv(forward)
    eigenvalues = np.linalg.eigvalsh((inverse+inverse.swapaxes(-1, -2))/2.)
    skew = abs(inverse[..., 1, 0]-inverse[..., 0, 1])/2.
    determinant = np.linalg.det(inverse)
    return {'minimum': eigenvalues[..., 0], 'maximum': eigenvalues[..., 1],
            'skew': skew, 'determinant': determinant,
            'passed': ((eigenvalues[..., 0] >= .52) & (eigenvalues[..., 1] <= 1.8)
                       & (skew <= .24) & (determinant > 0.))}


class RankOneInverseContractTests(unittest.TestCase):
    def test_direct_matrix_oracle_matches_cones_in_rotated_frames(self):
        from joint_jaw_boundary_field import rank_one_inverse_soc_slacks
        rng = np.random.default_rng(772)
        d = rng.uniform(-2., 3., 4096)
        b = rng.uniform(-2., 2., 4096)
        cone = np.all(rank_one_inverse_soc_slacks(d, b) >= 0., axis=-1)
        self.assertTrue(np.any(cone))
        self.assertTrue(np.any(~cone))
        reference = direct_inverse_metrics(d, b)
        for angle in [0., .37, 1.7, -2.2]:
            measured = direct_inverse_metrics(d, b, angle)
            np.testing.assert_array_equal(cone, measured['passed'])
            # Stay away from the inverse singularity when comparing values;
            # pass/fail equivalence above still includes those hard cases.
            stable = abs(d) > .01
            for name in ['minimum', 'maximum', 'skew', 'determinant']:
                np.testing.assert_allclose(measured[name][stable], reference[name][stable],
                                           atol=2e-10, rtol=2e-12)

    def test_positive_determinant_does_not_accept_large_transverse_shear(self):
        from joint_jaw_boundary_field import rank_one_inverse_soc_slacks
        metrics = direct_inverse_metrics(1., 3.)
        self.assertAlmostEqual(float(metrics['determinant']), 1.)
        self.assertAlmostEqual(float(metrics['minimum']), -.5)
        self.assertAlmostEqual(float(metrics['maximum']), 2.5)
        self.assertFalse(bool(metrics['passed']))
        self.assertTrue(np.any(rank_one_inverse_soc_slacks(1., 3.) < 0.))

    def test_nonpositive_directional_jacobian_is_rejected(self):
        from joint_jaw_boundary_field import rank_one_inverse_soc_slacks
        d = np.array([-4., -1., -.01, 0.])
        b = np.array([0., 1., -2., 0.])
        self.assertFalse(np.any(np.all(rank_one_inverse_soc_slacks(d, b) >= 0., axis=-1)))
        self.assertFalse(np.any(direct_inverse_metrics(d[:-1], b[:-1])['passed']))

    def test_known_original_contract_boundaries_and_both_sides(self):
        from joint_jaw_boundary_field import rank_one_inverse_soc_slacks
        # At b=0 the inverse diagonal limits occur at D=1/.52 and 1/1.8.
        # At D=1, b=.48 is the original skew-half-difference boundary.
        boundary = [(1./.52, 0., 0), (1./1.8, 0., 1), (1., .48, 2)]
        for d, b, active in boundary:
            slack = rank_one_inverse_soc_slacks(d, b)
            self.assertAlmostEqual(float(slack[active]), 0., places=13)
            self.assertTrue(np.all(np.delete(slack, active) > 0.))
        for d, b, expected in [(1./.52-1e-6, 0., True), (1./.52+1e-6, 0., False),
                                (1./1.8+1e-6, 0., True), (1./1.8-1e-6, 0., False),
                                (1., .48-1e-6, True), (1., .48+1e-6, False)]:
            self.assertEqual(bool(direct_inverse_metrics(d, b)['passed']), expected)
            self.assertEqual(bool(np.all(rank_one_inverse_soc_slacks(d, b) >= 0.)), expected)


if __name__ == '__main__': unittest.main()
