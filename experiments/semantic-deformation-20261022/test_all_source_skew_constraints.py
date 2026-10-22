"""Direct matrix-oracle checks without a photo, solve, or changed physical gate."""
import unittest

import numpy as np
from scipy.sparse import eye, vstack

from all_source_skew_constraints import all_source_skew_constraints
from joint_jaw_boundary_field import all_corner_inverse_check, corner_gradient_rows
from forward import cell_jacobians


def problem(width=3, height=2, spacing=(1., 1.), mapping=None):
    if mapping is None:
        mapping = np.arange(width*height)
    mapping = np.asarray(mapping)
    size = int(np.count_nonzero(mapping >= 0))
    return {'grid': {'origin': [-2., 5.], 'spacing': list(spacing),
                     'width': width, 'height': height},
            'mapping': mapping, 'hessian': eye(size, format='csc')}


def expand_slacks(block, z):
    retained = block['b']-block['A']@z
    provenance = block['original_halfspace_to_row']
    result = np.full(provenance.shape, .48)
    nonzero = provenance >= 0
    result[nonzero] = retained[provenance[nonzero]]
    return result


class AllSourceSkewTests(unittest.TestCase):
    def test_violation_in_adjacent_cell_is_not_hidden_by_old_corner_cuts(self):
        source = problem(); across = np.array([1., 0.]); down = np.array([0., 1.])
        # All four corners of the left cell pass. The adjacent right cell has
        # inverse skew .3 at its top-right corner, beyond the original .24.
        z = np.array([0., 0., 0., 0., 0., .6])
        p, b = corner_gradient_rows(source, [[c, 0, 0] for c in range(4)], across, down)
        old = vstack([-.48*p-b, -.48*p+b])
        self.assertTrue(np.all(.48-old@z >= 0.))
        oracle, _ = all_corner_inverse_check(source, z, across)
        self.assertAlmostEqual(oracle['inverse_absolute_skew_max'], .3)
        full = all_source_skew_constraints(source, across, down)
        self.assertTrue(np.any(full['b']-full['A']@z < 0.),
                        'all-corner assembly must reject violations outside the old active cuts')

    def test_all_corner_constraints_match_independent_direct_inverse_oracle(self):
        source = problem(6, 5, spacing=(1.7, 1.7))
        angle = .61
        across = np.array([np.cos(angle), np.sin(angle)])
        down = np.array([-np.sin(angle), np.cos(angle)])
        block = all_source_skew_constraints(source, across, down)
        rng = np.random.default_rng(831); base = rng.uniform(-1., 1., 30)
        decisions = []
        for strength in [0., .05, .25]:
            z = 1.7*strength*base
            forward = cell_jacobians((z[:, None]*across).reshape(5, 6, 2)/1.7)
            determinant = np.linalg.det(forward)
            self.assertGreater(float(determinant.min()), 0.)
            inverse = np.linalg.inv(forward)
            measured_skew = .5*abs(inverse[..., 1, 0]-inverse[..., 0, 1])
            margin = expand_slacks(block, z).min(axis=0)/(2.*determinant)
            np.testing.assert_allclose(margin, .24-measured_skew, atol=2e-16, rtol=2e-15)
            oracle, _ = all_corner_inverse_check(source, z, across)
            self.assertAlmostEqual(oracle['inverse_absolute_skew_max'], float(measured_skew.max()))
            passes = bool(np.all(block['b']-block['A']@z >= 0.))
            self.assertEqual(passes, oracle['inverse_absolute_skew_max'] <= .24)
            decisions.append(passes)
        self.assertEqual(set(decisions), {False, True})

    def test_rotated_frame_and_physical_spacing_preserve_known_shear_limit(self):
        angle = .73
        across = np.array([np.cos(angle), np.sin(angle)])
        down = np.array([-np.sin(angle), np.cos(angle)])
        source = problem(4, 3, spacing=(2., 3.))
        y, x = np.mgrid[:3, :4]
        world = np.stack([-2.+2.*x, 5.+3.*y], axis=-1)
        block = all_source_skew_constraints(source, across, down)
        # Affine displacement u=gradient dot x gives a constant Jacobian. The
        # direct numerical matrix inverse supplies the expected skew margin.
        for shear in [-.7, -.3, .3, .7]:
            gradient = .2*across+shear*down
            z = (world@gradient).ravel()
            inverse = np.linalg.inv(np.eye(2)+np.outer(across, gradient))
            skew = .5*abs(inverse[1, 0]-inverse[0, 1])
            actual = expand_slacks(block, z).min(axis=0)/(2.*np.linalg.det(np.linalg.inv(inverse)))
            np.testing.assert_allclose(actual, .24-skew, atol=5e-16, rtol=1e-14)
            self.assertEqual(bool(np.all(block['b']-block['A']@z >= 0.)), abs(shear) < .576)
        # Reversing the down axis swaps signs, never the feasible region.
        flipped = all_source_skew_constraints(source, across, -down)
        np.testing.assert_allclose(expand_slacks(block, z)[::-1], expand_slacks(flipped, z), atol=0., rtol=0.)

    def test_fixed_zero_mapping_omits_only_tautologies_and_exact_duplicates(self):
        full_source = problem(4, 3)
        across = np.array([1., 0.]); down = np.array([0., 1.])
        full = all_source_skew_constraints(full_source, across, down)
        free = np.array([11, 10, 7, 6, 3, 2])
        mapping = np.full(12, -1); mapping[free] = np.arange(len(free))
        source = problem(4, 3, mapping=mapping)
        reduced = all_source_skew_constraints(source, across, down)
        z = np.linspace(-.2, .3, len(free)); scalar = np.zeros(12); scalar[free] = z
        def original_rows(block):
            index = block['original_halfspace_to_row'].ravel()
            rows = np.zeros((len(index), block['A'].shape[1]))
            rows[index >= 0] = block['A'][index[index >= 0]].toarray()
            return rows
        # Column permutation changes floating-point summation order, but the
        # original coefficients must be bit-for-bit identical after restriction.
        np.testing.assert_array_equal(original_rows(reduced), original_rows(full)[:, free])
        np.testing.assert_allclose(expand_slacks(reduced, z), expand_slacks(full, scalar), atol=1e-16, rtol=0.)
        np.testing.assert_array_equal(source['mapping'], mapping)
        self.assertEqual(reduced['A'].shape[1], len(free))
        report = reduced['report']
        self.assertEqual(report['source_corners'], 24)
        self.assertEqual(report['covered_source_corners'], 24)
        self.assertEqual(report['covered_active_source_corners'], report['active_source_corners'])
        self.assertGreater(report['tautological_zero_halfspaces_removed'], 0)
        self.assertGreater(report['exact_duplicate_halfspaces_removed'], 0)
        self.assertEqual(report['halfspaces_before_reduction'], report['retained_halfspaces']+
                         report['tautological_zero_halfspaces_removed']+report['exact_duplicate_halfspaces_removed'])
        self.assertEqual(sum(c.dim for c in reduced['cones']), len(reduced['b']))

    def test_all_fixed_grid_needs_no_constraints_and_is_not_invertibility_proof(self):
        source = problem(mapping=np.full(6, -1))
        empty = all_source_skew_constraints(source, [1., 0.], [0., 1.])
        self.assertEqual(empty['A'].shape, (0, 0)); self.assertEqual(empty['cones'], [])
        self.assertEqual(empty['report']['covered_source_corners'], 8)
        self.assertEqual(empty['report']['active_source_corners'], 0)
        np.testing.assert_array_equal(expand_slacks(empty, np.empty(0)), np.full((2, 4, 1, 2), .48))
        source = problem(); block = all_source_skew_constraints(source, [1., 0.], [0., 1.])
        # u=-x has d=b=0. The original skew halfspaces permit it; the direct
        # inverse oracle must still reject it as singular.
        z = np.array([0., -1., -2., 0., -1., -2.])
        np.testing.assert_array_equal(expand_slacks(block, z), np.zeros((2, 4, 1, 2)))
        oracle, _ = all_corner_inverse_check(source, z, np.array([1., 0.]))
        self.assertFalse(oracle['accepted'])
        self.assertFalse(block['report']['strict_invertibility_certified'])

    def test_invalid_original_mapping_frame_and_grid_are_rejected(self):
        source = problem()
        for across, down in [([2., 0.], [0., 1.]), ([1., 0.], [.1, 1.]),
                             ([1., 0.], [0., np.nan])]:
            with self.subTest(across=across, down=down), self.assertRaisesRegex(ValueError, 'orthonormal'):
                all_source_skew_constraints(source, across, down)
        for mapping in [[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 6], [-2, 0, 1, 2, 3, 4]]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                all_source_skew_constraints(problem(mapping=mapping), [1., 0.], [0., 1.])
        for spacing in [(0., 1.), (np.inf, 1.), (-1., 1.)]:
            with self.subTest(spacing=spacing), self.assertRaisesRegex(ValueError, 'grid'):
                all_source_skew_constraints(problem(spacing=spacing), [1., 0.], [0., 1.])


if __name__ == '__main__':
    unittest.main()
