"""Analytical curve and small source-grid checks; no photographs."""
import unittest

import numpy as np

from curve_similarity_field import (curve_windows, assemble_curve_operator,
                                    assemble_endpoint_jet_operator, _solve_jet_equalities, solve_problem)
from portable_field import build_problem, evaluate_motion, solve_problem as solve_membrane


def complex_points(points):
    points = np.asarray(points)
    return points[..., 0]+1j*points[..., 1]


def residual(windows, values):
    return np.sum(windows['coefficients']*values, axis=1)


class CurveWindowTests(unittest.TestCase):
    def source(self):
        return np.array([[0., 0.], [6., 0.], [6., 12.], [12., 12.]])

    def test_finite_global_rotation_scale_and_translation_have_zero_residual(self):
        windows = curve_windows(self.source(), half_window_px=2., center_spacing_px=2.)
        z = complex_points(windows['points'])
        for alpha, shift in [(1., 3.+4j), (np.exp(1.2j), -2.+5j), (1.4*np.exp(.7j), 1.-3j)]:
            displacement = (alpha-1)*z+shift
            np.testing.assert_allclose(residual(windows, displacement), 0., atol=1e-14, rtol=0.)

    def test_new_corner_and_local_stretch_have_nonzero_closed_form_residual(self):
        windows = curve_windows([[0., 0.], [12., 0.]], half_window_px=2., center_spacing_px=2.)
        x = windows['points'][..., 0]
        middle = np.flatnonzero(windows['centers_s'] == 6.)[0]
        for direction in [1., 1j]:
            displacement = direction*.1*np.maximum(x-6., 0.)
            self.assertAlmostEqual(abs(residual(windows, displacement)[middle]), .2, places=14)

    def test_collinear_source_subdivision_preserves_windows_and_arc_weights(self):
        original = curve_windows(self.source(), half_window_px=2., center_spacing_px=2.)
        split = curve_windows([[0., 0.], [3., 0.], [6., 0.], [6., 6.], [6., 12.], [9., 12.], [12., 12.]],
                              half_window_px=2., center_spacing_px=2.)
        for key in ['points', 'coefficients', 'weights', 'centers_s']:
            np.testing.assert_allclose(original[key], split[key], atol=1e-14, rtol=0.)
        self.assertEqual(float(original['weights'].sum()), 20.)

    def test_reverse_traversal_and_spatial_rotation_preserve_penalty(self):
        original = curve_windows(self.source(), half_window_px=2., center_spacing_px=2.)
        reversed_curve = curve_windows(self.source()[::-1], half_window_px=2., center_spacing_px=2.)
        displacement = .02*complex_points(original['points'])**2
        reversed_displacement = .02*complex_points(reversed_curve['points'])**2
        np.testing.assert_allclose(residual(reversed_curve, reversed_displacement),
                                   -residual(original, displacement)[::-1], atol=1e-14, rtol=0.)
        gamma = np.exp(.7j)
        rotated_source = gamma*complex_points(self.source())+(20.-7j)
        rotated_curve = curve_windows(np.column_stack([rotated_source.real, rotated_source.imag]),
                                      half_window_px=2., center_spacing_px=2.)
        np.testing.assert_allclose(residual(rotated_curve, gamma*displacement),
                                   residual(original, displacement), atol=2e-13, rtol=0.)

    def test_physical_scale_has_pixel_residual_and_source_arclength_weight_units(self):
        original = curve_windows(self.source(), half_window_px=2., center_spacing_px=2.)
        scaled = curve_windows(3*self.source(), half_window_px=6., center_spacing_px=6.)
        displacement = .02*complex_points(original['points'])**2
        old_residual = residual(original, displacement)
        new_residual = residual(scaled, 3*displacement)
        np.testing.assert_allclose(new_residual, 3*old_residual, atol=1e-13, rtol=0.)
        old_energy = np.sum(original['weights']*abs(old_residual)**2)
        new_energy = np.sum(scaled['weights']*abs(new_residual)**2)
        self.assertAlmostEqual(new_energy/old_energy, 27., places=12)
        # Density has inverse-pixel units: rescaling density by 1/3 makes
        # the weighted energy scale as displacement² (9), not as a volume.
        self.assertAlmostEqual((new_energy/3)/old_energy, 9., places=12)

    def test_short_curve_or_degenerate_window_chord_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exceed two half windows'):
            curve_windows([[0., 0.], [4., 0.]], half_window_px=2.)
        with self.assertRaisesRegex(ValueError, 'degenerate window chord'):
            curve_windows([[0., 0.], [2., 0.], [0., 0.], [0., 8.]], half_window_px=4.)

    def test_reference_radius_normalization_matches_quadratic_but_detects_single_radius_null_mode(self):
        for radius in [4., 12.]:
            windows = curve_windows([[0., 0.], [72., 0.]], half_window_px=radius)
            x = windows['points'][..., 0]
            normalization = (12./radius)**2
            # d=.5*kappa*x² on a straight source has C_r d=kappa*r²,
            # so the reference-normalized residual is exactly kappa*12².
            quadratic = .5*.001*x**2
            np.testing.assert_allclose(normalization*residual(windows, quadratic),
                                       .001*12.**2, atol=3e-14, rtol=0.)
            periodic = .1*np.sin(2*np.pi*x/12.)
            value = normalization*residual(windows, periodic)
            if radius == 12.:
                np.testing.assert_allclose(value, 0., atol=2e-14, rtol=0.)
            else:
                # sin(period=12) is invisible to +/-12, but +/-4 gives
                # (2*cos(2*pi/3)-2)*.1*sin = -.3*sin; normalization is 9.
                np.testing.assert_allclose(value, -2.7*np.sin(2*np.pi*windows['centers_s']/12.),
                                           atol=3e-14, rtol=0.)
                self.assertGreater(float(abs(value).max()), 2.)


class CurveFieldTests(unittest.TestCase):
    def make(self, delta=(0., 0.), **kwargs):
        source = np.array([[12., 10.], [12., 12.], [16., 12.]])
        stride = kwargs.pop('stride', 1)
        problem = build_problem((25, 25), source, source+delta, [[6., 16.]],
            crop=[0, 0, 25, 25], target_weights=1000., stride=stride, **kwargs)
        interface = {'schema': 'reviewed-visible-source-interface-v1', 'segments': [
            {'id': 'left-observed', 'source_points': [[4., 4.], [8., 10.], [9., 18.], [12., 20.]]},
            {'id': 'right-observed', 'source_points': [[20., 4.], [18., 10.], [17., 18.], [12., 20.]]}]}
        return problem, interface

    def test_identity_field_is_exact_and_does_not_claim_hard_shape_or_quality(self):
        problem, interface = self.make()
        result = solve_problem(problem, interface, half_window_px=2.)
        np.testing.assert_array_equal(result['source_displacement'], 0.)
        self.assertEqual(result['report']['curve_shape_energy'], 0.)
        self.assertEqual(result['report']['shape_residual_max_px'], 0.)
        self.assertFalse(result['report']['chord_angles_hard_constrained'])
        self.assertFalse(result['aesthetic_approval'])

    def test_hermitian_solution_matches_independent_real_block_spd_solve(self):
        problem, interface = self.make(delta=(.8, -.2))
        original_h, original_rhs = problem['hessian'].copy(), problem['rhs'].copy()
        assembled = assemble_curve_operator(problem, interface, half_window_px=2.)
        c = assembled['operator'].toarray()
        gram = c.conjugate().T @ (assembled['weights'][:, None]*c)
        augmented = original_h.toarray()+.05*gram
        self.assertGreater(float(np.linalg.eigvalsh(augmented).min()), 0.)
        self.assertGreater(float(abs(augmented.imag).max()), 1e-6)
        real_block = np.block([[augmented.real, -augmented.imag],
                               [augmented.imag, augmented.real]])
        packed_rhs = original_rhs.T.ravel()
        expected = np.linalg.solve(real_block, packed_rhs)
        result = solve_problem(problem, interface, half_window_px=2.)
        np.testing.assert_allclose(result['solution'].T.ravel(), expected, rtol=0, atol=1e-10)
        independent_energy = expected @ real_block @ expected-2*packed_rhs @ expected
        self.assertAlmostEqual(result['report']['objective_without_constant'], independent_energy, places=8)
        self.assertEqual((original_h != problem['hessian']).nnz, 0)
        np.testing.assert_array_equal(original_rhs, problem['rhs'])
        self.assertLess(result['report']['projected_backward_error'], 1e-10)

    def test_shape_prior_reduces_curve_residual_while_source_protection_remains_exact(self):
        problem, interface = self.make(delta=(.8, -.2))
        baseline = solve_membrane(problem)
        assembled = assemble_curve_operator(problem, interface, half_window_px=2.)
        baseline_residual = assembled['operator'] @ (baseline['solution'][:, 0]+1j*baseline['solution'][:, 1])
        baseline_shape_energy = .05*np.sum(assembled['weights']*abs(baseline_residual)**2)
        result = solve_problem(problem, interface, half_window_px=2.)
        self.assertLess(result['report']['curve_shape_energy'], baseline_shape_energy)
        yy, xx = np.meshgrid(np.linspace(15, 17, 7), np.linspace(5, 7, 7))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        np.testing.assert_array_equal(evaluate_motion(problem, result['solution'], points), 0.)
        field = result['source_displacement']
        np.testing.assert_array_equal(field[[0, 1, -2, -1]], 0.)
        np.testing.assert_array_equal(field[:, [0, 1, -2, -1]], 0.)
        self.assertEqual(result['fixed_source_motion_max_px'], 0.)

    def test_zero_density_recovers_the_original_two_component_solution(self):
        problem, interface = self.make(delta=(.8, -.2))
        expected = solve_membrane(problem)
        result = solve_problem(problem, interface, half_window_px=2., density=0.)
        np.testing.assert_allclose(result['solution'], expected['solution'], rtol=0, atol=1e-11)

    def test_reduction_and_background_equalities_cannot_be_silently_ignored(self):
        problem, interface = self.make(delta=(.8, -.2), stride=4)
        with self.assertRaisesRegex(ValueError, 'identity reduction'):
            solve_problem(problem, interface, half_window_px=2.)
        problem, interface = self.make(delta=(.8, -.2), lines=[
            {'start': [4., 8.], 'end': [18., 22.], 'samples': 2}])
        with self.assertRaisesRegex(ValueError, 'no background line equalities'):
            solve_problem(problem, interface, half_window_px=2.)

    def test_single_scale_is_bitwise_legacy_and_multiscale_uses_sum_of_normalized_grams(self):
        problem, interface = self.make(delta=(.8, -.2))
        interface['segments'] = [{'id': 'long-observed',
            'source_points': [[4., 4.], [20., 4.], [20., 20.], [4., 20.]]}]
        legacy = solve_problem(problem, interface)
        single = solve_problem(problem, interface, half_windows_px=(12.,))
        np.testing.assert_array_equal(legacy['source_displacement'], single['source_displacement'])
        self.assertEqual(legacy['report']['curve_shape_energy'], single['report']['curve_shape_energy'])
        multi = solve_problem(problem, interface, half_windows_px=(4., 12.))
        expected_h = problem['hessian'].toarray().astype(complex)
        z = multi['solution'][:, 0]+1j*multi['solution'][:, 1]
        energies = []
        for radius, scale_report in zip([4., 12.], multi['report']['scales']):
            assembled = assemble_curve_operator(problem, interface, half_window_px=radius)
            c, w = assembled['operator'].toarray(), assembled['weights']
            normalized = (12./radius)**2*c
            expected_h += .05*normalized.conjugate().T @ (w[:, None]*normalized)
            energy = float(.05*np.sum(w*abs(normalized @ z)**2))
            energies.append(energy)
            self.assertAlmostEqual(scale_report['weighted_shape_energy'], energy, places=12)
            self.assertAlmostEqual(scale_report['raw_shape_residual_max_px'], float(abs(c @ z).max()), places=12)
        expected_rhs = problem['rhs'][:, 0]+1j*problem['rhs'][:, 1]
        np.testing.assert_allclose(z, np.linalg.solve(expected_h, expected_rhs), atol=1e-10, rtol=0.)
        self.assertAlmostEqual(multi['report']['curve_shape_energy'], sum(energies), places=12)

    def test_membrane_jets_preserve_endpoint_motion_and_interior_tangent_derivative(self):
        problem, interface = self.make(delta=(.8, -.2))
        # Endpoints at free grid vertices, including a reversed/down-left
        # incident segment, exercise the physical one-sided grid derivative.
        interface['segments'] = [
            {'id': 'left', 'source_points': [[8., 8.], [12., 12.], [16., 16.]]},
            {'id': 'right', 'source_points': [[16., 8.], [14., 12.], [10., 18.]]}]
        baseline = solve_membrane(problem)
        result = solve_problem(problem, interface, half_window_px=2., endpoint_reference='membrane-jet')
        jet = assemble_endpoint_jet_operator(problem, interface)
        e = jet['operator']
        np.testing.assert_allclose(e @ result['solution'], e @ baseline['solution'], atol=1e-10, rtol=0.)
        self.assertEqual(e.shape[0], 8)
        report = result['report']['endpoint_constraints']
        self.assertEqual(report['original_complex_rows'], 8)
        self.assertLess(report['value_max_residual_px'], 1e-9)
        self.assertLess(report['tangent_max_residual'], 1e-9)
        self.assertLess(report['kkt_backward_error'], 1e-10)
        self.assertFalse(report['normal_derivative_constrained'])
        self.assertEqual(result['report']['factorization_count'], 2)
        self.assertLessEqual(result['report']['objective_without_constant'],
            report['baseline_augmented_objective_without_constant']+1e-9)
        self.assertFalse(result['aesthetic_approval'])

    def test_jet_solution_matches_independent_dense_kkt(self):
        problem, interface = self.make(delta=(.8, -.2))
        interface['segments'] = [{'id': 'free-endpoints',
            'source_points': [[8.5, 8.5], [12., 10.], [15.5, 17.5]]}]
        assembled = assemble_curve_operator(problem, interface, half_window_px=2.)
        c, w = assembled['operator'].toarray(), assembled['weights']
        h = problem['hessian'].toarray().astype(complex)
        rhs = problem['rhs'][:, 0]+1j*problem['rhs'][:, 1]
        augmented = h+.05*c.conjugate().T @ (w[:, None]*c)
        e = assemble_endpoint_jet_operator(problem, interface)['operator'].toarray()
        g = e @ np.linalg.solve(h, rhs)
        kkt = np.block([[augmented, e.T], [e, np.zeros((len(e), len(e)))]])
        expected = np.linalg.solve(kkt, np.r_[rhs, g])
        result = solve_problem(problem, interface, half_window_px=2., endpoint_reference='membrane-jet')
        z = complex_points(result['solution'])
        np.testing.assert_allclose(z, expected[:len(z)], atol=1e-10, rtol=0.)
        np.testing.assert_allclose(augmented @ z-rhs+e.T @ result['multipliers'], 0., atol=1e-9, rtol=0.)
        self.assertGreater(float(abs(augmented @ z-rhs).max()), 1e-5)

    def test_jets_allow_fixed_and_repeated_rows_and_zero_density_matches_baseline(self):
        problem, interface = self.make(delta=(.8, -.2))
        interface['segments'].append({**interface['segments'][0], 'id': 'same-curve'})
        interface['segments'].append({'id': 'fixed-curve', 'source_points': [[0., 0.], [0., 12.]]})
        result = solve_problem(problem, interface, half_window_px=2., endpoint_reference='membrane-jet')
        report = result['report']['endpoint_constraints']
        self.assertEqual(report['original_complex_rows'], 16)
        self.assertGreater(report['zero_rows'], 0)
        self.assertLess(report['rank'], report['original_complex_rows']-report['zero_rows'])
        baseline = solve_membrane(problem)
        zero = solve_problem(problem, interface, half_window_px=2., density=0., endpoint_reference='membrane-jet')
        np.testing.assert_allclose(zero['solution'], baseline['solution'], atol=1e-10, rtol=0.)
        legacy = solve_problem(problem, interface, half_window_px=2.)
        explicit = solve_problem(problem, interface, half_window_px=2., endpoint_reference='free')
        np.testing.assert_array_equal(legacy['source_displacement'], explicit['source_displacement'])

    def test_jet_operator_uses_incident_cell_at_grid_lines_not_an_offset_point(self):
        problem, _ = self.make()
        grid = problem['grid']
        yy, xx = np.mgrid[:grid['height'], :grid['width']]
        # A nodal x²+y² field is piecewise linear along each grid axis.
        nodal = ((xx*2.)**2+(yy*2.)**2).ravel()
        z = nodal[problem['mapping'] >= 0]
        for p, q, expected in [([12., 12.], [10., 10.], -44/np.sqrt(2.)),
                                ([12., 12.], [14., 14.], 52/np.sqrt(2.)),
                                ([12., 12.], [12., 14.], 26.),
                                ([np.nextafter(12., 0.), 12.], [14., 12.], 22.)]:
            interface = {'schema': 'reviewed-visible-source-interface-v1', 'segments': [
                {'id': 'incident', 'source_points': [p, q]}]}
            assembled = assemble_endpoint_jet_operator(problem, interface)
            value = assembled['operator'] @ z
            self.assertAlmostEqual(value[1], expected, places=12)
            self.assertAlmostEqual(value[0], p[0]**2+p[1]**2, places=11)
        # An affine similarity has the exact finite displacement and derivative,
        # including at a corner approached through the negative quadrant.
        alpha = 1.2*np.exp(.7j)-1
        affine = alpha*(xx*2.+1j*yy*2.)+(2.+3j)
        z = affine.ravel()[problem['mapping'] >= 0]
        interface['segments'][0]['source_points'] = [[12., 12.], [10., 10.]]
        assembled = assemble_endpoint_jet_operator(problem, interface)
        actual = assembled['operator'] @ z
        self.assertAlmostEqual(abs(actual[0]-(alpha*(12.+12j)+2.+3j)), 0., places=12)
        self.assertAlmostEqual(abs(actual[1]-alpha*(-1.-1j)/np.sqrt(2.)), 0., places=12)

    def test_jet_operator_identity_exterior_and_invalid_mode(self):
        problem, _ = self.make()
        problem['image_shape'] = [40, 40]
        interface = {'schema': 'reviewed-visible-source-interface-v1', 'segments': [
            {'id': 'exterior', 'source_points': [[28., 12.], [32., 20.]]}]}
        e = assemble_endpoint_jet_operator(problem, interface)['operator']
        self.assertEqual(e.nnz, 0)
        result = solve_problem(problem, interface, half_window_px=2., endpoint_reference='membrane-jet')
        self.assertEqual(result['report']['endpoint_constraints']['rank'], 0)
        with self.assertRaisesRegex(ValueError, 'endpoint_reference'):
            solve_problem(problem, interface, half_window_px=2., endpoint_reference='unknown')

    def test_inconsistent_redundant_or_fixed_jet_targets_and_nonfinite_rows_reject(self):
        from scipy.sparse import csc_matrix, eye
        from scipy.sparse.linalg import splu
        factor = splu(eye(2, format='csc', dtype=complex))
        e = csc_matrix([[1., 0.], [2., 0.], [0., 0.]])
        for reference in [[1., 3., 0.], [1., 2., .1]]:
            with self.assertRaisesRegex(ValueError, 'inconsistent endpoint'):
                _solve_jet_equalities(factor, np.zeros(2, complex), e, reference)
        with self.assertRaisesRegex(ValueError, 'finite matching endpoint'):
            _solve_jet_equalities(factor, np.zeros(2, complex), e, [1., np.nan, 0.])
        e[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'finite matching endpoint'):
            _solve_jet_equalities(factor, np.zeros(2, complex), e, [1., 2., 0.])


if __name__ == '__main__':
    unittest.main()
