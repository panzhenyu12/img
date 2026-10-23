"""Small analytical graph tests; no photographs or system builds."""
import unittest

import numpy as np

from arap_field import graph_arap_energy, graph_rotations, solve_problem
from portable_field import build_problem, evaluate_motion, solve_problem as solve_membrane


class GraphARAPTests(unittest.TestCase):
    def points(self):
        xx, yy = np.meshgrid(np.arange(6)*2., np.arange(5)*3.)
        return np.stack([xx, yy], axis=-1)

    def test_identity_rotations_equal_original_membrane_energy(self):
        displacement = np.random.default_rng(13).normal(size=(5, 6, 2))
        hx, hy = 2., 3.
        original = hx*hy*(np.sum((np.diff(displacement, axis=1)/hx)**2)
                           + np.sum((np.diff(displacement, axis=0)/hy)**2))
        identity = np.broadcast_to(np.eye(2), (5, 6, 2, 2))
        self.assertAlmostEqual(graph_arap_energy(displacement, (hx, hy), identity),
                               original, places=11)

    def test_translation_and_finite_rotation_have_zero_energy(self):
        angle = 1.2
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        for expected in [np.eye(2), rotation]:
            displacement = self.points() @ (expected-np.eye(2)).T + [5., -3.]
            rotations = graph_rotations(displacement, (2., 3.))
            np.testing.assert_allclose(rotations, np.broadcast_to(expected, rotations.shape),
                                       rtol=0, atol=1e-14)
            self.assertLess(graph_arap_energy(displacement, (2., 3.), rotations), 1e-25)

    def test_uniform_stretch_costs_energy_and_is_not_fitted_as_a_similarity(self):
        displacement = .2*self.points()
        rotations = graph_rotations(displacement, (2., 3.))
        np.testing.assert_allclose(rotations, np.broadcast_to(np.eye(2), rotations.shape),
                                   rtol=0, atol=1e-14)
        # 49 edges, weight*rest_length²=cell_area=6, extension²=.04.
        self.assertAlmostEqual(graph_arap_energy(displacement, (2., 3.), rotations),
                               49*6*.04, places=12)

    def test_local_rotation_minimizes_full_endpoint_energy_without_reflections(self):
        displacement = np.random.default_rng(19).normal(size=(5, 6, 2))
        rotations = graph_rotations(displacement, (2., 3.))
        identity = np.broadcast_to(np.eye(2), rotations.shape)
        self.assertLess(graph_arap_energy(displacement, (2., 3.), rotations),
                        graph_arap_energy(displacement, (2., 3.), identity))
        np.testing.assert_allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-14)
        # Local optimality against independent finite perturbations at one node.
        angle = .01
        for sign in [-1, 1]:
            spin = np.array([[np.cos(angle), -sign*np.sin(angle)],
                             [sign*np.sin(angle), np.cos(angle)]])
            changed = rotations.copy()
            changed[2, 3] = changed[2, 3] @ spin
            self.assertGreater(graph_arap_energy(displacement, (2., 3.), changed),
                               graph_arap_energy(displacement, (2., 3.), rotations))


class PortableARAPTests(unittest.TestCase):
    def make(self, delta=(0., 0.), **kwargs):
        source = np.array([[20., 16.], [20., 20.], [24., 16.], [24., 20.]])
        stride = kwargs.pop('stride', 1)
        return build_problem((41, 41), source, source+delta, [[8., 18.]],
                             crop=[0, 0, 41, 41], target_weights=1000., stride=stride, **kwargs)

    def test_identity_is_exact_and_reports_convergence_without_quality_approval(self):
        result = solve_problem(self.make())
        np.testing.assert_array_equal(result['source_displacement'], 0.)
        self.assertEqual(result['geometry']['determinant_min'], 1.)
        self.assertTrue(result['report']['converged'])
        self.assertEqual(result['report']['iterations'], 1)
        self.assertFalse(result['report']['global_optimality_claimed'])
        self.assertFalse(result['aesthetic_approval'])

    def test_local_global_objective_decreases_from_the_original_membrane_solution(self):
        problem = self.make(delta=(.8, -.2))
        baseline = solve_membrane(problem)
        result = solve_problem(problem)
        z = baseline['solution']
        error = problem['target_operator'] @ z-problem['target_deltas']
        expected_initial = (np.sum(problem['target_weights'][:, None]*error**2)
            + .05*4*np.sum((problem['laplacian'] @ z)**2)
            + .001*graph_arap_energy(baseline['source_displacement'], (2., 2.)))
        report = result['report']
        self.assertAlmostEqual(report['initial_graph_objective']['total'], expected_initial, places=12)
        self.assertEqual(report['factorization_count'], 1)
        self.assertLessEqual(report['iterations'], 8)
        self.assertLess(result['target_max_error_px'], .001)
        self.assertLess(report['final_graph_objective']['total'], report['initial_graph_objective']['total'])
        for iteration in report['energy_history']:
            self.assertLessEqual(iteration['after_global']['total'], iteration['before']['total']+1e-12)
            self.assertLessEqual(iteration['after_local']['total'], iteration['after_global']['total']+1e-12)
            self.assertLess(iteration['global_linear_backward_error'], 1e-10)

    def test_fixed_pixel_support_and_zero_collar_remain_exact_while_rotations_are_free(self):
        problem = self.make(delta=(.8, -.2))
        result = solve_problem(problem)
        yy, xx = np.meshgrid(np.linspace(17, 19, 9), np.linspace(7, 9, 9))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        np.testing.assert_array_equal(evaluate_motion(problem, result['solution'], points), 0.)
        field = result['source_displacement']
        np.testing.assert_array_equal(field[[0, 1, -2, -1]], 0.)
        np.testing.assert_array_equal(field[:, [0, 1, -2, -1]], 0.)
        rotations = result['rotations'].reshape(-1, 2, 2)[problem['locked_nodes']]
        self.assertGreater(float(abs(rotations-np.eye(2)).max()), 1e-5)

    def test_iteration_limit_is_not_reported_as_nonlinear_convergence(self):
        result = solve_problem(self.make(delta=(.8, -.2)), max_iterations=1)
        report = result['report']
        self.assertFalse(report['converged'])
        self.assertEqual(report['status'], 'ARAPIterationLimit')
        self.assertGreater(report['energy_history'][-1]['max_node_step_px'], report['step_tolerance_px'])
        self.assertFalse(result['aesthetic_approval'])

    def test_unsupported_constraints_and_excess_iterations_are_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity reduction'):
            solve_problem(self.make(delta=(.8, -.2), stride=4))
        with self.assertRaisesRegex(ValueError, 'no background line equalities'):
            solve_problem(self.make(delta=(.8, -.2), lines=[
                {'start': [4., 14.], 'end': [26., 36.], 'samples': 2}]))
        with self.assertRaisesRegex(ValueError, '1 to 8'):
            solve_problem(self.make(), max_iterations=9)


if __name__ == '__main__':
    unittest.main()
