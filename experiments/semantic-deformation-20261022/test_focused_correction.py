"""Local constraint coverage and unchanged-input checks, without photographs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, eye, save_npz

from focused_correction import focused_source_corners, solve_focused_reduced
from adapt import sha


class FocusedCorrectionTests(unittest.TestCase):
    def test_failed_pixel_footprint_includes_cells_between_its_four_corners(self):
        grid = {'origin': [10, 20], 'spacing': [2., 2.], 'width': 6, 'height': 6}
        field = np.zeros((11, 11, 2), dtype=np.float32)
        field[4:6, 5, 0] = 3.
        source = np.zeros((6, 6, 2))
        previous = [0, 25, 50, 75]
        active, report = focused_source_corners(field, source, grid, previous)
        self.assertGreater(report['failed_inverse_cells'], 0)
        # The x=4..5 destination cell maps from x=4 to x=8: the complete
        # source enclosure includes source cell x=3, not only endpoint cells.
        self.assertIn(2*5+3, report['inverse_footprint_cell_ids'])
        for old in previous:
            self.assertIn(old, active)
        for corner in active:
            self.assertTrue(all(k*25+corner % 25 in active for k in range(4)))
        self.assertEqual(report['source_cell_halo'], 1)

    def test_source_low_eigenvalue_cells_are_added_without_final_pixel_failure(self):
        grid = {'origin': [0, 0], 'spacing': [2., 2.], 'width': 5, 'height': 5}
        field = np.zeros((9, 9, 2), dtype=np.float32)
        source = np.zeros((5, 5, 2)); source[2, 2, 0] = 1.2
        active, report = focused_source_corners(field, source, grid, [])
        self.assertEqual(report['failed_inverse_cells'], 0)
        self.assertGreater(report['low_source_cells'], 0)
        self.assertGreater(len(active), 0)

    def fixture(self, folder):
        previous, inp = folder/'previous', folder/'input'
        previous.mkdir(); (inp/'proposal').mkdir(parents=True)
        grid = {'origin': [0, 0], 'spacing': [2., 2.], 'width': 3, 'height': 3}
        mapping = np.full(9, -1, dtype=int); mapping[4] = 0
        h, r, rhs = eye(1, format='csc'), eye(1, format='csc'), np.array([[3., 1.]])
        (inp/'contract.json').write_text(json.dumps({'grid': grid, 'crop': [0, 0, 5, 5]}))
        np.savez(inp/'data.npz', node_to_variable=mapping)
        save_npz(inp/'proposal/hessian.npz', h); np.save(inp/'proposal/rhs.npy', rhs)
        save_npz(previous/'reduction.npz', r)
        np.save(previous/'solution.npy', np.zeros((1, 2)))
        np.save(previous/'coefficients.npy', np.zeros((1, 2)))
        np.save(previous/'source-forward.npy', np.zeros((3, 3, 2)))
        np.save(previous/'inverse-field.npy', np.zeros((5, 5, 2), dtype=np.float32))
        (previous/'solver-evidence.json').write_text(json.dumps({'rounds': [
            {'all_active_corner_ids': [0, 4, 8, 12]}]}))
        files = [inp/'contract.json', inp/'data.npz', inp/'proposal/hessian.npz', inp/'proposal/rhs.npy']
        (previous/'protocol.json').write_text(json.dumps({'input': str(inp),
            'inputs_sha256': {str(p): sha(p) for p in files}}))
        return previous, h, rhs, r, grid, mapping

    def test_exactly_one_bounded_solve_preserves_inputs_and_time_limit_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, h, rhs, r, grid, mapping = self.fixture(Path(folder))
            report = {'status': 'MaxTime', 'primal_candidate_available': True,
                      'active_cones_pass': True, 'solver_time_limit_seconds': 12.}
            with patch('focused_correction.validate_input', return_value={'verified_input_hashes': {}}), \
                    patch('focused_correction.focused_source_corners', return_value=(
                        [0, 1, 4, 5, 8, 9, 12, 13], {'selected_source_cells': 2, 'new_source_cells': 1})), \
                    patch('focused_correction._solve_round', return_value=(np.zeros((1, 2)), report)) as solve:
                result = solve_focused_reduced(h, rhs, r, grid, mapping, None, previous)
            self.assertEqual(solve.call_count, 1)
            np.testing.assert_array_equal(solve.call_args.args[1], rhs)
            np.testing.assert_array_equal(rhs, [[3., 1.]])
            self.assertTrue(result['report']['source_proxy_pass'])
            self.assertFalse(result['report']['source_convex_gate_pass'])
            self.assertTrue(result['report']['diagnostic_only'])
            self.assertTrue(result['report']['final_actual_f32_geometry_and_RGB_pending'])

    def test_passing_field_with_old_active_cells_does_not_repeat_same_solve(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, h, rhs, r, grid, mapping = self.fixture(Path(folder))
            with patch('focused_correction.validate_input', return_value={'verified_input_hashes': {}}), \
                    patch('focused_correction._solve_round') as solve:
                with self.assertRaisesRegex(ValueError, 'no new source constraints'):
                    solve_focused_reduced(h, rhs, r, grid, mapping, None, previous)
                solve.assert_not_called()

    def test_remaining_failure_already_covered_does_not_repeat_same_solve(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, h, rhs, r, grid, mapping = self.fixture(Path(folder))
            source = np.zeros((3, 3, 2)); source[1, 1, 0] = .95
            np.save(previous/'source-forward.npy', source)
            np.save(previous/'solution.npy', [[.95, 0.]])
            np.save(previous/'coefficients.npy', [[.95, 0.]])
            (previous/'solver-evidence.json').write_text(json.dumps({'rounds': [
                {'all_active_corner_ids': list(range(16))}]}))
            with patch('focused_correction.validate_input', return_value={'verified_input_hashes': {}}), \
                    patch('focused_correction._solve_round') as solve:
                with self.assertRaisesRegex(ValueError, 'no new source constraints'):
                    solve_focused_reduced(h, rhs, r, grid, mapping, None, previous)
                solve.assert_not_called()

    def test_changed_reduction_is_rejected_before_solver(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, h, rhs, r, grid, mapping = self.fixture(Path(folder))
            with patch('focused_correction.validate_input', return_value={'verified_input_hashes': {}}), \
                    patch('focused_correction._solve_round') as solve:
                with self.assertRaisesRegex(ValueError, 'reduction'):
                    solve_focused_reduced(h, rhs, r*.9, grid, mapping, None, previous)
                solve.assert_not_called()

    def test_changed_original_rhs_hash_is_rejected_before_solver(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, h, rhs, r, grid, mapping = self.fixture(Path(folder))
            inp = Path(json.loads((previous/'protocol.json').read_text())['input'])
            np.save(inp/'proposal/rhs.npy', rhs*.5)
            with patch('focused_correction.validate_input', return_value={'verified_input_hashes': {}}), \
                    patch('focused_correction._solve_round') as solve:
                with self.assertRaisesRegex(ValueError, 'hash'):
                    solve_focused_reduced(h, rhs, r, grid, mapping, None, previous)
                solve.assert_not_called()

    def test_infeasibility_certificate_keeps_previous_field_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, h, rhs, r, grid, mapping = self.fixture(Path(folder))
            report = {'status': 'PrimalInfeasible', 'primal_candidate_available': False}
            with patch('focused_correction.validate_input', return_value={'verified_input_hashes': {}}), \
                    patch('focused_correction.focused_source_corners', return_value=(
                        [0, 1, 4, 5, 8, 9, 12, 13], {'selected_source_cells': 2, 'new_source_cells': 1})), \
                    patch('focused_correction._solve_round', return_value=(None, report)):
                result = solve_focused_reduced(h, rhs, r, grid, mapping, None, previous)
            np.testing.assert_array_equal(result['solution'], np.zeros((1, 2)))
            self.assertFalse(result['report']['source_convex_gate_pass'])
            self.assertEqual(result['report']['selected_stage'], 'previous_case_no_primal_candidate')


if __name__ == '__main__':
    unittest.main()
