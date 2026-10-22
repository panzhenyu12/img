"""A moving interior minimum must not escape the original whole-interval cut."""
from types import SimpleNamespace
import unittest

import clarabel
import numpy as np
from scipy.sparse import csr_matrix, eye

from joint_jaw_boundary_field import GridCurve


def fixture():
    problem = {'grid': {'origin': [0., 0.], 'spacing': [1., 1.], 'width': 2, 'height': 2},
               'mapping': np.arange(4), 'hessian': eye(4, format='csc')}
    curve = GridCurve([[0., 0.], [1., 1.]], problem, [1., 0.], [0., 1.])
    audit = SimpleNamespace(size=4, specifications=[], jaws=[curve],
        domains=[(np.array([[0., curve.length, 1.]]), np.empty(0))],
        old=SimpleNamespace(arcs=[{'id': 'jaw'}]))
    return curve, audit


class IntervalExchangeTests(unittest.TestCase):
    def test_straight_or_unknown_source_turn_domain_has_no_certificate(self):
        from source_interval_exchange import SourceIntervalExchange
        curve, audit = fixture()
        audit.specifications = [('straight-interface', curve, np.empty((0, 3)), .2, .5)]
        result = SourceIntervalExchange(audit, []).discover(np.zeros(4))
        self.assertEqual(result['new_certificates'], 0)

    def test_one_certificate_covers_the_whole_piece_when_the_minimum_moves(self):
        from source_interval_exchange import SourceIntervalExchange
        curve, audit = fixture(); exchange = SourceIntervalExchange(audit, [])
        # Both endpoints .15 are positive, interior quadratic reaches -.125.
        z = np.array([.15, -.4, -.4, .15])
        self.assertEqual(exchange.discover(z)['new_certificates'], 1)
        # Different coefficients move the violating minimum, same interval.
        self.assertEqual(exchange.discover(np.array([.2, -.5, -.4, .1]))['new_certificates'], 0)
        self.assertEqual(exchange.keys, [('sign', 'jaw', 0)])
        base = {'P': 2*eye(4, format='csc'), 'q': -2*z,
                'A': csr_matrix([[1., 0., 0., 0.], [0., 0., 0., 1.]]),
                'b': np.array([.15, .15]), 'cones': [clarabel.ZeroConeT(2)]}
        qp = exchange.augment(base)
        settings = clarabel.DefaultSettings(); settings.verbose = False
        settings.tol_feas = 1e-9; settings.tol_gap_abs = 1e-10; settings.tol_gap_rel = 1e-10
        solved = clarabel.DefaultSolver(qp['P'], qp['q'], qp['A'], qp['b'], qp['cones'], settings).solve()
        self.assertEqual(str(solved.status), 'Solved')
        checked, _ = curve.continuous_sign_check(np.asarray(solved.x[:4]), None, audit.domains[0][0])
        self.assertEqual(checked['violations'], 0)
        np.testing.assert_allclose(np.asarray(solved.x)[[0, 3]], [.15, .15], atol=1e-10, rtol=0.)

    def test_existing_auxiliary_columns_and_original_qp_are_kept_verbatim(self):
        from source_interval_exchange import SourceIntervalExchange
        _, audit = fixture(); exchange = SourceIntervalExchange(audit, [])
        exchange.discover(np.array([.15, -.4, -.4, .15]))
        # Fifth variable stands for an existing certificate, not a source node.
        base = {'P': eye(5, format='csc'), 'q': np.arange(5.),
                'A': csr_matrix([[1., 2., 0., 0., 3.]]), 'b': np.array([7.]),
                'cones': [clarabel.ZeroConeT(1)]}
        qp = exchange.augment(base)
        self.assertEqual(qp['P'].shape, (11, 11))
        np.testing.assert_array_equal(qp['A'][:1, :5].toarray(), base['A'].toarray())
        np.testing.assert_array_equal(qp['P'][:5, :5].toarray(), base['P'].toarray())
        np.testing.assert_array_equal(qp['q'][:5], base['q'])
        self.assertEqual(qp['A'][:1, 5:].nnz, 0)
        self.assertEqual(qp['A'][1:, 4].nnz, 0)
        self.assertEqual(qp['P'][5:].nnz, 0)
        self.assertEqual(float(abs(qp['q'][5:]).sum()), 0.)

    def test_frozen_piece_is_not_duplicated_and_unknown_keys_are_rejected(self):
        from source_interval_exchange import SourceIntervalExchange
        _, audit = fixture(); key = ('sign', 'jaw', 0)
        exchange = SourceIntervalExchange(audit, [key])
        self.assertEqual(exchange.discover(np.array([.15, -.4, -.4, .15]))['new_certificates'], 0)
        with self.assertRaisesRegex(ValueError, 'original'):
            SourceIntervalExchange(audit, [('sign', 'jaw', 1)])

    def test_original_hard_zero_uses_linear_factor_without_degenerate_psd_auxiliaries(self):
        from source_interval_exchange import SourceIntervalExchange
        curve, audit = fixture(); audit.domains[0] = (audit.domains[0][0], np.array([0.]))
        exchange = SourceIntervalExchange(audit, [])
        # q(t)=t*(2*t-1), with the original endpoint constrained to zero.
        result = exchange.discover(np.array([0., -.5, -.5, 1.]))
        self.assertEqual(result['new_certificates'], 0)
        self.assertEqual(result['new_factorized_pieces'], 1)
        base = {'P': eye(4, format='csc'), 'q': np.zeros(4),
                'A': csr_matrix([[1., 0., 0., 0.]]), 'b': np.zeros(1), 'cones': [clarabel.ZeroConeT(1)]}
        qp = exchange.augment(base)
        self.assertEqual(qp['P'].shape, (4, 4))
        self.assertEqual(qp['A'].shape[0], 3)
        self.assertLess((qp['b']-qp['A']@np.array([0., -.5, -.5, 1.])).min(), 0.)


if __name__ == '__main__': unittest.main()
