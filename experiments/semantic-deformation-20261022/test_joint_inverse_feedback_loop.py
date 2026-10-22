"""Controller regressions: global failures and immutable source constraints."""
import unittest
from types import SimpleNamespace

import clarabel
import numpy as np
from scipy.sparse import csr_matrix, eye


def actual(violation=0., *, fixed=True, curves=True):
    return {'geometry': {'symmetric_eigenvalue_min': .52-violation,
        'symmetric_eigenvalue_max': 1.2, 'skew_absolute_max': .1,
        'jacobian_determinant_min': .5, 'boundary_max_displacement': 0.,
        'passed_strict': violation == 0.},
        'fixed': {'changed': 0 if fixed else 1, 'inverse_max_px': 0. if fixed else .001},
        'sampled_curves': {'sampled_passed': curves},
        'curve_response': {'passed': curves}}


class FeedbackLoopTests(unittest.TestCase):
    def test_complete_solver_state_requires_finite_slack_and_dual(self):
        from probe_joint_inverse_feedback_loop import finite_solver_state
        good=SimpleNamespace(x=[1.,2.],s=[1.,2.,3.],z=[1.,2.,3.])
        self.assertTrue(finite_solver_state(good,2,3))
        for key,value in [('s',[1.,np.nan,3.]),('z',[1.,2.,np.inf]),('s',[1.,2.])]:
            bad=SimpleNamespace(**vars(good));setattr(bad,key,value)
            self.assertFalse(finite_solver_state(bad,2,3))

    def test_crossed_source_cells_are_relocated_and_prediction_error_is_measured(self):
        from test_inverse_field_linearization import fixture
        from probe_joint_inverse_feedback import final_corner_linearization,inverse_field
        from probe_joint_inverse_feedback_loop import prediction_diagnostic
        grid,mapping,z,across=fixture();z*=.1
        down=np.array([-across[1],across[0]])
        linear=final_corner_linearization(grid,mapping,z,across,down,
            np.array([[corner,8,8] for corner in range(4)]))
        field,_=inverse_field(-z,mapping,grid,across)
        report=prediction_diagnostic(linear,-z,field,grid,mapping,across)
        self.assertGreater(report['source_cell_changes'],0)
        self.assertGreater(report['inverse_scalar_prediction_max_error_px'],1e-7)
        self.assertFalse(report['cells_frozen_for_acceptance'])
        self.assertLess(report['fresh_sample_closure_max_source_px'],1e-7)

    def test_source_failure_never_advances_even_when_actual_field_improves(self):
        from joint_inverse_feedback_controller import FeedbackState
        state=FeedbackState(np.zeros(2),actual(.1),[[0,0,0]])
        row=state.observe('Solved',np.array([.1,0.]),{'accepted':False},actual(.01),[],1)
        self.assertEqual(row['action'],'exchange_source')
        np.testing.assert_array_equal(state.anchor,np.zeros(2))
        row=state.observe('Solved',np.array([.1,0.]),{'accepted':False},actual(.001),[],0)
        self.assertEqual(row['action'],'stop')
        self.assertEqual(row['reason'],'source_failure_without_new_exact_constraint')
        np.testing.assert_array_equal(state.anchor,np.zeros(2))

    def test_new_global_failure_is_exchanged_and_nonimprovement_shrinks_trust(self):
        from joint_inverse_feedback_controller import FeedbackState
        state=FeedbackState(np.zeros(2),actual(.1),[[0,0,0]])
        row=state.observe('Solved',np.array([.1,0.]),{'accepted':True},actual(.11),[[3,10,9]],0)
        self.assertEqual(row['action'],'shrink')
        self.assertEqual(state.radius,.125)
        self.assertIn((3,10,9),state.corners)
        np.testing.assert_array_equal(state.anchor,np.zeros(2))
        row=state.observe('AlmostSolved',np.array([.1,0.]),{'accepted':True},actual(.01),[],0)
        self.assertEqual(row['action'],'advance')
        self.assertFalse(state.accepted)
        np.testing.assert_array_equal(state.anchor,[.1,0.])
        self.assertEqual(state.radius,.125)

    def test_geometry_success_cannot_hide_curve_fixed_or_solver_failure(self):
        from joint_inverse_feedback_controller import FeedbackState
        for status,evidence,reason in [
            ('Solved',actual(curves=False),'curve_failure_without_geometry_direction'),
            ('Solved',actual(fixed=False),'fixed_identity_failure'),
            ('AlmostSolved',actual(),'solver_not_solved_without_geometry_direction'),
            ('MaxTime',actual(),'terminal_solver_status')]:
            state=FeedbackState(np.zeros(1),actual(.1),[[0,0,0]])
            row=state.observe(status,np.array([.1]),{'accepted':True},evidence,[],0)
            self.assertEqual(row['reason'],reason)
            self.assertFalse(state.accepted)
            np.testing.assert_array_equal(state.anchor,[0.])
        state=FeedbackState(np.zeros(1),actual(.1),[[0,0,0]])
        row=state.observe('Solved',np.array([.1]),{'accepted':True},actual(),[],0)
        self.assertEqual(row['action'],'accept')
        self.assertTrue(state.accepted)

    def test_failed_trials_count_and_both_step_and_frozen_boxes_hold(self):
        from joint_inverse_feedback_controller import FeedbackState
        state=FeedbackState(np.zeros(1),actual(.1),[[0,0,0]])
        self.assertEqual(state.observe('Solved',np.array([.2]),{'accepted':True},actual(.08),[],0)['action'],'advance')
        # Inside the per-step .25 box but outside the frozen total .25 box.
        row=state.observe('Solved',np.array([.3]),{'accepted':True},actual(.01),[],0)
        self.assertEqual(row['reason'],'trust_box_failure')
        self.assertEqual(state.attempts,2)
        for vector in [np.array([np.nan]),np.array([np.inf])]:
            state=FeedbackState(np.zeros(1),actual(.1),[[0,0,0]])
            row=state.observe('Solved',vector,None,None,[],0)
            self.assertEqual(row['reason'],'nonfinite_trial')
            self.assertEqual(state.attempts,1)

    def test_exact_source_cut_ledger_does_not_promote_frozen_duplicates(self):
        from joint_inverse_feedback_controller import ExactCutLedger
        base={'A':csr_matrix([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,0.,0.]]),
              'b':np.array([1.,1.,2.,3.]),
              'cones':[clarabel.NonnegativeConeT(2),clarabel.SecondOrderConeT(2)]}
        ledger=ExactCutLedger(base)
        duplicate={'A':base['A'].copy(),'b':base['b'].copy(),'cones':base['cones']}
        self.assertEqual(ledger.add(duplicate)['new_cone_blocks'],0)
        fresh={'A':csr_matrix([[1.,0.,0.],[0.,1.,0.]]),'b':np.array([1.,4.]),
               'cones':[clarabel.NonnegativeConeT(2)]}
        self.assertEqual(ledger.add(fresh)['new_cone_blocks'],1)
        self.assertEqual(ledger.add(fresh)['new_cone_blocks'],0)
        self.assertEqual(ledger.blocks[0]['A'].shape,(1,3))

    def test_reassembly_keeps_original_and_exact_cuts_but_replaces_tangent(self):
        from joint_inverse_feedback_controller import assemble_trial
        base={'A':csr_matrix([[1.,2.,3.]]),'b':np.array([4.]),'cones':[clarabel.ZeroConeT(1)]}
        exact={'A':csr_matrix([[5.,6.,0.]]),'b':np.array([7.]),'cones':[clarabel.NonnegativeConeT(1)]}
        first={'A':csr_matrix([[8.,9.,0.]]),'b':np.array([10.]),'cones':[clarabel.NonnegativeConeT(1)]}
        second={'A':csr_matrix([[11.,12.,0.]]),'b':np.array([13.]),'cones':[clarabel.NonnegativeConeT(1)]}
        one=assemble_trial(base,[exact],first,np.zeros(2),np.zeros(2),.25)
        two=assemble_trial(base,[exact],second,np.zeros(2),np.array([.1,0.]),.125)
        np.testing.assert_array_equal(one['A'][:2].toarray(),two['A'][:2].toarray())
        np.testing.assert_array_equal(two['A'][2].toarray(),second['A'].toarray())
        self.assertFalse(any(np.array_equal(row,[8.,9.,0.]) for row in two['A'].toarray()))
        self.assertEqual(two['A'][3:,2:].nnz,0)
        np.testing.assert_allclose(two['b'][-4:], [.225,.125,.025,.125],atol=1e-16,rtol=0.)


if __name__=='__main__':unittest.main()
