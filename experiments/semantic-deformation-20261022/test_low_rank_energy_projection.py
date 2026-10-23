import unittest
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from low_rank_energy_projection import project_energy_balls


class LowRankEnergyProjectionTests(unittest.TestCase):
    def test_correction_uses_original_energy_metric_and_preserves_unseen_mode(self):
        # a=(1,1,0), H=diag(4,1,9), base=(2,0,7).
        # H^-1 a=(1/4,1,0), so the nearest point on a*x=1 is
        # base - (1/(5/4)) H^-1 a = (1.8,-.8,7).
        H=csc_matrix(np.diag([4.,1.,9.]))
        base=np.array([2.,0.,7.])
        out=project_energy_balls(H,base,[{'operator':csr_matrix([[1.,1.,0.]]),'cap':1.}])
        np.testing.assert_allclose(out['solution'],[1.8,-.8,7.],rtol=0.,atol=2e-7)
        np.testing.assert_array_equal(base,[2.,0.,7.])

    def test_preserved_total_is_not_spent_to_meet_local_budget(self):
        out=project_energy_balls(csc_matrix(np.eye(2)),np.array([2.,0.]),
            [{'operator':csr_matrix([[1.,0.]]),'cap':1.}],preserve=csr_matrix([[1.,1.]]))
        np.testing.assert_allclose(out['solution'],[1.,1.],rtol=0.,atol=2e-7)
        self.assertAlmostEqual(float(sum(out['solution'])),2.,places=12)

    def test_redundant_constraints_do_not_create_fictitious_degrees_of_freedom(self):
        # Both balls say |x|<=1, despite duplicated/scaled rows.
        out=project_energy_balls(csc_matrix(np.eye(2)),np.array([2.,5.]),[
            {'operator':csr_matrix([[1.,0.],[1.,0.]]),'cap':2.},
            {'operator':csr_matrix([[3.,0.]]),'cap':9.}])
        np.testing.assert_allclose(out['solution'],[1.,5.],rtol=0.,atol=2e-7)
        self.assertEqual(out['report']['constraint_space_rank'],1)

    def test_equivalent_constraint_units_do_not_delete_preserved_measurements(self):
        # Scaling an entire equation or an entire energy ball changes units,
        # not its feasible set. Both small representations still imply (1,1).
        for ball_scale,equality_scale in [(1.,1e-10),(1e-10,1.),(1e10,1e10)]:
            out=project_energy_balls(csc_matrix(np.eye(2)),np.array([2.,0.]),
                [{'operator':csr_matrix([[ball_scale,0.]]),'cap':ball_scale**2}],
                preserve=csr_matrix([[equality_scale,equality_scale]]))
            np.testing.assert_allclose(out['solution'],[1.,1.],rtol=0.,atol=2e-7)


if __name__=='__main__':unittest.main()
