"""Independent geometric checks for the joint source-field diagnostic."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.sparse import eye

from joint_jaw_boundary_field import GridCurve, polynomial_minima, _independent_equalities, solve_joint_field, source_sign_regions, all_corner_inverse_check, hard_zero_sign_piece_rows, interface_chord_conditions, interface_displacement_bending, jaw_start_displacement_regularizer, curve_window_motion


def grid_problem():
    return {'grid': {'origin': [0., 0.], 'spacing': [1., 1.], 'width': 11, 'height': 11},
            'mapping': np.arange(121), 'hessian': eye(121, format='csc')}


class JointSourceCurveTests(unittest.TestCase):
    def test_resume_reuses_exact_first_qp_and_debits_original_round_and_time(self):
        import clarabel
        from test_jaw_curve_qp import curved
        problem={'grid':{'origin':[-50.,-30.],'spacing':[10.,10.],'width':11,'height':11},
                 'mapping':np.arange(121),'hessian':eye(121,format='csc'),'target_weights':np.zeros(1)}
        neck=[{'source_points':[[0.,30.],[0.,40.]],'uncertainty_radius_px':1.}]
        def solver_with(status,fold):
            def solver(p,q,a,b,cones,settings):
                x=np.zeros(p.shape[0])
                if fold:x[6*11+6]=-30.
                answer=SimpleNamespace(status=status,x=x,s=np.zeros(a.shape[0]),z=np.zeros(a.shape[0]),
                    iterations=1,r_prim=1e-11,r_dual=0.,obj_val=0.,obj_val_dual=0.)
                return SimpleNamespace(solve=lambda:answer)
            return solver
        with patch('clarabel.DefaultSolver',side_effect=solver_with(clarabel.SolverStatus.AlmostSolved,True)):
            initial=solve_joint_field(problem,curved(),0.,[],neck,max_rounds=1,allow_strict_polish=False)
        reuse={'source_report':'frozen-test-first-round','subproblem_sha256':initial['diagnostic_qp']['sha256'],
               'state':initial['diagnostic_solver_state'],'row':initial['report']['refinement'][0],
               'prior_elapsed_seconds':initial['report']['total_source_solve_seconds']}
        with patch('clarabel.DefaultSolver',side_effect=solver_with(clarabel.SolverStatus.Solved,False)) as called:
            result=solve_joint_field(problem,curved(),0.,[],neck,allow_strict_polish=False,reuse_first_round=reuse)
            self.assertEqual(called.call_count,1,'the unchanged failed first problem must not be solved again')
        report=result['report'];self.assertTrue(report['source_continuous_accepted'])
        self.assertEqual(len(report['refinement']),2)
        self.assertTrue(report['refinement'][0]['solver_state_reused_without_resolve'])
        self.assertEqual(report['prior_source_solve_seconds'],reuse['prior_elapsed_seconds'])
        self.assertEqual(report['total_source_solve_seconds'],report['prior_source_solve_seconds']+report['current_source_solve_seconds'])
        with patch('clarabel.DefaultSolver') as called:
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                solve_joint_field(problem,curved(),0.,[],neck,reuse_first_round={**reuse,'subproblem_sha256':'wrong'})
            self.assertEqual(called.call_count,0)
        with patch('clarabel.DefaultSolver') as called:
            stopped=solve_joint_field(problem,curved(),0.,[],neck,reuse_first_round={**reuse,'prior_elapsed_seconds':60.})
            self.assertEqual(called.call_count,0)
            self.assertFalse(stopped['report']['source_continuous_accepted'])
            self.assertEqual(len(stopped['report']['refinement']),0)

    def test_intermediate_almost_solved_can_only_supply_new_valid_cuts(self):
        import clarabel
        from test_jaw_curve_qp import curved
        problem={'grid':{'origin':[-50.,-30.],'spacing':[10.,10.],'width':11,'height':11},
                 'mapping':np.arange(121),'hessian':eye(121,format='csc'),'target_weights':np.zeros(1)}
        calls=[]
        def solver(p,q,a,b,cones,settings):
            x=np.zeros(p.shape[0]);status=clarabel.SolverStatus.Solved
            if not calls:x[6*11+6]=-30.;status=clarabel.SolverStatus.AlmostSolved
            calls.append(status)
            answer=SimpleNamespace(status=status,x=x,s=np.zeros(a.shape[0]),z=np.zeros(a.shape[0]),
                iterations=1,r_prim=1e-11,r_dual=0.,obj_val=0.,obj_val_dual=0.)
            return SimpleNamespace(solve=lambda:answer)
        with patch('clarabel.DefaultSolver',side_effect=solver):
            result=solve_joint_field(problem,curved(),0.,[],[
                {'source_points':[[0.,30.],[0.,40.]],'uncertainty_radius_px':1.}],allow_strict_polish=False)
        self.assertEqual(calls,[clarabel.SolverStatus.AlmostSolved,clarabel.SolverStatus.Solved])
        self.assertGreater(result['report']['refinement'][0]['new_geometry_corners'],0)
        self.assertTrue(result['report']['refinement'][0]['intermediate_almost_solved_separation_only'])
        self.assertTrue(result['report']['source_continuous_accepted'])

    def test_almost_solved_without_new_cuts_and_failed_statuses_cannot_loop_or_accept(self):
        import clarabel
        from test_jaw_curve_qp import curved
        problem={'grid':{'origin':[-50.,-30.],'spacing':[10.,10.],'width':11,'height':11},
                 'mapping':np.arange(121),'hessian':eye(121,format='csc'),'target_weights':np.zeros(1)}
        for status,fold in [(clarabel.SolverStatus.AlmostSolved,False),
                            (clarabel.SolverStatus.MaxTime,True),(clarabel.SolverStatus.PrimalInfeasible,True),
                            ('Unknown',True)]:
            calls=[]
            def solver(p,q,a,b,cones,settings):
                calls.append(status);x=np.zeros(p.shape[0])
                if fold:x[6*11+6]=-30.
                answer=SimpleNamespace(status=status,x=x,s=np.zeros(a.shape[0]),z=np.zeros(a.shape[0]),
                    iterations=1,r_prim=0.,r_dual=0.,obj_val=0.,obj_val_dual=0.)
                return SimpleNamespace(solve=lambda:answer)
            with self.subTest(status=status),patch('clarabel.DefaultSolver',side_effect=solver):
                result=solve_joint_field(problem,curved(),0.,[],[
                    {'source_points':[[0.,30.],[0.,40.]],'uncertainty_radius_px':1.}],allow_strict_polish=False)
                self.assertEqual(len(calls),1)
                self.assertFalse(result['report']['source_continuous_accepted'])

    def test_jaw_start_energy_exact_linear_and_degree_seven_integrals(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[1.,1.],[8.,8.]],problem,[1.,0.],[0.,1.]);knots=curve.knots.copy()
        length=5.3;regularizer=jaw_start_displacement_regularizer(curve,length)
        for constant in [0.,7.]:
            np.testing.assert_allclose(regularizer['operator']@np.full(121,constant),0.,atol=2e-15)
        slope=.7;linear=(slope*((xx-1.)+(yy-1.))/np.sqrt(2.)).ravel()
        curvature=.12;quadratic=(2.*curvature*(xx-1.)*(yy-1.)).ravel()
        for z,expected in [(linear,.5*slope**2*length**3),
                           (quadratic,5./21*curvature**2*length**5)]:
            measured=float(regularizer['weights']@((regularizer['operator']@z)**2))
            self.assertAlmostEqual(measured,expected,places=10)
            self.assertAlmostEqual(float(z@(regularizer['hessian']@z)),expected,places=10)
        np.testing.assert_array_equal(curve.knots,knots)
        self.assertEqual(regularizer['report']['window_end_px'],length)

    def test_jaw_start_energy_catches_growth_after_zero_endpoint_tangent(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[1.,1.],[8.,8.]],problem,[1.,0.],[0.,1.])
        # u(s)=s^2: both displacement and tangent are zero at the start, yet
        # the first half-pixel already gains slope 1. A point gate misses it.
        z=(2.*(xx-1.)*(yy-1.)).ravel();regularizer=jaw_start_displacement_regularizer(curve,6.)
        self.assertEqual(float((curve.op([0.])@z)[0]),0.)
        self.assertAlmostEqual(float((curve.op([.5])@z)[0]),.25,places=12)
        np.testing.assert_allclose(regularizer['operator']@z,2.*regularizer['positions'],atol=1e-12)
        self.assertGreater(float(regularizer['weights']@((regularizer['operator']@z)**2)),1000.)

    def test_jaw_start_motion_reports_interior_extremum_not_just_window_endpoint(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[1.,1.],[8.,8.]],problem,[1.,0.],[0.,1.])
        z=(np.sqrt(2.)*((xx-1.)+(yy-1.))-2.*(xx-1.)*(yy-1.)).ravel()
        report=curve_window_motion(curve,z,2.)
        self.assertAlmostEqual(report['at_window_end_px'],0.,places=12)
        self.assertAlmostEqual(report['maximum_abs_motion_px'],1.,places=12)
        self.assertAlmostEqual(report['maximum_at_source_arclength_px'],1.,places=12)

    def test_interface_displacement_bending_leaves_source_and_arclength_affine_motion_free(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[5.,1.],[5.,9.]],problem,[1.,0.],[0.,1.])
        regularizer=interface_displacement_bending(curve,.05,12.,2.)
        for z in [np.zeros(121),np.full(121,3.),(2.+.3*(yy-1.)).ravel()]:
            np.testing.assert_allclose(regularizer['operator']@z,0.,atol=2e-15)
        curved=GridCurve([[4.,1.],[5.,5.],[3.,9.]],problem,[1.,0.],[0.,1.])
        bending=interface_displacement_bending(curved,.05,12.,2.)
        self.assertEqual(float(np.sum((bending['operator']@np.zeros(121))**2)),0.)
        np.testing.assert_array_equal(curved.points,[[4.,1.],[5.,5.],[3.,9.]])

    def test_interface_displacement_bending_penalizes_concentration_at_same_endpoint_slopes(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[5.,1.],[5.,5.]],problem,[1.,0.],[0.,1.])
        regularizer=interface_displacement_bending(curve,.05,12.,1.)
        distributed=np.array([0.,0.,1.,3.,6.]);concentrated=np.array([0.,0.,0.,3.,6.])
        np.testing.assert_array_equal(distributed[[0,-1]],concentrated[[0,-1]])
        np.testing.assert_array_equal(np.diff(distributed)[[0,-1]],np.diff(concentrated)[[0,-1]])
        energies=[]
        for values in [distributed,concentrated]:
            z=np.interp(yy,np.arange(1.,6.),values).ravel()
            energies.append(regularizer['weight']*np.sum((regularizer['operator']@z)**2))
        self.assertGreater(energies[1],energies[0]*2.9)

    def test_interface_displacement_bending_catches_single_radius_periodic_blind_mode(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[5.,1.],[5.,9.]],problem,[1.,0.],[0.,1.])
        # Periodic piecewise-linear motion has period 2 source pixels. The
        # radius-2 chord cannot see it even between nodes; the local bending can.
        z=np.cos(np.pi*yy).ravel();centers=np.linspace(2.,6.,71)
        source,rows,_=curve.chord(centers,2.)
        np.testing.assert_allclose(rows@z,0.,atol=2e-14)
        regularizer=interface_displacement_bending(curve,.05,12.,1.)
        self.assertGreater(regularizer['weight']*np.sum((regularizer['operator']@z)**2),100.)

    def test_interface_displacement_bending_physical_scaling_and_endpoint_quadrature(self):
        yy,xx=np.mgrid[:11,:11];energies=[]
        for scale in [1.,3.]:
            problem=grid_problem();problem['grid']['spacing']=[scale,scale]
            curve=GridCurve(scale*np.array([[5.,1.],[5.,9.]]),problem,[1.,0.],[0.,1.])
            regularizer=interface_displacement_bending(curve,.05,12.*scale,2.*scale)
            z=(scale*(yy-1.)**2).ravel()
            energies.append(regularizer['weight']*np.sum((regularizer['operator']@z)**2))
        self.assertAlmostEqual(energies[1]/energies[0],3.**3,places=11)
        # Same convention as jaw: interior-node rectangle quadrature spans
        # L-h, not all L; do not claim exact resolution-invariant integration.
        regularizer=interface_displacement_bending(
            GridCurve([[5.,1.],[5.,9.]],grid_problem(),[1.,0.],[0.,1.]),.05,12.,2.)
        self.assertAlmostEqual(regularizer['report']['quadrature_span_px'],6.)
        self.assertAlmostEqual(energies[0],.05*12.**4*6.*4.)

    def test_interface_band_rejects_concentrated_bending_in_both_turn_directions(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        y=np.array([1.,3.,5.,7.,9.]);source_x=np.array([3.6,3.,2.8,3.,3.6])
        # Same endpoints and same total slope change, but curvature has been
        # moved from the first two source turns into the final turn.
        target_x=np.array([3.6,3.225,2.95,2.775,3.6])
        for direction in [1.,-1.]:
            source=np.column_stack([5.+direction*(source_x-5.),y])
            curve=GridCurve(source,problem,[1.,0.],[0.,1.]);knots=curve.knots.copy()
            z=(direction*np.interp(yy,y,target_x-source_x)).ravel()
            original=curve.source_turn_regions(1.)
            old,_,_=curve.continuous_chord_check(z,original,1.,0.)
            self.assertEqual(old['violations'],0,'old sign-only gate must miss this bend concentration')
            checks=[c.continuous_chord_check(z,regions,radius,rho)[0]
                for _,c,regions,radius,rho in interface_chord_conditions('interface',curve,1.,(.5,1.5))]
            self.assertTrue(any(check['violations'] for check in checks),
                'relative bend band must reject same-turn over-flattening and over-concentration')
            self.assertEqual(len(checks),2)
            self.assertTrue(all(check['violations'] for check in checks))
            np.testing.assert_array_equal(curve.knots,knots)
            np.testing.assert_array_equal(curve.source_turn_regions(1.),original)
            for _,c,regions,radius,rho in interface_chord_conditions('interface',curve,1.,(.5,1.5)):
                unchanged,_,_=c.continuous_chord_check(np.zeros(121),regions,radius,rho)
                self.assertEqual(unchanged['violations'],0)

    def test_interface_band_does_not_assign_unknown_zero_turn_domains(self):
        curve=GridCurve([[2.,1.],[2.5,5.],[3.,9.]],grid_problem(),[1.,0.],[0.,1.])
        self.assertEqual(len(curve.source_turn_regions(1.)),0)
        for _,_,regions,_,_ in interface_chord_conditions('interface',curve,1.,(.5,1.5)):
            self.assertEqual(len(regions),0)

    def test_source_rejects_out_of_range_nonfinite_or_nonvector_arclength(self):
        curve=GridCurve([[1.,1.],[3.,5.]],grid_problem(),[1.,0.],[0.,1.])
        for bad in [[-1.], [curve.length+1.], [np.nan], [np.inf], 0., [[0.]]]:
            with self.subTest(arclength=bad):
                with self.assertRaisesRegex(ValueError,'arclength'):
                    curve.source(bad)
                with self.assertRaisesRegex(ValueError,'arclength'):
                    curve.op(bad)
        np.testing.assert_array_equal(curve.source([0.,curve.length]),curve.points)

    def test_original_parameter_knots_are_preserved_without_clipping(self):
        original=np.array([0.,np.nextafter(np.sqrt(20.),np.inf)])
        curve=GridCurve([[1.,1.],[3.,5.]],grid_problem(),[1.,0.],[0.,1.],source_knots=original)
        np.testing.assert_array_equal(curve.knots,original)
        np.testing.assert_array_equal(curve.source(original),curve.points)
        with self.assertRaisesRegex(ValueError,'arclength'):
            curve.source([np.nextafter(original[-1],np.inf)])
        with self.assertRaisesRegex(ValueError,'disagree'):
            GridCurve([[1.,1.],[3.,5.]],grid_problem(),[1.,0.],[0.,1.],source_knots=[0.,5.])

    def test_legal_continuous_curve_observations_cannot_hide_interior_fold(self):
        problem=grid_problem();z=np.zeros((11,11));z[6,6]=-3.;z=z.ravel()
        curve=GridCurve([[1.5,1.],[1.2,3.],[1.1,5.],[1.2,7.],[1.5,9.]],problem,[1.,0.],[0.,1.])
        regions=curve.source_turn_regions(1.)
        self.assertGreater(len(regions),0)
        chord,_,_=curve.continuous_chord_check(z,regions,1.,.5)
        sign,_=curve.continuous_sign_check(z,1.)
        self.assertEqual(chord['violations'],0)
        self.assertEqual(sign['violations'],0)
        self.assertEqual(float((curve.area_row()@z)[0]),0.)
        checked,bad=all_corner_inverse_check(problem,z,np.array([1.,0.]))
        self.assertFalse(checked['accepted'])
        self.assertFalse(checked['positive_source_determinant'])
        self.assertLess(checked['minimum_source_determinant'],0.)
        self.assertGreater(len(bad),0)

    def test_inverse_oracle_rejects_unsupported_anisotropic_grid(self):
        problem=grid_problem();problem['grid']['spacing']=[1.,2.]
        with self.assertRaisesRegex(ValueError,'isotropic'):
            all_corner_inverse_check(problem,np.zeros(121),np.array([1.,0.]))

    def test_exact_area_integrates_bilinear_field_on_diagonal_source_edges(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11];z=(xx*yy).ravel().astype(float)
        curve=GridCurve([[1.,1.],[3.,5.]],problem,[1.,0.],[0.,1.])
        # Along x=(y+1)/2, u=xy; integral_1^5 u dy = 80/3.
        self.assertAlmostEqual(float((curve.area_row()@z)[0]),80./3.,places=11)
        subdivided=GridCurve([[1.,1.],[1.7,2.4],[3.,5.]],problem,[1.,0.],[0.,1.])
        self.assertAlmostEqual(float((subdivided.area_row()@z)[0]),80./3.,places=11)
        # Endpoint trapezoids would give 32: this must detect that wrong rule.
        self.assertGreater(abs(float((curve.area_row()@z)[0])-32.),5.)

    def test_closed_area_uses_curved_target_edges_not_warped_vertices(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11];z=(xx*yy).ravel().astype(float)
        curve=GridCurve([[0.,0.],[1.,1.],[0.,1.],[0.,0.]],problem,[1.,0.],[0.,1.])
        self.assertAlmostEqual(float((curve.area_row()@z)[0]),1./3.,places=12)
        self.assertGreater(abs(float((curve.area_row()@z)[0])-.5),.16)

    def test_cubic_margin_finds_interior_negative_minimum_missed_by_nodes(self):
        s=np.linspace(0.,1.,4)
        values=(s-.19)**2*(1.+s)-.004
        self.assertTrue(np.all(values>0))
        location,minimum=polynomial_minima(values)
        self.assertAlmostEqual(location[0],.19,places=11)
        self.assertAlmostEqual(minimum[0],-.004,places=12)

    def test_interval_coefficients_reproduce_direct_source_chord_and_sign(self):
        problem=grid_problem();rng=np.random.default_rng(741);z=rng.normal(size=121)
        curve=GridCurve([[1.,1.],[3.,4.],[2.,7.],[5.,9.]],problem,[1.,0.],[0.,1.])
        cuts=curve.chord_partition(1.)
        fractions=np.array([.031,.173,.493,.827,.979])
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if hi-lo<1e-10:continue
            rows,constant=curve.chord_polynomial([lo,hi,-1.],1.,.5)
            s=(1.-fractions)*lo+fractions*hi
            source,op,_=curve.chord(s,1.)
            expected=-((1.-.5)*source+op@z)
            actual=np.polynomial.polynomial.polyval(fractions,constant+rows@z)
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=2e-12)
        for lo,hi in zip(curve.breaks[:-1],curve.breaks[1:]):
            rows,constant=curve.sign_polynomial([lo,hi,-1.])
            self.assertEqual(rows.getrow(3).nnz,0)
            actual=np.polynomial.polynomial.polyval(fractions,constant+rows@z)
            expected=-(curve.op((1.-fractions)*lo+fractions*hi)@z)
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=2e-12)

    def test_hard_zero_quadratic_factor_detects_interior_negative_lobe(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        curve=GridCurve([[1.,1.],[2.,2.]],problem,[1.,0.],[0.,1.])
        region=[0.,curve.length,1.]
        for zero_end in [0.,curve.length]:
            rows=hard_zero_sign_piece_rows(curve,region,[zero_end])
            for a,b,accepted in [(-.1,1.,False),(.1,-1.,False),(1.,-1.,True),(.1,.7,True)]:
                if zero_end==0.:z=(a*(xx-1.)+b*(xx-1.)*(yy-1.)).ravel()
                else:z=(a*(2.-xx)+b*(xx-1.)*(2.-yy)).ravel()
                self.assertEqual(bool(np.all(rows@z>=0.)),accepted)
                self.assertEqual(float((curve.op([zero_end])@z)[0]),0.)
                checked,_=curve.continuous_sign_check(z,1.)
                self.assertEqual(checked['violations']==0,accepted)
        # Merely small/unknown/soft-zero endpoints do not authorize factoring.
        self.assertIsNone(hard_zero_sign_piece_rows(curve,region,[curve.length/2.]))

    def test_conflicting_area_equalities_cannot_be_silently_dropped(self):
        from scipy.sparse import csr_matrix
        row=csr_matrix([[1.,2.]])
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            _independent_equalities([row,2*row],[1.,3.])
        eq,rhs,count=_independent_equalities([row,2*row],[1.,2.])
        self.assertEqual(count,1)
        np.testing.assert_allclose(eq@np.array([.2,.4]),rhs,atol=1e-12)

    def test_fixed_support_cannot_fake_nonzero_jaw_area_or_export_solution(self):
        from test_jaw_curve_qp import curved
        problem={'grid':{'origin':[-50.,-30.],'spacing':[10.,10.],'width':11,'height':11},
                 'mapping':np.full(121,-1),'hessian':eye(1,format='csc'),'target_weights':np.zeros(1)}
        result=solve_joint_field(problem,curved(),100.,[],[])
        self.assertFalse(result['report']['source_continuous_accepted'])
        self.assertIsNone(result['diagnostic_solution'])
        self.assertFalse(result['report']['executable'])
        self.assertIn('zero-support',result['report']['rejection'])

    def test_strict_polish_waits_for_unchanged_problem_and_runs_only_once(self):
        import clarabel
        from test_jaw_curve_qp import curved
        problem={'grid':{'origin':[-50.,-30.],'spacing':[10.,10.],'width':11,'height':11},
                 'mapping':np.arange(121),'hessian':eye(121,format='csc'),'target_weights':np.zeros(1)}
        calls=[]
        def solver(p,q,a,b,cones,settings):
            calls.append((p,q,a,b,cones,settings.tol_feas))
            answer=SimpleNamespace(status=clarabel.SolverStatus.Solved,x=np.zeros(p.shape[0]),
                s=np.zeros(a.shape[0]),z=np.zeros(a.shape[0]),iterations=1,r_prim=0.,r_dual=0.,obj_val=0.,obj_val_dual=0.)
            return SimpleNamespace(solve=lambda:answer)
        geometry_report={'violating_source_cell_corners':1,'accepted':False}
        with patch('clarabel.DefaultSolver',side_effect=solver),patch(
                'joint_jaw_boundary_field.all_corner_inverse_check',return_value=(geometry_report,np.array([[0,1,1]]))):
            result=solve_joint_field(problem,curved(),0.,[],[
                {'source_points':[[0.,30.],[0.,40.]],'uncertainty_radius_px':1.}])
        self.assertEqual([call[-1] for call in calls],[1e-12,1e-12,1e-13])
        for index in range(5):self.assertIs(calls[-2][index],calls[-1][index])
        self.assertFalse(result['report']['source_continuous_accepted'])
        self.assertEqual([row['same_subproblem_strict_polish'] for row in result['report']['refinement']],[False,False,True])
        self.assertEqual(result['report']['refinement'][-2]['subproblem_sha256'],result['report']['refinement'][-1]['subproblem_sha256'])
        self.assertIsNotNone(result['diagnostic_qp'])

    def test_sign_check_inspects_quadratic_inside_source_grid_intervals(self):
        problem=grid_problem();yy,xx=np.mgrid[:11,:11]
        z=((xx-1.45)*(yy-1.45)-.01).ravel()
        curve=GridCurve([[1.1,1.1],[1.9,1.9]],problem,[1.,0.],[0.,1.])
        self.assertTrue(np.all(curve.op([0.,curve.length])@z>0.))
        checked,where=curve.continuous_sign_check(z,1.)
        self.assertGreater(checked['violations'],0)
        self.assertAlmostEqual(checked['minimum'],-.01,places=12)

    def test_off_midline_chin_preserves_radius_sign_flip_and_zero_sites(self):
        problem=grid_problem()
        # A left source arc crosses x=5 before reaching its off-midline chin.
        curve=GridCurve([[2.,1.],[4.,5.],[6.,8.]],problem,[1.,0.],[0.,1.])
        for amount,expected in [(100.,[1.,-1.]),(-100.,[-1.,1.])]:
            regions,zeros=source_sign_regions(curve,np.array([5.,0.]),0.,amount)
            self.assertEqual(regions[0,2],expected[0]);self.assertEqual(regions[-1,2],expected[1])
            self.assertEqual(zeros[0],0.);self.assertEqual(zeros[-1],curve.length)
            self.assertEqual(len(zeros),3)
            np.testing.assert_allclose(curve.source(zeros[1:2]),[[5.,6.5]],atol=1e-12)
        np.testing.assert_array_equal(curve.points,[[2.,1.],[4.,5.],[6.,8.]])


if __name__=='__main__':unittest.main()
