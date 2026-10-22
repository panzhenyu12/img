"""Behavior checks for equal-height coupling on unchanged synthetic arcs."""
import copy
import unittest

import numpy as np

from jaw_bilateral import build_bilateral_coupling
from jaw_curve_qp import _Problem, _basis, solve_jaw_curve_profile
from test_jaw_curve_qp import curved, source_arc


class JawBilateralTests(unittest.TestCase):
    def test_equal_fraction_shrink_has_zero_energy_but_equal_pixels_do_not(self):
        y=np.arange(0.,61.,10.)
        geometry=curved(left=np.column_stack([-30+30*(y/60)**2,y]),
            right=np.column_stack([60-60*(y/60)**2,y]))
        problem=_Problem(geometry,5.,8.)
        # Include the true source corners so their radii are represented
        # exactly, without approximation by a different control partition.
        for arc in problem.arcs:arc['nodes']=arc['knots'].copy()
        pairing=build_bilateral_coupling(problem,_basis)
        proportional=[];equal_pixels=[]
        for arc in problem.arcs:
            source=problem.source(arc,arc['nodes'])
            radius=(source-problem.origin)@problem.across-problem.midline
            proportional.extend(-.1*radius)
            equal_pixels.extend((1. if arc['id']=='left-jaw' else -1.)*(1-source[:,1]/60.))
        self.assertLess(pairing.energy(np.asarray(proportional)),1e-22)
        # Here b=2a, so the residual is 2/3*(1-y/60), whose
        # squared integral over y=0..60 is exactly 80/9 px^3.
        self.assertAlmostEqual(pairing.energy(np.asarray(equal_pixels)),80/9,places=10)

    def test_multiple_source_intersections_are_excluded_before_pairing(self):
        geometry=curved(left=np.array([[-40.,0.],[-35.,30.],[-25.,50.],
                [-15.,40.],[-5.,55.],[0.,60.]]),
            right=np.array([[40.,0.],[35.,20.],[20.,40.],[0.,60.]]))
        original=copy.deepcopy(geometry);problem=_Problem(geometry,5.,8.)
        pairing=build_bilateral_coupling(problem,_basis);report=pairing.describe()
        # Origin y=-10: three left intersections at down heights 50..60.
        self.assertFalse(np.any((pairing.heights>50.)&(pairing.heights<60.)))
        excluded=[r['height_interval_px'] for r in report['excluded_height_intervals']
                  if r['reason']=='multiple_source_intersections']
        self.assertEqual(excluded,[[50.,60.]])
        for lo,hi in problem.arcs[0]['nonmonotone']:
            self.assertFalse(np.any((pairing.source_arclength[:,0]>lo)
                                   &(pairing.source_arclength[:,0]<hi)))
        self.assertEqual(geometry,original)

    def test_asymmetric_solve_preserves_area_and_reduces_fraction_mismatch(self):
        y=np.arange(0.,61.,10.)
        geometry=curved(left=np.column_stack([-45+45*(y/60)**2,y]),
            right=np.column_stack([50-50*(y/60)**1.4,y]))
        options=dict(node_spacing_px=5.,window_radius_px=8.,smoothing_length_px=8.)
        old=solve_jaw_curve_profile(geometry,100.,**options)
        paired=solve_jaw_curve_profile(geometry,100.,bilateral_weight=1.,**options)
        heights=np.arange(.031,59.95,.071)
        def mismatch(result):
            fractions=[]
            for profile in result['profiles']:
                source,s=source_arc(geometry,profile['arc_id'])
                query=np.interp(heights,source[:,1],s)
                radius=np.interp(heights,source[:,1],source[:,0])
                displacement=np.interp(query,profile['arclength_px'],profile['across_displacement_px'])
                fractions.append(-displacement/radius)
            return float(np.sqrt(np.mean((fractions[0]-fractions[1])**2)))
        for result in [old,paired]:
            self.assertTrue(result['report']['accepted_numerically'],result['report'])
            self.assertLess(abs(result['report']['signed_area_error_px2']),1e-8)
            self.assertTrue(all(c['continuous_violating_subintervals']==0
                for c in result['report']['continuous_check']))
            self.assertEqual(result['report']['endpoint_displacement_max_px'],0.)
            self.assertLessEqual(result['report']['sign_violation_px'],1e-10)
        self.assertEqual(old['required_signed_area_delta_px2'],paired['required_signed_area_delta_px2'])
        self.assertLess(mismatch(paired),.9*mismatch(old))
        report=paired['report']['bilateral_coupling']
        self.assertEqual(report['weight'],1.)
        self.assertEqual(report['reference_kind'],'piecewise-linear beta control preference; not a lambda-zero solution')
        self.assertFalse(report['anatomical_correspondence_claimed'])

    def test_explicit_zero_weight_preserves_old_profile_values(self):
        geometry=curved()
        options=dict(node_spacing_px=5.,window_radius_px=8.)
        old=solve_jaw_curve_profile(geometry,60.,**options)
        zero=solve_jaw_curve_profile(geometry,60.,bilateral_weight=0.,**options)
        self.assertEqual(old,zero)


if __name__=='__main__':unittest.main()
