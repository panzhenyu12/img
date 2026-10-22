import unittest

import numpy as np
from scipy.sparse import csc_matrix

from joint_jaw_boundary_field import GridCurve,jaw_prefix_intent_block,jaw_prefix_intent_blocks,check_jaw_prefix_intent
from probe_jaw_plateau import PlateauProblem
from test_jaw_curve_qp import curved


def problem(size=21,origin=(0.,0.)):
    return {'grid':{'origin':list(origin),'spacing':[1.,1.],'width':size,'height':size},
            'mapping':np.arange(size*size),'hessian':csc_matrix((size*size,size*size))}


class JawPrefixIntentTests(unittest.TestCase):
    def test_unweighted_prefix_integrates_linear_and_quadratic_with_fixed_outer_scale(self):
        p=problem();yy,xx=np.mgrid[:21,:21]
        curve=GridCurve([[1.,1.],[15.,15.]],p,[1.,0.],[0.,1.]);scale=12.
        slope=.4;curvature=.06
        fields=[(slope*((xx-1.)+(yy-1.))/np.sqrt(2.)).ravel(),
                (2.*curvature*(xx-1.)*(yy-1.)).ravel()]
        for prefix in [2.,6.,12.]:
            block=jaw_prefix_intent_block(curve,prefix,scale,100.)
            expected=[scale**2*slope**2*prefix,4./3.*scale**2*curvature**2*prefix**3]
            for z,energy in zip(fields,expected):
                residual=block['operator']@z
                self.assertAlmostEqual(float(residual@residual),energy,places=10)
            np.testing.assert_allclose(block['operator']@np.full(441,2.),0.,atol=1e-14)

    def test_full_window_budget_can_hide_early_concentration_but_nested_cap_rejects(self):
        p=problem();yy,xx=np.mgrid[:21,:21]
        curve=GridCurve([[5.,1.],[5.,19.]],p,[1.,0.],[0.,1.]);scale=12.
        # Synthetic reference intent u=(s/12)^3. Candidate spends modest
        # total energy entirely in [0,2] and then stays flat; endpoint u0=0.
        z=(.1*np.minimum(np.maximum(yy-1.,0.),2.)).ravel()
        blocks=[jaw_prefix_intent_block(curve,r,scale,scale**2*9.*r**5/(5.*12.**6)) for r in [2.,6.,12.]]
        full=check_jaw_prefix_intent(blocks[-1:],z)
        nested=check_jaw_prefix_intent(blocks,z)
        self.assertTrue(full['accepted'])
        self.assertFalse(nested['accepted'])
        self.assertFalse(nested['constraints'][0]['accepted'])
        self.assertGreater(nested['constraints'][0]['actual_motion']['at_window_end_px'],(2./12.)**3*40.)

    def test_soc_slack_is_exact_energy_ball_and_does_not_pool_sides(self):
        p=problem();yy,xx=np.mgrid[:21,:21]
        curve=GridCurve([[5.,1.],[5.,19.]],p,[1.,0.],[0.,1.])
        block=jaw_prefix_intent_block(curve,2.,12.,.02)
        for slope,accepted in [(0.,True),(.005,True),(.02,False)]:
            z=(slope*(yy-1.)).ravel();slack=block['b']-block['A']@z
            self.assertEqual(bool(np.linalg.norm(slack[1:])<=slack[0]),accepted)
            self.assertEqual(check_jaw_prefix_intent([block],z)['accepted'],accepted)
        # An underspent independent constraint must not offset another cap.
        other=jaw_prefix_intent_block(curve,2.,12.,100.)
        self.assertFalse(check_jaw_prefix_intent([block,other],(.02*(yy-1.)).ravel())['accepted'])

    def test_plateau_caps_keep_original_profile_and_scale_with_amount_squared(self):
        preferred=PlateauProblem(curved(),2.,16.);p=problem(101,(-50.,-30.))
        curves=[GridCurve(preferred.source(a,a['knots']),p,preferred.across,preferred.down,source_knots=a['knots'])
                for a in preferred.arcs]
        positive=jaw_prefix_intent_blocks(preferred,curves,100.,.13,12.,2.)
        negative=jaw_prefix_intent_blocks(preferred,curves,-100.,.13,12.,2.)
        half=jaw_prefix_intent_blocks(preferred,curves,50.,.13,12.,2.)
        self.assertEqual(len(positive),6)
        for a,b,c in zip(positive,negative,half):
            self.assertAlmostEqual(a['energy_cap'],b['energy_cap'],places=12)
            self.assertAlmostEqual(a['energy_cap'],4*c['energy_cap'],places=12)
            self.assertEqual(a['scale_px'],12.)
            self.assertLess(abs(a['report']['preference_antiderivative_difference_px3']),1e-9)
        self.assertEqual([b['prefix_px'] for b in positive],[2.,6.,12.,2.,6.,12.])


if __name__=='__main__':unittest.main()
