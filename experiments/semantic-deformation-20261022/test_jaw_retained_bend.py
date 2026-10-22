"""Behavior checks for the isolated fixed-rho source-bend target proposal."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import jaw_curve_qp as core
from probe_jaw_retained_bend import RetainedProblem, solve_retained_bend_profile
from probe_jaw_plateau import OriginalProblem, PlateauProblem
from test_jaw_curve_qp import curved, source_arc


def dense_retention(geometry, result, radius):
    # Independent polygon walker and explicit cross products; no candidate
    # margin/numerator routine is used to measure source-bend retention.
    frame = geometry['frame']; axes = np.array([frame['across'],frame['down']]).T
    result_rows = []
    for profile in result['profiles']:
        source, knots = source_arc(geometry,profile['arc_id'])
        s = np.arange(radius+.0137,knots[-1]-radius,.0173)
        source_q, target_q = [], []
        for shift in [-radius,0.,radius]:
            query = s+shift
            xy = np.column_stack([np.interp(query,knots,source[:,axis]) for axis in [0,1]])
            uv = (xy-frame['origin'])@axes
            source_q.append(uv.copy())
            uv[:,0] += np.interp(query,profile['arclength_px'],profile['across_displacement_px'])
            target_q.append(uv)
        side = 1. if profile['arc_id']=='left-jaw' else -1.
        def n(points):
            a,b,c = points
            return side*((c[:,0]-b[:,0])*(b[:,1]-a[:,1])-(b[:,0]-a[:,0])*(c[:,1]-b[:,1]))
        ns,nt = n(source_q),n(target_q)
        known = ns>1e-6
        for lo,hi in profile['source_nonmonotone_height_intervals_px']:
            known &= ~((s+radius>lo)&(s-radius<hi))
        self_count = int(known.sum())
        if self_count:
            result_rows.append((float((nt[known]/ns[known]).min()),float((nt-.5*ns)[known].min())))
    return result_rows


class RetainedBendTests(unittest.TestCase):
    def test_same_plateau_without_retention_loses_over_half_the_source_bend(self):
        geometry = curved(); before = copy.deepcopy(geometry)
        with patch.object(core,'_Problem',PlateauProblem):
            old = core.solve_jaw_curve_profile(geometry,100.,bilateral_weight=1.)
        self.assertTrue(old['report']['accepted_numerically'],old['report'])
        old_retention = dense_retention(geometry,old,16.)
        self.assertLess(min(row[0] for row in old_retention),.5)
        self.assertLess(min(row[1] for row in old_retention),0.)
        for amount in [100.,-100.]:
            candidate = solve_retained_bend_profile(geometry,amount)
            self.assertTrue(candidate['report']['accepted_numerically'],candidate['report'])
            for ratio, margin in dense_retention(geometry,candidate,16.):
                self.assertGreaterEqual(ratio,.5-1e-8)
                self.assertGreaterEqual(margin,-1e-8)
            self.assertLess(abs(candidate['report']['signed_area_error_px2']),1e-8)
            self.assertEqual(candidate['required_signed_area_delta_px2'],OriginalProblem(geometry,2.,16.).beta_area(amount,.13))
            self.assertTrue(all(r['continuous_violating_subintervals']==0
                for r in candidate['report']['continuous_check']))
            margins = candidate['report']['retained_bend_continuous_check']
            self.assertTrue(all('continuous_min_retained_bend_margin_px2' in r for r in margins))
            self.assertTrue(all('continuous_min_numerator' not in r for r in margins))
        self.assertEqual(geometry,before)
        self.assertIs(core._Problem,OriginalProblem)

    def test_preference_sign_bounds_source_domain_and_unknown_tail_are_unchanged(self):
        geometry = curved(left=np.array([[-40.,0.],[-38.,12.],[-33.,24.],[-24.,36.],[-13.,48.],[-1.,62.],[0.,60.]]))
        old,new = PlateauProblem(geometry,5.,8.),RetainedProblem(geometry,5.,8.)
        for a,b in zip(old.arcs,new.arcs):
            np.testing.assert_array_equal(old.beta(a,a['nodes'],100.,.13),new.beta(b,b['nodes'],100.,.13))
            np.testing.assert_array_equal(old.convex_regions(a),new.convex_regions(b))
            self.assertEqual(a['nonmonotone'],b['nonmonotone'])
        candidate = solve_retained_bend_profile(geometry,60.,node_spacing_px=5.,window_radius_px=8.)
        self.assertTrue(candidate['report']['accepted_numerically'],candidate['report'])
        self.assertTrue(candidate['profiles'][0]['source_nonmonotone_height_intervals_px'])
        for ratio,margin in dense_retention(geometry,candidate,8.):
            self.assertGreaterEqual(ratio,.5-1e-8);self.assertGreaterEqual(margin,-1e-8)

    def test_collinear_subdivision_and_rigid_frame_preserve_intent(self):
        geometry = curved();left,_=source_arc(geometry,'left-jaw');right,_=source_arc(geometry,'right-jaw')
        subdivided=curved(left=np.insert(left,2,.37*left[1]+.63*left[2],axis=0),
                          right=np.insert(right,3,.61*right[2]+.39*right[3],axis=0))
        angle=.41;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        translation=np.array([91.,-14.]);moved=copy.deepcopy(geometry)
        moved['contour']=(np.asarray(moved['contour'])@rotation.T+translation).tolist()
        moved['frame']['origin']=(np.asarray(moved['frame']['origin'])@rotation.T+translation).tolist()
        for axis in ['across','down']:moved['frame'][axis]=(np.asarray(moved['frame'][axis])@rotation.T).tolist()
        results=[solve_retained_bend_profile(g,60.,node_spacing_px=5.,window_radius_px=8.) for g in [geometry,subdivided,moved]]
        self.assertTrue(all(r['report']['accepted_numerically'] for r in results))
        for name in ['left-jaw','right-jaw']:
            length=source_arc(geometry,name)[1][-1];s=np.linspace(0.,length,301)[1:-1]
            rows=[next(p for p in r['profiles'] if p['arc_id']==name) for r in results]
            u=[np.interp(s,p['arclength_px'],p['across_displacement_px']) for p in rows]
            np.testing.assert_allclose(u[1],u[0],atol=2e-6,rtol=0.)
            np.testing.assert_allclose(u[2],u[0],atol=2e-6,rtol=0.)

    def test_zero_is_exact_identity_with_original_source_gate(self):
        geometry=curved();result=solve_retained_bend_profile(geometry,0.)
        self.assertTrue(result['report']['accepted_numerically'])
        self.assertEqual(result['candidate_signed_area_delta_px2'],0.)
        self.assertTrue(all(np.all(np.asarray(p['across_displacement_px'])==0.) for p in result['profiles']))
        self.assertEqual(result['source_geometry'],geometry)


if __name__=='__main__':unittest.main()
