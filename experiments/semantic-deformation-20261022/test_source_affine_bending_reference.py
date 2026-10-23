import unittest
import numpy as np
from source_affine_bending_reference import discrete_reference_bending, source_affine_reference
from probe_jaw_plateau import OriginalProblem
from test_jaw_curve_qp import curved
import copy


class SourceAffineBendingReferenceTests(unittest.TestCase):
    def test_reference_changes_only_linear_and_constant_terms_of_existing_energy(self):
        # h=2, ell=1: H=(1/8) outer([1,-2,1],[1,-2,1]).
        # The reference [2,1,2] has D2=.5, so shift is [-.25,.5,-.25].
        item=discrete_reference_bending(np.array([0.,2.,4.]),np.array([2.,1.,2.]),1.)
        np.testing.assert_array_equal(item['hessian'],.125*np.outer([1.,-2.,1.],[1.,-2.,1.]))
        np.testing.assert_array_equal(item['linear_shift'],[-.25,.5,-.25])
        self.assertEqual(item['constant'],.5)
        for u in [np.array([2.,1.,2.]),np.array([0.,0.,0.]),np.array([1.,3.,2.])]:
            literal=.125*((u[0]-2*u[1]+u[2])-2)**2
            expanded=u@item['hessian']@u+2*item['linear_shift']@u+item['constant']
            self.assertAlmostEqual(float(expanded),literal)

    def test_nonzero_reference_endpoints_are_not_clamped_to_hard_zero(self):
        # Affine in arclength has zero D2 even when both endpoint values are nonzero.
        ref=np.array([4.,5.,6.,7.]);before=ref.copy()
        item=discrete_reference_bending(np.arange(4.),ref,12.)
        np.testing.assert_array_equal(item['linear_shift'],np.zeros(4))
        np.testing.assert_array_equal(ref,before)
        self.assertEqual(item['constant'],0.)

    def test_energy_and_linear_terms_have_physical_similarity_units(self):
        u=np.array([0.,.3,1.,.2]);ref=np.array([2.,1.2,.7,.3]);nodes=np.arange(4.)*2
        first=discrete_reference_bending(nodes,ref,3.)
        scale=7.;last=discrete_reference_bending(nodes*scale,ref*scale,3.*scale)
        def energy(item,x):return x@item['hessian']@x+2*item['linear_shift']@x+item['constant']
        self.assertAlmostEqual(energy(last,u*scale)/energy(first,u),scale**3)
        np.testing.assert_allclose(last['linear_shift'],scale**2*first['linear_shift'],atol=1e-11)

    def test_shared_coefficient_preserves_continuous_area_and_signed_amount(self):
        geometry=curved();before=copy.deepcopy(geometry);problem=OriginalProblem(geometry,2.,16.)
        plus=source_affine_reference(problem,100.,.13,12.)
        minus=source_affine_reference(problem,-100.,.13,12.)
        zero=source_affine_reference(problem,0.,.13,12.)
        self.assertEqual(geometry,before)
        self.assertAlmostEqual(sum(c['continuous_affine_jaw_area_px2'] for c in plus['curves']),problem.beta_area(100.,.13))
        self.assertGreater(plus['report']['width_fraction'],0.)
        self.assertEqual(minus['report']['width_fraction'],-plus['report']['width_fraction'])
        np.testing.assert_allclose(minus['linear_shift'],-plus['linear_shift'],atol=0.,rtol=0.)
        np.testing.assert_array_equal(zero['linear_shift'],np.zeros(len(zero['linear_shift'])))
        for c,a in zip(plus['curves'],problem.arcs):
            k=plus['report']['width_fraction'];r=(problem.source(a,a['nodes'])-problem.origin)@problem.across-problem.midline
            np.testing.assert_array_equal(c['reference_displacement'],-k*r)
        self.assertNotEqual(plus['curves'][0]['reference_displacement'][0],0.)

    def test_rigid_frame_change_and_physical_scale_preserve_width_fraction(self):
        g=curved();angle=.37;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        scale=3.;translation=np.array([100.,-20.]);m=copy.deepcopy(g)
        m['contour']=(np.asarray(g['contour'])@rot.T*scale+translation).tolist()
        m['frame']['origin']=(np.asarray(g['frame']['origin'])@rot.T*scale+translation).tolist()
        for key in ['across','down']:m['frame'][key]=(np.asarray(g['frame'][key])@rot.T).tolist()
        # Midline is expressed in frame-local physical coordinates.
        if 'midline' in m['frame']:m['frame']['midline']*=scale
        p=source_affine_reference(OriginalProblem(g,2.,16.),100.,.13,12.)
        q=source_affine_reference(OriginalProblem(m,6.,48.),100.,.13,36.)
        self.assertAlmostEqual(p['report']['width_fraction'],q['report']['width_fraction'])
        np.testing.assert_allclose(q['linear_shift'],scale**2*p['linear_shift'],atol=2e-8,rtol=2e-10)


if __name__=='__main__':unittest.main()
