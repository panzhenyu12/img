"""Behavior checks for source gates retained during final-map correction."""
import copy
import unittest

import numpy as np
from scipy.sparse import eye

from joint_jaw_boundary_field import GridCurve, jaw_prefix_intent_blocks
from probe_jaw_plateau import OriginalProblem, PlateauProblem
from test_jaw_curve_qp import curved


def fixture(amount=0.):
    geometry=curved()
    grid={'origin':[-50.,-30.],'spacing':[1.,1.],'width':101,'height':101}
    mapping=np.arange(10201)
    problem={'grid':grid,'mapping':mapping,'hessian':eye(len(mapping),format='csc')}
    old=OriginalProblem(geometry,2.,16.);preference=PlateauProblem(geometry,2.,16.)
    jaws=[GridCurve(old.source(a,a['knots']),problem,old.across,old.down,source_knots=a['knots'])
          for a in old.arcs]
    budgets=jaw_prefix_intent_blocks(preference,jaws,amount,.13,12.,2.)
    report={'source_geometry':geometry,'grid':grid,'amount':amount,'jaw_radius_px':16.,
            'interface_radius_px':12.,'interface_source_relative_bend_band':[.5,1.5],
            'requested_jaw_and_full_contour_area_px2':old.beta_area(amount,.13),
            'interface_displacement_bending':{'requested_node_spacing_px':2.},
            'jaw_start_displacement_regularization':{'length_px':12.},
            'jaw_start_intent_constraints':{'enabled':True,'budgets':[b['report'] for b in budgets]}}
    return report,{'segments':[]},mapping,problem


class FrozenSourceGateTests(unittest.TestCase):
    def test_identity_passes_all_gates_and_zero_budgets_remain_six_separate_constraints(self):
        from joint_source_constraints import SourceConstraintAudit
        reference,interface,mapping,_=fixture()
        audit=SourceConstraintAudit(reference,interface,mapping)
        result=audit.audit(np.zeros(len(mapping)))
        self.assertTrue(result['accepted'])
        self.assertEqual(len(result['jaw_start_intent']['constraints']),6)
        self.assertEqual(result['zero_max_px'],0.)
        self.assertEqual(result['jaw_area_px2'],0.)

    def test_frozen_original_prefix_budget_cannot_be_silently_changed(self):
        from joint_source_constraints import SourceConstraintAudit
        reference,interface,mapping,_=fixture(100.)
        reference=copy.deepcopy(reference)
        reference['jaw_start_intent_constraints']['budgets'][0]['energy_cap_px3']*=1.01
        with self.assertRaisesRegex(ValueError,'budget'):
            SourceConstraintAudit(reference,interface,mapping)

    def test_source_audit_keeps_endpoint_and_prefix_failures_even_with_safe_geometry(self):
        from joint_source_constraints import SourceConstraintAudit
        reference,interface,mapping,_=fixture(100.)
        audit=SourceConstraintAudit(reference,interface,mapping)
        # Translation has identity differential but moves original hard-zero
        # sites. A safe matrix is insufficient to authorize the source field.
        moved=audit.audit(np.full(len(mapping),.125))
        self.assertTrue(moved['all_source_corner_geometry']['accepted'])
        self.assertAlmostEqual(moved['zero_max_px'],.125)
        self.assertFalse(moved['accepted'])
        yy,xx=np.mgrid[:101,:101]
        rapid=(.2*(yy-30.)).ravel()
        onset=audit.audit(rapid)
        self.assertTrue(onset['all_source_corner_geometry']['accepted'])
        self.assertGreater(onset['jaw_start_intent']['violations'],0)
        self.assertFalse(onset['accepted'])

    def test_new_corner_cut_rejects_fold_and_keeps_auxiliary_columns_zero(self):
        from joint_source_constraints import SourceConstraintAudit
        reference,interface,mapping,_=fixture()
        audit=SourceConstraintAudit(reference,interface,mapping)
        z=np.zeros(len(mapping));z[50*101+50]=4.
        result=audit.audit(z)
        self.assertFalse(result['all_source_corner_geometry']['accepted'])
        block=audit.additional_cuts(z,len(mapping)+6)
        self.assertGreater(block['new_geometry_corners'],0)
        self.assertEqual(block['A'][:,len(mapping):].nnz,0)
        slack=block['b']-block['A']@np.r_[z,np.zeros(6)]
        offset=0;failed=0
        for cone in block['cones']:
            part=slack[offset:offset+cone.dim];offset+=cone.dim
            if type(cone).__name__=='SecondOrderConeT':failed+=part[0]<np.linalg.norm(part[1:])
            else:failed+=np.any(part<0)
        self.assertGreater(failed,0)
        repeated=audit.additional_cuts(z,len(mapping)+6)
        self.assertEqual(repeated['new_constraints'],0)

    def test_bend_band_rejects_same_sign_concentration_that_sign_only_allows(self):
        from joint_source_constraints import SourceConstraintAudit
        reference,interface,mapping,_=fixture()
        # Known parabola and quadratic height displacement: both curves retain
        # turn sign, but the target numerator is twice the source numerator.
        y=np.arange(0.,41.,2.)
        points=np.column_stack([.005*(y-20.)**2,y])
        interface={'segments':[{'id':'observed','source_points':points.tolist()}]}
        yy,xx=np.mgrid[:101,:101];world_y=yy-30.
        z=(.005*(world_y-20.)**2).ravel()
        strict=SourceConstraintAudit(reference,interface,mapping)
        result=strict.audit(z)
        interface_rows=[r for r in result['curves'] if r['id'].startswith('observed')]
        self.assertTrue(any(r['violations'] for r in interface_rows))
        original=copy.deepcopy(reference);original['interface_source_relative_bend_band']=None
        sign_only=SourceConstraintAudit(original,interface,mapping).audit(z)
        self.assertFalse(any(r['violations'] for r in sign_only['curves'] if r['id']=='observed'))
        cuts=strict.additional_cuts(z,len(mapping))
        self.assertGreater(cuts['new_curve_planes'],0)


if __name__=='__main__':unittest.main()
