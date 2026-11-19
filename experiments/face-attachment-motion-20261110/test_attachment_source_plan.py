"""Source component permissions, including the observed soft-edge conflict."""
import unittest
import numpy as np
from attachment_source_plan import plan,verify,partition


class SourceAttachmentPlan(unittest.TestCase):
    def typed_fixture(self):
        accessory=np.zeros((40,64),np.float32)
        accessory[7:21,7:43]=.12
        accessory[8:20,8:42]=.9
        accessory[26:35,49:57]=.12
        accessory[27:33,50:56]=.9
        eyewear=np.zeros_like(accessory);eyewear[8:20,8:42]=.9
        earring=np.zeros_like(accessory);earring[27:33,50:56]=.9
        left=np.zeros_like(accessory,bool);left[12:16,12:20]=True
        right=np.zeros_like(accessory,bool);right[12:16,29:37]=True
        return accessory,{'leftEye':left,'rightEye':right},{
            'eyewear':eyewear,'earring':earring,'observed':np.ones_like(accessory,bool)}

    def test_source_eyewear_stays_whole_and_fixed_while_separate_earring_moves(self):
        accessory,barriers,subtypes=self.typed_fixture();before=accessory.copy()
        support,items=plan(accessory,barriers,subtypes=subtypes)
        self.assertFalse(support[7:21,7:43].any(), 'whole frame/lens instance stays fixed, not just eye intersections')
        self.assertTrue(support[26:35,49:57].all(), 'entire separate earring retains soft-boundary permission')
        self.assertEqual(len(items),1)
        self.assertEqual(int(support.sum()),72)
        np.testing.assert_array_equal(accessory,before)

    def test_mixed_soft_connected_instance_is_not_split_into_moving_branches(self):
        accessory,barriers,subtypes=self.typed_fixture()
        accessory[19:28,41:51]=.12
        support,items=plan(accessory,barriers,subtypes=subtypes)
        self.assertFalse(support.any());self.assertEqual(items,[])
        groups=partition(accessory,barriers,subtypes)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['kind'],'unresolved')
        self.assertEqual(groups[0]['coreLabels'],[1,2])
        self.assertTrue(groups[0]['support'][32,55])

    def test_true_moving_earring_still_rejects_independent_protection(self):
        accessory,barriers,subtypes=self.typed_fixture()
        protected=np.zeros_like(accessory,bool);protected[34,52]=True
        barriers['user']=protected
        with self.assertRaisesRegex(ValueError,'protected user'):
            plan(accessory,barriers,subtypes=subtypes)

    def test_unknown_subtype_or_unobserved_branch_stays_fixed_without_relabeling(self):
        for unobserved in [False,True]:
            accessory,barriers,subtypes=self.typed_fixture()
            if unobserved:subtypes['observed'][34,52]=False
            else:subtypes['earring'][:]=0
            support,items=plan(accessory,barriers,subtypes=subtypes)
            self.assertFalse(support.any());self.assertEqual(items,[])
            self.assertEqual(partition(accessory,barriers,subtypes)[1]['kind'],'unresolved')

    def test_typed_verifier_cannot_reunion_fixed_eyewear_or_drop_earring_soft_edge(self):
        accessory,barriers,subtypes=self.typed_fixture()
        support,_=plan(accessory,barriers,subtypes=subtypes)
        self.assertEqual(len(verify(support,accessory,barriers,subtypes=subtypes)),1)
        for bad in [accessory>.02,support&(accessory>=.5),np.zeros_like(support)]:
            with self.assertRaisesRegex(ValueError,'differs'):
                verify(bad,accessory,barriers,subtypes=subtypes)

    def fixture(self):
        evidence=np.zeros((20,24),np.float32)
        evidence[4:10,6:12]=.9
        evidence[10:13,6:12]=.12
        evidence[10:13,18:21]=.12  # a separate weak classifier island
        return evidence

    def test_core_and_its_existing_soft_boundary_have_one_motion_permission(self):
        evidence=self.fixture();before=evidence.copy()
        support,items=plan(evidence,{})
        self.assertEqual(len(items),1)
        self.assertTrue(support[11,8])
        self.assertFalse((evidence>=.5)[11,8], 'legacy core-only release misses this material')
        self.assertFalse(support[11,19], 'no disconnected weak-island promotion')
        np.testing.assert_array_equal(evidence,before)
        self.assertEqual(int(support.sum()),54)

    def test_any_true_protection_on_soft_material_is_a_conflict_not_a_trim(self):
        for name in ('user','hair','feature','garment'):
            blocked=np.zeros((20,24),bool);blocked[11,8]=True
            with self.assertRaisesRegex(ValueError,'protected '+name):
                plan(self.fixture(),{name:blocked})

    def test_nearby_protection_is_not_expanded_into_the_source_component(self):
        blocked=np.zeros((20,24),bool);blocked[13,8]=True
        support,_=plan(self.fixture(),{'user':blocked})
        self.assertTrue(support[12,8]);self.assertFalse(support[13,8])

    def test_missing_or_tiny_evidence_does_not_claim_an_attachment(self):
        for evidence in [np.zeros((20,24)),np.pad(np.ones((3,3)),((1,16),(1,20)))]:
            support,items=plan(evidence,{})
            self.assertFalse(support.any());self.assertEqual(items,[])

    def test_downstream_cannot_replace_full_support_by_a_core_only_release(self):
        evidence=self.fixture()
        with self.assertRaisesRegex(ValueError,'differs'):
            verify(evidence>=.5,evidence,{})
        support,_=plan(evidence,{})
        self.assertEqual(len(verify(support,evidence,{})),1)


if __name__=='__main__':unittest.main()
