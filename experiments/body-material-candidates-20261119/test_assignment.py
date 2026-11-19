"""Behavior of source-evidence selection; no trained weights or photo labels."""
import unittest
import numpy as np

from assignment import select_candidate, select_person_candidate, connected_torso_proposal, body_witnesses


class SourceRoleSelectionTest(unittest.TestCase):
    def test_rejected_middle_bone_sample_does_not_make_endpoints_adjacent(self):
        points = [dict(x=.1, y=.9, visibility=1) for _ in range(33)]
        for i, x, y in [(11,.125,.25),(13,.875,.25),(15,.875,.75)]:
            points[i] = dict(x=x, y=y, visibility=1)
        classes = np.zeros((18, 12, 16), np.uint8)
        classes[3] = 255
        classes[3,3,8] = 0
        classes[0,3,8] = 255  # Unobserved middle of the upper-arm interval.
        witnesses, groups = body_witnesses(points, classes, 16, 12)
        mask = np.zeros((1,12,16), bool)
        for x, y in witnesses['left-arm']: mask[0,y,x] = True
        selected = select_candidate(mask, witnesses['left-arm'], [[0,0]], groups['left-arm'])
        self.assertIsNone(selected['index'],
            'two separated samples cannot establish an observed bone interval')

    def test_subtracting_a_sleeve_does_not_assign_its_isolated_fringe_to_the_torso(self):
        proposal = np.zeros((12, 16), bool)
        proposal[1:11, 6:14] = True  # Observed connected chest/jacket.
        proposal[3:9, 2] = True    # Sleeve fringe outside a coarse proposal.
        kept, evidence = connected_torso_proposal(proposal, [[8, 4], [9, 8]])
        self.assertTrue(kept[1:11, 6:14].all())
        self.assertFalse(kept[3:9, 2].any())
        self.assertEqual(evidence['unassignedPixels'], 6)
        self.assertTrue(np.array_equal(proposal[3:9, 2], np.ones(6, bool)))
        # A missing torso observation cannot silently choose the largest blob.
        empty, evidence = connected_torso_proposal(proposal, [[4, 4]])
        self.assertFalse(empty.any())
        self.assertEqual(evidence['status'], 'no-observed-torso-component')

    def test_whole_person_cannot_replace_independent_sleeve(self):
        whole = np.ones((12, 16), dtype=bool)
        sleeve = np.zeros_like(whole)
        sleeve[2:10, 1:5] = True
        incomplete = sleeve.copy()
        incomplete[7:] = False
        masks = np.stack([whole, incomplete, sleeve])
        # Independently specified material witnesses: a sleeve spans the arm;
        # the chest and opposite arm are other regions even in one garment.
        chosen = select_candidate(masks, [[3, 3], [3, 6], [3, 8]], [[8, 5], [13, 5]])
        self.assertEqual(chosen['index'], 2)
        self.assertEqual(chosen['status'], 'selected-source-proposal')

    def test_joint_uncertainty_does_not_discard_two_observed_arm_segments(self):
        whole = np.ones((12, 16), dtype=bool)
        sleeve = np.zeros_like(whole)
        sleeve[3:10, 1:5] = True
        forearm_only = sleeve.copy()
        forearm_only[:6] = False
        positive = [[3, 2], [3, 3], [3, 4], [3, 6], [3, 7], [3, 8]]
        masks = np.stack([whole, forearm_only, sleeve])
        chosen = select_candidate(masks, positive, [[8, 5]], groups=[[0, 1, 2], [3, 4, 5]])
        self.assertEqual(chosen['index'], 2)
        self.assertEqual(chosen['unresolvedWitnesses'], [0])
        self.assertFalse(masks[chosen['index'], 2, 3])  # No filling an uncertain joint.
        rejected = select_candidate(masks[:2], positive, [[8, 5]], groups=[[0, 1, 2], [3, 4, 5]])
        self.assertIsNone(rejected['index'])  # An entirely absent upper arm cannot pass.

    def test_whole_instance_requires_both_body_coverage_and_foreground_agreement(self):
        person = np.zeros((12, 16), bool)
        person[1:11, 2:14] = True
        part = person.copy()
        part[:, 8:] = False
        scene = np.ones_like(person)
        observed = np.where(person, .95, .05)
        chosen = select_person_candidate(np.stack([scene, part, person]), observed, [[4, 3], [12, 4], [8, 8]])
        self.assertEqual(chosen['index'], 2)


if __name__ == '__main__':
    unittest.main()
