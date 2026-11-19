import unittest
import numpy as np

from geometry import visible_motion_identity


class ExplicitMotionIdentityTest(unittest.TestCase):
    def test_complete_rear_support_does_not_turn_occluders_or_fixed_coat_into_targets(self):
        roles = [
            {'id':'shirt','role':'moving-rear','polygon':[[1,1],[18,1],[18,18],[1,18]]},
            {'id':'coat','role':'stationary-scene','polygon':[[15,2],[18,2],[18,17],[15,17]]},
            {'id':'bag','role':'stationary-front','polygon':[[2,12],[8,12],[8,18],[2,18]]},
            {'id':'sweater-front-arm','role':'stationary-front','polygon':[[0,0],[19,0],[19,19]]},
        ]
        sleeve = np.zeros((20,20),bool)
        sleeve[5:9,4:14] = True
        sleeve[6,7] = False  # An enclosed proposal miss is not a rear observation.
        observed = np.ones((20,20))
        observed[10,10] = 0
        motion, _ = visible_motion_identity(roles, (20,20), observed, {'sweater-front-arm':sleeve})
        self.assertTrue(motion[11,10])  # Visible, unoccluded shirt still moves.
        for y,x in [(8,16),(14,5),(6,7),(7,10),(10,10)]:
            self.assertFalse(motion[y,x], (x,y))
        self.assertFalse(sleeve[6,7])  # No source/optical hole was filled in place.

    def test_unrecognized_motion_role_is_not_assumed_movable(self):
        with self.assertRaises(ValueError):
            visible_motion_identity([{'id':'coat','role':'unknown','polygon':[[1,1],[2,1],[2,2]]}],
                (4,4), np.ones((4,4)), {})

    def test_replacement_uses_explicit_role_instead_of_sample_specific_name(self):
        roles = [
            {'id':'blouse','role':'moving-rear','polygon':[[0,0],[7,0],[7,7],[0,7]]},
            {'id':'crossed-sleeve','role':'stationary-front','polygon':[[0,0],[7,0],[7,7],[0,7]]},
        ]
        sleeve = np.zeros((8,8),bool)
        sleeve[2:5,2:6] = True
        motion,_ = visible_motion_identity(roles,(8,8),np.ones((8,8)),{'crossed-sleeve':sleeve})
        self.assertTrue(motion[6,6])
        self.assertFalse(motion[3,3])
        with self.assertRaises(ValueError):
            visible_motion_identity(roles,(8,8),np.ones((8,8)),{'unbound-role':sleeve})

    def test_wrong_shape_or_non_categorical_proposal_is_not_an_identity(self):
        roles = [{'id':'sleeve','role':'stationary-front','polygon':[[0,0],[3,0],[3,3]]}]
        for proposal in [np.zeros((3,4),bool),np.full((4,4),.5)]:
            with self.assertRaises(ValueError):
                visible_motion_identity(roles,(4,4),np.ones((4,4)),{'sleeve':proposal})


if __name__ == '__main__':
    unittest.main()
