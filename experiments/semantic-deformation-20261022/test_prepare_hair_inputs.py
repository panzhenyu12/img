"""Signed neck-target regressions using synthetic source coordinates only."""
import copy
import unittest

import numpy as np

from prepare_hair_inputs import check_signed_target_evidence


class HairInputTests(unittest.TestCase):
    def evidence(self):
        source = np.column_stack([np.arange(23, dtype=float)+100, np.arange(23, dtype=float)+200])
        pairs = []
        for amount in [60., -60.]:
            destinations = source.copy()
            destinations[5:, 0] += np.arange(18)*amount/60./8.
            controls = {'targets': [{'position': q.tolist(), 'delta': (p-q).tolist(), 'weight': 1.}
                                    for p, q in zip(source, destinations)]}
            provenance = {'source_relative_width': {'amount': amount}, 'source_neck_support': {
                'base_target_count': 5, 'added_target_count': 18,
                'source': source[5:].tolist(), 'requested': destinations[5:].tolist(),
                'source_photo_sha256': 'same-source', 'graph_sha256': 'same-graph'}}
            pairs.append((controls, provenance))
        return pairs

    def test_both_signed_neck_ledgers_are_retained_as_original_evidence(self):
        positive, negative = self.evidence()
        report = check_signed_target_evidence(*positive, *negative, -60.)
        self.assertEqual(report['neck_rows'], 18)
        self.assertEqual(report['selected_amount'], -60.)
        self.assertEqual(report['signed_neck_displacement_max_error'], 0.)

    def test_negative_face_amount_cannot_keep_positive_eighteen_neck_targets(self):
        positive, negative = self.evidence()
        wrong = copy.deepcopy(negative)
        wrong[0]['targets'][5:] = copy.deepcopy(positive[0]['targets'][5:])
        with self.assertRaisesRegex(ValueError, 'neck destinations'):
            check_signed_target_evidence(*positive, *wrong, -60.)

    def test_changing_metadata_to_match_mixed_neck_still_fails_signed_evidence(self):
        positive, negative = self.evidence()
        wrong = copy.deepcopy(negative)
        wrong[0]['targets'][5:] = copy.deepcopy(positive[0]['targets'][5:])
        wrong[1]['source_neck_support']['requested'] = copy.deepcopy(positive[1]['source_neck_support']['requested'])
        with self.assertRaisesRegex(ValueError, 'opposite signed neck'):
            check_signed_target_evidence(*positive, *wrong, -60.)

    def test_changed_original_weight_is_rejected(self):
        positive, negative = self.evidence()
        negative[0]['targets'][6]['weight'] = .5
        with self.assertRaisesRegex(ValueError, 'weights'):
            check_signed_target_evidence(*positive, *negative, -60.)

    def test_unregistered_amount_cannot_borrow_another_amount_neck(self):
        positive, negative = self.evidence()
        with self.assertRaisesRegex(ValueError, 'exact signed baseline'):
            check_signed_target_evidence(*positive, *negative, -30.)


if __name__ == '__main__':
    unittest.main()
