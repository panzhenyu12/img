import copy
import unittest
from prompted import validate_queries


class PromptContractTest(unittest.TestCase):
    def setUp(self):
        self.request = {'manualCorrection': True, 'queries': [
            {'id': 'sleeve', 'positive': [[10, 20], [20, 30]], 'negative': [[60, 60]]}]}

    def test_variable_bounded_point_count(self):
        self.assertEqual(len(validate_queries(self.request, (100, 100), [])[0]['negative']), 1)

    def test_evaluation_core_exclusion(self):
        with self.assertRaisesRegex(ValueError, 'evaluation core'):
            validate_queries(self.request, (100, 100), [[9, 19, 11, 21]])

    def test_outside_source(self):
        bad = copy.deepcopy(self.request)
        bad['queries'][0]['positive'][0] = [100, 20]
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_queries(bad, (100, 100), [])

    def test_conflicting_labels(self):
        bad = copy.deepcopy(self.request)
        bad['queries'][0]['negative'] = [[10, 20]]
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            validate_queries(bad, (100, 100), [])

    def test_manual_claim_and_point_budget(self):
        bad = copy.deepcopy(self.request)
        bad['manualCorrection'] = False
        with self.assertRaisesRegex(ValueError, 'explicit'):
            validate_queries(bad, (100, 100), [])
        bad = copy.deepcopy(self.request)
        bad['queries'][0]['positive'].append([30, 40])
        with self.assertRaisesRegex(ValueError, 'at most two'):
            validate_queries(bad, (100, 100), [])


if __name__ == '__main__':
    unittest.main()
