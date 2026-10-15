"""Regression against archived real inference, including original failures."""
import importlib.util
import itertools
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('semantic_selection', HERE / 'select.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)
raw = json.loads((selection.RAW / 'report.json').read_text())


class SemanticSelectionRegression(unittest.TestCase):
    def test_hair_does_not_select_high_score_full_face(self):
        photo = next(p for p in raw['images'] if p['name'] == 'portrait-hair')
        case = next(c for c in photo['rows'] if c['prompt']['protocol'] == 'component-box-positive')
        official = case['alphas'][case['officialScoreArgmax']]
        self.assertGreater(official['skinCoreRetainedFraction'], .70)
        for candidates in itertools.permutations(case['alphas']):
            chosen, _ = selection.choose(candidates, 'hair')
            result = next(c for c in candidates if c['index'] == chosen)
            self.assertLess(result['skinCoreRetainedFraction'], .001)
            self.assertGreater(result['parserCoreRetainedFraction'], .95)
            self.assertNotEqual(chosen, case['officialScoreArgmax'])

    def test_unresolved_negative_point_is_not_silently_ignored(self):
        for name in ['portrait-woman', 'portrait-hair']:
            photo = next(p for p in raw['images'] if p['name'] == name)
            case = next(c for c in photo['rows'] if c['prompt']['protocol'].endswith('skin-negative'))
            chosen, decisions = selection.choose(case['alphas'], 'hair')
            self.assertIsNone(chosen, name)
            self.assertTrue(all(not d['eligible'] for d in decisions))

    def test_previously_compatible_prompts_are_stable(self):
        for photo in raw['images']:
            if photo['name'] == 'portrait-hair':
                continue
            for case in photo['rows']:
                if case['prompt']['protocol'] != 'component-box-positive':
                    continue
                chosen, _ = selection.choose(case['alphas'], case['prompt']['kind'])
                self.assertEqual(chosen, case['officialScoreArgmax'])

    def test_no_fallback_to_incompatible_candidate(self):
        self.assertEqual(selection.choose([], 'hair'), (None, []))
        failed = dict(index=0, score=1.0, positiveAlpha=0.0, negativeAlpha=[],
                      skinCoreRetainedFraction=0.0, parserCoreRetainedFraction=1.0)
        chosen, decisions = selection.choose([failed], 'hair')
        self.assertIsNone(chosen)
        self.assertIn('positive point excluded', decisions[0]['reasons'])


if __name__ == '__main__':
    unittest.main()
