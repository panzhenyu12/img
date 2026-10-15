"""Regression from the 13 real source pixels worsened by a mismatched B.

The real frozen trace is intentionally reused: it detects the pipeline bug
between an otherwise passing solver and the final background compositor.
"""
from pathlib import Path
import json
import unittest

import numpy as np
from PIL import Image

from pipeline import correct_foreground


class FinalBackgroundConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.here = Path(__file__).resolve().parent
        cls.root = cls.here.parents[1]
        cls.saved = np.load(cls.here / 'outputs-v2-monotonic/source-solve.npz')
        layer = cls.root / 'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-trace-inverse/100/layers'
        state = json.loads((layer / 'state.json').read_text())
        x, y = state['origin']; w, h = state['width'], state['height']
        cls.source = np.asarray(Image.open(layer / 'source-crop.png').convert('RGB')).astype(float) / 255
        scene = np.load(cls.root / 'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz')
        cls.observed = np.floor(scene['scenePrior'][y:y+h, x:x+w] * 255 + .5) / 255
        cls.final = np.load(cls.root / 'experiments/alpha-feasibility-skin-v1-20261015/qa/v37-scene-b-only-v2-feather/100/background-replaced.npy').astype(float) / 255
        cls.alpha = cls.saved['alpha'].astype(float)
        cls.old = cls.saved['sourceForegroundByte'].astype(float) / 255

    def residual(self, foreground):
        return np.abs(self.source - self.alpha[..., None] * foreground - (1-self.alpha[..., None]) * self.final)

    def test_saved_v2_exhibits_the_actual_background_counterexample(self):
        previous = self.saved['correctedForegroundByte'].astype(float) / 255
        changed = np.any(previous != self.old, axis=-1)
        worse = changed & np.any(self.residual(previous) > self.residual(self.old) + 1e-8, axis=-1)
        self.assertEqual(int(worse.sum()), 13)

    def test_final_byte_foreground_never_worsens_actual_source_background_fit(self):
        result = correct_foreground(self.source, self.alpha, self.old,
                                    self.final,
                                    self.saved['skinCore'], self.saved['allowed'],
                                    self.saved['transition'])
        new = result['correctedForegroundByte'].astype(float) / 255
        changed = np.any(new != self.old, axis=-1)
        worse = changed & np.any(self.residual(new) > self.residual(self.old) + 1e-8, axis=-1)
        self.assertEqual(int(worse.sum()), 0, f'{int(worse.sum())} pixels worsen against the actual final B')


if __name__ == '__main__':
    unittest.main(verbosity=2)
