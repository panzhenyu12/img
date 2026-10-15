"""Validate retained inference artifacts without rerunning or imitating the model."""
from pathlib import Path
import hashlib
import json
import unittest

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class ArtifactContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((HERE / 'outputs-v2-crop-seed/report.json').read_text())

    def test_model_provenance(self):
        model = ROOT / '.models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt'
        self.assertEqual(model.stat().st_size, 215569778)
        self.assertEqual(sha(model), self.report['modelSha256'])
        vendor = json.loads((HERE / 'vendor-revision.json').read_text())
        self.assertEqual(vendor['commit'], self.report['vendorCommit'])

    def test_protocol_and_retained_failure(self):
        self.assertFalse(self.report['accepted'])
        self.assertEqual({r['name'] for r in self.report['rows']},
                         {'user-face-contour', 'portrait-woman', 'portrait-hair'})
        for row in self.report['rows']:
            with self.subTest(case=row['name']):
                case = ROOT / 'experiments/joint-skin-matting-v1-20261015/prompts-v2' / row['name']
                meta = json.loads((case / 'report.json').read_text())
                source = ROOT / row['source']
                self.assertEqual(sha(source), row['sourceSha256'])
                self.assertEqual(row['sourceSha256'], meta['sourceSha256'])
                self.assertEqual(row['graphSha256'], meta['graphSha256'])
                seed = case / 'crop-point/alphas-full.npy'
                full = case / 'full-point/alphas-full.npy'
                self.assertEqual(sha(seed), row['seedAlphaSha256'])
                self.assertEqual(sha(full), row['fullAlphaSha256'])
                z = np.load(seed)[2]
                f = np.load(full)[2]
                output = HERE / row['directory']
                raw = output / 'alpha-raw.npy'
                self.assertEqual(sha(raw), row['rawAlphaSha256'])
                a = np.load(raw)
                self.assertEqual(a.shape, np.asarray(Image.open(source)).shape[:2])
                self.assertTrue(np.isfinite(a).all())
                self.assertTrue(((0 <= a) & (a <= 1)).all())
                self.assertTrue(np.array_equal(np.asarray(Image.open(output / 'input-mask.png')), (z > .5).astype('uint8') * 255))
                self.assertTrue(np.array_equal(np.asarray(Image.open(output / 'alpha.png')), np.floor(a * 255 + .5).astype('uint8')))
                core = (z >= .99) & (f >= .99)
                self.assertEqual(int(core.sum()), row['consensusOpaquePixels'])
                self.assertEqual(int((core & (a < .9)).sum()), row['consensusOpaqueLostBelow09'])
                self.assertEqual(int((core & (a < .5)).sum()), row['consensusOpaqueLostBelow05'])
                self.assertEqual(int(((z < .01) & (f < .01) & (a > .05)).sum()), row['agreedBackgroundLeakedOver005'])
                self.assertEqual(int(((z > .5) != (f > .5)).sum()), row['v2FullVsCropBinaryDiff'])
                self.assertGreater(row['consensusOpaqueLostBelow05'], 0)


if __name__ == '__main__':
    unittest.main()
