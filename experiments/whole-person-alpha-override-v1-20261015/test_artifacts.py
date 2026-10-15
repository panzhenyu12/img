"""Independent artifact checks for the single-factor alpha intervention."""
from pathlib import Path
import hashlib,json,unittest
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'experiments/whole-face-shape-v2-20261015/outputs-scoped-owner/user-face-contour'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

class AlphaOverrideArtifacts(unittest.TestCase):
    def test_frozen_sources_binary_and_raw_model_index(self):
        m=json.loads((HERE/'source-manifest.json').read_text())
        for p,d in m['sourceFiles'].items():self.assertEqual(sha(HERE/p),d,p)
        r=json.loads((HERE/'outputs/report.json').read_text())
        self.assertEqual(r['binarySha256'],m['binarySha256'])
        self.assertEqual(sha(HERE/'outputs'/('image-render-'+r['binarySha256'])),r['binarySha256'])
        a=json.loads((HERE/'inputs/manifest.json').read_text())
        self.assertEqual(sha(ROOT/a['source']),a['sourceSha256'])
        self.assertEqual(sha(HERE/'inputs/physical-alpha-f32le.bin'),a['rawSha256'])
        raw=np.fromfile(HERE/'inputs/physical-alpha-f32le.bin',dtype='<f4').reshape(1024,1024)
        self.assertTrue(np.array_equal(raw,np.load(ROOT/a['source'])[a['index']]))
        for row in r['rows']:
            d=HERE/'outputs'/f"{row['mode']}-{row['amount']}"
            self.assertEqual(sha(d/'candidate.png'),row['outputSha256'])
            self.assertEqual(sha(d/'graph.json'),row['graphSha256'])

    def test_alpha_changes_but_geometry_frozen_pixels_and_zero_remain_exact(self):
        r=json.loads((HERE/'outputs/report.json').read_text());source=np.asarray(Image.open(ROOT/r['source']).convert('RGBA'))
        raw=np.fromfile(HERE/'inputs/physical-alpha-f32le.bin',dtype='<f4').reshape(1024,1024)
        self.assertTrue(np.array_equal(source,np.asarray(Image.open(HERE/'outputs/raw-0/candidate.png'))))
        self.assertTrue(np.array_equal(np.asarray(Image.open(BASE/'natural-30/candidate.png')),np.asarray(Image.open(HERE/'outputs/control-30/candidate.png'))))
        for amount in [30,-30]:
            d=HERE/f'outputs/raw-{amount}/debug';base=BASE/f'natural-{amount}/debug'
            l,t,w,h=json.loads((d/'layers.json').read_text())['crop']
            alpha=np.fromfile(d/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
            self.assertTrue(np.array_equal(alpha,raw[t:t+h,l:l+w]))
            for p in ['flow.json','requested.json','source-xy-f32le.bin','frozen.png','head-component.png','owner-support.png']:
                self.assertEqual((d/p).read_bytes(),(base/p).read_bytes(),p)
            out=np.asarray(Image.open(d.parent/'candidate.png').convert('RGBA'))
            changed=np.asarray(Image.open(d/'changed.png'))>0;fixed=np.asarray(Image.open(d/'frozen.png'))>0
            self.assertTrue(np.array_equal(out[...,3],source[...,3]))
            self.assertTrue(np.array_equal(out[t:t+h,l:l+w][fixed|~changed],source[t:t+h,l:l+w][fixed|~changed]))
            roi=np.zeros(source.shape[:2],bool);roi[t:t+h,l:l+w]=True
            self.assertTrue(np.array_equal(out[~roi],source[~roi]))

if __name__=='__main__':unittest.main()
