"""Recorded-artifact integrity, geometry and composition checks; not aesthetic tests."""
from pathlib import Path
import hashlib,json,unittest
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

class ScopedOwnerArtifacts(unittest.TestCase):
    def test_sources_and_immutable_binary_match(self):
        manifest=json.loads((HERE/'scoped-owner-source-manifest.json').read_text())
        for name,digest in manifest['sourceFiles'].items():self.assertEqual(sha(HERE/name),digest,name)
        out=HERE/'outputs-scoped-owner/user-face-contour'
        report=json.loads((out/'report.json').read_text())
        self.assertEqual(report['binarySha256'],manifest['binarySha256'])
        self.assertEqual(sha(out/('image-render-'+report['binarySha256'])),report['binarySha256'])
        self.assertEqual(sha(ROOT/report['source']),report['sourceSha256'])
        for row in report['rows']:
            d=out/f"{row['preset']}-{row['amount']}"
            self.assertEqual(sha(d/'candidate.png'),row['outputSha256'])
            self.assertEqual(sha(d/'graph.json'),row['graphSha256'])

    def test_zero_roi_coverage_and_recorded_composition(self):
        out=HERE/'outputs-scoped-owner/user-face-contour'
        report=json.loads((out/'report.json').read_text())
        source=np.asarray(Image.open(ROOT/report['source']).convert('RGBA'))
        self.assertTrue(np.array_equal(source,np.asarray(Image.open(out/'natural-0/candidate.png'))))
        for amount in [-30,30]:
            d=out/f'natural-{amount}';debug=d/'debug';output=np.asarray(Image.open(d/'candidate.png').convert('RGBA'))
            left,top,w,h=json.loads((debug/'layers.json').read_text())['crop']
            roi=np.zeros(source.shape[:2],bool);roi[top:top+h,left:left+w]=True
            self.assertTrue(np.array_equal(source[~roi],output[~roi]))
            self.assertTrue(np.array_equal(source[...,3],output[...,3]))
            crop=source[top:top+h,left:left+w]
            self.assertTrue(np.array_equal(crop,np.asarray(Image.open(debug/'crop-source.png'))))
            alpha=np.fromfile(debug/'moving-physical-f32le.bin',dtype='<f4').reshape(h,w)
            target=np.fromfile(debug/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w)
            xy=np.fromfile(debug/'source-xy-f32le.bin',dtype='<f4').reshape(h,w,2)
            sample=np.fromfile(debug/'sample-premult-rgba-f32le.bin',dtype='<f4').reshape(h,w,4)
            changed=np.asarray(Image.open(debug/'changed.png'))>0
            frozen=np.asarray(Image.open(debug/'frozen.png'))>0
            for a in [alpha,target]:self.assertTrue(np.all(np.isfinite(a)&(a>=0)&(a<=1)))
            self.assertTrue(np.all(np.isfinite(xy)))
            self.assertFalse(np.any(changed&frozen))
            self.assertFalse(np.any(changed&(alpha==0)&(target==0)))
            # Source coordinates can be evaluated and then discarded when both
            # old/new coverage are zero; the public pixel must remain exact.
            self.assertTrue(np.array_equal(crop[~changed],output[top:top+h,left:left+w][~changed]))
            bg=np.asarray(Image.open(debug/'reconstructed-background.png').convert('RGBA')).astype(np.float32)
            weight=bg[...,3]/255*(1-target)
            denom=sample[...,3]+weight
            pred=np.clip(np.floor((sample[...,:3]+bg[...,:3]*weight[...,None])/np.maximum(denom[...,None],1e-6)+.5),0,255)
            delta=np.abs(pred-output[top:top+h,left:left+w,:3].astype(float))
            self.assertLessEqual(float(delta[changed].max()),1)
            # Scoping ownership was causally separate from this one-head output.
            initial=np.asarray(Image.open(HERE/f'outputs/user-face-contour/natural-{amount}/candidate.png'))
            self.assertTrue(np.array_equal(output,initial))

if __name__=='__main__':unittest.main()
