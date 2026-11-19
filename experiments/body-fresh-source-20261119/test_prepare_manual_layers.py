"""Source-stage contracts without model weights or historical entry points."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import prepare_manual_layers as prepare


class ManualSourceStagesTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.source=self.base/'source.png'
        yy,xx=np.mgrid[:32,:40]
        self.rgb=np.stack([xx*5,yy*7,(xx+yy)*3],axis=-1).astype(np.uint8)
        Image.fromarray(self.rgb).save(self.source)
        self.request={'version':'explicit-source-contour/1','source':str(self.source),
          'sourceSHA256':prepare.sha(self.source),'role':'stationary-front',
          'polygons':[[[10,8],[29,8],[29,24],[10,24]]],'uncertaintyPixels':3,
          'provenance':'manual-source-image-contour'}
        self.save_request()

    def save_request(self):
        (self.base/'front-request.json').write_text(json.dumps(self.request))

    def prepare_matte(self):
        prepare.front_prepare(self.base,self.source)
        self.target=self.base/'source-front-matte-selected-domain'
        tri=np.array(Image.open(self.target/'trimap.png'))
        raw=np.full(tri.shape,.375,dtype=np.float32)
        alpha=raw.copy();alpha[tri==0]=0;alpha[tri==255]=1
        weight=self.base/'fake-weights';weight.write_bytes(b'test-only')
        with patch.object(prepare,'MEMATTE_WEIGHTS',weight), \
             patch.object(prepare,'predict_matte',return_value=(raw,alpha,{'testStub':True})) as infer, \
             contextlib.redirect_stdout(io.StringIO()):
            prepare.front_matte(self.base,self.source)
        np.testing.assert_array_equal(infer.call_args.args[0],self.rgb)
        np.testing.assert_array_equal(infer.call_args.args[1],tri)
        return raw,alpha

    def test_numeric_import_does_not_activate_model_or_historical_pipeline(self):
        code="import sys; import prepare_manual_layers; assert not any(n in sys.modules for n in ['torch','worker','run_mematte','prepare_explicit_front','prepare_manual_front']); assert not any('/.run/' in p for p in sys.path)"
        run=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,text=True,capture_output=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_changed_source_request_and_trimap_are_rejected_before_inference(self):
        prepare.front_prepare(self.base,self.source)
        with self.assertRaises(FileExistsError):
            prepare.front_prepare(self.base,self.source)
        self.request['uncertaintyPixels']=4;self.save_request()
        with patch.object(prepare,'predict_matte') as infer:
            with self.assertRaisesRegex(ValueError,'changed after freeze'):
                prepare.front_matte(self.base,self.source)
            infer.assert_not_called()
        self.request['uncertaintyPixels']=3;self.save_request()
        target=self.base/'source-front-matte-selected-domain'
        tri=np.array(Image.open(target/'trimap.png'));tri[0,0]=128
        Image.fromarray(tri).save(target/'trimap.png')
        with self.assertRaisesRegex(ValueError,'changed after freeze'):
            prepare.front_matte(self.base,self.source)

    def test_matte_preserves_raw_and_official_results_and_refuses_overwrite(self):
        raw,alpha=self.prepare_matte()
        np.testing.assert_array_equal(np.load(self.target/'alpha-raw.npy'),raw)
        np.testing.assert_array_equal(np.load(self.target/'alpha.npy'),alpha)
        report=json.loads((self.target/'report.json').read_text())
        self.assertTrue(report['sourceOnly']);self.assertFalse(report['motionRead'])
        self.assertFalse(report['automaticGeneralizationClaim'])
        self.assertEqual(report['sourceSHA256'],prepare.sha(self.source))
        with self.assertRaises(FileExistsError):
            prepare.front_matte(self.base,self.source)

    def test_underlay_reads_only_declared_alpha_and_preserves_known_context(self):
        _,alpha=self.prepare_matte();selected=alpha>0
        result=self.rgb.copy();result[selected]=[45,63,87]
        with patch.object(prepare,'inpaint',return_value=(Image.fromarray(result),{'testStub':True})) as infer, \
             contextlib.redirect_stdout(io.StringIO()):
            prepare.underlay(self.base,self.source)
        np.testing.assert_array_equal(infer.call_args.args[1],selected)
        actual=np.array(Image.open(self.base/'source-underlay/underlay.png'))
        np.testing.assert_array_equal(actual[~selected],self.rgb[~selected])
        report=json.loads((self.base/'source-underlay/report.json').read_text())
        self.assertEqual(report['frontAlphaSHA256'],prepare.sha(self.target/'alpha.npy'))
        self.assertEqual(report['outsideMaskChanged'],0)

    def test_underlay_rejects_repainting_known_context(self):
        self.prepare_matte();result=self.rgb.copy();result[0,0]=255
        with patch.object(prepare,'inpaint',return_value=(Image.fromarray(result),{})):
            with self.assertRaisesRegex(ValueError,'repainted observed source context'):
                prepare.underlay(self.base,self.source)


if __name__=='__main__':
    unittest.main()
