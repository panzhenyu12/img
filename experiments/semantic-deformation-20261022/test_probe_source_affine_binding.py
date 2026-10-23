"""No optimization is needed to test the source identity contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_source_affine_bending as probe
from jaw_target_profile import _geometry_sha


class AffineSourceBindingTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source.png';source.write_bytes(b'frozen source pixels')
        mask=root/'mask.png';mask.write_bytes(b'frozen mask pixels')
        geometry={'contour':[[1,2],[3,4]]}
        proof=root/'report.json'
        proof.write_text(json.dumps({'source_image':str(source),'source_geometry':geometry,
                                    'hashes':{str(source):probe.sha(source)}}))
        protocol=root/'source-protocol.json'
        protocol.write_text(json.dumps({'source_geometry':geometry,
            'inputs_sha256':{str(p):probe.sha(p) for p in [source,mask,proof]}}))
        baseline={'source_protocol':str(protocol),'source_protocol_sha256':probe.sha(protocol),
                  'source_geometry':geometry,'source_geometry_sha256':_geometry_sha(geometry)}
        return baseline,source,mask,protocol,proof

    def test_unrelated_image_is_rejected_before_assembly_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            baseline=root/'baseline.json'
            baseline.write_text(json.dumps({'controller':'probe-plateau-retain-half-source-bend-v1',
                                            'report':{'accepted_numerically':True}}))
            source=root/'unrelated.png';source.write_bytes(b'unrelated')
            output=root/'result'
            with patch.object(probe,'assemble',side_effect=AssertionError('must not solve')) as solve:
                with self.assertRaisesRegex(ValueError,'source protocol'):
                    probe.run(baseline,source,output)
            solve.assert_not_called()
            self.assertFalse(output.exists())

    def test_original_role_is_accepted_but_hash_bound_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline,source,mask,protocol,proof=self.fixture(Path(folder))
            bindings=probe.check_source_binding(baseline,source)
            for path in [source,protocol,proof]:
                self.assertEqual(bindings[str(path.resolve())],probe.sha(path))
            with self.assertRaisesRegex(ValueError,'image role'):
                probe.check_source_binding(baseline,mask)

    def test_modified_protocol_or_source_is_rejected(self):
        for changed in ['protocol','source','proof']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as folder:
                baseline,source,_,protocol,proof=self.fixture(Path(folder))
                path={'protocol':protocol,'source':source,'proof':proof}[changed]
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'changed'):
                    probe.check_source_binding(baseline,source)

    def test_different_geometry_cannot_reuse_the_frozen_image_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline,source,*_=self.fixture(Path(folder))
            baseline['source_geometry']={'contour':[[9,8],[7,6]]}
            baseline['source_geometry_sha256']=_geometry_sha(baseline['source_geometry'])
            with self.assertRaisesRegex(ValueError,'geometry'):
                probe.check_source_binding(baseline,source)

    def test_snapshot_preserves_direct_and_transitive_algorithm_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,bindings=probe.snapshot_local_code(Path(folder))
            records=json.loads(manifest.read_text())['files']
            by_name={Path(row['source']).name:row for row in records}
            for name in ['probe_source_affine_bending.py','source_affine_bending_reference.py',
                         'probe_jaw_retained_bend.py','probe_jaw_plateau.py','jaw_curve_qp.py',
                         'jaw_bilateral.py','jaw_collocation.py','jaw_target_profile.py',
                         'interface_bend_oracle.py','measure_jaw_preference_start.py']:
                row=by_name[name]
                self.assertEqual(Path(row['source']).read_bytes(),Path(row['snapshot']).read_bytes())
                self.assertEqual(bindings[row['source']],bindings[row['snapshot']])


if __name__=='__main__':unittest.main()
