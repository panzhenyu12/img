"""Exercise the real export CLI/loader with tiny synthetic source identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jaw_target_profile import load_profile
from test_jaw_curve_qp import curved


class ExportJawCurveTests(unittest.TestCase):
    def test_cli_freezes_sources_loads_in_adapter_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); geometry = curved()
            image = root/'source.png'; image.write_bytes(b'synthetic hash-bound image identity; no image decode')
            evidence = root/'evidence.json'; evidence.write_text(json.dumps({'geometry': geometry}))
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            protocol = root/'protocol.json'
            protocol.write_text(json.dumps({'source_geometry': geometry,
                'inputs_sha256': {str(p): sha(p) for p in [image, evidence]}}))
            output = root/'candidate'
            command = [sys.executable, str(Path(__file__).with_name('export_jaw_curve_profile.py')),
                       '--protocol', str(protocol), '--amount', '60', '--smoothing', '8',
                       '--node-spacing', '5', '--window-radius', '8', '--bilateral-weight', '1', '--output', str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            summary = json.loads(completed.stdout)
            profile = load_profile(output/'profile.json', geometry, 60.)
            self.assertEqual(summary['profile_sha256'], sha(output/'profile.json'))
            self.assertLess(abs(profile['verification']['closed_polygon_area_error_px2']), 1e-8)
            self.assertEqual(profile['verification']['adapter_max_target_error_px'], 0.)
            for field in ['qp_source', 'qp_report', 'export_source', 'bilateral_source']:
                path = Path(profile[field+'_path'])
                self.assertTrue(path.is_relative_to(output.resolve()))
                self.assertEqual(profile[field+'_sha256'], sha(path))
            self.assertEqual(profile['source_protocol_sha256'], sha(profile['source_protocol']))
            self.assertEqual(profile['source_protocol_sha256'], sha(protocol))
            self.assertEqual(profile['parameters']['smoothing_length_px'], 8.)
            self.assertEqual(profile['parameters']['bilateral_weight'], 1.)
            self.assertEqual(profile['report']['bilateral_coupling']['weight'], 1.)
            manifest = json.loads(Path(profile['executed_code_manifest']).read_text())
            helper = next(row for row in manifest if Path(row['snapshot']).name == 'jaw_bilateral.py')
            self.assertEqual(helper['sha256'], sha(profile['bilateral_source_path']))
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('refusing to overwrite', again.stderr)
            self.assertEqual(summary['profile_sha256'], sha(output/'profile.json'))
            image.write_bytes(b'changed source identity')
            with self.assertRaisesRegex(ValueError, 'source image or evidence changed'):
                load_profile(output/'profile.json', geometry, 60.)


if __name__ == '__main__':
    unittest.main()
