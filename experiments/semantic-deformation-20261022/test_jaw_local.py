"""Optional jaw evidence checks; ignored source artifacts are required."""
import json
import os
from pathlib import Path
import subprocess
import unittest

from prepare_jaw_input import (expected_targets, verify_sidecar, controller_request,
                               check_controller_output, ROOT)
from test_jaw_input import fixture


class JawLocalTests(unittest.TestCase):
    def test_existing_source_boundaries_are_recomputed_and_wrong_order_rejected(self):
        base = ROOT/'test/results/semantic-deformation-20261022'
        audit_path = base/'controller-source-audit/woman-audit.json'
        sidecar_path = base/'source-region-sidecars-v1/source-index-coordinate-v2/woman.json'
        if not audit_path.exists() or not sidecar_path.exists():
            self.skipTest('ignored source evidence unavailable')
        audit = json.loads(audit_path.read_text()); sidecar = json.loads(sidecar_path.read_text())
        graph = Path(next(p for p in sidecar['provenance']['input_sha256'] if p.endswith('/graph.json')))
        verify_sidecar(audit, sidecar, graph)
        sidecar['sides']['left']['direction_in_source_row_order'] *= -1
        with self.assertRaisesRegex(ValueError, 'boundary/order'):
            verify_sidecar(audit, sidecar, graph)

    def test_rust_cli_matches_independent_signed_polynomial(self):
        binary = Path(os.environ.get('JAW_CONTROLLER_BINARY',
            ROOT/'experiments/semantic-deformation-20261022/rust-controls/target/debug/jaw-arc-targets'))
        if not binary.exists():
            self.skipTest('jaw binary unavailable; set JAW_CONTROLLER_BINARY')
        args = fixture()
        for amount in [-50., 0., 50.]:
            wanted, gains, ids = expected_targets(*args, amount)
            request = controller_request(args[0], args[3], args[4], amount)
            completed = subprocess.run([str(binary.resolve())], input=json.dumps(request),
                                       text=True, capture_output=True, check=True)
            check_controller_output(json.loads(completed.stdout), args[0], wanted, gains, ids)


if __name__ == '__main__':
    unittest.main()
