"""Independent counterexample: transported edge may lie in unknown coverage.

This freezes and tests the pre-fix compositor. It intentionally documents a
failed protection property; it does not edit the experiment or production.
"""
from pathlib import Path
import importlib.util
import json
import sys
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location('frozen_occlusion', HERE / 'frozen/occlusion_hypotheses-before-unknown-fix.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)

shape = (12, 20)
old = np.zeros(shape)
new = np.zeros(shape)
source_border = np.zeros(shape, bool)
transported_border = np.zeros(shape, bool)
source_border[3:9, 3] = True
old[3:9, 3] = .5
# A finite translation carries a mixed boundary to a disjoint destination.
# The exact connectedness decision must not be bypassed by that destination.
new[3:9, 12] = .5
transported_border[3:9, 12] = True
allowed = np.ones(shape, bool)
hair = np.ones(shape)
cover, unknown, _, _ = subject.connected_coverage(new, old, hair, source_border, allowed)
edge = source_border | transported_border
hair_core = np.zeros(shape, bool)
pre_fix_writes = (edge & ~hair_core) | cover
assert unknown[3:9, 12].all()
assert not cover[3:9, 12].any()
assert (pre_fix_writes & unknown).sum() == 6
report = {
    'case': 'Transported mixed border in disconnected newly-covered component',
    'unknownPixels': int(unknown.sum()),
    'preFixUnknownWritePixels': int((pre_fix_writes & unknown).sum()),
    'expectedUnknownWritePixels': 0,
    'status': 'Confirmed pre-fix contract violation; no real-case unknown pixels',
}
from occlusion_hypotheses import write_domains
fixed_a, fixed_b = write_domains(edge, hair_core, cover, cover, unknown, unknown)
assert not (fixed_a & unknown).any()
assert not (fixed_b & unknown).any()
report['fixedUnknownWritesA'] = int((fixed_a & unknown).sum())
report['fixedUnknownWritesB'] = int((fixed_b & unknown).sum())
report['fixedScriptSha256'] = hashlib.sha256((HERE.parent / 'occlusion_hypotheses.py').read_bytes()).hexdigest()
before = HERE.parent / 'occlusion-before-unknown-write-gate'
after = HERE.parent / 'occlusion-corrected-prompts'
frozen_pngs = sorted(before.rglob('*.png'))
assert len(frozen_pngs) == 90
assert all(p.read_bytes() == (after / p.relative_to(before)).read_bytes() for p in frozen_pngs)
report['realPngFilesByteExactAfterFix'] = len(frozen_pngs)
(HERE / 'unknown-counterexample.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
