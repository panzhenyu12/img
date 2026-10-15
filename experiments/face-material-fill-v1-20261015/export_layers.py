"""Export exact frozen V36 layers, including samples skipped by fixed output masks.

Diagnostic-only source must be built explicitly before running. Never consumes
the shared production target. Decoded output must equal the frozen V36 image.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import argparse
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
built = ROOT / '.run/material-diagnostic-target/release/image-render'
binary = HERE / ('image-render-' + sha(built))
if not binary.exists():
    shutil.copy2(built, binary)
parser = argparse.ArgumentParser()
parser.add_argument('--photo', choices=['portrait-woman','portrait-hair','user-face-contour'], default='portrait-woman')
parser.add_argument('--amounts', type=int, nargs='+', default=[100,-100])
args = parser.parse_args()
source = ROOT / {'portrait-woman':'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png',
                 'portrait-hair':'test/results/face-core-review/portrait-hair-source.png',
                 'user-face-contour':'test/results/user-face-contour-20261014/1024/canonical-source.png'}[args.photo]
for amount in args.amounts:
    baseline = ROOT / f'experiments/face-source-curve-v36-20261015/outputs/{args.photo}-{amount}'
    prefix = 'woman' if args.photo == 'portrait-woman' else args.photo
    dest = HERE / f'{prefix}-{amount}-full'
    dest.mkdir(exist_ok=False)
    shutil.copy2(baseline / 'graph.json', dest / 'graph.json')
    run = subprocess.run([str(binary), '--input', str(source), '--graph', str(dest / 'graph.json'), '--output', str(dest / 'candidate.png')],
                         env={**os.environ, 'FACE_LAYERS_DEBUG_DIR': str(dest / 'layers')}, capture_output=True)
    (dest / 'render.log').write_bytes(run.stdout + run.stderr)
    run.check_returncode()
    exact = np.array_equal(np.asarray(Image.open(dest / 'candidate.png')), np.asarray(Image.open(baseline / 'candidate.png')))
    assert exact, f'Diagnostic changes output: {amount}'
    (dest / 'provenance.json').write_text(json.dumps({
        'source': str(source), 'sourceSha256': sha(source), 'binary': str(binary),
        'binarySha256': sha(binary), 'graphSha256': sha(dest / 'graph.json'),
        'baseline': str(baseline / 'candidate.png'), 'diagnosticRGBAExact': exact,
        'diagnosticSourceSha256': sha(HERE / 'core/src/face.rs'),
        'layers': {p.name: sha(p) for p in sorted((dest / 'layers').iterdir())},
        'inverseCoordinates': 'H,W,2 little-endian f32 crop-local source pixel index (no +0.5); full sample ignores output freeze/ROI/selection skip but uses unchanged field',
    }, indent=2))
    print(str(dest), flush=True)
