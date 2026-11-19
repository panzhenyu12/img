"""Lazy access to the existing local MEMatte and LaMa runtimes.

Importing source preparation does not import Torch, a segmentation proposal,
or any historical experiment entry point. Model/vendor files remain external
local prerequisites; this module neither downloads nor copies them.
"""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
MEMATTE_WEIGHTS = ROOT/'.models/mematte-20261020/MEMatte-AIM500-ViTS.pth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _activate_runtime():
    # Preserve the established local inference dependency order. Numerical
    # source-only stages do not need or activate any of these directories.
    paths = [ROOT/'.run/mematte-20261117-deps',
             ROOT/'.run/body-source-20261117-deps',
             ROOT/'experiments/sam2matting-tiny-probe-20261015/deps',
             ROOT/'.run/algorithm-tools',
             ROOT/'.run/mematte-20261020/detectron2-vendor/build/lib.macosx-10.15-x86_64-cpython-311']
    sys.path[:0] = [str(path) for path in paths]


def predict_matte(rgb, trimap, weights=MEMATTE_WEIGHTS):
    _activate_runtime()
    sys.path.insert(0, str(ROOT/'experiments/body-matting-model-review-20261020'))
    from run_mematte import official_prediction
    return official_prediction(rgb, trimap, weights)


def inpaint(image, mask):
    _activate_runtime()
    sys.path.insert(0, str(ROOT/'services/background-worker'))
    from worker import LamaWorker
    return LamaWorker(threads=3).inpaint(
        image, mask, max_side=max(mask.shape), context_mode='full',
        mask_dilation=0, seam_blend='none')
