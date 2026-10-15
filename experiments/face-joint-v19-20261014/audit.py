"""Read frozen original/input/debug only; no detection or quality score."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_user_face_contour import raster_coverage

base = ROOT / "test/results/user-face-contour-20261014/1024"
out = base / "face-joint-v19-candidate"
params = json.loads((base / "face-graph.json").read_text())["operations"][0]["params"]
source = np.asarray(Image.open(base / "canonical-source.png").convert("RGBA"))
masks = {k: raster_coverage(v, 1024, 1024) for k, v in params["semanticMasks"].items() if v.get("shape") == "raster"}
person = raster_coverage(params["subjectMask"], 1024, 1024)
report = {"scope": "local RGB-refined alpha from parser seed, no physical gold or automatic rating", "rows": []}
for amount in [100, -100, 30, -30]:
    debug = out / f"debug{amount}"
    geometry = json.loads((debug / "flow-geometry.json").read_text())
    ox, oy = geometry["origin"]
    old = np.asarray(Image.open(debug / "source-physical-alpha.png")) / 255
    new = np.asarray(Image.open(debug / "full-transported-physical-alpha.png")) / 255
    target = np.asarray(Image.open(out / f"faceWidth{amount}.png").convert("RGBA"))
    changed = np.any(source != target, axis=2)
    row = {"amount": amount, "physicalComponents": [], "outerRows": [], "sceneChanged": int((changed & (person <= 1e-6) & (masks["face"] <= 1e-6)).sum()), "allPixelsY700PlusChanged": int(changed[700:].sum())}
    for alpha in [old, new]:
        components, count = label(alpha > .5)
        sizes = np.bincount(components.ravel())[1:]
        row["physicalComponents"].append({"count": int(count), "sizesDescending": sorted([int(v) for v in sizes], reverse=True)})
    for y in [495, 525, 535, 545, 560, 600, 620, 700, 716]:
        spans = []
        for alpha in [old, new]:
            xx = np.nonzero(alpha[y - oy] > .5)[0]
            spans.append(None if not len(xx) else [int(xx[0] + ox), int(xx[-1] + ox)])
        row["outerRows"].append({"y": y, "source": spans[0], "transported": spans[1]})
    report["rows"].append(row)
(out / "read-only-audit-replay.json").write_text(json.dumps(report, indent=2) + "\n")
