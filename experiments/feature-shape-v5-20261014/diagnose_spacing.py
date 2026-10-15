"""Single +30 source-pose causal replay; no production artifacts are modified."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("EYE_ELASTIC_DIAGNOSTIC_OUT", str(HERE / "spacing30-causal")))
SOURCE = ROOT / "test/results/user-face-contour-20261014/1024/canonical-source.png"
GRAPH = HERE / "qa/user-eyeSpacing30-graph.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    worker = Path(os.environ["EYE_ELASTIC_DIAGNOSTIC_BINARY"])
    reference = json.loads(GRAPH.read_text())
    variants = {"fixed-nose": reference, "no-nose-boundary": copy.deepcopy(reference)}
    for name in ["noseRoot", "noseBridge"]:
        variants["no-nose-boundary"]["operations"][0]["params"]["semanticMasks"].pop(name)
    rows = []
    for name, graph in variants.items():
        path = OUT / (name + "-graph.json")
        path.write_text(json.dumps(graph, indent=2) + "\n")
        env = dict(os.environ, EYE_RIGID_INPUT=str(SOURCE), EYE_RIGID_GRAPH=str(path),
                   EYE_RIGID_OUTPUT=str(OUT / (name + ".png")),
                   EYE_RIGID_DIAGNOSTICS_OUT=str(OUT / (name + "-geometry.json")),
                   EYE_ELASTIC_MAP_OUT=str(OUT / (name + "-mapping.json")))
        command = [str(worker), "geometry::tests::eye_rigid_photographic_diagnostics", "--exact", "--ignored", "--nocapture"]
        start = time.monotonic()
        proc = subprocess.run(command, env=env, check=True, capture_output=True)
        (OUT / (name + ".log")).write_bytes(proc.stdout + proc.stderr)
        rows.append({"name": name, "wallSeconds": time.monotonic() - start,
                     "graphSha256": sha(path), "outputSha256": sha(OUT / (name + ".png")),
                     "mappingSha256": sha(OUT / (name + "-mapping.json"))})
        print(name, flush=True)
    box = (380, 310, 635, 440)
    images = [Image.open(path).convert("RGB").crop(box) for path in
              [SOURCE, OUT / "fixed-nose.png", OUT / "no-nose-boundary.png"]]
    width, height = images[0].size
    result = Image.new("RGB", (width * 3, height + 24), "#222")
    draw = ImageDraw.Draw(result)
    for i, (image, title) in enumerate(zip(images, ["Original", "+30 fixed nose", "+30 ablate nose boundary"])):
        result.paste(image, (i * width, 24))
        draw.text((i * width + 5, 6), title, fill="white")
    result.save(OUT / "causal-comparison.png")
    result.resize((width * 6, (height + 24) * 2), Image.Resampling.NEAREST).save(OUT / "causal-comparison-2x.png")
    report = {"scope": "Actual public image pipeline with test-only inverse mapper instrumentation; only eyeSpacing=+30. Second variant removes two static nasal boundary polygons, leaving requested eye pose and original eye envelope unchanged. This is a causal ablation, not a quality candidate.",
              "source": str(SOURCE), "sourceSha256": sha(SOURCE), "binary": str(worker),
              "binarySha256": sha(worker), "cropBox": box, "rows": rows}
    (OUT / "replay.json").write_text(json.dumps(report, indent=2) + "\n")

if __name__ == "__main__":
    main()
