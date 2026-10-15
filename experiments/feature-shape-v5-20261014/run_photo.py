"""Replay immutable cached photographs, graphs and Rust test instrumentation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QA = Path(os.environ.get("EYE_RIGID_QA_DIR", str(HERE / "qa")))
BASE = ROOT / "test/results/feature-svf-v4b-20261014/image-render-final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases = json.loads((HERE / "qa" / "cases.json").read_text())
    QA.mkdir(parents=True, exist_ok=True)
    if QA != HERE / "qa":
        for case in cases:
            case["output"] = str(QA / (case["label"] + "-candidate.png"))
            case["diagnostics"] = str(QA / (case["label"] + "-geometry.json"))
    wanted = set(sys.argv[1:])
    if wanted:
        cases = [row for row in cases if row["label"] in wanted]
    worker = Path(os.environ["EYE_RIGID_TEST_BINARY"])
    signature = sha(worker)
    baseline_signature = sha(BASE)
    rows = []
    for case in cases:
        source, graph, output, diag = (Path(case[k]) for k in ("input", "graph", "output", "diagnostics"))
        baseline = QA / (case["label"] + "-v4b.png")
        subprocess.run([str(BASE), "--input", str(source), "--graph", str(graph), "--output", str(baseline)], check=True, capture_output=True)
        env = dict(os.environ)
        env.update(EYE_RIGID_INPUT=str(source), EYE_RIGID_GRAPH=str(graph), EYE_RIGID_OUTPUT=str(output), EYE_RIGID_DIAGNOSTICS_OUT=str(diag))
        start = time.monotonic()
        command = [str(worker), "geometry::tests::eye_rigid_photographic_diagnostics", "--exact", "--ignored", "--nocapture"]
        result = subprocess.run(command, check=True, capture_output=True, env=env)
        (QA / (case["label"] + ".log")).write_bytes(result.stdout + result.stderr)
        geometry = json.loads(diag.read_text())
        params = json.loads(graph.read_text())["operations"][0]["params"]
        original = Image.open(source).convert("RGBA")
        w, h = original.size
        masks = params["semanticMasks"]
        points = [p for name in ("leftEye", "rightEye", "leftBrow", "rightBrow") for p in masks[name]["points"]]
        xs, ys = [p["x"] * w for p in points], [p["y"] * h for p in points]
        box = (max(0, int(min(xs) - 30)), max(0, int(min(ys) - 30)), min(w, int(max(xs) + 30)), min(h, int(max(ys) + 65)))
        tiles = [Image.open(p).convert("RGB").crop(box) for p in (source, baseline, output)]
        cw, ch = tiles[0].size
        sheet = Image.new("RGB", (cw * 3, ch + 24), "#222")
        draw = ImageDraw.Draw(sheet)
        for i, (tile, title) in enumerate(zip(tiles, ("Original", "V4b", "V5 candidate"))):
            sheet.paste(tile, (i * cw, 24))
            draw.text((i * cw + 5, 6), title + "  " + case["label"], fill="white")
        sheet.save(QA / (case["label"] + "-comparison.png"))
        sheet.resize((cw * 6, (ch + 24) * 2), Image.Resampling.NEAREST).save(QA / (case["label"] + "-comparison-2x.png"))
        rows.append(dict(case=case, sourceSha256=sha(source), graphSha256=sha(graph), baselineSha256=sha(baseline), outputSha256=sha(output), geometrySha256=sha(diag), diagnostics=geometry, renderWallSeconds=time.monotonic()-start, cropBox=box))
        assert sha(worker) == signature and sha(BASE) == baseline_signature
        print(case["label"], [{"name":r["name"],"area":r["detectedPolygonForwardAreaRatio"],"maxEdgeError":r["maxPolygonEdgeRelativeError"],"maxRigidErrorPx":r["maxRigidVertexErrorPixels"],"blocked":r["rigidSupportBlocked"],"poseScale":r.get("poseScale"),"transition":r["rigidTransitionPixels"]} for r in geometry], flush=True)
    report = dict(scope="Frozen actual native public apply_graph_json including final opacity/mask composite and instrumented final Rust inverse mapper. Photograph crops are visual evidence, not a segmented iris truth.", testBinary=str(worker), testBinarySha256=signature, baselineBinary=str(BASE), baselineBinarySha256=baseline_signature, rows=rows)
    (QA / ("photo-report-" + ("selected" if wanted else "all") + ".json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
