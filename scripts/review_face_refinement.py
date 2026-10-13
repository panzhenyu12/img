#!/usr/bin/env python3
"""Replay all nine face controls against an explicit, verified old binary.

Requires the local fresh detector cache (not shipped model inference). This
produces evidence, protection checks and timings; it never awards quality
scores automatically. All failures and source/control identities are retained.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["faceWidth", "temples", "cheekbones", "jaw", "foreheadWidth",
        "chinWidth", "chinPoint", "forehead", "chin"]
NAMES = ["fixture-portrait", "portrait-woman", "portrait-makeup",
         "portrait-roll", "portrait-hair", "portrait-glasses-street"]
FEATURES = ["leftEye", "rightEye", "leftBrow", "rightBrow", "lips", "teeth",
            "nose", "noseBridge", "noseWings"]


def polygon_interior(mask, size):
    """Pixel-center point-in-polygon; protects the actual feature, without
    pretending its enclosing rectangle/ellipse is the observed anatomy."""
    width, height = size
    xx, yy = np.meshgrid((np.arange(width) + .5) / width,
                         (np.arange(height) + .5) / height)
    selected = np.zeros((height, width), dtype=bool)
    points = mask.get("points", [])
    for n, a in enumerate(points):
        b = points[(n + 1) % len(points)]
        if abs(b["y"] - a["y"]) < 1e-12:
            continue
        selected ^= ((a["y"] > yy) != (b["y"] > yy)) & (xx < (b["x"] - a["x"]) * (yy - a["y"]) / (b["y"] - a["y"]) + a["x"])
    return selected


def coverage(mask, size):
    """Match Rust raster sampling at pixel centers, including its float math."""
    width, height = size
    mw, mh = mask["width"], mask["height"]
    a = np.asarray(mask["data"], np.float32).reshape(mh, mw) / np.float32(255)
    xx = np.clip((np.arange(width, dtype=np.float32) + .5) / width * mw - .5, 0, mw - 1)
    yy = np.clip((np.arange(height, dtype=np.float32) + .5) / height * mh - .5, 0, mh - 1)
    x0, y0 = xx.astype(int), yy.astype(int)
    x1, y1 = np.minimum(x0 + 1, mw - 1), np.minimum(y0 + 1, mh - 1)
    tx, ty = xx - x0, yy - y0
    top = a[y0[:, None], x0] * (1 - tx) + a[y0[:, None], x1] * tx
    bottom = a[y1[:, None], x0] * (1 - tx) + a[y1[:, None], x1] * tx
    return top * (1 - ty[:, None]) + bottom * ty[:, None]


def render(binary, source, output, graph):
    begin = time.perf_counter()
    subprocess.run([str(binary), "--input", str(source), "--output", str(output),
                    "--graph", str(graph)], check=True, capture_output=True)
    return round((time.perf_counter() - begin) * 1000, 2)


def comparison(source, before, after, box, path, title, scale=1, baseline_label="ea5b404 baseline"):
    images = [Image.open(p).convert("RGB").crop(box) for p in (source, before, after)]
    w, h = images[0].size
    out = Image.new("RGB", (w * 3 * scale, h * scale + 30), "#171c22")
    d = ImageDraw.Draw(out)
    for n, (image, label) in enumerate(zip(images, ["Original", baseline_label, "Refined"])):
        out.paste(image.resize((w*scale, h*scale), Image.Resampling.NEAREST), (n*w*scale, 30))
        d.text((n*w*scale + 5, 7), label + (" | " + title if n == 2 else ""), fill="white")
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / ".run/quality-baseline/image-render")
    # Use an immutable copy: cargo test may replace the regular target binary
    # while building CARGO_BIN_EXE for integration tests, even with a filter.
    ap.add_argument("--candidate", type=Path, default=ROOT / ".run/face-refinement-candidate-image-render")
    ap.add_argument("--baseline-label", default="ea5b404")
    ap.add_argument("--allow-neck-lighting", action="store_true", help="Explicitly allow observed semantic neck illumination changes in contractions; real scene background remains fixed")
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--keys", nargs="+", choices=KEYS, default=KEYS)
    ap.add_argument("--values", nargs="+", type=int, default=[-60,60])
    ap.add_argument("--output", type=Path, default=ROOT / "test/results/face-refinement-v3")
    ap.add_argument("--doc-output", type=Path, default=ROOT / "doc/qa/face-refinement-v3")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.doc_output.mkdir(parents=True, exist_ok=True)
    # Snapshot both inputs to read-only, content-named executables. Builds in
    # any target directory cannot replace these files during an evaluation.
    original_binary_paths = {"baseline": str(args.baseline), "candidate": str(args.candidate)}
    for name in ("baseline", "candidate"):
        binary = getattr(args, name)
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        snapshot = args.output / (name + "-" + digest + "-image-render")
        if not snapshot.exists():
            shutil.copy2(binary, snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"binary changed while copying snapshot: {binary}")
        snapshot.chmod(0o555)
        setattr(args, name, snapshot)
    report = {"baselineCommit": args.baseline_label, "neckLightingContract": args.allow_neck_lighting, "baselineSha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              "candidateSha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
              "timingIncludes": "CLI image/graph decoding, validation, kernel and PNG output; concurrent machine work may affect latency",
              "sourceBinaryPaths": original_binary_paths,
              "snapshotBinaryPaths": {"baseline": str(args.baseline), "candidate": str(args.candidate)},
              "rating": "No automatic rating; all real-photo outputs need visual assessment", "cases": []}
    def verify_binaries():
        for binary, key in [(args.baseline, "baselineSha256"), (args.candidate, "candidateSha256")]:
            if hashlib.sha256(binary.read_bytes()).hexdigest() != report[key]:
                report["invalidated"] = True
                (args.doc_output / "replay-report.json").write_text(json.dumps(report, indent=2)+"\n")
                raise SystemExit(f"binary changed during replay: {binary}; whole run is invalid")
    for name in args.names:
        seed = ROOT / f"test/results/face-core-fresh/{name}-faceWidth-60.json"
        source = ROOT / f"test/results/face-core-fresh/{name}-source.png"
        graph = json.loads(seed.read_text())
        params = graph["operations"][0]["params"]
        original = np.asarray(Image.open(source).convert("RGBA"))
        h, w = original.shape[:2]
        masks = {k: coverage(m, (w, h)) for k, m in params["semanticMasks"].items() if m.get("shape") == "raster"}
        feature_interiors = {k: polygon_interior(m, (w, h)) for k, m in params["semanticMasks"].items() if k in FEATURES and m.get("shape") == "polygon"}
        person = coverage(params["subjectMask"], (w,h)) if params.get("subjectMask",{}).get("shape")=="raster" else None
        protect = coverage(params["protectMask"], (w,h)) if params.get("protectMask",{}).get("shape")=="raster" else None
        outline = params["semanticMasks"]["faceContour"]["points"]
        xs, ys = [p["x"] * w for p in outline], [p["y"] * h for p in outline]
        pad = max(xs) - min(xs)
        box = (max(0, int(min(xs)-pad*.16)), max(0, int(min(ys)-pad*.12)),
               min(w, int(max(xs)+pad*.16)), min(h, int(max(ys)+pad*.22)))
        region = params["region"]
        xx, yy = np.meshgrid((np.arange(w)+.5)/w, (np.arange(h)+.5)/h)
        outside = ((xx < region["x"]) | (xx > region["x"]+region["width"]) |
                   (yy < region["y"]) | (yy > region["y"]+region["height"]))
        for key in args.keys:
            for value in args.values:
                verify_binaries()
                case = f"{name}-{key}-{value}"
                params["adjustments"] = {key: value}
                graph_path = args.output / "current-graph.json"
                graph_path.write_text(json.dumps(graph, separators=(",", ":")))
                before, after = args.output / (case+"-before.png"), args.output / (case+"-after.png")
                old_ms = render(args.baseline, source, before, graph_path)
                verify_binaries()
                new_ms = render(args.candidate, source, after, graph_path)
                verify_binaries()
                result = np.asarray(Image.open(after).convert("RGBA"))
                changed = np.any(result != original, axis=2)
                row = {"name": name, "key": key, "value": value, "seedGraph": str(seed.relative_to(ROOT)),
                       "beforeMs": old_ms, "afterMs": new_ms, "outsideChanges": int(changed[outside].sum()),
                       "protectedChanges": {k: int(changed[a >= .5].sum()) for k,a in masks.items() if k in ("hair", "ears", "accessories")}}
                row["featureInteriorChanges"] = {k: int(changed[a].sum()) for k, a in feature_interiors.items()}
                # Pure contraction preserves all original head-free output.
                contracts = (value > 0) == (key in ("faceWidth", "temples", "cheekbones", "jaw", "chinPoint"))
                row["oldHeadExteriorChanges"] = int(changed[masks["face"] <= 1e-6].sum())
                neck = masks.get("neck", np.zeros((h,w)))
                head_free = masks["face"] <= 1e-6
                row["semanticNeckChanges"] = int(changed[head_free & (neck > .8)].sum())
                row["headAndNeckExteriorChanges"] = int(changed[head_free & (neck <= .8)].sum())
                row["userProtectedChanges"] = int(changed[protect >= 1-1e-6].sum()) if protect is not None else 0
                row["sceneBackgroundChanges"] = int(changed[head_free & (person <= 1e-6)].sum()) if person is not None else None
                row["pureContraction"] = contracts
                contraction_protected = (row["headAndNeckExteriorChanges"] == 0 and row["sceneBackgroundChanges"] == 0) if args.allow_neck_lighting else row["oldHeadExteriorChanges"] == 0
                row["hardChecksPassed"] = (row["outsideChanges"] == 0 and not any(row["protectedChanges"].values())
                    and not any(row["featureInteriorChanges"].values())
                    and row["userProtectedChanges"] == 0 and (not contracts or contraction_protected))
                report["cases"].append(row)
                comparison(source, before, after, box, args.output/(case+"-comparison.png"), f"{key}={value}", baseline_label=args.baseline_label+" baseline")
                if name == "portrait-makeup" and key in ("faceWidth", "chin", "jaw"):
                    comparison(source, before, after, box, args.doc_output/(case+"-comparison.png"), f"{key}={value}", baseline_label=args.baseline_label+" baseline")
                print(case, row["hardChecksPassed"], f"{old_ms:.0f}->{new_ms:.0f} ms", flush=True)
        report_path = args.doc_output / "replay-report.json"
        report_path.write_text(json.dumps(report, indent=2)+"\n")
    (args.output / "current-graph.json").unlink(missing_ok=True)
    verify_binaries()
    report["finalBaselineSha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    report["finalCandidateSha256"] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    (args.doc_output / "replay-report.json").write_text(json.dumps(report, indent=2)+"\n")
    if not all(c["hardChecksPassed"] for c in report["cases"]):
        raise SystemExit("hard protection regression; failures retained in replay-report.json")


if __name__ == "__main__":
    main()
