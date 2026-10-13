"""Fixed-photo replay for local redness control, separate from skin smoothing.

Visual scores are manual. Luma/detail checks describe regressions and do not
turn changes into an aesthetic score. Every run verifies immutable binaries.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from check_skin_acne_quality import SAMPLES, lesion_metrics
from check_skin_detail_quality import ROOT, comparison, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--label", default="skin-redness-v5")
    args = parser.parse_args()
    out = ROOT / "test/results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    binaries = {"before": args.reference.resolve(), "after": args.candidate.resolve()}
    hashes = {name: sha(binary) for name, binary in binaries.items()}
    rows = []
    cases = {"zero": 0, "gentle": -40, "strong": -100, "red": 40}
    sources = []
    cache = ROOT / "test/results/native-face-contour"
    for source in sorted(cache.glob("*-source.png")):
        sid = source.name.removesuffix("-source.png")
        if sid == "portrait-roll":
            continue
        params = json.loads((cache / f"{sid}-graph.json").read_text())["operations"][0]["params"]
        params = {k: v for k, v in params.items() if k in {
            "region", "mask", "protectMask", "landmarks", "semanticMasks"}}
        image = Image.open(source).convert("RGBA")
        r = params["region"]
        crop = (int(r["x"]*image.width), int(r["y"]*image.height),
                int((r["x"]+r["width"])*image.width), int((r["y"]+r["height"])*image.height))
        sources.append((sid, source, params, crop))
    sid = "portrait-glasses-street"
    source = ROOT / f"test/results/face-core-final/{sid}-source.png"
    if source.exists():
        params = json.loads((source.parent / f"{sid}-faceWidth-60.json").read_text())["operations"][0]["params"]
        params = {k: v for k, v in params.items() if k in {
            "region", "mask", "protectMask", "landmarks", "semanticMasks"}}
        image = Image.open(source).convert("RGBA")
        r = params["region"]
        crop = (int(r["x"]*image.width), int(r["y"]*image.height),
                int((r["x"]+r["width"])*image.width), int((r["y"]+r["height"])*image.height))
        sources.append((sid, source, params, crop))
    for sid, config in SAMPLES.items():
        source = ROOT / f"test/results/skin-acne/{sid}-source.png"
        params = {"region": config["region"], "semanticMasks": {"skin": {
            "shape": "polygon", "points": [{"x": x, "y": y} for x, y in config["points"]],
            "feather": 3}}}
        sources.append((sid, source, params, config["crop"]))
    for sid, source, params, crop in sources:
        image = Image.open(source).convert("RGBA")
        original = np.asarray(image)
        r = params["region"]
        yy, xx = np.indices(original.shape[:2])
        nx, ny = (xx+.5)/image.width, (yy+.5)/image.height
        outside = (nx<r["x"]) | (nx>r["x"]+r["width"]) | (ny<r["y"]) | (ny>r["y"]+r["height"])
        original_light = original[:, :, :3] @ np.array([.2126, .7152, .0722])
        for case, amount in cases.items():
            graph = {"version": "1.0", "sourceAssetId": sid, "operations": [{
                "id": "local-redness-replay", "type": "skin.retouch", "enabled": True,
                "params": {**params, "smooth": 0, "redness": amount}}]}
            graph_path = out / f"{sid}-{case}.json"
            graph_path.write_text(json.dumps(graph))
            panels = [(image, "Original")]
            row = {"sample": sid, "case": case, "params": {"smooth": 0, "redness": amount},
                   "source_sha256": sha(source), "graph_sha256": sha(graph_path)}
            for name, binary in binaries.items():
                assert sha(binary) == hashes[name]
                output = out / f"{sid}-{case}-{name}.png"
                start = time.perf_counter()
                subprocess.run([str(binary), "--input", str(source), "--graph", str(graph_path),
                                "--output", str(output)], check=True, capture_output=True)
                row[name+"_native_total_ms"] = (time.perf_counter()-start)*1000
                assert sha(binary) == hashes[name]
                edited_image = Image.open(output).convert("RGBA")
                edited = np.asarray(edited_image)
                changed = np.any(edited != original, axis=2)
                light = edited[:, :, :3] @ np.array([.2126, .7152, .0722])
                row[name+"_outside_changes"] = int(changed[outside].sum())
                row[name+"_alpha_changes"] = int((edited[:, :, 3] != original[:, :, 3]).sum())
                row[name+"_max_luma_shift"] = float(np.abs(light-original_light).max())
                if sid in SAMPLES:
                    row[name+"_lesions"] = {l["id"]: lesion_metrics(edited, l)
                        for l in SAMPLES[sid]["lesions"]}
                if amount == 0:
                    assert not changed.any(), "zero must be identity"
                if amount < 0:
                    assert row[name+"_max_luma_shift"] <= 0.8, "redness must preserve luminance"
                assert not changed[outside].any()
                assert row[name+"_alpha_changes"] == 0
                panels.append((edited_image, name))
            comparison(panels, crop, out/f"{sid}-{case}-comparison.png")
            row["comparison"] = str((out/f"{sid}-{case}-comparison.png").relative_to(ROOT))
            rows.append(row)
    report = {"binary_sha256": hashes, "rows": rows,
              "independent_originals": len(sources), "derived_roll_excluded": True,
              "scope": "Local redness, no smoothing or AI. Native total includes I/O. Scores are visual."}
    (ROOT/f"doc/qa/{args.label}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"runs": len(rows), "binary_sha256": hashes, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
