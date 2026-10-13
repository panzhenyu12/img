"""Frozen native comparison for adaptive under-eye correction.

Uses six cached full-resolution parsing graphs; records missing/zero-effect
targets without treating them as visual passes. No detection is re-run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from check_skin_detail_quality import ROOT, comparison, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / ".run/dark-circle-before-image-render")
    parser.add_argument("--label", default="under-eye-v1")
    parser.add_argument("--extra-glasses", action="store_true", help="Include an independent cached eyeglasses portrait with its original parsing limitations.")
    args = parser.parse_args()
    output = ROOT / "test/results" / args.label
    output.mkdir(parents=True, exist_ok=True)
    binaries = [("before", args.reference), ("after", args.candidate)]
    hashes = {label: sha(binary) for label, binary in binaries}
    rows = []
    inputs = [(path, path.with_name(path.name.removesuffix("-source.png") + "-graph.json"))
              for path in sorted((ROOT / "test/results/native-face-contour").glob("*-source.png"))]
    if args.extra_glasses:
        inputs.append((ROOT / "test/results/face-core-mls-matting/portrait-glasses-street-source.png",
                       ROOT / "test/results/face-core-mls-matting/portrait-glasses-street-faceWidth-60.json"))
    for source_path, parsing_path in inputs:
        sample = source_path.name.removesuffix("-source.png")
        params = json.loads(parsing_path.read_text())["operations"][0]["params"]
        params = {key: value for key, value in params.items() if key in {"region", "mask", "protectMask", "landmarks", "semanticMasks"}}
        image = Image.open(source_path).convert("RGBA")
        original = np.asarray(image)
        region = params["region"]
        crop = (int(region["x"] * image.width), int(region["y"] * image.height),
                min(image.width, int((region["x"] + region["width"]) * image.width)),
                min(image.height, int((region["y"] + region["height"]) * image.height)))
        yy, xx = np.indices(original.shape[:2])
        outside = ((xx + .5) / image.width < region["x"]) | ((xx + .5) / image.width > region["x"] + region["width"]) | ((yy + .5) / image.height < region["y"]) | ((yy + .5) / image.height > region["y"] + region["height"])
        for amount in [0, 30, 60, 100]:
            graph = {"version": "1.0", "sourceAssetId": sample, "operations": [{"id": "under-eye", "type": "feature.adjust", "enabled": True, "params": {**params, "adjustments": {"darkCircle": amount}}}]}
            graph_path = output / f"{sample}-{amount}.json"
            graph_path.write_text(json.dumps(graph))
            panels = [(image, "Original")]
            row = {"sample": sample, "amount": amount, "source_sha256": sha(source_path), "graph_sha256": sha(graph_path), "width": image.width, "height": image.height}
            row["parsing_sha256"] = sha(parsing_path)
            row["semantic_masks"] = sorted(params.get("semanticMasks", {}))
            for label, binary in binaries:
                assert sha(binary) == hashes[label]
                result_path = output / f"{sample}-{amount}-{label}.png"
                begin = time.perf_counter()
                subprocess.run([str(binary), "--input", str(source_path), "--graph", str(graph_path), "--output", str(result_path)], check=True, capture_output=True)
                row[label + "_native_total_ms"] = round((time.perf_counter() - begin) * 1000, 2)
                assert sha(binary) == hashes[label]
                result_image = Image.open(result_path).convert("RGBA")
                result = np.asarray(result_image)
                changed = np.any(original != result, axis=2)
                row[label + "_changed_pixels"] = int(changed.sum())
                row[label + "_outside_changes"] = int(changed[outside].sum())
                row[label + "_alpha_changes"] = int((result[:, :, 3] != original[:, :, 3]).sum())
                row[label + "_max_channel_delta"] = int(np.abs(result[:, :, :3].astype(int) - original[:, :, :3].astype(int)).max())
                assert not row[label + "_outside_changes"] and not row[label + "_alpha_changes"]
                if amount == 0:
                    assert not changed.any()
                panels.append((result_image, label + f" / {amount}"))
            comparison(panels, crop, output / f"{sample}-{amount}-comparison.png")
            row["evidence"] = str((output / f"{sample}-{amount}-comparison.png").relative_to(ROOT))
            rows.append(row)
    report = {"binary_sha256": hashes, "color_source_sha256": sha(ROOT / "core/image-core/src/color.rs"), "rows": rows, "timing_scope": "native total includes PNG and startup; simultaneous development load", "note": "All pixel checks are contracts, not aesthetic scores; cached parsing, portrait-roll is a rotated derivative. Eyeglasses source uses an older cached parser without accessories/ears masks."}
    (ROOT / "doc/qa" / (args.label + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "changed_range": [min(r["after_changed_pixels"] for r in rows if r["amount"]), max(r["after_changed_pixels"] for r in rows)], "max_delta": max(r["after_max_channel_delta"] for r in rows), "native_total_ms": [min(r["after_native_total_ms"] for r in rows), max(r["after_native_total_ms"] for r in rows)]}, indent=2))


if __name__ == "__main__":
    main()
