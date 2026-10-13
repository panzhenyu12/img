"""Frozen native makeup replay; pixel contracts do not award aesthetic scores.

Uses existing parsing graphs, records input/binary hashes and full-size crops.
No detector is rerun, and the glasses case retains its old parsing limitations.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from check_skin_detail_quality import ROOT, comparison, run, sha
from review_face_refinement import polygon_interior


def bounds(masks, names, size):
    points = [p for name in names for p in masks.get(name, {}).get("points", [])]
    if not points:
        raise ValueError(f"Missing polygon for {names}")
    width, height = size
    return (max(0, int(min(p["x"] for p in points) * width) - 12),
            max(0, int(min(p["y"] for p in points) * height) - 12),
            min(width, int(max(p["x"] for p in points) * width) + 13),
            min(height, int(max(p["y"] for p in points) * height) + 13))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--label", default="makeup-material-v1")
    ap.add_argument("--names", nargs="+")
    args = ap.parse_args()
    binaries = {"before": args.baseline.resolve(), "after": args.candidate.resolve()}
    hashes = {key: sha(path) for key, path in binaries.items()}
    qa = ROOT / "doc/qa" / args.label
    output = ROOT / "test/results" / args.label
    qa.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    inputs = [(path, path.with_name(path.name.removesuffix("-source.png") + "-graph.json"))
              for path in sorted((ROOT / "test/results/native-face-contour").glob("*-source.png"))]
    inputs.append((ROOT / "test/results/face-core-mls-matting/portrait-glasses-street-source.png",
                   ROOT / "test/results/face-core-mls-matting/portrait-glasses-street-faceWidth-60.json"))
    rows = []
    bench_hash = sha(args.bench.resolve()) if args.bench else None
    for source, parsing in inputs:
        sample = source.name.removesuffix("-source.png")
        if args.names and sample not in args.names:
            continue
        params = json.loads(parsing.read_text())["operations"][0]["params"]
        params = {k: v for k, v in params.items() if k in {"region", "mask", "protectMask", "landmarks", "semanticMasks"}}
        image = Image.open(source).convert("RGBA")
        original = np.asarray(image)
        yy, xx = np.indices(original.shape[:2])
        region = params["region"]
        inside_roi = (((xx + .5) / image.width >= region["x"]) &
                      ((xx + .5) / image.width <= region["x"] + region["width"]) &
                      ((yy + .5) / image.height >= region["y"]) &
                      ((yy + .5) / image.height <= region["y"] + region["height"]))
        teeth = polygon_interior(params["semanticMasks"]["teeth"], image.size)
        for key, names in [("browColor", ["leftBrow", "rightBrow"]), ("lipColor", ["lips"])]:
            crop = bounds(params["semanticMasks"], names, image.size)
            for amount in [-100, -60, 0, 60, 100]:
                graph = {"version": "1.0", "sourceAssetId": sample, "operations": [{"id": "makeup", "type": "feature.adjust", "enabled": True, "params": {**params, "adjustments": {key: amount}}}]}
                graph_path = output / f"{sample}-{key}-{amount}.json"
                graph_path.write_text(json.dumps(graph))
                row = {"sample": sample, "key": key, "amount": amount,
                       "source_sha256": sha(source), "parsing_sha256": sha(parsing),
                       "graph_sha256": sha(graph_path), "width": image.width, "height": image.height,
                       "semantic_masks": sorted(params["semanticMasks"])}
                panels = [(image, "Original")]
                for label, binary in binaries.items():
                    result_path = output / f"{sample}-{key}-{amount}-{label}.png"
                    elapsed, result = run(binary, hashes[label], source, graph_path, result_path)
                    assert result.returncode == 0, result.stderr
                    result_image = Image.open(result_path).convert("RGBA")
                    pixels = np.asarray(result_image)
                    changed = np.any(pixels != original, axis=2)
                    row[label + "_native_total_ms"] = round(elapsed, 2)
                    row[label + "_changed_pixels"] = int(changed.sum())
                    row[label + "_outside_changes"] = int(changed[~inside_roi].sum())
                    row[label + "_alpha_changes"] = int((pixels[:, :, 3] != original[:, :, 3]).sum())
                    row[label + "_inner_mouth_changes"] = int(changed[teeth].sum())
                    assert not row[label + "_outside_changes"] and not row[label + "_alpha_changes"]
                    # Older baseline may tint feathered enamel; the candidate
                    # is required to keep the teeth polygon bit-exact.
                    if label == "after":
                        assert not row[label + "_inner_mouth_changes"], row
                    if amount == 0:
                        assert not changed.any()
                    panels.append((result_image, label + f" / {key}={amount}"))
                before = np.asarray(panels[1][0])
                after = np.asarray(panels[2][0])
                edited = np.any(after != original, axis=2)
                row["candidate_vs_baseline_pixels"] = int(np.any(before != after, axis=2).sum())
                if key == "lipColor" and edited.any():
                    weights = np.array([.2126, .7152, .0722])
                    delta = np.abs((after[:, :, :3].astype(float) - original[:, :, :3]) @ weights)
                    row["candidate_luma_delta_p99"] = float(np.percentile(delta[edited], 99))
                    row["baseline_luma_delta_p99"] = float(np.percentile(np.abs((before[:, :, :3].astype(float) - original[:, :, :3]) @ weights)[edited], 99))
                comparison(panels, crop, qa / f"{sample}-{key}-{amount}.png")
                if args.bench and amount == 60:
                    assert sha(args.bench.resolve()) == bench_hash
                    row["candidate_graph_benchmark"] = json.loads(subprocess.check_output([str(args.bench.resolve()), str(source), str(graph_path)], text=True))
                    assert sha(args.bench.resolve()) == bench_hash
                rows.append(row)
    report = {"binary_sha256": hashes, "benchmark_sha256": bench_hash, "rows": rows,
              "note": "Protection and luminance checks are contracts, not aesthetic ratings. Seven cached inputs include one rotated derivative and one older glasses parser; detection/initialization not measured. Native total includes process and PNG I/O."}
    (qa / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "zero_cases": sum(r["amount"] == 0 for r in rows), "report": str(qa / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
