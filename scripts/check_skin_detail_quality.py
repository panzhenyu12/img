"""Preserve identical-parameter original / ea5b404 / multiscale skin evidence.

Pixel checks only verify edit contracts. Native times include process startup,
decode, graph validation, edit and encode; these are not isolated hot edit times.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "smooth-natural": {"smooth": 60, "texture": 60},
    "smooth-strong": {"smooth": 100, "texture": 0},
    "texture-preserved": {"smooth": 80, "texture": 100},
    "whiten": {"whiten": 60},
    "warm": {"warmth": 40},
    "cool": {"warmth": -40},
    "red": {"redness": 40},
    "neutralize-red": {"redness": -40},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary, expected_sha, source, graph, output):
    assert sha(binary) == expected_sha, f"binary changed before spawn: {binary}"
    start = time.perf_counter()
    result = subprocess.run([str(binary), "--input", str(source), "--output", str(output),
                             "--graph", str(graph)], capture_output=True, text=True)
    elapsed = (time.perf_counter() - start) * 1000
    assert sha(binary) == expected_sha, f"binary changed during spawn: {binary}"
    return elapsed, result


def comparison(panels, crop, output):
    x0, y0, x1, y1 = crop
    w, h = x1 - x0, y1 - y0
    canvas = Image.new("RGB", (w * len(panels), h + 30), "#161b22")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(panels):
        canvas.paste(panel.crop(crop), (i * w, 30))
        draw.text((i * w + 8, 8), label, fill="white")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / ".run/skin-detail-v2-image-render")
    parser.add_argument("--baseline-binary", type=Path, default=ROOT / ".run/quality-baseline/image-render")
    parser.add_argument("--baseline-label", default="ea5b404 guided")
    parser.add_argument("--baseline-revision", default="ea5b4045a45fc5233d751c35aa945db5da7aa106")
    parser.add_argument("--output", type=Path, default=ROOT / "test/results/skin-detail")
    parser.add_argument("--report", type=Path, default=ROOT / "doc/qa/skin-detail-validation.json")
    parser.add_argument("--candidate-label", default="Multiscale")
    options = parser.parse_args()
    hashes = {str(p): sha(p) for p in [options.binary, options.baseline_binary]}
    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "test/results/native-face-contour"
    rows = []
    for source_path in sorted(cache.glob("*-source.png")):
        sid = source_path.name.removesuffix("-source.png")
        params = json.loads((cache / f"{sid}-graph.json").read_text())["operations"][0]["params"]
        image = Image.open(source_path).convert("RGBA")
        original = np.asarray(image)
        region = params["region"]
        x0, y0 = max(0, int(region["x"] * image.width)), max(0, int(region["y"] * image.height))
        x1 = min(image.width, int((region["x"] + region["width"]) * image.width))
        y1 = min(image.height, int((region["y"] + region["height"]) * image.height))
        yy, xx = np.indices(original.shape[:2])
        support = ((xx + .5) / image.width >= region["x"]) & ((xx + .5) / image.width <= region["x"] + region["width"]) & ((yy + .5) / image.height >= region["y"]) & ((yy + .5) / image.height <= region["y"] + region["height"])
        for case, adjustments in CASES.items():
            graph = {"version": "1.0", "sourceAssetId": "local", "operations": [{
                "id": "skin-detail-quality", "type": "skin.retouch", "enabled": True,
                "params": {**{k: v for k, v in params.items() if k in {
                    "region", "mask", "protectMask", "landmarks", "semanticMasks"}}, **adjustments}}]}
            graph_path = out / f"{sid}-{case}.json"
            graph_path.write_text(json.dumps(graph))
            panels = [(image, "Original")]
            row = {"sample": sid, "case": case, "parameters": adjustments,
                   "input_sha256": sha(source_path), "graph_sha256": sha(graph_path),
                   "width": image.width, "height": image.height}
            for label, binary in [("baseline", options.baseline_binary), ("current", options.binary)]:
                output = out / f"{sid}-{case}-{label}.png"
                ms, result = run(binary, hashes[str(binary)], source_path, graph_path, output)
                row[label + "_native_total_ms"] = round(ms, 2)
                if result.returncode:
                    row[label + "_error"] = result.stderr
                    continue
                edited_image = Image.open(output).convert("RGBA")
                edited = np.asarray(edited_image)
                changed = np.any(edited != original, axis=2)
                row[label + "_changed_pixels"] = int(changed.sum())
                row[label + "_outside_support_changed"] = int((changed & ~support).sum())
                row[label + "_alpha_changed"] = int((edited[:, :, 3] != original[:, :, 3]).sum())
                panels.append((edited_image, options.baseline_label if label == "baseline" else options.candidate_label + " / " + case))
            comparison(panels, (x0, y0, x1, y1), out / f"{sid}-{case}-comparison.png")
            # The center cheek/skin region stays at original pixel scale. Full
            # support crops above retain all face features and boundary context.
            cx, cy = (x0 + x1) // 2, y0 + (y1 - y0) * 3 // 5
            crop = (max(x0, cx - 96), max(y0, cy - 96), min(x1, cx + 96), min(y1, cy + 96))
            comparison(panels, crop, out / f"{sid}-{case}-100pct.png")
            row["comparison"] = str((out / f"{sid}-{case}-comparison.png").relative_to(ROOT))
            row["crop_100pct"] = str((out / f"{sid}-{case}-100pct.png").relative_to(ROOT))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"input_sha256", "graph_sha256"}}), flush=True)
    assert all(sha(Path(path)) == expected for path, expected in hashes.items())
    report = {"baseline_revision": options.baseline_revision,
              "baseline_label": options.baseline_label,
              "baseline_binary_sha256": sha(options.baseline_binary),
              "current_binary_sha256": sha(options.binary), "samples": 6,
              "color_source_sha256": sha(ROOT / "core/image-core/src/color.rs"),
              "binary_checks": "sha256 before and after every spawn, then final hash",
              "independent_originals": 5, "timing_scope": "native total process including I/O",
              "rows": rows}
    options.report.resolve().write_text(json.dumps(report, indent=2) + "\n")
    assert len(rows) == 48, "expected all six cached sample variants and eight cases"
    assert all(not r.get(label + "_error") and r[label + "_outside_support_changed"] == 0
               and r[label + "_alpha_changed"] == 0 for r in rows for label in ["baseline", "current"])


if __name__ == "__main__":
    main()
