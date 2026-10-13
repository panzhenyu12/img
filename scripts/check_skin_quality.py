"""Replay cached real-photo masks and keep old/new skin outputs for inspection.

Uses the PIL/NumPy workspace runtime. Timings include native process startup,
PNG decode/encode and the edit. Pixel metrics are contracts, not beauty scores.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--baseline-binary", type=Path, help="A verified pre-guided-filter build; omit for current-only evidence")
    options = args.parse_args()
    binary = ROOT / "core/image-core/target/release/image-render"
    cache = ROOT / "test/results/native-face-contour"
    out = ROOT / "test/results/skin-guided"
    out.mkdir(parents=True, exist_ok=True)
    cases = {"smooth-natural": {"smooth": 60, "texture": 60},
             "smooth-strong": {"smooth": 100, "texture": 0},
             "texture-preserved": {"smooth": 80, "texture": 100},
             "whiten": {"whiten": 60}, "warm": {"warmth": 40},
             "cool": {"warmth": -40}, "red": {"redness": 40},
             "neutralize-red": {"redness": -40}}
    reports = []
    for source_path in sorted(cache.glob("*-source.png")):
        sid = source_path.name.removesuffix("-source.png")
        original_graph = json.loads((cache / f"{sid}-graph.json").read_text())
        params = original_graph["operations"][0]["params"]
        image = Image.open(source_path).convert("RGBA")
        original = np.asarray(image)
        for name, adjustment in cases.items():
            graph = {"version": "1.0", "sourceAssetId": "local", "operations": [{
                "id": "skin-quality", "type": "skin.retouch", "enabled": True,
                "params": {**{k: v for k, v in params.items() if k in {"region", "mask", "protectMask", "landmarks", "semanticMasks"}}, **adjustment}}]}
            graph_path = out / f"{sid}-{name}.json"
            graph_path.write_text(json.dumps(graph))
            target = out / f"{sid}-{name}.png"
            start = time.monotonic()
            result = subprocess.run([str(binary), "--input", str(source_path), "--output", str(target), "--graph", str(graph_path)], capture_output=True, text=True)
            ms = (time.monotonic() - start) * 1000
            if result.returncode:
                reports.append({"sample": sid, "case": name, "error": result.stderr})
                continue
            edited_image = Image.open(target).convert("RGBA")
            edited = np.asarray(edited_image)
            region = params["region"]
            yy, xx = np.indices(original.shape[:2])
            selected = ((xx + .5) / image.width >= region["x"]) & ((xx + .5) / image.width <= region["x"] + region["width"]) & ((yy + .5) / image.height >= region["y"]) & ((yy + .5) / image.height <= region["y"] + region["height"])
            changed = np.any(edited != original, axis=2)
            row = {"sample": sid, "case": name, "native_total_ms": round(ms, 2), "changed_pixels": int(changed.sum()), "outside_support_changed": int((changed & ~selected).sum()), "alpha_changed": int(np.count_nonzero(edited[:, :, 3] != original[:, :, 3]))}
            panels = [(image, "Original")]
            if options.baseline_binary:
                baseline = out / f"{sid}-{name}-baseline.png"
                run = subprocess.run([str(options.baseline_binary), "--input", str(source_path), "--output", str(baseline), "--graph", str(graph_path)], capture_output=True)
                if run.returncode == 0:
                    panels.append((Image.open(baseline).convert("RGBA"), "Previous Gaussian"))
            panels.append((edited_image, "Guided / " + name))
            x = max(0, int(region["x"] * image.width)); y = max(0, int(region["y"] * image.height))
            w = min(image.width - x, int(region["width"] * image.width)); h = min(image.height - y, int(region["height"] * image.height))
            canvas = Image.new("RGB", (w * len(panels), h + 30), "#161b22")
            draw = ImageDraw.Draw(canvas)
            for i, (panel, label) in enumerate(panels):
                canvas.paste(panel.crop((x, y, x+w, y+h)), (i*w, 30))
                draw.text((i*w+8, 8), label, fill="white")
            comparison = out / f"{sid}-{name}-comparison.png"
            canvas.save(comparison)
            row["comparison"] = str(comparison.relative_to(ROOT))
            reports.append(row)
            print(json.dumps(row), flush=True)
    (ROOT / "doc/qa/skin-guided-validation.json").write_text(json.dumps(reports, indent=2) + "\n")
    assert reports, "No cached source images; run real-photo face diagnostics first"
    assert all(not r.get("error") and r["outside_support_changed"] == 0 and r["alpha_changed"] == 0 for r in reports), "Retouching contract failed; see report"


if __name__ == "__main__":
    main()
