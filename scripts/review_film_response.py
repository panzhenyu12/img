#!/usr/bin/env python3
"""Reproducible real-photo film review, including engine-only hot timings.

Use a Python with Pillow/numpy and a built graph_quality_bench executable.
The report preserves the executable/source hashes; scores require visual QA.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ["provia", "astia", "velvia", "classicNegative", "eterna", "acros"]
PHOTOS = ["portrait-makeup", "portrait-dark", "portrait-hair", "portrait-woman", "cityscape", "street-crowd"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "core/image-core/target/release/examples/graph_quality_bench")
    parser.add_argument("--out", type=Path, default=ROOT / "test/results/film-response-v1")
    parser.add_argument("--previous", type=Path, help="Existing frozen v1 outputs for exact profile comparison")
    parser.add_argument("--qa", type=Path, help="Save compact grids, 100%% face crops and report for versioned documentation")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.previous:
        args.previous = args.previous.resolve()
    if args.qa:
        args.qa = args.qa.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    before = digest(binary)
    report = {"binary_sha256": before, "film_source_sha256": digest(ROOT / "core/image-core/src/film.rs"), "scope": "1024px long edge, first/hot core timing excludes decode/encode, detection and browser initialization", "presets": PRESETS, "cases": [], "changes_vs_previous": [], "crops": []}
    for name in PHOTOS:
        original = ROOT / "test/images" / (name + ".jpg")
        image = ImageOps.exif_transpose(Image.open(original)).convert("RGBA")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        path = args.out / (name + "-source.png")
        image.save(path)
        cells = []
        for preset in ["neutral", *PRESETS]:
            graph = {"operations": [{"id": "film", "type": "filter.film", "enabled": True, "params": {"preset": preset, "intensity": 100, "seed": 42}}]}
            gp = args.out / (name + "-" + preset + ".json")
            gp.write_text(json.dumps(graph))
            op = args.out / (name + "-" + preset + ".png")
            if digest(binary) != before:
                raise RuntimeError("review executable changed during replay")
            process = subprocess.run([str(binary), str(path), str(gp), str(op)], capture_output=True, text=True, check=True)
            if digest(binary) != before:
                raise RuntimeError("review executable changed during replay")
            timing = json.loads(process.stdout)
            result = Image.open(op).convert("RGBA")
            a, b = np.array(image), np.array(result)
            if not np.array_equal(a[:, :, 3], b[:, :, 3]):
                raise AssertionError("film changed alpha")
            report["cases"].append({"photo": name, "preset": preset, "source_sha256": digest(original), "render_input_sha256": digest(path), "output_sha256": digest(op), **timing})
            if args.previous:
                old = args.previous / (name + "-" + preset + ".png")
                old_pixels = np.array(Image.open(old).convert("RGBA"))
                assert old_pixels.shape == b.shape, "comparison inputs have different dimensions"
                changed = int(np.any(old_pixels != b, axis=2).sum())
                if preset != "velvia":
                    assert changed == 0, (name, preset, "unrelated profile changed")
                report["changes_vs_previous"].append({"photo": name, "preset": preset, "previous_output_sha256": digest(old), "changed_pixels": changed})
            thumb = result.convert("RGB")
            thumb.thumbnail((300, 330), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (320, 365), "#f2f2f2")
            cell.paste(thumb, ((320 - thumb.width) // 2, 5))
            ImageDraw.Draw(cell).text((10, 345), preset, fill="black")
            cells.append(cell)
        grid = Image.new("RGB", (320 * 4, 365 * 2), "#ddd")
        for i, cell in enumerate(cells):
            grid.paste(cell, ((i % 4) * 320, (i // 4) * 365))
        grid.save(args.out / (name + "-grid.jpg"), quality=94)
        if args.qa:
            args.qa.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.out / (name + "-grid.jpg"), args.qa / (name + "-grid.jpg"))
            seed = ROOT / f"test/results/face-core-fresh/{name}-faceWidth-60.json"
            if not seed.exists():
                seed = ROOT / f"test/results/under-eye-v5/{name}-60.json"
            if args.previous and seed.exists():
                points = json.loads(seed.read_text())["operations"][0]["params"]["semanticMasks"]["faceContour"]["points"]
                xs, ys = [p["x"] * image.width for p in points], [p["y"] * image.height for p in points]
                box = (max(0, int(min(xs))-8), max(0, int(min(ys))-8), min(image.width, int(max(xs))+8), min(image.height, int(max(ys))+8))
                panels = [(path, "Original"), (args.previous / (name + "-velvia.png"), "V1 / 100"), (args.out / (name + "-velvia.png"), "V2 / 100")]
                cw, ch = box[2]-box[0], box[3]-box[1]
                sheet = Image.new("RGB", (cw*3, ch+28), "#171c22")
                draw = ImageDraw.Draw(sheet)
                for i, (panel, label) in enumerate(panels):
                    sheet.paste(Image.open(panel).convert("RGB").crop(box), (i*cw, 28))
                    draw.text((i*cw+8, 8), label, fill="white")
                evidence = args.qa / (name + "-velvia-100pct.png")
                sheet.save(evidence)
                report["crops"].append({"photo": name, "bounds": box, "source_detection_graph_sha256": digest(seed), "scale": "100% pixels; normalized existing outline only defines the crop, not the filter", "evidence": str(evidence.relative_to(ROOT))})
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    if args.qa:
        (args.qa / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"cases": len(report["cases"]), "max_hot_ms": max(c["hot_median_ms"] for c in report["cases"]), "report": str(args.out / "report.json")}))


if __name__ == "__main__":
    main()
