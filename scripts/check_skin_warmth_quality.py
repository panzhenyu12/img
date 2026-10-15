"""Replay identical skin temperature graphs against immutable native binaries.

The source/graph/binary hashes and colour-contract measurements are evidence,
not an automatic aesthetic score. Existing parsing caches are deliberately not
regenerated between versions; their provenance/coverage limitations remain.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from check_skin_acne_quality import SAMPLES
from check_skin_detail_quality import ROOT, comparison, sha


CASES = {
    "zero": {"warmth": 0},
    "cool40": {"warmth": -40},
    "cool100": {"warmth": -100},
    "warm40": {"warmth": 40},
    "warm100": {"warmth": 100},
    "smooth-control": {"smooth": 60, "texture": 60},
    "whiten-control": {"whiten": 60},
    "redness-control": {"redness": -40},
}


def sources():
    items = []
    cache = ROOT / "test/results/native-face-contour"
    for source in sorted(cache.glob("*-source.png")):
        sid = source.name.removesuffix("-source.png")
        if sid == "portrait-roll":
            continue
        graph = cache / f"{sid}-graph.json"
        params = json.loads(graph.read_text())["operations"][0]["params"]
        items.append((sid, source, params))
    sid = "portrait-glasses-street"
    source = ROOT / f"test/results/face-core-final/{sid}-source.png"
    if source.exists():
        params = json.loads((source.parent / f"{sid}-faceWidth-60.json").read_text())[
            "operations"][0]["params"]
        items.append((sid, source, params))
    for sid, config in SAMPLES.items():
        items.append((sid, ROOT / f"test/results/skin-acne/{sid}-source.png", {
            "region": config["region"], "semanticMasks": {"skin": {
                "shape": "polygon", "points": [{"x": x, "y": y}
                                                for x, y in config["points"]], "feather": 3}}}))
    return items


def checked_run(binary, expected_sha, source, graph, output):
    assert sha(binary) == expected_sha
    start = time.perf_counter()
    subprocess.run([str(binary), "--input", str(source), "--graph", str(graph),
                    "--output", str(output)], check=True, capture_output=True)
    elapsed = (time.perf_counter() - start) * 1000
    assert sha(binary) == expected_sha
    return elapsed


def contact_sheet(rows, case, out):
    relevant = [r for r in rows if r["case"] == case]
    thumb_width = 630
    panels = []
    for row in relevant:
        image = Image.open(ROOT / row["comparison"]).convert("RGB")
        height = round(image.height * thumb_width / image.width)
        thumb = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
        labelled = Image.new("RGB", (thumb_width, height + 24), "#161b22")
        labelled.paste(thumb, (0, 24))
        ImageDraw.Draw(labelled).text((6, 6), row["sample"], fill="white")
        panels.append(labelled)
    contact = Image.new("RGB", (thumb_width * 3, max(sum(p.height for p in panels[i::3])
                                                       for i in range(3))), "#161b22")
    offsets = [0, 0, 0]
    for i, panel in enumerate(panels):
        column = i % 3
        contact.paste(panel, (column * thumb_width, offsets[column]))
        offsets[column] += panel.height
    contact.save(out / f"{case}-contact.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-bench", type=Path)
    parser.add_argument("--reference-bench", type=Path)
    parser.add_argument("--candidate-color-source", type=Path,
                        default=ROOT / "core/image-core/src/color.rs")
    parser.add_argument("--label", default="skin-warmth-v1")
    args = parser.parse_args()
    out = ROOT / "test/results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    binaries = {"before": args.reference.resolve(), "after": args.candidate.resolve()}
    hashes = {name: sha(binary) for name, binary in binaries.items()}
    benches = {"before": args.reference_bench, "after": args.candidate_bench}
    bench_hashes = {name: sha(path) for name, path in benches.items() if path}
    rows, timings = [], []
    samples = sources()
    for sid, source, params in samples:
        source_hash = sha(source)
        params = {k: v for k, v in params.items() if k in {
            "region", "mask", "protectMask", "landmarks", "semanticMasks"}}
        image = Image.open(source).convert("RGBA")
        original = np.asarray(image)
        original_rgb = original[:, :, :3].astype(np.float64)
        original_light = original_rgb @ np.array([.2126, .7152, .0722])
        original_yellow = original_rgb[:, :, 1] - original_rgb[:, :, 2]
        r = params["region"]
        crop = (int(r["x"]*image.width), int(r["y"]*image.height),
                int((r["x"]+r["width"])*image.width), int((r["y"]+r["height"])*image.height))
        yy, xx = np.indices(original.shape[:2])
        nx, ny = (xx+.5)/image.width, (yy+.5)/image.height
        outside = (nx<r["x"]) | (nx>r["x"]+r["width"]) | (ny<r["y"]) | (ny>r["y"]+r["height"])
        for case, adjustments in CASES.items():
            graph = {"version": "1.0", "sourceAssetId": sid, "operations": [{
                "id": "skin-temperature-replay", "type": "skin.retouch", "enabled": True,
                "params": {**params, **adjustments}}]}
            graph_path = out / f"{sid}-{case}.json"
            graph_path.write_text(json.dumps(graph))
            row = {"sample": sid, "case": case, "params": adjustments,
                   "source_sha256": source_hash, "graph_sha256": sha(graph_path),
                   "width": image.width, "height": image.height}
            panels = [(image, "Original")]
            edited_arrays = {}
            for name, binary in binaries.items():
                assert sha(source) == source_hash
                assert sha(graph_path) == row["graph_sha256"]
                output = out / f"{sid}-{case}-{name}.png"
                row[name+"_native_total_ms"] = checked_run(binary, hashes[name], source, graph_path, output)
                assert sha(source) == source_hash
                assert sha(graph_path) == row["graph_sha256"]
                edited_image = Image.open(output).convert("RGBA")
                edited = np.asarray(edited_image)
                edited_arrays[name] = edited
                edited_rgb = edited[:, :, :3].astype(np.float64)
                changed = np.any(edited != original, axis=2)
                light = edited_rgb @ np.array([.2126, .7152, .0722])
                row[name+"_png_sha256"] = sha(output)
                row[name+"_changed_pixels"] = int(changed.sum())
                row[name+"_outside_changes"] = int(changed[outside].sum())
                row[name+"_alpha_changes"] = int((edited[:, :, 3] != original[:, :, 3]).sum())
                row[name+"_max_luma_shift"] = float(np.abs(light-original_light).max())
                new_yellow = edited_rgb[:, :, 1]-edited_rgb[:, :, 2]
                row[name+"_new_blue_crossings"] = int(((original_yellow >= 2) &
                    (new_yellow < 0) & changed).sum())
                red_shift = ((edited_rgb[:, :, 0]-edited_rgb[:, :, 1]) -
                    (original_rgb[:, :, 0]-original_rgb[:, :, 1]))
                row[name+"_mean_absolute_red_axis_shift"] = float(np.abs(red_shift[changed]).mean()) if changed.any() else 0
                assert not changed[outside].any()
                assert row[name+"_alpha_changes"] == 0
                if case == "zero":
                    assert not changed.any()
                if "warmth" in adjustments:
                    assert row[name+"_max_luma_shift"] <= 0.8
                    if name == "after":
                        assert row[name+"_new_blue_crossings"] == 0
                panels.append((edited_image, name + " / " + case))
            row["same_rgba_as_reference"] = bool(np.array_equal(edited_arrays["before"], edited_arrays["after"]))
            if case.endswith("control") or case == "zero":
                assert row["same_rgba_as_reference"], "unrelated skin controls must remain unchanged"
            comparison(panels, crop, out/f"{sid}-{case}-comparison.png")
            row["comparison"] = str((out/f"{sid}-{case}-comparison.png").relative_to(ROOT))
            if not sid.startswith("acne-") and case in {"cool100", "warm100"}:
                for name, bench in benches.items():
                    if not bench:
                        continue
                    assert sha(bench) == bench_hashes[name]
                    result = subprocess.run([str(bench.resolve()), str(source), str(graph_path)],
                                            check=True, capture_output=True, text=True)
                    assert sha(bench) == bench_hashes[name]
                    timings.append({"sample": sid, "case": case, "version": name,
                                    "graph_sha256": sha(graph_path), **json.loads(result.stdout)})
            rows.append(row)
    for case in ["cool40", "cool100", "warm40", "warm100"]:
        contact_sheet(rows, case, out)
    assert all(sha(binary) == hashes[name] for name, binary in binaries.items())
    report = {"binary_sha256": hashes, "bench_sha256": bench_hashes,
              "binary_color_source_sha256": sha(args.candidate_color_source),
              "workspace_color_source_sha256": sha(ROOT / "core/image-core/src/color.rs"),
              "independent_originals": len(samples), "derived_roll_excluded": True,
              "scope": "Skin warmth only; existing masks fixed. Native total includes I/O. Scores require visual review.",
              "rows": rows, "same_process_timings": timings}
    (ROOT/f"doc/qa/{args.label}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"graphs": len(rows), "independent_originals": len(samples),
        "before_cold100_blue_crossings": sum(r["before_new_blue_crossings"] for r in rows if r["case"] == "cool100"),
        "after_cold100_blue_crossings": sum(r["after_new_blue_crossings"] for r in rows if r["case"] == "cool100"),
        "timing_rows": len(timings), "report": f"doc/qa/{args.label}.json"}, indent=2))


if __name__ == "__main__":
    main()
