"""Fixed real-acne repair strokes, identical old/new parameters and native cores.

This is a manual spot-heal challenge, not an automatic acne detector. Metrics
verify defect contrast and exact edit contracts; aesthetic scores require review
of original-scale crops. Brush circles were fixed before rendering candidates.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

from check_skin_acne_quality import SAMPLES, lesion_metrics
from check_skin_detail_quality import ROOT, comparison, sha

STROKES = {
    "acne-teen": [(206, 540, 12), (232, 578, 12)],
    "acne-oily": [(790, 180, 21), (370, 112, 10)],
    "acne-cystic": [(278, 611, 16)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / ".run/heal-v1-image-render")
    parser.add_argument("--baseline", type=Path, default=ROOT / ".run/heal-baseline-image-render")
    parser.add_argument("--label", default="v1")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    builds = [("Baseline", args.baseline.resolve()), (args.label, args.candidate.resolve())]
    hashes = {name: sha(binary) for name, binary in builds}
    out = ROOT / f"test/results/heal/{args.label}"
    evidence = ROOT / f"doc/qa/heal-{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {row["id"]: row for row in json.loads((ROOT / "test/images/manifest.json").read_text())}
    rows = []
    for sid, strokes in STROKES.items():
        entry = manifest[sid]
        original_path = ROOT / "test/images" / entry["path"]
        assert sha(original_path) == entry["sha256"], f"source changed: {original_path}"
        source_image = ImageOps.exif_transpose(Image.open(original_path)).convert("RGBA")
        ratio = 1024 / max(source_image.size)
        source_image = source_image.resize(tuple(round(v * ratio) for v in source_image.size), Image.Resampling.LANCZOS)
        source = out / f"{sid}-source.png"
        source_image.save(source)
        original = np.asarray(source_image)
        h, w = original.shape[:2]
        yy, xx = np.indices((h, w))
        support = np.zeros((h, w), dtype=bool)
        for x, y, radius in strokes:
            support |= np.hypot(xx - x, yy - y) < radius
        graph = {"version": "1.0", "sourceAssetId": sid, "operations": [{
            "id": "manual-blemish-heal", "type": "repair.heal", "enabled": True,
            "params": {"strokes": [{"x": x / w, "y": y / h, "radius": r / min(w, h), "strength": 100} for x, y, r in strokes]}}]}
        graph_path = out / f"{sid}-graph.json"
        graph_path.write_text(json.dumps(graph, indent=2) + "\n")
        row = {"sample": sid, "original_sha256": sha(original_path), "source_sha256": sha(source),
               "graph_sha256": sha(graph_path), "fixed_strokes_pixels": strokes,
               "lesions": {l["id"]: {"annotation": l, "Original": lesion_metrics(original, l)} for l in SAMPLES[sid]["lesions"]}}
        panels = [(source_image, "Original")]
        for name, binary in builds:
            output = out / f"{sid}-{name}.png"
            times = []
            for _ in range(args.repeat):
                assert sha(binary) == hashes[name]
                begin = time.perf_counter()
                subprocess.run([str(binary), "--input", str(source), "--graph", str(graph_path), "--output", str(output)], capture_output=True, check=True)
                times.append((time.perf_counter() - begin) * 1000)
                assert sha(binary) == hashes[name]
            edited_image = Image.open(output).convert("RGBA")
            edited = np.asarray(edited_image)
            changed = np.any(original != edited, axis=2)
            # Normalized JSON floats can move an exact-radius coordinate by a
            # few ulps; allow only 1e-3 px of support, never a whole pixel.
            exact_support = np.zeros((h, w), dtype=bool)
            for x, y, radius in strokes:
                exact_support |= np.hypot(xx - x, yy - y) < radius + 1e-3
            row[name] = {"outside_brush_changes": int(changed[~exact_support].sum()),
                         "alpha_changes": int(np.any(original[:, :, 3] != edited[:, :, 3])),
                         "hidden_rgb_changes": int(np.any(original != edited, axis=2)[original[:, :, 3] == 0].sum()),
                         "changed_pixels": int(changed.sum()),
                         "output_sha256": sha(output), "native_total_ms": times,
                         "native_total_median_ms": float(np.median(times))}
            for lesion in SAMPLES[sid]["lesions"]:
                row["lesions"][lesion["id"]][name] = lesion_metrics(edited, lesion)
            panels.append((edited_image, name))
        comparison(panels, SAMPLES[sid]["crop"], evidence / f"{sid}-100pct.png")
        for index, (x, y, radius) in enumerate(strokes):
            span = radius + 12
            crop = (max(0, x - span), max(0, y - span), min(w, x + span), min(h, y + span))
            path = evidence / f"{sid}-spot-{index + 1}-100pct.png"
            comparison(panels, crop, path)
            zoom = Image.open(path)
            zoom.resize((zoom.width * 4, zoom.height * 4), Image.Resampling.NEAREST).save(path.with_name(path.stem.replace('100pct', '400pct') + '.png'))
        row["evidence"] = str((evidence / f"{sid}-100pct.png").relative_to(ROOT))
        rows.append(row)
    assert all(sha(binary) == hashes[name] for name, binary in builds)
    report = {"label": args.label, "binary_sha256": hashes,
              "algorithm_source_sha256": sha(ROOT / "core/image-core/src/geometry.rs"),
              "scope": "three independent real acne photographs; five fixed manual spots; 1024px prepared inputs",
              "timing_scope": "native total incl. process startup, PNG decode, edit graph validation, processing, encode; not isolated hot edit time",
              "selection_protocol": STROKES, "rows": rows,
              "note": "No automatic aesthetic score. Original-scale and enlarged crops require visual review. Other unmarked acne is deliberately left in the source."}
    (ROOT / f"doc/qa/heal-{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    for row in rows:
        for name, _ in builds:
            assert row[name]["outside_brush_changes"] == 0
            assert row[name]["alpha_changes"] == 0
            assert row[name]["hidden_rgb_changes"] == 0


if __name__ == "__main__":
    main()
