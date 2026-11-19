"""Replay frozen yoga layers without refitting their alpha, material or scene.

The crop is valid only after the ordinary, unmodified full-canvas field proves
that every moving destination and all four sampling neighbours are inside it.
Outside-crop identity placeholders are never evidence of background ownership.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from visible_rear_geometry import build_visible_rear_geometry


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "test/results/body-anatomical-inputs/body-yoga-mediapipe/source.png"
GRAPH = SOURCE.parent / "person-0-waist.json"
LAYERS = ROOT / "test/results/body-waist-20261116/automatic-source-replay/layers"
TRACE = ROOT / "test/results/body-waist-20261116/native-authoritative"
VIEW = {"x": 401, "y": 332, "width": 202, "height": 176}
EXPECTED = {
    "source": "7a648e911e8cda8fb54bc6f2d12f3aa8c57d05719121607bd31e8dec6958cc11",
    "graph": "f67654ec718c343c14ceda59b55e4003e2cf98d1d421d9b89f312ba96d30fdf8",
    "source.rgba": "034c9d7d4df41031fccbaf0e4916b27ac2a7b57e28261e49629f89a0f855ca1c",
    "alpha.f64": "71fadd30ad117407dafc952f893ffcc2b7da34b9a59fc7819e5d877395a2f730",
    "premul.f64": "25fba512abc68ad00817a42bd23e4562fd36bb82baedc8e2b42612eededf7a75",
    "background.f64": "8f7e53adc5ad9eb3805c558392988763331fceef79b6a38bd75e30cbb29be482",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def support_audit(field: np.ndarray, view: dict) -> dict:
    """Use the compositor's full-canvas clamp, never a crop-boundary clamp."""
    if field.ndim != 3 or field.shape[2] != 2 or not np.isfinite(field).all():
        raise ValueError("field must be a finite full-canvas H×W×2 array")
    h, w = field.shape[:2]
    active_y, active_x = np.where(np.any(field != 0, axis=2))
    left, top = view["x"], view["y"]
    right, bottom = left + view["width"], top + view["height"]
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError("prepared crop must lie inside the source canvas")

    def inside(xs, ys):
        return (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)

    destination_known = inside(active_x, active_y)
    sx = np.clip(active_x + field[active_y, active_x, 0].astype(float), 0, w - 1)
    sy = np.clip(active_y + field[active_y, active_x, 1].astype(float), 0, h - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = sx - x0, sy - y0
    all_known = np.ones(len(active_x), bool)
    weighted_known = all_known.copy()
    for xs, ys, weight in [(x0, y0, (1 - tx) * (1 - ty)),
                           (x1, y0, tx * (1 - ty)),
                           (x0, y1, (1 - tx) * ty), (x1, y1, tx * ty)]:
        known = inside(xs, ys)
        all_known &= known
        weighted_known &= known | (weight == 0)
    bad = ~(destination_known & all_known)
    indices = np.flatnonzero(bad)[:8]
    return {
        "nonzeroFieldPixels": len(active_x),
        "unknownDestinationPixels": int((~destination_known).sum()),
        "unknownFourNeighbourPixels": int((~all_known).sum()),
        "unknownNonzeroWeightNeighbourPixels": int((~weighted_known).sum()),
        "allMovingDestinationsAndFourNeighboursKnown": bool(not bad.any()),
        "sourceCoordinatesUseFullCanvasClamp": True,
        "firstUnsupported": [{"destination": [int(active_x[i]), int(active_y[i])],
                              "sourceXY": [float(sx[i]), float(sy[i])]} for i in indices],
    }


def frozen_source():
    paths = {"source": SOURCE, "graph": GRAPH,
             **{name: LAYERS / name for name in EXPECTED if name not in ("source", "graph")}}
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise ValueError("frozen yoga source, graph or layer hash changed")
    source = np.array(Image.open(SOURCE).convert("RGBA"))
    x, y, w, h = (VIEW[k] for k in ("x", "y", "width", "height"))
    crop = source[y:y + h, x:x + w]
    if not np.array_equal(crop, np.fromfile(LAYERS / "source.rgba", np.uint8).reshape(h, w, 4)):
        raise ValueError("frozen layer source crop differs from the source photo")
    a = np.fromfile(LAYERS / "alpha.f64", "<f8").reshape(h, w)
    p = np.fromfile(LAYERS / "premul.f64", "<f8").reshape(h, w, 3)
    b = np.fromfile(LAYERS / "background.f64", "<f8").reshape(h, w, 3)
    if not all(np.isfinite(v).all() for v in (a, p, b)):
        raise ValueError("nonfinite frozen source material")
    if not (np.all((a >= 0) & (a <= 1)) and np.all((b >= 0) & (b <= 255))
            and np.all((p >= 0) & (p <= 255 * a[..., None]))):
        raise ValueError("frozen source material is out of physical bounds")
    latent = p + (1 - a[..., None]) * b
    if not np.array_equal(np.floor(latent + .5), crop[..., :3]):
        raise ValueError("frozen latent reconstruction fails the explicit nearest-byte contract")
    audit = {"hashes": hashes, "sourceCrop": VIEW,
             "sourceReconstruction": "nearest-byte", "sourceBytesRoundTripExact": True,
             "maxEncodingResidual": float(abs(crop[..., :3] - latent).max()),
             "sourceLayerModified": False, "encodingResidualTransportedOrAdded": False,
             "newModelInference": False, "manualGeometry": False}
    # Despite the historical name "target", the producer writes this array
    # before motion; it separately exports sampled destination "warped-alpha".
    producer_hash = sha(TRACE / "body-trace.rs")
    if producer_hash != "e6cb0c474675663a24e21e782e7783bcf555655e3282b1435ea1f6e42a8adea6":
        raise ValueError("frozen alpha producer changed")
    target = np.fromfile(TRACE / "+30/target-alpha.f32", "<f4").reshape(h, w)
    opposite = np.fromfile(TRACE / "-30/target-alpha.f32", "<f4").reshape(h, w)
    if not np.array_equal(target, opposite) or not np.array_equal(target.astype(float), a):
        raise ValueError("frozen source alpha differs between requests or from the paired layer")
    provenance = {"producer": str(TRACE / "body-trace.rs"), "producerSHA256": producer_hash,
                  "sourceConstructionLines": [1020, 1028], "destinationSamplingLine": 1144,
                  "separateExportLines": [1182, 1183], "plusMinusSourceAlphaIdentical": True,
                  "pairedF64EqualsSourceF32": True,
                  "interpretation": "Legacy source part/subject coverage estimate with optical preserve; not a new matting prediction or destination-warped coverage.",
                  "warpedDifferencePixels": {}}
    for sign in ("+30", "-30"):
        warped = np.fromfile(TRACE / sign / "warped-alpha.f32", "<f4").reshape(h, w)
        provenance["warpedDifferencePixels"][sign] = int(np.count_nonzero(warped != target))
    audit["sourceAlphaProvenance"] = provenance
    return source, a, b, latent, audit


def prepared_arrays(out: Path, source, alpha, background, latent) -> Path:
    """Called only after all requested original/candidate fields pass support."""
    prepared = out / "prepared"
    prepared.mkdir()
    h, w = source.shape[:2]
    x, y, cw, ch = (VIEW[k] for k in ("x", "y", "width", "height"))
    a = np.zeros((h, w)); a[y:y + ch, x:x + cw] = alpha
    s = source[..., :3].astype(float); s[y:y + ch, x:x + cw] = latent
    b = source[..., :3].astype(float); b[y:y + ch, x:x + cw] = background
    for name, value in [("front-alpha", np.zeros((h, w))),
                        ("front-rgb", np.zeros((h, w, 3))),
                        ("rear-alpha", a), ("rear-source-rgb", s), ("scene-rgb", b)]:
        value.astype("<f8").tofile(prepared / f"{name}.f64")
    return prepared


def run(command: list[str]) -> dict:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"ordinary Rust replay failed ({process.returncode}): {process.stderr.strip()}")
    return json.loads(process.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executable", type=Path,
                        default=ROOT / "core/image-core/target/release/examples/body-layered-replay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source, alpha, background, latent, audit = frozen_source()
    audit["executableSHA256"] = sha(args.executable)
    audit["driverSHA256"] = sha(Path(__file__))
    audit["visibleGeometrySHA256"] = sha(Path(__file__).with_name("visible_rear_geometry.py"))
    h, w = source.shape[:2]
    graph = json.loads(GRAPH.read_text())
    params = graph["operations"][0]["params"]
    # The helper stops every optical ray at the anatomical region. Its entire
    # sampling halo must already be inside the frozen crop before making it a
    # full-size array; zero outside is only a memory placeholder.
    r = params["region"]
    region_box = [r["x"] * w, r["y"] * h,
                  (r["x"] + r["width"]) * w, (r["y"] + r["height"]) * h]
    x, y, cw, ch = (VIEW[k] for k in ("x", "y", "width", "height"))
    if not (x + 1 <= region_box[0] <= region_box[2] <= x + cw - 2
            and y + 1 <= region_box[1] <= region_box[3] <= y + ch - 2):
        raise ValueError("visible geometry's source sampling domain escapes the known crop")
    full_alpha = np.zeros((h, w)); full_alpha[y:y + ch, x:x + cw] = alpha
    candidate, evidence = build_visible_rear_geometry(
        params, w, h, np.zeros((h, w)), full_alpha, continuation="pchip")
    save(args.out / "source-evidence.json", evidence)
    audit["visibleGeometryRegionAndHaloKnown"] = True
    audit["addedGeometryPixels"] = evidence["addedGeometryPixels"]
    audit["status"] = "field-preflight-in-progress"
    audit["rows"] = []
    save(args.out / "report.json", audit)
    # Field-only never reads a prepared layer or fabricates an image. Both
    # geometry variants must pass before any crop-to-canvas embedding exists.
    for name, configuration in [("original", params), ("candidate", candidate)]:
        for amount in (0, -30, 30):
            label = f"{name}-{amount:+d}"
            g = copy.deepcopy(graph)
            g["operations"][0]["params"] = copy.deepcopy(configuration)
            g["operations"][0]["params"]["adjustments"]["waist"] = amount
            graph_path, field_path = args.out / f"{label}.json", args.out / f"{label}.f32"
            save(graph_path, g)
            try:
                call = run([str(args.executable), "--field-only", "--source", str(SOURCE),
                            "--graph", str(graph_path), "--save-field", str(field_path)])
            except RuntimeError as error:
                audit["rows"].append({"name": name, "amount": amount, "label": label,
                                      "fieldError": str(error), "support": {
                                          "allMovingDestinationsAndFourNeighboursKnown": False}})
                save(args.out / "report.json", audit)
                continue
            field = np.fromfile(field_path, "<f4").reshape(h, w, 2)
            support = support_audit(field, VIEW)
            audit["rows"].append({"name": name, "amount": amount, "label": label,
                                  "fieldSHA256": sha(field_path), "fieldOnly": call,
                                  "support": support})
            save(args.out / "report.json", audit)
    if not all(row["support"]["allMovingDestinationsAndFourNeighboursKnown"] for row in audit["rows"]):
        audit["status"] = ("blocked-by-ordinary-field-rejection" if any("fieldError" in row for row in audit["rows"])
                           else "blocked-by-frozen-crop-coverage")
        save(args.out / "report.json", audit)
        print(json.dumps({"status": audit["status"], "report": str(args.out / "report.json")}))
        return
    prepared = prepared_arrays(args.out, source, alpha, background, latent)
    audit["outsideCropIdentityPlaceholderNeverReadByMotion"] = True
    for row in audit["rows"]:
        label = row["label"]
        output = args.out / f"{label}.png"
        row["render"] = run([str(args.executable), "--source", str(SOURCE),
            "--prepared", str(prepared), "--graph", str(args.out / f"{label}.json"),
            "--source-reconstruction", "nearest-byte", "--output", str(output)])
        rgba = np.array(Image.open(output).convert("RGBA"))
        field = np.fromfile(args.out / f"{label}.f32", "<f4").reshape(h, w, 2)
        changed = np.any(rgba != source, axis=2)
        row["changedPixels"] = int(changed.sum())
        row["zeroFieldPixelChanges"] = int((changed & ~np.any(field != 0, axis=2)).sum())
        outside = changed.copy(); outside[y:y + ch, x:x + cw] = False
        row["outsideCropChanges"] = int(outside.sum())
        if row["zeroFieldPixelChanges"] or row["outsideCropChanges"] or (row["amount"] == 0 and changed.any()):
            raise AssertionError("ordinary replay changed a frozen or zero-request pixel")
        save(args.out / "report.json", audit)
    frozen_source()  # Checks every pinned source byte again after rendering.
    audit["frozenSourceHashesPreserved"] = True
    audit["status"] = "rendered-awaiting-independent-visual-review"
    audit["eightPointClaim"] = False
    save(args.out / "report.json", audit)
    for amount in (-30, 30):
        items = [("source", Image.fromarray(source)),
                 ("original geometry", Image.open(args.out / f"original-{amount:+d}.png")),
                 ("visible PCHIP geometry", Image.open(args.out / f"candidate-{amount:+d}.png"))]
        for tag, box, scale in [("full", (0, 0, w, h), 1),
                                 ("waist", (420, 345, 575, 495), 3)]:
            pw, ph = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
            panel = Image.new("RGB", (pw * 3, ph + 24), "#222")
            draw = ImageDraw.Draw(panel)
            for index, (label, image) in enumerate(items):
                panel.paste(image.convert("RGB").crop(box).resize((pw, ph), Image.Resampling.NEAREST),
                            (index * pw, 24))
                draw.text((index * pw + 4, 5), label, fill="white")
            panel.save(args.out / f"{tag}-{amount:+d}.png")
    print(json.dumps({"status": audit["status"], "addedGeometryPixels": audit["addedGeometryPixels"],
                      "report": str(args.out / "report.json")}))


if __name__ == "__main__":
    main()
