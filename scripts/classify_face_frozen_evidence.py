"""Offline classification of frozen-constraint evidence; not production segmentation.

Never infers another instance from missing head-class probability. Outputs hard
constraint proposals and unknown provenance separately. Unknown is not a moving
foreground label and is not permission to repaint background or change alpha.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_propagation, distance_transform_edt, label

NAMES = {
    0: "outside_audit",
    1: "explicit_user_protection",
    2: "known_other_instance",
    3: "observed_clothing",
    4: "fixed_lower_context",
    5: "unknown_near_selected_material",
    6: "unknown_remote",
    7: "explicit_partial_protection",
}
COLORS = np.array([[0, 0, 0], [255, 0, 255], [255, 64, 64], [32, 80, 255],
                   [40, 180, 255], [255, 220, 0], [255, 128, 0], [160, 80, 200]], dtype=np.uint8)


@dataclass
class Evidence:
    rgb: np.ndarray
    original_frozen: np.ndarray
    selected_head: np.ndarray
    owner_support: np.ndarray
    head: np.ndarray
    person: np.ndarray
    neck: np.ndarray
    hair: np.ndarray
    user_protection: np.ndarray
    other_instance: np.ndarray
    clothing: np.ndarray
    down_coordinate: np.ndarray
    chin_coordinate: float
    face_height: float
    parser_pitch: float


def raster_sample(mask, full_size, crop):
    """Rust PreparedMask's pixel-center raster convention, float32 arithmetic.

    Deliberately rejects unsupported protection representations instead of
    silently dropping strokes, polygons, or invalid evidence.
    """
    left, top, width, height = crop
    if mask is None:
        return np.zeros((height, width), dtype=np.float32)
    if mask.get("shape") != "raster" or mask.get("strokes"):
        raise ValueError("offline evidence reader requires a pre-rasterized mask without strokes")
    mw, mh = mask["width"], mask["height"]
    data = np.asarray(mask["data"], dtype=np.float32)
    if mw <= 0 or mh <= 0 or data.size != mw * mh or not np.isfinite(data).all():
        raise ValueError("invalid raster evidence")
    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("raster evidence outside byte range")
    data = data.reshape(mh, mw) / np.float32(255)
    fw, fh = full_size
    yy, xx = np.mgrid[top:top + height, left:left + width].astype(np.float32)
    px = np.clip((xx + np.float32(.5)) / np.float32(fw) * np.float32(mw) - np.float32(.5), 0, mw - 1)
    py = np.clip((yy + np.float32(.5)) / np.float32(fh) * np.float32(mh) - np.float32(.5), 0, mh - 1)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    x1, y1 = np.minimum(x0 + 1, mw - 1), np.minimum(y0 + 1, mh - 1)
    tx, ty = px - x0, py - y0
    result = ((data[y0, x0] * (1 - tx) + data[y0, x1] * tx) * (1 - ty)
              + (data[y1, x0] * (1 - tx) + data[y1, x1] * tx) * ty)
    return 1 - result if mask.get("invert", False) else result


def classify(e: Evidence):
    shape = e.original_frozen.shape
    arrays = [e.selected_head, e.owner_support, e.head, e.person, e.neck, e.hair,
              e.user_protection, e.other_instance, e.clothing, e.down_coordinate]
    if any(a.shape != shape for a in arrays) or e.rgb.shape != (*shape, 3):
        raise ValueError("evidence dimensions differ")
    if (not np.isfinite([e.face_height, e.parser_pitch, e.chin_coordinate]).all()
            or e.face_height <= 0 or e.parser_pitch <= 0
            or not all(np.isfinite(a).all() for a in arrays) or not np.isfinite(e.rgb).all()):
        raise ValueError("nonfinite evidence or invalid scale")
    explicit_user = e.user_protection >= 1 - 1e-6
    partial_user = (e.user_protection > 0) & ~explicit_user
    explicit_other, explicit_clothes = e.other_instance >= .8, e.clothing >= .8
    active = e.original_frozen | explicit_user | partial_user | explicit_other | explicit_clothes
    if np.any(e.selected_head):
        head_distance, nearest = distance_transform_edt(~e.selected_head, return_indices=True)
        rgb_distance = np.linalg.norm(e.rgb.astype(float) - e.rgb[nearest[0], nearest[1]].astype(float), axis=-1)
    else:
        head_distance = np.full(shape, np.inf)
        rgb_distance = np.full(shape, np.inf)
    uncertainty = 2 + 1.5 * e.parser_pitch
    # This uses the existing head/upper-neck motion scope (B + .30 * face height),
    # not a new claim that a particular RGB patch is clothing or another person.
    lower_floor = e.chin_coordinate + .30 * e.face_height
    neck_entry = e.chin_coordinate + .05 * e.face_height
    lower_allowed = ((e.down_coordinate >= neck_entry)
                     & (e.head <= 1 / 255) & ~e.owner_support & (e.hair <= 1 / 255))
    lower_seeds = lower_allowed & (e.neck >= .95)
    # The person parser can omit most of a bare shoulder. RGB may bridge that
    # gap from observed neck support, but only inside the lower context scope.
    # Chroma tolerates illumination changes; this is an explicitly heuristic
    # continuity cue, not a skin/hair/garment classifier.
    rgb = e.rgb.astype(float)
    chroma = rgb[..., :2] / np.maximum(rgb.sum(axis=-1, keepdims=True), 1)
    lower_rgb_distance = np.full(shape, np.inf)
    if np.count_nonzero(lower_seeds) >= 16:
        reference = chroma[lower_seeds]
        center = np.median(reference, axis=0)
        covariance = np.cov(reference.T) + np.eye(2) * .0001
        residual = chroma - center
        lower_rgb_distance = np.sqrt(np.maximum(0, np.einsum(
            "...i,ij,...j->...", residual, np.linalg.inv(covariance), residual)))
    continuous_lower = lower_allowed & ((e.person >= .5) | (lower_rgb_distance <= 3) | lower_seeds)
    connected_lower = binary_propagation(lower_seeds, mask=continuous_lower)
    lower_distance = (distance_transform_edt(~connected_lower) if np.any(connected_lower)
                      else np.full(shape, np.inf))
    # RGB can support local material continuity, but cannot prove identity.
    # Strong color similarity only classifies uncertainty as near material;
    # it never promotes it to selected foreground or deletes user protection.
    close = head_distance <= 3 * uncertainty
    rgb_continuous = rgb_distance <= 32.0  # Euclidean distance in byte RGB; heuristic only.
    near_evidence = close & (rgb_continuous | e.owner_support)
    result = np.zeros(shape, dtype=np.uint8)
    result[active] = 6
    result[active & near_evidence] = 5
    lower = (active & (e.person >= .8) & (lower_distance <= 2 * uncertainty)
             & (e.down_coordinate >= lower_floor)
             & (head_distance > 3 * uncertainty) & (e.hair <= 1 / 255)
             & (e.head <= 1 / 255) & ~e.owner_support & np.any(e.selected_head))
    result[lower] = 4
    result[partial_user] = 7
    result[explicit_clothes] = 3
    result[explicit_other] = 2
    result[explicit_user] = 1
    hard = np.isin(result, [1, 2, 3, 4])
    unknown = np.isin(result, [5, 6])
    return {
        "classes": result, "hard": hard, "unknown": unknown,
        "head_distance": head_distance, "rgb_distance": rgb_distance,
        "connected_lower": connected_lower, "lower_floor": float(lower_floor),
        "lower_distance": lower_distance, "lower_rgb_distance": lower_rgb_distance,
        "neck_entry": float(neck_entry), "uncertainty": float(uncertainty),
        "partial_user_strength": np.where(partial_user, e.user_protection, 0),
    }


def independent_masks(document, kind, full_size, crop):
    result = raster_sample(None, full_size, crop)
    for item in document.get(kind, []):
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            raise ValueError("independent masks require provenance")
        if kind == "other_instance_masks":
            if not item.get("instance_id") or item["instance_id"] == document.get("selected_instance_id"):
                raise ValueError("other-instance evidence needs a distinct instance identity")
        result = np.maximum(result, raster_sample(item["mask"], full_size, crop))
    return result


def load_case(case, source, independent=None):
    params = json.loads((case / "graph.json").read_text())["operations"][0]["params"]
    debug = case / "debug"
    layers = json.loads((debug / "layers.json").read_text())
    crop = layers["crop"]
    left, top, width, height = crop
    source_image = Image.open(source).convert("RGB")
    rgb = np.asarray(source_image.crop((left, top, left + width, top + height)))
    if not np.array_equal(rgb, np.asarray(Image.open(debug / "crop-source.png").convert("RGB"))):
        raise ValueError("source image differs from frozen trace")
    def mask(name):
        return raster_sample(params["semanticMasks"].get(name), source_image.size, crop)
    def saved(name):
        a = np.asarray(Image.open(debug / name).convert("L")) != 0
        if a.shape != (height, width):
            raise ValueError("trace dimensions differ")
        return a
    flow = json.loads((debug / "flow.json").read_text())
    yy, xx = np.mgrid[top:top + height, left:left + width]
    down = (np.stack([xx + .5, yy + .5], axis=-1) - flow["origin"]) @ np.asarray(flow["down"])
    headmask = params["semanticMasks"]["face"]
    pitch = max(source_image.width / headmask["width"], source_image.height / headmask["height"], 1)
    evidence = independent or {}
    e = Evidence(rgb, saved("frozen.png"), saved("head-component.png"), saved("owner-support.png"),
                 mask("face"), raster_sample(params.get("subjectMask"), source_image.size, crop),
                 mask("neck"), mask("hair"), raster_sample(params.get("protectMask"), source_image.size, crop),
                 independent_masks(evidence, "other_instance_masks", source_image.size, crop),
                 independent_masks(evidence, "clothing_masks", source_image.size, crop), down,
                 flow["bounds"][3], flow["bounds"][3] - flow["bounds"][1], pitch)
    return e, crop


def summarize(e, result, crop):
    classes = result["classes"]
    components, n = label(e.original_frozen)
    entries = []
    for i in range(1, n + 1):
        y, x = np.where(components == i)
        ys, xs = y + crop[1], x + crop[0]
        hd, rd = result["head_distance"][y, x], result["rgb_distance"][y, x]
        finite = lambda a: [float(v) if np.isfinite(v) else None for v in np.quantile(a, [0, .5, 1])]
        entries.append({"id": i, "pixels": len(x), "bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                        "classification": {NAMES[k]: int(np.sum(classes[y, x] == k)) for k in NAMES if np.any(classes[y, x] == k)},
                        "head_distance_min_median_max": finite(hd), "nearest_head_rgb_distance_min_median_max": finite(rd),
                        "lower_rgb_chroma_distance_min_median_max": finite(result["lower_rgb_distance"][y, x]),
                        "lower_contact_distance_min_median_max": finite(result["lower_distance"][y, x]),
                        "connected_lower_fraction": float(np.mean(result["connected_lower"][y, x])),
                        "body_axis_min_max": [float(e.down_coordinate[y, x].min()), float(e.down_coordinate[y, x].max())]})
    return {"original_frozen_count": int(e.original_frozen.sum()), "candidate_hard_count": int(result["hard"].sum()),
            "unknown_count": int(result["unknown"].sum()),
            "classes": {NAMES[k]: int(np.sum(classes == k)) for k in NAMES if k},
            "lower_context_floor": result["lower_floor"], "parser_uncertainty": result["uncertainty"],
            "components": entries,
            "limitation": "Unknown denotes missing evidence, never foreground ownership or permission to move/repaint. Lower context is a motion-scope hypothesis, not garment recognition. RGB support is heuristic. No automatic segmentation or aesthetic-quality claim."}


def run_case(case, source, output, independent=None):
    e, crop = load_case(case, source, independent)
    result = classify(e)
    output.mkdir(parents=True, exist_ok=False)
    for name, a in [("classes.png", result["classes"]), ("candidate-hard.png", result["hard"] * 255),
                    ("unknown.png", result["unknown"] * 255)]:
        Image.fromarray(a.astype(np.uint8)).save(output / name)
    result["partial_user_strength"].astype("<f4").tofile(output / "partial-user-protection.f32")
    overlay = e.rgb.copy()
    active = result["classes"] != 0
    overlay[active] = (.45 * overlay[active] + .55 * COLORS[result["classes"][active]]).astype(np.uint8)
    Image.fromarray(overlay).save(output / "classification-overlay.png")
    report = summarize(e, result, crop)
    report["case"] = str(case)
    report["source"] = str(source)
    report["crop"] = crop
    paths = [case / "graph.json", case / "debug/frozen.png", case / "debug/head-component.png",
             case / "debug/owner-support.png", case / "debug/flow.json", source]
    report["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report["classifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--independent-evidence", type=Path)
    args = p.parse_args()
    independent = json.loads(args.independent_evidence.read_text()) if args.independent_evidence else None
    report = run_case(args.case, args.source, args.output, independent)
    print(json.dumps({k: report[k] for k in ["case", "original_frozen_count", "candidate_hard_count", "unknown_count", "classes"]}))
