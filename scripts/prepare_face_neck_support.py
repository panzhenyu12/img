"""Offline source-silhouette support experiment for faceWidth.

The face/neck semantic boundary is an internal skin boundary.  It must not be
used as if it were the visible person/background silhouette.  This adapter
adds controls from the source subject silhouette near the lower face and neck.
It retains every original request, including requests that conflict with the
new support.  No output photograph or independent QA contour is an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt, map_coordinates

from prepare_bounded_face import touched_nodes, validate_trace_source
from refine_face_width_intent import smoothstep


def sample_scalar(image, points):
    return map_coordinates(image, [points[:, 1], points[:, 0]], order=1,
                           mode="constant", cval=0)


def raster(mask, size):
    if mask.get("shape") != "raster":
        raise ValueError("Expected source semantic raster")
    data = np.asarray(mask["data"], dtype=float)
    if (data.size != mask["width"] * mask["height"] or
            not np.isfinite(data).all() or data.min() < 0 or data.max() > 255):
        raise ValueError("Invalid semantic raster")
    image = Image.fromarray(data.astype(np.uint8).reshape(mask["height"], mask["width"]))
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=float) / 255


def source_contour(subject, spacing, with_topology=False):
    # OpenCV is the existing isolated algorithm tool dependency.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".run/algorithm-tools"))
    import cv2
    contours, _ = cv2.findContours((subject >= .5).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No source subject silhouette")
    # Components are kept separate; no joins are invented between people/hair.
    gy, gx = np.gradient(subject)
    result, component_ids, sequence, component_sizes = [], [], [], []
    for contour in contours:
        points = contour[:, 0].astype(float)
        if len(points) < 12:
            continue
        # Project the integer contour onto its source mask's 0.5 level set.
        for _ in range(12):
            gradient = np.stack([sample_scalar(gx, points), sample_scalar(gy, points)], axis=-1)
            norm2 = np.sum(gradient ** 2, axis=-1)
            step = ((.5 - sample_scalar(subject, points)) /
                    np.maximum(norm2, 1e-6))[:, None] * gradient
            length = np.linalg.norm(step, axis=-1)
            # Central differences underestimate the slope of a one-pixel
            # binary edge. Damping prevents alternating between its sides.
            points += .5 * step * np.minimum(1, 1 / np.maximum(length, 1e-9))[:, None]
        closed = np.vstack([points, points[0]])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=-1))]
        if arc[-1] < spacing * 4:
            continue
        at = np.arange(0, arc[-1], spacing)
        result.append(np.stack([np.interp(at, arc, closed[:, c]) for c in range(2)], axis=-1))
        component_ids.extend([len(component_sizes)]*len(at))
        sequence.extend(range(len(at)))
        component_sizes.append(len(at))
    if not result:
        raise ValueError("No usable source silhouette")
    points = np.concatenate(result)
    if with_topology:
        return points, {"component_ids": component_ids, "sequence": sequence,
                        "component_sizes": component_sizes}
    return points


def continuous_branch(ids, original_ids, topology, point_count):
    """Keep the original circular order; never join gaps by sorting by y."""
    if topology is None:
        raise ValueError("Neck continuation requires source contour topology")
    component = np.asarray(topology["component_ids"], int)
    sequence = np.asarray(topology["sequence"], int)
    sizes = np.asarray(topology["component_sizes"], int)
    if component.shape != (point_count,) or sequence.shape != (point_count,):
        raise ValueError("Source contour topology dimensions disagree")
    source_ids = original_ids[ids]
    components = np.unique(component[source_ids])
    if len(components) != 1:
        raise ValueError("Cannot join separate source subject components")
    c = int(components[0])
    if c < 0 or c >= len(sizes) or sizes[c] < 4:
        raise ValueError("Invalid source contour component")
    steps = sequence[source_ids]
    order = np.argsort(steps)
    steps, ids = steps[order], ids[order]
    if len(np.unique(steps)) != len(steps) or steps.min() < 0 or steps.max() >= sizes[c]:
        raise ValueError("Invalid source contour sequence")
    gaps = (np.roll(steps, -1)-steps) % sizes[c]
    cuts = np.flatnonzero(gaps != 1)
    if len(cuts) != 1:
        raise ValueError("Visible neck support must be one open contiguous source arc")
    start = int(cuts[0]+1) % len(ids)
    return np.roll(ids, -start)


def neck_controls(points, neck, skin, frame, amount, profile_kind="face-height", topology=None):
    """Select visible skin silhouette and apply the existing signed intent.

    Frame coordinates are photograph indices, not native texel centres.
    The support range is an explicit experimental policy, not a universal
    anatomical claim.  Occluded sides produce no controls instead of a guessed
    neck boundary through hair or clothes.
    """
    if not np.isfinite(amount) or not -100 <= amount <= 100:
        raise ValueError("faceWidth amount must be within [-100,100]")
    points = np.asarray(points, float)
    if (neck.shape != skin.shape or not np.isfinite(neck).all() or
            not np.isfinite(skin).all() or not np.any(neck >= .5) or not np.any(skin >= .5)):
        raise ValueError("Insufficient source neck/skin raster evidence")
    origin, across, down = (np.asarray(frame[k], float) for k in ["origin", "across", "down"])
    left, top, right, bottom = np.asarray(frame["bounds"], float)
    basis = np.stack([across, down])
    if not np.allclose(basis @ basis.T, np.eye(2), atol=1e-5, rtol=0):
        raise ValueError("Face frame must be orthonormal")
    width, height = right-left, bottom-top
    if min(width, height) < 4 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Invalid face geometry")
    uv = (points-origin) @ basis.T
    y = (uv[:, 1]-top)/height
    lateral = abs((uv[:, 0]-(left+right)*.5)/(width*.5))
    neck_distance = distance_transform_edt(neck < .5)
    skin_distance = distance_transform_edt(skin < .5)
    selected = ((y >= .65) & (y <= 1.18) & (lateral >= .45) &
                (lateral <= 1.55) &
                (sample_scalar(neck_distance, points) <= .12*width) &
                (sample_scalar(skin_distance, points) <= .04*width))
    original_ids = np.flatnonzero(selected)
    p, uv, y, lateral = points[selected], uv[selected], y[selected], lateral[selected]
    if len(p) < 4:
        raise ValueError("Insufficient visible source neck/face silhouette support")
    side = np.sign(uv[:, 0]-(left+right)*.5)
    profile = smoothstep(.17, .34, y) * (1-smoothstep(.88, 1., y)) * smoothstep(.08, .28, lateral)
    continuation = []
    if profile_kind == "neck-arclength":
        for sign in [-1, 1]:
            ids = np.flatnonzero(side == sign)
            if not len(ids):
                continue
            ids = continuous_branch(ids, original_ids, topology, len(points))
            if uv[ids[0], 1] > uv[ids[-1], 1]:
                ids = ids[::-1]
            if len(ids) < 4:
                raise ValueError("Too few visible neck samples for anatomical continuation")
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[ids], axis=0), axis=-1))]
            if arc[-1] < .15*height or np.max(np.diff(arc)) > .12*height:
                raise ValueError("Visible neck branch is short or disconnected")
            # The face's chin-height fade is inappropriate for the exterior
            # neck: it forces its entire movement to zero over a short strip.
            # Instead join the measured upper-neck request to the observed
            # lower skin silhouette over this branch's own source arclength.
            # Zero endpoint derivatives apply to the requested displacement,
            # not to the original anatomical curve or to the solved field.
            t = arc/arc[-1]
            entry = float(profile[ids[0]])
            profile[ids] = entry*(1-t**3*(10+t*(-15+6*t)))
            continuation.append({"side": sign, "source_indices": ids.tolist(),
                                 "length": float(arc[-1]), "entry_profile": entry})
    elif profile_kind != "face-height":
        raise ValueError("Unknown neck support profile")
    requested = p-(side*width*.065*profile*amount/100)[:, None]*across
    return p, requested, {"normalized_down": y.tolist(), "profile": profile.tolist(),
                         "side": side.tolist(), "profile_kind": profile_kind,
                         "continuation": continuation,
                         "source_contour_topology_verified": profile_kind == "neck-arclength"}


def verified_frame(trace, provenance):
    reference = Path(provenance["source_trace"]).resolve()
    if trace.resolve() != reference:
        raise ValueError("Frame must use the base problem's verified source trace")
    flow_bytes = (trace/"flow.json").read_bytes()
    expected = provenance.get("input_sha256", {}).get("flow.json")
    if not expected or hashlib.sha256(flow_bytes).hexdigest() != expected:
        raise ValueError("Frame flow differs from the base problem's recorded source hash")
    return json.loads(flow_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--frame-trace", type=Path, required=True)
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument("--profile", choices=["face-height", "neck-arclength"], default="face-height")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.weight <= 0 or not np.isfinite(args.weight):
        parser.error("Weight must be positive and finite")
    problem = json.loads((args.base/"controls.json").read_text())
    provenance = json.loads((args.base/"provenance.json").read_text())
    source = ImageOps.exif_transpose(Image.open(args.source)).convert("RGBA")
    validate_trace_source(Path(provenance["source_trace"]), np.asarray(source))
    graph_bytes = args.graph.read_bytes()
    if hashlib.sha256(graph_bytes).hexdigest() != provenance.get("feature_graph_sha256"):
        raise ValueError("Source masks must come from the original feature graph")
    params = json.loads(graph_bytes)["operations"][0]["params"]
    subject = raster(params["subjectMask"], source.size)
    neck = raster(params["semanticMasks"]["neck"], source.size)
    skin = raster(params["semanticMasks"]["skin"], source.size)
    flow = verified_frame(args.frame_trace, provenance)
    frame = {k: flow[k] for k in ["origin", "across", "down", "bounds"]}
    frame["origin"] = (np.asarray(frame["origin"])-.5).tolist()
    raw_source = np.asarray(flow["source"][:128], float)-.5
    arc_spacing = float(np.linalg.norm(np.roll(raw_source, -1, axis=0)-raw_source, axis=-1).sum()/128)
    points, topology = source_contour(subject, arc_spacing, with_topology=True)
    p, q, report = neck_controls(points, neck, skin, frame, args.amount, args.profile, topology)
    grid = problem["grid"]
    touched_nodes(q, np.asarray(grid["origin"]), np.asarray(grid["spacing"]), grid["width"], grid["height"])
    original_count = len(problem["targets"])
    problem["targets"].extend({"position": dest.tolist(), "delta": (src-dest).tolist(),
                               "weight": args.weight} for src, dest in zip(p, q))
    report.update({"source": p.tolist(), "requested": q.tolist(), "weight": args.weight,
                   "spacing": arc_spacing, "base_target_count": original_count,
                   "added_target_count": len(p), "source_photo_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                   "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
                   "policy": "Source subject 0.5 contour near visible lower-face/neck skin; signed existing faceWidth profile; all original controls retained",
                   "limitation": "Source segmentation may miss visible contour. This is a request-support experiment, not an anatomy or visual-quality guarantee."})
    provenance["source_neck_support"] = report
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"controls.json").write_text(json.dumps(problem))
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output/"neck-support.json").write_text(json.dumps(report, indent=2))
    view = source.convert("RGB"); draw = ImageDraw.Draw(view)
    for a, b in zip(p, q):
        draw.line([tuple(a), tuple(b)], fill="yellow", width=1)
        draw.ellipse((a[0]-1.5,a[1]-1.5,a[0]+1.5,a[1]+1.5),fill="red")
    view.save(args.output/"source-support.png")
    print(json.dumps({"output": str(args.output), "added": len(p), "spacing": arc_spacing}))


if __name__ == "__main__":
    main()
