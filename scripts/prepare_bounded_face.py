"""Offline adapters for the constrained inverse-field experiment.

`prepare` reads existing whole-head control traces, not rendered displacement.
`sample` expands a solved grid to the single-photograph sampler's trace format.
The grid already contains all constraints; sample adds no taper or mask blend.
Images and model evidence remain in ignored test/results directories.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt, map_coordinates


def touched_nodes(points, origin, spacing, width, height):
    """All four vertices enclosing each point, for exact bilinear constraints."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not points.size:
        return set()
    uv = (points - origin) / spacing
    if np.any(uv < -1e-6) or np.any(uv > [width - 1 + 1e-6, height - 1 + 1e-6]):
        raise ValueError("constraint outside grid")
    ij = np.floor(uv).astype(int)
    ij = np.minimum(np.maximum(ij, 0), [width - 2, height - 2])
    cells = (ij[:, 1] * width + ij[:, 0])[:, None]
    return set((cells + np.array([0, 1, width, width + 1])[None, :]).ravel().tolist())


def raw_controls(trace):
    """Both native diagnostics store absolute x+.5/y+.5 point coordinates."""
    if (trace / "flow.json").exists():
        flow = json.loads((trace / "flow.json").read_text())
        source, requested = np.asarray(flow["source"], float), np.asarray(flow["requested"], float)
    else:
        rows = json.loads((trace / "controls.json").read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError("unknown native control trace format")
        source, requested = np.asarray([row["source"] for row in rows], float), np.asarray([row["requested"] for row in rows], float)
    if source.ndim != 2 or source.shape[1] != 2 or source.shape != requested.shape or not len(source) or not np.isfinite(source).all() or not np.isfinite(requested).all():
        raise ValueError("invalid raw control dimensions or nonfinite coordinates")
    return source - 0.5, requested - 0.5


def validate_trace_source(trace, source_rgba):
    """Prove absolute source coordinates by comparing native source-crop bytes.

    A crop width alone is not the source image size. Exact crop pixels plus
    absolute placement detect resized, rotated, or unrelated source traces.
    Missing pixel evidence refuses the switch; no guessed rescaling occurs.
    """
    if (trace / "layers.json").exists():
        metadata_path = trace / "layers.json"
        crop = json.loads(metadata_path.read_text())["crop"]
        crop_path = trace / "crop-source.png"
    elif (trace / "metadata.json").exists():
        metadata_path = trace / "metadata.json"
        meta = json.loads(metadata_path.read_text())
        crop = [meta[k] for k in ["left", "top", "width", "height"]]
        crop_path = trace / "crop.png"
    else:
        raise ValueError(f"missing source crop metadata: {trace}")
    if len(crop) != 4 or any(not isinstance(v, int) for v in crop):
        raise ValueError("invalid source crop coordinates")
    left, top, width, height = crop
    image_height, image_width = source_rgba.shape[:2]
    if left < 0 or top < 0 or width < 2 or height < 2 or left+width > image_width or top+height > image_height or not crop_path.exists():
        raise ValueError(f"source crop dimensions/evidence incompatible with {image_width}x{image_height}: {trace}")
    native = np.asarray(Image.open(crop_path).convert("RGBA"))
    expected = source_rgba[top:top+height, left:left+width]
    if native.shape != expected.shape or not np.array_equal(native, expected):
        raise ValueError(f"source crop pixels do not match --source at recorded coordinates: {trace}; use its exact canonical source, never mix image resolutions")
    points, requested = raw_controls(trace)
    if np.any(points < [left-1e-5, top-1e-5]) or np.any(points > [left+width-1+1e-5, top+height-1+1e-5]):
        raise ValueError("raw source controls lie outside verified source crop")
    if np.any(requested < [0, 0]) or np.any(requested > [image_width-1, image_height-1]):
        raise ValueError("raw requested controls lie outside source image coordinates")
    return {"crop": crop, "source_dimensions": [image_width, image_height],
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "crop_source_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
            "coordinate_convention": "native absolute texel centres +.5 converted once to photograph indices",
            "verification": "decoded native source-crop RGBA equals canonical source at absolute crop position"}


def retarget_attachment(attachment, source, requested):
    """One shared translation from new raw perimeter motion at source contact.

    Old materialTarget/contactTarget are deliberately unused. Four nearest
    source perimeter samples are interpolated at each distributed contact
    point, then averaged. This is geometry intent, not proof of contact/alpha.
    """
    material = np.asarray(attachment.get("material", []), float) - 0.5
    contact = np.asarray(attachment.get("contact", []), float) - 0.5
    if material.ndim != 2 or contact.ndim != 2 or material.shape[1:] != (2,) or contact.shape[1:] != (2,) or not len(material) or not len(contact) or not np.isfinite(material).all() or not np.isfinite(contact).all():
        raise ValueError("switched attachment requires finite source material and source contact geometry")
    if len(source) < 128 or source.shape != requested.shape:
        raise ValueError("switched attachment requires 128 raw perimeter controls before feature anchors")
    perimeter, targets = source[:128], requested[:128]
    distance = np.linalg.norm(contact[:, None]-perimeter[None], axis=-1)
    indices = np.argsort(distance, axis=1)[:, :4]
    chosen = np.take_along_axis(distance, indices, axis=1)
    # Refuse a contact far away from the source contour rather than attaching
    # an unrelated object. The threshold is relative to observed face span.
    max_local_distance = 0.25 * float(np.max(np.ptp(perimeter, axis=0)))
    if np.any(chosen[:, 0] > max_local_distance):
        raise ValueError("source attachment contact too far from new raw contour")
    weights = 1.0 / np.maximum(chosen, 1.0)**2
    weights /= weights.sum(axis=1, keepdims=True)
    contact_delta = np.sum((perimeter[indices]-targets[indices])*weights[:, :, None], axis=1)
    delta = contact_delta.mean(axis=0)
    return material, contact, delta, {"control_kind": "raw perimeter source/requested",
        "method": "nearest 4 inverse-distance-squared interpolation at each source contact; one shared distributed-contact average",
        "contact_count": len(contact), "indices": indices.tolist(), "weights": weights.tolist(),
        "nearest_control_distance_max": float(chosen[:, 0].max()),
        "contact_request_spread": float(np.linalg.norm(contact_delta-delta, axis=1).max()),
        "old_attachment_targets_used": False}


def swept_attachment_nodes(points, delta, origin, spacing, width, height, sweep):
    """All cells intersecting continuous movement and 1px optical fringe.

    Bounding boxes of each source point's translation segment are conservative
    geometric support. Including all four cell vertices prevents subpixel gaps;
    no pixel alpha is inferred from this support.
    """
    width, height = int(width), int(height)
    points = np.asarray(points, float)
    destination = points-delta
    low = np.minimum(points, destination)-1.0 if sweep else destination-1.0
    high = np.maximum(points, destination)+1.0 if sweep else destination+1.0
    lo_uv, hi_uv = (low-origin)/spacing, (high-origin)/spacing
    if np.any(lo_uv < 0) or np.any(hi_uv > [width-1, height-1]):
        raise ValueError("attachment source/requested sweep or optical fringe outside grid")
    lo = np.maximum(np.floor(np.nextafter(lo_uv, -np.inf)).astype(int), 0)
    hi = np.minimum(np.floor(np.nextafter(hi_uv, np.inf)).astype(int), [width-2, height-2])
    nodes = set()
    for a, b in zip(lo, hi):
        for y in range(a[1], b[1]+1):
            for x in range(a[0], b[0]+1):
                i = int(y*width+x)
                nodes.update([i, i+1, i+width, i+width+1])
    return nodes


def distant_context_nodes(support, spacing, width, height, padding, extra=None):
    """Geometry extent only; callers union this with all semantic hard nodes.

    An explicitly added edit layer may extend the old owner extent. Its support
    changes only this distance-based context rule, never source protection.
    """
    support = np.asarray(support, bool)
    if extra is not None:
        if np.shape(extra) != support.shape:
            raise ValueError("extra geometry context dimensions differ from source crop")
        support = support | np.asarray(extra, bool)
    if not support.any() or padding < 0 or not np.isfinite(padding):
        raise ValueError("invalid geometry context support or padding")
    distance = distance_transform_edt(~support)
    gx, gy = np.meshgrid(np.arange(width) * spacing[0], np.arange(height) * spacing[1])
    far = map_coordinates(distance, [gy, gx], order=1) > padding
    return set(np.flatnonzero(far).tolist())


def prepare(args):
    trace = args.trace.resolve()
    meta = json.loads((trace / "layers.json").read_text())
    left, top, cw, ch = meta["crop"]
    width, height = np.ceil((np.array([cw, ch]) - 1) / args.spacing).astype(int) + 1
    spacing = (np.array([cw, ch]) - 1) / (np.array([width, height]) - 1)
    origin = np.array([left, top], dtype=float)
    # Native whole-head traces store x+.5/y+.5; image texel indices use x/y.
    intent_trace = args.control_trace.resolve() if args.control_trace else trace
    source, requested = raw_controls(intent_trace)
    switched_source_evidence = None
    if args.control_trace:
        if not args.source:
            raise ValueError("--control-trace requires --source to verify exact source coordinates and resolution")
        canonical = np.asarray(ImageOps.exif_transpose(Image.open(args.source)).convert("RGBA"))
        switched_source_evidence = {"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "reference": validate_trace_source(trace, canonical),
            "intent": validate_trace_source(intent_trace, canonical)}
        # Fail here if independent target points do not fit the reference grid.
        touched_nodes(requested, origin, spacing, width, height)
    targets = [{"position": q.tolist(), "delta": (p - q).tolist(), "weight": 1.0}
               for p, q in zip(source, requested)]
    frozen_path = args.frozen if args.frozen else trace / "frozen.png"
    if args.freeze_evidence:
        if args.frozen:
            raise ValueError("choose frozen override or classified evidence, not both")
        frozen_path = args.freeze_evidence / "candidate-hard.png"
    frozen = np.asarray(Image.open(frozen_path).convert("L")) != 0
    if frozen.shape != (ch, cw):
        raise ValueError("frozen dimensions differ from crop")
    yy, xx = np.where(frozen)
    fixed = touched_nodes(np.stack([xx + left, yy + top], axis=-1), origin,
                          spacing, width, height)
    unknown_evidence = None
    if args.freeze_evidence:
        unknown = np.asarray(Image.open(args.freeze_evidence / "unknown.png").convert("L")) != 0
        partial = np.fromfile(args.freeze_evidence / "partial-user-protection.f32", dtype="<f4")
        evidence_report = json.loads((args.freeze_evidence / "report.json").read_text())
        hashes = evidence_report.get("input_sha256", {})
        frozen_hash = hashlib.sha256((trace / "frozen.png").read_bytes()).hexdigest()
        if evidence_report.get("crop") != meta["crop"] or frozen_hash not in hashes.values():
            raise ValueError("classified evidence belongs to a different frozen trace")
        if unknown.shape != frozen.shape or partial.size != cw * ch:
            raise ValueError("classified evidence dimensions differ")
        if np.any(partial != 0):
            raise ValueError("partial user protection needs explicit solver policy; refusing to drop it")
        if args.unknown_stiffness <= 0 or not np.isfinite(args.unknown_stiffness):
            raise ValueError("unknown evidence requires positive finite zero-motion stiffness")
        yy, xx = np.where(unknown)
        weight = args.unknown_stiffness / float(np.prod(spacing))
        targets.extend({"position": [float(x + left), float(y + top)],
                        "delta": [0.0, 0.0], "weight": weight} for y, x in zip(yy, xx))
        unknown_evidence = {"pixels": len(xx), "stiffness_per_grid_area": args.unknown_stiffness,
            "per_pixel_weight": weight,
            "report_sha256": hashlib.sha256((args.freeze_evidence / "report.json").read_bytes()).hexdigest(),
            "hard_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "unknown_sha256": hashlib.sha256((args.freeze_evidence / "unknown.png").read_bytes()).hexdigest(),
            "limitation": "Unknowns retain a soft zero-motion prior; this is not selected-material truth."}
    feature_evidence = {}
    if args.graph:
        if not args.source:
            raise ValueError("--graph requires --source to establish image coordinates")
        image_width, image_height = Image.open(args.source).size
        params = json.loads(args.graph.read_text())["operations"][0]["params"]
        for name in ["leftEye", "rightEye", "leftBrow", "rightBrow", "nose", "lips"]:
            mask = params.get("semanticMasks", {}).get(name, {})
            if mask.get("shape") != "polygon" or not mask.get("points"):
                raise ValueError(f"missing feature polygon: {name}")
            canvas = Image.new("L", (cw, ch))
            points = [(p["x"] * image_width - left, p["y"] * image_height - top)
                      for p in mask["points"]]
            ImageDraw.Draw(canvas).polygon(points, fill=255)
            core = distance_transform_edt(np.asarray(canvas) > 0) >= 2.0
            yy, xx = np.where(core)
            if not len(xx):
                raise ValueError(f"empty feature core: {name}")
            nodes = touched_nodes(np.stack([xx + left, yy + top], axis=-1), origin,
                                  spacing, width, height)
            fixed.update(nodes)
            feature_evidence[name] = {"core_pixels": len(xx), "nodes": len(nodes)}
    groups = []
    attachments_path = trace / "attachments.json"
    attachments = json.loads(attachments_path.read_text()) if attachments_path.exists() else []
    rigid_evidence = []
    for attachment in attachments:
        interpolation = None
        if args.control_trace:
            material, contact, delta, interpolation = retarget_attachment(attachment, source, requested)
            spread = 0.0  # The new shared request is one translation by construction.
        else:
            material = np.asarray(attachment["material"], dtype=float) - 0.5
            target = np.asarray(attachment["materialTarget"], dtype=float) - 0.5
            translations = material - target
            delta = np.mean(translations, axis=0)
            spread = float(np.max(np.linalg.norm(translations - delta, axis=-1)))
            if spread > 0.001:
                raise ValueError("attachment trace is not a rigid translation")
            contact = np.asarray(attachment["contact"], dtype=float) - 0.5
            old_contact_target = np.asarray(attachment["contactTarget"], dtype=float) - 0.5
            if contact.shape != old_contact_target.shape or np.max(np.linalg.norm(contact-old_contact_target-delta, axis=-1)) > 0.001:
                raise ValueError("source contact and material do not share the recorded rigid translation")
        source_points = np.concatenate([material, contact])
        nodes = swept_attachment_nodes(source_points, delta, origin, spacing, width, height,
                                       args.control_trace is not None or args.rigid_target_weight is not None)
        group = {"nodes": sorted(nodes), "delta": delta.tolist()}
        if args.rigid_target_weight is not None:
            group["target_weight"] = args.rigid_target_weight
        groups.append(group)
        rigid_evidence.append({"node_count": len(nodes), "source_spread": spread,
                               "delta": delta.tolist(), "fixed_conflicts": len(nodes & fixed),
                               "source_material": material.tolist(), "source_contact": contact.tolist(),
                               "request_interpolation": interpolation,
                               "sweep": "continuous conservative point-translation boxes with 1px fringe; all cell vertices",
                               "actual_coverage_requires_verification": True})
    # Optionally pin distant context. This is a geometry support, not alpha.
    extra_context_evidence = None
    if args.context_padding is not None:
        support = np.asarray(Image.open(trace / "owner-support.png").convert("L")) != 0
        extra = None
        if getattr(args, "context_support_extra", None):
            extra_path = args.context_support_extra
            extra = np.asarray(Image.open(extra_path).convert("L")) != 0
            extra_context_evidence = {"path": str(extra_path.resolve()),
                "sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
                "meaning": "Union with source owner only for distant geometry context; no user, semantic or feature protection removed"}
        context_nodes = distant_context_nodes(support, spacing, width, height, args.context_padding, extra)
        required_fixed = fixed.copy()
        fixed.update(context_nodes)
    else:
        if getattr(args, "context_support_extra", None):
            raise ValueError("extra context support requires an explicit context padding")
        context_nodes, required_fixed = set(), fixed.copy()
    problem = {"grid": {"origin": origin.tolist(), "spacing": spacing.tolist(),
                         "width": int(width), "height": int(height)},
               "targets": targets, "fixed_nodes": sorted(fixed), "rigid_groups": groups,
               "smoothness": args.smoothness, "regularization": 1e-6}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "controls.json").write_text(json.dumps(problem) + "\n")
    provenance = {"source_trace": str(trace), "crop": meta["crop"],
                  "input_sha256": {name: hashlib.sha256((trace / name).read_bytes()).hexdigest()
                                   for name in ["layers.json", "flow.json", "frozen.png"]},
                  "targets": "raw requested; no inverse-weight amplification or residual loop",
                  "rigid_evidence": rigid_evidence,
                  "rigid_limitation": "attachment positions reuse existing experiment evidence; some makeup labels are manual",
                  "fixed_nodes": len(fixed), "context_padding": args.context_padding}
    provenance["feature_cores"] = feature_evidence
    provenance["context_fixed_nodes"] = sorted(context_nodes)
    provenance["required_fixed_nodes"] = sorted(required_fixed)
    if extra_context_evidence:
        provenance["extra_context_support"] = extra_context_evidence
    if args.control_trace:
        intent_name = "flow.json" if (intent_trace / "flow.json").exists() else "controls.json"
        provenance["control_trace"] = {"path": str(intent_trace),
            "file": intent_name, "sha256": hashlib.sha256((intent_trace / intent_name).read_bytes()).hexdigest(),
            "meaning": "Raw edit request from an independent native target trace; reference trace supplies material constraints only."}
        provenance["control_trace"]["source_evidence"] = switched_source_evidence
    provenance["unknown_motion_prior"] = unknown_evidence
    if args.graph:
        provenance["feature_graph_sha256"] = hashlib.sha256(args.graph.read_bytes()).hexdigest()
    if args.frozen:
        provenance["frozen_override"] = {"path": str(args.frozen.resolve()),
            "sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
            "limitation": "External diagnostic mask; separate evidence must justify its ownership."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "grid": problem["grid"],
                      "targets": len(targets), "fixed_nodes": len(fixed), "rigid": rigid_evidence}))


def require_bilinear_field(field):
    # Historical bilinear outputs have no representation tag. Cubic latent
    # coefficients are NOT nodal displacement samples despite identical shape.
    if field.get("representation") not in (None, "bilinear"):
        raise ValueError("bilinear sampler/audit cannot evaluate this field representation; use its matching basis")


def sample(args):
    problem = json.loads(args.controls.read_text())
    field = json.loads(args.field.read_text())
    if "field" in field:
        field = field["field"]
    require_bilinear_field(field)
    provenance = json.loads(args.controls.with_name("provenance.json").read_text())
    grid = problem["grid"]
    left, top, cw, ch = provenance["crop"]
    values = np.asarray(field["displacement"], dtype=float).reshape(grid["height"], grid["width"], 2)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite solved displacement")
    yy, xx = np.mgrid[:ch, :cw]
    uv = [(yy + top - grid["origin"][1]) / grid["spacing"][1],
          (xx + left - grid["origin"][0]) / grid["spacing"][0]]
    offsets = np.stack([map_coordinates(values[:, :, c], uv, order=1, mode="nearest")
                        for c in range(2)], axis=-1).astype("<f4")
    args.output.mkdir(parents=True, exist_ok=False)
    offsets.tofile(args.output / "inverse.f32")
    # Pass through the already constrained geometry without a second mask gate.
    np.ones((ch, cw), dtype="<f4").tofile(args.output / "color.f32")
    Image.fromarray(np.zeros((ch, cw), dtype=np.uint8)).save(args.output / "frozen.png")
    (args.output / "metadata.json").write_text(json.dumps({"left": left, "top": top,
        "width": cw, "height": ch, "geometry": "bounded inverse grid; no post-solve taper"}) + "\n")
    print(args.output)


def audit(args):
    """Independently measure effective forward positions, not just inverse fit."""
    problem = json.loads(args.controls.read_text())
    field = json.loads(args.field.read_text())
    applied_problem = problem
    if "field" in field:
        applied_problem = field.get("applied_controls", problem)
        field = field["field"]
    require_bilinear_field(field)
    provenance = json.loads(args.controls.with_name("provenance.json").read_text())
    grid = problem["grid"]
    values = np.asarray(field["displacement"], dtype=float).reshape(grid["height"], grid["width"], 2)

    def evaluate(points):
        uv = (points - grid["origin"]) / grid["spacing"]
        return np.stack([map_coordinates(values[:, :, c], [uv[:, 1], uv[:, 0]],
                                         order=1, mode="nearest") for c in range(2)], axis=-1)

    # Whole-head flow always records 128 perimeter controls before feature anchors.
    intent_trace = Path(provenance.get("control_trace", {}).get("path", provenance["source_trace"]))
    source, requested = raw_controls(intent_trace)
    source, requested = source[:128], requested[:128]
    actual = source.copy()
    for _ in range(80):
        actual = source - evaluate(actual)
    closure = np.linalg.norm(actual + evaluate(actual) - source, axis=-1)
    displacement = actual - source
    desired = requested - source
    magnitude = np.linalg.norm(desired, axis=-1)
    active = magnitude > 0.5
    ratio = np.sum(desired[active] * displacement[active], axis=-1) / magnitude[active] ** 2
    weights = magnitude[active]
    fixed = np.asarray(problem["fixed_nodes"], dtype=int)
    flat = values.reshape(-1, 2)
    rigid_checks = []
    for index, group in enumerate(applied_problem["rigid_groups"]):
        requested_delta = np.asarray(problem["rigid_groups"][index]["delta"])
        translations = field.get("rigid_translations", [])
        actual_delta = np.asarray(translations[index]["actual"] if translations else group["delta"])
        check = {"requested": requested_delta.tolist(), "actual": actual_delta.tolist(),
            "request_error": float(np.linalg.norm(actual_delta - requested_delta)),
            "node_equality_max_error": float(np.linalg.norm(flat[group["nodes"]] - actual_delta, axis=-1).max())}
        stored = provenance.get("rigid_evidence", [])
        if index < len(stored) and "source_material" in stored[index] and "source_contact" in stored[index]:
            source_geometry = np.concatenate([stored[index]["source_material"], stored[index]["source_contact"]])
            required = swept_attachment_nodes(source_geometry, actual_delta, np.asarray(grid["origin"]),
                np.asarray(grid["spacing"]), grid["width"], grid["height"], False)
            missing = sorted(required-set(group["nodes"]))
            destination = source_geometry-actual_delta
            inverse_error = float(np.linalg.norm(destination+evaluate(destination)-source_geometry, axis=-1).max())
            check.update({"actual_geometry_fringe_missing_nodes": missing,
                "source_material_contact_inverse_closure_max": inverse_error,
                "actual_geometry_covered": not missing and inverse_error <= 1e-7 and check["node_equality_max_error"] <= 1e-7})
        else:
            check["actual_geometry_covered"] = None
            check["coverage_limitation"] = "Source material/contact not stored in this adapter provenance; use its separate evidence verifier."
        rigid_checks.append(check)
    summary = {"contour_forward_closure_max": float(closure.max()),
               "active_contour_count": int(active.sum()),
               "requested_max": float(magnitude.max()),
               "actual_max": float(np.linalg.norm(displacement, axis=-1).max()),
               "active_error_rms": float(np.sqrt(np.mean(np.sum((actual[active] - requested[active]) ** 2, axis=-1)))),
               "active_projection_ratio_mean": float(np.average(ratio, weights=weights)),
               "active_projection_ratio_range": [float(ratio.min()), float(ratio.max())],
               "fixed_nodes_max": float(np.linalg.norm(flat[fixed], axis=-1).max()) if len(fixed) else 0,
               "grid_max_movement": float(np.linalg.norm(flat, axis=-1).max()),
               "rigid_groups": rigid_checks,
               "limitation": "Geometry diagnostics do not measure beauty or background plausibility."}
    args.output.write_text(json.dumps({"summary": summary, "source": source.tolist(),
        "requested": requested.tolist(), "actual": actual.tolist()}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--control-trace", type=Path, help="Use another native trace's raw requested shape with the reference material constraints")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spacing", type=float, default=8.0)
    p.add_argument("--smoothness", type=float, default=0.05)
    p.add_argument("--context-padding", type=float)
    p.add_argument("--context-support-extra", type=Path,
                   help="Explicit source-crop geometry support added to owner before distant-context pinning; hard protection is unchanged")
    p.add_argument("--frozen", type=Path, help="Explicit diagnostic frozen-mask override, recorded in provenance")
    p.add_argument("--freeze-evidence", type=Path, help="Independent hard/unknown evidence classification directory")
    p.add_argument("--unknown-stiffness", type=float, default=1.0,
                   help="Zero-motion penalty for uncertain material per grid-cell area")
    p.add_argument("--graph", type=Path, help="Protect the six feature cores while editing face shape")
    p.add_argument("--source", type=Path)
    p.add_argument("--rigid-target-weight", type=float,
                   help="Jointly solve rigid translation; cover its full source/requested sweep")
    p.set_defaults(run=prepare)
    p = sub.add_parser("sample")
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=sample)
    p = sub.add_parser("audit")
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=audit)
    args = parser.parse_args()
    if args.command == "prepare" and (args.spacing <= 0 or args.smoothness < 0):
        parser.error("spacing must be positive and smoothness nonnegative")
    args.run(args)
