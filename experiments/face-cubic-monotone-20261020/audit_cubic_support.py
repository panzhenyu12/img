#!/usr/bin/env python3
"""Independently audit fixed/rigid B3 coefficients at actual source geometry.

Reads original controls, original solved coefficients and source provenance.
Checks every coefficient required by every actual material/contact point's
continuous +/-1px square fringe. No interpolation, repair or new optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(ids, width, height):
    result = set()
    for i in ids:
        if i < 0 or i >= width*height:
            raise ValueError("protection index outside grid")
        x, y = i % width, i // width
        result.update(yy*width+xx for yy in range(max(0, y-1), min(height, y+2))
                      for xx in range(max(0, x-1), min(width, x+2)))
    return result


def required_coefficients(points, grid):
    """Union of all 4x4 B3 stencils touched by continuous square fringes.

    nextafter deliberately includes the neighboring cell at exact knot hits.
    This is conservative also where an endpoint's basis weight is zero.
    """
    width, height = grid["width"], grid["height"]
    origin, spacing = np.asarray(grid["origin"]), np.asarray(grid["spacing"])
    lower = np.floor(np.nextafter((points-1-origin)/spacing, -np.inf)).astype(int)
    upper = np.floor(np.nextafter((points+1-origin)/spacing, np.inf)).astype(int)
    nodes, outside = set(), set()
    for a, b in zip(lower, upper):
        for yy in range(a[1]-1, b[1]+3):
            for xx in range(a[0]-1, b[0]+3):
                if 0 <= xx < width and 0 <= yy < height:
                    nodes.add(yy*width+xx)
                else:
                    outside.add((xx, yy))
    return nodes, outside


def audit(path):
    files = {name: path/name for name in ["controls.json", "field.json", "provenance.json"]}
    c, f, p = [json.loads(files[name].read_text()) for name in files]
    errors = []
    if f.get("representation") not in {
        "cubic-bspline-convolution-latent-monotone",
        "cubic-bspline-bernstein-certified",
    }:
        raise ValueError("not a cubic coefficient field")
    grid = c["grid"]
    if grid != f["grid"]:
        raise ValueError("field and original control grids differ")
    width, height = grid["width"], grid["height"]
    values = np.asarray(f["displacement"], float)
    if values.shape != (width*height, 2) or not np.isfinite(values).all():
        raise ValueError("invalid raw coefficient array")
    if c.get("background_lines"):
        errors.append("unsupported background line")
    boundary = [y*width+x for y in range(height) for x in range(width)
                if x in [0, width-1] or y in [0, height-1]]
    fixed = expand(c["fixed_nodes"]+boundary, width, height)
    fixed_max = float(np.linalg.norm(values[sorted(fixed)], axis=1).max())
    if np.any(values[sorted(fixed)] != 0):
        errors.append("expanded fixed coefficients are not exactly zero")
    groups = c["rigid_groups"]
    evidence, translations = p.get("rigid_evidence", []), f["rigid_translations"]
    if len(groups) != len(evidence) or len(groups) != len(translations):
        raise ValueError("rigid source evidence/translation count differs")
    expanded = [expand(g["nodes"], width, height) for g in groups]
    for i in range(len(expanded)):
        for j in range(i+1, len(expanded)):
            if expanded[i] & expanded[j]:
                errors.append(f"rigid halos overlap: {i}/{j}")
    rows = []
    for i, (group, source, translation, support) in enumerate(zip(groups, evidence, translations, expanded)):
        actual = np.asarray(translation["actual"], float)
        requested = np.asarray(group["delta"], float)
        if actual.shape != (2,) or not np.isfinite(actual).all():
            raise ValueError("invalid actual rigid translation")
        if translation["requested"] != group["delta"] or translation["target_weight"] != group.get("target_weight"):
            errors.append(f"rigid {i} changed original request")
        if group.get("target_weight") is None and not np.array_equal(actual, requested):
            errors.append(f"rigid {i} hard requested translation changed")
        conflict = support & fixed
        if conflict and (group.get("target_weight") is not None or np.any(requested != 0)):
            errors.append(f"rigid {i} halo conflicts with fixed; flexible zero request is also a movable DOF")
        material = np.asarray(source["source_material"], float)
        contact = np.asarray(source["source_contact"], float)
        if any(a.ndim != 2 or a.shape[1] != 2 or not len(a) or not np.isfinite(a).all() for a in [material, contact]):
            raise ValueError(f"rigid {i} lacks finite source material/contact geometry")
        points = np.concatenate([material, contact])
        moved = points-actual
        required, outside = required_coefficients(moved, grid)
        missing = required-support
        coefficient_error = float(np.linalg.norm(values[sorted(support)]-actual, axis=1).max())
        if missing or outside:
            errors.append(f"rigid {i} actual geometry+1px B3 support uncovered")
        if np.any(values[sorted(support)] != actual):
            errors.append(f"rigid {i} coefficients do not share exact actual translation")
        rows.append({"group": i, "source_material_points": len(material), "source_contact_points": len(contact),
                     "requested": requested.tolist(), "actual": actual.tolist(),
                     "original_nodes": len(group["nodes"]), "expanded_nodes": len(support),
                     "actual_material_fringe_cubic_required": len(required),
                     "missing_coefficients": sorted(missing), "out_of_grid_coefficient_coordinates": sorted(outside),
                     "fixed_conflicts": sorted(conflict), "coefficient_shared_error_max": coefficient_error,
                     "actual_destination_bbox": [moved.min(axis=0).tolist(), moved.max(axis=0).tolist()]})
    return {"case": str(path.resolve()), "input_sha256": {k: sha(v) for k,v in files.items()},
            "audit_script_sha256": sha(Path(__file__)), "representation": f["representation"],
            "expanded_fixed_coefficients": len(fixed), "fixed_coefficient_error_max": fixed_max,
            "rigid_groups": rows, "passed": not errors, "rejection_reasons": errors,
            "method": "Independently reconstructed one-ring support; full 4x4 cubic stencil for each actual source material/contact +/-1px continuous square fringe",
            "limitation": "Checks supplied source geometry, not its semantic completeness. Coefficient protection and final pixel reconstruction/appearance require distinct checks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.case:
        try:
            report = audit(path)
        except (KeyError, ValueError, OSError) as exc:
            report = {"case": str(path.resolve()), "passed": False, "rejection_reasons": [str(exc)]}
        reports.append(report)
        print(json.dumps({"case": path.name, "passed": report["passed"],
                          "expanded_fixed_coefficients": report.get("expanded_fixed_coefficients"),
                          "groups": [{k:r[k] for k in ["group","expanded_nodes","actual_material_fringe_cubic_required","missing_coefficients","coefficient_shared_error_max"]}
                                     for r in report.get("rigid_groups", [])],
                          "errors": report["rejection_reasons"]}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2)+"\n")
    raise SystemExit(0 if all(r["passed"] for r in reports) else 2)


if __name__ == "__main__":
    main()
