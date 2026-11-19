"""Source-only visible waist contour evidence; never an optical-layer swap.

This isolated experiment updates missing geometric support connected to a
waist seed. It does not change any motion authorization or render source.
Hidden/front-covered/protected boundaries are unknown, not model truth.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def bounded_contour_interpolator(t: list[float], radii: list[float]):
    """The actual computational interpolator; undefined outside its knots."""
    from scipy.interpolate import PchipInterpolator

    if len(t) != len(radii) or len(t) < 2 or not np.isfinite(t + radii).all():
        raise ValueError("contour interpolation needs finite paired knots")
    if any(b <= a for a, b in zip(t, t[1:])) or any(radius < 0 for radius in radii):
        raise ValueError("contour knots must be ordered with nonnegative radius")
    return PchipInterpolator(t, radii, extrapolate=False)


def _sample(array: np.ndarray, x: float, y: float, width: int, height: int) -> float:
    """World pixel-center coordinates, matching normal raster mask sampling."""
    h, w = array.shape
    if not (0 <= x <= width and 0 <= y <= height):
        return 0.0
    px = float(np.clip(x * w / width - 0.5, 0, w - 1))
    py = float(np.clip(y * h / height - 0.5, 0, h - 1))
    x0, y0 = int(px), int(py)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = px - x0, py - y0
    return float((array[y0, x0] * (1 - tx) + array[y0, x1] * tx) * (1 - ty)
                 + (array[y1, x0] * (1 - tx) + array[y1, x1] * tx) * ty)


def _raster(mask: dict, name: str) -> np.ndarray:
    if mask.get("shape") != "raster" or mask.get("strokes") or mask.get("invert"):
        raise ValueError(f"{name} requires a frozen plain raster for this experiment")
    w, h = mask.get("width", 0), mask.get("height", 0)
    values = np.asarray(mask.get("data", []), dtype=float)
    if w <= 0 or h <= 0 or values.size != w * h:
        raise ValueError(f"{name} dimensions are invalid")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 255):
        raise ValueError(f"{name} must contain finite byte coverage")
    return values.reshape(h, w) / 255.0


def build_visible_rear_geometry(
    params: dict,
    width: int,
    height: int,
    front_alpha: np.ndarray,
    rear_alpha: np.ndarray,
    *,
    continuation: str = "pchip",
) -> tuple[dict, dict]:
    """Return new waist geometry and auditable, source-only observations.

    Candidate params differ only in partMask.data. rear_alpha is sampled to
    locate the first visible material edge along a connected anatomical ray;
    it is never copied wholesale into a semantic or motion-permission mask.
    The default continues computational support across unknown intervals.
    ``none`` exists only to reproduce the rejected hard-splice experiment.
    """
    if params.get("bodyPart") != "waist":
        raise ValueError("visible rear geometry currently supports waist only")
    if continuation not in ("none", "pchip"):
        raise ValueError("continuation must be none or pchip")
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    for label, array in [("front", front_alpha), ("rear", rear_alpha)]:
        if array.shape != (height, width) or not np.isfinite(array).all():
            raise ValueError(f"{label} coverage must match the source dimensions")
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"{label} coverage must be in [0,1]")
    part = _raster(params.get("partMask", {}), "partMask")
    subject = _raster(params.get("subjectMask", {}), "subjectMask")
    protect = (_raster(params["protectMask"], "protectMask")
               if params.get("protectMask") else np.zeros((1, 1)))
    landmarks = params.get("landmarks", [])
    if len(landmarks) < 3:
        raise ValueError("waist requires shoulder/waist/hip axis points")
    points = np.asarray([[p["x"] * width, p["y"] * height] for p in landmarks[:3]], float)
    if not np.isfinite(points).all():
        raise ValueError("axis points must be finite")
    bone = points[2] - points[0]
    bone_length = float(np.linalg.norm(bone))
    if bone_length < 1.0:
        raise ValueError("waist axis is degenerate")
    direction = bone / bone_length
    normal = np.array([-direction[1], direction[0]])
    length = max(1.0, bone_length * 0.65)
    origin = points[1] - direction * length * 0.5
    region = params.get("region", {"x": 0, "y": 0, "width": 1, "height": 1})

    def in_region(p: np.ndarray) -> bool:
        return (region["x"] * width <= p[0] <= (region["x"] + region["width"]) * width
                and region["y"] * height <= p[1] <= (region["y"] + region["height"]) * height)

    def read(array: np.ndarray, p: np.ndarray) -> float:
        return _sample(array, float(p[0]), float(p[1]), width, height)

    def visible_reason(p: np.ndarray) -> str | None:
        if not in_region(p) or not (0 <= p[0] <= width and 0 <= p[1] <= height):
            return "anatomical-domain-boundary"
        # Check authorization/visibility BEFORE accessing hidden rear content.
        if read(protect, p) > 0:
            return "protected"
        if read(front_alpha, p) > 0:
            return "front-occluded"
        return None

    def observe(center: np.ndarray, sign: float) -> dict:
        unknown = lambda reason: {"status": "unknown", "reason": reason, "boundaryPoint": None}
        blocked = visible_reason(center)
        if blocked:
            return unknown(blocked)
        if read(part, center) < 0.5 or read(subject, center) < 0.5:
            return unknown("no-anatomical-seed")
        previous_alpha = read(rear_alpha, center)
        if previous_alpha < 0.5:
            return unknown("seed-has-no-visible-rear")
        previous_distance = 0.0
        # A half-pixel sampling step observes the connected material run.
        # There is no jump across background gaps to another arm or object.
        for distance in np.arange(0.5, math.hypot(width, height) + 0.5, 0.5):
            point = center + normal * sign * distance
            blocked = visible_reason(point)
            if blocked:
                return unknown(blocked)
            alpha = read(rear_alpha, point)
            if alpha < 0.5:
                fraction = (previous_alpha - 0.5) / (previous_alpha - alpha)
                radius = previous_distance + (distance - previous_distance) * fraction
                boundary = center + normal * sign * radius
                return {"status": "measured", "reason": "first-visible-rear-edge",
                        "radius": float(radius), "boundaryPoint": boundary.tolist()}
            if read(subject, point) < 0.5:
                return unknown("outside-original-subject")
            previous_alpha, previous_distance = alpha, float(distance)
        return unknown("no-visible-boundary")

    count = int(math.ceil(length)) + 1
    sections = []
    for index, t in enumerate(np.linspace(0.0, 1.0, count)):
        center = origin + direction * length * t
        sections.append({"t": float(t), "center": center.tolist(),
                         "left": observe(center, 1.0), "right": observe(center, -1.0)})

    # Evidence stays untouched. A separate bounded computational curve may
    # continue across unknown rows; its anchors never read hidden rear alpha.
    curves = {}
    support_report = {}
    if continuation == "pchip":
        def coarse_anchor(center: np.ndarray, sign: float) -> float | None:
            if not in_region(center) or read(part, center) < 0.5:
                return None
            previous, previous_distance = read(part, center), 0.0
            for distance in np.arange(0.5, math.hypot(width, height) + 0.5, 0.5):
                point = center + normal * sign * distance
                if not in_region(point):
                    return None
                value = read(part, point)
                if value < 0.5:
                    fraction = (previous - 0.5) / (previous - value)
                    return previous_distance + (distance - previous_distance) * fraction
                previous, previous_distance = value, float(distance)
            return None

        for side, sign in [("left", 1.0), ("right", -1.0)]:
            known = [i for i, s in enumerate(sections) if s[side]["status"] == "measured"]
            if len(known) < 2:
                support_report[side] = {"status": "insufficient-visible-evidence"}
                continue
            knots = {i: sections[i][side]["radius"] for i in known}
            priors = []
            missing_endpoints = []
            for endpoint in (0, count - 1):
                if endpoint in knots:
                    continue
                radius = coarse_anchor(np.asarray(sections[endpoint]["center"]), sign)
                if radius is not None:
                    knots[endpoint] = radius
                    priors.append({"t": sections[endpoint]["t"], "radius": radius,
                                   "kind": "original-part-computational-prior-not-visible-truth"})
                else:
                    missing_endpoints.append(float(sections[endpoint]["t"]))
            if missing_endpoints:
                # A partial curve pasted back onto unchanged endpoint rows
                # recreates the rejected V1 unknown/known geometry seam.
                support_report[side] = {"status": "insufficient-endpoint-support",
                                        "missingEndpoints": missing_endpoints,
                                        "visibleKnots": len(known), "priorAnchors": priors}
                continue
            indices = sorted(knots)
            ts = [sections[i]["t"] for i in indices]
            values = [knots[i] for i in indices]
            curves[side] = bounded_contour_interpolator(ts, values)
            support_report[side] = {"status": "bounded-pchip", "domain": [ts[0], ts[-1]],
                                    "visibleKnots": len(known), "priorAnchors": priors,
                                    "knots": [{"t": t, "radius": radius,
                                               "kind": "visible-measurement" if i in known else "computational-prior"}
                                              for i, t, radius in zip(indices, ts, values)],
                                    "hiddenRearRead": False, "extrapolation": False,
                                    "unknownEvidencePromoted": False}

    candidate = copy.deepcopy(params)
    # Keep the caller's literal values, including fractional research masks,
    # at every unobserved/protected/unmodified pixel. Quantizing the whole mask
    # here would alter data outside the accepted measurement corridor.
    new_data = candidate["partMask"]["data"]
    mh, mw = part.shape
    added = 0
    for y in range(mh):
        for x in range(mw):
            if part[y, x] >= 1.0:
                continue
            point = np.array([(x + 0.5) * width / mw, (y + 0.5) * height / mh])
            local = point - origin
            t = float(np.dot(local, direction) / length)
            across = float(np.dot(local, normal))
            if not 0 <= t <= 1 or not in_region(point) or read(protect, point) > 0:
                continue
            if continuation == "none" and read(front_alpha, point) > 0:
                continue
            if read(subject, point) < 0.5:
                continue
            side = "left" if across >= 0 else "right"
            if continuation == "pchip":
                if side not in curves:
                    continue
                radius = float(curves[side](t))
                if not math.isfinite(radius):
                    continue
            else:
                slot = t * (count - 1)
                lo, hi = int(slot), min(int(slot) + 1, count - 1)
                first, last = sections[lo][side], sections[hi][side]
                # V1 intentionally records why hard splicing visible support
                # onto the old hidden mask fails at an unknown/known seam.
                if first["status"] != "measured" or last["status"] != "measured":
                    continue
                radius = first["radius"] + (last["radius"] - first["radius"]) * (slot - lo)
            pixel_footprint = math.hypot(normal[0] * width / mw, normal[1] * height / mh)
            coverage = float(np.clip(0.5 + (radius - abs(across)) / pixel_footprint, 0, 1))
            measured_byte = int(round(coverage * 255))
            if measured_byte > new_data[y * mw + x]:
                new_data[y * mw + x] = measured_byte
                added += 1
    report = {"method": "source-visible-waist-rays-v1" if continuation == "none" else "source-visible-waist-pchip-v2", "sourceOnly": True,
              "manualGeometry": False, "opticalLayersModified": False,
              "motionAuthorizationModified": False, "partMaskResolution": [mw, mh],
              "addedGeometryPixels": added, "sections": sections,
              "computationalSupport": support_report,
              "limitations": ["unknown evidence stays unknown; computational continuation is not hidden waist truth",
                              "connected material is not anatomical identity; missing arm ownership remains ambiguous",
                              "this experiment measures waist only and does not establish general usability"]}
    return candidate, report
