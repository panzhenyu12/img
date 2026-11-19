//! Isolated anatomical contour target candidate.
//!
//! The production target already follows the eye axis, but its vertical
//! bands use fixed fractions of face height. This candidate derives knots
//! from observed feature landmarks and projects the outward normal onto the
//! face-width axis. Width controls therefore do not shorten the chin. Smooth
//! scalar weights are not a guarantee of a smooth or injective
//! target contour. This module is research-only until real-photo holdouts
//! show a stable visual gain.

use crate::engine::percent;
use serde_json::Value;

#[path = "face_profile_normal.rs"]
mod normal;

type Point = (f32, f32);

fn dot(a: Point, b: Point) -> f32 {
    a.0 * b.0 + a.1 * b.1
}

fn sub(a: Point, b: Point) -> Point {
    (a.0 - b.0, a.1 - b.1)
}

fn add(a: Point, b: Point) -> Point {
    (a.0 + b.0, a.1 + b.1)
}

fn scale(p: Point, amount: f32) -> Point {
    (p.0 * amount, p.1 * amount)
}

fn center(mask: &Value, width: f32, height: f32) -> Option<Point> {
    let points = mask.get("points")?.as_array()?;
    let mut sum = (0.0, 0.0);
    let mut count = 0.0;
    for point in points {
        let x = point.get("x")?.as_f64()? as f32 * width;
        let y = point.get("y")?.as_f64()? as f32 * height;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        sum = add(sum, (x, y));
        count += 1.0;
    }
    (count > 0.0).then(|| scale(sum, 1.0 / count))
}

fn smooth_c2(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    // Quintic smoothstep has zero first and second derivative at both knots.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn smooth_c1(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn adaptive_saturate(raw: f32) -> f32 {
    // C-infinity and odd, unlike a hard clamp. The cap remains conservative
    // relative to the historical 0.11 normalized bound.
    0.11 * libm::tanhf(raw / 0.10)
}

fn adaptive_local_width(half_width: f32, frame_width: f32) -> f32 {
    if !half_width.is_finite() || !frame_width.is_finite() || frame_width <= 0.0 {
        return frame_width.max(1.0);
    }
    // Preserve a conservative floor/ceiling for partial or concave contours,
    // while using the observed section radius as the displacement basis.
    adaptive_radius_scale(2.0 * half_width / frame_width, 1.0) * frame_width
}

fn signed_area(points: &[Point]) -> f32 {
    points
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let b = points[(i + 1) % points.len()];
            a.0 * b.1 - b.0 * a.1
        })
        .sum::<f32>()
        * 0.5
}

/// Keep the requested contour inside a conservative injective envelope. The
/// renderer has its own field checks, but a target edge that already folds or
/// collapses can make any continuous raster field look torn at strong values.
/// Blending toward the observed contour preserves the slider direction while
/// removing the unsafe part of the request instead of emitting a folded target.
fn topology_safe_target(source: &[Point], requested: Vec<Point>) -> Vec<Point> {
    if source.len() != requested.len() || source.len() < 3 {
        return source.to_vec();
    }
    let source_area = signed_area(source);
    if !source_area.is_finite() || source_area.abs() <= 1e-4 {
        return source.to_vec();
    }
    let safe = |lambda: f32| {
        let candidate: Vec<_> = source
            .iter()
            .zip(&requested)
            .map(|(&a, &b)| add(a, scale(sub(b, a), lambda)))
            .collect();
        let area = signed_area(&candidate);
        if !area.is_finite() || area.signum() != source_area.signum() || area.abs() < source_area.abs() * 0.20 {
            return None;
        }
        for i in 0..source.len() {
            let next = (i + 1) % source.len();
            let source_edge = sub(source[next], source[i]);
            let target_edge = sub(candidate[next], candidate[i]);
            let source_length = dot(source_edge, source_edge).sqrt();
            let target_length = dot(target_edge, target_edge).sqrt();
            if source_length <= 1e-4
                || target_length < source_length * 0.20
                || dot(source_edge, target_edge) < -0.15 * source_length * target_length
            {
                return None;
            }
        }
        Some(candidate)
    };
    for lambda in [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
        if let Some(candidate) = safe(lambda) {
            return candidate;
        }
    }
    source.to_vec()
}

/// Convert a measured section width into a bounded multiplier. Keeping this
/// mapping separate makes the geometry contract testable without requiring a
/// full image render and prevents invalid narrow/concave sections from
/// producing an unbounded displacement.
fn adaptive_radius_scale(measured: f32, reference: f32) -> f32 {
    if !measured.is_finite() || !reference.is_finite() || reference <= 0.0 {
        return reference.max(1.0);
    }
    (measured * reference).clamp(0.55 * reference, 1.10 * reference)
}

fn piecewise_zero_slope(knots: &[(f32, f32)], x: f32) -> f32 {
    if knots.is_empty() {
        return 0.0;
    }
    if x <= knots[0].0 {
        return knots[0].1;
    }
    for pair in knots.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if x <= x1 {
            let t = (x - x0) / (x1 - x0).max(1e-6);
            return y0 + (y1 - y0) * smooth_c2(t);
        }
    }
    knots.last().unwrap().1
}

/// Build an opt-in contour target using feature-derived anatomical bands.
/// `frame` is `(origin, across, down, left, right, top, bottom)` where the
/// bounds are coordinates in the frame basis (the same convention as
/// `face::Frame::coordinates`), not absolute image coordinates.
pub(crate) fn target_contour(
    polygon: &[Point],
    frame: (Point, Point, Point, f32, f32, f32, f32),
    semantics: Option<&Value>,
    width: f32,
    height: f32,
    values: &Value,
) -> Result<Vec<Point>, &'static str> {
    if [
        "faceWidth",
        "temples",
        "cheekbones",
        "jaw",
        "chinWidth",
        "chinPoint",
        "foreheadWidth",
        "forehead",
        "chin",
    ]
    .iter()
    .all(|key| percent(values, key) == 0.0)
    {
        return Ok(polygon.to_vec());
    }
    if polygon.len() < 3 {
        return Err("segmented face profile requires a closed contour");
    }
    let (origin, across, down, left, right, top, bottom) = frame;
    if ![
        origin.0, origin.1, across.0, across.1, down.0, down.1, left, right, top, bottom, width,
        height,
    ]
    .iter()
    .all(|x| x.is_finite())
        || right <= left
        || bottom <= top
        || width <= 0.0
        || height <= 0.0
        || (dot(across, across) - 1.0).abs() > 1e-3
        || (dot(down, down) - 1.0).abs() > 1e-3
        || dot(across, down).abs() > 1e-3
    {
        return Err("invalid segmented face coordinate frame");
    }
    let extent = bottom - top;
    let frame_width = right - left;
    let coordinates = |p: Point| {
        let d = sub(p, origin);
        (dot(d, across), dot(d, down))
    };
    let semantic_v = |name: &str| -> Result<Option<f32>, &'static str> {
        semantics
            .and_then(|m| m.get(name))
            .map(|mask| {
                let p = center(mask, width, height).ok_or("invalid feature landmarks")?;
                Ok((coordinates(p).1 - top) / extent)
            })
            .transpose()
    };
    let eye_v = semantic_v("leftEye")?
        .zip(semantic_v("rightEye")?)
        .map(|(a, b)| (a + b) * 0.5)
        .unwrap_or(0.30);
    let mouth_v = semantic_v("lips")?.unwrap_or(eye_v + 0.32);
    let nose_v = semantic_v("nose")?.unwrap_or(eye_v + (mouth_v - eye_v) * 0.53);
    // Reject inconsistent anatomical evidence rather than sorting mislabeled
    // eyes/nose/mouth into plausible-looking positions or panicking in clamp.
    if !(eye_v >= 0.05 && nose_v >= eye_v + 0.02 && mouth_v >= nose_v + 0.02 && mouth_v <= 0.90) {
        return Err("feature landmarks are not ordered eyes/nose/mouth within the face");
    }
    let jaw_v = mouth_v + (1.0 - mouth_v) * 0.35;
    let chin_v = 0.985_f32;

    let face_width = percent(values, "faceWidth");
    let temples = percent(values, "temples");
    let cheekbones = percent(values, "cheekbones");
    let jaw = percent(values, "jaw");
    let chin_width = percent(values, "chinWidth");
    let chin_point = percent(values, "chinPoint");
    let forehead_width = percent(values, "foreheadWidth");
    let inward = |v: f32| {
        let face = piecewise_zero_slope(
            &[
                (0.0, 0.0),
                (eye_v * 0.82, 0.10),
                ((eye_v + nose_v) * 0.5, 0.68),
                (mouth_v, 0.88),
                (jaw_v, 1.0),
                (chin_v, 0.10),
                (1.0, 0.0),
            ],
            v,
        );
        let temple = piecewise_zero_slope(&[(0.0, 0.0), (eye_v, 1.0), (nose_v, 0.0)], v);
        let cheek = piecewise_zero_slope(&[(eye_v * 0.72, 0.0), (nose_v, 1.0), (mouth_v, 0.0)], v);
        let jaw_shape = piecewise_zero_slope(&[(mouth_v, 0.0), (jaw_v, 1.0), (chin_v, 0.0)], v);
        let chin_shape = piecewise_zero_slope(&[(mouth_v, 0.0), (chin_v, 1.0), (1.0, 0.0)], v);
        let forehead_shape =
            piecewise_zero_slope(&[(0.0, 0.0), (eye_v * 0.5, 1.0), (eye_v, 0.0)], v);
        (face_width * 0.065 * face
            + temples * 0.050 * temple
            + cheekbones * 0.055 * cheek
            + jaw * 0.065 * jaw_shape
            - chin_width * 0.050 * chin_shape
            - forehead_width * 0.060 * forehead_shape
            + chin_point * 0.055 * chin_shape)
            .clamp(-0.11, 0.11)
    };
    let normals = normal::outward_normals(polygon, frame_width * 0.04)?;
    Ok(polygon
        .iter()
        .enumerate()
        .map(|(index, &point)| {
            let (_, v) = coordinates(point);
            let v = ((v - top) / extent).clamp(0.0, 1.0);
            let amount = frame_width * inward(v);
            let normal = normals[index];
            let lateral_target = sub(point, scale(across, amount * dot(normal, across)));
            // Preserve the established axial control contract. Width-profile
            // experiments must not silently disable length sliders.
            let vertical = extent
                * (percent(values, "chin") * 0.075 * smooth_c1((v - 0.74) / 0.26)
                    - percent(values, "forehead") * 0.060 * (1.0 - smooth_c1(v / 0.30)));
            add(lateral_target, scale(down, vertical))
        })
        .collect())
}

/// Research candidate v2 for contour reshaping.
///
/// The legacy target expresses every control as a fixed image-space lateral
/// offset.  That makes the same slider value over-shrink a narrow jaw and can
/// create a visible kink when several controls saturate at the same hard
/// clamp.  This candidate keeps the controls in anatomical feature bands but
/// derives the displacement from the observed local half-width and transports
/// it along the observed contour normal.  The combined signed control is
/// passed through a smooth saturation curve, so changing a slider cannot
/// introduce a derivative jump at the cap.  It is intentionally opt-in until
/// real-photo review accepts the material/source path as well as the curve.
pub(crate) fn target_contour_adaptive(
    polygon: &[Point],
    frame: (Point, Point, Point, f32, f32, f32, f32),
    semantics: Option<&Value>,
    width: f32,
    height: f32,
    values: &Value,
) -> Result<Vec<Point>, &'static str> {
    let (origin, across, down, left, right, top, bottom) = frame;
    if polygon.len() < 3 {
        return Err("adaptive face profile requires a closed contour");
    }
    let extent = bottom - top;
    let frame_width = right - left;
    if ![
        origin.0, origin.1, across.0, across.1, down.0, down.1, left, right, top, bottom,
        width, height,
    ]
    .iter()
    .all(|x| x.is_finite())
        || right <= left
        || bottom <= top
        || width <= 0.0
        || height <= 0.0
        || (dot(across, across) - 1.0).abs() > 1e-3
        || (dot(down, down) - 1.0).abs() > 1e-3
        || dot(across, down).abs() > 1e-3
    {
        return Err("invalid adaptive face coordinate frame");
    }
    let coordinates = |p: Point| {
        let d = sub(p, origin);
        (dot(d, across), dot(d, down))
    };
    let semantic_v = |name: &str| -> Result<Option<f32>, &'static str> {
        semantics
            .and_then(|m| m.get(name))
            .map(|mask| {
                let p = center(mask, width, height).ok_or("invalid feature landmarks")?;
                Ok((coordinates(p).1 - top) / extent)
            })
            .transpose()
    };
    let eye_v = semantic_v("leftEye")?
        .zip(semantic_v("rightEye")?)
        .map(|(a, b)| (a + b) * 0.5)
        .unwrap_or(0.30);
    let mouth_v = semantic_v("lips")?.unwrap_or(eye_v + 0.32);
    let nose_v = semantic_v("nose")?.unwrap_or(eye_v + (mouth_v - eye_v) * 0.53);
    if !(eye_v >= 0.05 && nose_v >= eye_v + 0.02 && mouth_v >= nose_v + 0.02 && mouth_v <= 0.90)
    {
        return Err("feature landmarks are not ordered eyes/nose/mouth within the face");
    }

    // Convert the source polygon once into the anatomical frame. Intersections
    // with a horizontal section provide a robust local radius even when the
    // observed contour is tilted or non-elliptical.
    let local_polygon: Vec<Point> = polygon.iter().map(|&p| coordinates(p)).collect();
    let horizontal_half_width = |v: f32| -> Option<f32> {
        let mut intersections = Vec::new();
        for pair in local_polygon
            .iter()
            .copied()
            .zip(local_polygon.iter().copied().cycle().skip(1))
            .take(local_polygon.len())
        {
            let (a, b) = pair;
            if (a.1 <= v && v < b.1) || (b.1 <= v && v < a.1) {
                let t = (v - a.1) / (b.1 - a.1);
                intersections.push(a.0 + (b.0 - a.0) * t);
            }
        }
        if intersections.len() < 2 {
            return None;
        }
        let min = intersections.iter().copied().fold(f32::INFINITY, f32::min);
        let max = intersections
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        ((max - min).is_finite() && max > min).then_some((max - min) * 0.5)
    };
    let base_half_width = (frame_width * 0.5).max(1.0);
    let local_half_width = |v: f32| {
        let samples = [v - 0.015, v, v + 0.015]
            .into_iter()
            .filter_map(horizontal_half_width)
            .collect::<Vec<_>>();
        if samples.is_empty() {
            base_half_width
        } else {
            let mut sorted = samples;
            sorted.sort_by(f32::total_cmp);
            sorted[sorted.len() / 2]
        }
    };
    let face_width = percent(values, "faceWidth");
    let temples = percent(values, "temples");
    let cheekbones = percent(values, "cheekbones");
    let jaw = percent(values, "jaw");
    let chin_width = percent(values, "chinWidth");
    let chin_point = percent(values, "chinPoint");
    let forehead_width = percent(values, "foreheadWidth");
    let inward = |v: f32| {
        let face = piecewise_zero_slope(
            &[
                (0.0, 0.0),
                (eye_v * 0.82, 0.10),
                ((eye_v + nose_v) * 0.5, 0.68),
                (mouth_v, 0.88),
                (mouth_v + (1.0 - mouth_v) * 0.35, 1.0),
                (0.985, 0.10),
                (1.0, 0.0),
            ],
            v,
        );
        let temple = piecewise_zero_slope(&[(0.0, 0.0), (eye_v, 1.0), (nose_v, 0.0)], v);
        let cheek = piecewise_zero_slope(
            &[(eye_v * 0.72, 0.0), (nose_v, 1.0), (mouth_v, 0.0)],
            v,
        );
        let jaw_shape = piecewise_zero_slope(
            &[(mouth_v, 0.0), (mouth_v + (1.0 - mouth_v) * 0.35, 1.0), (0.985, 0.0)],
            v,
        );
        let chin_shape = piecewise_zero_slope(&[(mouth_v, 0.0), (0.985, 1.0), (1.0, 0.0)], v);
        let forehead_shape = piecewise_zero_slope(&[(0.0, 0.0), (eye_v * 0.5, 1.0), (eye_v, 0.0)], v);
        adaptive_saturate(
            face_width * 0.065 * face
                + temples * 0.050 * temple
                + cheekbones * 0.055 * cheek
                + jaw * 0.065 * jaw_shape
                - chin_width * 0.050 * chin_shape
                - forehead_width * 0.060 * forehead_shape
                + chin_point * 0.055 * chin_shape,
        )
    };
    let normals = normal::outward_normals(polygon, (frame_width * 0.04).max(1.0))?;
    let requested: Vec<_> = polygon
        .iter()
        .enumerate()
        .map(|(index, &point)| {
            let (_, local_v) = coordinates(point);
            let v = ((local_v - top) / extent).clamp(0.0, 1.0);
            let amount = adaptive_local_width(local_half_width(v), frame_width) * inward(v);
            let lateral = dot(normals[index], across).abs().clamp(0.0, 1.0);
            // Fade normal transport at a nearly vertical chin tangent; width
            // controls then reshape the jaw without shortening the chin.
            let normal_gate = smooth_c2((lateral - 0.30) / 0.55)
                * smooth_c2((0.965 - v) / 0.12);
            let normal_move = scale(normals[index], -amount * normal_gate);
            let axial_move = scale(across, -amount * dot(normals[index], across));
            let lateral_move = add(scale(axial_move, 0.72), scale(normal_move, 0.28));
            let vertical = extent
                * (percent(values, "chin") * 0.075 * smooth_c1((v - 0.74) / 0.26)
                    - percent(values, "forehead") * 0.060 * (1.0 - smooth_c1(v / 0.30)));
            add(add(point, lateral_move), scale(down, vertical))
        })
        .collect();
    Ok(topology_safe_target(polygon, requested))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn oval() -> Vec<Point> {
        (0..64)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / 64.0;
                (100.0 + 60.0 * a.cos(), 120.0 + 90.0 * a.sin())
            })
            .collect()
    }

    #[test]
    fn zero_controls_are_identity() {
        let source = oval();
        let target = target_contour(
            &source,
            (
                (100.0, 120.0),
                (1.0, 0.0),
                (0.0, 1.0),
                -60.0,
                60.0,
                -90.0,
                90.0,
            ),
            None,
            200.0,
            240.0,
            &json!({}),
        )
        .unwrap();
        assert_eq!(target, source);
    }

    #[test]
    fn face_width_moves_both_sides_inward() {
        let source = oval();
        let target = target_contour(
            &source,
            (
                (100.0, 120.0),
                (1.0, 0.0),
                (0.0, 1.0),
                -60.0,
                60.0,
                -90.0,
                90.0,
            ),
            None,
            200.0,
            240.0,
            &json!({"faceWidth": 100.0}),
        )
        .unwrap();
        let left = source
            .iter()
            .zip(&target)
            .filter(|(p, _)| (75.0..175.0).contains(&p.1))
            .min_by(|(a, _), (b, _)| a.0.total_cmp(&b.0))
            .unwrap();
        let right = source
            .iter()
            .zip(&target)
            .filter(|(p, _)| (75.0..175.0).contains(&p.1))
            .max_by(|(a, _), (b, _)| a.0.total_cmp(&b.0))
            .unwrap();
        assert!(
            left.1 .0 > left.0 .0,
            "left source={:?} target={:?}",
            left.0,
            left.1
        );
        assert!(
            right.1 .0 < right.0 .0,
            "right source={:?} target={:?}",
            right.0,
            right.1
        );
    }

    #[test]
    fn feature_knots_change_the_jaw_profile_without_moving_the_forehead() {
        let source = oval();
        let target = target_contour(
            &source,
            ((100.0, 120.0), (1.0, 0.0), (0.0, 1.0), -60.0, 60.0, -90.0, 90.0),
            Some(&json!({"leftEye":{"points":[{"x":0.4,"y":0.35}]},"rightEye":{"points":[{"x":0.6,"y":0.35}]},"lips":{"points":[{"x":0.5,"y":0.56}]}})),
            200.0,
            240.0,
            &json!({"jaw": 1.0}),
        ).unwrap();
        let forehead = source
            .iter()
            .zip(&target)
            .filter(|(p, _)| p.1 < 65.0)
            .all(|(a, b)| a == b);
        let jaw_changed = source
            .iter()
            .zip(&target)
            .any(|(a, b)| a.1 > 170.0 && a != b);
        assert!(forehead && jaw_changed);
    }

    #[test]
    fn no_supported_control_silently_becomes_identity() {
        let source = oval();
        for key in ["forehead", "foreheadWidth", "chin"] {
            for amount in [-100, 100] {
                let target = target_contour(
                    &source,
                    (
                        (100.0, 120.0),
                        (1.0, 0.0),
                        (0.0, 1.0),
                        -60.0,
                        60.0,
                        -90.0,
                        90.0,
                    ),
                    None,
                    200.0,
                    240.0,
                    &json!({key: amount}),
                )
                .unwrap();
                assert_ne!(target, source, "{key}={amount} was silently ignored");
            }
        }
    }

    #[test]
    fn inconsistent_landmarks_do_not_crash_the_editor() {
        let result = std::panic::catch_unwind(|| {
            target_contour(
                &oval(),
                (
                    (100.0, 120.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    -60.0,
                    60.0,
                    -90.0,
                    90.0,
                ),
                Some(&json!({"lips":{"points":[{"x":0.5,"y":0.875}]}})),
                200.0,
                240.0,
                &json!({"jaw":100}),
            )
        });
        assert!(result.is_ok(), "a detector outlier must not panic");
        assert!(
            result.unwrap().is_err(),
            "invalid evidence must be rejected explicitly"
        );
    }

    #[test]
    fn width_controls_do_not_change_chin_length() {
        let source = oval();
        for key in ["faceWidth", "chinWidth", "chinPoint"] {
            let target = target_contour(
                &source,
                (
                    (100.0, 120.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    -60.0,
                    60.0,
                    -90.0,
                    90.0,
                ),
                None,
                200.0,
                240.0,
                &json!({key:100}),
            )
            .unwrap();
            for i in [15, 16, 17] {
                assert!(
                    (target[i].1 - source[i].1).abs() < 1e-4,
                    "{key} changed chin length at {i}"
                );
            }
        }
    }

    #[test]
    fn signed_strength_symmetry_and_roll_are_consistent() {
        let source = oval();
        let frame = (
            (100.0, 120.0),
            (1.0, 0.0),
            (0.0, 1.0),
            -60.0,
            60.0,
            -90.0,
            90.0,
        );
        let rotate = |p: Point| (240.0 - p.1, p.0);
        let rotated: Vec<_> = source.iter().copied().map(rotate).collect();
        for key in [
            "faceWidth",
            "temples",
            "cheekbones",
            "jaw",
            "chinWidth",
            "chinPoint",
            "foreheadWidth",
            "forehead",
            "chin",
        ] {
            let positive =
                target_contour(&source, frame, None, 200.0, 240.0, &json!({key:60})).unwrap();
            let negative =
                target_contour(&source, frame, None, 200.0, 240.0, &json!({key:-60})).unwrap();
            let half =
                target_contour(&source, frame, None, 200.0, 240.0, &json!({key:30})).unwrap();
            let turned = target_contour(
                &rotated,
                (
                    rotate(frame.0),
                    (0.0, 1.0),
                    (-1.0, 0.0),
                    -60.0,
                    60.0,
                    -90.0,
                    90.0,
                ),
                None,
                240.0,
                200.0,
                &json!({key:60}),
            )
            .unwrap();
            for i in 0..source.len() {
                let expected = rotate(positive[i]);
                assert!(
                    dot(sub(turned[i], expected), sub(turned[i], expected)) < 1e-7,
                    "roll changed {key} at {i}"
                );
                for c in [0, 1] {
                    let component = |p: Point| if c == 0 { p.0 } else { p.1 };
                    assert!(
                        (component(positive[i]) + component(negative[i])
                            - 2.0 * component(source[i]))
                        .abs()
                            < 5e-5,
                        "sign bias {key}"
                    );
                    assert!(
                        (component(positive[i])
                            - component(source[i])
                            - 2.0 * (component(half[i]) - component(source[i])))
                        .abs()
                            < 5e-5,
                        "strength nonlinearity {key}"
                    );
                }
                let j = (32 + source.len() - i) % source.len();
                assert!(
                    (positive[i].0 + positive[j].0 - 200.0).abs() < 1e-4,
                    "mirror x {key}"
                );
                assert!(
                    (positive[i].1 - positive[j].1).abs() < 1e-4,
                    "mirror y {key}"
                );
            }
        }
    }

    #[test]
    fn adaptive_width_is_identity_and_symmetric_at_zero() {
        let source = oval();
        let frame = ((100.0, 120.0), (1.0, 0.0), (0.0, 1.0), -60.0, 60.0, -90.0, 90.0);
        assert_eq!(target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({})).unwrap(), source);
        let positive = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": 60})).unwrap();
        let negative = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": -60})).unwrap();
        for ((p, a), b) in source.iter().zip(&positive).zip(&negative) {
            assert!(((a.0 + b.0) * 0.5 - p.0).abs() < 1e-4);
            assert!(((a.1 + b.1) * 0.5 - p.1).abs() < 1e-4);
        }
    }

    #[test]
    fn adaptive_profile_uses_local_radius_and_does_not_shorten_chin() {
        let source: Vec<Point> = (0..128)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / 128.0;
                // A tapered lower face exposes the local-radius scaling that a
                // fixed image-space offset misses.
                let v = a.sin();
                (100.0 + (55.0 - 14.0 * v.max(0.0)) * a.cos(), 120.0 + 88.0 * v)
            })
            .collect();
        let frame = ((100.0, 120.0), (1.0, 0.0), (0.0, 1.0), -60.0, 60.0, -90.0, 90.0);
        let target = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": 100})).unwrap();
        assert_eq!(adaptive_local_width(0.275, 1.0), 0.55);
        assert_eq!(adaptive_local_width(0.55, 1.0), 1.10);
        assert!(adaptive_local_width(0.35, 1.0) < adaptive_local_width(0.50, 1.0));
        let max_y = source.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        let mut bottom = source
            .iter()
            .zip(&target)
            .filter(|(p, _)| p.1 > max_y - 1.0);
        let bottom_dy = bottom
            .by_ref()
            .map(|(a, b)| (a.1 - b.1).abs())
            .fold(0.0_f32, f32::max);
        assert!(bottom_dy < 1e-3, "width edit shortened the chin by {bottom_dy}");
    }

    #[test]
    fn adaptive_saturation_has_no_hard_clamp_jump() {
        let source = oval();
        let frame = ((100.0, 120.0), (1.0, 0.0), (0.0, 1.0), -60.0, 60.0, -90.0, 90.0);
        let a = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": 99})).unwrap();
        let b = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": 100})).unwrap();
        let c = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": 98})).unwrap();
        let first = adaptive_saturate(0.099) - adaptive_saturate(0.098);
        let second = adaptive_saturate(0.100) - adaptive_saturate(0.099);
        let third = adaptive_saturate(0.101) - adaptive_saturate(0.100);
        assert!(first > 0.0 && second > 0.0 && third > 0.0);
        assert!((first - second).abs() < 0.001 && (second - third).abs() < 0.001,
            "smooth saturation derivative jumped: {first}, {second}, {third}");
        // The target itself remains finite and monotonic around the same
        // slider range; this catches accidental hard caps in the composition.
        let first_target = a.iter().zip(&b).map(|(x, y)| ((y.0 - x.0).powi(2) + (y.1 - x.1).powi(2)).sqrt()).sum::<f32>();
        let second_target = b.iter().zip(&c).map(|(x, y)| ((y.0 - x.0).powi(2) + (y.1 - x.1).powi(2)).sqrt()).sum::<f32>();
        assert!(first_target.is_finite() && second_target.is_finite() && first_target > 0.0 && second_target > 0.0);
    }

    #[test]
    fn adaptive_target_preserves_polygon_winding_for_strong_signed_width() {
        let source = oval();
        let frame = ((100.0, 120.0), (1.0, 0.0), (0.0, 1.0), -60.0, 60.0, -90.0, 90.0);
        let signed_area = |points: &[Point]| {
            points.iter().enumerate().map(|(i, a)| {
                let b = points[(i + 1) % points.len()];
                a.0 * b.1 - b.0 * a.1
            }).sum::<f32>() * 0.5
        };
        let source_sign = signed_area(&source).signum();
        for amount in [-100, -60, 60, 100] {
            let target = target_contour_adaptive(&source, frame, None, 200.0, 240.0, &json!({"faceWidth": amount})).unwrap();
            assert_eq!(signed_area(&target).signum(), source_sign, "strong width flipped winding at {amount}");
        }
    }

    #[test]
    fn topology_guard_reduces_a_folded_target_to_the_nearest_safe_contour() {
        let source = oval();
        let mut folded = source.clone();
        for (i, point) in folded.iter_mut().enumerate() {
            if (16..=48).contains(&i) {
                point.0 = 100.0 - (point.0 - 100.0) * 2.8;
            }
        }
        let target = topology_safe_target(&source, folded);
        assert_eq!(signed_area(&target).signum(), signed_area(&source).signum());
        for i in 0..source.len() {
            let next = (i + 1) % source.len();
            let a = sub(source[next], source[i]);
            let b = sub(target[next], target[i]);
            assert!(dot(a, b) >= -0.15 * dot(a, a).sqrt() * dot(b, b).sqrt());
            assert!(dot(b, b).sqrt() >= dot(a, a).sqrt() * 0.20);
        }
    }
}
