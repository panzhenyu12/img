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
}
