//! Public entry regression: measured eye-width changes the aperture, not the iris.
use image_core::{apply_graph_json, ImageError, RgbaImage};
use serde_json::{json, Value};

fn fixture() -> (RgbaImage, Value) {
    let pixels = (0..160)
        .flat_map(|y| {
            (0..256).flat_map(move |x| [x as u8, y as u8, ((x * 31 + y * 17) % 251) as u8, 255])
        })
        .collect();
    let eye = |cx: f64| {
        let points: Vec<_> = (0..32)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 32.0;
                json!({"x":(cx+24.0*angle.cos())/256.0,"y":(76.0+6.0*angle.sin())/160.0})
            })
            .collect();
        json!({"shape":"polygon","points":points,"feather":0})
    };
    let iris = |cx: f64| {
        json!({"iris":{
        "center":{"x":cx/256.0,"y":76.0/160.0},
        "boundary":[{"x":(cx+9.0)/256.0,"y":76.0/160.0},
                    {"x":cx/256.0,"y":67.0/160.0},
                    {"x":(cx-9.0)/256.0,"y":76.0/160.0},
                    {"x":cx/256.0,"y":85.0/160.0}]}})
    };
    (
        RgbaImage::new(256, 160, pixels).unwrap(),
        json!({
            "adjustments":{"eyeWidth":30},
            "semanticMasks":{"leftEye":eye(80.0),"rightEye":eye(176.0)},
            "eyeGeometry":{"leftEye":iris(75.0),"rightEye":iris(181.0)}
        }),
    )
}

fn run(source: &RgbaImage, kind: &str, params: &Value) -> Result<RgbaImage, ImageError> {
    apply_graph_json(
        source,
        &json!({"operations":[{"type":kind,"params":params}]}).to_string(),
    )
}

#[test]
fn measured_width_public_graph_moves_corners_both_ways_without_warping_iris() {
    let (source, mut params) = fixture();
    for amount in [-30, 30] {
        params["adjustments"]["eyeWidth"] = json!(amount);
        let output = run(&source, "feature.adjust", &params).unwrap();
        let mut legacy = params.clone();
        legacy.as_object_mut().unwrap().remove("eyeGeometry");
        let legacy = run(&source, "feature.adjust", &legacy).unwrap();
        let mut observed = 0;
        let mut legacy_changed = 0;
        for (cx, iris_cx) in [(80, 75), (176, 181)] {
            for x in [cx - 24, cx + 24] {
                let i = (76 * 256 + x) as usize * 4;
                let delta = output.pixels[i] as i32 - source.pixels[i] as i32;
                assert!(
                    delta * (cx - x) * amount > 0,
                    "canthus must respond in requested direction: {amount}, {x}, {delta}"
                );
            }
            for y in 71..=81 {
                for x in iris_cx - 8..=iris_cx + 8 {
                    let (dx, dy) = ((x - iris_cx) as f64, (y - 76) as f64);
                    // A strict interior stays away from polygon approximation and filtering halos.
                    if dx * dx + dy * dy < 64.0
                        && ((x - cx) as f64 / 24.0).powi(2) + (dy / 6.0).powi(2) < 0.8
                    {
                        let i = (y * 256 + x) as usize * 4;
                        observed += 1;
                        assert_eq!(
                            &output.pixels[i..i + 4],
                            &source.pixels[i..i + 4],
                            "observed iris at {x},{y}"
                        );
                        legacy_changed +=
                            usize::from(legacy.pixels[i..i + 4] != source.pixels[i..i + 4]);
                    }
                }
            }
        }
        assert!(
            observed > 100 && legacy_changed > 50,
            "nonempty material and discriminating old-path control"
        );
        assert_eq!(
            &output.pixels[..256 * 4],
            &source.pixels[..256 * 4],
            "outside support"
        );
    }
}

#[test]
fn measured_width_composition_keeps_size_motion_without_extra_iris_stretch() {
    let (source, base) = fixture();
    for size in [-30, 30] {
        let mut size_only = base.clone();
        size_only["adjustments"] = json!({"eyeSize":size});
        let expected = run(&source, "feature.adjust", &size_only).unwrap();
        for width in [-30, 30] {
            let mut params = size_only.clone();
            params["adjustments"]["eyeWidth"] = json!(width);
            let output = run(&source, "feature.adjust", &params).unwrap();
            let mut size_changes = 0;
            for cx in [75, 181] {
                for y in 74..=78 {
                    for x in cx - 3..=cx + 3 {
                        let i = (y * 256 + x) as usize * 4;
                        for channel in 0..4 {
                            assert!(
                                (output.pixels[i + channel] as i16
                                    - expected.pixels[i + channel] as i16)
                                    .abs()
                                    <= 1,
                                "width adds iris distortion to size: {size},{width},{x},{y}"
                            );
                        }
                        size_changes +=
                            usize::from(expected.pixels[i..i + 4] != source.pixels[i..i + 4]);
                    }
                }
            }
            assert!(size_changes > 20, "eyeSize still changes iris material");
        }
    }
}

#[test]
fn measured_eye_input_is_strict_even_when_adjustment_is_zero() {
    let (source, mut params) = fixture();
    params["adjustments"]["eyeWidth"] = json!(0);
    assert_eq!(run(&source, "feature.adjust", &params).unwrap(), source);
    let mut invalid = Vec::new();
    for replacement in [
        json!({}),
        json!({"leftEye":{}}),
        json!({"leftEye":{"iris":{}}}),
        json!({"unknownEye":{}}),
    ] {
        let mut p = params.clone();
        p["eyeGeometry"] = replacement;
        invalid.push(p);
    }
    let mut p = params.clone();
    p["eyeGeometry"]["leftEye"]["iris"]["center"]["x"] = json!(-0.01);
    invalid.push(p);
    let mut p = params.clone();
    p["eyeGeometry"]["leftEye"]["iris"]["center"]["visibility"] = json!(1);
    invalid.push(p);
    let mut p = params.clone();
    p["eyeGeometry"]["leftEye"]["iris"]["boundary"] = json!([]);
    invalid.push(p);
    let mut p = params.clone();
    p["eyeGeometry"]["leftEye"]["iris"]["boundary"][2] =
        p["eyeGeometry"]["leftEye"]["iris"]["boundary"][0].clone();
    invalid.push(p);
    let mut p = params.clone();
    p["semanticMasks"]
        .as_object_mut()
        .unwrap()
        .remove("leftEye");
    invalid.push(p);
    let mut p = params.clone();
    p["semanticMasks"]["leftEye"] =
        json!({"shape":"ellipse","x":0.1,"y":0.1,"width":0.2,"height":0.2});
    invalid.push(p);
    for p in invalid {
        assert!(
            run(&source, "feature.adjust", &p).is_err(),
            "invalid measured geometry accepted: {p}"
        );
    }
    assert!(run(
        &source,
        "face.adjust",
        &json!({"adjustments":{},"eyeGeometry":params["eyeGeometry"]})
    )
    .is_err());
}

#[test]
fn explicit_eye_geometry_cannot_silently_mix_with_unmeasured_active_eye() {
    let (source, mut params) = fixture();
    params["eyeGeometry"]
        .as_object_mut()
        .unwrap()
        .remove("rightEye");
    assert!(run(&source, "feature.adjust", &params).is_err());
    params["semanticMasks"]
        .as_object_mut()
        .unwrap()
        .remove("rightEye");
    assert!(
        run(&source, "feature.adjust", &params).is_ok(),
        "one explicitly selected measured eye is valid"
    );
}

#[test]
fn iris_sampling_conflict_is_not_silently_clamped_to_no_effect() {
    let (source, mut params) = fixture();
    // Sideward gaze: the outer canthus cannot cross this visible iris.
    let iris = &mut params["eyeGeometry"]["leftEye"]["iris"];
    iris["center"]["x"] = json!(70.0 / 256.0);
    for p in iris["boundary"].as_array_mut().unwrap() {
        p["x"] = json!(p["x"].as_f64().unwrap() - 5.0 / 256.0);
    }
    params["adjustments"]["eyeWidth"] = json!(-100);
    let error = run(&source, "feature.adjust", &params)
        .err()
        .expect("canthus crosses observed iris")
        .to_string();
    assert!(error.contains("iris sampling guard"), "{error}");
}
