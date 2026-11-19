//! Source-anchored light eyelid closure, independent of image composition.
//!
//! This ordinary native path extracts the established canthi-frame experiment.
//! It deliberately does not enable the pending generic eyeHeight operation:
//! a separate ocular/lid material compositor is required to retain the iris.

use super::{
    feature_flow_steps, feature_polygon_signed_distance, integrate_feature_flow,
    invalid, percent, symmetry_feature, FeatureBrowAnchor, FeatureFrame,
    FeatureMotionDomain, LocalWarp, Region,
};
use crate::{ImageError, RgbaImage};
use serde_json::{json, Value};
use std::sync::Arc;

const LEFT_BROW: &[usize] = &[70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const RIGHT_BROW: &[usize] = &[300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
const LEFT_EYE: &[usize] = &[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE: &[usize] = &[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384];
const FACE_OVAL: &[usize] = &[
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
];

/// Absolute source pixel-index coordinates, in row-major destination order.
/// `stationary` always has width*height entries; unlike the historical test
/// observer, an empty protected-mask set is represented by explicit false.
#[derive(Clone, Debug)]
pub struct EyeClosureField {
    pub width: u32,
    pub height: u32,
    pub inverse: Vec<[f32; 2]>,
    pub stationary: Vec<bool>,
    pub integration_steps: usize,
}

fn dimensions(width: u32, height: u32) -> Result<usize, ImageError> {
    if width <= 1 || height <= 1 || width as u64 * height as u64 > 32_000_000 {
        return Err(ImageError::InvalidDimensions);
    }
    Ok(width as usize * height as usize)
}

fn closing_amount(amount: i32) -> Result<(), ImageError> {
    if !(-30..=0).contains(&amount) {
        return Err(invalid("eye closure supports only integer light-closing strengths from -30 to 0"));
    }
    Ok(())
}

/// Convert a full MediaPipe source observation into this field's parameters.
///
/// Input landmarks are normalized physical image coordinates. Feature points
/// are converted exactly once as `(normalized * dimension - .5) / dimension`.
/// The face admission region uses the ORIGINAL physical-normalized oval plus
/// 20% padding: it is an outer boundary, not a pixel-center feature point.
/// Neither this adapter nor the field detects anatomy or fits source curves.
pub fn eye_closure_params_from_landmarks(
    width: u32,
    height: u32,
    landmarks: &[[f64; 2]],
    amount: i32,
) -> Result<Value, ImageError> {
    dimensions(width, height)?;
    closing_amount(amount)?;
    if landmarks.len() < 478 || landmarks.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("eye closure requires at least 478 finite source landmarks"));
    }
    let size = [width as f64, height as f64];
    for ids in [LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE, FACE_OVAL] {
        for &index in ids {
            for axis in 0..2 {
                let point = landmarks[index][axis] * size[axis] - 0.5;
                if point < 0.0 || point > size[axis] - 1.0 {
                    return Err(invalid("adapted eye feature leaves the source pixel-center domain; no clamping"));
                }
            }
        }
    }
    let polygon = |ids: &[usize]| -> Vec<Value> {
        ids.iter().map(|&index| json!({
            "x": (landmarks[index][0] * size[0] - 0.5) / size[0],
            "y": (landmarks[index][1] * size[1] - 0.5) / size[1],
        })).collect()
    };
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for &index in FACE_OVAL {
        for axis in 0..2 {
            low[axis] = low[axis].min(landmarks[index][axis]);
            high[axis] = high[axis].max(landmarks[index][axis]);
        }
    }
    for axis in 0..2 {
        let padding = (high[axis] - low[axis]) * 0.2;
        low[axis] = (low[axis] - padding).max(0.0);
        high[axis] = (high[axis] + padding).min(1.0);
        if high[axis] <= low[axis] {
            return Err(invalid("source face region has no extent"));
        }
    }
    let mut masks = serde_json::Map::new();
    for (name, ids) in [
        ("leftEye", LEFT_EYE), ("rightEye", RIGHT_EYE),
        ("leftBrow", LEFT_BROW), ("rightBrow", RIGHT_BROW),
        ("face", FACE_OVAL), ("faceOutline", FACE_OVAL),
    ] {
        masks.insert(name.into(), json!({
            "shape": "polygon", "points": polygon(ids), "feather": 2, "invert": false,
        }));
    }
    Ok(json!({
        "region": {"x": low[0], "y": low[1], "width": high[0]-low[0], "height": high[1]-low[1]},
        "landmarks": polygon(FACE_OVAL), "semanticMasks": masks,
        "adjustments": {"eyeHeight": amount},
    }))
}

fn validate_points(points: &Value, source: &RgbaImage) -> Result<(), ImageError> {
    let points = points.as_array().filter(|p| p.len() >= 3)
        .ok_or_else(|| invalid("eye closure requires source feature polygons"))?;
    for point in points {
        let object = point.as_object().filter(|p| p.len() == 2 && p.contains_key("x") && p.contains_key("y"))
            .ok_or_else(|| invalid("source feature points require only finite x/y"))?;
        for (axis, size) in [("x", source.width), ("y", source.height)] {
            let value = object[axis].as_f64().filter(|v| v.is_finite())
                .ok_or_else(|| invalid("source feature points require only finite x/y"))?;
            if value * (size as f64) < 0.0 || value * (size as f64) > size as f64 - 1.0 {
                return Err(invalid("native source feature leaves the pixel-index domain"));
            }
        }
    }
    Ok(())
}

fn validate_params(source: &RgbaImage, params: &Value) -> Result<i32, ImageError> {
    let object = params.as_object().ok_or_else(|| invalid("eye closure parameters must be an object"))?;
    if object.keys().any(|key| !matches!(key.as_str(), "region" | "landmarks" | "semanticMasks" | "adjustments")) {
        return Err(invalid("eye closure field accepts no extra adjustments, selections or research parameters"));
    }
    let values = params.get("adjustments").and_then(Value::as_object)
        .filter(|v| v.len() == 1 && v.contains_key("eyeHeight"))
        .ok_or_else(|| invalid("eye closure requires only the eyeHeight adjustment"))?;
    let amount = values["eyeHeight"].as_f64()
        .filter(|v| v.is_finite() && (-30.0..=0.0).contains(v) && v.fract()==0.0)
        .ok_or_else(|| invalid("eye closure supports only integer light-closing strengths from -30 to 0"))? as i32;
    let region = params.get("region").and_then(Value::as_object)
        .filter(|v| v.len() == 4)
        .ok_or_else(|| invalid("eye closure requires an explicit source region"))?;
    let mut coordinates = [0.0; 4];
    for (i, key) in ["x", "y", "width", "height"].iter().enumerate() {
        coordinates[i] = region.get(*key).and_then(Value::as_f64).filter(|v| v.is_finite())
            .ok_or_else(|| invalid("eye closure region must be finite"))?;
    }
    let [x, y, w, h] = coordinates;
    if x < 0.0 || y < 0.0 || w <= 0.0 || h <= 0.0 || x+w > 1.0 || y+h > 1.0 {
        return Err(invalid("eye closure region must stay inside the physical normalized image boundary"));
    }
    let masks = params.get("semanticMasks").and_then(Value::as_object)
        .ok_or_else(|| invalid("eye closure requires explicit source eye and brow polygons"))?;
    for name in ["leftEye", "rightEye", "leftBrow", "rightBrow"] {
        if !masks.contains_key(name) {
            return Err(invalid("eye closure requires both source eyes and both source brows"));
        }
    }
    for (name, mask) in masks {
        if !matches!(name.as_str(), "leftEye" | "rightEye" | "leftBrow" | "rightBrow" | "face" | "faceOutline") {
            return Err(invalid("unsupported semantic material in eye closure field parameters"));
        }
        let object = mask.as_object().ok_or_else(|| invalid("eye closure semantic masks must be polygons"))?;
        if object.keys().any(|key| !matches!(key.as_str(), "shape" | "points" | "feather" | "invert"))
            || mask.get("shape").and_then(Value::as_str) != Some("polygon")
            || mask.get("invert").is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(invalid("eye closure accepts only ordinary non-inverted source polygons"));
        }
        if mask.get("feather").is_some_and(|v| !v.as_f64().is_some_and(|n| n.is_finite() && (0.0..=100.0).contains(&n))) {
            return Err(invalid("invalid source polygon feather"));
        }
        validate_points(mask.get("points").unwrap_or(&Value::Null), source)?;
    }
    if let Some(points) = params.get("landmarks") {
        validate_points(points, source)?;
    }
    Ok(amount)
}

// This is the historical canthi-frame algorithm, not the detector centroid or
// a global face axis. Keep ordering and f32 arithmetic unchanged for replay.
fn eyelid_source_frame(params: &Value, name: &str, source: &RgbaImage) -> Option<FeatureFrame> {
    let points = symmetry_feature(params, name, source)?;
    let mut endpoints = None;
    let mut longest = 0.0_f32;
    for (i, &a) in points.iter().enumerate() {
        for &b in &points[i + 1..] {
            let squared = (b.0 - a.0).powi(2) + (b.1 - a.1).powi(2);
            if squared > longest {
                longest = squared;
                endpoints = Some((a, b));
            }
        }
    }
    let (mut a, mut b) = endpoints?;
    if b.0 < a.0 { std::mem::swap(&mut a, &mut b); }
    let width = longest.sqrt();
    if width < 2.0 { return None; }
    let across = ((b.0 - a.0) / width, (b.1 - a.1) / width);
    let center = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
    let height = points.iter().map(|&(x, y)| {
        ((x - center.0) * -across.1 + (y - center.1) * across.0).abs() * 2.0
    }).fold(0.0_f32, f32::max);
    if height < 1.0 || height >= width { return None; }
    for k in 1..32 {
        let t = k as f32 / 32.0;
        let point = (a.0 * (1.0 - t) + b.0 * t, a.1 * (1.0 - t) + b.1 * t);
        if feature_polygon_signed_distance(&points, point) > 0.01 { return None; }
    }
    Some(FeatureFrame { center, across, width, height })
}

/// Prepare a light-closing field for an integer strength in [-30, 0].
/// The continuous-in-strength generator is unchanged from the established
/// 0/-15/-30 replay; expanded integer strengths require their own QA evidence.
///
/// `params` contains native feature points normalized as pixel-index/dimension.
/// Use `eye_closure_params_from_landmarks` for physical-normalized observations;
/// this function NEVER applies another half-pixel shift. Region bounds retain
/// their original physical-normalized contract. The result is a geometry field,
/// not an edited image or an assurance that ocular/lash materials are correct.
pub fn prepare_eye_closure_field(source: &RgbaImage, params: &Value) -> Result<EyeClosureField, ImageError> {
    prepare_eye_closure_field_impl::<true>(source, params)
}

// Only the two ordinary eyelid warps below contribute velocity. Outside both
// ellipses their initial velocity is exactly zero, so the existing integrator
// would return the input point before any RK4 steps. This admission avoids its
// coverage lookup and generic dispatch; it does not reduce integration work
// for any moving point. Preserve the velocity function's f32 operation order
// rather than using a rounded bounding box to classify support boundaries.
fn closure_support_contains(point: (f32, f32), changes: &[LocalWarp]) -> bool {
    changes.iter().any(|warp| {
        let delta = (point.0 - warp.center.0, point.1 - warp.center.1);
        let u = delta.0 * warp.across.0 + delta.1 * warp.across.1;
        let v = -delta.0 * warp.across.1 + delta.1 * warp.across.0;
        let d = (u / warp.radius.0).powi(2) + (v / warp.radius.1).powi(2);
        // Match `if d >= 1 { continue }` in feature_flow_velocity, including
        // admitting an unexpected NaN to the ordinary path for validation.
        !(d >= 1.0)
    })
}

fn prepare_eye_closure_field_impl<const SUPPORT_ADMISSION: bool>(
    source: &RgbaImage,
    params: &Value,
) -> Result<EyeClosureField, ImageError> {
    let count = dimensions(source.width, source.height)?;
    if count.checked_mul(4) != Some(source.pixels.len()) {
        return Err(ImageError::InvalidDimensions);
    }
    let amount = validate_params(source, params)?;
    let mut frames = Vec::new();
    for name in ["leftEye", "rightEye"] {
        frames.push(eyelid_source_frame(params, name, source)
            .ok_or_else(|| invalid(&format!("{name}: source aperture does not support an observed eyelid closure")))?);
    }
    let identity = || (0..count).map(|i| [(i % source.width as usize) as f32, (i / source.width as usize) as f32]).collect();
    if amount == 0 {
        return Ok(EyeClosureField { width: source.width, height: source.height,
            inverse: identity(), stationary: vec![true; count], integration_steps: 0 });
    }
    let brows = Arc::new(["leftBrow", "rightBrow"].iter().map(|name| {
        FeatureBrowAnchor::new(symmetry_feature(params, name, source).expect("validated source brow polygon"))
    }).collect());
    let amount = percent(&params["adjustments"], "eyeHeight");
    let changes: Vec<_> = frames.into_iter().map(|frame| {
        let scale_x = frame.width / 0.26;
        let scale_y = frame.height / 0.10;
        LocalWarp {
            center: frame.center, across: frame.across,
            radius: ((0.2 * scale_x).max(1.0), (0.13 * scale_y).max(1.0)),
            scale: (0.0, -0.3 * amount), translation: (0.0, 0.0), angle: 0.0,
            preserve_eyes: false, whole_eye_size: false, whole_eye_tilt: false,
            whole_eye_spacing: false, eye_width_material: None, whole_eye_inner: 0.0,
            eye_size_brows: Some(Arc::clone(&brows)),
        }
    }).collect();
    let region = Region::read(params.get("region"));
    let stationary = vec![false; count];
    let raw_coverage = |x: f32, y: f32| {
        if region.contains(x / source.width as f32, y / source.height as f32) { 1.0 } else { 0.0 }
    };
    let domain = FeatureMotionDomain::new(source, &stationary, raw_coverage);
    let coverage = |x, y| domain.coverage(x, y);
    let steps = feature_flow_steps(&changes, &[], &[], domain.gradient);
    let mut inverse = Vec::with_capacity(count);
    for y in 0..source.height {
        for x in 0..source.width {
            let point = (x as f32, y as f32);
            let mapped = if !region.contains(point.0 / source.width as f32, point.1 / source.height as f32)
                || SUPPORT_ADMISSION && !closure_support_contains(point, &changes) {
                point
            } else {
                integrate_feature_flow(point, &changes, &[], &[], &coverage, steps, 1.0)
            };
            if !mapped.0.is_finite() || !mapped.1.is_finite()
                || mapped.0 < 0.0 || mapped.1 < 0.0
                || mapped.0 > (source.width-1) as f32 || mapped.1 > (source.height-1) as f32 {
                return Err(invalid("eye closure field leaves the source pixel-index domain"));
            }
            inverse.push([mapped.0, mapped.1]);
        }
    }
    Ok(EyeClosureField { width: source.width, height: source.height,
        inverse, stationary, integration_steps: steps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> Vec<[f64; 2]> {
        let mut points = vec![[0.5, 0.5]; 478];
        for (i, &id) in FACE_OVAL.iter().enumerate() {
            let a = i as f64 * std::f64::consts::TAU / FACE_OVAL.len() as f64;
            points[id] = [0.5 + 0.35*a.cos(), 0.5 + 0.4*a.sin()];
        }
        for (ids, cx, cy, rx, ry) in [
            (LEFT_EYE, 40.5, 45.5, 12.0, 5.0),
            (RIGHT_EYE, 88.5, 45.5, 12.0, 5.0),
            (LEFT_BROW, 40.5, 30.5, 14.0, 2.0),
            (RIGHT_BROW, 88.5, 30.5, 14.0, 2.0),
        ] {
            for (i, &id) in ids.iter().enumerate() {
                let a = i as f64 * std::f64::consts::TAU / ids.len() as f64;
                points[id] = [(cx+rx*a.cos())/128.0, (cy+ry*a.sin())/96.0];
            }
        }
        points
    }

    fn source() -> RgbaImage {
        RgbaImage::new(128, 96, vec![127; 128*96*4]).unwrap()
    }

    #[test]
    fn source_adapter_shifts_feature_centers_once_and_preserves_region_boundary() {
        let points = observation();
        let params = eye_closure_params_from_landmarks(128, 96, &points, -15).unwrap();
        let first = &params["semanticMasks"]["leftEye"]["points"][0];
        assert_eq!(first["x"].as_f64().unwrap()*128.0, 52.0);
        assert_eq!(first["y"].as_f64().unwrap()*96.0, 45.0);
        let frame = eyelid_source_frame(&params, "leftEye", &source()).unwrap();
        assert_eq!(frame.center, (40.0, 45.0));
        // Original physical oval has x bounds .15/.85. Its 20% padded
        // boundary is .01/.99 and must not inherit the point shift.
        assert!((params["region"]["x"].as_f64().unwrap()-0.01).abs() < 1e-14);
        assert!((params["region"]["width"].as_f64().unwrap()-0.98).abs() < 1e-14);
        let field = prepare_eye_closure_field(&source(), &params).unwrap();
        assert_eq!(field.inverse[45*128+40], [40.0,45.0]);
    }

    #[test]
    fn light_closure_has_real_vertical_motion_exact_zero_and_positive_sampled_cells() {
        let source = source();
        let points = observation();
        let zero = prepare_eye_closure_field(&source,
            &eye_closure_params_from_landmarks(128,96,&points,0).unwrap()).unwrap();
        assert!(zero.stationary.iter().all(|&v|v));
        for (i, point) in zero.inverse.iter().enumerate() {
            assert_eq!(*point, [(i%128) as f32, (i/128) as f32]);
        }
        let fields: Vec<_> = [-15,-30].into_iter().map(|amount| {
            prepare_eye_closure_field(&source,
                &eye_closure_params_from_landmarks(128,96,&points,amount).unwrap()).unwrap()
        }).collect();
        for field in &fields {
            assert_eq!(field.stationary.len(),128*96);
            assert!(field.stationary.iter().all(|&v| !v));
            assert!(field.inverse[41*128+40][1] < 40.95,
                "upper eyelid inverse must move into the original upper side");
            assert!(field.inverse[49*128+40][1] > 49.05);
            assert_eq!(field.inverse[30*128+40], [40.0,30.0], "source brow stays fixed");
            for y in 0..95 {
                for x in 0..127 {
                    let p=field.inverse[y*128+x];
                    let px=field.inverse[y*128+x+1];
                    let py=field.inverse[(y+1)*128+x];
                    let det=(px[0]-p[0])*(py[1]-p[1])-(px[1]-p[1])*(py[0]-p[0]);
                    assert!(det>0.0,"sampled closure cell folded at {x},{y}: {det}");
                }
            }
        }
        assert!(fields[1].inverse[49*128+40][1] > fields[0].inverse[49*128+40][1]);
    }

    #[test]
    fn field_rejects_unvalidated_scope_and_bad_source_geometry_even_at_zero() {
        let points = observation();
        for amount in [1,15,-31,30] {
            assert!(eye_closure_params_from_landmarks(128,96,&points,amount).is_err());
        }
        assert!(eye_closure_params_from_landmarks(128,96,&points[..477],0).is_err());
        let mut params=eye_closure_params_from_landmarks(128,96,&points,0).unwrap();
        params["adjustments"]["eyeWidth"]=json!(0);
        assert!(prepare_eye_closure_field(&source(),&params).is_err());
        params["adjustments"].as_object_mut().unwrap().remove("eyeWidth");
        params["mask"]=json!({"shape":"ellipse"});
        assert!(prepare_eye_closure_field(&source(),&params).is_err());
        params.as_object_mut().unwrap().remove("mask");
        params["adjustments"]["eyeHeight"]=json!(-1.5);
        assert!(prepare_eye_closure_field(&source(),&params).is_err());
        params["adjustments"]["eyeHeight"]=json!(0);
        params["semanticMasks"]["leftEye"]["points"][0]["x"]=json!(1.0);
        assert!(prepare_eye_closure_field(&source(),&params).is_err());
    }

    #[test]
    fn all_31_integer_candidates_keep_finite_positive_cells_and_move_the_eye() {
        let source=source();
        let points=observation();
        for amount in -30..=0 {
            let params=eye_closure_params_from_landmarks(128,96,&points,amount).unwrap();
            let field=prepare_eye_closure_field(&source,&params).unwrap();
            assert!(field.inverse.iter().flatten().all(|v|v.is_finite()),"nonfinite field at {amount}");
            for x in [40,88] {
                assert_eq!(field.inverse[30*128+x],[x as f32,30.0],"source brow moved at {amount}");
            }
            for y in 0..95 {
                for x in 0..127 {
                    let p=field.inverse[y*128+x];
                    let px=field.inverse[y*128+x+1];
                    let py=field.inverse[(y+1)*128+x];
                    let det=(px[0]-p[0])*(py[1]-p[1])-(px[1]-p[1])*(py[0]-p[0]);
                    assert!(det>0.0,"sampled closure cell folded at strength {amount}, pixel {x},{y}: {det}");
                }
            }
            // Measure motion over the eye interior, not an arbitrary remote
            // pixel. Quantized output need not change at every pixel or tick.
            let mut eye_motion=0.0_f64;
            for y in 40..=50 {
                for x in 28..=52 {
                    eye_motion+=(field.inverse[y*128+x][1]-y as f32).abs() as f64;
                }
            }
            if amount==0 {
                assert_eq!(eye_motion,0.0);
            } else {
                assert!(eye_motion>0.0,"nonzero strength {amount} has no eye-interior motion");
            }
        }
    }

    #[test]
    fn support_admission_is_bit_identical_to_full_mapper_for_all_strengths_and_tilted_eyes() {
        let source = source();
        for (left_angle, right_angle) in [(0.0_f64, 0.0_f64), (0.37, -0.29)] {
            let mut points = observation();
            // Tilt each aperture in source physical pixel coordinates, not in
            // normalized coordinates (the image is intentionally non-square).
            // Fractional offsets exercise support edges away from integers.
            for (ids, cx, angle, dx, dy) in [
                (LEFT_EYE, 40.5, left_angle, 0.173, -0.219),
                (RIGHT_EYE, 88.5, right_angle, -0.311, 0.137),
            ] {
                let (sin, cos) = angle.sin_cos();
                for &id in ids {
                    let x = points[id][0] * 128.0 - cx;
                    let y = points[id][1] * 96.0 - 45.5;
                    points[id] = [
                        (cx + x * cos - y * sin + dx) / 128.0,
                        (45.5 + x * sin + y * cos + dy) / 96.0,
                    ];
                }
            }
            for amount in -30..=0 {
                let params = eye_closure_params_from_landmarks(128, 96, &points, amount).unwrap();
                let optimized = prepare_eye_closure_field(&source, &params).unwrap();
                // Disabling only admission retains the full pre-optimization
                // domain, velocity and RK4 path as the differential reference.
                let reference = prepare_eye_closure_field_impl::<false>(&source, &params).unwrap();
                assert_eq!(optimized.width, reference.width);
                assert_eq!(optimized.height, reference.height);
                assert_eq!(optimized.stationary, reference.stationary);
                assert_eq!(optimized.integration_steps, reference.integration_steps);
                assert_eq!(optimized.inverse.len(), reference.inverse.len());
                for (i, (actual, expected)) in optimized.inverse.iter().zip(&reference.inverse).enumerate() {
                    for axis in 0..2 {
                        assert_eq!(actual[axis].to_bits(), expected[axis].to_bits(),
                            "support admission changed f32 bits at pixel {},{}, axis {axis}, amount {amount}, angles {left_angle}/{right_angle}",
                            i % 128, i / 128);
                    }
                }
            }
        }
    }
}
