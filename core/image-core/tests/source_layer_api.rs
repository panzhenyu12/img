//! Tests linked to the ordinary rlib: cfg(test) research switches are absent.
use image_core::{
    adjust_body_with_source_layers, adjust_face_with_source_layers, BodySourceLayers,
    FaceObservationRegion, FaceSourceLayers, RgbaImage,
};
use serde_json::json;

fn source() -> RgbaImage {
    RgbaImage::new(4, 3, (0..12).flat_map(|i| [20 + i, 80, 120, 255]).collect()).unwrap()
}

fn layers<'a>(source: &'a RgbaImage, alpha: &'a [f32]) -> FaceSourceLayers<'a> {
    FaceSourceLayers {
        observed_source: source,
        optical_alpha: alpha,
        observation: FaceObservationRegion {
            left: 0,
            top: 0,
            width: 4,
            height: 3,
        },
        scene: source,
        motion_interface: None,
        moving_attachment_support: None,
    }
}

#[test]
fn valid_zero_face_request_is_exact_and_source_is_immutable() {
    let source = source();
    let original = source.clone();
    let alpha = [0.6; 12];
    let output = adjust_face_with_source_layers(
        &source,
        &json!({"adjustments":{"faceWidth":0}}),
        &layers(&source, &alpha),
    )
    .unwrap();
    assert_eq!(output, original);
    assert_eq!(source, original);
}

#[test]
fn zero_face_request_still_rejects_another_source_and_invalid_layer_arrays() {
    let source = source();
    let mut unrelated = source.clone();
    unrelated.pixels[0] += 1;
    let p = json!({"adjustments":{"faceWidth":0}});
    assert!(
        adjust_face_with_source_layers(&unrelated, &p, &layers(&source, &[1.; 12]))
            .unwrap_err()
            .to_string()
            .contains("different decoded source")
    );
    for alpha in [
        vec![1.; 11],
        vec![f32::NAN; 12],
        vec![-0.01; 12],
        vec![1.01; 12],
    ] {
        assert!(adjust_face_with_source_layers(&source, &p, &layers(&source, &alpha)).is_err());
    }
    let mut context = layers(&source, &[1.; 12]);
    context.observation.left = u32::MAX;
    assert!(adjust_face_with_source_layers(&source, &p, &context).is_err());
    context.observation.left = 0;
    context.observation.width = 0;
    assert!(adjust_face_with_source_layers(&source, &p, &context).is_err());
}

#[test]
fn private_switches_and_bad_motion_evidence_cannot_bypass_public_validation() {
    let source = source();
    let alpha = [1.; 12];
    let mut context = layers(&source, &alpha);
    for key in [
        "experimentalFaceProfile",
        "__physicalAlpha",
        "__sourcePlatePrior",
    ] {
        let p = json!({"adjustments":{}, key:"source-domain-continuous-v1"});
        assert!(adjust_face_with_source_layers(&source, &p, &context).is_err());
    }
    let interface = json!({"mask":{"shape":"raster","width":4,"height":3,"data":[255]},"contour":{"shape":"polygon","points":[]}});
    context.motion_interface = Some(&interface);
    assert!(adjust_face_with_source_layers(&source, &json!({"adjustments":{}}), &context).is_err());
    context.motion_interface = None;
    let attachment =
        json!({"shape":"raster","width":4,"height":3,"data":[0,0,0,0,0,0,0,0,0,0,0,128]});
    context.moving_attachment_support = Some(&attachment);
    assert!(adjust_face_with_source_layers(&source, &json!({"adjustments":{}}), &context).is_err());
}

#[test]
fn zero_body_request_keeps_fixed_background_contract() {
    let source = source();
    let alpha = [0.; 12];
    let mut background = source.clone();
    let p = json!({"adjustments":{"waist":0}});
    let run = |background: &RgbaImage| {
        adjust_body_with_source_layers(
            &source,
            &p,
            &BodySourceLayers {
                width: 4,
                height: 3,
                optical_alpha: &alpha,
                paired_background: Some(background),
            },
        )
    };
    assert_eq!(run(&background).unwrap(), source);
    background.pixels[0] += 1;
    assert!(
        run(&background).is_err(),
        "explicit incompatible background must not be silently changed even at zero"
    );
}

// Analytic portrait over an independently known textured background. This
// fixture does not use the renderer's outline/matting/flow helpers.
fn analytic_portrait() -> (RgbaImage, RgbaImage, Vec<f32>, serde_json::Value) {
    let (w, h) = (96usize, 128usize);
    let mut source_pixels = Vec::new();
    let mut scene_pixels = Vec::new();
    let mut alpha = Vec::new();
    let mut face = Vec::new();
    let mut skin = Vec::new();
    let mut neck = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let qx = (x as f32 + 0.5 - 48.) / 24.;
            let qy = (y as f32 + 0.5 - 48.) / 35.;
            let a = ((1. - qx.hypot(qy)) * 24. + 0.5).clamp(0., 1.);
            let n = if (40..56).contains(&x) && (74..h).contains(&y) {
                1.
            } else {
                0.
            };
            let material = a.max(n);
            let b = [20 + (x % 9) as u8, 40 + (y % 11) as u8, 130];
            let f = [210 + (x % 7) as u8, 150 + (y % 5) as u8, 90];
            for c in 0..3 {
                source_pixels
                    .push((material * f[c] as f32 + (1. - material) * b[c] as f32).round() as u8);
            }
            source_pixels.push(255);
            scene_pixels.extend_from_slice(&[b[0], b[1], b[2], 255]);
            alpha.push(material);
            face.push((a * 255.).round() as u8);
            skin.push((material * 255.).round() as u8);
            neck.push(if n > 0. && a < 0.5 { 255u8 } else { 0 });
        }
    }
    let outline = json!({"shape":"polygon","points":(0..64).map(|i| {
        let theta=i as f64*std::f64::consts::TAU/64.;
        json!({"x":(48.+24.*theta.cos())/w as f64,"y":(48.+35.*theta.sin())/h as f64})
    }).collect::<Vec<_>>()});
    let raster = |data: Vec<u8>| json!({"shape":"raster","width":w,"height":h,"data":data});
    let params = json!({"adjustments":{"faceWidth":60},"subjectMask":raster(skin.clone()),
        "semanticMasks":{"face":raster(face.clone()),"faceOutline":outline.clone(),
        "faceContourMask":raster(face),"faceContour":outline,"skin":raster(skin),"neck":raster(neck)}});
    (
        RgbaImage::new(w as u32, h as u32, source_pixels).unwrap(),
        RgbaImage::new(w as u32, h as u32, scene_pixels).unwrap(),
        alpha,
        params,
    )
}

#[test]
fn ordinary_face_api_moves_the_contour_both_ways_and_honors_user_protection() {
    let (source, scene, alpha, mut params) = analytic_portrait();
    let layers = FaceSourceLayers {
        observed_source: &source,
        optical_alpha: &alpha,
        observation: FaceObservationRegion {
            left: 0,
            top: 0,
            width: source.width,
            height: source.height,
        },
        scene: &scene,
        motion_interface: None,
        moving_attachment_support: None,
    };
    let span = |image: &RgbaImage| {
        let xs: Vec<_> = (0..96)
            .filter(|&x| image.pixels[(60 * 96 + x) * 4] > 120)
            .collect();
        xs.last().unwrap() - xs.first().unwrap()
    };
    let original = source.clone();
    for value in [-60, 60] {
        params["adjustments"]["faceWidth"] = json!(value);
        let output = adjust_face_with_source_layers(&source, &params, &layers).unwrap();
        assert!(
            (span(&output) as i32 - span(&source) as i32) * value < 0,
            "contour should respond in the requested direction"
        );
        let mut protected = params.clone();
        protected["protectMask"] =
            json!({"shape":"rect","region":{"x":0.25,"y":0.35,"width":0.125,"height":0.25}});
        let fixed = adjust_face_with_source_layers(&source, &protected, &layers).unwrap();
        for y in 45..76 {
            for x in 24..36 {
                let i = (y * 96 + x) * 4;
                assert_eq!(
                    &fixed.pixels[i..i + 4],
                    &source.pixels[i..i + 4],
                    "user-protected material at {x},{y}"
                );
            }
        }
        assert_ne!(
            fixed, source,
            "protection must not turn the whole edit into a silent no-op"
        );
    }
    assert_eq!(source, original);
}
