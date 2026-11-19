//! Regression for the source-image + categorical-label seam, not supplied alpha.
use image_core::{
    prepare_eye_closure_source, EyeClosureField, EyeMaterialLabels, EyeSourceGeometry, RgbaImage,
};
use serde_json::json;
use std::collections::BTreeMap;

fn fixture() -> (RgbaImage, EyeSourceGeometry) {
    let (w, h) = (80, 50);
    let mut pixels = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let u = x as f64 - 10.;
            let v = y as f64 - 25.;
            let s = (std::f64::consts::PI * (u / 52.).clamp(0., 1.)).sin();
            let p = if u > 0. && u < 52. && v > -7.25 * s && v < 6.75 * s {
                if x == 36 && y == 20 {
                    [242, 246, 249, 255]
                } else {
                    [55 + y as u8, 75 + y as u8, 95 + y as u8, 255]
                }
            } else {
                [175, 124, 102, 255]
            };
            pixels.extend(p);
        }
    }
    let knots: Vec<f64> = (0..=32).map(|i| 52. * i as f64 / 32.).collect();
    let upper = knots
        .iter()
        .map(|u| -7.25 * (std::f64::consts::PI * u / 52.).sin())
        .collect();
    let lower = knots
        .iter()
        .map(|u| 6.75 * (std::f64::consts::PI * u / 52.).sin())
        .collect();
    (
        RgbaImage::new(w, h, pixels).unwrap(),
        EyeSourceGeometry {
            eye: "leftEye".into(),
            origin: [10., 25.],
            axis: [1., 0.],
            width: 52.,
            crop: [0, 0, w as usize, h as usize],
            knots,
            upper,
            lower,
        },
    )
}
fn identity(source: &RgbaImage) -> EyeClosureField {
    let (w, h) = (source.width as usize, source.height as usize);
    EyeClosureField {
        width: source.width,
        height: source.height,
        inverse: (0..w * h)
            .map(|i| [(i % w) as f32, (i / w) as f32])
            .collect(),
        stationary: vec![false; w * h],
        integration_steps: 0,
    }
}

#[test]
fn source_ocular_annotation_preserves_a_glint_under_closing_without_freezing_lid_motion() {
    let (source, geometry) = fixture();
    let labels: EyeMaterialLabels = serde_json::from_value(json!({"ocularAnchors":[{
        "shape":"rect","region":{"x":30,"y":19,"width":13,"height":5}}]}))
    .unwrap();
    let materials = BTreeMap::from([("leftEye".into(), labels)]);
    let manual = prepare_eye_closure_source(&source, &[geometry.clone()], &materials).unwrap();
    let automatic = prepare_eye_closure_source(&source, &[geometry], &BTreeMap::new()).unwrap();
    let mut field = identity(&source);
    for y in 16..27 {
        for x in 30..43 {
            field.inverse[y * 80 + x][1] -= 0.75;
        }
    }
    let old = automatic.render_field(&field).unwrap();
    let result = manual.render_field(&field).unwrap();
    let glint = (20 * 80 + 36) * 4;
    assert!(
        old.pixels[glint..glint + 3] != source.pixels[glint..glint + 3],
        "fixture exposes moving/occluded glint"
    );
    assert_eq!(
        result.pixels[glint..glint + 3],
        source.pixels[glint..glint + 3],
        "observed glint stays on the eyeball"
    );
    assert!(
        (16..19).any(|y| (30..43).any(|x| {
            let i = (y * 80 + x) * 4;
            result.pixels[i..i + 3] != source.pixels[i..i + 3]
        })),
        "lid still responds"
    );
    assert_eq!(
        &result.pixels[..80 * 4],
        &source.pixels[..80 * 4],
        "outside motion support stays exact"
    );
    assert_eq!(
        manual.render_field(&identity(&source)).unwrap(),
        source,
        "zero keeps original bytes"
    );
}

#[test]
fn unknown_material_blocks_only_an_actual_transport_dependency() {
    let (source, geometry) = fixture();
    let labels: EyeMaterialLabels = serde_json::from_value(json!({"unknown":[{
        "shape":"rect","region":{"x":36,"y":20,"width":1,"height":1}}]}))
    .unwrap();
    let p = prepare_eye_closure_source(
        &source,
        &[geometry],
        &BTreeMap::from([("leftEye".into(), labels)]),
    )
    .unwrap();
    let mut field = identity(&source);
    assert_eq!(p.render_field(&field).unwrap(), source);
    // Destination is not unknown, but its bilinear footprint includes unknown.
    field.inverse[21 * 80 + 36][1] = 20.75;
    assert!(p
        .render_field(&field)
        .unwrap_err()
        .to_string()
        .contains("unknown source material"));
    field.stationary[21 * 80 + 36] = true;
    assert_eq!(p.render_field(&field).unwrap(), source);
    field = identity(&source);
    field.inverse[20 * 80 + 36][0] += 0.25;
    assert!(p
        .render_field(&field)
        .unwrap_err()
        .to_string()
        .contains("unknown source material"));
}

#[test]
fn source_preparation_rejects_cross_eye_ownership_and_invalid_geometry_even_at_zero() {
    let (source, g) = fixture();
    let shape = json!({"shape":"rect","region":{"x":35,"y":21,"width":2,"height":2}});
    let mut label: EyeMaterialLabels =
        serde_json::from_value(json!({"ocularAnchors":[shape.clone()],"unknown":[shape.clone()]}))
            .unwrap();
    assert!(prepare_eye_closure_source(
        &source,
        &[g.clone()],
        &BTreeMap::from([("leftEye".into(), label.clone())])
    )
    .is_err());
    label.unknown.clear();
    let mut right = g.clone();
    right.eye = "rightEye".into();
    assert!(prepare_eye_closure_source(
        &source,
        &[g.clone(), right],
        &BTreeMap::from([
            ("leftEye".into(), label.clone()),
            ("rightEye".into(), label)
        ])
    )
    .is_err());
    for bad in [f64::NAN, 1e20] {
        let mut invalid = g.clone();
        invalid.width = bad;
        assert!(prepare_eye_closure_source(&source, &[invalid], &BTreeMap::new()).is_err());
    }
    let mut invalid = g.clone();
    invalid.origin[0] = -1.;
    assert!(prepare_eye_closure_source(&source, &[invalid], &BTreeMap::new()).is_err());
    let mut transparent = source;
    transparent.pixels[3] = 254;
    assert!(prepare_eye_closure_source(&transparent, &[g], &BTreeMap::new()).is_err());
    assert!(serde_json::from_value::<EyeMaterialLabels>(json!({"ocularAnchors":[{"shape":"rect","region":{"x":1,"y":1,"width":2,"height":2},"opacity":0.5}]})).is_err());
}

#[test]
fn continuous_ocular_stroke_keeps_interior_glint_without_freezing_the_lid() {
    let (source, geometry) = fixture();
    let labels: EyeMaterialLabels = serde_json::from_value(json!({"ocularAnchors":[{
        "shape":"polyline","points":[{"x":30,"y":20},{"x":42,"y":20}],"radius":1.0}]}))
    .unwrap();
    let prepared = prepare_eye_closure_source(
        &source,
        &[geometry],
        &BTreeMap::from([("leftEye".into(), labels)]),
    )
    .unwrap();
    let mut field = identity(&source);
    for y in 16..27 {
        for x in 30..43 {
            field.inverse[y * 80 + x][1] -= 0.75;
        }
    }
    let result = prepared.render_field(&field).unwrap();
    let glint = (20 * 80 + 36) * 4; // Halfway between the two drawn vertices.
    assert_eq!(
        &result.pixels[glint..glint + 3],
        &source.pixels[glint..glint + 3]
    );
    assert!(
        (16..19).any(|y| (30..43).any(|x| {
            let i = (y * 80 + x) * 4;
            result.pixels[i..i + 3] != source.pixels[i..i + 3]
        })),
        "a source identity stroke must not disable eyelid movement"
    );
    assert_eq!(prepared.render_field(&identity(&source)).unwrap(), source);
}

#[test]
fn continuous_unknown_stroke_checks_its_interior_and_rejects_invalid_corridors() {
    let (source, geometry) = fixture();
    let shape = json!({"shape":"polyline","points":[{"x":30,"y":20},{"x":42,"y":20}],"radius":0.6});
    let labels: EyeMaterialLabels =
        serde_json::from_value(json!({"unknown":[shape.clone()]})).unwrap();
    let prepared = prepare_eye_closure_source(
        &source,
        &[geometry.clone()],
        &BTreeMap::from([("leftEye".into(), labels)]),
    )
    .unwrap();
    let mut field = identity(&source);
    field.inverse[21 * 80 + 36][1] = 20.75;
    assert!(prepared
        .render_field(&field)
        .unwrap_err()
        .to_string()
        .contains("unknown source material"));
    for (points, radius) in [
        (json!([{"x":30,"y":20}]), 1.0),
        (json!([{"x":30,"y":20},{"x":30,"y":20}]), 1.0),
        (json!([{"x":0,"y":20},{"x":42,"y":20}]), 1.0),
        (json!([{"x":30,"y":20},{"x":42,"y":20}]), 0.0),
    ] {
        let label: EyeMaterialLabels = serde_json::from_value(
            json!({"unknown":[{"shape":"polyline","points":points,"radius":radius}]}),
        )
        .unwrap();
        assert!(prepare_eye_closure_source(
            &source,
            &[geometry.clone()],
            &BTreeMap::from([("leftEye".into(), label)])
        )
        .is_err());
    }
}
