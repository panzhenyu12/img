//! Explicit source-material evidence through the normal face renderer.
//!
//! These fixtures supply known material identities. They test the renderer's
//! ownership contracts, not the accuracy of automatic garment segmentation.

use super::*;

const WIDTH: usize = 160;
const HEIGHT: usize = 320;

fn garment_mask(data: &[u8]) -> Value {
    json!({"shape":"raster","width":WIDTH,"height":HEIGHT,"data":data})
}

fn assert_garment_unchanged(source: &RgbaImage, output: &RgbaImage, garment: &[u8]) {
    for (i, &material) in garment.iter().enumerate() {
        if material > 0 {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "fixed garment changed at {},{}",
                i % WIDTH,
                i / WIDTH,
            );
        }
    }
}

#[test]
fn explicit_source_material_corrects_cloth_without_freezing_real_neck() {
    let (mut source, mut params) = super::tests::joint_portrait();
    // A broad visible neck gives the collar nonzero lateral motion under the
    // original face-width field. A tiny centred neck is almost stationary at
    // its base even without correct material evidence, hiding this failure.
    for y in 195..HEIGHT {
        for x in 50..110 {
            let i = y * WIDTH + x;
            if params["semanticMasks"]["faceContourMask"]["data"][i]
                .as_u64()
                .unwrap()
                < 128
            {
                params["semanticMasks"]["skin"]["data"][i] = json!(255);
                params["semanticMasks"]["neck"]["data"][i] = json!(255);
                params["subjectMask"]["data"][i] = json!(255);
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    205 + (x % 7) as u8,
                    145 + (y % 5) as u8,
                    90,
                    255,
                ]);
            }
        }
    }
    let mut garment = vec![0_u8; WIDTH * HEIGHT];
    // The old source parser still includes the central shirt in its neck
    // support. Independent material evidence corrects it without editing the
    // original semantic masks or pretending to be user-authored protection.
    params["__sourceSkinSupport"] = params["semanticMasks"]["skin"].clone();
    params["__sourceNeckSupport"] = params["semanticMasks"]["neck"].clone();
    // Raised sides of an open collar sit below the face while leaving the
    // central neck connected. Their source positions are inside the actual
    // face-width field; the lower flat hem alone can be outside its support.
    for y in 210..HEIGHT {
        for x in 50..110 {
            if y < 225 && (70..90).contains(&x) {
                continue;
            }
            let i = y * WIDTH + x;
            garment[i] = 255;
            // Clothing and skin deliberately share the same colour family.
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                190 + 35 * (x % 2) as u8,
                145 + 15 * (y % 2) as u8,
                95,
                255,
            ]);
            params["__sourceSkinSupport"]["data"][i] = json!(0);
            params["__sourceNeckSupport"]["data"][i] = json!(0);
        }
    }
    params["__sourceGarment"] = garment_mask(&garment);
    for amount in [-100., 100.] {
        params["adjustments"] = json!({"faceWidth":amount});
        let output = adjust(&source, &params).unwrap();
        assert_garment_unchanged(&source, &output, &garment);

        let changed_neck = (180..225)
            .flat_map(|y| (65..95).map(move |x| y * WIDTH + x))
            .filter(|&i| {
                params["semanticMasks"]["neck"]["data"][i]
                    .as_u64()
                    .unwrap_or(0)
                    > 0
                    && output.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4]
            })
            .count();
        assert!(
            changed_neck > 0,
            "stationary cloth must not cancel the visible adjacent neck edit"
        );

        let mut no_material_evidence = params.clone();
        for key in [
            "__sourceGarment",
            "__sourceSkinSupport",
            "__sourceNeckSupport",
        ] {
            no_material_evidence.as_object_mut().unwrap().remove(key);
        }
        let baseline = adjust(&source, &no_material_evidence).unwrap();
        let changed_cloth = garment
            .iter()
            .enumerate()
            .filter(|&(i, &material)| {
                material > 0 && baseline.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4]
            })
            .count();
        assert!(
            changed_cloth > 0,
            "fixture must expose the original source-material error"
        );
    }
}

struct TraceDirectory(std::path::PathBuf);

impl Drop for TraceDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn overlapping_hair_and_accessory_cannot_be_background_donors() {
    let (source, mut params) = super::tests::joint_portrait();
    let rect = |x: f32, y: f32, width: f32, height: f32| json!({"shape":"rect","region":{"x":x,"y":y,"width":width,"height":height}});
    // The actual fixed masks overlap. A hair exception must not erase the
    // independent accessory identity at the same source pixel.
    params["semanticMasks"]["hair"] =
        rect(124.0 / 160.0, 100.0 / 320.0, 22.0 / 160.0, 20.0 / 320.0);
    params["semanticMasks"]["accessories"] =
        rect(130.0 / 160.0, 105.0 / 320.0, 10.0 / 160.0, 10.0 / 320.0);
    params["semanticMasks"]["ears"] = rect(130.0 / 160.0, 115.0 / 320.0, 10.0 / 160.0, 4.0 / 320.0);
    params["adjustments"] = json!({"faceWidth":60});
    let trace = TraceDirectory(std::env::temp_dir().join(format!(
        "image-core-overlapping-hair-accessory-{}",
        std::process::id()
    )));
    params["__faceFieldTrace"] = json!(trace.0);
    params["__faceProbePoints"] = json!([[135.5, 110.5], [126.5, 102.5], [135.5, 116.5]]);
    let output = adjust(&source, &params).unwrap();
    let probes: Value =
        serde_json::from_slice(&std::fs::read(trace.0.join("probes.json")).unwrap()).unwrap();
    let rendered: Value =
        serde_json::from_slice(&std::fs::read(trace.0.join("render-probes.json")).unwrap())
            .unwrap();
    assert_eq!(probes[0]["attachmentFixed"], json!(true));
    assert_eq!(rendered[0]["donorExcluded"], json!(false));

    // Feed the renderer's donor decision into the real background fill. The
    // bright overlapping accessory must not contaminate the exposed hole.
    let mut pixels = [20_u8, 40, 60, 255].repeat(25);
    pixels[13 * 4..13 * 4 + 4].copy_from_slice(&[240, 10, 10, 255]);
    let background = RgbaImage::new(5, 5, pixels).unwrap();
    let mut holes = vec![false; 25];
    holes[12] = true;
    let mut donor_frozen = vec![false; 25];
    donor_frozen[13] = rendered[0]["donorFrozen"].as_bool().unwrap();
    let mut layer = vec![0_u8; 25];
    layer[13] = 2;
    let filled = reconstruct(
        &background,
        &holes,
        &[0.0; 25],
        &donor_frozen,
        &[false; 25],
        &layer,
        None,
    )
    .unwrap();
    assert_eq!(&filled.pixels[12 * 4..12 * 4 + 4], &[20, 40, 60, 255]);

    assert_eq!(
        rendered[0]["donorFrozen"],
        json!(true),
        "overlapping accessory lost its donor protection: {}",
        rendered[0]
    );
    // Hair alone remains eligible for the existing stationary-hair treatment.
    assert_eq!(rendered[1]["donorFrozen"], json!(false));
    assert_eq!(rendered[2]["donorExcluded"], json!(false));
    assert_eq!(rendered[2]["donorFrozen"], json!(true));
    for (x, y) in [(135_usize, 110_usize), (126, 102), (135, 116)] {
        let i = (y * WIDTH + x) * 4;
        assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
    }
}

#[test]
fn skin_coloured_high_collar_cannot_be_released_as_inferred_portrait_fringe() {
    let (mut source, mut params) = super::tests::joint_portrait();
    // A distinct head silhouette enables the actual inferred-fringe recovery
    // path. Its head alpha must agree with its own outline, not the face oval.
    let head: Vec<Point> = (0..64)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 64.;
            (80. + 45. * a.cos(), 120. + 100. * a.sin())
        })
        .collect();
    let head_data: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| byte(alpha(((i % WIDTH) as f32 + 0.5, (i / WIDTH) as f32 + 0.5), &head) * 255.))
        .collect();
    let head_outline = json!({"shape":"polygon","points":head.iter().map(|p|
        json!({"x":p.0/WIDTH as f32,"y":p.1/HEIGHT as f32})).collect::<Vec<_>>()});
    params["landmarks"] = params["semanticMasks"]["faceContour"]["points"].clone();
    params["semanticMasks"]["face"] = garment_mask(&head_data);
    params["semanticMasks"]["faceOutline"] = head_outline;

    let mut garment = vec![0_u8; WIDTH * HEIGHT];
    for y in 222..245 {
        for x in [68_usize, 69, 90, 91] {
            let i = y * WIDTH + x;
            garment[i] = 255;
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[208, 147, 90, 255]);
            params["semanticMasks"]["skin"]["data"][i] = json!(0);
            params["semanticMasks"]["neck"]["data"][i] = json!(0);
            params["subjectMask"]["data"][i] = json!(255);
        }
    }
    params["__sourceGarment"] = garment_mask(&garment);
    let trace = TraceDirectory(std::env::temp_dir().join(format!(
        "image-core-material-high-collar-{}",
        std::process::id()
    )));
    params["__faceFieldTrace"] = json!(trace.0);
    params["__faceProbePoints"] = json!([[69.5, 230.5], [70.5, 230.5], [90.5, 230.5]]);

    for amount in [-60., 60.] {
        params["adjustments"] = json!({"faceWidth":amount});
        let output = adjust(&source, &params).unwrap();
        let probes: Value =
            serde_json::from_slice(&std::fs::read(trace.0.join("probes.json")).unwrap()).unwrap();
        for i in [0_usize, 2] {
            assert_eq!(
                probes[i]["inferredFixed"],
                json!(true),
                "fixture must exercise overlapping inferred-person protection"
            );
            assert_eq!(
                probes[i]["frozen"],
                json!(true),
                "explicit garment was released by RGB fringe recovery: {}",
                probes[i]
            );
            assert_eq!(probes[i]["colorAlpha"], json!(0.0));
            assert_eq!(probes[i]["forward"], probes[i]["p"]);
        }
        assert_garment_unchanged(&source, &output, &garment);
        // A contact boundary may reduce adjacent skin motion below byte
        // quantization. It must preserve that skin's actual source identity.
        assert_eq!(probes[1]["frozen"], json!(false));
        assert_eq!(probes[1]["jointOwned"], json!(true));
        assert!(probes[1]["colorAlpha"].as_f64().unwrap() > 0.5);
    }
}

/// Explicit offline source-evidence replay through the real, normal geometry
/// implementation. It does not relax the public graph's validation schema.
#[test]
#[ignore]
fn replay_explicit_source_material() {
    let path = std::env::var("FACE_MATERIAL_REPLAY_SPEC").expect("explicit local replay spec");
    let spec: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for case in spec.as_array().unwrap() {
        let graph: Value =
            serde_json::from_slice(&std::fs::read(case["graph"].as_str().unwrap()).unwrap())
                .unwrap();
        let mut operation: crate::GraphOperation =
            serde_json::from_value(graph["operations"][0].clone()).unwrap();
        assert_eq!(operation.kind, "face.adjust");
        let mut evidence = Vec::new();
        for key in [
            "__sourceGarment",
            "__sourceSkinSupport",
            "__sourceNeckSupport",
        ] {
            if let Some(mask) = operation.params.as_object_mut().unwrap().remove(key) {
                evidence.push((key, mask));
            }
        }
        crate::validation::operation(&operation).expect("valid original operation");
        for (key, mask) in evidence {
            operation.params[key] = mask;
        }
        let source = image::open(case["source"].as_str().unwrap())
            .unwrap()
            .into_rgba8();
        let source = RgbaImage::new(source.width(), source.height(), source.into_raw()).unwrap();
        let start = std::time::Instant::now();
        let output = adjust(&source, &operation.params).unwrap();
        image::save_buffer_with_format(
            case["output"].as_str().unwrap(),
            &output.pixels,
            output.width,
            output.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();
        eprintln!("{} {} ms", case["id"], start.elapsed().as_millis());
    }
}
