//! Public graph-entry regressions for weakly parsed, physically opaque neck edges.
//! The reference uses independently known material labels on the same image;
//! it is not a comparison between implementation feature switches.

use super::tests::{joint_portrait, public_adjust};
use super::*;

const W: usize = 160;
const H: usize = 320;
const BACKGROUND: [u8; 4] = [20, 40, 60, 255];

fn raster(data: Vec<u8>) -> Value {
    json!({"shape":"raster", "width":W, "height":H, "data":data})
}

fn set_mask_pixel(params: &mut Value, name: &str, x: usize, y: usize, value: u8) {
    params["semanticMasks"][name]["data"][y * W + x] = json!(value);
}

/// A known opaque neck is continuous with the face. Its two boundary columns
/// have deliberately weak parsing, although the RGB material is unambiguous.
/// These columns lie below/farther than the old face-polygon uncertainty band.
fn parsed_neck_edge() -> (RgbaImage, Value, Value) {
    let (mut source, mut observed) = joint_portrait();
    for i in 0..W * H {
        if source.pixels[i * 4] < 100 {
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&BACKGROUND);
        }
    }
    for y in 209..H {
        for x in 70..90 {
            let i = y * W + x;
            // Known alternating skin reflectance makes the subpixel motion
            // measurable. The actual -60 expansion here is only 0.03–0.09 px:
            // a four-level/px ramp cannot move by three decoded byte levels,
            // and previously measured F/B estimation error as “motion”.
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                if x % 2 == 0 { 170 } else { 240 },
                130,
                90,
                255,
            ]);
        }
    }

    let mut hair = vec![0_u8; W * H];
    for y in 28..32 {
        for x in 75..85 {
            let i = y * W + x;
            hair[i] = 255;
            set_mask_pixel(&mut observed, "face", x, y, 255);
            observed["subjectMask"]["data"][i] = json!(255);
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[35, 25, 20, 255]);
        }
    }
    observed["semanticMasks"]["hair"] = raster(hair);
    assert_ne!(
        observed["semanticMasks"]["face"], observed["semanticMasks"]["faceContourMask"],
        "this fixture must enter the independently parsed head path"
    );

    let mut omitted = observed.clone();
    for y in 210..220 {
        for x in [70, 89] {
            let i = y * W + x;
            assert_eq!(observed["semanticMasks"]["face"]["data"][i], 0);
            assert_eq!(observed["semanticMasks"]["faceContourMask"]["data"][i], 0);
            assert_eq!(observed["semanticMasks"]["neck"]["data"][i], 255);
            assert_eq!(observed["subjectMask"]["data"][i], 255);
            set_mask_pixel(&mut omitted, "skin", x, y, 10);
            set_mask_pixel(&mut omitted, "neck", x, y, 0);
        }
    }
    (source, omitted, observed)
}

fn pixel(image: &RgbaImage, x: usize, y: usize) -> &[u8] {
    &image.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4]
}

fn rgb_error(a: &[u8], b: &[u8]) -> u8 {
    (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap()
}

fn face_span(image: &RgbaImage) -> (usize, usize) {
    let xs: Vec<_> = (0..W).filter(|&x| pixel(image, x, 120)[0] > 150).collect();
    (*xs.first().unwrap(), *xs.last().unwrap())
}

#[test]
fn independent_head_weak_neck_fringe_follows_known_material_in_both_directions() {
    let (source, omitted, observed) = parsed_neck_edge();
    let before_span = face_span(&source);
    for amount in [-60, 60] {
        let mut omitted = omitted.clone();
        let mut observed = observed.clone();
        omitted["adjustments"] = json!({"faceWidth":amount});
        observed["adjustments"] = json!({"faceWidth":amount});
        let actual = public_adjust(&source, &omitted).unwrap();
        let reference = public_adjust(&source, &observed).unwrap();
        let after_span = face_span(&actual);
        assert!(
            if amount > 0 {
                after_span.0 > before_span.0 && after_span.1 < before_span.1
            } else {
                after_span.0 < before_span.0 && after_span.1 > before_span.1
            },
            "fringe recovery cancelled the requested face-width change {amount}: {before_span:?}->{after_span:?}"
        );

        let mut changed_reference = 0;
        let mut changed_actual = 0;
        let mut total_error = 0_u32;
        for y in 210..220 {
            for x in [70, 89] {
                let original = pixel(&source, x, y);
                let expected = pixel(&reference, x, y);
                let result = pixel(&actual, x, y);
                if amount < 0 {
                    // Expansion samples toward the adjacent interior column:
                    // left even stripe brightens; right odd stripe darkens.
                    // This sign is fixed by the photographed source pattern,
                    // independently of matte estimates or renderer controls.
                    assert!(
                        if x == 70 { result[0] >= original[0] } else { result[0] <= original[0] },
                        "expansion sampled the wrong source stripe at {x},{y}: {original:?}->{result:?}"
                    );
                }
                changed_reference += usize::from(rgb_error(expected, original) >= 3);
                changed_actual += usize::from(rgb_error(result, original) >= 3);
                total_error += rgb_error(result, expected) as u32;
                assert_eq!(
                    result[3], 255,
                    "transport introduced transparency at {x},{y}"
                );
                assert!(
                    rgb_error(result, expected) <= 24,
                    "weak parsing stranded/contaminated physical neck material {amount} at {x},{y}: {result:?}, known material {expected:?}"
                );
            }
        }
        assert!(
            changed_reference >= 4,
            "fixture failed to exercise neck-edge motion for {amount}"
        );
        assert!(
            changed_actual >= 4,
            "weakly parsed source edge stayed frozen for {amount}"
        );
        assert!(
            total_error <= 20 * 8,
            "excessive neck-edge material error for {amount}: {total_error}"
        );
        for y in 280..H {
            for x in 70..90 {
                assert_eq!(
                    pixel(&actual, x, y),
                    pixel(&source, x, y),
                    "low neck moved at {x},{y}"
                );
            }
        }
    }
}

#[test]
fn independent_head_fringe_recovery_preserves_explicit_and_attachment_constraints() {
    let (source, mut params, _) = parsed_neck_edge();
    let mut protected = vec![0_u8; W * H];
    protected[213 * W + 70] = 255;
    params["protectMask"] = raster(protected);
    for name in ["ears", "accessories"] {
        params["semanticMasks"][name] = raster(vec![0_u8; W * H]);
    }
    set_mask_pixel(&mut params, "ears", 70, 217, 255);
    set_mask_pixel(&mut params, "accessories", 89, 213, 255);
    set_mask_pixel(&mut params, "hair", 89, 217, 255);
    let before_span = face_span(&source);
    for amount in [-60, 60] {
        params["adjustments"] = json!({"faceWidth":amount});
        let output = public_adjust(&source, &params).unwrap();
        for (x, y, name) in [
            (70, 213, "explicit user protection"),
            (70, 217, "ear"),
            (89, 213, "accessory"),
            (89, 217, "hair"),
        ] {
            assert_eq!(
                pixel(&output, x, y),
                pixel(&source, x, y),
                "{name} changed at {x},{y} for {amount}"
            );
        }
        for y in 28..32 {
            for x in 75..85 {
                assert_eq!(
                    pixel(&output, x, y),
                    pixel(&source, x, y),
                    "independent head hair moved at {x},{y}"
                );
            }
        }
        assert_ne!(
            face_span(&output),
            before_span,
            "local protection suppressed the whole edit"
        );
    }
}

#[test]
fn ambiguous_person_confidence_does_not_paint_unreached_scene() {
    let (source, mut params) = joint_portrait();
    for y in 80..240 {
        for x in 12..64 {
            params["subjectMask"]["data"][y * W + x] = json!(128);
        }
    }
    for amount in [-100, 100] {
        params["adjustments"] = json!({"faceWidth":amount});
        let output = public_adjust(&source, &params).unwrap();
        // This column band is independently known scene, farther from the
        // observed portrait than the full requested lateral width change.
        // Nearby background CAN be legitimately covered during expansion;
        // comparing the first changed legacy pixel falsely forbade that.
        for y in 80..240 {
            for x in 12..30 {
                assert!(pixel(&source, x, y)[0] < 100);
                assert_eq!(
                    pixel(&output, x, y),
                    pixel(&source, x, y),
                    "unreached scene changed at {x},{y} for {amount}"
                );
            }
        }
        assert_ne!(face_span(&output), face_span(&source));
    }
}

#[test]
fn independent_head_fringe_zero_and_unsupported_paths_keep_their_contracts() {
    let (source, params, _) = parsed_neck_edge();
    let mut zero = params.clone();
    zero["adjustments"] = json!({"faceWidth":0});
    assert_eq!(public_adjust(&source, &zero).unwrap(), source);

    for variant in ["shared-head-matte", "missing-neck", "empty-neck"] {
        let mut fallback = params.clone();
        fallback["adjustments"] = json!({"faceWidth":60});
        match variant {
            "shared-head-matte" => {
                fallback["semanticMasks"]["face"] =
                    fallback["semanticMasks"]["faceContourMask"].clone();
            }
            "missing-neck" => {
                fallback["semanticMasks"]
                    .as_object_mut()
                    .unwrap()
                    .remove("neck");
            }
            "empty-neck" => {
                fallback["semanticMasks"]["neck"] = raster(vec![0_u8; W * H]);
            }
            _ => unreachable!(),
        }
        let output = public_adjust(&source, &fallback).unwrap();
        assert_ne!(
            face_span(&output),
            face_span(&source),
            "supported face editing stopped for {variant}"
        );
        for y in 210..220 {
            for x in [70, 89] {
                assert_eq!(
                    pixel(&output, x, y),
                    pixel(&source, x, y),
                    "unsupported fringe route changed at {x},{y}: {variant}"
                );
            }
        }
    }
}
