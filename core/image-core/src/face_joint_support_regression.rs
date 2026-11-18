//! Public-entry material support contracts for the shared face/neck layer.
//! Expectations come from the known opaque source material, never a legacy
//! implementation switch or the implementation's own target contour.

use super::tests::{joint_portrait, public_adjust};
use super::*;

const W: usize = 160;
const H: usize = 320;
const BACKGROUND: [u8; 4] = [20, 40, 60, 255];

fn pixel(image: &RgbaImage, x: usize, y: usize) -> &[u8] {
    &image.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4]
}

fn rgb_error(a: &[u8], b: &[u8]) -> u8 {
    (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap()
}

fn set_mask(params: &mut Value, name: &str, x: usize, y: usize, value: u8) {
    params["semanticMasks"][name]["data"][y * W + x] = json!(value);
}

/// Completely missed parsing on an otherwise opaque, contiguous neck edge.
/// RGB and person confidence remain identical in omitted and observed cases.
fn completely_omitted_neck_edge() -> (RgbaImage, Value, Value) {
    let (mut source, mut observed) = joint_portrait();
    for i in 0..W * H {
        if source.pixels[i * 4] < 100 {
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&BACKGROUND);
        }
    }
    for y in 209..H {
        for x in 70..90 {
            let i = y * W + x;
            // A 70-level alternating source stripe resolves the independently
            // measured 0.03–0.09 px expansion. The previous four-level ramp's
            // >=3-level “motion” was dominated by F/B estimation error.
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
            set_mask(&mut observed, "face", x, y, 255);
            observed["subjectMask"]["data"][i] = json!(255);
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[35, 25, 20, 255]);
        }
    }
    observed["semanticMasks"]["hair"] = json!({"shape":"raster","width":W,"height":H,"data":hair});
    assert_ne!(
        observed["semanticMasks"]["face"],
        observed["semanticMasks"]["faceContourMask"]
    );
    let mut omitted = observed.clone();
    for y in 210..220 {
        for x in [70, 89] {
            let i = y * W + x;
            assert_eq!(observed["semanticMasks"]["face"]["data"][i], 0);
            assert_eq!(observed["semanticMasks"]["faceContourMask"]["data"][i], 0);
            assert_eq!(observed["semanticMasks"]["neck"]["data"][i], 255);
            assert_eq!(observed["subjectMask"]["data"][i], 255);
            set_mask(&mut omitted, "skin", x, y, 0);
            set_mask(&mut omitted, "neck", x, y, 0);
        }
    }
    (source, omitted, observed)
}

#[test]
fn completely_omitted_opaque_fringe_moves_like_known_neck_material() {
    let (source, omitted, observed) = completely_omitted_neck_edge();
    let mut failures = Vec::new();
    for amount in [-60, 60] {
        let mut omitted = omitted.clone();
        let mut observed = observed.clone();
        omitted["adjustments"] = json!({"faceWidth":amount});
        observed["adjustments"] = json!({"faceWidth":amount});
        let actual = public_adjust(&source, &omitted).unwrap();
        let reference = public_adjust(&source, &observed).unwrap();
        let mut changed_reference = 0;
        let mut changed_actual = 0;
        let mut total_error = 0_u32;
        let mut max_error = 0;
        for y in 210..220 {
            for x in [70, 89] {
                let original = pixel(&source, x, y);
                let expected = pixel(&reference, x, y);
                let result = pixel(&actual, x, y);
                if amount < 0 {
                    assert!(
                        if x == 70 { result[0] >= original[0] } else { result[0] <= original[0] },
                        "expansion sampled the wrong source stripe at {x},{y}: {original:?}->{result:?}"
                    );
                }
                changed_reference += usize::from(rgb_error(expected, original) >= 3);
                changed_actual += usize::from(rgb_error(result, original) >= 3);
                let error = rgb_error(result, expected);
                total_error += error as u32;
                max_error = max_error.max(error);
                assert_eq!(result[3], 255, "new transparency at {x},{y}");
            }
        }
        eprintln!(
            "complete omission {amount}: changed actual/reference={changed_actual}/{changed_reference}; max={max_error}, total={total_error}"
        );
        assert!(
            changed_reference >= 4,
            "fixture does not exercise edge motion"
        );
        if changed_actual < 4 || max_error > 24 || total_error > 20 * 8 {
            failures.push(format!(
                "amount={amount}: changed actual/reference={changed_actual}/{changed_reference}, max={max_error}, total={total_error}"
            ));
        }
        for y in 280..H {
            for x in 70..90 {
                assert_eq!(pixel(&actual, x, y), pixel(&source, x, y));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fully missed parsing stranded/contaminated continuous opaque material: {failures:?}"
    );
}

#[test]
fn joint_expansion_transports_upper_neck_beyond_its_original_support() {
    let (source, mut params) = joint_portrait();
    params["adjustments"] = json!({"faceWidth":-100});
    let output = public_adjust(&source, &params).unwrap();
    // The face ends above row 209, while the independently specified opaque
    // neck is x=70..89. Its shared transition must expand into BOTH adjacent
    // background columns instead of being clipped at the old material edge.
    for x in [69, 90] {
        let mut covered_rows = 0;
        for y in 210..220 {
            let original = pixel(&source, x, y);
            let result = pixel(&output, x, y);
            assert!(original[0] < 40 && original[2] == 130);
            assert_eq!(result[3], 255);
            // Known skin has substantially higher R and lower B than the
            // known background. This asserts newly transported skin coverage.
            covered_rows += usize::from(result[0] >= original[0] + 8 && result[2] < original[2]);
        }
        assert!(
            covered_rows >= 4,
            "shared upper neck was clipped to its original support at column {x}: covered {covered_rows}/10 rows"
        );
    }
    // Shared support is local, not permission to expand the entire neck.
    for y in 280..H {
        for x in 65..95 {
            assert_eq!(pixel(&output, x, y), pixel(&source, x, y));
        }
    }
}
