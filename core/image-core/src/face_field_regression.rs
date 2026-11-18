//! Public image-level checks for the connected face/neck velocity field.
//! An opaque source-coordinate ramp is a physical texture oracle: its output
//! order and displacement can be checked without reading interpolation controls.

use super::tests::{joint_portrait, public_adjust};
use super::*;

const W: usize = 160;
const H: usize = 320;
const EYE_Y: std::ops::Range<usize> = 111..119;
const LEFT_EYE_X: std::ops::Range<usize> = 52..62;
const RIGHT_EYE_X: std::ops::Range<usize> = 98..108;
const HAIR_X: std::ops::Range<usize> = 66..70;
const HAIR_Y: std::ops::Range<usize> = 98..136;

fn pixel(image: &RgbaImage, x: usize, y: usize) -> &[u8] {
    &image.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4]
}

fn rectangle_polygon(xs: std::ops::Range<usize>, ys: std::ops::Range<usize>) -> Value {
    let (left, right) = (xs.start as f32 / W as f32, xs.end as f32 / W as f32);
    let (top, bottom) = (ys.start as f32 / H as f32, ys.end as f32 / H as f32);
    json!({"shape":"polygon", "points":[
        {"x":left,"y":top}, {"x":right,"y":top},
        {"x":right,"y":bottom}, {"x":left,"y":bottom}
    ]})
}

fn textured_joint_portrait(with_hair: bool) -> (RgbaImage, Value) {
    let (mut source, mut params) = joint_portrait();
    // Keep all source alpha/ownership evidence unchanged. For known opaque
    // facial skin, B = 10 + 2*x gives two code values per source pixel. The
    // vertical invariance ensures vertical transport cannot fake an x change.
    for y in 0..H {
        for x in 0..W {
            if params["semanticMasks"]["faceContourMask"]["data"][y * W + x]
                .as_u64()
                .unwrap()
                == 255
            {
                source.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4].copy_from_slice(&[
                    220,
                    150,
                    10 + 2 * x as u8,
                    255,
                ]);
            }
        }
    }
    for (name, xs) in [("leftEye", LEFT_EYE_X), ("rightEye", RIGHT_EYE_X)] {
        params["semanticMasks"][name] = rectangle_polygon(xs.clone(), EYE_Y);
        for y in EYE_Y {
            for x in xs.clone() {
                source.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4].copy_from_slice(&[
                    5 + (x % 3) as u8,
                    15,
                    25,
                    255,
                ]);
            }
        }
    }
    if with_hair {
        // A known opaque foreground strand crosses the cheek next to the eye.
        // Its fixed material boundary must influence adjacent moving skin.
        let mut hair = vec![0_u8; W * H];
        for y in HAIR_Y {
            for x in HAIR_X {
                hair[y * W + x] = 255;
                source.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4].copy_from_slice(&[
                    35,
                    25 + (y % 3) as u8,
                    20,
                    255,
                ]);
            }
        }
        params["semanticMasks"]["hair"] =
            json!({"shape":"raster", "width":W, "height":H, "data":hair});
    }
    (source, params)
}

fn skin_span(image: &RgbaImage, y: usize) -> (usize, usize) {
    let xs: Vec<_> = (0..W).filter(|&x| pixel(image, x, y)[0] > 150).collect();
    (*xs.first().unwrap(), *xs.last().unwrap())
}

#[test]
fn joint_eye_side_texture_stays_ordered_while_the_contour_moves_and_features_stay_exact() {
    for with_hair in [false, true] {
        let (source, mut params) = textured_joint_portrait(with_hair);
        for amount in [-60, 60] {
            params["adjustments"] = json!({"faceWidth":amount});
            let output = public_adjust(&source, &params).unwrap();
            let before = skin_span(&source, 123);
            let after = skin_span(&output, 123);
            assert!(
                if amount > 0 {
                    after.0 >= before.0 + 2 && after.1 + 2 <= before.1
                } else {
                    after.0 + 2 <= before.0 && after.1 >= before.1 + 2
                },
                "eye/strand protection cancelled meaningful contour motion: amount={amount}, hair={with_hair}, {before:?}->{after:?}"
            );

            for xs in [LEFT_EYE_X, RIGHT_EYE_X] {
                for y in EYE_Y {
                    for x in xs.clone() {
                        assert_eq!(
                            pixel(&output, x, y),
                            pixel(&source, x, y),
                            "eye material moved at {x},{y}: amount={amount}, hair={with_hair}"
                        );
                    }
                }
            }
            if with_hair {
                for y in HAIR_Y {
                    for x in HAIR_X {
                        assert_eq!(
                            pixel(&output, x, y),
                            pixel(&source, x, y),
                            "fixed hair strand moved at {x},{y}: amount={amount}"
                        );
                    }
                }
            }

            let mut moved_texture_pixels = 0;
            // These rows avoid eye pixels; the columns remain several pixels
            // inside the edited silhouette and outside the fixed hair strand.
            // Consequently their RGB is transported opaque skin, not a mixed
            // foreground/background edge on which this coordinate oracle fails.
            for y in [106, 123, 128] {
                let original_span = skin_span(&source, y);
                let edited_span = skin_span(&output, y);
                let maximum_contour_shift = original_span
                    .0
                    .abs_diff(edited_span.0)
                    .max(original_span.1.abs_diff(edited_span.1));
                let allowed_code_shift = 2 * (maximum_contour_shift + 1);
                let mut previous_code = None;
                for x in 49..65 {
                    let result = pixel(&output, x, y);
                    let original = pixel(&source, x, y);
                    assert!(
                        result[0] >= 210 && (145..=155).contains(&result[1]) && result[3] == 255,
                        "eye-side skin gained a hole or foreign material at {x},{y}: {result:?}, amount={amount}, hair={with_hair}"
                    );
                    let code_shift = result[2].abs_diff(original[2]) as usize;
                    assert!(
                        code_shift <= allowed_code_shift,
                        "interior texture overshot the observed contour movement at {x},{y}: code shift={code_shift}, allowed={allowed_code_shift}, amount={amount}, hair={with_hair}"
                    );
                    moved_texture_pixels += usize::from(code_shift >= 2);
                    if let Some(previous) = previous_code {
                        let step = result[2] as i16 - previous;
                        // The source slope is exactly two code values per
                        // pixel. Allow at most 2x local stretch, and never a
                        // reversal of the known source material ordering.
                        assert!(
                            (0..=4).contains(&step),
                            "eye-side texture reversed or stretched beyond 2x at {x},{y}: step={step}, amount={amount}, hair={with_hair}"
                        );
                    }
                    previous_code = Some(result[2] as i16);
                }
            }
            assert!(
                moved_texture_pixels >= 4,
                "contour moved but adjacent skin did not: amount={amount}, hair={with_hair}"
            );
            for y in 280..H {
                for x in 70..90 {
                    assert_eq!(
                        pixel(&output, x, y),
                        pixel(&source, x, y),
                        "low neck moved at {x},{y}"
                    );
                }
            }
        }
    }
}

#[test]
fn geodesic_neck_candidate_stops_before_the_lower_neck_interface_in_both_directions() {
    let (source, mut params) = textured_joint_portrait(true);
    for amount in [-60, 60] {
        params["adjustments"] = json!({"faceWidth": amount});
        params["__geodesicNeck"] = json!(true);
        let output = super::adjust(&source, &params).unwrap();
        let before = skin_span(&source, 123);
        let after = skin_span(&output, 123);
        assert!(
            if amount > 0 {
                after.0 >= before.0 + 2 && after.1 + 2 <= before.1
            } else {
                after.0 + 2 <= before.0 && after.1 >= before.1 + 2
            },
            "geodesic neck falloff cancelled the face edit: amount={amount}, {before:?}->{after:?}"
        );
        // The synthetic neck is deliberately much longer than the measured
        // falloff.  Once targetAlpha is zero at the lower interface, every
        // observed neck texel must remain source-exact; an isolated transported
        // pixel would violate the layer contract even if the face still moved.
        for y in 220..H {
            for x in 70..90 {
                assert_eq!(
                    pixel(&output, x, y),
                    pixel(&source, x, y),
                    "lower-neck isolated pixel at {x},{y}, amount={amount}"
                );
            }
        }
    }
}

#[test]
fn joint_contraction_does_not_paint_foreground_over_observed_background() {
    let (source, mut params) = joint_portrait();
    params["adjustments"] = json!({"faceWidth": 60});
    let output = public_adjust(&source, &params).unwrap();
    let face = params["semanticMasks"]["faceContourMask"]["data"]
        .as_array()
        .unwrap();
    let neck = params["semanticMasks"]["neck"]["data"].as_array().unwrap();
    let mut changed_background = 0;
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            if face[i].as_u64().unwrap() == 0
                && neck[i].as_u64().unwrap() == 0
                && pixel(&output, x, y) != pixel(&source, x, y)
            {
                changed_background += 1;
            }
        }
    }
    assert_eq!(
        changed_background, 0,
        "contraction painted {changed_background} observed background pixels"
    );
}
