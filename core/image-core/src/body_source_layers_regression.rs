use super::*;
use serde_json::json;

fn cloth_fixture(amount: f32, semantic_cut: bool) -> (RgbaImage, Value, Vec<f32>, RgbaImage) {
    let (w, h) = (160usize, 200usize);
    let mut alpha = vec![0.0; w * h];
    let mut driver = vec![0u8; w * h];
    let mut pixels = vec![255; w * h * 4];
    let background =
        RgbaImage::new(w as u32, h as u32, [180, 150, 120, 255].repeat(w * h)).unwrap();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let cloth = (20..140).contains(&x) && (10..190).contains(&y);
            let color = if cloth {
                [30 + x as u8, 40, 210]
            } else {
                [180, 150, 120]
            };
            pixels[i * 4..i * 4 + 3].copy_from_slice(&color);
            if cloth {
                alpha[i] = 1.0;
                if !(semantic_cut && (70..112).contains(&y) && (100..128).contains(&x)) {
                    driver[i] = 255;
                }
            }
        }
    }
    let params = json!({"bodyPart":"waist",
        "region":{"x":0.01,"y":0.01,"width":0.98,"height":0.98},
        "landmarks":[{"x":0.5,"y":0.05},{"x":0.5,"y":0.5},{"x":0.5,"y":0.95}],
        "subjectMask":{"shape":"raster","width":w,"height":h,"data":alpha.iter().map(|a|a*255.0).collect::<Vec<_>>()},
        "partMask":{"shape":"raster","width":w,"height":h,"data":driver},
        "adjustments":{"waist":amount}});
    (
        RgbaImage::new(w as u32, h as u32, pixels).unwrap(),
        params,
        alpha,
        background,
    )
}

fn layers<'a>(source: &RgbaImage, alpha: &'a [f32], background: &'a RgbaImage) -> SourceLayers<'a> {
    SourceLayers {
        width: source.width,
        height: source.height,
        optical_alpha: alpha,
        paired_background: Some(background),
    }
}

#[test]
fn a_semantic_partition_does_not_cut_one_optical_cloth_layer() {
    for amount in [-60.0, 60.0] {
        let (source, cut, alpha, background) = cloth_fixture(amount, true);
        let (_, complete, _, _) = cloth_fixture(amount, false);
        let material = layers(&source, &alpha, &background);
        let actual = adjust_with_source_layers(&source, &cut, &material).unwrap();
        let expected = adjust_with_source_layers(&source, &complete, &material).unwrap();
        assert!(
            actual == expected,
            "semantic-only hole must not split one continuous material at {amount}"
        );
        let at = (90 * 160 + 110) * 4;
        let response = actual.pixels[at] as f32 - source.pixels[at] as f32;
        assert!(
            response * amount > 0.0,
            "retain signed cloth motion, response={response}, amount={amount}"
        );
    }
}

#[test]
fn physical_apertures_move_with_material_and_reveal_the_fixed_textured_plate() {
    for amount in [-60.0, 60.0] {
        let (mut source, params, mut alpha, mut background) = cloth_fixture(amount, false);
        for y in 0..200 {
            for x in 0..160 {
                let i = y * 160 + x;
                let b = [
                    30 + (x % 13) as u8,
                    80 + (y % 17) as u8,
                    120 + ((x + y) % 19) as u8,
                ];
                background.pixels[i * 4..i * 4 + 3].copy_from_slice(&b);
                if (78..106).contains(&y) && (105..125).contains(&x) {
                    alpha[i] = 0.0;
                }
                if alpha[i] == 0.0 {
                    source.pixels[i * 4..i * 4 + 3].copy_from_slice(&b);
                }
            }
        }
        let before = source.clone();
        let output =
            adjust_with_source_layers(&source, &params, &layers(&source, &alpha, &background))
                .unwrap();
        // These lie strictly inside the translated hole, not on its soft
        // sampling fringe. Revealing an observed lower layer must reproduce
        // its original local texture, not an output-dependent fill.
        let revealed_x = if amount > 0.0 { 104 } else { 125 };
        let revealed = (90 * 160 + revealed_x) * 4;
        assert_eq!(
            output.pixels[revealed..revealed + 4],
            background.pixels[revealed..revealed + 4]
        );
        assert_ne!(
            source.pixels[revealed..revealed + 4],
            background.pixels[revealed..revealed + 4]
        );
        let covered_x = if amount > 0.0 { 123 } else { 105 };
        let covered = (90 * 160 + covered_x) * 4;
        assert_eq!(alpha[90 * 160 + covered_x], 0.0);
        assert_ne!(
            output.pixels[covered..covered + 4],
            source.pixels[covered..covered + 4],
            "source optical zero must not freeze a destination later covered by cloth"
        );
        let mut protected = params.clone();
        protected["protectMask"] = json!({"shape":"rect","region":{"x":105.0/160.0,"y":78.0/200.0,"width":20.0/160.0,"height":28.0/200.0}});
        let fixed =
            adjust_with_source_layers(&source, &protected, &layers(&source, &alpha, &background))
                .unwrap();
        for y in 78..106 {
            for x in 105..125 {
                let i = (y * 160 + x) * 4;
                assert_eq!(
                    fixed.pixels[i..i + 4],
                    source.pixels[i..i + 4],
                    "explicitly protected lower material must stay exact"
                );
            }
        }
        assert!(source == before, "source input is immutable");
    }
}

#[test]
fn a_missing_background_prior_is_not_replaced_by_the_original_opaque_cloth() {
    let (source, params, _, _) = cloth_fixture(60.0, false);
    let alpha = vec![1.0; 160 * 200];
    let request = SourceLayers {
        width: 160,
        height: 200,
        optical_alpha: &alpha,
        paired_background: None,
    };
    assert!(
        adjust_with_source_layers(&source, &params, &request).is_err(),
        "no observed background must be reported, not silently copied from the foreground"
    );
}

#[test]
fn fractional_source_layers_are_continuous_across_zero_without_mutating_inputs() {
    let (mut source, mut params, mut alpha, mut background) = cloth_fixture(0.0, false);
    for y in 0..200 {
        for x in 0..160 {
            let i = y * 160 + x;
            let a = if (10..190).contains(&y) {
                ((x as f32 - 18.0) / 4.0)
                    .clamp(0.0, 1.0)
                    .min((142.0 - x as f32) / 4.0)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            alpha[i] = a;
            let b = [
                150 + (x % 7) as u8,
                100 + (y % 11) as u8,
                80 + ((x + y) % 13) as u8,
            ];
            background.pixels[i * 4..i * 4 + 3].copy_from_slice(&b);
            let foreground = [30.0 + x as f32, 40.0, 210.0];
            for c in 0..3 {
                source.pixels[i * 4 + c] = byte(a * foreground[c] + (1.0 - a) * b[c] as f32);
            }
        }
    }
    let original_source = source.clone();
    let original_alpha = alpha.clone();
    let original_background = background.clone();
    for amount in [0.0_f32, -0.01, 0.001, -1.0, 0.01, -0.001, 1.0, 0.0] {
        params["adjustments"]["waist"] = json!(amount);
        let output =
            adjust_with_source_layers(&source, &params, &layers(&source, &alpha, &background))
                .unwrap();
        let max_difference = output
            .pixels
            .iter()
            .zip(&source.pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        if amount == 0.0 {
            assert!(
                output == source,
                "zero edit must reconstruct source bytes exactly"
            );
        } else if amount.abs() <= 0.01 {
            assert!(
                max_difference <= 1,
                "nonzero transport has a finite cross-zero color jump: {max_difference}"
            );
        } else {
            assert!(
                max_difference > 0,
                "the source-layer renderer must retain actual signed motion"
            );
        }
    }
    assert!(source == original_source);
    assert_eq!(alpha, original_alpha);
    assert!(background == original_background);
}

#[test]
fn optical_coverage_cannot_override_region_selection_or_explicit_protection() {
    let (mut source, mut params, _, background) = cloth_fixture(60.0, false);
    let alpha = vec![1.0; 160 * 200];
    params["mask"] = json!({"shape":"rect","region":{"x":0.5,"y":0.0,"width":0.5,"height":1.0}});
    params["protectMask"] = json!({"shape":"rect","region":{"x":106.0/160.0,"y":80.0/200.0,"width":10.0/160.0,"height":20.0/200.0}});
    for y in 80..100 {
        for x in 106..116 {
            source.pixels[(y * 160 + x) * 4..(y * 160 + x) * 4 + 3].copy_from_slice(&[10, 240, 20]);
        }
    }
    for amount in [-60.0, 60.0] {
        params["adjustments"]["waist"] = json!(amount);
        let output =
            adjust_with_source_layers(&source, &params, &layers(&source, &alpha, &background))
                .unwrap();
        assert!(output != source, "authorized material must still move");
        for y in 0..200 {
            for x in 0..160 {
                let i = (y * 160 + x) * 4;
                let protected = (80..100).contains(&y) && (106..116).contains(&x);
                if x < 80 || x == 159 || y == 0 || y == 199 || protected {
                    assert_eq!(
                        output.pixels[i..i + 4],
                        source.pixels[i..i + 4],
                        "write permission widened at {x},{y}"
                    );
                }
                if !protected {
                    assert!(
                        output.pixels[i + 1] < 200,
                        "fixed foreground copied into movable cloth at {x},{y}"
                    );
                }
            }
        }
    }
}

#[test]
fn valid_backgrounds_remain_fixed_and_do_not_cross_contaminate_cached_colors() {
    let (source, params, alpha, first) = cloth_fixture(60.0, false);
    let mut second = first.clone();
    for (i, &a) in alpha.iter().enumerate() {
        if a == 1.0 {
            second.pixels[i * 4..i * 4 + 3].copy_from_slice(&[20, 200, 40]);
        }
    }
    let render =
        |b| adjust_with_source_layers(&source, &params, &layers(&source, &alpha, b)).unwrap();
    let one = render(&first);
    let two = render(&second);
    let again = render(&first);
    assert!(
        one == again,
        "another valid fixed plate must not poison cached source colors"
    );
    assert!(
        one != two,
        "newly exposed material must use the caller's actual fixed plate"
    );
    let empty = vec![0.0; alpha.len()];
    let empty_layers = SourceLayers {
        width: 160,
        height: 200,
        optical_alpha: &empty,
        paired_background: None,
    };
    assert!(adjust_with_source_layers(&source, &params, &empty_layers).unwrap() == source);
}

#[test]
fn explicit_invalid_source_layers_reject_even_for_a_zero_request() {
    let (source, params, alpha, background) = cloth_fixture(0.0, false);
    let render = |request: &SourceLayers<'_>| adjust_with_source_layers(&source, &params, request);
    assert!(render(&SourceLayers {
        width: 200,
        height: 160,
        ..layers(&source, &alpha, &background)
    })
    .is_err());
    assert!(render(&SourceLayers {
        optical_alpha: &alpha[..alpha.len() - 1],
        ..layers(&source, &alpha, &background)
    })
    .is_err());
    for invalid in [f32::NAN, f32::INFINITY, -0.01, 255.0] {
        let mut bad = alpha.clone();
        bad[90 * 160 + 90] = invalid;
        assert!(render(&layers(&source, &bad, &background)).is_err());
    }
    let mut bad_background = background.clone();
    bad_background.pixels[0] ^= 1; // Optical zero requires the observed source as B.
    let before = bad_background.clone();
    assert!(render(&layers(&source, &alpha, &bad_background)).is_err());
    assert!(
        before == bad_background,
        "validating a fixed plate must never modify the caller's image"
    );
    let mut transparent = background.clone();
    transparent.pixels[3] = 128;
    assert!(render(&layers(&source, &alpha, &transparent)).is_err());
    let mut transparent_source = source.clone();
    transparent_source.pixels[3] = 128;
    assert!(adjust_with_source_layers(
        &transparent_source,
        &params,
        &layers(&source, &alpha, &background)
    )
    .is_err());
}

#[test]
fn material_touching_the_canvas_keeps_body_clamped_sampling() {
    let (w, h) = (64usize, 80usize);
    let source = RgbaImage::new(
        w as u32,
        h as u32,
        (0..w * h)
            .flat_map(|i| [30 + (i % w) as u8, 100, 200, 255])
            .collect(),
    )
    .unwrap();
    let alpha = vec![1.0; w * h];
    let background = RgbaImage::new(w as u32, h as u32, [0, 0, 0, 255].repeat(w * h)).unwrap();
    for amount in [-80.0, 80.0] {
        let params = json!({"bodyPart":"waist","region":{"x":0.,"y":0.,"width":1.,"height":1.},
            "landmarks":[{"x":0.5,"y":0.},{"x":0.5,"y":0.5},{"x":0.5,"y":1.}],
            "subjectMask":{"shape":"rect","region":{"x":0.,"y":0.,"width":1.,"height":1.}},"adjustments":{"waist":amount}});
        let output =
            adjust_with_source_layers(&source, &params, &layers(&source, &alpha, &background))
                .unwrap();
        for pixel in output.pixels.chunks_exact(4) {
            assert_eq!(
                &pixel[1..],
                [100, 200, 255],
                "transparent extrapolation must not expose a false canvas-edge background strip"
            );
        }
        assert!(
            output != source,
            "clamping the canvas must not disable material transport"
        );
    }
}

#[test]
fn an_arm_driver_does_not_measure_the_whole_optical_torso() {
    let (w, h) = (120usize, 160usize);
    let mut pixels = [20, 40, 110, 255].repeat(w * h);
    let background = RgbaImage::new(w as u32, h as u32, pixels.clone()).unwrap();
    let mut alpha = vec![0.0; w * h];
    let mut arm = vec![0u8; w * h];
    for y in 24..136 {
        for x in 36..100 {
            let i = y * w + x;
            alpha[i] = 1.0;
            if x < 56 {
                arm[i] = 255;
            }
            pixels[i * 4..i * 4 + 3].copy_from_slice(&[120 + x as u8, 110, 70]);
        }
    }
    let source = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
    let mut params = json!({"bodyPart":"left-upper-arm","region":{"x":0.23,"y":0.05,"width":0.3,"height":0.9},
        "landmarks":[{"x":0.375,"y":0.15},{"x":0.375,"y":0.85}],
        "subjectMask":{"shape":"raster","width":w,"height":h,"data":alpha.iter().map(|a|a*255.0).collect::<Vec<_>>()},
        "partMask":{"shape":"raster","width":w,"height":h,"data":arm},"adjustments":{"arms":80}});
    let start = |im: &RgbaImage| (0..w).find(|&x| im.pixels[(80 * w + x) * 4] > 120).unwrap();
    for amount in [-80, 80] {
        params["adjustments"]["arms"] = json!(amount);
        let output =
            adjust_with_source_layers(&source, &params, &layers(&source, &alpha, &background))
                .unwrap();
        if amount > 0 {
            assert!(start(&output) > start(&source));
        } else {
            assert!(start(&output) < start(&source));
        }
        for y in 24..136 {
            for x in 65..100 {
                let i = (y * w + x) * 4;
                assert_eq!(
                    output.pixels[i..i + 4],
                    source.pixels[i..i + 4],
                    "optical torso must not replace the arm's geometric radius"
                );
            }
        }
    }
}

#[test]
fn fixed_plate_validation_distinguishes_alpha_rounding_from_wrong_source_colors() {
    // True mixture: C=10, F=0, B=100, alpha=.9. The nearest f32 alpha is
    // slightly below .9; that is not evidence that fixed B must be changed.
    let source = RgbaImage::new(1, 1, vec![10, 10, 10, 255]).unwrap();
    let background = RgbaImage::new(1, 1, vec![100, 100, 100, 255]).unwrap();
    let params = json!({"adjustments":{"waist":0}});
    let alpha = [0.9];
    let output = adjust_with_source_layers(&source, &params, &layers(&source, &alpha, &background));
    assert!(
        output.is_ok(),
        "representational alpha rounding must not reject a feasible fixed plate"
    );
    assert!(output.unwrap() == source);
    let invalid = [0.9 - 4e-7];
    assert!(
        adjust_with_source_layers(&source, &params, &layers(&source, &invalid, &background))
            .is_err(),
        "multiple alpha ULPs of infeasibility are not ordinary float rounding"
    );
}
