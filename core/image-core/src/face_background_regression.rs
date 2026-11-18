//! Hidden stationary backdrop must not acquire a seam from semantic labels.
//!
//! The independent reference is an affine RGB plane, whose harmonic extension
//! through a bounded hole is the same plane. Hair/scene labels distinguish
//! observed donor eligibility, but both belong to this stationary backdrop.

use super::*;

fn affine_backdrop_with_labels(diagonal_labels: bool) {
    const W: usize = 24;
    const H: usize = 24;
    let truth = |x: usize, y: usize| -> [u8; 4] {
        [
            20 + 7 * y as u8,
            15 + 3 * (x + y) as u8,
            30 + 5 * x as u8,
            255,
        ]
    };
    let mut pixels = Vec::with_capacity(W * H * 4);
    let mut holes = vec![false; W * H];
    let mut subject = vec![0.0; W * H];
    let frozen = vec![false; W * H];
    let mut donor_excluded = vec![false; W * H];
    let mut known_layer = vec![0; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            // Only the observed source carries the semantic labels. The
            // concealed strip contains opaque foreground, not background RGB.
            holes[i] = (8..16).contains(&x) && (3..21).contains(&y);
            subject[i] = if holes[i] { 1.0 } else { 0.0 };
            let hair = if diagonal_labels {
                x + y >= 24
            } else {
                y >= 12
            };
            donor_excluded[i] = hair || holes[i];
            known_layer[i] = if hair && !holes[i] { 2 } else { 0 };
            pixels.extend_from_slice(&if holes[i] {
                [205, 135, 110, 255]
            } else {
                truth(x, y)
            });
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let output = reconstruct(
        &source,
        &holes,
        &subject,
        &frozen,
        &donor_excluded,
        &known_layer,
        None,
    )
    .unwrap();

    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let actual = &output.pixels[i * 4..i * 4 + 4];
            if holes[i] {
                let expected = truth(x, y);
                for c in 0..3 {
                    assert!(
                        actual[c].abs_diff(expected[c]) <= 2,
                        "semantic label introduced a stationary backdrop seam at {x},{y}: \
                         {actual:?}, expected {expected:?}; diagonal={diagonal_labels}"
                    );
                }
                assert_eq!(actual[3], 255);
            } else {
                assert_eq!(
                    actual,
                    &source.pixels[i * 4..i * 4 + 4],
                    "observed stationary backdrop changed at {x},{y}"
                );
            }
        }
    }
}

#[test]
fn hidden_stationary_backdrop_keeps_a_smooth_ramp_across_hair_scene_labels() {
    affine_backdrop_with_labels(false);
}

#[test]
fn diagonal_hair_scene_labels_do_not_draw_a_staircase_into_the_hidden_backdrop() {
    affine_backdrop_with_labels(true);
}

#[test]
fn donor_excluded_zero_alpha_bridges_keep_the_observed_underlayer_exact() {
    const W: usize = 11;
    const H: usize = 11;
    const DARK: [u8; 4] = [65, 40, 25, 255];
    const YELLOW: [u8; 4] = [235, 210, 80, 255];
    const FOREGROUND: [u8; 4] = [205, 145, 110, 255];
    let mut pixels = vec![0; W * H * 4];
    let mut holes = vec![false; W * H];
    let mut alpha = vec![0.0; W * H];
    let frozen = vec![false; W * H];
    let mut donor_excluded = vec![false; W * H];
    let known_layer = vec![0; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let excluded_island = (3..8).contains(&x) && (3..8).contains(&y);
            donor_excluded[i] = excluded_island;
            pixels[i * 4..i * 4 + 4].copy_from_slice(if excluded_island { &YELLOW } else { &DARK });
        }
    }
    // The excluded island may be used as a temporary bridge to reconstruct
    // this opaque hole. Exclusion forbids borrowing its color as a donor; it
    // does not make the observed zero-alpha island an output reconstruction.
    let hole = 5 * W + 5;
    holes[hole] = true;
    alpha[hole] = 1.0;
    pixels[hole * 4..hole * 4 + 4].copy_from_slice(&FOREGROUND);
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let output = reconstruct(
        &source,
        &holes,
        &alpha,
        &frozen,
        &donor_excluded,
        &known_layer,
        None,
    )
    .unwrap();
    assert_ne!(
        &output.pixels[hole * 4..hole * 4 + 4],
        &FOREGROUND,
        "the real opaque foreground hole was not reconstructed"
    );
    assert_eq!(output.pixels[hole * 4 + 3], 255);
    for i in 0..W * H {
        if !holes[i] {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "temporary donor bridge overwrote observed underlayer at {},{}; excluded={}",
                i % W,
                i / W,
                donor_excluded[i]
            );
        }
    }
}

#[test]
fn low_alpha_reveals_the_estimated_local_underlayer_instead_of_remote_dark_donors() {
    const W: usize = 11;
    const H: usize = 11;
    const DARK: [u8; 4] = [65, 40, 25, 255];
    const YELLOW: [u8; 4] = [235, 210, 80, 255];
    const FOREGROUND: [u8; 4] = [205, 145, 110, 255];
    const ALPHA: f32 = 0.05;
    let mut pixels = Vec::with_capacity(W * H * 4);
    let mut background_pixels = Vec::with_capacity(W * H * 4);
    let mut holes = vec![false; W * H];
    let mut alpha = vec![0.0; W * H];
    let frozen = vec![false; W * H];
    let mut donor_excluded = vec![false; W * H];
    let known_layer = vec![0; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let edge = (3..8).contains(&x) && (3..8).contains(&y);
            if edge {
                let mut mixed = [0; 4];
                for c in 0..3 {
                    mixed[c] = (ALPHA * FOREGROUND[c] as f32 + (1.0 - ALPHA) * YELLOW[c] as f32)
                        .round() as u8;
                }
                mixed[3] = 255;
                pixels.extend_from_slice(&mixed);
                background_pixels.extend_from_slice(&YELLOW);
                alpha[i] = ALPHA;
                holes[i] = true;
                donor_excluded[i] = true;
            } else {
                pixels.extend_from_slice(&DARK);
                background_pixels.extend_from_slice(&DARK);
            }
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let estimated_background = RgbaImage::new(W as u32, H as u32, background_pixels).unwrap();
    let output = reconstruct(
        &source,
        &holes,
        &alpha,
        &frozen,
        &donor_excluded,
        &known_layer,
        Some(&estimated_background),
    )
    .unwrap();
    // Only 5% of this source patch was foreground. Its known F/B estimate
    // therefore supplies the already observed local backdrop; far darker
    // eligible donors must not replace it just because a queue reaches it.
    for y in 3..8 {
        for x in 3..8 {
            let i = y * W + x;
            let actual = &output.pixels[i * 4..i * 4 + 4];
            for c in 0..3 {
                assert!(
                    actual[c].abs_diff(YELLOW[c]) <= 2,
                    "remote donor replaced low-alpha local backdrop at {x},{y}: {actual:?}"
                );
            }
            assert_eq!(actual[3], 255);
        }
    }
}

#[test]
fn uncertain_foreground_color_cannot_cast_a_false_shadow_on_observed_flat_background() {
    const W: usize = 15;
    const H: usize = 15;
    let bg = [88_u8, 172, 179, 255];
    let fg = [147_u8, 104, 76, 255];
    let a = 0.73_f32;
    let mut pixels = bg.repeat(W * H);
    let mut estimated = pixels.clone();
    let mut holes = vec![false; W * H];
    let mut alpha = vec![0.0; W * H];
    for y in 5..10 {
        for x in 5..10 {
            let i = y * W + x;
            holes[i] = true;
            alpha[i] = a;
            for c in 0..3 {
                pixels[4 * i + c] = byte(a * fg[c] as f32 + (1.0 - a) * bg[c] as f32);
            }
            // A coupled estimator can trade a too-bright foreground against
            // a too-dark background. Nearby visible backdrop disambiguates it.
            estimated[4 * i..4 * i + 4].copy_from_slice(&[52, 136, 152, 255]);
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let prior = RgbaImage::new(W as u32, H as u32, estimated).unwrap();
    let output = reconstruct(
        &source,
        &holes,
        &alpha,
        &vec![false; W * H],
        &holes,
        &vec![0; W * H],
        Some(&prior),
    )
    .unwrap();
    for i in 0..W * H {
        for c in 0..3 {
            assert!(
                output.pixels[4 * i + c].abs_diff(bg[c]) <= 2,
                "foreground ambiguity invented a backdrop shadow at {},{}: {:?}",
                i % W,
                i / W,
                &output.pixels[4 * i..4 * i + 4]
            );
        }
    }
}

#[test]
fn background_context_constraint_is_continuous_when_source_feasibility_crosses_a_donor() {
    const W: usize = 7;
    const H: usize = 7;
    const SOURCE_CODE: u8 = 100;
    const DONOR_CODE: u8 = 200;
    let center = 3 * W + 3;
    let mut pixels = [DONOR_CODE, DONOR_CODE, DONOR_CODE, 255].repeat(W * H);
    pixels[center * 4..center * 4 + 4].copy_from_slice(&[
        SOURCE_CODE,
        SOURCE_CODE,
        SOURCE_CODE,
        255,
    ]);
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    // Holding C and B-prior at 100 while alpha crosses 0.5 makes the nearby
    // donor B=200 enter the physically possible interval continuously. This
    // does not identify a unique true B: the contract is a continuous solve,
    // not agreement with an arbitrarily selected background at the center.
    let prior = source.clone();
    let mut holes = vec![false; W * H];
    holes[center] = true;
    let frozen = vec![false; W * H];
    let known_layer = vec![0; W * H];
    let mut alpha = vec![0.0; W * H];
    let mut previous: Option<(f32, [u8; 3])> = None;
    for step in 450..=550 {
        let a = step as f32 / 1000.0;
        alpha[center] = a;
        let output = reconstruct(
            &source,
            &holes,
            &alpha,
            &frozen,
            &holes,
            &known_layer,
            Some(&prior),
        )
        .unwrap();
        let actual: [u8; 3] = output.pixels[center * 4..center * 4 + 3]
            .try_into()
            .unwrap();
        if let Some((previous_alpha, previous_rgb)) = previous {
            for c in 0..3 {
                assert!(
                    actual[c].abs_diff(previous_rgb[c]) <= 2,
                    "background constraint jumped as a donor crossed the feasible interval: \
                     alpha {previous_alpha:.3}->{a:.3}, RGB {previous_rgb:?}->{actual:?}"
                );
            }
        }
        // C=aF+(1-a)B and F in [0,255] imply this necessary interval.
        // Integer output gets outward rounding; no particular B is demanded.
        let lower = ((SOURCE_CODE as f32 - 255.0 * a) / (1.0 - a))
            .max(0.0)
            .floor();
        let upper = (SOURCE_CODE as f32 / (1.0 - a)).min(255.0).ceil();
        for &channel in &actual {
            assert!(
                (lower..=upper).contains(&(channel as f32)),
                "background violates source-mixture feasibility at alpha {a:.3}: \
                 {actual:?}, allowed [{lower},{upper}]"
            );
        }
        assert_eq!(output.pixels[center * 4 + 3], 255);
        for i in 0..W * H {
            if i != center {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "observed context changed at alpha {a:.3}, pixel {i}"
                );
            }
        }
        previous = Some((a, actual));
    }
}

#[test]
fn isolated_hole_still_applies_source_evidence_without_four_neighbor_donors() {
    const W: usize = 5;
    const H: usize = 5;
    let center = 2 * W + 2;
    let source = RgbaImage::new(W as u32, H as u32, [100, 100, 100, 255].repeat(W * H)).unwrap();
    let mut prior = source.clone();
    prior.pixels[center * 4..center * 4 + 4].copy_from_slice(&[250, 250, 250, 255]);
    let mut holes = vec![false; W * H];
    holes[center] = true;
    let mut alpha = vec![0.0; W * H];
    alpha[center] = 0.05;
    let mut frozen = vec![false; W * H];
    for neighbor in [center - W, center + W, center - 1, center + 1] {
        frozen[neighbor] = true;
    }
    let output = reconstruct(
        &source,
        &holes,
        &alpha,
        &frozen,
        &holes,
        &vec![0; W * H],
        Some(&prior),
    )
    .unwrap();
    // All four immediate neighbors are protected, so harmonic adjacency gives
    // no donors. Source C=100 and a=.05 still constrain B, and observed 100
    // pixels remain visible diagonally and beyond the protected ring. Neither
    // this valid evidence nor the mixture bounds may be skipped with the BFS.
    let lower = ((100.0_f32 - 0.5 - 255.0 * 0.05) / 0.95).floor();
    let upper = ((100.0_f32 + 0.5) / 0.95).ceil();
    let actual = &output.pixels[center * 4..center * 4 + 4];
    for &channel in &actual[..3] {
        assert!(
            (lower..=upper).contains(&(channel as f32)),
            "missing spatial donors bypassed source evidence: {actual:?}, allowed [{lower},{upper}]"
        );
    }
    assert_eq!(actual[3], 255);
    for i in 0..W * H {
        if i != center {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "isolated-hole reconstruction changed observed/protected pixel {i}"
            );
        }
    }
}

#[test]
fn almost_opaque_alpha_does_not_drop_source_bounds_at_a_byte_threshold() {
    const W: usize = 5;
    const H: usize = 5;
    let center = 2 * W + 2;
    let mut source =
        RgbaImage::new(W as u32, H as u32, [255, 255, 255, 255].repeat(W * H)).unwrap();
    let prior = source.clone();
    source.pixels[center * 4..center * 4 + 4].copy_from_slice(&[0, 0, 0, 255]);
    let mut holes = vec![false; W * H];
    holes[center] = true;
    let mut alpha = vec![0.0; W * H];
    let frozen = vec![false; W * H];
    let known_layer = vec![0; W * H];
    let mut previous: Option<(f32, [u8; 3])> = None;
    for step in 9959..=9962 {
        let a = step as f32 / 10000.0;
        alpha[center] = a;
        let output = reconstruct(
            &source,
            &holes,
            &alpha,
            &frozen,
            &holes,
            &known_layer,
            Some(&prior),
        )
        .unwrap();
        let actual: [u8; 3] = output.pixels[center * 4..center * 4 + 3]
            .try_into()
            .unwrap();
        if let Some((previous_alpha, previous_rgb)) = previous {
            for c in 0..3 {
                // The physically feasible upper endpoint varies by fewer
                // than 4 codes here; allow rounding, never a ~130-code jump.
                assert!(
                    actual[c].abs_diff(previous_rgb[c]) <= 5,
                    "near-one alpha disabled source constraints abruptly: \
                     alpha {previous_alpha:.4}->{a:.4}, RGB {previous_rgb:?}->{actual:?}"
                );
            }
        }
        // With C=0 rounded to a byte, F>=0 implies (1-a)B<=0.5.
        // B becomes unconstrained only as this bound naturally reaches 255.
        let upper = (0.5 / (1.0 - a)).min(255.0).ceil();
        for &channel in &actual {
            assert!(
                channel as f32 <= upper,
                "almost-opaque alpha lost source feasibility: alpha={a:.4}, RGB={actual:?}, max={upper}"
            );
        }
        assert_eq!(output.pixels[center * 4 + 3], 255);
        previous = Some((a, actual));
    }
}

#[test]
fn partial_scene_observations_do_not_draw_a_boundary_into_a_hidden_ramp() {
    const W: usize = 64;
    const H: usize = 64;
    let truth = |x: usize, y: usize| -> [u8; 4] {
        // The concealed strip is an affine plane. Visible fine texture is
        // deliberately separate, so smoothing the output cannot pass.
        let detail = if x >= 40 {
            if (x + y) % 2 == 0 {
                20_i16
            } else {
                -20_i16
            }
        } else {
            0
        };
        std::array::from_fn(|c| {
            if c == 3 {
                255
            } else {
                (80 + 10 * c as i16 + x as i16 + detail) as u8
            }
        })
    };
    let mut pixels = Vec::with_capacity(W * H * 4);
    let mut holes = vec![false; W * H];
    let mut subject = vec![0.0; W * H];
    let mut excluded = vec![false; W * H];
    let mut scene = vec![None; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            holes[i] = (24..32).contains(&x);
            subject[i] = if x < 32 { 1.0 } else { 0.0 };
            excluded[i] = x < 32;
            pixels.extend_from_slice(&if x < 32 {
                [205, 135, 110, 255]
            } else {
                truth(x, y)
            });
            // Independent observations stop on a diagonal. Their support
            // boundary is not an edge in the physical hidden backdrop.
            if holes[i] && y >= 32 + (x - 24) {
                let rgb = truth(x, y);
                scene[i] = Some([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]);
            }
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let output = reconstruct_with_scene(
        &source,
        &holes,
        &subject,
        &vec![false; W * H],
        &excluded,
        &vec![0; W * H],
        None,
        Some(&scene),
    )
    .unwrap();

    // Ignoring the scene observations would return the sole visible boundary
    // color at x32 throughout this one-sided hole. Reject that smooth no-op.
    for y in 50..58 {
        for x in 24..32 {
            let i = y * W + x;
            let expected = truth(x, y);
            for c in 0..3 {
                assert!(
                    output.pixels[i * 4 + c].abs_diff(expected[c]) <= 2,
                    "reliable hidden background was ignored at {x},{y}: {:?}, truth {expected:?}",
                    &output.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }
    let mut worst_jump = (0_u8, 0, 0);
    for x in 24..32 {
        let boundary = 32 + (x - 24);
        for y in boundary - 5..boundary + 5 {
            for c in 0..3 {
                let jump = output.pixels[(y * W + x) * 4 + c]
                    .abs_diff(output.pixels[((y + 1) * W + x) * 4 + c]);
                if jump > worst_jump.0 {
                    worst_jump = (jump, x, y);
                }
            }
        }
    }
    // True vertical slope is zero. Allow quantization and a gradual uncertain
    // transition, but never paint the observation support as an eight-level seam.
    assert!(
        worst_jump.0 <= 3,
        "partial scene observations drew a false boundary: max adjacent jump {} at ({},{})->({},{}), true vertical slope 0",
        worst_jump.0, worst_jump.1, worst_jump.2, worst_jump.1, worst_jump.2 + 1
    );
    for i in 0..W * H {
        if !holes[i] {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "scene repair changed observed texture or retained foreground at {},{}",
                i % W,
                i / W
            );
        }
        assert_eq!(output.pixels[i * 4 + 3], 255);
    }
}

#[test]
fn continuous_scene_reconstruction_keeps_observed_low_alpha_texture() {
    const W: usize = 32;
    const H: usize = 32;
    const ALPHA: f32 = 0.05;
    const FOREGROUND: [u8; 3] = [200, 120, 80];
    let truth = |x: usize, y: usize| -> [u8; 4] {
        let detail = if y % 2 == 0 { 12_i16 } else { -12_i16 };
        std::array::from_fn(|c| {
            if c == 3 {
                255
            } else {
                (80 + 10 * c as i16 + x as i16 + detail) as u8
            }
        })
    };
    let mut pixels = Vec::with_capacity(W * H * 4);
    let mut background_pixels = Vec::with_capacity(W * H * 4);
    let mut holes = vec![false; W * H];
    let mut subject = vec![0.0; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let b = truth(x, y);
            background_pixels.extend_from_slice(&b);
            holes[i] = (12..20).contains(&x) && (4..28).contains(&y);
            let mut observed = b;
            if holes[i] {
                subject[i] = ALPHA;
                for c in 0..3 {
                    // Independent forward image formation, not the renderer's
                    // inverse estimator: the fine backdrop remains 95% visible.
                    observed[c] =
                        (ALPHA * FOREGROUND[c] as f32 + (1.0 - ALPHA) * b[c] as f32).round() as u8;
                }
            }
            pixels.extend_from_slice(&observed);
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let background = RgbaImage::new(W as u32, H as u32, background_pixels).unwrap();
    let output = reconstruct_with_scene(
        &source,
        &holes,
        &subject,
        &vec![false; W * H],
        &holes,
        &vec![0; W * H],
        Some(&background),
        Some(&vec![None; W * H]),
    )
    .unwrap();
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let actual = &output.pixels[i * 4..i * 4 + 4];
            if holes[i] {
                let expected = truth(x, y);
                for c in 0..3 {
                    assert!(
                        actual[c].abs_diff(expected[c]) <= 2,
                        "scene continuity erased observed mixed-edge texture at {x},{y}: {actual:?}, truth {expected:?}"
                    );
                }
            } else {
                assert_eq!(actual, &source.pixels[i * 4..i * 4 + 4]);
            }
            assert_eq!(actual[3], 255);
        }
    }
    for y in (6..26).step_by(2) {
        let bright = output.pixels[(y * W + 16) * 4] as i16;
        let dark = output.pixels[((y + 1) * W + 16) * 4] as i16;
        assert!(
            bright - dark >= 20,
            "known 24-level backdrop texture was flattened at row {y}: {bright}->{dark}"
        );
    }
}

#[test]
fn isolated_opaque_hole_uses_scene_evidence_without_available_neighbor_donors() {
    const W: usize = 3;
    const H: usize = 3;
    const CENTER: usize = 4;
    const FOREGROUND: [u8; 4] = [167, 87, 64, 255];
    const BACKGROUND: [u8; 4] = [218, 232, 241, 255];
    let mut pixels = Vec::with_capacity(W * H * 4);
    for i in 0..W * H {
        // Distinct protected colors also detect accidental donor borrowing
        // and any overwrite outside the single hidden-background texel.
        pixels.extend_from_slice(&[17 + 17 * i as u8, 31 + 7 * i as u8, 43 + 11 * i as u8, 255]);
    }
    pixels[CENTER * 4..CENTER * 4 + 4].copy_from_slice(&FOREGROUND);
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let mut holes = vec![false; W * H];
    let mut alpha = vec![0.0; W * H];
    let mut frozen = vec![true; W * H];
    let mut scene = vec![None; W * H];
    holes[CENTER] = true;
    alpha[CENTER] = 1.0;
    frozen[CENTER] = false;
    scene[CENTER] = Some([
        BACKGROUND[0] as f32,
        BACKGROUND[1] as f32,
        BACKGROUND[2] as f32,
    ]);

    // An opaque foreground gives no source-mixture observation of hidden B.
    // The F/B initializer intentionally retains C; its precision must be zero.
    // All adjacent pixels are protected, so the only accepted RGB evidence is
    // the independent scene observation. The remaining energy is exactly
    // E(B) = ||B - [218, 232, 241]||^2, with that literal RGB as its minimizer.
    let output = reconstruct_source_background(
        &source,
        &holes,
        &alpha,
        &frozen,
        &holes,
        &vec![0; W * H],
        Some(&source),
        Some(&scene),
        true,
    )
    .unwrap();

    assert_eq!(
        &output.pixels[CENTER * 4..CENTER * 4 + 4],
        &BACKGROUND,
        "a hidden background with sole accepted scene evidence retained its opaque foreground initializer"
    );
    for i in 0..W * H {
        if i != CENTER {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "isolated background reconstruction changed protected pixel {},{}",
                i % W,
                i / W
            );
        }
    }
}
