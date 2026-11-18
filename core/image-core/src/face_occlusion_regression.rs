//! Explicit depth-order experiments for hair next to the facial silhouette.
//! The public default treats unspecified hair as a fixed foreground occluder.
//! Only a test-only independently supplied rear-hair mask permits the moving
//! face layer to cover stationary hair; that hair must never be transported.

use super::tests::{joint_portrait, public_adjust};
use super::*;

const W: usize = 160;
const H: usize = 320;
const HAIR_ROWS: std::ops::Range<usize> = 92..176;
const CHECK_ROWS: std::ops::Range<usize> = 112..144;

fn pixel(image: &RgbaImage, x: usize, y: usize) -> &[u8] {
    &image.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4]
}

fn raster(data: &[u8]) -> Value {
    json!({"shape":"raster","width":W,"height":H,"data":data})
}

struct HairFixture {
    source: RgbaImage,
    params: Value,
    hair: Vec<u8>,
    // Leftmost and rightmost nonzero source-face alpha per row. These are
    // observed fixture labels, independent of requested target/flow geometry.
    face_edges: Vec<(usize, usize)>,
}

fn portrait_with_adjacent_hair() -> HairFixture {
    let (mut source, mut params) = joint_portrait();
    let mut hair = vec![0; W * H];
    let mut face_edges = vec![(0, 0); H];
    for y in HAIR_ROWS {
        let face_xs: Vec<_> = (0..W)
            .filter(|&x| {
                params["semanticMasks"]["faceContourMask"]["data"][y * W + x]
                    .as_u64()
                    .unwrap()
                    > 0
            })
            .collect();
        let left = *face_xs.first().unwrap();
        let right = *face_xs.last().unwrap();
        face_edges[y] = (left, right);
        for distance in 1..=18 {
            for x in [left - distance, right + distance] {
                let i = y * W + x;
                assert_eq!(params["semanticMasks"]["faceContourMask"]["data"][i], 0);
                assert_eq!(params["semanticMasks"]["skin"]["data"][i], 0);
                hair[i] = 255;
                // Preserve a visible coordinate-dependent hair texture so
                // bit equality catches stationary hair accidentally warped.
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    18 + (x % 4) as u8,
                    22 + (y % 5) as u8,
                    28 + ((x + y) % 3) as u8,
                    255,
                ]);
                params["semanticMasks"]["face"]["data"][i] = json!(255);
                params["subjectMask"]["data"][i] = json!(255);
            }
        }
    }
    params["semanticMasks"]["hair"] = raster(&hair);
    HairFixture {
        source,
        params,
        hair,
        face_edges,
    }
}

fn skin_span(image: &RgbaImage, y: usize) -> (usize, usize) {
    let skin: Vec<_> = (0..W).filter(|&x| pixel(image, x, y)[0] > 150).collect();
    (*skin.first().unwrap(), *skin.last().unwrap())
}

fn near_hair_covered_rows(fixture: &HairFixture, output: &RgbaImage, side: usize) -> usize {
    CHECK_ROWS
        .filter(|&y| {
            let (left, right) = fixture.face_edges[y];
            (1..=3).any(|distance| {
                let x = if side == 0 {
                    left - distance
                } else {
                    right + distance
                };
                assert_eq!(fixture.hair[y * W + x], 255);
                let actual = pixel(output, x, y);
                // The known source facial material is [220,155,100]. Dark
                // hair/blue scene cannot satisfy this visible-skin predicate.
                actual[0] > 150 && actual[1] > 95 && actual[2] < 120
            })
        })
        .count()
}

#[test]
fn confirmed_rear_hair_can_be_occluded_by_expansion_without_being_warped() {
    let fixture = portrait_with_adjacent_hair();
    let mut params = fixture.params.clone();
    params["adjustments"] = json!({"faceWidth":-100});
    params["__rearHairMask"] = raster(&fixture.hair);
    let output = adjust(&fixture.source, &params).unwrap();
    for side in 0..2 {
        let covered = near_hair_covered_rows(&fixture, &output, side);
        assert!(
            covered >= 8,
            "confirmed rear hair blocked meaningful face expansion on side {side}: {covered}/32 rows"
        );
    }
    for y in HAIR_ROWS {
        let (left, right) = fixture.face_edges[y];
        for distance in 13..=18 {
            for x in [left - distance, right + distance] {
                assert_eq!(
                    pixel(&output, x, y),
                    pixel(&fixture.source, x, y),
                    "uncovered stationary rear-hair texture moved at {x},{y}"
                );
            }
        }
    }
}

#[test]
fn confirmed_rear_hair_stays_exact_during_actual_face_contraction() {
    let fixture = portrait_with_adjacent_hair();
    let mut params = fixture.params.clone();
    params["adjustments"] = json!({"faceWidth":100});
    params["__rearHairMask"] = raster(&fixture.hair);
    let output = adjust(&fixture.source, &params).unwrap();
    let before = skin_span(&fixture.source, 120);
    let after = skin_span(&output, 120);
    assert!(
        after.0 >= before.0 + 2 && after.1 + 2 <= before.1,
        "rear-hair handling suppressed requested slimming: {before:?}->{after:?}"
    );
    for i in 0..W * H {
        if fixture.hair[i] > 0 {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &fixture.source.pixels[i * 4..i * 4 + 4],
                "contraction moved stationary rear hair at {},{}",
                i % W,
                i / W
            );
        }
    }
}

#[test]
fn public_default_never_overwrites_hair_without_rear_depth_evidence() {
    let fixture = portrait_with_adjacent_hair();
    for amount in [-100, 100] {
        let mut params = fixture.params.clone();
        params["adjustments"] = json!({"faceWidth":amount});
        let output = public_adjust(&fixture.source, &params).unwrap();
        assert_ne!(output, fixture.source, "hair protection cancelled the edit");
        for i in 0..W * H {
            if fixture.hair[i] > 0 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &fixture.source.pixels[i * 4..i * 4 + 4],
                    "unspecified/front hair was overwritten at {},{} for {amount}",
                    i % W,
                    i / W
                );
            }
        }
    }
}

#[test]
fn explicit_protection_overrides_confirmed_rear_hair_occlusion() {
    let fixture = portrait_with_adjacent_hair();
    let mut params = fixture.params.clone();
    params["adjustments"] = json!({"faceWidth":-100});
    params["__rearHairMask"] = raster(&fixture.hair);
    let mut protected = vec![0; W * H];
    for y in 120..126 {
        let (left, right) = fixture.face_edges[y];
        for distance in 1..=4 {
            protected[y * W + left - distance] = 255;
            protected[y * W + right + distance] = 255;
        }
    }
    params["protectMask"] = raster(&protected);
    let output = adjust(&fixture.source, &params).unwrap();
    for i in 0..W * H {
        if protected[i] > 0 {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &fixture.source.pixels[i * 4..i * 4 + 4],
                "rear-depth permission overrode explicit protection at {},{}",
                i % W,
                i / W
            );
        }
    }
    for side in 0..2 {
        assert!(
            near_hair_covered_rows(&fixture, &output, side) >= 8,
            "small explicit protection suppressed all other rear-hair occlusion on side {side}"
        );
    }
}

fn textured_portrait_with_adjacent_hair() -> HairFixture {
    let mut fixture = portrait_with_adjacent_hair();
    for y in HAIR_ROWS {
        let (left, right) = fixture.face_edges[y];
        for x in left..=right {
            let i = y * W + x;
            fixture.source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                192 + (x % 8) as u8 * 4,
                132 + (y % 8) as u8 * 3,
                82 + ((x + y) % 11) as u8,
                255,
            ]);
        }
    }
    fixture
}

fn shared_cheek_trace(params: &mut Value, fixture: &HairFixture, amount: i32, label: &str) {
    let Some(root) = std::env::var_os("FACE_SHARED_TEST_TRACE") else {
        return;
    };
    let path = std::path::PathBuf::from(root).join(format!("{amount}-{label}"));
    std::fs::create_dir_all(&path).expect("create optional shared cheek trace directory");
    let mut probes = Vec::new();
    for y in [92, 96, 104, 112, 120, 128, 136, 143, 152, 168, 175] {
        let (left, right) = fixture.face_edges[y];
        for offset in [-14, -8, -3, -1, 0, 1, 3, 8] {
            for x in [left as i32 + offset, right as i32 - offset] {
                probes.push(json!([x as f32 + 0.5, y as f32 + 0.5]));
            }
        }
    }
    params["__faceFieldTrace"] = json!(path);
    params["__faceProbePoints"] = json!(probes);
    std::fs::write(
        path.join("input-params.json"),
        serde_json::to_vec(params).unwrap(),
    )
    .expect("write optional shared cheek trace inputs");
    // Record promotion separately from field ownership. This is diagnostic
    // data only; none of the pixel assertions use a promoted-mask oracle.
    let polygon = points(&params["semanticMasks"]["faceContour"], W as f32, H as f32);
    let frame = Frame::new(&polygon, params.get("semanticMasks"), W as f32, H as f32);
    if let Some(promoted) =
        upper_width::promote_interface(params, &polygon, frame, W as u32, H as u32)
    {
        std::fs::write(
            path.join("promoted-params.json"),
            serde_json::to_vec(&promoted).unwrap(),
        )
        .expect("write optional shared cheek promoted inputs");
    }
}

/// Observe the actual output pixels from the Rust renderer. The shared-cheek
/// flag is intentionally test-only and is not accepted by the public schema;
/// the unmodified public graph path is rendered separately as a baseline.
/// The coordinate pattern makes a protected patch a meaningful assertion:
/// the same patch must visibly move when no explicit protection is supplied.
/// Width is measured from source skin pigment, not the target polygon,
/// velocity field, promoted mask, or any implementation-derived geometry.
#[test]
fn shared_cheek_moderate_width_changes_visible_sides_and_respects_protection() {
    check_shared_cheek_visible_sides_and_protection("shared-cheek-curve-v4");
}

#[test]
fn periodic_curve_changes_visible_sides_and_respects_protection() {
    check_shared_cheek_visible_sides_and_protection("shared-cheek-ear-periodic-v1");
}

fn check_shared_cheek_visible_sides_and_protection(mode: &str) {
    let fixture = textured_portrait_with_adjacent_hair();
    let mut protected = vec![0_u8; W * H];
    for y in 116..122 {
        for x in 47..53 {
            assert!(pixel(&fixture.source, x, y)[0] > 150);
            protected[y * W + x] = 255;
        }
    }
    assert_eq!(protected.iter().filter(|&&p| p == 255).count(), 36);

    for amount in [-30_i32, 30] {
        let mut params = fixture.params.clone();
        params["adjustments"] = json!({"faceWidth":amount});
        params["protectMask"] = raster(&protected);
        let baseline = public_adjust(&fixture.source, &params).unwrap();
        params.as_object_mut().unwrap().remove("protectMask");
        params["experimentalFaceProfile"] = json!(mode);
        shared_cheek_trace(
            &mut params,
            &fixture,
            amount,
            &format!("{mode}-unprotected"),
        );
        let unprotected = adjust(&fixture.source, &params).unwrap();
        let would_move = protected
            .iter()
            .enumerate()
            .filter(|&(i, &p)| {
                p > 0
                    && unprotected.pixels[i * 4..i * 4 + 4]
                        != fixture.source.pixels[i * 4..i * 4 + 4]
            })
            .count();
        params["protectMask"] = raster(&protected);
        shared_cheek_trace(&mut params, &fixture, amount, &format!("{mode}-protected"));
        let output = adjust(&fixture.source, &params).unwrap();
        let mut moved_rows = [0_usize; 2];
        let mut reversed_rows = [0_usize; 2];
        for y in CHECK_ROWS {
            let before = skin_span(&fixture.source, y);
            let after = skin_span(&output, y);
            for (side, delta) in [
                after.0 as i32 - before.0 as i32,
                before.1 as i32 - after.1 as i32,
            ]
            .into_iter()
            .enumerate()
            {
                let directional = delta * amount.signum();
                moved_rows[side] += usize::from(directional >= 1);
                reversed_rows[side] += usize::from(directional < 0);
            }
        }
        let mut protected_changes = 0;
        let mut background_changes = 0;
        let mut background_count = 0;
        let mut distant_hair_changes = 0;
        let mut distant_hair_count = 0;
        for (i, &p) in protected.iter().enumerate() {
            let changed =
                output.pixels[i * 4..i * 4 + 4] != fixture.source.pixels[i * 4..i * 4 + 4];
            protected_changes += usize::from(p > 0 && changed);
            // The blue scene gutters lie outside the entire source
            // head/hair envelope. Immediate face-background neighbours are
            // excluded: expansion may legitimately cover those destinations.
            // This is a source-fixture label, not a field-support predicate.
            if i % W < 20 || i % W >= 140 {
                assert_eq!(fixture.params["subjectMask"]["data"][i], 0);
                background_count += 1;
                background_changes += usize::from(changed);
            }
        }
        for y in HAIR_ROWS {
            let (left, right) = fixture.face_edges[y];
            for distance in 16..=18 {
                for x in [left - distance, right + distance] {
                    assert_eq!(fixture.hair[y * W + x], 255);
                    distant_hair_count += 1;
                    distant_hair_changes +=
                        usize::from(pixel(&output, x, y) != pixel(&fixture.source, x, y));
                }
            }
        }
        eprintln!(
            "shared cheek {amount}: moved rows={moved_rows:?}/32, reversed={reversed_rows:?}, protected counterfactual={would_move}/36, protected changes={protected_changes}, exterior changes={background_changes}/{background_count}, distant hair changes={distant_hair_changes}/{distant_hair_count}; row128 source={:?}, public baseline={:?}, candidate={:?}",
            skin_span(&fixture.source, 128), skin_span(&baseline, 128), skin_span(&output, 128)
        );
        assert!(
            would_move > 0,
            "explicit protection was not exercised at {amount}"
        );
        assert_eq!(
            protected_changes, 0,
            "explicit protection leaked at {amount}"
        );
        assert!(
            background_count > 10_000,
            "fixture omitted external background"
        );
        assert_eq!(background_changes, 0, "external backdrop moved at {amount}");
        assert!(distant_hair_count > 400, "fixture omitted distant hair");
        assert_eq!(distant_hair_changes, 0, "distant hair moved at {amount}");
        assert_eq!(
            reversed_rows,
            [0, 0],
            "a visible side moved backwards at {amount}"
        );
        assert!(
            moved_rows.iter().all(|&count| count >= 16),
            "moderate faceWidth {amount} did not visibly change both facial sides: {moved_rows:?}/32 rows"
        );
    }
}

/// Promotion changes a source material domain when a width control leaves
/// zero. Its optical preparation must still converge to the decoded source
/// image, independently of whether explicit protection is present. A one-code
/// tolerance allows final 8-bit rounding; it does not permit a new hair/skin
/// plate or a changed material class to appear at infinitesimal motion.
#[test]
fn shared_cheek_crossing_zero_has_no_finite_optical_jump() {
    check_shared_cheek_zero_continuity("shared-cheek-curve-v4");
}

#[test]
fn periodic_curve_crossing_zero_has_no_finite_optical_jump() {
    check_shared_cheek_zero_continuity("shared-cheek-ear-periodic-v1");
}

fn check_shared_cheek_zero_continuity(mode: &str) {
    let fixture = textured_portrait_with_adjacent_hair();
    let largest_difference = |a: &RgbaImage, b: &RgbaImage| {
        (0..W * H)
            .flat_map(|i| (0..3).map(move |channel| (i, channel)))
            .map(|(i, channel)| {
                (
                    a.pixels[i * 4 + channel].abs_diff(b.pixels[i * 4 + channel]),
                    i % W,
                    i / W,
                    channel,
                )
            })
            .max()
            .unwrap()
    };
    let mut failures = Vec::new();
    for has_protection in [false, true] {
        let mut params = fixture.params.clone();
        params["experimentalFaceProfile"] = json!(mode);
        if has_protection {
            let mut protected = vec![0_u8; W * H];
            for y in 116..122 {
                for x in 47..53 {
                    protected[y * W + x] = 255;
                }
            }
            params["protectMask"] = raster(&protected);
        }
        params["adjustments"] = json!({"faceWidth":0});
        let zero = adjust(&fixture.source, &params).unwrap();
        assert_eq!(
            zero, fixture.source,
            "shared-cheek zero was not bit-exact: protection={has_protection}"
        );
        for magnitude in [0.01_f64, 0.001] {
            let mut pair = Vec::new();
            for amount in [-magnitude, magnitude] {
                params["adjustments"] = json!({"faceWidth":amount});
                let actual = adjust(&fixture.source, &params).unwrap();
                let difference = largest_difference(&fixture.source, &actual);
                eprintln!(
                    "shared cheek near zero {amount}, protection={has_protection}: maximum RGB change={} at {},{}, channel={}",
                    difference.0, difference.1, difference.2, difference.3
                );
                if difference.0 > 1 {
                    failures.push(format!(
                        "amount={amount}, protection={has_protection}: {} RGB codes at {},{}, channel={}",
                        difference.0, difference.1, difference.2, difference.3
                    ));
                }
                assert!(
                    actual.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255),
                    "near-zero width introduced transparent pixels"
                );
                pair.push(actual);
            }
            let cross_zero = largest_difference(&pair[0], &pair[1]);
            if cross_zero.0 > 2 {
                failures.push(format!(
                    "crossing ±{magnitude}, protection={has_protection}: {} RGB codes at {},{}, channel={}",
                    cross_zero.0, cross_zero.1, cross_zero.2, cross_zero.3
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "finite optical jumps: {}",
        failures.join("; ")
    );
}

#[test]
fn empty_observed_ear_mask_keeps_the_shared_curve_render_identical() {
    let fixture = textured_portrait_with_adjacent_hair();
    for amount in [-30, 30] {
        let mut params = fixture.params.clone();
        params["adjustments"] = json!({"faceWidth":amount});
        params["semanticMasks"]["ears"] = raster(&vec![0_u8; W * H]);
        params["experimentalFaceProfile"] = json!("shared-cheek-curve-v4");
        let reference = adjust(&fixture.source, &params).unwrap();
        assert!(
            reference.pixels != fixture.source.pixels,
            "empty-ear comparison omitted actual face editing at {amount}"
        );
        params["experimentalFaceProfile"] = json!("shared-cheek-ear-v1");
        let actual = adjust(&fixture.source, &params).unwrap();
        let changed = actual
            .pixels
            .chunks_exact(4)
            .zip(reference.pixels.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            changed, 0,
            "an empty ear mask changed {changed} rendered pixels at {amount}"
        );
    }
}

/// The scene contains a connected, opaque neck with a visible source stripe.
/// At isolated source pixels the head/face/neck classifiers miss the material,
/// while the independent skin and selected-person masks still observe it.
/// Enabling motion for an ear elsewhere must not turn these non-ear pixels
/// into fixed old-image pins. This runs both complete material pipelines;
/// no private ownership/frozen flag is used as an assertion oracle.
#[test]
fn observed_ear_policy_does_not_freeze_connected_non_ear_neck_pins() {
    let mut fixture = textured_portrait_with_adjacent_hair();
    for y in 209..240 {
        for x in 70..90 {
            let i = y * W + x;
            assert_eq!(fixture.params["semanticMasks"]["skin"]["data"][i], 255);
            fixture.source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                if x % 2 == 0 { 170 } else { 240 },
                130,
                90,
                255,
            ]);
        }
    }
    // A labelled lateral ear patch enables the observed-ear policy. Explicit
    // user protection covers this patch in both runs, so the test does not
    // require an unrelated ear edit as evidence of neck motion.
    let mut ears = vec![0_u8; W * H];
    let mut protected = vec![0_u8; W * H];
    for y in 113..127 {
        for x in 46..50 {
            let i = y * W + x;
            assert_eq!(
                fixture.params["semanticMasks"]["faceContourMask"]["data"][i],
                255
            );
            ears[i] = 255;
            protected[i] = 255;
        }
    }
    fixture.params["semanticMasks"]["ears"] = raster(&ears);
    fixture.params["protectMask"] = raster(&protected);
    let mut pins = Vec::new();
    for y in [210, 212, 214, 216] {
        for x in [71, 88] {
            let i = y * W + x;
            assert_eq!(fixture.params["subjectMask"]["data"][i], 255);
            assert_eq!(fixture.params["semanticMasks"]["skin"]["data"][i], 255);
            assert_eq!(ears[i], 0);
            assert_eq!(protected[i], 0);
            for kind in ["face", "faceContourMask", "neck"] {
                fixture.params["semanticMasks"][kind]["data"][i] = json!(0);
            }
            pins.push((x, y));
        }
    }
    let mut failures = Vec::new();
    for amount in [-60, 60] {
        let mut params = fixture.params.clone();
        params["adjustments"] = json!({"faceWidth":amount});
        params["experimentalFaceProfile"] = json!("shared-cheek-curve-v4");
        let reference = adjust(&fixture.source, &params).unwrap();
        params["experimentalFaceProfile"] = json!("shared-cheek-ear-v1");
        let actual = adjust(&fixture.source, &params).unwrap();
        let mut reference_motion = 0;
        let mut actual_motion = 0;
        for &(x, y) in &pins {
            let source = pixel(&fixture.source, x, y);
            let expected = pixel(&reference, x, y);
            let observed = pixel(&actual, x, y);
            if expected != source {
                reference_motion += 1;
                if observed == source {
                    failures.push(format!(
                        "amount={amount}: non-ear pin {x},{y} was frozen; source={source:?}, curve={expected:?}, ear={observed:?}"
                    ));
                }
            }
            actual_motion += usize::from(observed != source);
        }
        for (i, &p) in protected.iter().enumerate() {
            if p != 0 {
                assert_eq!(
                    &actual.pixels[i * 4..i * 4 + 4],
                    &fixture.source.pixels[i * 4..i * 4 + 4],
                    "observed-ear policy overrode explicit ear protection at {},{}",
                    i % W,
                    i / W
                );
            }
        }
        eprintln!(
            "non-ear neck pins {amount}: curve moved={reference_motion}/8, observed-ear moved={actual_motion}/8"
        );
        assert!(
            reference_motion > 0,
            "neck fixture produced no measurable reference motion at {amount}"
        );
    }
    assert!(
        failures.is_empty(),
        "ear policy froze non-ear material: {}",
        failures.join("; ")
    );
}

const EAR_SCENE: [u8; 4] = [20, 40, 60, 255];

struct ConnectedEarFixture {
    source: RgbaImage,
    params: Value,
    physical_alpha: Vec<f32>,
    ear_alpha: Vec<f32>,
}

/// The ear's source optical support is known independently of the face-only
/// requested polygon: it protrudes left of the original facial contour.
fn portrait_with_connected_ear() -> ConnectedEarFixture {
    let (mut source, mut params) = joint_portrait();
    let mut physical_alpha = vec![0.0_f32; W * H];
    for (i, pixel) in source.pixels.chunks_exact_mut(4).enumerate() {
        if pixel[0] > 150 {
            physical_alpha[i] = 1.0;
        } else {
            pixel.copy_from_slice(&EAR_SCENE);
        }
    }
    let mut ears = vec![0_u8; W * H];
    let mut ear_alpha = vec![0.0_f32; W * H];
    let ear_color = [240.0_f32, 70.0, 40.0];
    let mut mixed_pixels = 0;
    let mut protruding_pixels = 0;
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let radius = (((x as f32 + 0.5 - 39.0) / 5.0).powi(2)
                + ((y as f32 + 0.5 - 120.0) / 15.0).powi(2))
            .sqrt();
            let opacity = (0.5 + (1.0 - radius) * 5.0 / 3.0).clamp(0.0, 1.0);
            if opacity <= 0.0 {
                continue;
            }
            mixed_pixels += usize::from(opacity > 0.05 && opacity < 0.95);
            protruding_pixels += usize::from(
                opacity > 0.5 && params["semanticMasks"]["faceContourMask"]["data"][i] == 0,
            );
            for channel in 0..3 {
                source.pixels[i * 4 + channel] = byte(
                    opacity * ear_color[channel]
                        + (1.0 - opacity) * source.pixels[i * 4 + channel] as f32,
                );
            }
            physical_alpha[i] = opacity + (1.0 - opacity) * physical_alpha[i];
            ear_alpha[i] = opacity;
            ears[i] = byte(opacity * 255.0);
            let old_head = params["semanticMasks"]["face"]["data"][i].as_u64().unwrap();
            params["semanticMasks"]["face"]["data"][i] = json!(old_head.max(ears[i] as u64));
            params["subjectMask"]["data"][i] = json!(255);
        }
    }
    assert!(
        mixed_pixels > 100,
        "source ear omitted its mixed optical edge"
    );
    assert!(
        protruding_pixels > 50,
        "source ear was merely a face-interior patch"
    );
    // The independent head silhouette includes the ear, while the facial
    // contour/landmarks still describe the original facial anatomy.
    let outline = params["semanticMasks"]["faceOutline"]["points"]
        .as_array_mut()
        .unwrap();
    let side = outline
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a["x"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["x"].as_f64().unwrap())
        })
        .unwrap()
        .0;
    outline[side]["x"] = json!(34.0 / W as f64);
    params["semanticMasks"]["ears"] = raster(&ears);
    params["experimentalFaceProfile"] = json!("shared-cheek-ear-v1");
    params["__physicalAlpha"] = json!({
        "shape":"raster", "width":W, "height":H,
        "data":physical_alpha.iter().map(|a| a * 255.0).collect::<Vec<_>>(),
        "sourceRegion":{"x":30.0 / W as f64,"y":101.0 / H as f64,
            "width":18.0 / W as f64,"height":38.0 / H as f64}
    });
    ConnectedEarFixture {
        source,
        params,
        physical_alpha,
        ear_alpha,
    }
}

/// A protruding source ear is independently composited over a known constant
/// scene, including a three-pixel mixed optical edge. Source alpha is supplied
/// only in that local region to isolate reconstruction of the vacated B plate
/// from a separate matting error. The actual renderer must transport material
/// away before this test asks for the underlying scene to become visible.
#[test]
fn connected_ear_contraction_reveals_the_scene_instead_of_its_old_pigment() {
    let ConnectedEarFixture {
        source,
        mut params,
        ear_alpha,
        ..
    } = portrait_with_connected_ear();
    params["adjustments"] = json!({"faceWidth":100});
    let trace = std::env::temp_dir().join(format!(
        "image-core-connected-ear-background-{}",
        std::process::id()
    ));
    params["__faceFieldTrace"] = json!(trace);
    params["__faceProbePoints"] = json!((112..129)
        .flat_map(|y| (33..38).map(move |x| [x as f32 + 0.5, y as f32 + 0.5]))
        .collect::<Vec<_>>());
    let output = adjust(&source, &params).unwrap();
    let probes: Value =
        serde_json::from_slice(&std::fs::read(trace.join("render-probes.json")).unwrap()).unwrap();
    let mut vacated_pixels = 0;
    let mut residual_pixels = Vec::new();
    for probe in probes.as_array().unwrap() {
        let x = probe["p"][0].as_f64().unwrap().floor() as usize;
        let y = probe["p"][1].as_f64().unwrap().floor() as usize;
        let membership = probe["sampledAlpha"].as_f64().unwrap();
        let movement = probe["inverseDisplacement"][0].as_f64().unwrap();
        if ear_alpha[y * W + x] < 0.5 || membership > 0.02 || movement > -1.0 {
            continue;
        }
        vacated_pixels += 1;
        let actual = pixel(&output, x, y);
        let error = (0..3)
            .map(|c| actual[c].abs_diff(EAR_SCENE[c]))
            .max()
            .unwrap();
        // Any 8-bit foreground with at most the measured remaining coverage
        // can contribute no more than 255*a codes. Allow three codes for
        // source/output quantization; the known scene itself is constant.
        if error as f64 > 3.0 + 255.0 * membership {
            residual_pixels.push((x, y, actual.to_vec(), error, membership));
        }
    }
    let ear_left = |image: &RgbaImage, y: usize| {
        (0..60)
            .find(|&x| {
                let color = pixel(image, x, y);
                color[0] as i32 - 2 * color[1] as i32 > 0
            })
            .expect("visible independent ear pigment")
    };
    let before = ear_left(&source, 120);
    let after = ear_left(&output, 120);
    eprintln!("connected ear contraction: vacated={vacated_pixels}, residual={}, visible edge={before}->{after}, trace={trace:?}", residual_pixels.len());
    assert!(vacated_pixels >= 6,
        "ear-material counterfactual did not actually uncover enough source ear pixels: {vacated_pixels}; trace={trace:?}");
    assert!(residual_pixels.is_empty(),
        "vacated connected-ear pixels retained old pigment instead of the known scene: {residual_pixels:?}; trace={trace:?}");
    assert!(
        after >= before + 1,
        "visible outer ear edge did not move inward: {before}->{after}"
    );
    std::fs::remove_dir_all(trace).unwrap();
}

#[test]
fn connected_ear_expansion_covers_scene_outside_the_face_only_target() {
    let ConnectedEarFixture {
        source,
        mut params,
        physical_alpha,
        ear_alpha,
    } = portrait_with_connected_ear();
    params["adjustments"] = json!({"faceWidth":-100});
    let trace = std::env::temp_dir().join(format!(
        "image-core-connected-ear-expansion-{}",
        std::process::id()
    ));
    params["__faceFieldTrace"] = json!(trace);
    params["__faceProbePoints"] = json!((108..133)
        .flat_map(|y| (26..36).map(move |x| [x as f32 + 0.5, y as f32 + 0.5]))
        .collect::<Vec<_>>());
    let output = adjust(&source, &params).unwrap();
    let probes: Value =
        serde_json::from_slice(&std::fs::read(trace.join("render-probes.json")).unwrap()).unwrap();
    // Independent bilinear transport of the fixture's physical ear coverage,
    // using the renderer's actual inverse coordinates. This does not use its
    // changed/write flags or its own estimated physical alpha as the oracle.
    let coverage = |values: &[f32], x: f64, y: f64| {
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        assert!(x0 + 1 < W && y0 + 1 < H);
        let (u, v) = (x.fract(), y.fract());
        (1.0 - u) * (1.0 - v) * values[y0 * W + x0] as f64
            + u * (1.0 - v) * values[y0 * W + x0 + 1] as f64
            + (1.0 - u) * v * values[(y0 + 1) * W + x0] as f64
            + u * v * values[(y0 + 1) * W + x0 + 1] as f64
    };
    let mut newly_covered = 0;
    let mut missing_pigment = Vec::new();
    for probe in probes.as_array().unwrap() {
        let x = probe["p"][0].as_f64().unwrap().floor() as usize;
        let y = probe["p"][1].as_f64().unwrap().floor() as usize;
        let i = y * W + x;
        let dx = probe["inverseDisplacement"][0].as_f64().unwrap();
        let dy = probe["inverseDisplacement"][1].as_f64().unwrap();
        let sx = x as f64 + dx;
        let sy = y as f64 + dy;
        let transported_ear = coverage(&ear_alpha, sx, sy);
        let transported_union = coverage(&physical_alpha, sx, sy);
        if physical_alpha[i] != 0.0
            || probe["targetAlpha"].as_f64().unwrap() > 1e-6
            || transported_ear < 0.6
            || (transported_ear - transported_union).abs() > 1e-6
        {
            continue;
        }
        assert_eq!(pixel(&source, x, y), EAR_SCENE);
        assert!(dx > 1.0, "fixture did not actually expand the left ear");
        assert!(!probe["frozen"].as_bool().unwrap());
        assert!(probe["oldAlpha"].as_f64().unwrap() <= 1e-6);
        newly_covered += 1;
        // Ear R-2G=100, scene=-60, original face=-90. With independently
        // transported ear coverage >=.6 this location must contain the unique
        // red ear pigment, not the untouched scene. Quantization allowance is
        // far smaller than their >=96-code separating colour score.
        let color = pixel(&output, x, y);
        let score = color[0] as i32 - 2 * color[1] as i32;
        if score < 20 {
            missing_pigment.push((x, y, transported_ear, color.to_vec(), score));
        }
    }
    eprintln!("connected ear expansion: independent newly covered={newly_covered}, missing={}, trace={trace:?}", missing_pigment.len());
    assert!(newly_covered >= 6,
        "fixture did not exercise expansion beyond original ear AND face-only target: {newly_covered}; trace={trace:?}");
    assert!(missing_pigment.is_empty(),
        "legitimately transported ear was clipped outside the face-only target: {missing_pigment:?}; trace={trace:?}");
    std::fs::remove_dir_all(trace).unwrap();
}

/// An independently photographed opaque ornament has a green pigment that
/// cannot be made by any convex mixture of this fixture's skin/neck/backdrop.
/// Its optical edge has three source rings (65%, 25%, 10% coverage), so the
/// test includes partial attachment alpha rather than only a binary interior.
fn portrait_with_green_attachment(kind: &str) -> (RgbaImage, Value, Vec<bool>) {
    let (mut source, mut params) = joint_portrait();
    let mut attachment = vec![0_u8; W * H];
    let mut pigment_support = vec![false; W * H];
    let pigment = [5.0_f32, 245.0, 10.0];
    for y in 0..H {
        for x in 0..W {
            let dx = if x < 49 { 49 - x } else { x.saturating_sub(53) };
            let dy = if y < 109 {
                109 - y
            } else {
                y.saturating_sub(129)
            };
            // A diamond-shaped PSF footprint gives a known antialiased source
            // edge with no diagonal point farther than its labelled radius.
            let a = match dx + dy {
                0 => 1.0,
                1 => 0.65,
                2 => 0.25,
                3 => 0.10,
                _ => continue,
            };
            let i = y * W + x;
            assert_eq!(
                params["semanticMasks"]["faceContourMask"]["data"][i], 255,
                "ornament fixture must be on known opaque facial material"
            );
            for c in 0..3 {
                source.pixels[i * 4 + c] =
                    byte(a * pigment[c] + (1.0 - a) * source.pixels[i * 4 + c] as f32);
            }
            attachment[i] = byte(a * 255.0);
            pigment_support[i] = true;
        }
    }
    params["semanticMasks"][kind] = raster(&attachment);
    (source, params, pigment_support)
}

fn green_pigment_score(rgba: &[u8]) -> i32 {
    // Skin scores -85, all original background texels <=5, neck <=-50.
    // Even the 10%-coverage ornament fringe scores above 40. A value >20
    // outside the independently known source support therefore identifies
    // copied ornament colour, not legitimate skin/background reconstruction.
    5 * rgba[1] as i32 - 3 * rgba[0] as i32 - 2 * rgba[2] as i32
}

/// Unlike the on-face ornament fixture, this observed ear component has no
/// source-material connection to the face or neck. Its five mixed rings are
/// independently composited source truth, including a weak outer footprint
/// close to the editable cheek. No explicit protection mask makes it pass.
#[test]
fn disconnected_ear_keeps_all_mixed_pigment_and_other_people_during_real_face_edits() {
    let (mut source, mut params) = joint_portrait();
    let mut ears = vec![0_u8; W * H];
    let mut pigment_support = vec![false; W * H];
    let mut other_person_support = vec![false; W * H];
    let mut mixed_count = 0;
    let pigment = [5.0_f32, 245.0, 10.0];
    for y in 0..H {
        for x in 0..W {
            let dx = if x < 32 { 32 - x } else { x.saturating_sub(33) };
            let dy = if y < 112 {
                112 - y
            } else {
                y.saturating_sub(126)
            };
            let opacity = match dx + dy {
                0 => 1.0,
                1 => 0.65,
                2 => 0.40,
                3 => 0.30,
                4 => 0.20,
                5 => 0.15,
                _ => continue,
            };
            let i = y * W + x;
            for kind in ["faceContourMask", "skin", "neck"] {
                assert_eq!(
                    params["semanticMasks"][kind]["data"][i], 0,
                    "disconnected source ear overlapped {kind} at {x},{y}"
                );
            }
            for channel in 0..3 {
                source.pixels[i * 4 + channel] = byte(
                    opacity * pigment[channel]
                        + (1.0 - opacity) * source.pixels[i * 4 + channel] as f32,
                );
            }
            ears[i] = byte(opacity * 255.0);
            pigment_support[i] = true;
            mixed_count += usize::from(opacity < 0.5);
            // Head/person observations can include a disconnected component;
            // they do not assert that its optical boundary belongs to skin.
            params["semanticMasks"]["face"]["data"][i] = json!(ears[i]);
            params["subjectMask"]["data"][i] = json!(255);
            assert!(
                green_pigment_score(&source.pixels[i * 4..i * 4 + 4]) > 20,
                "the independent source pigment marker is not distinguishable"
            );
        }
    }
    params["semanticMasks"]["ears"] = raster(&ears);
    assert!(
        mixed_count > 100,
        "fixture omitted low-confidence real mixed coverage"
    );
    // A vertical strip of observed backdrop separates every ear texel from
    // face/neck material. This source assertion is independent of joint_owned.
    for y in 107..132 {
        for x in 39..41 {
            let i = y * W + x;
            assert!(!pigment_support[i]);
            for kind in ["faceContourMask", "skin", "neck"] {
                assert_eq!(params["semanticMasks"][kind]["data"][i], 0);
            }
        }
    }
    // A separate person's coordinate texture is also in the editable image
    // region but outside the selected head/skin/neck. It is not an alpha donor.
    for y in 102..150 {
        for x in 133..140 {
            let i = y * W + x;
            for kind in ["face", "faceContourMask", "skin", "neck"] {
                assert_eq!(params["semanticMasks"][kind]["data"][i], 0);
            }
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                110 + (x % 5) as u8,
                70 + (y % 7) as u8,
                115,
                255,
            ]);
            params["subjectMask"]["data"][i] = json!(255);
            other_person_support[i] = true;
        }
    }
    let mut failures = Vec::new();
    for mode in ["shared-cheek-ear-v1", "shared-cheek-ear-periodic-v1"] {
        for amount in [-30_i32, 30] {
            params["experimentalFaceProfile"] = json!(mode);
            params["adjustments"] = json!({"faceWidth":amount});
            let output = adjust(&source, &params).unwrap();
            let moved_rows = CHECK_ROWS
                .clone()
                .filter(|&y| {
                    let before = skin_span(&source, y).1 as i32;
                    let after = skin_span(&output, y).1 as i32;
                    (before - after) * amount.signum() >= 1
                })
                .count();
            assert!(moved_rows>=16,
                "{mode} {amount}: safety check omitted actual face contour editing ({moved_rows}/32 rows)");
            let mut ear_changes = 0;
            let mut weak_edge_changes = 0;
            let mut other_changes = 0;
            let mut leaked_pigment = 0;
            let mut first_failure = None;
            for i in 0..W * H {
                let original = &source.pixels[i * 4..i * 4 + 4];
                let actual = &output.pixels[i * 4..i * 4 + 4];
                let changed = actual != original;
                if pigment_support[i] {
                    ear_changes += usize::from(changed);
                    weak_edge_changes += usize::from(changed && ears[i] < 128);
                    if changed && first_failure.is_none() {
                        first_failure = Some((i % W, i / W));
                    }
                } else {
                    assert!(
                        green_pigment_score(original) <= 5,
                        "unlabelled pigment in the independent source oracle"
                    );
                    leaked_pigment += usize::from(green_pigment_score(actual) > 20);
                }
                other_changes += usize::from(other_person_support[i] && changed);
            }
            if ear_changes != 0 || other_changes != 0 || leaked_pigment != 0 {
                failures.push(format!("{mode} {amount}: ear={ear_changes}, weak edge={weak_edge_changes}, other person={other_changes}, copied pigment={leaked_pigment}, first changed ear={first_failure:?}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "disconnected observed-ear material moved: {}",
        failures.join("; ")
    );
}

#[test]
fn fixed_ear_and_accessory_keep_their_mixed_edge_without_copying_pigment() {
    for kind in ["accessories", "ears"] {
        let (source, mut params, pigment_support) = portrait_with_green_attachment(kind);
        let before = skin_span(&source, 151);
        for amount in [-100, 100] {
            params["adjustments"] = json!({"faceWidth":amount});
            let output = public_adjust(&source, &params).unwrap();
            let after = skin_span(&output, 151);
            assert!(
                if amount < 0 {
                    after.1 > before.1
                } else {
                    after.1 < before.1
                },
                "{kind} fixture cancelled the actual face edit: {amount}, {before:?}->{after:?}"
            );
            let mut mixed_edge_pixels = 0;
            for i in 0..W * H {
                let original = &source.pixels[i * 4..i * 4 + 4];
                let actual = &output.pixels[i * 4..i * 4 + 4];
                if pigment_support[i] {
                    assert_eq!(
                        actual,
                        original,
                        "{kind} source pigment/antialias edge moved at {},{} for {amount}",
                        i % W,
                        i / W
                    );
                    let a = params["semanticMasks"][kind]["data"][i].as_u64().unwrap();
                    mixed_edge_pixels += usize::from(a > 0 && a < 128);
                } else {
                    assert!(
                        green_pigment_score(original) <= 5,
                        "source oracle contains unlabelled green pigment"
                    );
                    assert!(green_pigment_score(actual) <= 20,
                        "{kind} pigment was copied outside its source support at {},{} for {amount}: {original:?}->{actual:?}", i % W, i / W);
                }
            }
            assert!(
                mixed_edge_pixels > 50,
                "fixture omitted the weak attachment edge"
            );
        }
    }
}
