//! An interior aperture is not the visible outer waist silhouette.
//! The opaque material beside it must not acquire an axial motion step.

use super::*;
use serde_json::json;

fn fixture(open_to_scene: bool) -> Vec<f32> {
    let (w, h) = (160, 200);
    let mut alpha = vec![0.0; w * h];
    for y in 10..190 {
        for x in 20..140 {
            let aperture = (70..112).contains(&y)
                && (if open_to_scene { 0..72 } else { 60..72 }).contains(&x);
            if !aperture {
                alpha[y * w + x] = 1.0;
            }
        }
    }
    alpha
}

fn profile(alpha: &[f32], amount: f32) -> ContourProfile {
    ContourProfile::new(
        Warp {
            axis: Axis {
                origin: (80.0, 10.0),
                direction: (0.0, 1.0),
                length: 180.0,
                radius: 65.0,
            },
            amount,
            center: 0.5,
            extent: 0.55,
            limb: false,
        },
        alpha,
        160,
        200,
        false,
        4.0,
    )
}

#[test]
fn off_axis_internal_aperture_does_not_cut_opaque_outer_material_motion() {
    let alpha = fixture(false);
    for amount in [-0.6, 0.6] {
        let p = profile(&alpha, amount);
        let motion: Vec<_> = (45..140)
            .map(|y| {
                assert_eq!(alpha[y * 160 + 30], 1.0);
                p.displacement((30.5, y as f32 + 0.5)).0
            })
            .collect();
        let jump = motion.windows(2).map(|v| (v[1] - v[0]).abs()).fold(0.0, f32::max);
        assert!(jump < 1.0, "continuous opaque source has a {jump}px adjacent motion jump at {amount}");
        assert!(motion.iter().map(|v| v.abs()).fold(0.0, f32::max) > 3.0,
            "the aperture must not erase the requested waist edit");
    }
}

#[test]
fn a_gap_connected_to_the_scene_is_still_the_outer_silhouette() {
    let alpha = fixture(true);
    let p = profile(&alpha, 0.6);
    assert!(p.sections[32].1 < 10.0, "do not bridge an observed exterior gap");
    assert_eq!(p.displacement((30.5, 100.5)), (0.0, 0.0));
}

#[test]
fn the_off_axis_aperture_remains_optically_absent_and_source_is_not_modified() {
    let alpha = fixture(false);
    let before = alpha.clone();
    let p = profile(&alpha, 0.6);
    assert_eq!(alpha, before);
    assert_eq!(alpha[90 * 160 + 65], 0.0);
    assert!(p.sections[32].1 > 50.0, "measure the external waist contour across an enclosed aperture");
}

#[test]
fn a_single_pixel_diagonal_channel_remains_connected_to_the_scene() {
    let mut alpha = vec![1.0; 32 * 32];
    for i in 0..=15 {
        alpha[i * 32 + i] = 0.0;
    }
    let support = exterior_contour_support(&alpha, 32, 32);
    assert_eq!(support, alpha, "an eight-connected scene opening must not be closed by geometry");
}

fn photo_fixture(amount: i32, protect_aperture: bool) -> (RgbaImage, Value) {
    let (w, h) = (160, 200);
    let alpha = fixture(false);
    let mut pixels = vec![255; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let color = if alpha[i] == 1.0 {
                [30 + x as u8, 40, 180]
            } else if (60..72).contains(&x) && (70..112).contains(&y) {
                // A distinct, observed lower layer through a real aperture.
                [20, 210, 30]
            } else {
                [160, 150, 140]
            };
            pixels[i * 4..i * 4 + 3].copy_from_slice(&color);
        }
    }
    let mask = json!({"shape":"raster","width":w,"height":h,
        "data":alpha.iter().map(|a| a * 255.0).collect::<Vec<_>>()});
    let mut params = json!({"bodyPart":"waist",
        "region":{"x":0.01,"y":0.01,"width":0.98,"height":0.98},
        "landmarks":[{"x":0.5,"y":0.05},{"x":0.5,"y":0.5},{"x":0.5,"y":0.95}],
        "subjectMask":mask,"adjustments":{"waist":amount}});
    if protect_aperture {
        params["protectMask"] = json!({"shape":"rect", "region":{
            "x":60.0 / w as f32,"y":70.0 / h as f32,
            "width":12.0 / w as f32,"height":42.0 / h as f32}});
    }
    (RgbaImage::new(w as u32, h as u32, pixels).unwrap(), params)
}

#[test]
fn fixed_lower_material_stays_exact_beside_continuously_moving_cloth() {
    for amount in [-60, 60] {
        let (source, params) = photo_fixture(amount, true);
        let result = adjust(&source, &params).unwrap();
        for y in 70..112 {
            for x in 60..72 {
                let i = (y * 160 + x) * 4;
                assert_eq!(result.pixels[i..i + 4], source.pixels[i..i + 4]);
            }
        }
        // The red channel is a source-coordinate ruler, independent of the
        // profile internals. Its axial discontinuity exposes the old tear.
        let jumps = (50..135).map(|y| {
            let a = result.pixels[(y * 160 + 32) * 4] as i32;
            let b = result.pixels[((y + 1) * 160 + 32) * 4] as i32;
            (a - b).abs()
        }).max().unwrap();
        assert!(jumps <= 2, "protected internal material caused a cloth seam: {jumps}");
        assert_ne!(result, source, "retain the requested contour change");
    }
}

#[test]
fn geometric_enclosure_does_not_paint_over_a_real_visible_aperture() {
    for amount in [-60, 60] {
        let (source, params) = photo_fixture(amount, false);
        let result = adjust(&source, &params).unwrap();
        // This original aperture core remains uncovered for these two
        // requests. Geometry-only support must not turn it into cloth alpha.
        for y in 82..100 {
            for x in 66..70 {
                let i = (y * 160 + x) * 4;
                assert_eq!(result.pixels[i..i + 4], source.pixels[i..i + 4],
                    "visible lower layer was painted over at ({x},{y})");
            }
        }
        assert_ne!(result, source);
    }
}
