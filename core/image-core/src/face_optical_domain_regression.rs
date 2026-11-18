//! Independent analytic portrait over a known affine backdrop, viewed through
//! two different source crops. Coordinates describe the same physical scene.
use super::estimate;
use crate::{engine::byte, face_matting, RgbaImage};

fn scene(
    width: usize,
    height: usize,
    offset_x: usize,
    offset_y: usize,
) -> (RgbaImage, Vec<f32>, Vec<[u8; 3]>) {
    let mut pixels = Vec::with_capacity(width * height * 4);
    let mut alpha = Vec::with_capacity(width * height);
    let mut truth = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let sx = x as f32 - offset_x as f32;
            let sy = y as f32 - offset_y as f32;
            let radius = (((sx - 55.0) / 25.0).powi(2) + ((sy - 55.0) / 30.0).powi(2)).sqrt();
            let a = ((1.1 - radius) / 0.2).clamp(0.0, 1.0);
            let background = [
                30.0 + sx * 0.45,
                50.0 + sy * 0.35,
                80.0 + sx * 0.12 + sy * 0.21,
            ];
            let foreground = [
                180.0 + (sy * 0.4).sin() * 10.0,
                130.0 + (sx * 0.2).sin() * 7.0,
                90.0,
            ];
            alpha.push(a);
            truth.push(background.map(byte));
            pixels.extend(
                (0..3)
                    .map(|c| byte(a * foreground[c] + (1.0 - a) * background[c]))
                    .chain([255]),
            );
        }
    }
    (
        RgbaImage::new(width as u32, height as u32, pixels).unwrap(),
        alpha,
        truth,
    )
}

#[test]
fn same_observed_edge_retains_background_estimate_when_render_view_is_padded_and_shifted() {
    let (small, a, truth) = scene(120, 120, 0, 0);
    let (large, b, _) = scene(150, 150, 7, 13);
    // The two views really contain identical source evidence at each shared
    // physical location; these are not two independently generated portraits.
    for y in 0..120 {
        for x in 0..120 {
            let i = y * 120 + x;
            let j = (y + 13) * 150 + x + 7;
            assert_eq!(a[i], b[j]);
            assert_eq!(
                &small.pixels[4 * i..4 * i + 4],
                &large.pixels[4 * j..4 * j + 4]
            );
        }
    }
    let (_, old_small) = face_matting::estimate(&small, &a).unwrap();
    let (_, old_large) = face_matting::estimate(&large, &b).unwrap();
    let (_, stable_small) = estimate(&small, &a).unwrap();
    let (_, stable_large) = estimate(&large, &b).unwrap();
    let (mut old_difference, mut stable_difference) = (0_u8, 0_u8);
    let (mut old_error, mut stable_error, mut count) = (0.0, 0.0, 0_usize);
    for y in 0..120 {
        for x in 0..120 {
            let i = y * 120 + x;
            let j = (y + 13) * 150 + x + 7;
            if a[i] <= 0.0 || a[i] >= 1.0 {
                continue;
            }
            for c in 0..3 {
                old_difference = old_difference
                    .max(old_small.pixels[4 * i + c].abs_diff(old_large.pixels[4 * j + c]));
                stable_difference = stable_difference
                    .max(stable_small.pixels[4 * i + c].abs_diff(stable_large.pixels[4 * j + c]));
                old_error += old_small.pixels[4 * i + c].abs_diff(truth[i][c]) as f64;
                stable_error += stable_small.pixels[4 * i + c].abs_diff(truth[i][c]) as f64;
                count += 1;
            }
        }
    }
    assert!(count > 0, "the fixture must observe a genuine mixed edge");
    assert!(
        old_difference > 1,
        "old size-dependent estimator did not reproduce"
    );
    assert!(
        stable_difference <= 1,
        "unchanged source edge was recolored by view padding: {stable_difference}"
    );
    // Invariance is not accuracy: keep the independent truth error visible,
    // without tuning this regression to a favorable quality threshold.
    eprintln!(
        "mixed-edge channels={count}, old padding delta={old_difference}, stable delta={stable_difference}, known-B MAE old={}, stable={}",
        old_error / count as f64,
        stable_error / count as f64,
    );
}
