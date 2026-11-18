//! Source-layer contracts through the actual reconstruction and sampler.
//!
//! The independent fixture has foreground 180 over backdrop 20. The excluded
//! clothing pixel belongs to the visible lower layer, but is not a scene donor.
//! These tests do not establish that inferred layers in a photograph are true.

use super::*;

fn grayscale(values: &[u8]) -> RgbaImage {
    RgbaImage::new(
        values.len() as u32,
        1,
        values.iter().flat_map(|&v| [v, v, v, 255]).collect(),
    )
    .unwrap()
}

fn observed(alpha: &[f32]) -> RgbaImage {
    grayscale(
        &alpha
            .iter()
            .map(|&a| byte(180.0 * a + 20.0 * (1.0 - a)))
            .collect::<Vec<_>>(),
    )
}

fn reconstruct_plate(
    source: &RgbaImage,
    alpha: &[f32],
    holes: &[bool],
    excluded: &[bool],
) -> RgbaImage {
    let count = alpha.len();
    reconstruct_with_scene(
        source,
        holes,
        alpha,
        &vec![false; count],
        excluded,
        &vec![0; count],
        Some(&grayscale(&vec![20; count])),
        None,
    )
    .unwrap()
}

fn gray_values(image: &RgbaImage) -> Vec<u8> {
    image.pixels.chunks_exact(4).map(|p| p[0]).collect()
}

#[test]
fn complete_source_plate_reconstructs_every_observed_texel_after_pairing() {
    let alpha = [0.0, 0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0, 0.0];
    let source = observed(&alpha);
    let mut background = reconstruct_plate(&source, &alpha, &alpha.map(|a| a > 0.0), &[false; 9]);
    assert_eq!(gray_values(&background), [20; 9]);
    let foreground = layer_pair::pair(&source, &alpha, &mut background).unwrap();
    for i in 0..alpha.len() {
        let (pixel, membership) = sample(&foreground, &alpha, i as f32, 0.0);
        for (c, channel) in pixel.iter().enumerate().take(3) {
            assert_eq!(
                byte(channel + (1.0 - membership) * background.pixels[i * 4 + c] as f32),
                source.pixels[i * 4 + c],
                "source reconstruction at texel {i}, channel {c}"
            );
        }
    }
}

#[test]
fn excluded_zero_alpha_clothing_stays_visible_without_becoming_a_scene_donor() {
    let alpha = [0.0, 0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0, 0.0];
    let mut source = observed(&alpha);
    source.pixels[4..8].copy_from_slice(&[240, 240, 240, 255]);
    let mut excluded = [false; 9];
    excluded[1] = true;
    let mut background = reconstruct_plate(&source, &alpha, &alpha.map(|a| a > 0.0), &excluded);
    let expected = [20, 240, 20, 20, 20, 20, 20, 20, 20];
    assert_eq!(gray_values(&background), expected);
    layer_pair::pair(&source, &alpha, &mut background).unwrap();
    assert_eq!(gray_values(&background), expected);
}

#[test]
fn destination_thin_holes_do_not_define_a_complete_source_plate() {
    let alpha = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let source = observed(&alpha);
    let mut thin = [false; 9];
    thin[2] = true;
    let destination_plate = reconstruct_plate(&source, &alpha, &thin, &[false; 9]);
    let source_plate = reconstruct_plate(&source, &alpha, &alpha.map(|a| a > 0.0), &[false; 9]);
    assert_eq!(
        gray_values(&destination_plate),
        [20, 20, 20, 180, 180, 180, 180, 20, 20],
        "the unexposed opaque source interior is still foreground color"
    );
    assert_eq!(gray_values(&source_plate), [20; 9]);
}

#[test]
fn fixed_source_plate_stays_continuous_across_destination_coverage_switch() {
    let alpha = [0.0, 0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0, 0.0];
    let source = observed(&alpha);
    let mut background = reconstruct_plate(&source, &alpha, &alpha.map(|a| a > 0.0), &[false; 9]);
    let foreground = layer_pair::pair(&source, &alpha, &mut background).unwrap();
    // The same destination texel reads across an opaque source center.
    // Its hole flag closes and reopens without selecting a different source B.
    let samples = [3.99999, 4.0, 4.00001].map(|x| sample(&foreground, &alpha, x, 0.0));
    assert_eq!(samples.map(|(_, a)| a < 1.0 - 1e-6), [true, false, true]);
    let colors = samples.map(|(pixel, a)| pixel[0] + (1.0 - a) * background.pixels[12] as f32);
    assert!((colors[0] - colors[1]).abs() < 0.01, "{colors:?}");
    assert!((colors[2] - colors[1]).abs() < 0.01, "{colors:?}");
    assert_eq!(colors[1], 180.0);
}
