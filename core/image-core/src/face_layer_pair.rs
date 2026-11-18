//! Isolated source-consistent color pairing for an opaque portrait layer.
//!
//! A regularized F/B estimate is a prior, not an exact decomposition. Once the
//! background plate is chosen, project its byte RGB into the source-mixture
//! feasible interval and derive F from that SAME plate. The renderer must use
//! this paired plate and foreground together; changing B later breaks the
//! invariant. No opacity, geometry, or original source texel is changed.
//! Exactness here means byte reconstruction at source texel centers, not an
//! exact floating-point factorization. Subpixel transport still carries the
//! quantization error of the byte foreground.

use crate::RgbaImage;

pub(crate) fn pair(
    source: &RgbaImage,
    alpha: &[f32],
    background: &mut RgbaImage,
) -> Option<RgbaImage> {
    let count = source.width as usize * source.height as usize;
    if source.width != background.width
        || source.height != background.height
        || source.pixels.len() != count * 4
        || background.pixels.len() != count * 4
        || alpha.len() != count
        || source.pixels.chunks_exact(4).any(|p| p[3] != 255)
        || background.pixels.chunks_exact(4).any(|p| p[3] != 255)
        || alpha
            .iter()
            .any(|a| !a.is_finite() || !(0.0..=1.0).contains(a))
    {
        return None;
    }
    let mut foreground = source.clone();
    for (i, &a) in alpha.iter().enumerate() {
        let a = a as f64;
        if a == 1.0 {
            // All observed color belongs to this opaque moving material.
            continue;
        }
        if a == 0.0 {
            background.pixels[i * 4..i * 4 + 4].copy_from_slice(&source.pixels[i * 4..i * 4 + 4]);
            continue;
        }
        for c in 0..3 {
            let at = i * 4 + c;
            let observed = source.pixels[at] as f64;
            // C itself is always in this interval, so a feasible integer B
            // exists even at very small alpha. Choose the closest byte B to
            // the proposed plate; do not clip an infeasible F afterwards.
            // Express the interval as offsets from its known feasible point
            // B=C. Subtracting nearly equal terms in (C-255*a)/(1-a)
            // can round above C at tiny alpha, making ceil invert [lo, hi].
            let ratio = a / (1.0 - a);
            let lo = (observed - (255.0 - observed) * ratio).ceil().max(0.0);
            let hi = (observed + observed * ratio).floor().min(255.0);
            let b = (background.pixels[at] as f64).clamp(lo, hi);
            background.pixels[at] = b as u8;
            // This equivalent form also keeps F=C exactly when B=C; the
            // cancelled numerator of the direct formula can become zero.
            let f = b + (observed - b) / a;
            // The feasible interval already bounds f. The clamp only removes
            // floating-point roundoff before byte conversion.
            foreground.pixels[at] = f.round().clamp(0.0, 255.0) as u8;
        }
    }
    Some(foreground)
}

/// Sample the source-consistent premultiplied foreground without quantizing F.
///
/// Images must be matching opaque canvases with finite alpha in [0, 1]. The
/// physical path uses `pair` to project B first. An explicitly assigned encoded
/// contrast layer may instead retain signed residuals against a reviewed scene
/// prior; that is an editing hypothesis, not physical F/B recovery. In either
/// case the SAME immutable plate must be used for final composition.
/// Coordinates, weights, out-of-bounds handling and membership match `sample`.
pub(crate) fn sample_residual(
    source: &RgbaImage,
    alpha: &[f32],
    background: &RgbaImage,
    x: f32,
    y: f32,
) -> ([f32; 4], f32) {
    let w = source.width as usize;
    let h = source.height as usize;
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (tx, ty) = (x - x.floor(), y - y.floor());
    let mut pixel = [0.0; 4];
    let mut membership = 0.0;
    for (x, y, weight) in [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ] {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            continue;
        }
        let i = y as usize * w + x as usize;
        let a = alpha[i];
        membership += a * weight;
        for (c, value) in pixel.iter_mut().enumerate().take(3) {
            let observed = source.pixels[i * 4 + c] as f64;
            let b = background.pixels[i * 4 + c] as f64;
            // P=C-(1-a)B, written without subtracting C from a nearly
            // identical B when alpha is tiny. One final f32 rounding avoids
            // both byte-F quantization and an extra product cancellation.
            let premultiplied = ((observed - b) + a as f64 * b) as f32;
            *value += premultiplied * weight;
        }
        pixel[3] += a * weight;
    }
    (pixel, membership.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(pixels: &[[u8; 4]]) -> RgbaImage {
        RgbaImage {
            width: pixels.len() as u32,
            height: 1,
            pixels: pixels.iter().flatten().copied().collect(),
        }
    }
    fn reconstruct(f: &RgbaImage, b: &RgbaImage, a: &[f32]) -> Vec<u8> {
        (0..a.len())
            .flat_map(|i| {
                // Exercise the renderer's f32 sampler and composition order,
                // including the final-alpha division, rather than an ideal
                // f64 mixture that can hide renderer rounding differences.
                let (pixel, membership) = super::super::sample(
                    f,
                    a,
                    (i % f.width as usize) as f32,
                    (i / f.width as usize) as f32,
                );
                let bg_weight = b.pixels[i * 4 + 3] as f32 / 255.0 * (1.0 - membership);
                let final_alpha = pixel[3] + bg_weight;
                (0..3).map(move |c| {
                    ((pixel[c] + b.pixels[i * 4 + c] as f32 * bg_weight) / final_alpha)
                        .round()
                        .clamp(0.0, 255.0) as u8
                })
            })
            .collect()
    }

    fn composite(pixel: [f32; 4], membership: f32, background: &[u8]) -> [u8; 4] {
        let bg_weight = background[3] as f32 / 255.0 * (1.0 - membership);
        let final_alpha = pixel[3] + bg_weight;
        let mut output = [0; 4];
        for c in 0..3 {
            output[c] = if final_alpha > 1e-6 {
                ((pixel[c] + background[c] as f32 * bg_weight) / final_alpha)
                    .round()
                    .clamp(0.0, 255.0) as u8
            } else {
                0
            };
        }
        output[3] = (final_alpha * 255.0).round().clamp(0.0, 255.0) as u8;
        output
    }

    #[test]
    fn residual_transport_preserves_a_known_subpixel_mixture() {
        let source = image(&[[100, 100, 100, 255], [102, 102, 102, 255]]);
        let alpha = [0.6; 2];
        let mut background = image(&[[0, 0, 0, 255]; 2]);
        let foreground = pair(&source, &alpha, &mut background).unwrap();
        let (old, old_membership) = super::super::sample(&foreground, &alpha, 0.2, 0.0);
        assert_eq!(
            composite(old, old_membership, &background.pixels[..4])[0],
            101
        );
        let (pixel, membership) = sample_residual(&source, &alpha, &background, 0.2, 0.0);
        // Constant black backdrop: .8*100+.2*102 = 100.4, rounding to 100.
        assert!((pixel[0] - 100.4).abs() < 1e-5);
        assert_eq!(
            composite(pixel, membership, &background.pixels[..4]),
            [100, 100, 100, 255]
        );
    }

    #[test]
    fn residual_keeps_a_tiny_nonzero_foreground_without_cancellation() {
        let source = image(&[[91, 47, 19, 255]]);
        for a in [f32::from_bits(1), f32::MIN_POSITIVE, 1e-20] {
            let mut background = image(&[[132, 105, 45, 255]]);
            pair(&source, &[a], &mut background).unwrap();
            let (pixel, membership) = sample_residual(&source, &[a], &background, 0.0, 0.0);
            assert_eq!(membership, a);
            for c in 0..3 {
                assert_eq!(pixel[c], a * source.pixels[c] as f32);
                assert!(pixel[c] > 0.0);
            }
        }
    }

    #[test]
    fn residual_bilinear_transport_matches_an_independent_constant_plate_solution() {
        let source = RgbaImage {
            width: 2,
            height: 2,
            pixels: [64, 128, 191, 100]
                .into_iter()
                .flat_map(|c| [c, c, c, 255])
                .collect(),
        };
        let alpha = [0.5, 1.0, 0.75, 0.6];
        let mut background = RgbaImage {
            width: 2,
            height: 2,
            pixels: [128, 128, 128, 255].repeat(4),
        };
        pair(&source, &alpha, &mut background).unwrap();
        assert!(background
            .pixels
            .chunks_exact(4)
            .all(|p| p == [128, 128, 128, 255]));
        for x in [0.0_f32, 0.13, 0.27, 0.61, 0.87, 1.0] {
            for y in [0.0_f32, 0.13, 0.27, 0.61, 0.87, 1.0] {
                let (pixel, membership) = sample_residual(&source, &alpha, &background, x, y);
                // With spatially constant B, the composited transport equals
                // ordinary interpolation of observed C, independent of alpha.
                let (t, u) = (x as f64, y as f64);
                let truth = (1.0 - t) * (1.0 - u) * 64.0
                    + t * (1.0 - u) * 128.0
                    + (1.0 - t) * u * 191.0
                    + t * u * 100.0;
                let observed = pixel[0] + (1.0 - membership) * 128.0;
                assert!(
                    (observed as f64 - truth).abs() < 4e-5,
                    "x={x}, y={y}, observed={observed}, truth={truth}"
                );
            }
        }
    }

    #[test]
    fn residual_identity_covers_opaque_and_alpha_endpoints() {
        let source = image(
            &(0..=255)
                .map(|c| [c, 255 - c, c / 2, 255])
                .collect::<Vec<_>>(),
        );
        for a in [
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::from_bits(612_368_385),
            1e-7,
            0.001,
            0.5,
            0.569151,
            f32::from_bits(1.0_f32.to_bits() - 1),
            1.0,
        ] {
            for prior in [0, 17, 128, 254, 255] {
                let alpha = vec![a; 256];
                let mut background = image(&vec![[prior, 255 - prior, prior, 255]; 256]);
                pair(&source, &alpha, &mut background).unwrap();
                for i in 0..256 {
                    let (pixel, membership) =
                        sample_residual(&source, &alpha, &background, i as f32, 0.0);
                    assert_eq!(
                        composite(pixel, membership, &background.pixels[i * 4..i * 4 + 4]),
                        source.pixels[i * 4..i * 4 + 4],
                        "alpha={a}, prior={prior}, pixel={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_source_plate_is_continuous_across_zero_transport() {
        let source = image(&[
            [100, 100, 100, 255],
            [91, 91, 91, 255],
            [110, 110, 110, 255],
        ]);
        let alpha = [0.2, 0.569151, 0.8];
        let mut background = image(&[[80, 80, 80, 255]; 3]);
        pair(&source, &alpha, &mut background).unwrap();
        let plate = background.pixels.clone();
        for shift in [-0.01_f32, -0.001, -1e-5, 0.0, 1e-5, 0.001, 0.01] {
            let (pixel, membership) =
                sample_residual(&source, &alpha, &background, 1.0 + shift, 0.0);
            let other = if shift < 0.0 { 100.0 } else { 110.0 };
            let truth = 91.0 + (other - 91.0) * shift.abs();
            let observed = pixel[0] + (1.0 - membership) * 80.0;
            assert!(
                (observed - truth).abs() < 3e-5,
                "shift={shift}, observed={observed}, truth={truth}"
            );
            assert_eq!(
                composite(pixel, membership, &background.pixels[4..8]),
                [91, 91, 91, 255]
            );
        }
        assert_eq!(background.pixels, plate);
    }

    #[test]
    fn residual_sampler_keeps_existing_bilinear_boundary_semantics() {
        let source = RgbaImage {
            width: 2,
            height: 2,
            pixels: vec![
                10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
            ],
        };
        let alpha = [1.0; 4];
        let mut background = source.clone();
        pair(&source, &alpha, &mut background).unwrap();
        for x in [-2.0, -0.2, 0.0, 0.3, 1.0, 1.2, 2.0] {
            for y in [-2.0, -0.2, 0.0, 0.7, 1.0, 1.2, 2.0] {
                assert_eq!(
                    sample_residual(&source, &alpha, &background, x, y),
                    super::super::sample(&source, &alpha, x, y)
                );
            }
        }
    }

    #[test]
    fn tiny_alpha_does_not_invert_the_feasible_byte_interval() {
        // Direct evaluation of (C-255*a)/(1-a) used to round above the
        // necessarily feasible C, then ceil produced lo > hi and panicked.
        for a in [f32::from_bits(612_368_385), f32::from_bits(648_548_915)] {
            let source = image(&[[255, 247, 1, 255]]);
            let mut background = image(&[[0, 16, 255, 255]]);
            let f = pair(&source, &[a], &mut background).unwrap();
            assert_eq!(background.pixels, source.pixels);
            assert_eq!(f.pixels, source.pixels);
            assert_eq!(reconstruct(&f, &background, &[a]), [255, 247, 1]);
        }
    }

    #[test]
    fn changing_the_background_requires_a_new_foreground_pair() {
        let source = image(&[[100, 80, 60, 255]]);
        let old_f = image(&[[180, 120, 60, 255]]);
        let mut background = image(&[[100, 60, 20, 255]]);
        assert_eq!(reconstruct(&old_f, &background, &[0.5]), [140, 90, 40]);
        let f = pair(&source, &[0.5], &mut background).unwrap();
        assert_eq!(&f.pixels[..3], &[100, 100, 100]);
        assert_eq!(reconstruct(&f, &background, &[0.5]), [100, 80, 60]);
    }

    #[test]
    fn every_source_byte_reconstructs_for_fractional_alpha_and_extreme_plates() {
        for a in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1e-7,
            0.001,
            0.05,
            0.25,
            0.5,
            0.569151,
            0.9,
            0.999999,
            f32::from_bits(1.0_f32.to_bits() - 1),
        ] {
            let source = image(
                &(0..=255)
                    .map(|c| [c, 255 - c, c / 2, 255])
                    .collect::<Vec<_>>(),
            );
            for b in [0, 17, 84, 132, 254, 255] {
                let mut background = image(&vec![[b, 255 - b, b, 255]; 256]);
                let alpha = vec![a; 256];
                let f = pair(&source, &alpha, &mut background).unwrap();
                let expected: Vec<_> = source
                    .pixels
                    .chunks_exact(4)
                    .flat_map(|p| p[..3].iter().copied())
                    .collect();
                assert_eq!(
                    reconstruct(&f, &background, &alpha),
                    expected,
                    "alpha={a}, B={b}"
                );
            }
        }
    }

    #[test]
    fn known_alpha_endpoints_keep_the_observed_material_exact() {
        let source = image(&[[91, 47, 19, 255], [230, 160, 80, 255]]);
        let mut background = image(&[[132, 105, 45, 255], [7, 8, 9, 255]]);
        let f = pair(&source, &[0., 1.], &mut background).unwrap();
        assert_eq!(&background.pixels[..4], &source.pixels[..4]);
        assert_eq!(&f.pixels[4..], &source.pixels[4..]);
        assert_eq!(&background.pixels[4..], &[7, 8, 9, 255]);
    }

    #[test]
    fn tiny_transport_is_continuous_even_when_the_plate_changes_across_zero() {
        let source = image(&[[100, 80, 60, 255], [91, 47, 19, 255], [110, 80, 50, 255]]);
        let alpha = [0.2_f32, 0.569151, 0.8];
        for shift in [-0.01_f32, -0.001, 0.0, 0.001, 0.01] {
            // Opposite signs deliberately select very different valid plate
            // priors, as happens when the revealed-hole set changes.
            let prior = if shift < 0.0 {
                [84, 37, 12, 255]
            } else {
                [132, 105, 45, 255]
            };
            let mut b = image(&[prior; 3]);
            let f = pair(&source, &alpha, &mut b).unwrap();
            let t = shift.abs();
            let (pixel, membership) = super::super::sample(&f, &alpha, 1.0 + shift, 0.0);
            let bg_weight = 1.0 - membership;
            let final_alpha = pixel[3] + bg_weight;
            for c in 0..3 {
                let observed =
                    ((pixel[c] + bg_weight * b.pixels[4 + c] as f32) / final_alpha).round();
                let error = (observed - source.pixels[4 + c] as f32).abs();
                assert!(
                    error <= 1.0 + 510.0 * t,
                    "shift={shift}, channel={c}, error={error}"
                );
                if shift == 0.0 {
                    assert_eq!(observed, source.pixels[4 + c] as f32);
                }
            }
        }
    }

    #[test]
    fn unsupported_alpha_or_transparency_does_not_mutate_the_plate() {
        let source = image(&[[91, 47, 19, 200]]);
        let original = image(&[[132, 105, 45, 255]]);
        let mut b = original.clone();
        assert!(pair(&source, &[0.5], &mut b).is_none());
        assert_eq!(b.pixels, original.pixels);
        let opaque = image(&[[91, 47, 19, 255]]);
        let mut transparent_background = original.clone();
        transparent_background.pixels[3] = 200;
        let unchanged = transparent_background.clone();
        assert!(pair(&opaque, &[0.5], &mut transparent_background).is_none());
        assert_eq!(transparent_background.pixels, unchanged.pixels);
        for a in [f32::NAN, -0.1, 1.1] {
            assert!(pair(&opaque, &[a], &mut b).is_none());
            assert_eq!(b.pixels, original.pixels);
        }
    }
}
