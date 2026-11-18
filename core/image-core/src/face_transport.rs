//! Physical-layer transport in the project's existing sRGB byte space.
//! RGB is 0..255; opacity/coverage are 0..1. This is not a linear-light solver.
//! Input RGB must be straight, decomposed foreground, not a flattened photograph.
//! Coverage is physical coverage, not a parser's class probability. If bitmap
//! alpha already represents that coverage, pass an all-one coverage map.

use crate::RgbaImage;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LayerSample {
    pub premul: [f32; 3],
    pub opacity: f32,
    pub coverage: f32,
}

pub(crate) fn sample(
    foreground: &RgbaImage,
    coverage: &[f32],
    source_x: f32,
    source_y: f32,
) -> LayerSample {
    let width = foreground.width as usize;
    let height = foreground.height as usize;
    assert_eq!(foreground.pixels.len(), width * height * 4);
    assert_eq!(coverage.len(), width * height);
    assert!(source_x.is_finite() && source_y.is_finite());
    // Integer coordinates denote texel centres. Outside the supplied layer
    // has zero opacity; clamping would extend the crop-edge foreground.
    if source_x <= -1.0 || source_y <= -1.0 || source_x >= width as f32 || source_y >= height as f32
    {
        return LayerSample::default();
    }
    let (x0, y0) = (source_x.floor() as i64, source_y.floor() as i64);
    let (tx, ty) = (source_x - x0 as f32, source_y - y0 as f32);
    let mut result = LayerSample::default();
    for (x, y, weight) in [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ] {
        if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
            continue;
        }
        let i = y as usize * width + x as usize;
        let physical = coverage[i].clamp(0.0, 1.0);
        let opacity = physical * foreground.pixels[i * 4 + 3] as f32 / 255.0;
        for c in 0..3 {
            result.premul[c] += foreground.pixels[i * 4 + c] as f32 * opacity * weight;
        }
        result.opacity += opacity * weight;
        result.coverage += physical * weight;
    }
    result
}

/// Old visible foreground must be replaced by its actual lower layer even
/// when the transported foreground still covers part of this destination.
pub(crate) fn needs_underlay(old_opacity: f32, new_opacity: f32) -> bool {
    old_opacity > 0.0 && new_opacity < 1.0
}

/// A transported foreground sample may only create new foreground ownership
/// inside the destination silhouette.  Without this gate, an inverse field
/// can pull a face texel into a background destination whose old opacity was
/// zero; compositing then paints the face over the scene and makes the
/// background appear to move with the edit.  Existing foreground may shrink
/// outside the target and reveal the lower layer, so the old opacity remains
/// an explicit part of the decision.
pub(crate) fn allows_new_front_ownership(
    _old_opacity: f32,
    new_opacity: f32,
    destination_coverage: f32,
) -> bool {
    let new = new_opacity.clamp(0.0, 1.0);
    let target = destination_coverage.clamp(0.0, 1.0);
    if new <= 1e-6 {
        return true;
    }
    // New ownership is valid only where the moved silhouette says foreground
    // is allowed.  A half-covered edge is accepted to retain antialiasing.
    // Callers that had old ownership outside this contour must clear that
    // ownership and resolve the resulting hole from the lower layer.
    target >= 0.5
}

/// `underlay` is true lower-layer RGBA, never the old flattened destination.
/// Unknown required underlay stays unresolved instead of copying old skin.
pub(crate) fn over(front: LayerSample, underlay: Option<[u8; 4]>) -> Option<[u8; 4]> {
    let fa = front.opacity.clamp(0.0, 1.0);
    let bg = if fa == 1.0 { [0; 4] } else { underlay? };
    let ba = bg[3] as f32 / 255.0;
    let opacity = fa + (1.0 - fa) * ba;
    let mut out = [0; 4];
    if opacity > 0.0 {
        for c in 0..3 {
            out[c] = ((front.premul[c] + bg[c] as f32 * ba * (1.0 - fa)) / opacity)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    out[3] = (opacity * 255.0).round().clamp(0.0, 255.0) as u8;
    Some(out)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum UnderlayError {
    InvalidInput,
    Unresolved { pixels: usize, first_index: usize },
}

/// Bounded deterministic propagation from caller-certified stationary seeds.
/// `holes` includes partial disocclusions. No semantic neck exception exists.
/// Unobserved lower-layer colour can extend behind opaque foreground. Its
/// foreground RGB never becomes a donor; all non-hole output bytes stay exact.
/// This extends observed colours, not hidden texture: it is not neural inpaint.
/// The caller must supply seeds belonging to this actual lower layer. These
/// booleans cannot themselves distinguish backdrop from stationary hair.
pub(crate) fn reconstruct_underlay(
    source: &RgbaImage,
    holes: &[bool],
    moving_alpha: &[f32],
    clean_stationary_seed: &[bool],
    max_distance: u32,
) -> Result<RgbaImage, UnderlayError> {
    let (w, h) = (source.width as usize, source.height as usize);
    let count = w.checked_mul(h).ok_or(UnderlayError::InvalidInput)?;
    if count == 0
        || count.checked_mul(4) != Some(source.pixels.len())
        || holes.len() != count
        || moving_alpha.len() != count
        || clean_stationary_seed.len() != count
        || moving_alpha
            .iter()
            .any(|a| !a.is_finite() || !(0.0..=1.0).contains(a))
    {
        return Err(UnderlayError::InvalidInput);
    }
    if !holes.iter().any(|&hole| hole) {
        return Ok(source.clone());
    }
    let mut distance = vec![usize::MAX; count];
    let mut colors = vec![[0.0_f32; 4]; count];
    let mut queue = Vec::new();
    for i in 0..count {
        // Exactly zero moving coverage is required. Even a fractional skin
        // fringe can contaminate all descendants of a propagation seed.
        if holes[i] || !clean_stationary_seed[i] || moving_alpha[i] != 0.0 {
            continue;
        }
        distance[i] = 0;
        let rgba = &source.pixels[i * 4..i * 4 + 4];
        let a = rgba[3] as f32 / 255.0;
        colors[i] = [
            rgba[0] as f32 * a,
            rgba[1] as f32 * a,
            rgba[2] as f32 * a,
            a,
        ];
        queue.push(i);
    }
    let mut head = 0;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        if distance[i] >= max_distance as usize {
            continue;
        }
        for n in neighbors(i, w, h) {
            if distance[n] != usize::MAX {
                continue;
            }
            distance[n] = distance[i] + 1;
            queue.push(n);
        }
    }
    let mut missing = (0..count).filter(|&i| holes[i] && distance[i] == usize::MAX);
    if let Some(first_index) = missing.next() {
        return Err(UnderlayError::Unresolved {
            pixels: 1 + missing.count(),
            first_index,
        });
    }
    // Average only earlier wavefront values, never old RGB inside the hole.
    // A separate distance pass makes ties independent of neighbour order.
    for &i in &queue {
        if distance[i] == 0 {
            continue;
        }
        let mut sum = [0.0; 4];
        let mut donors = 0;
        for n in neighbors(i, w, h).filter(|&n| distance[n] == distance[i] - 1) {
            for c in 0..4 {
                sum[c] += colors[n][c];
            }
            donors += 1;
        }
        debug_assert!(donors > 0);
        for c in 0..4 {
            colors[i][c] = sum[c] / donors as f32;
        }
    }
    let mut result = source.clone();
    for i in (0..count).filter(|&i| holes[i]) {
        let a = colors[i][3];
        for c in 0..3 {
            result.pixels[i * 4 + c] = if a > 0.0 {
                (colors[i][c] / a).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
        }
        result.pixels[i * 4 + 3] = (a * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    Ok(result)
}

fn neighbors(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % w, i / w);
    [
        (x > 0).then(|| i - 1),
        (x + 1 < w).then(|| i + 1),
        (y > 0).then(|| i - w),
        (y + 1 < h).then(|| i + w),
    ]
    .into_iter()
    .flatten()
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

    #[test]
    fn subpixel_transport_premultiplies_before_filtering() {
        let source = image(&[[200, 80, 40, 255], [0, 255, 0, 0]]);
        let front = sample(&source, &[1.0, 1.0], 0.5, 0.0);
        assert_eq!(front.premul, [100.0, 40.0, 20.0]);
        assert_eq!(front.opacity, 0.5);
        assert_eq!(front.coverage, 1.0);
    }

    #[test]
    fn partial_disocclusion_requires_true_underlay() {
        let fg = image(&[[200, 80, 40, 255]; 2]);
        let front = sample(&fg, &[1.0, 0.0], 0.5, 0.0);
        assert!(needs_underlay(1.0, front.opacity));
        assert_eq!(over(front, None), None);
        assert_eq!(
            over(front, Some([20, 160, 200, 255])),
            Some([110, 120, 120, 255])
        );
        // Retaining the former face RGB as backdrop reproduces the old edge.
        assert_ne!(
            over(front, Some([200, 80, 40, 255])),
            Some([110, 120, 120, 255])
        );
    }

    #[test]
    fn moving_skin_is_not_a_donor_even_if_explicitly_marked_clean() {
        let bg = [20, 160, 200, 255];
        let source = image(&[bg, [200, 80, 40, 255], [200, 80, 40, 255], bg]);
        let out = reconstruct_underlay(
            &source,
            &[false, true, true, false],
            &[0.0, 1.0, 0.2, 0.0],
            &[true; 4],
            3,
        )
        .unwrap();
        assert_eq!(out, image(&[bg; 4]));
    }

    #[test]
    fn both_half_pixel_directions_match_independent_soft_edge_values() {
        let source = image(&[[200, 80, 40, 255]; 5]);
        let alpha = [1.0, 1.0, 0.5, 0.0, 0.0];
        for (x, expected) in [(1.5, [155, 100, 80, 255]), (2.5, [65, 140, 160, 255])] {
            assert_eq!(
                over(sample(&source, &alpha, x, 0.0), Some([20, 160, 200, 255])),
                Some(expected)
            );
        }
    }

    #[test]
    fn identity_reconstructs_known_foreground_and_lower_layer() {
        let fg = image(&[[200, 80, 40, 255]; 3]);
        let expected = [
            [200, 80, 40, 255],
            [110, 120, 120, 255],
            [20, 160, 200, 255],
        ];
        for (i, rgba) in expected.into_iter().enumerate() {
            assert_eq!(
                over(
                    sample(&fg, &[1.0, 0.5, 0.0], i as f32, 0.0),
                    Some([20, 160, 200, 255])
                ),
                Some(rgba)
            );
        }
        assert_eq!(
            over(sample(&fg, &[1.0; 3], 0.0, 0.0), None),
            Some(expected[0])
        );
    }

    #[test]
    fn flattened_rgb_is_not_a_valid_foreground_even_with_correct_underlay() {
        let flattened = image(&[[110, 120, 120, 255]]);
        let doubled = over(
            sample(&flattened, &[0.5], 0.0, 0.0),
            Some([20, 160, 200, 255]),
        );
        assert_eq!(doubled, Some([65, 140, 160, 255]));
        assert_ne!(doubled, Some([110, 120, 120, 255]));
    }

    #[test]
    fn coverage_and_intrinsic_opacity_are_distinct_and_applied_once() {
        let fg = image(&[[200, 80, 40, 128]]);
        let front = sample(&fg, &[0.5], 0.0, 0.0);
        assert_eq!(front.coverage, 0.5);
        assert!((front.opacity - 64.0 / 255.0).abs() < 1e-6);
        assert_eq!(over(front, Some([0, 255, 0, 0])), Some([200, 80, 40, 64]));
        assert_eq!(
            over(LayerSample::default(), Some([0, 255, 0, 0])),
            Some([0; 4])
        );
    }

    #[test]
    fn no_certified_donor_returns_explicit_failure() {
        let source = image(&[[20, 160, 200, 255], [200, 80, 40, 255], [200, 80, 40, 255]]);
        let holes = [false, false, true];
        for seeds in [[false; 3], [false, true, false]] {
            assert_eq!(
                reconstruct_underlay(&source, &holes, &[0.0, 1.0, 1.0], &seeds, 5),
                Err(UnderlayError::Unresolved {
                    pixels: 1,
                    first_index: 2
                })
            );
        }
    }

    #[test]
    fn hidden_underlay_extends_under_opaque_skin_without_using_its_colour() {
        let bg = [20, 160, 200, 255];
        let skin = [200, 80, 40, 255];
        let source = image(&[bg, skin, skin, skin]);
        let out = reconstruct_underlay(
            &source,
            &[false, false, false, true],
            &[0.0, 1.0, 1.0, 1.0],
            &[true, true, true, false],
            3,
        )
        .unwrap();
        assert_eq!(out, image(&[bg, skin, skin, bg]));
    }

    #[test]
    fn propagation_has_a_finite_band_and_preserves_every_non_hole_byte() {
        let source = image(&[[10, 20, 30, 128], [1, 2, 3, 4], [200, 80, 40, 255]]);
        let holes = [false, false, true];
        assert_eq!(
            reconstruct_underlay(&source, &holes, &[0.0, 0.0, 1.0], &[true, false, false], 1),
            Err(UnderlayError::Unresolved {
                pixels: 1,
                first_index: 2
            })
        );
        let out = reconstruct_underlay(&source, &holes, &[0.0, 0.0, 1.0], &[true, false, false], 2)
            .unwrap();
        assert_eq!(
            out,
            image(&[[10, 20, 30, 128], [1, 2, 3, 4], [10, 20, 30, 128]])
        );
        assert_eq!(source.pixels[8..], [200, 80, 40, 255]);
    }

    #[test]
    fn wavefront_averages_transparent_rgba_in_premultiplied_space() {
        let source = image(&[[200, 80, 40, 255], [255; 4], [0, 255, 0, 0]]);
        let out = reconstruct_underlay(
            &source,
            &[false, true, false],
            &[0.0, 1.0, 0.0],
            &[true, false, true],
            1,
        )
        .unwrap();
        assert_eq!(out.pixels[4..8], [200, 80, 40, 128]);
    }

    #[test]
    fn invalid_metadata_cannot_silently_admit_a_donor() {
        let source = image(&[[255; 4]]);
        assert_eq!(
            reconstruct_underlay(&source, &[true], &[f32::NAN], &[true], 4),
            Err(UnderlayError::InvalidInput)
        );
        assert_eq!(
            reconstruct_underlay(&source, &[], &[0.0], &[true], 4),
            Err(UnderlayError::InvalidInput)
        );
    }

    #[test]
    fn two_dimensional_sampling_and_crop_exterior_use_one_coverage_coordinate() {
        let fg = RgbaImage {
            width: 2,
            height: 2,
            pixels: vec![200, 80, 40, 255].repeat(4),
        };
        let alpha = [1.0, 0.0, 0.0, 1.0];
        let front = sample(&fg, &alpha, 0.25, 0.75);
        assert_eq!(front.premul, [75.0, 30.0, 15.0]);
        assert_eq!(front.opacity, 0.375);
        assert_eq!(front.coverage, 0.375);
        assert_eq!(sample(&fg, &alpha, 2.0, 0.0), LayerSample::default());
        assert_eq!(sample(&fg, &alpha, -0.5, 0.0).opacity, 0.5);
    }

    #[test]
    fn new_face_texel_cannot_paint_outside_destination_silhouette() {
        assert!(!allows_new_front_ownership(0.0, 0.8, 0.0));
        assert!(!allows_new_front_ownership(0.0, 0.8, 0.49));
        assert!(allows_new_front_ownership(0.0, 0.8, 0.5));
        assert!(allows_new_front_ownership(0.0, 0.8, 1.0));
    }

    #[test]
    fn old_foreground_outside_target_must_be_cleared_before_underlay() {
        // A contraction removes old ownership outside the requested target;
        // the caller then marks a hole and resolves the lower layer.
        assert!(!allows_new_front_ownership(0.8, 0.2, 0.0));
        assert!(allows_new_front_ownership(0.8, 0.0, 0.0));
    }
}
