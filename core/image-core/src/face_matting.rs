//! Germer et al., "Fast Multi-Level Foreground Estimation", arXiv:2006.14970.
//! Rust translation of the multi-scale 2×2 F/B updates in PyMatting's official
//! estimate_foreground_ml (nearest resize, 10 small / 2 large iterations,
//! regularization=1e-5, gradient_weight=1). The paper's estimator does not
//! estimate alpha. The separate local color-line refinement below is a
//! project adaptation and only applies to opaque composite photographs.

use crate::{engine::byte, RgbaImage};
use std::collections::VecDeque;

/// Semantic confidence is not physical opacity. In the uncertain, local
/// transition, project RGB onto the line between nearby *observed* opaque
/// foreground/background colors. This is a constrained color-line refinement,
/// not a learned matting model. Ambiguous colors and distant donors retain the
/// supplied matte; no unknown face or background color is invented.
pub(crate) fn refine_alpha(source: &RgbaImage, alpha: &[f32]) -> Vec<f32> {
    refine_alpha_with_border(source, alpha, 0)
}

/// Portrait class confidence can saturate before the physical RGB edge: the
/// last skin/hair mixed texel can have confidence 1.0. Reserve two pixels on
/// either side as a color-line trimap, while the eroded interior remains exact.
/// Other callers retain the confidence-only refinement above.
pub(crate) fn refine_portrait_alpha(source: &RgbaImage, alpha: &[f32]) -> Vec<f32> {
    refine_alpha_with_border(source, alpha, 2)
}

fn refine_alpha_with_border(source: &RgbaImage, alpha: &[f32], border: usize) -> Vec<f32> {
    if source.pixels.chunks_exact(4).any(|p| p[3] != 255) {
        return alpha.to_vec();
    }
    let (w, h) = (source.width as usize, source.height as usize);
    let count = w * h;
    let neighbors = |i: usize| {
        let (x, y) = (i % w, i / w);
        [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
    };
    let donors = |foreground: bool| {
        let mut nearest = vec![usize::MAX; count];
        let mut distance = vec![u8::MAX; count];
        let mut queue = VecDeque::new();
        for (i, &a) in alpha.iter().enumerate() {
            let certain = if foreground { a >= 0.98 } else { a <= 0.02 };
            let (x, y) = (i % w, i / w);
            let eroded = certain
                && (border == 0
                    || (y.saturating_sub(border)..=(y + border).min(h - 1)).all(|yy| {
                        (x.saturating_sub(border)..=(x + border).min(w - 1)).all(|xx| {
                            let a = alpha[yy * w + xx];
                            if foreground {
                                a >= 0.98
                            } else {
                                a <= 0.02
                            }
                        })
                    }));
            if eroded {
                nearest[i] = i;
                distance[i] = 0;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            if distance[i] >= 12 {
                continue;
            }
            for n in neighbors(i).into_iter().flatten() {
                if distance[n] > distance[i] + 1 {
                    nearest[n] = nearest[i];
                    distance[n] = distance[i] + 1;
                    queue.push_back(n);
                }
            }
        }
        (nearest, distance)
    };
    let (foreground, foreground_distance) = donors(true);
    let (background, background_distance) = donors(false);
    let observed = |i: usize, foreground: bool| {
        let (x, y) = (i % w, i / w);
        let mut mean = [0.0_f32; 3];
        let mut n = 0;
        for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let j = yy * w + xx;
                if if foreground {
                    alpha[j] >= 0.98
                } else {
                    alpha[j] <= 0.02
                } {
                    for (c, value) in mean.iter_mut().enumerate() {
                        *value += source.pixels[j * 4 + c] as f32;
                    }
                    n += 1;
                }
            }
        }
        mean.map(|v| v / n.max(1) as f32)
    };
    let mut refined = alpha.to_vec();
    for (i, a) in refined.iter_mut().enumerate() {
        let confident = *a <= 0.02 || *a >= 0.98;
        let in_trimapped_border = border > 0
            && if *a >= 0.98 {
                foreground_distance[i] > 0 && background_distance[i] <= (border * 2 + 1) as u8
            } else {
                background_distance[i] > 0 && foreground_distance[i] <= (border * 2 + 1) as u8
            };
        if (confident && !in_trimapped_border)
            || foreground[i] == usize::MAX
            || background[i] == usize::MAX
        {
            continue;
        }
        let f = observed(foreground[i], true);
        let b = observed(background[i], false);
        let delta = std::array::from_fn::<_, 3, _>(|c| f[c] - b[c]);
        let denominator: f32 = delta.iter().map(|v| v * v).sum();
        if denominator < 25.0 * 25.0 {
            continue;
        }
        let candidate = (delta
            .iter()
            .enumerate()
            .map(|(c, d)| (source.pixels[i * 4 + c] as f32 - b[c]) * d)
            .sum::<f32>()
            / denominator)
            .clamp(0.0, 1.0);
        let residual: f32 = (0..3)
            .map(|c| (source.pixels[i * 4 + c] as f32 - (b[c] + candidate * delta[c])).powi(2))
            .sum();
        if residual <= 3.0 * 12.0 * 12.0 {
            *a = candidate;
        }
    }
    refined
}

/// Recover only a corroborated, connected portrait trimap omission. Class
/// zero is not sufficient physical-background evidence if the selected-person
/// observation and local RGB line both support the adjacent connected skin.
/// No semantic label, geometry, or frozen output pixel is changed here.
pub(crate) fn recover_connected_portrait_fringe(
    source: &RgbaImage,
    alpha: &[f32],
    owned: &[bool],
    person: &[f32],
    semantic_edge: &[f32],
    forbidden: &[bool],
    fringe: usize,
) -> Vec<f32> {
    if source.pixels.chunks_exact(4).any(|p| p[3] != 255) {
        return alpha.to_vec();
    }
    let (w, h) = (source.width as usize, source.height as usize);
    let count = w * h;
    let neighbors = |i: usize| {
        let (x, y) = (i % w, i / w);
        [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
    };
    let distances = |seeds: &[bool], limit: usize| {
        let mut nearest = vec![usize::MAX; count];
        let mut dist = vec![usize::MAX; count];
        let mut q = VecDeque::new();
        for (i, &yes) in seeds.iter().enumerate() {
            if yes {
                nearest[i] = i;
                dist[i] = 0;
                q.push_back(i);
            }
        }
        while let Some(i) = q.pop_front() {
            if dist[i] >= limit {
                continue;
            }
            for n in neighbors(i).into_iter().flatten() {
                if dist[n] > dist[i] + 1 {
                    dist[n] = dist[i] + 1;
                    nearest[n] = nearest[i];
                    q.push_back(n);
                }
            }
        }
        (nearest, dist)
    };
    let certain = |i: usize, fg: bool| {
        !forbidden[i]
            && if fg {
                owned[i] && alpha[i] >= 0.98
            } else {
                alpha[i] <= 0.02 && person[i] <= 1.0 / 255.0
            }
    };
    let eroded = |fg: bool| {
        (0..count)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                certain(i, fg)
                    && (y.saturating_sub(1)..=(y + 1).min(h - 1)).all(|yy| {
                        (x.saturating_sub(1)..=(x + 1).min(w - 1))
                            .all(|xx| certain(yy * w + xx, fg))
                    })
            })
            .collect::<Vec<bool>>()
    };
    let (_, distance) = distances(owned, fringe);
    let (foreground, fd) = distances(&eroded(true), 12);
    let (background, bd) = distances(&eroded(false), 12);
    let observed = |i: usize, fg: bool| {
        let (x, y) = (i % w, i / w);
        let mut sum = [0.0f32; 3];
        let mut n = 0.0f32;
        for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let j = yy * w + xx;
                if certain(j, fg) {
                    for (c, value) in sum.iter_mut().enumerate() {
                        *value += source.pixels[j * 4 + c] as f32;
                    }
                    n += 1.0;
                }
            }
        }
        sum.map(|v| v / n.max(1.0))
    };
    let mut result = alpha.to_vec();
    let mut proposals = vec![0.0_f32; count];
    for i in 0..count {
        if alpha[i] > 1e-6
            || forbidden[i]
            || person[i] <= 0.15
            || distance[i] > fringe
            || fd[i] > 12
            || bd[i] > 12
        {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let near_semantic = (y.saturating_sub(2)..=(y + 2).min(h - 1)).any(|yy| {
            (x.saturating_sub(2)..=(x + 2).min(w - 1)).any(|xx| semantic_edge[yy * w + xx] > 0.02)
        });
        if !near_semantic {
            continue;
        }
        let f = observed(foreground[i], true);
        let b = observed(background[i], false);
        let delta = [f[0] - b[0], f[1] - b[1], f[2] - b[2]];
        let denominator = delta.iter().map(|x| x * x).sum::<f32>();
        if denominator < 25.0 * 25.0 {
            continue;
        }
        let candidate = ((0..3)
            .map(|c| (source.pixels[i * 4 + c] as f32 - b[c]) * delta[c])
            .sum::<f32>()
            / denominator)
            .clamp(0.0, 1.0);
        let residual = (0..3)
            .map(|c| (source.pixels[i * 4 + c] as f32 - (b[c] + candidate * delta[c])).powi(2))
            .sum::<f32>();
        proposals[i] = candidate;
        if residual <= 3.0 * 12.0 * 12.0 && candidate > 1.0 / 255.0 {
            result[i] = candidate;
        }
    }
    // A local color-line threshold can reject a few samples in an otherwise
    // continuous physical edge because neighboring skin has a small lighting
    // or chromaticity gradient. Propagate only an already supported alpha to
    // immediately connected, RGB-compatible proposals. Every proposal still
    // passes the same person, ownership-distance, semantic, donor and explicit
    // protection gates above; this never dilates an arbitrary semantic mask.
    // Use simultaneous rounds so raster scan order cannot change the result.
    for _ in 0..fringe {
        let previous = result.clone();
        let mut added = false;
        for i in 0..count {
            if result[i] > 1e-6 || proposals[i] < 0.5 {
                continue;
            }
            if neighbors(i).into_iter().flatten().any(|j| {
                if previous[j] < 0.5 || (previous[j] - proposals[i]).abs() > 0.15 || forbidden[j] {
                    return false;
                }
                let color_distance = (0..3)
                    .map(|c| {
                        (source.pixels[i * 4 + c] as f32 - source.pixels[j * 4 + c] as f32).powi(2)
                    })
                    .sum::<f32>();
                color_distance <= 3.0 * 8.0 * 8.0
            }) {
                result[i] = proposals[i];
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    result
}

fn resize<T: Copy>(input: &[T], iw: usize, ih: usize, w: usize, h: usize) -> Vec<T> {
    (0..w * h)
        .map(|i| input[(i / w * ih / h).min(ih - 1) * iw + (i % w * iw / w).min(iw - 1)])
        .collect()
}

pub(crate) fn estimate(source: &RgbaImage, alpha: &[f32]) -> Option<(RgbaImage, RgbaImage)> {
    if source.pixels.chunks_exact(4).any(|p| p[3] != 255) {
        return None;
    }
    let (iw, ih) = (source.width as usize, source.height as usize);
    let image: Vec<[f32; 3]> = source
        .pixels
        .chunks_exact(4)
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    let (mut fm, mut bm) = ([0.0; 3], [0.0; 3]);
    let (mut fc, mut bc) = (0, 0);
    for (&a, p) in alpha.iter().zip(&image) {
        if a > 0.9 {
            for c in 0..3 {
                fm[c] += p[c];
            }
            fc += 1;
        }
        if a < 0.1 {
            for c in 0..3 {
                bm[c] += p[c];
            }
            bc += 1;
        }
    }
    if fc == 0 || bc == 0 {
        return None;
    }
    for c in 0..3 {
        fm[c] /= fc as f32 + 1e-5;
        bm[c] /= bc as f32 + 1e-5;
    }
    let (mut f, mut b) = (vec![fm], vec![bm]);
    let (mut pw, mut ph) = (1, 1);
    let levels = (iw.max(ih) as f32).log2().ceil().max(1.0) as usize;
    for level in 0..=levels {
        let fraction = level as f32 / levels as f32;
        let w = (iw as f32).powf(fraction).round().max(1.0) as usize;
        let h = (ih as f32).powf(fraction).round().max(1.0) as usize;
        let colors = resize(&image, iw, ih, w, h);
        let alphas = resize(alpha, iw, ih, w, h);
        f = resize(&f, pw, ph, w, h);
        b = resize(&b, pw, ph, w, h);
        let iterations = if w <= 32 && h <= 32 { 10 } else { 2 };
        for _ in 0..iterations {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let a = alphas[i];
                    let inverse = 1.0 - a;
                    let (mut aa, ab, mut bb) = (a * a, a * inverse, inverse * inverse);
                    let mut rhs_f = colors[i].map(|v| v * a);
                    let mut rhs_b = colors[i].map(|v| v * inverse);
                    for j in [
                        y * w + x.saturating_sub(1),
                        y * w + (x + 1).min(w - 1),
                        y.saturating_sub(1) * w + x,
                        (y + 1).min(h - 1) * w + x,
                    ] {
                        let weight = 1e-5 + (a - alphas[j]).abs();
                        aa += weight;
                        bb += weight;
                        for c in 0..3 {
                            rhs_f[c] += weight * f[j][c];
                            rhs_b[c] += weight * b[j][c];
                        }
                    }
                    let determinant = (aa * bb - ab * ab).max(1e-15);
                    for c in 0..3 {
                        f[i][c] = ((bb * rhs_f[c] - ab * rhs_b[c]) / determinant).clamp(0.0, 1.0);
                        b[i][c] = ((aa * rhs_b[c] - ab * rhs_f[c]) / determinant).clamp(0.0, 1.0);
                    }
                }
            }
        }
        pw = w;
        ph = h;
    }
    let convert = |colors: Vec<[f32; 3]>| RgbaImage {
        width: source.width,
        height: source.height,
        pixels: colors
            .iter()
            .flat_map(|c| {
                [
                    byte(c[0] * 255.0),
                    byte(c[1] * 255.0),
                    byte(c[2] * 255.0),
                    255,
                ]
            })
            .collect(),
    };
    Some((convert(f), convert(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovered_known_alpha_edge_transports_to_background_without_skin_islands() {
        // Independent physical ground truth: a three-pixel parser omission at
        // an antialiased skin edge. A small chromaticity shift crosses the
        // single-pixel color-line cutoff, but the actual skin is still opaque.
        let (w, h) = (48usize, 28usize);
        let b = [20.0_f32, 30.0, 40.0];
        let mut pixels = vec![0_u8; w * h * 4];
        let mut truth = vec![0.0_f32; w * h];
        let mut alpha = truth.clone();
        let mut person = truth.clone();
        let mut owned = vec![false; w * h];
        let mut forbidden = owned.clone();
        let mut fg = vec![[190.0_f32, 130.0, 100.0]; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                truth[i] = if x >= 12 {
                    1.0
                } else if x == 11 {
                    0.3
                } else {
                    0.0
                };
                person[i] = truth[i];
                if x >= 15 {
                    alpha[i] = 1.0;
                    owned[i] = true;
                }
                if (12..=14).contains(&x) && (8..=14).contains(&y) {
                    let shift = if y == 8 || y == 14 { 12.0 } else { 16.0 };
                    fg[i][1] += shift;
                    fg[i][2] += shift;
                }
                for c in 0..3 {
                    pixels[i * 4 + c] =
                        (b[c] * (1.0 - truth[i]) + fg[i][c] * truth[i]).round() as u8;
                }
                pixels[i * 4 + 3] = 255;
            }
        }
        // A different person's nearby opaque skin is explicitly excluded.
        // It is only two pixels from the target's physical edge.
        for y in 20..24 {
            let i = y * w + 10;
            for c in 0..3 {
                pixels[i * 4 + c] = fg[i][c] as u8;
            }
            person[i] = 1.0;
            forbidden[i] = true;
        }
        let image = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let recovered = recover_connected_portrait_fringe(
            &image,
            &alpha,
            &owned,
            &person,
            &vec![1.0; w * h],
            &forbidden,
            5,
        );
        for y in 2..h - 2 {
            for x in 10..18 {
                let i = y * w + x;
                assert!(
                    (recovered[i] - truth[i]).abs() < 0.01,
                    "physical opacity at {x},{y}: {} != {}",
                    recovered[i],
                    truth[i]
                );
            }
        }
        for i in 0..w * h {
            if forbidden[i] || person[i] == 0.0 {
                assert_eq!(recovered[i], alpha[i]);
            }
        }
        // For both contraction and expansion, compare transported physical
        // opacity against the analytic translated half-plane, rather than
        // treating darker newly revealed background as a failure.
        for displacement in [-6_isize, 6] {
            for y in 2..h - 2 {
                for x in 7..w - 7 {
                    let sx = (x as isize - displacement) as usize;
                    let si = y * w + sx;
                    let expected_alpha = if sx >= 12 {
                        1.0
                    } else if sx == 11 {
                        0.3
                    } else {
                        0.0
                    };
                    for c in 0..3 {
                        let actual = b[c] * (1.0 - recovered[si]) + fg[si][c] * recovered[si];
                        let expected = b[c] * (1.0 - expected_alpha) + fg[si][c] * expected_alpha;
                        assert!((actual-expected).abs()<1.0,"transport {displacement} at {x},{y} channel {c}: {actual} != {expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn connected_fringe_uses_observed_rgb_and_rejects_fixed_or_separate_people() {
        let (w, h) = (48usize, 24usize);
        let mut pixels = [20u8, 30, 40, 255].repeat(w * h);
        let mut alpha = vec![0.0; w * h];
        let mut person = alpha.clone();
        let mut owned = vec![false; w * h];
        let mut forbidden = owned.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x >= 15 {
                    alpha[i] = 1.0;
                    owned[i] = true;
                }
                if x >= 12 {
                    person[i] = 1.0;
                    pixels[i * 4..i * 4 + 4].copy_from_slice(&[190, 130, 100, 255]);
                }
                if x == 11 {
                    person[i] = 0.3;
                    pixels[i * 4..i * 4 + 4].copy_from_slice(&[71, 60, 58, 255]);
                }
                // Disconnected same-colored person and explicit fixed texel.
                if x == 4 {
                    person[i] = 1.0;
                    pixels[i * 4..i * 4 + 4].copy_from_slice(&[190, 130, 100, 255]);
                }
            }
        }
        forbidden[12 * w + 12] = true;
        let image = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let result = recover_connected_portrait_fringe(
            &image,
            &alpha,
            &owned,
            &person,
            &vec![1.0; w * h],
            &forbidden,
            5,
        );
        assert!(result[10 * w + 12] > 0.98);
        assert!((result[10 * w + 11] - 0.3).abs() < 0.02);
        assert_eq!(result[12 * w + 12], 0.0);
        assert_eq!(result[10 * w + 4], 0.0);
        for i in 0..w * h {
            if person[i] <= 1.0 / 255.0 || alpha[i] > 0.0 {
                assert_eq!(result[i], alpha[i]);
            }
        }
        let mut transparent = image.clone();
        transparent.pixels[3] = 128;
        assert_eq!(
            recover_connected_portrait_fringe(
                &transparent,
                &alpha,
                &owned,
                &person,
                &vec![1.0; w * h],
                &forbidden,
                5
            ),
            alpha
        );
    }
    #[test]
    fn connected_fringe_refuses_rgb_inconsistent_with_skin_background_line() {
        let (w, h) = (32usize, 20usize);
        let mut pixels = [20u8, 30, 40, 255].repeat(w * h);
        let mut alpha = vec![0.0; w * h];
        let mut person = alpha.clone();
        let mut owned = vec![false; w * h];
        for y in 0..h {
            for x in 12..w {
                let i = y * w + x;
                person[i] = 1.0;
                pixels[i * 4..i * 4 + 4].copy_from_slice(if x >= 15 {
                    &[190, 130, 100, 255]
                } else {
                    &[20, 240, 30, 255]
                });
                if x >= 15 {
                    alpha[i] = 1.0;
                    owned[i] = true;
                }
            }
        }
        let image = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let result = recover_connected_portrait_fringe(
            &image,
            &alpha,
            &owned,
            &person,
            &vec![1.0; w * h],
            &vec![false; w * h],
            5,
        );
        assert_eq!(result[10 * w + 13], 0.0);
    }
    #[test]
    fn portrait_trimap_corrects_saturated_mixed_edge_and_keeps_certain_interior() {
        let (w, h) = (64, 16);
        // A semantic label is binary despite a real two-pixel skin/hair
        // antialias fringe. Confidence-only refinement cannot fix this case.
        let truth: Vec<f32> = (0..w * h)
            .map(|i| (((i % w) as f32 - 30.0) / 3.0).clamp(0.0, 1.0))
            .collect();
        let confidence: Vec<f32> = truth
            .iter()
            .map(|&a| if a >= 0.5 { 1.0 } else { 0.0 })
            .collect();
        let foreground = [224.0, 165.0, 127.0];
        let background = [25.0, 18.0, 12.0];
        let source = RgbaImage::new(
            w as u32,
            h as u32,
            truth
                .iter()
                .flat_map(|&a| {
                    [
                        byte(foreground[0] * a + background[0] * (1.0 - a)),
                        byte(foreground[1] * a + background[1] * (1.0 - a)),
                        byte(foreground[2] * a + background[2] * (1.0 - a)),
                        255,
                    ]
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(refine_alpha(&source, &confidence), confidence);
        let refined = refine_portrait_alpha(&source, &confidence);
        for (i, (&a, &expected)) in refined.iter().zip(&truth).enumerate() {
            assert!(
                (a - expected).abs() < 0.008,
                "mixed physical edge at {i}: {a} != {expected}"
            );
            if i % w < 29 || i % w > 34 {
                assert_eq!(a, confidence[i], "certain interior at {i}");
            }
        }
    }
    #[test]
    fn local_color_line_corrects_class_confidence_without_erasing_the_physical_soft_edge() {
        let (w, h) = (64, 24);
        let truth: Vec<f32> = (0..w * h)
            .map(|i| (((i % w) as f32 - 30.0) / 4.0).clamp(0.0, 1.0))
            .collect();
        let confidence: Vec<f32> = (0..w * h)
            .map(|i| (((i % w) as f32 - 26.0) / 12.0).clamp(0.0, 1.0))
            .collect();
        let foreground = [220.0, 60.0, 30.0];
        let background = [20.0, 110.0, 230.0];
        let source = RgbaImage::new(
            w as u32,
            h as u32,
            truth
                .iter()
                .flat_map(|&a| {
                    [
                        byte(foreground[0] * a + background[0] * (1.0 - a)),
                        byte(foreground[1] * a + background[1] * (1.0 - a)),
                        byte(foreground[2] * a + background[2] * (1.0 - a)),
                        255,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let refined = refine_alpha(&source, &confidence);
        let before: f32 = truth
            .iter()
            .zip(&confidence)
            .map(|(a, b)| (a - b).abs())
            .sum();
        let after: f32 = truth.iter().zip(&refined).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            after < before * 0.02,
            "semantic opacity error {before} -> {after}"
        );
        assert!((refined[32] - 0.5).abs() < 0.01);
        assert!((refined[31] - 0.25).abs() < 0.01);
    }
    #[test]
    fn ambiguous_identical_colors_and_rgba_do_not_invent_a_matting_boundary() {
        let alpha: Vec<f32> = (0..32).map(|i| i as f32 / 31.0).collect();
        let source = RgbaImage::new(
            32,
            1,
            vec![[100, 100, 100, 255]; 32]
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();
        assert_eq!(refine_alpha(&source, &alpha), alpha);
        let source = RgbaImage::new(
            32,
            1,
            vec![[100, 100, 100, 128]; 32]
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();
        assert_eq!(refine_alpha(&source, &alpha), alpha);
    }
    #[test]
    fn recovers_foreground_color_instead_of_warping_its_mixed_background_halo() {
        let (w, h) = (64, 32);
        let foreground = [220.0, 60.0, 30.0];
        let background = [20.0, 110.0, 230.0];
        let alpha: Vec<f32> = (0..w * h)
            .map(|i| ((i % w) as f32 - 20.0) / 22.0)
            .map(|a| a.clamp(0.0, 1.0))
            .collect();
        let pixels: Vec<u8> = alpha
            .iter()
            .flat_map(|&a| {
                [
                    byte(foreground[0] * a + background[0] * (1.0 - a)),
                    byte(foreground[1] * a + background[1] * (1.0 - a)),
                    byte(foreground[2] * a + background[2] * (1.0 - a)),
                    255,
                ]
            })
            .collect();
        let source = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let (f, b) = estimate(&source, &alpha).unwrap();
        let mut raw_error = 0.0;
        let mut f_error = 0.0;
        let mut count = 0;
        for (i, &a) in alpha.iter().enumerate() {
            if (0.2..0.8).contains(&a) {
                for (c, &expected) in foreground.iter().enumerate() {
                    raw_error += (source.pixels[i * 4 + c] as f32 - expected).abs();
                    f_error += (f.pixels[i * 4 + c] as f32 - expected).abs();
                    assert!((b.pixels[i * 4 + c] as f32 - background[c]).abs() < 12.0);
                }
                count += 3;
            }
        }
        assert!(
            f_error / (count as f32) < 8.0,
            "mean foreground error {}",
            f_error / count as f32
        );
        assert!(
            f_error < raw_error * 0.12,
            "foreground contamination {raw_error} -> {f_error}"
        );
    }
    #[test]
    fn transparent_rgba_is_not_mistaken_for_an_opaque_composite_photo() {
        let source = RgbaImage::new(2, 1, vec![220, 60, 30, 128, 20, 110, 230, 0]).unwrap();
        assert!(estimate(&source, &[1.0, 0.0]).is_none());
        let source = RgbaImage::new(2, 1, vec![220, 60, 30, 255, 20, 110, 230, 255]).unwrap();
        assert!(estimate(&source, &[1.0, 1.0]).is_none());
    }
}
