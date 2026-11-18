//! Source-only evidence for the backdrop at a portrait's outer edge.
//!
//! A parser's zero is not an opacity measurement: the decoded RGB transition
//! can extend beyond it. Fit a local affine background only where an eroded,
//! observed scene patch supports that model, then derive the minimum opacity
//! needed to explain the source RGB with a physically bounded foreground.
//! No target contour, slider value, transported pixel or generated fill is
//! evidence here. Textured/occluded regions deliberately return no estimate.
//!
//! A wider decoded-tail band with monotone residual connectivity remains a
//! test-only candidate. It passes synthetic tails but produced fragments at a
//! real hair tip and retained a thin neck line; it is NOT used by prepare().
//! Moving scene donors farther away changes the background model even inside
//! the original band, while boolean path rejection can alternate underlayers.
//! These limitations require actual-photo ablation before any default change.

use crate::RgbaImage;
use nalgebra::{Matrix3, Vector3};
use std::collections::VecDeque;

pub(crate) struct SceneEdgeEvidence {
    /// Conservative source-RGB projection into the admitted scene envelope.
    /// This supplies the opacity lower bound, not a point estimate of B.
    /// None preserves the caller's existing underlayer path, never black.
    pub background: Vec<Option<[f32; 3]>>,
    /// Necessary source opacity for the estimate above, not a unique matte.
    /// Combine with existing opacity by max, never by replacing solid skin.
    pub minimum_alpha: Vec<f32>,
    /// A source-plane point estimate, or the robust center of compatible
    /// admitted contexts. Independent of the source-mixture projection above.
    pub model_center: Vec<Option<[f32; 3]>>,
    /// Empirical donor residual, including envelope spread for fallback models.
    /// This is not a bound on hidden-scene truth and is not a likelihood weight.
    pub model_error: Vec<Option<[f32; 3]>>,
}

impl SceneEdgeEvidence {
    /// Color reconstruction receives one source-model prior per accepted pixel.
    /// A later rejection (including experimental tail filtering) remains final.
    pub fn color_prior(&self) -> Vec<Option<[f32; 3]>> {
        self.background
            .iter()
            .enumerate()
            .map(|(i, projection)| projection.map(|bound| self.model_center[i].unwrap_or(bound)))
            .collect()
    }
}

#[derive(Clone, Copy)]
struct Plane {
    center: (f32, f32),
    radius: f32,
    coefficients: [[f32; 3]; 3],
    /// Per-channel approximation error observed on the clean scene donors.
    /// An affine fit is not an exact background measurement: these residuals
    /// cannot subsequently be counted as evidence of moving foreground.
    maximum_error: [f32; 3],
}

impl Plane {
    fn evaluate(&self, x: usize, y: usize) -> [f32; 3] {
        let basis = [
            1.0,
            (x as f32 - self.center.0) / self.radius,
            (y as f32 - self.center.1) / self.radius,
        ];
        self.coefficients
            .map(|p| p.iter().zip(basis).map(|(a, b)| a * b).sum::<f32>())
    }
}

fn neighbors(
    i: usize,
    blocked: &[bool],
    width: usize,
    height: usize,
) -> impl Iterator<Item = usize> {
    let (x, y) = (i % width, i / width);
    // Erosion, source-edge support and donor patches all use square raster
    // neighborhoods. A unit-cost diagonal keeps this distance consistent with
    // those squares. Never tunnel between protected pixels at a corner.
    [
        (x > 0).then(|| i - 1),
        (x + 1 < width).then(|| i + 1),
        (y > 0).then(|| i - width),
        (y + 1 < height).then(|| i + width),
        (x > 0 && y > 0 && !blocked[i - 1] && !blocked[i - width]).then(|| i - width - 1),
        (x + 1 < width && y > 0 && !blocked[i + 1] && !blocked[i - width]).then(|| i - width + 1),
        (x > 0 && y + 1 < height && !blocked[i - 1] && !blocked[i + width]).then(|| i + width - 1),
        (x + 1 < width && y + 1 < height && !blocked[i + 1] && !blocked[i + width])
            .then(|| i + width + 1),
    ]
    .into_iter()
    .flatten()
}

fn distances(
    seeds: &[bool],
    blocked: &[bool],
    width: usize,
    height: usize,
    limit: usize,
) -> (Vec<usize>, Vec<usize>) {
    let count = width * height;
    let mut distance = vec![usize::MAX; count];
    let mut nearest = vec![usize::MAX; count];
    let mut queue = VecDeque::new();
    for i in 0..count {
        if seeds[i] && !blocked[i] {
            distance[i] = 0;
            nearest[i] = i;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        if distance[i] >= limit {
            continue;
        }
        let next = distance[i] + 1;
        for j in neighbors(i, blocked, width, height) {
            if !blocked[j] && next < distance[j] {
                distance[j] = next;
                nearest[j] = nearest[i];
                queue.push_back(j);
            }
        }
    }
    (distance, nearest)
}

/// A line-of-sight test additionally prevents a small local fitting patch from
/// taking samples across an authoritative hair/ear/user-protection barrier.
fn visible(a: usize, b: usize, blocked: &[bool], width: usize) -> bool {
    let (ax, ay) = ((a % width) as isize, (a / width) as isize);
    let (bx, by) = ((b % width) as isize, (b / width) as isize);
    let steps = (bx - ax).unsigned_abs().max((by - ay).unsigned_abs());
    for step in 0..=steps {
        let t = step as f32 / steps.max(1) as f32;
        let x = (ax as f32 + (bx - ax) as f32 * t).round() as usize;
        let y = (ay as f32 + (by - ay) as f32 * t).round() as usize;
        if blocked[y * width + x] {
            return false;
        }
    }
    true
}

fn fit_plane(
    source: &RgbaImage,
    scene: &[bool],
    blocked: &[bool],
    center: usize,
    radius: usize,
) -> Option<Plane> {
    let (width, height) = (source.width as usize, source.height as usize);
    let (cx, cy) = (center % width, center / width);
    let mut normal = Matrix3::<f64>::zeros();
    let mut rhs = [Vector3::<f64>::zeros(); 3];
    let mut observations = Vec::new();
    // Every sample and straight ray stays inside this rectangular fitting
    // patch. With no blocked texel in the patch, every visibility query is
    // exactly true; skipping those rays does not change donor order or sums.
    let patch_has_blocker = (cy.saturating_sub(radius)..=(cy + radius).min(height - 1)).any(|y| {
        blocked[y * width + cx.saturating_sub(radius)..=y * width + (cx + radius).min(width - 1)]
            .iter()
            .any(|&b| b)
    });
    for y in cy.saturating_sub(radius)..=(cy + radius).min(height - 1) {
        for x in cx.saturating_sub(radius)..=(cx + radius).min(width - 1) {
            let i = y * width + x;
            if !scene[i] || (patch_has_blocker && !visible(center, i, blocked, width)) {
                continue;
            }
            let basis = Vector3::new(
                1.0,
                (x as f64 - cx as f64) / radius as f64,
                (y as f64 - cy as f64) / radius as f64,
            );
            let color = std::array::from_fn::<_, 3, _>(|c| source.pixels[i * 4 + c] as f64);
            normal += basis * basis.transpose();
            for c in 0..3 {
                rhs[c] += basis * color[c];
            }
            observations.push((basis, color));
        }
    }
    if observations.len() < 12 {
        return None;
    }
    // A line of donors cannot justify a two-dimensional extrapolation.
    let inverse = normal.try_inverse()?;
    if inverse.norm() > 8.0 {
        return None;
    }
    let coefficients = rhs.map(|r| inverse * r);
    let mut squared = [0.0_f64; 3];
    let mut maximum = [0.0_f64; 3];
    for (basis, color) in &observations {
        for c in 0..3 {
            let error = (coefficients[c].dot(basis) - color[c]).abs();
            squared[c] += error * error;
            maximum[c] = maximum[c].max(error);
        }
    }
    // Flat channels must not dilute an invalid fit in another channel (for
    // example a repeating red stripe with constant green and blue).
    let rms = squared
        .into_iter()
        .map(|sum| (sum / observations.len() as f64).sqrt())
        .fold(0.0_f64, f64::max);
    // These are model-validity tolerances in decoded byte units. They do not
    // change with editing strength. A sharp pattern or texture is not a ramp.
    if rms > 2.5 || maximum.into_iter().any(|error| error > 8.0) {
        return None;
    }
    if coefficients
        .iter()
        .any(|c| c[1].hypot(c[2]) / radius as f64 > 3.0)
    {
        return None;
    }
    Some(Plane {
        center: (cx as f32, cy as f32),
        radius: radius as f32,
        coefficients: coefficients.map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]),
        maximum_error: maximum.map(|v| v as f32),
    })
}

/// Each value comes only from an already accepted, finite source plane.
/// Retaining one robust point does not turn correlated contexts into multiple
/// independent high-precision observations. Input order cannot change it.
fn context_center(predictions: &[[f32; 3]], low: [f32; 3], high: [f32; 3]) -> [f32; 3] {
    assert!(!predictions.is_empty());
    std::array::from_fn(|c| {
        let mut values: Vec<f32> = predictions.iter().map(|p| p[c]).collect();
        values.sort_by(f32::total_cmp);
        let n = values.len();
        let median = if n % 2 == 0 {
            0.5 * (values[n / 2 - 1] + values[n / 2])
        } else {
            values[n / 2]
        };
        if low[c] <= high[c] {
            median.clamp(low[c], high[c])
        } else {
            // The existing compatibility gate permits <=.5 byte empty overlap.
            // Its midpoint avoids an invalid clamp without changing that gate.
            0.5 * (low[c] + high[c])
        }
    })
}

/// If C=aF+(1-a)B and F is a finite byte-range RGB, each channel imposes a
/// lower bound on a. Half a byte accounts for the source's quantization. This
/// bound also represents a dark edge shadow even when it is not collinear
/// with the average skin colour; a skin/background line fit alone rejects it.
fn opacity_lower_bound(observed: &[u8], background: [f32; 3]) -> f32 {
    (0..3).fold(0.0_f32, |bound, c| {
        let b = background[c];
        let lo = (observed[c] as f32 - 0.5).max(0.0);
        let hi = (observed[c] as f32 + 0.5).min(255.0);
        let required = if b > hi {
            (b - hi) / b.max(1e-6)
        } else if b < lo {
            (lo - b) / (255.0 - b).max(1e-6)
        } else {
            0.0
        };
        bound.max(required).clamp(0.0, 1.0)
    })
}

/// Upper end of the *observed residual's* opacity-bound interval, including
/// scene-model error and source quantization. This is not an upper bound on
/// the true matte. It lets two adjacent decoded samples be compared without
/// treating a one-byte quantization change as increasing material opacity.
fn residual_upper_bound(observed: &[u8], background: [f32; 3], error: [f32; 3]) -> f32 {
    (0..3).fold(0.0_f32, |bound, c| {
        let low = (background[c] - error[c]).max(0.0);
        let high = (background[c] + error[c]).min(255.0);
        let source_low = (observed[c] as f32 - 0.5).max(0.0);
        let source_high = (observed[c] as f32 + 0.5).min(255.0);
        let darker = (high - source_low).max(0.0) / high.max(1e-6);
        let brighter = (source_high - low).max(0.0) / (255.0 - low).max(1e-6);
        bound.max(darker).max(brighter).clamp(0.0, 1.0)
    })
}

pub(crate) fn prepare(
    source: &RgbaImage,
    alpha: &[f32],
    moving_owned: &[bool],
    blocked: &[bool],
    portrait_support: &[f32],
    fringe: usize,
) -> SceneEdgeEvidence {
    prepare_impl(
        source,
        alpha,
        moving_owned,
        blocked,
        portrait_support,
        fringe,
        false,
    )
}

/// Experimental and deliberately unavailable to production callers. Synthetic
/// success does not override its failed real-photo boundary review.
#[cfg(test)]
fn prepare_extended_tail_candidate(
    source: &RgbaImage,
    alpha: &[f32],
    moving_owned: &[bool],
    blocked: &[bool],
    portrait_support: &[f32],
    fringe: usize,
) -> SceneEdgeEvidence {
    prepare_impl(
        source,
        alpha,
        moving_owned,
        blocked,
        portrait_support,
        fringe,
        true,
    )
}

fn prepare_impl(
    source: &RgbaImage,
    alpha: &[f32],
    moving_owned: &[bool],
    blocked: &[bool],
    portrait_support: &[f32],
    fringe: usize,
    extended_tail_candidate: bool,
) -> SceneEdgeEvidence {
    let (width, height) = (source.width as usize, source.height as usize);
    let count = width * height;
    let mut evidence = SceneEdgeEvidence {
        background: vec![None; count],
        minimum_alpha: vec![0.0; count],
        model_center: vec![None; count],
        model_error: vec![None; count],
    };
    if count == 0
        || [
            alpha.len(),
            moving_owned.len(),
            blocked.len(),
            portrait_support.len(),
        ]
        .iter()
        .any(|&n| n != count)
        || source.pixels.len() != count * 4
        || source.pixels.chunks_exact(4).any(|p| p[3] != 255)
        || alpha.iter().chain(portrait_support).any(|v| !v.is_finite())
    {
        return evidence;
    }
    let fringe = fringe.clamp(2, 12);
    // A semantic localization margin and a decoded RGB transition are two
    // different uncertainties. Observe at most one additional margin, rather
    // than assuming that the semantic zero ends the image's mixed material.
    // This is a bounded candidate band, not permission to move every residual.
    let decoded_fringe = if extended_tail_candidate {
        2 * fringe
    } else {
        fringe
    };
    let reach = 3 * fringe + 2;
    // Include every observed portrait class, not just the moving facial matte.
    // An eroded scene seed must be far enough from this union to omit the
    // decoded antialias fringe even if one semantic class becomes zero early.
    let support: Vec<_> = (0..count)
        .map(|i| alpha[i] > 1.0 / 255.0 || portrait_support[i] > 1.0 / 255.0 || blocked[i])
        .collect();
    // Erosion uses an unobstructed transform: barriers themselves are support
    // seeds and keep a scene seed away from their mixed decoded RGB border.
    let (support_distance, _) = distances(
        &support,
        &vec![false; count],
        width,
        height,
        decoded_fringe + 1,
    );
    let scene: Vec<_> = (0..count)
        .map(|i| !blocked[i] && !support[i] && support_distance[i] > decoded_fringe)
        .collect();
    let (scene_distance, nearest_scene) = distances(&scene, blocked, width, height, reach);
    let (moving_distance, _) = distances(moving_owned, blocked, width, height, decoded_fringe);
    let mut fitted = vec![false; count];
    let mut planes = vec![None; count];
    let mut residual_upper = if extended_tail_candidate {
        vec![0.0; count]
    } else {
        Vec::new()
    };
    let mut shells = if extended_tail_candidate {
        vec![Vec::new(); decoded_fringe + 1]
    } else {
        Vec::new()
    };
    for i in 0..count {
        if blocked[i]
            || scene[i]
            || moving_distance[i] > decoded_fringe
            || scene_distance[i] > reach
        {
            continue;
        }
        let seed = nearest_scene[i];
        if seed == usize::MAX || !visible(i, seed, blocked, width) {
            continue;
        }
        if !fitted[seed] {
            // Validate the model on an observed context at least as wide as
            // its allowed extrapolation. A short linear piece of a repeating
            // texture is not evidence that the entire hidden band is a ramp.
            planes[seed] = fit_plane(source, &scene, blocked, seed, reach);
            fitted[seed] = true;
        }
        let background = if let Some(plane) = planes[seed] {
            let background = plane.evaluate(i % width, i / width);
            if background.iter().any(|&v| !(0.0..=255.0).contains(&v)) {
                continue;
            }
            evidence.model_center[i] = Some(background);
            evidence.model_error[i] = Some(plane.maximum_error);
            if extended_tail_candidate {
                residual_upper[i] = residual_upper_bound(
                    &source.pixels[i * 4..i * 4 + 3],
                    background,
                    plane.maximum_error,
                );
            }
            std::array::from_fn(|c| {
                (source.pixels[i * 4 + c] as f32).clamp(
                    (background[c] - plane.maximum_error[c]).max(0.0),
                    (background[c] + plane.maximum_error[c]).min(255.0),
                )
            })
        } else if !extended_tail_candidate
            && (alpha[i] > 1.0 / 255.0 || portrait_support[i] > 1.0 / 255.0)
        {
            // A single failed nearest patch is not proof that every observed
            // patch is invalid. Consider all visible, independently validated
            // scene models within the SAME source reach. Disagreement rejects
            // the inference; model uncertainty is retained as a union envelope.
            // The initial matte may itself have missed this decoded edge.
            // Requiring that estimate to be positive made valid source
            // evidence disappear exactly where it was needed to correct the
            // underestimate. Independent observed portrait support may admit
            // the same contexts. With neither source support nor opacity, a
            // failed nearest model remains a rejection: farther short linear
            // views of a periodic scene cannot recruit unsupported backdrop.
            // Distance, erosion, visibility and model compatibility still gate
            // every admitted prediction; support itself is never made alpha.
            let (x, y) = (i % width, i / width);
            let mut low = [255.0_f32; 3];
            let mut high = [0.0_f32; 3];
            let mut common_low = [0.0_f32; 3];
            let mut common_high = [255.0_f32; 3];
            let mut models = 0;
            let mut model_predictions = Vec::new();
            let mut incompatible = false;
            'models: for sy in y.saturating_sub(reach)..=(y + reach).min(height - 1) {
                for sx in x.saturating_sub(reach)..=(x + reach).min(width - 1) {
                    let j = sy * width + sx;
                    if !scene[j] || !visible(i, j, blocked, width) {
                        continue;
                    }
                    if !fitted[j] {
                        planes[j] = fit_plane(source, &scene, blocked, j, reach);
                        fitted[j] = true;
                    }
                    let Some(model) = planes[j] else {
                        continue;
                    };
                    let b = model.evaluate(x, y);
                    if b.iter().any(|&v| !(0.0..=255.0).contains(&v)) {
                        continue;
                    }
                    models += 1;
                    model_predictions.push(b);
                    for c in 0..3 {
                        let lo = (b[c] - model.maximum_error[c]).max(0.0);
                        let hi = (b[c] + model.maximum_error[c]).min(255.0);
                        low[c] = low[c].min(lo);
                        high[c] = high[c].max(hi);
                        common_low[c] = common_low[c].max(lo);
                        common_high[c] = common_high[c].min(hi);
                        if common_low[c] > common_high[c] + 0.5 {
                            incompatible = true;
                            // Intersection can only shrink as models are
                            // added, so disagreement cannot recover later.
                            break 'models;
                        }
                    }
                }
            }
            if models == 0 || incompatible {
                continue;
            }
            // These are the exact same admitted contexts and the exact same
            // compatibility/union envelopes as the accepted fallback. Their
            // nearest-to-source projection remains the opacity-bound evidence.
            // A separate robust center is used once for color reconstruction.
            let center = context_center(&model_predictions, common_low, common_high);
            evidence.model_center[i] = Some(center);
            evidence.model_error[i] = Some(std::array::from_fn(|c| {
                (center[c] - low[c]).abs().max((high[c] - center[c]).abs())
            }));
            std::array::from_fn(|c| (source.pixels[i * 4 + c] as f32).clamp(low[c], high[c]))
        } else {
            continue;
        };
        evidence.minimum_alpha[i] =
            opacity_lower_bound(&source.pixels[i * 4..i * 4 + 3], background);
        evidence.background[i] = Some(background);
        if extended_tail_candidate {
            shells[moving_distance[i]].push(i);
        }
    }
    if !extended_tail_candidate {
        return evidence;
    }
    // Trace a decoded tail outwards from an observed moving layer. A candidate
    // must remain connected through positive source residual and may only
    // decrease (within the measured uncertainty) at each outward shell. A
    // separate background mark, or one beyond a clean pixel, is not a tail.
    // Reject both opacity AND background replacement when this evidence fails.
    let mut connected = moving_owned.to_vec();
    for shell in shells.iter().skip(1) {
        for &i in shell {
            let a = evidence.minimum_alpha[i];
            let follows_tail = a > 0.0
                && neighbors(i, blocked, width, height).any(|j| {
                    connected[j]
                        && moving_distance[j] < moving_distance[i]
                        && (moving_owned[j] || a <= residual_upper[j])
                });
            if follows_tail {
                connected[i] = true;
            } else {
                evidence.minimum_alpha[i] = 0.0;
                evidence.background[i] = None;
            }
        }
    }
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::byte;

    const W: usize = 64;
    const H: usize = 40;

    #[test]
    fn a_local_linear_part_of_periodic_scene_cannot_become_foreground() {
        let (mut source, alpha, owned, blocked, support, _) = fixture(false);
        for y in 0..H {
            for x in 29..W {
                let i = y * W + x;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    20 + (x % 16) as u8,
                    50 + (y % 16) as u8,
                    130,
                    255,
                ]);
            }
        }
        let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
        for y in 8..H - 8 {
            for x in 29..34 {
                let i = y * W + x;
                assert_eq!(
                    e.minimum_alpha[i], 0.0,
                    "stationary repeating backdrop was reclassified at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn low_contrast_scene_model_error_is_not_missing_foreground() {
        for period in [3, 4, 7] {
            for phase in [0, 1, 2] {
                for diagonal in [false, true] {
                    let (mut source, alpha, owned, blocked, support, _) = fixture(false);
                    for y in 0..H {
                        for x in 29..W {
                            let i = y * W + x;
                            // Below the fit's RMS rejection threshold.
                            // An approximate background's ordinary residual
                            // is not evidence of moving foreground.
                            let position = x + if diagonal { y } else { 0 };
                            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                                85 + (3 * ((position + phase) % period) / (period - 1)) as u8,
                                170 + (3 * ((y + phase) % period) / (period - 1)) as u8,
                                178,
                                255,
                            ]);
                        }
                    }
                    let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
                    let mut false_pixels = 0;
                    let mut maximum_alpha = 0.0_f32;
                    for y in 8..H - 8 {
                        for x in 29..37 {
                            let i = y * W + x;
                            false_pixels += usize::from(e.minimum_alpha[i] > 1e-6);
                            maximum_alpha = maximum_alpha.max(e.minimum_alpha[i]);
                            if let Some(b) = e.background[i] {
                                for c in 0..3 {
                                    assert_eq!(
                                        byte(b[c]),
                                        source.pixels[i * 4 + c],
                                        "observable background changed at {x},{y}"
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(
                        false_pixels, 0,
                        "{false_pixels} backdrop texels became moving opacity; maximum alpha \
                         {maximum_alpha}; period={period}, phase={phase}, diagonal={diagonal}"
                    );
                }
            }
        }
    }

    #[test]
    fn candidate_connected_decoded_tail_beyond_semantic_fringe_is_not_left_behind() {
        for ramp in [false, true] {
            let (mut source, alpha, owned, blocked, support, _) = fixture(ramp);
            let tail = [0.30, 0.18, 0.12, 0.08, 0.05, 0.025];
            for y in 0..H {
                for (d, &a) in tail.iter().enumerate() {
                    let x = 29 + d;
                    let b = background(x, y, ramp);
                    for c in 0..3 {
                        source.pixels[(y * W + x) * 4 + c] =
                            byte(a * [25.0, 45.0, 30.0][c] + (1.0 - a) * b[c]);
                    }
                }
            }
            let e = prepare_extended_tail_candidate(&source, &alpha, &owned, &blocked, &support, 4);
            // The production entry point intentionally retains its existing
            // bounded behavior until this candidate passes real-photo review.
            let default = prepare(&source, &alpha, &owned, &blocked, &support, 4);
            assert!(default.background[20 * W + 34].is_none());
            assert_eq!(default.minimum_alpha[20 * W + 34], 0.0);
            for y in 8..H - 8 {
                for (d, &truth) in tail.iter().enumerate() {
                    let x = 29 + d;
                    let i = y * W + x;
                    let b = e.background[i].unwrap_or_else(|| {
                        panic!("connected decoded tail lost at {x},{y}, ramp={ramp}")
                    });
                    let a = e.minimum_alpha[i];
                    assert!(a > 0.005 && a <= truth + 0.01, "invalid tail alpha {a}");
                    for c in 0..3 {
                        assert!((b[c] - background(x, y, ramp)[c]).abs() < 0.8);
                        let observed = source.pixels[i * 4 + c] as f32;
                        let foreground = ((observed - (1.0 - a) * b[c]) / a).clamp(0.0, 255.0);
                        assert!((a * foreground + (1.0 - a) * b[c] - observed).abs() <= 0.51);
                    }
                }
                for x in 35..W {
                    let i = y * W + x;
                    assert_eq!(e.minimum_alpha[i], 0.0, "clean scene moved at {x},{y}");
                    assert!(
                        e.background[i].is_none(),
                        "clean scene repainted at {x},{y}"
                    );
                }
            }
        }
    }

    #[test]
    fn candidate_independent_background_dark_line_is_not_attached_to_moving_layer() {
        let (mut source, alpha, owned, blocked, support, _) = fixture(false);
        // The true facial tail ends at x=30. A stationary narrow scene mark
        // is separated by clean background and falls inside the added band.
        for y in 0..H {
            for x in 34..=35 {
                let i = y * W + x;
                source.pixels[i * 4..i * 4 + 3].copy_from_slice(&[45, 100, 105]);
            }
        }
        let e = prepare_extended_tail_candidate(&source, &alpha, &owned, &blocked, &support, 4);
        for y in 8..H - 8 {
            for x in 34..=35 {
                let i = y * W + x;
                assert_eq!(e.minimum_alpha[i], 0.0, "stationary scene mark moved");
                assert!(
                    e.background[i].is_none(),
                    "stationary scene mark was repainted"
                );
            }
        }
    }

    #[test]
    fn candidate_outward_increasing_residual_does_not_extend_a_decoded_tail() {
        let (mut source, alpha, owned, blocked, support, _) = fixture(false);
        // Even with no completely clean separating pixel, a stronger scene
        // mark after a weaker face tail is not continuation of a decaying edge.
        for y in 0..H {
            for x in 31..=34 {
                let b = background(x, y, false);
                for c in 0..3 {
                    source.pixels[(y * W + x) * 4 + c] =
                        byte(0.4 * [25.0, 45.0, 30.0][c] + 0.6 * b[c]);
                }
            }
        }
        let e = prepare_extended_tail_candidate(&source, &alpha, &owned, &blocked, &support, 4);
        for y in 8..H - 8 {
            for x in 29..=30 {
                assert!(e.minimum_alpha[y * W + x] > 0.03, "real edge was lost");
            }
            for x in 31..=34 {
                let i = y * W + x;
                assert_eq!(e.minimum_alpha[i], 0.0, "increasing scene residual moved");
                assert!(
                    e.background[i].is_none(),
                    "increasing scene residual repainted"
                );
            }
        }
    }

    fn fixture(
        ramp: bool,
    ) -> (
        RgbaImage,
        Vec<f32>,
        Vec<bool>,
        Vec<bool>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let mut pixels = Vec::new();
        let mut parsed = Vec::new();
        let mut owned = Vec::new();
        let mut truth = Vec::new();
        for y in 0..H {
            for x in 0..W {
                let a = match x {
                    0..=26 => 1.0,
                    27 => 0.85,
                    28 => 0.6,
                    29 => 0.3,
                    30 => 0.1,
                    _ => 0.0,
                };
                // A dark shadow at the mixed edge is not collinear with the
                // nearby opaque skin. It is still physically valid material.
                let f = if x >= 27 {
                    [25.0, 45.0, 30.0]
                } else {
                    [220.0, 155.0, 100.0]
                };
                let b = background(x, y, ramp);
                pixels.extend((0..3).map(|c| byte(a * f[c] + (1.0 - a) * b[c])));
                pixels.push(255);
                parsed.push(if x <= 28 { 1.0 } else { 0.0 });
                owned.push(x <= 28);
                truth.push(a);
            }
        }
        let support = parsed.clone();
        (
            RgbaImage::new(W as u32, H as u32, pixels).unwrap(),
            parsed,
            owned,
            vec![false; W * H],
            support,
            truth,
        )
    }

    fn background(x: usize, y: usize, ramp: bool) -> [f32; 3] {
        if ramp {
            [
                70.0 + x as f32 * 0.5,
                150.0 + y as f32 * 0.4,
                170.0 + (x + y) as f32 * 0.2,
            ]
        } else {
            [85.0, 170.0, 178.0]
        }
    }

    #[test]
    fn omitted_shadow_fringe_has_source_consistent_opacity_on_flat_and_ramp_scene() {
        for ramp in [false, true] {
            let (source, alpha, owned, blocked, support, truth) = fixture(ramp);
            let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
            for y in 8..H - 8 {
                for x in 29..=30 {
                    let i = y * W + x;
                    let b = e.background[i].expect("clean scene must support the omitted edge");
                    for c in 0..3 {
                        assert!((b[c] - background(x, y, ramp)[c]).abs() < 0.8);
                    }
                    let a = e.minimum_alpha[i];
                    assert!(
                        a > 0.03 && a <= truth[i] + 0.01,
                        "bound {a} is not physically justified at {x},{y}"
                    );
                    for c in 0..3 {
                        let observed = source.pixels[i * 4 + c] as f32;
                        let foreground = ((observed - (1.0 - a) * b[c]) / a).clamp(0.0, 255.0);
                        let reconstructed = a * foreground + (1.0 - a) * b[c];
                        assert!((reconstructed - observed).abs() <= 0.51);
                    }
                }
            }
            for y in 0..H {
                for x in 48..W {
                    let i = y * W + x;
                    assert!(e.background[i].is_none(), "distant scene changed");
                    assert_eq!(e.minimum_alpha[i], 0.0);
                }
            }
        }
    }

    #[test]
    fn omitted_fringe_support_is_consistent_for_axis_aligned_and_diagonal_edges() {
        const SIDE: usize = 72;
        const FRINGE: usize = 4;
        // The trimap and plane donor patches are raster squares. A diagonal
        // edge must receive the same square-width support as a vertical edge;
        // a Manhattan transform drops its outer half and pollutes clean seeds.
        for diagonal in [false, true] {
            let mut pixels = Vec::new();
            let mut alpha = Vec::new();
            let mut owned = Vec::new();
            let mut truth = Vec::new();
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let distance = if diagonal {
                        (x as f32 + y as f32 - 48.0) / 2.0
                    } else {
                        x as f32 - 24.0
                    };
                    // Integer Chebyshev shells: the next diagonal half-step
                    // outside the fourth square shell is already clean scene.
                    let a = (1.0 - distance.max(0.0).ceil() / 5.0).clamp(0.0, 1.0);
                    let b = background(x, y, true);
                    let f = [25.0, 45.0, 30.0];
                    pixels.extend((0..3).map(|c| byte(a * f[c] + (1.0 - a) * b[c])));
                    pixels.push(255);
                    alpha.push(if distance <= 0.0 { 1.0 } else { 0.0 });
                    owned.push(distance <= 0.0);
                    truth.push(a);
                }
            }
            let source = RgbaImage::new(SIDE as u32, SIDE as u32, pixels).unwrap();
            let blocked = vec![false; SIDE * SIDE];
            let e = prepare(&source, &alpha, &owned, &blocked, &alpha, FRINGE);
            for d in 1..=FRINGE {
                let (x, y) = (24 + d, if diagonal { 24 + d } else { 32 });
                let i = y * SIDE + x;
                let b = e.background[i].unwrap_or_else(|| {
                    panic!("missing fringe at distance {d}, diagonal={diagonal}")
                });
                let a = e.minimum_alpha[i];
                assert!(
                    a > 0.05 && a <= truth[i] + 0.01,
                    "unsupported bound {a} at distance {d}, diagonal={diagonal}"
                );
                for c in 0..3 {
                    assert!((b[c] - background(x, y, true)[c]).abs() < 0.8);
                    let observed = source.pixels[i * 4 + c] as f32;
                    let foreground = ((observed - (1.0 - a) * b[c]) / a).clamp(0.0, 255.0);
                    let reconstructed = a * foreground + (1.0 - a) * b[c];
                    assert!((reconstructed - observed).abs() <= 0.51);
                }
            }
        }
    }

    #[test]
    fn square_distance_never_crosses_blocked_corners() {
        let mut seeds = vec![false; 25];
        seeds[0] = true;
        let mut blocked = vec![false; 25];
        blocked[1] = true;
        blocked[5] = true;
        let (distance, _) = distances(&seeds, &blocked, 5, 5, 5);
        assert_eq!(distance[6], usize::MAX, "escaped a closed blocked corner");
        assert_eq!(distance[24], usize::MAX);

        blocked.fill(false);
        let (distance, _) = distances(&seeds, &blocked, 5, 5, 4);
        assert_eq!(distance[4], 4);
        assert_eq!(
            distance[24], 4,
            "diagonal square support must be full width"
        );
    }

    #[test]
    fn explicit_barrier_prevents_using_the_scene_across_it() {
        let (source, alpha, owned, mut blocked, support, _) = fixture(false);
        for y in 0..H {
            blocked[y * W + 31] = true;
        }
        let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
        for y in 0..H {
            for x in 0..=31 {
                let i = y * W + x;
                assert!(
                    e.background[i].is_none(),
                    "crossed the protected barrier at {x},{y}"
                );
                assert_eq!(e.minimum_alpha[i], 0.0);
            }
        }
    }

    #[test]
    fn patterned_backdrop_is_not_replaced_by_an_affine_guess() {
        let (mut source, alpha, owned, blocked, support, _) = fixture(false);
        for y in 0..H {
            for x in 31..W {
                let i = y * W + x;
                let level = if (x + y) % 2 == 0 { 50 } else { 210 };
                source.pixels[i * 4..i * 4 + 3].fill(level);
            }
        }
        let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
        assert!(e.background.iter().all(Option::is_none));
        assert!(e.minimum_alpha.iter().all(|&a| a == 0.0));
    }

    #[test]
    fn transparent_and_malformed_sources_produce_no_scene_evidence() {
        let (mut source, alpha, owned, blocked, support, _) = fixture(false);
        source.pixels[3] = 0;
        assert!(prepare(&source, &alpha, &owned, &blocked, &support, 4)
            .background
            .iter()
            .all(Option::is_none));
        assert!(prepare(&source, &[], &owned, &blocked, &support, 4)
            .background
            .iter()
            .all(Option::is_none));
    }
}

#[cfg(test)]
mod alternate_scene_tests {
    use super::*;
    const W: usize = 96;
    const H: usize = 96;
    fn fixture() -> (RgbaImage, Vec<f32>, Vec<bool>, Vec<bool>, Vec<f32>) {
        let mut pixels = vec![0; W * H * 4];
        let mut alpha = vec![0.0; W * H];
        let mut owned = vec![false; W * H];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let true_a = if x < 40 {
                    1.0
                } else if x == 40 {
                    0.7
                } else {
                    0.0
                };
                alpha[i] = if x < 40 {
                    1.0
                } else if x == 40 {
                    0.2
                } else {
                    0.0
                };
                owned[i] = x < 40;
                for c in 0..3 {
                    pixels[i * 4 + c] = (true_a * [210.0, 170.0, 140.0][c]
                        + (1.0 - true_a) * [20.0, 30.0, 40.0][c])
                        as u8;
                }
                pixels[i * 4 + 3] = 255;
            }
        }
        let support = alpha.clone();
        (
            RgbaImage::new(W as u32, H as u32, pixels).unwrap(),
            alpha,
            owned,
            vec![false; W * H],
            support,
        )
    }
    #[test]
    fn invalid_nearest_context_does_not_discard_consistent_visible_contexts() {
        let (mut source, alpha, owned, blocked, support) = fixture();
        // This observed object lies in the nearest context but outside several
        // other equally local views of the same clean wall. The wall remains
        // visible between the mixed source pixel and those alternative donors.
        source.pixels[(30 * W + 45) * 4] = 220;
        let observed_support: Vec<_> = (0..W * H)
            .map(|i| alpha[i] > 1.0 / 255.0 || support[i] > 1.0 / 255.0)
            .collect();
        let (dist, _) = distances(&observed_support, &blocked, W, H, 5);
        let scene: Vec<_> = (0..W * H)
            .map(|i| !observed_support[i] && dist[i] > 4)
            .collect();
        let (_, nearest) = distances(&scene, &blocked, W, H, 14);
        assert!(
            fit_plane(&source, &scene, &blocked, nearest[48 * W + 40], 14).is_none(),
            "test must exercise a rejected nearest plane"
        );
        let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
        let i = 48 * W + 40;
        let b = e.background[i].expect("valid alternative scene contexts should remain usable");
        for c in 0..3 {
            assert!((b[c] - [20.0, 30.0, 40.0][c]).abs() < 0.01);
        }
        let center = e.model_center[i].expect("accepted contexts must supply a color center");
        for c in 0..3 {
            assert!(
                (center[c] - [20., 30., 40.][c]).abs() < 0.01,
                "rejected nearest red outlier entered the reconstruction center: {center:?}"
            );
        }
        assert!(e.minimum_alpha[i] > 0.5 && e.minimum_alpha[i] <= 0.7);
        assert_eq!(alpha[i], 0.2);
    }
    #[test]
    fn visible_texture_between_portrait_and_far_wall_cannot_be_explained_away() {
        let (mut source, alpha, owned, blocked, support) = fixture();
        // The line is visible in the scene immediately behind the edited edge.
        // A farther flat patch cannot establish that the hidden edge is flat.
        for x in 45..W {
            source.pixels[(48 * W + x) * 4..(48 * W + x) * 4 + 3].copy_from_slice(&[190, 80, 60]);
        }
        let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
        let i = 48 * W + 40;
        assert!(
            e.background[i].is_none(),
            "a visible heterogeneous backdrop must reject flat extrapolation"
        );
        assert_eq!(e.minimum_alpha[i], 0.0);
        assert!(
            e.model_center[i].is_none(),
            "rejected contexts cannot produce a center"
        );
    }
    #[test]
    fn periodic_texture_with_nonzero_parser_confidence_is_not_moving_skin() {
        for period in [3, 7, 16] {
            let (mut source, mut alpha, owned, blocked, mut support) = fixture();
            for y in 0..H {
                for x in 40..W {
                    let i = y * W + x;
                    source.pixels[i * 4..i * 4 + 3].copy_from_slice(&[
                        20 + ((x % period) * 8) as u8,
                        30 + ((y % period) * 8) as u8,
                        40,
                    ]);
                    if x == 40 {
                        alpha[i] = 0.1;
                        support[i] = 0.1;
                    }
                }
            }
            let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
            for y in 20..H - 20 {
                let i = y * W + 40;
                assert_eq!(
                    e.minimum_alpha[i], 0.0,
                    "periodic scene was recruited at y={y},period={period}"
                );
                assert!(e.background[i].is_none());
            }
        }
    }
    #[test]
    fn underestimated_source_opacity_keeps_valid_context_evidence_and_protection() {
        let (mut source, mut alpha, owned, mut blocked, support) = fixture();
        source.pixels[(30 * W + 45) * 4] = 220;
        let i = 48 * W + 40;
        // The source photograph still has true coverage .7 at this pixel.
        // Changing only a deficient initial matte cannot invalidate otherwise
        // identical observed background evidence. In particular, 1/255 is
        // not a physical discontinuity in the decoded source material.
        let mut reference_bound = None;
        for initial in [0.0, 0.003, 1.0 / 255.0, 0.004, 0.2] {
            alpha[i] = initial;
            let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
            let backdrop = e.background[i].expect(
                "a zero source-opacity estimate must not suppress valid alternative source contexts",
            );
            for (actual, expected) in backdrop.into_iter().zip([20.0, 30.0, 40.0]) {
                assert!((actual - expected).abs() < 0.01);
            }
            assert!(e.minimum_alpha[i] > 0.5 && e.minimum_alpha[i] <= 0.7);
            if let Some(reference) = reference_bound {
                assert_eq!(e.minimum_alpha[i], reference,
                    "the same source RGB and accepted contexts must impose the same opacity bound");
            } else {
                reference_bound = Some(e.minimum_alpha[i]);
            }
        }
        alpha[i] = 0.2;
        blocked[i] = true;
        let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
        assert!(e.background[i].is_none());
        assert_eq!(e.minimum_alpha[i], 0.0);
    }

    #[test]
    fn clean_scene_color_remains_zero_opacity_when_nearest_context_fails() {
        let (mut source, mut alpha, owned, blocked, support) = fixture();
        source.pixels[(30 * W + 45) * 4] = 220;
        let i = 48 * W + 40;
        // Same source support and contaminated nearest donor as the omitted
        // foreground regression, but this observation is genuinely the wall.
        source.pixels[i * 4..i * 4 + 3].copy_from_slice(&[20, 30, 40]);
        for initial in [0.0, 0.003, 0.004, 0.2] {
            alpha[i] = initial;
            let e = prepare(&source, &alpha, &owned, &blocked, &support, 4);
            assert!(e.background[i].is_some(), "must exercise accepted alternative contexts");
            assert_eq!(e.minimum_alpha[i], 0.0, "clean scene became moving material");
        }
    }
}

#[cfg(test)]
mod reconstruction_center_tests {
    use super::*;
    #[test]
    fn admitted_context_order_does_not_change_color_center_or_half_byte_gap_handling() {
        let p = [
            [190., 200., 210.],
            [194., 198., 207.],
            [192., 201., 211.],
            [193., 197., 208.],
        ];
        let lo = [191., 198., 209.2];
        let hi = [194., 200., 209.0];
        let expected = context_center(&p, lo, hi);
        assert_eq!(expected, [192.5, 199., 209.1]);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                for k in 0..4 {
                    if i == k || j == k {
                        continue;
                    }
                    let l = (0..4).find(|x| *x != i && *x != j && *x != k).unwrap();
                    assert_eq!(context_center(&[p[i], p[j], p[k], p[l]], lo, hi), expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod color_admission_tests {
    use super::*;
    #[test]
    fn rejected_source_evidence_cannot_leak_a_reconstruction_center() {
        let evidence = SceneEdgeEvidence {
            background: vec![None, Some([190., 191., 192.])],
            minimum_alpha: vec![0., 0.1],
            model_center: vec![Some([200.; 3]), Some([200.; 3])],
            model_error: vec![Some([8.; 3]); 2],
        };
        assert_eq!(evidence.color_prior(), vec![None, Some([200.; 3])]);
    }
}
