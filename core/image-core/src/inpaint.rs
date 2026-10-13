//! Deterministic exemplar based inpainting for small object removals.
//!
//! This is a bounded Criminisi-style fill: a target boundary pixel gets a
//! priority from confidence and the continuation of boundary structure, then
//! copies the closest known exemplar patch. It is local and deterministic so native
//! and WASM produce the same result. It does not claim generative completion
//! of large or semantically complex scenes.

use crate::engine::{byte, number};
use crate::mask::{coverage, Region};
use crate::{ImageError, RgbaImage};
use serde_json::Value;

fn invalid(message: &str) -> ImageError {
    ImageError::InvalidOperation(message.to_owned())
}

fn neighbours(index: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let x = index % width;
    let y = index / width;
    [
        (x > 0).then(|| index - 1),
        (x + 1 < width).then(|| index + 1),
        (y > 0).then(|| index - width),
        (y + 1 < height).then(|| index + width),
    ]
    .into_iter()
    .flatten()
}

fn patch_score(
    target: usize,
    candidate: usize,
    radius: i32,
    width: usize,
    context: (&[bool], &[u8], &[u8]),
) -> (f32, usize) {
    let (known, pixels, original) = context;
    let tx = target as i32 % width as i32;
    let ty = target as i32 / width as i32;
    let cx = candidate as i32 % width as i32;
    let cy = candidate as i32 / width as i32;
    let mut error = 0.0;
    let mut compared = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let txp = tx + dx;
            let typ = ty + dy;
            if txp < 0
                || typ < 0
                || txp >= width as i32
                || typ >= pixels.len() as i32 / (width as i32 * 4)
            {
                continue;
            }
            let ti = typ as usize * width + txp as usize;
            if !known[ti] {
                continue;
            }
            let sxp = cx + dx;
            let syp = cy + dy;
            if sxp < 0
                || syp < 0
                || sxp >= width as i32
                || syp >= pixels.len() as i32 / (width as i32 * 4)
            {
                continue;
            }
            let si = syp as usize * width + sxp as usize;
            let a = ti * 4;
            let b = si * 4;
            for c in 0..3 {
                let difference = pixels[a + c] as f32 - original[b + c] as f32;
                error += difference * difference;
            }
            compared += 1;
        }
    }
    (error / compared.max(1) as f32, compared)
}

fn frontier_priority(
    index: usize,
    radius: i32,
    width: usize,
    height: usize,
    known: &[bool],
    confidence: &[f32],
    pixels: &[u8],
) -> (f32, f32) {
    let x = index % width;
    let y = index / width;
    let x0 = x.saturating_sub(radius as usize);
    let x1 = (x + radius as usize).min(width - 1);
    let y0 = y.saturating_sub(radius as usize);
    let y1 = (y + radius as usize).min(height - 1);
    let mut c = 0.0;
    let mut best_gradient = (0.0f32, 0.0f32);
    let gray = |i: usize| {
        pixels[i * 4] as f32 * 0.2126
            + pixels[i * 4 + 1] as f32 * 0.7152
            + pixels[i * 4 + 2] as f32 * 0.0722
    };
    for py in y0..=y1 {
        for px in x0..=x1 {
            let i = py * width + px;
            c += confidence[i];
            if px > 0
                && px + 1 < width
                && py > 0
                && py + 1 < height
                && [i - 1, i + 1, i - width, i + width]
                    .into_iter()
                    .all(|j| known[j])
            {
                let g = (
                    (gray(i + 1) - gray(i - 1)) * 0.5,
                    (gray(i + width) - gray(i - width)) * 0.5,
                );
                if g.0 * g.0 + g.1 * g.1
                    > best_gradient.0 * best_gradient.0 + best_gradient.1 * best_gradient.1
                {
                    best_gradient = g;
                }
            }
        }
    }
    c /= ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
    let unknown = |px: usize, py: usize| {
        if known[py * width + px] {
            0.0f32
        } else {
            1.0f32
        }
    };
    let nx = unknown((x + 1).min(width - 1), y) - unknown(x.saturating_sub(1), y);
    let ny = unknown(x, (y + 1).min(height - 1)) - unknown(x, y.saturating_sub(1));
    let length = nx.hypot(ny).max(1e-6);
    let data = (-best_gradient.1 * nx + best_gradient.0 * ny).abs() / length / 255.0 + 0.02;
    (c * data, c)
}

/// Fill the `mask`/`region` with known texture while excluding protected and
/// transparent pixels from both target and exemplar donors.
pub(crate) fn inpaint(source: &RgbaImage, params: &Value) -> Result<RgbaImage, ImageError> {
    let width = source.width as usize;
    let height = source.height as usize;
    let count = width
        .checked_mul(height)
        .ok_or(ImageError::InvalidDimensions)?;
    let region = Region::read(params.get("region"));
    let mask = params.get("mask");
    let protection = params.get("protectMask");
    let mut target = vec![false; count];
    let mut known = vec![false; count];
    let mut donors = vec![false; count];
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f32 + 0.5) / width as f32;
            let ny = (y as f32 + 0.5) / height as f32;
            let selected = region.contains(nx, ny)
                && mask
                    .map(|value| coverage(value, nx, ny, source.width, source.height) > 0.05)
                    .unwrap_or(true);
            let frozen = protection
                .map(|value| coverage(value, nx, ny, source.width, source.height) > 1e-6)
                .unwrap_or(false);
            let i = y * width + x;
            target[i] = selected && !frozen;
            donors[i] = !target[i] && !frozen && source.pixels[i * 4 + 3] > 0;
            // Protected subjects and transparent texels are also excluded
            // from patch matching evidence, not merely source patch copying.
            known[i] = donors[i];
        }
    }
    if !target.iter().any(|value| *value) {
        return Ok(source.clone());
    }
    if !donors.iter().any(|value| *value) {
        return Err(invalid(
            "inpaint requires at least one unprotected nontransparent background donor",
        ));
    }

    let requested_radius =
        number(params, "radius", 0.04).clamp(0.005, 0.18) * width.min(height) as f32;
    let mut patch_radius = requested_radius
        .round()
        .clamp(2.0, 8.0)
        .min((width.min(height).saturating_sub(1) / 2) as f32) as i32;
    // A slightly larger matching window disambiguates repeated texture
    // phase before a smaller patch is copied into the hole.
    let mut matching_radius =
        (patch_radius + 1).min((width.min(height).saturating_sub(1) / 2) as i32);
    let quality = number(params, "iterations", 16.0).clamp(1.0, 64.0) as usize;
    let search_radius = (width.max(height) as i32).min(96);
    // Exemplar patches are always from the immutable original background,
    // never from partially synthesised texture or protected foreground.
    let targets: Vec<usize> = (0..count).filter(|&i| target[i]).collect();
    let left = targets
        .iter()
        .map(|i| i % width)
        .min()
        .unwrap()
        .saturating_sub(search_radius as usize + patch_radius as usize);
    let right = (targets.iter().map(|i| i % width).max().unwrap()
        + search_radius as usize
        + patch_radius as usize
        + 1)
    .min(width);
    let top = targets
        .iter()
        .map(|i| i / width)
        .min()
        .unwrap()
        .saturating_sub(search_radius as usize + patch_radius as usize);
    let bottom = (targets.iter().map(|i| i / width).max().unwrap()
        + search_radius as usize
        + patch_radius as usize
        + 1)
    .min(height);
    let crop_width = right - left;
    let crop_height = bottom - top;
    let stride = crop_width + 1;
    let mut prefix = vec![0usize; (crop_width + 1) * (crop_height + 1)];
    for y in 0..crop_height {
        for x in 0..crop_width {
            prefix[(y + 1) * stride + x + 1] = usize::from(donors[(y + top) * width + x + left])
                + prefix[y * stride + x + 1]
                + prefix[(y + 1) * stride + x]
                - prefix[y * stride + x];
        }
    }
    let mut candidates = vec![false; count];
    loop {
        let r = matching_radius as usize;
        for y in top + r..bottom.saturating_sub(r) {
            for x in left + r..right.saturating_sub(r) {
                let x0 = x - left - r;
                let x1 = x - left + r + 1;
                let y0 = y - top - r;
                let y1 = y - top + r + 1;
                let total = prefix[y1 * stride + x1] + prefix[y0 * stride + x0]
                    - prefix[y0 * stride + x1]
                    - prefix[y1 * stride + x0];
                candidates[y * width + x] = total == (2 * r + 1).pow(2);
            }
        }
        if candidates.iter().any(|v| *v) {
            break;
        }
        // Small images or narrow legal background can lack a full requested
        // patch. Reduce the matching window, then the exemplar radius, rather
        // than treating an enlarged matching window as a missing donor.
        if matching_radius > patch_radius {
            matching_radius = patch_radius;
        } else if patch_radius > 1 {
            patch_radius -= 1;
            matching_radius = patch_radius;
        } else {
            return Err(invalid(
                "inpaint has no complete unprotected exemplar patch",
            ));
        }
    }
    let mut output = source.clone();
    let mut confidence: Vec<f32> = known.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    let mut remaining = targets.len();
    while remaining > 0 {
        let mut best_target = None;
        let mut best_priority = -1.0;
        let mut target_confidence = 0.0;
        for &i in &targets {
            if known[i] {
                continue;
            }
            if !neighbours(i, width, height).any(|n| known[n]) {
                continue;
            }
            let (priority, c) = frontier_priority(
                i,
                patch_radius,
                width,
                height,
                &known,
                &confidence,
                &output.pixels,
            );
            if priority > best_priority {
                best_priority = priority;
                best_target = Some(i);
                target_confidence = c;
            }
        }
        let Some(target_index) = best_target else {
            // Every remaining target island is disconnected from a usable
            // background. Reporting this is safer than smearing foreground.
            return Err(invalid(
                "inpaint target has no reachable unprotected nontransparent donor",
            ));
        };

        let tx = target_index as i32 % width as i32;
        let ty = target_index as i32 / width as i32;
        let radius = matching_radius;
        let mut best_candidate = None;
        let mut best_score = f32::INFINITY;
        let y0 = (ty - search_radius).max(0);
        let y1 = (ty + search_radius).min(height as i32 - 1);
        let x0 = (tx - search_radius).max(0);
        let x1 = (tx + search_radius).min(width as i32 - 1);
        let candidate_limit = 512 + quality * 32;
        let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize;
        let step = (area as f32 / candidate_limit as f32)
            .sqrt()
            .ceil()
            .max(1.0) as usize;
        for y in (y0..=y1).step_by(step) {
            for x in (x0..=x1).step_by(step) {
                let candidate = y as usize * width + x as usize;
                if !candidates[candidate] {
                    continue;
                }
                let (score, compared) = patch_score(
                    target_index,
                    candidate,
                    radius,
                    width,
                    (&known, &output.pixels, &source.pixels),
                );
                // Prefer candidates with more contextual evidence, then their
                // raster order for deterministic native/WASM output.
                let distance = (x - tx).pow(2) + (y - ty).pow(2);
                let adjusted = score + distance as f32 * 0.005 - compared as f32 * 1e-4;
                if adjusted < best_score {
                    best_score = adjusted;
                    best_candidate = Some(candidate);
                }
            }
        }
        if best_candidate.is_none() {
            // A sparse/occluded background can miss the regular sample grid.
            // Find one legal exemplar rather than silently leaving a hole.
            'fallback: for y in y0..=y1 {
                for x in x0..=x1 {
                    let i = y as usize * width + x as usize;
                    if candidates[i] {
                        best_candidate = Some(i);
                        break 'fallback;
                    }
                }
            }
        }
        let Some(mut candidate) = best_candidate else {
            return Err(invalid(
                "inpaint target has no complete known exemplar patch",
            ));
        };
        // Refine neighbouring candidates after bounded coarse raster search,
        // retaining sub-stride texture phases in repeating backgrounds.
        let cx = candidate as i32 % width as i32;
        let cy = candidate as i32 / width as i32;
        let refine = step.min(8) as i32;
        for y in (cy - refine).max(y0)..=(cy + refine).min(y1) {
            for x in (cx - refine).max(x0)..=(cx + refine).min(x1) {
                let i = y as usize * width + x as usize;
                if !candidates[i] {
                    continue;
                }
                let (score, compared) = patch_score(
                    target_index,
                    i,
                    radius,
                    width,
                    (&known, &output.pixels, &source.pixels),
                );
                let distance = (x - tx).pow(2) + (y - ty).pow(2);
                let adjusted = score + distance as f32 * 0.005 - compared as f32 * 1e-4;
                if adjusted < best_score {
                    best_score = adjusted;
                    candidate = i;
                }
            }
        }
        let cx = candidate as i32 % width as i32;
        let cy = candidate as i32 / width as i32;
        let mut correction = [0.0f32; 3];
        let mut comparisons = 0.0;
        for dy in -matching_radius..=matching_radius {
            for dx in -matching_radius..=matching_radius {
                let x = tx + dx;
                let y = ty + dy;
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }
                let ti = y as usize * width + x as usize;
                if !known[ti] {
                    continue;
                }
                let si = (cy + dy) as usize * width + (cx + dx) as usize;
                for (c, value) in correction.iter_mut().enumerate() {
                    *value += output.pixels[ti * 4 + c] as f32 - source.pixels[si * 4 + c] as f32;
                }
                comparisons += 1.0;
            }
        }
        if comparisons > 0.0 {
            for value in &mut correction {
                *value = (*value / comparisons).clamp(-20.0, 20.0);
            }
        }
        // Copy a complete unknown target patch, retaining the known boundary
        // exactly. This preserves exemplar texture instead of averaging it.
        for dy in -patch_radius..=patch_radius {
            for dx in -patch_radius..=patch_radius {
                let x = tx + dx;
                let y = ty + dy;
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }
                let ti = y as usize * width + x as usize;
                if !target[ti] || known[ti] {
                    continue;
                }
                let si = (cy + dy) as usize * width + (cx + dx) as usize;
                output.pixels[ti * 4..ti * 4 + 4]
                    .copy_from_slice(&source.pixels[si * 4..si * 4 + 4]);
                // Match only the patch's low-frequency illumination offset.
                // A constant correction keeps donor gradients and texture,
                // avoiding block seams without averaging the texture away.
                for (c, value) in correction.iter().enumerate() {
                    output.pixels[ti * 4 + c] = byte(output.pixels[ti * 4 + c] as f32 + value);
                }
                known[ti] = true;
                confidence[ti] = target_confidence;
                remaining -= 1;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stripe(width: u32, height: u32) -> RgbaImage {
        RgbaImage::new(
            width,
            height,
            (0..width * height)
                .flat_map(|i| {
                    let x = i % width;
                    let y = i / width;
                    let stripe = if (x / 3 + y / 3).is_multiple_of(2) {
                        40
                    } else {
                        220
                    };
                    [stripe, stripe / 2, 255 - stripe, 255]
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn exemplar_fill_reconstructs_a_small_textured_hole_deterministically() {
        let source = stripe(48, 32);
        let mut damaged = source.clone();
        for y in 12..20 {
            for x in 20..28 {
                let i = (y * 48 + x) * 4;
                damaged.pixels[i..i + 4].copy_from_slice(&[128, 128, 128, 255]);
            }
        }
        let params = json!({"mask":{"shape":"rect","region":{"x":20.0/48.0,"y":12.0/32.0,"width":8.0/48.0,"height":8.0/32.0}},"radius":0.04,"iterations":256});
        let first = inpaint(&damaged, &params).unwrap();
        let second = inpaint(&damaged, &params).unwrap();
        assert_eq!(first, second);
        let mut error = 0u64;
        for y in 12..20 {
            for x in 20..28 {
                let i = (y * 48 + x) * 4;
                for c in 0..3 {
                    error += (first.pixels[i + c] as i32 - source.pixels[i + c] as i32)
                        .unsigned_abs() as u64;
                }
            }
        }
        assert_eq!(
            error, 0,
            "periodic checker texture must match its original phase"
        );
        assert!(first.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn protected_person_and_transparent_texels_never_donate_or_change() {
        let mut source = stripe(32, 24);
        for y in 6..18 {
            for x in 13..19 {
                let i = (y * 32 + x) * 4;
                source.pixels[i..i + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }
        let params = json!({"mask":{"shape":"rect","region":{"x":8.0/32.0,"y":8.0/24.0,"width":16.0/32.0,"height":8.0/24.0}},"protectMask":{"shape":"rect","region":{"x":13.0/32.0,"y":6.0/24.0,"width":6.0/32.0,"height":12.0/24.0}},"radius":0.03,"iterations":256});
        let output = inpaint(&source, &params).unwrap();
        for y in 6..18 {
            for x in 13..19 {
                let i = (y * 32 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
        let mut transparent = source.clone();
        for y in 0..24 {
            for x in 0..32 {
                if x < 4 {
                    transparent.pixels[(y * 32 + x) * 4 + 3] = 0;
                }
            }
        }
        for pixel in transparent.pixels.chunks_exact_mut(4) {
            pixel[3] = 0;
        }
        let output = inpaint(
            &transparent,
            &json!({"mask":{"shape":"rect","region":{"x":0,"y":0,"width":4.0/32.0,"height":1}},"iterations":32}),
        );
        assert!(
            output.is_err(),
            "transparent-only canvas must not count as a donor"
        );
    }

    #[test]
    fn no_unprotected_donor_is_an_explicit_error() {
        let source = RgbaImage::new(12, 12, [100, 100, 100, 255].repeat(144)).unwrap();
        let result = inpaint(
            &source,
            &json!({"mask":{"shape":"rect","region":{"x":0.25,"y":0.25,"width":0.5,"height":0.5}},"protectMask":{"shape":"rect","region":{"x":0.25,"y":0.25,"width":0.5,"height":0.5},"invert":true}}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn matching_window_shrinks_to_fit_narrow_legal_background() {
        let background = [20, 190, 60, 77];
        let mut source = RgbaImage::new(20, 20, background.repeat(400)).unwrap();
        for y in 6..14 {
            for x in 6..14 {
                source.pixels[(y * 20 + x) * 4..(y * 20 + x) * 4 + 3]
                    .copy_from_slice(&[220, 20, 20]);
            }
        }
        let params = json!({
            "mask":{"shape":"rect","region":{"x":0.3,"y":0.3,"width":0.4,"height":0.4}},
            "protectMask":{"shape":"rect","region":{"x":0.3,"y":0.3,"width":0.1,"height":0.4}},
            "radius":0.1,"iterations":4
        });
        let output = inpaint(&source, &params).unwrap();
        for y in 0..20 {
            for x in 0..20 {
                let i = (y * 20 + x) * 4;
                if (6..8).contains(&x) && (6..14).contains(&y) {
                    assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
                } else {
                    assert_eq!(&output.pixels[i..i + 4], &background);
                }
            }
        }
    }

    #[test]
    fn vertical_and_horizontal_stripes_keep_their_original_frequency_and_phase() {
        for horizontal in [false, true] {
            let source = RgbaImage::new(
                48,
                48,
                (0..48 * 48)
                    .flat_map(|i| {
                        let axis = if horizontal { i / 48 } else { i % 48 };
                        let value = if axis % 6 < 3 { 35 } else { 225 };
                        [value, value, 255 - value, 181]
                    })
                    .collect(),
            )
            .unwrap();
            let mut damaged = source.clone();
            for y in 17..29 {
                for x in 17..29 {
                    damaged.pixels[(y * 48 + x) * 4..(y * 48 + x) * 4 + 4]
                        .copy_from_slice(&[255, 0, 0, 255]);
                }
            }
            let params = json!({"mask":{"shape":"rect","region":{"x":17.0/48.0,"y":17.0/48.0,"width":0.25,"height":0.25}},"radius":0.04,"iterations":32});
            let output = inpaint(&damaged, &params).unwrap();
            let differences = output
                .pixels
                .iter()
                .zip(&source.pixels)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                differences, 0,
                "copying must preserve stripe phase, amplitude and background alpha"
            );
        }
    }

    #[test]
    fn protected_and_transparent_rgb_cannot_influence_the_filled_background() {
        let make = |frozen: [u8; 3], transparent: [u8; 3]| {
            let mut source = stripe(48, 32);
            for y in 0..32 {
                for x in 0..48 {
                    let i = (y * 48 + x) * 4;
                    if x < 4 {
                        source.pixels[i..i + 4].copy_from_slice(&[
                            transparent[0],
                            transparent[1],
                            transparent[2],
                            0,
                        ]);
                    }
                    if (34..40).contains(&x) {
                        source.pixels[i..i + 4]
                            .copy_from_slice(&[frozen[0], frozen[1], frozen[2], 255]);
                    }
                    if (18..30).contains(&x) && (10..22).contains(&y) {
                        source.pixels[i..i + 4].copy_from_slice(&[128, 128, 128, 255]);
                    }
                }
            }
            source
        };
        let params = json!({"mask":{"shape":"rect","region":{"x":18.0/48.0,"y":10.0/32.0,"width":12.0/48.0,"height":12.0/32.0}},"protectMask":{"shape":"rect","region":{"x":34.0/48.0,"y":0,"width":6.0/48.0,"height":1}},"radius":0.04});
        let a = make([255, 0, 0], [0, 255, 0]);
        let b = make([0, 0, 255], [255, 0, 255]);
        let ca = inpaint(&a, &params).unwrap();
        let cb = inpaint(&b, &params).unwrap();
        for y in 0..32 {
            for x in 0..48 {
                let i = (y * 48 + x) * 4;
                if x < 4 || (34..40).contains(&x) {
                    assert_eq!(&ca.pixels[i..i + 4], &a.pixels[i..i + 4]);
                    assert_eq!(&cb.pixels[i..i + 4], &b.pixels[i..i + 4]);
                } else {
                    assert_eq!(
                        &ca.pixels[i..i + 4],
                        &cb.pixels[i..i + 4],
                        "ineligible pixel color altered patch matching"
                    );
                }
            }
        }
    }

    #[test]
    fn roi_and_polygon_limit_filling_and_a_fully_frozen_target_is_identity() {
        let source = stripe(32, 24);
        let params = json!({"region":{"x":0.25,"y":0.25,"width":0.25,"height":0.25},"mask":{"shape":"polygon","points":[{"x":0.1,"y":0.1},{"x":0.9,"y":0.1},{"x":0.5,"y":0.9}]},"radius":0.03,"iterations":1});
        let output = inpaint(&source, &params).unwrap();
        for y in 0..24 {
            for x in 0..32 {
                if !(8..16).contains(&x) || !(6..12).contains(&y) {
                    let i = (y * 32 + x) * 4;
                    assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
                }
            }
        }
        let frozen = json!({"mask":{"shape":"rect","region":{"x":0.25,"y":0.25,"width":0.25,"height":0.25}},"protectMask":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}});
        assert_eq!(inpaint(&source, &frozen).unwrap(), source);
    }

    #[test]
    #[ignore = "manual real-street quality check; RETOUCH_STREET_QA_INPUT must name OpenCV vtest frame 50"]
    fn real_street_small_person_removal_quality_and_timing() {
        let input = std::env::var("RETOUCH_STREET_QA_INPUT").expect("RETOUCH_STREET_QA_INPUT");
        let output = std::env::var("RETOUCH_STREET_QA_OUTPUT").expect("RETOUCH_STREET_QA_OUTPUT");
        let decoded = image::open(input).unwrap().into_rgba8();
        let source = RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw()).unwrap();
        assert_eq!((source.width, source.height), (768, 576));
        let params = json!({"mask":{"shape":"polygon","points":[
            {"x":684.0/768.0,"y":251.0/576.0},{"x":704.0/768.0,"y":258.0/576.0},
            {"x":714.0/768.0,"y":291.0/576.0},{"x":713.0/768.0,"y":327.0/576.0},
            {"x":723.0/768.0,"y":381.0/576.0},{"x":697.0/768.0,"y":383.0/576.0},
            {"x":685.0/768.0,"y":358.0/576.0},{"x":667.0/768.0,"y":371.0/576.0},
            {"x":658.0/768.0,"y":350.0/576.0},{"x":670.0/768.0,"y":321.0/576.0},
            {"x":672.0/768.0,"y":284.0/576.0}
        ]},"protectMask":{"shape":"rect","region":{"x":495.0/768.0,"y":204.0/576.0,"width":106.0/768.0,"height":116.0/576.0}},"radius":0.008,"iterations":32});
        let started = std::time::Instant::now();
        let rendered = inpaint(&source, &params).unwrap();
        let elapsed = started.elapsed();
        for y in 204..320 {
            for x in 495..601 {
                let i = (y * 768 + x) * 4;
                assert_eq!(&rendered.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
        let mut changed = 0;
        for (i, (a, b)) in rendered
            .pixels
            .chunks_exact(4)
            .zip(source.pixels.chunks_exact(4))
            .enumerate()
        {
            let nx = ((i % 768) as f32 + 0.5) / 768.0;
            let ny = ((i / 768) as f32 + 0.5) / 576.0;
            if coverage(params.get("mask").unwrap(), nx, ny, 768, 576) <= 0.05 {
                assert_eq!(a, b, "unselected street pixel changed at {i}");
            }
            changed += usize::from(a != b);
        }
        image::RgbaImage::from_raw(rendered.width, rendered.height, rendered.pixels)
            .unwrap()
            .save(output)
            .unwrap();
        eprintln!(
            "street_small_person: changed_pixels={changed}, time_ms={}",
            elapsed.as_millis()
        );
        assert!(changed > 2000);
    }
}
