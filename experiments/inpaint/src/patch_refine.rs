//! Model-guided, deterministic PatchMatch texture refinement.
//!
//! Barnes et al. SIGGRAPH 2009 supplies propagation and shrinking random
//! search; the model candidate initializes unknown evidence, and immutable
//! known context has higher matching weight. This is an optional texture
//! completion stage, not a semantic model or a reproduction of their paper.
use image_core::{ImageError, InpaintMasks, RgbaImage};

#[derive(Clone, Copy)]
struct Match {
    x: i32,
    y: i32,
    error: f32,
}

struct Context {
    w: usize,
    h: usize,
    rgb: Vec<[f32; 3]>,
    guide: Vec<[f32; 3]>,
    unknown: Vec<bool>,
    blocked: Vec<bool>,
    donor: Vec<bool>,
}

impl Context {
    fn score(&self, x: i32, y: i32, sx: i32, sy: i32, r: i32) -> f32 {
        if sx < 0 || sy < 0 || sx >= self.w as i32 || sy >= self.h as i32 {
            return f32::INFINITY;
        }
        if !self.donor[sy as usize * self.w + sx as usize] {
            return f32::INFINITY;
        }
        let mut score = 0.0;
        let mut weight = 0.0;
        // A bounded sampling footprint keeps browser work independent of
        // original megapixel count. Radius 2 uses every pixel.
        let stride = if r > 4 { 2 } else { 1 };
        for dy in (-r..=r).step_by(stride) {
            for dx in (-r..=r).step_by(stride) {
                let tx = x + dx;
                let ty = y + dy;
                if tx < 0 || ty < 0 || tx >= self.w as i32 || ty >= self.h as i32 {
                    continue;
                }
                let ti = ty as usize * self.w + tx as usize;
                if self.blocked[ti] {
                    continue;
                }
                let si = (sy + dy) as usize * self.w + (sx + dx) as usize;
                let q = if self.unknown[ti] { 0.35 } else { 4.0 };
                for c in 0..3 {
                    let d = self.guide[ti][c] - self.rgb[si][c];
                    score += q * d * d;
                }
                weight += q * 3.0;
            }
        }
        if weight > 0.0 {
            score / weight
        } else {
            f32::INFINITY
        }
    }
}

fn random(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn analysis(source: &RgbaImage, candidate: &RgbaImage, masks: &InpaintMasks) -> Context {
    let scale = (512.0 / source.width.max(source.height) as f32).min(1.0);
    let w = (source.width as f32 * scale).round().max(1.0) as usize;
    let h = (source.height as f32 * scale).round().max(1.0) as usize;
    let mut context = Context {
        w,
        h,
        rgb: vec![[0.0; 3]; w * h],
        guide: vec![[0.0; 3]; w * h],
        unknown: vec![false; w * h],
        blocked: vec![false; w * h],
        donor: vec![false; w * h],
    };
    // Positive area coverage, excluding the whole footprint if even one
    // original texel is protected, selected or not fully opaque. No erased
    // object or hidden RGB can become an exemplar after reduction.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let x0 = x * source.width as usize / w;
            let x1 = ((x + 1) * source.width as usize).div_ceil(w);
            let y0 = y * source.height as usize / h;
            let y1 = ((y + 1) * source.height as usize).div_ceil(h);
            let mut count = 0.0f32;
            let mut known_count = 0.0f32;
            for sy in y0..y1.min(source.height as usize) {
                for sx in x0..x1.min(source.width as usize) {
                    let si = sy * source.width as usize + sx;
                    let unknown = masks.target[si] > 0;
                    let blocked = masks.protection[si] > 0 || source.pixels[si * 4 + 3] != 255;
                    context.unknown[i] |= unknown;
                    context.blocked[i] |= blocked;
                    // The candidate is used only where the canonical target
                    // actually authorizes editing, never as known evidence.
                    for c in 0..3 {
                        context.guide[i][c] += if unknown {
                            candidate.pixels[si * 4 + c]
                        } else {
                            source.pixels[si * 4 + c]
                        } as f32;
                        if !unknown && !blocked {
                            context.rgb[i][c] += source.pixels[si * 4 + c] as f32;
                        }
                    }
                    count += 1.0;
                    if !unknown && !blocked {
                        known_count += 1.0;
                    }
                }
            }
            for c in 0..3 {
                context.guide[i][c] /= count;
                context.rgb[i][c] /= known_count.max(1.0);
            }
        }
    }
    context
}

pub(crate) fn refine(
    source: &RgbaImage,
    candidate: &RgbaImage,
    masks: &InpaintMasks,
) -> Result<RgbaImage, ImageError> {
    let mut ctx = analysis(source, candidate, masks);
    let radius = 6.min((ctx.w.min(ctx.h).saturating_sub(1) / 2) as i32);
    if radius < 1 {
        return Ok(candidate.clone());
    }
    // Integral occupancy makes full donor-patch validity O(1).
    let stride = ctx.w + 1;
    let mut integral = vec![0u32; stride * (ctx.h + 1)];
    for y in 0..ctx.h {
        for x in 0..ctx.w {
            let n = y * ctx.w + x;
            let i = (y + 1) * stride + x + 1;
            integral[i] = u32::from(ctx.unknown[n] || ctx.blocked[n])
                + integral[i - 1]
                + integral[i - stride]
                - integral[i - stride - 1];
        }
    }
    let mut donors = Vec::new();
    for y in radius as usize..ctx.h - radius as usize {
        for x in radius as usize..ctx.w - radius as usize {
            let l = x - radius as usize;
            let r = x + radius as usize + 1;
            let t = y - radius as usize;
            let b = y + radius as usize + 1;
            let occupied = integral[b * stride + r] + integral[t * stride + l]
                - integral[t * stride + r]
                - integral[b * stride + l];
            if occupied == 0 {
                ctx.donor[y * ctx.w + x] = true;
                donors.push((x as i32, y as i32));
            }
        }
    }
    if donors.is_empty() {
        return Err(ImageError::InvalidOperation("texture refinement needs an unprotected opaque known patch; use textureRefinement=none for this selection".into()));
    }
    let targets: Vec<_> = (0..ctx.w * ctx.h)
        .filter(|&i| ctx.unknown[i] && !ctx.blocked[i])
        .collect();
    let mut field = vec![
        Match {
            x: 0,
            y: 0,
            error: f32::INFINITY
        };
        ctx.w * ctx.h
    ];
    let mut seed = 0x91a26bc5;
    for &i in &targets {
        let x = (i % ctx.w) as i32;
        let y = (i / ctx.w) as i32;
        for _ in 0..24 {
            let (sx, sy) = donors[random(&mut seed) as usize % donors.len()];
            let error = ctx.score(x, y, sx, sy, radius);
            if error < field[i].error {
                field[i] = Match {
                    x: sx,
                    y: sy,
                    error,
                };
            }
        }
    }
    for r in [radius, radius.min(4), radius.min(2)] {
        for iteration in 0..4 {
            // Alternating sweeps propagate coherent translations from both
            // sides of the hole. Original source remains the only donor.
            for step in 0..targets.len() {
                let i = targets[if iteration % 2 == 0 {
                    step
                } else {
                    targets.len() - step - 1
                }];
                let x = (i % ctx.w) as i32;
                let y = (i / ctx.w) as i32;
                let mut best = field[i];
                best.error = ctx.score(x, y, best.x, best.y, r);
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= ctx.w as i32 || ny >= ctx.h as i32 {
                        continue;
                    }
                    let other = field[ny as usize * ctx.w + nx as usize];
                    if !other.error.is_finite() {
                        continue;
                    }
                    let sx = other.x - dx;
                    let sy = other.y - dy;
                    let error = ctx.score(x, y, sx, sy, r);
                    if error < best.error {
                        best = Match {
                            x: sx,
                            y: sy,
                            error,
                        };
                    }
                }
                let mut distance = ctx.w.max(ctx.h) as i32;
                while distance >= 1 {
                    let sx =
                        best.x + (random(&mut seed) % (2 * distance + 1) as u32) as i32 - distance;
                    let sy =
                        best.y + (random(&mut seed) % (2 * distance + 1) as u32) as i32 - distance;
                    let error = ctx.score(x, y, sx, sy, r);
                    if error < best.error {
                        best = Match {
                            x: sx,
                            y: sy,
                            error,
                        };
                    }
                    distance /= 2;
                }
                field[i] = best;
            }
            let mut votes = vec![[0.0f32; 4]; ctx.w * ctx.h];
            for &i in &targets {
                let x = (i % ctx.w) as i32;
                let y = (i / ctx.w) as i32;
                let m = field[i];
                let q = 1.0 / (25.0 + m.error);
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let tx = x + dx;
                        let ty = y + dy;
                        if tx < 0 || ty < 0 || tx >= ctx.w as i32 || ty >= ctx.h as i32 {
                            continue;
                        }
                        let ti = ty as usize * ctx.w + tx as usize;
                        if !ctx.unknown[ti] || ctx.blocked[ti] {
                            continue;
                        }
                        // Small-image patches have radius 1.
                        if m.x + dx < 0
                            || m.y + dy < 0
                            || m.x + dx >= ctx.w as i32
                            || m.y + dy >= ctx.h as i32
                        {
                            continue;
                        }
                        let si = (m.y + dy) as usize * ctx.w + (m.x + dx) as usize;
                        if ctx.unknown[si] || ctx.blocked[si] {
                            continue;
                        }
                        for c in 0..3 {
                            votes[ti][c] += q * ctx.rgb[si][c];
                        }
                        votes[ti][3] += q;
                    }
                }
            }
            for &i in &targets {
                if votes[i][3] > 0.0 {
                    for c in 0..3 {
                        ctx.guide[i][c] = votes[i][c] / votes[i][3];
                    }
                }
            }
        }
    }
    let mut output = candidate.clone();
    for y in 0..source.height as usize {
        for x in 0..source.width as usize {
            let i = y * source.width as usize + x;
            if masks.target[i] == 0 {
                continue;
            }
            let ax = x * ctx.w / source.width as usize;
            let ay = y * ctx.h / source.height as usize;
            let mut color = [0.0f32; 3];
            let mut weight = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = ax as i32 + dx;
                    let ny = ay as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= ctx.w as i32 || ny >= ctx.h as i32 {
                        continue;
                    }
                    let m = field[ny as usize * ctx.w + nx as usize];
                    if !m.error.is_finite() {
                        continue;
                    }
                    let sx = x as i32
                        + ((m.x - nx) as f32 * source.width as f32 / ctx.w as f32).round() as i32;
                    let sy = y as i32
                        + ((m.y - ny) as f32 * source.height as f32 / ctx.h as f32).round() as i32;
                    if sx < 0 || sy < 0 || sx >= source.width as i32 || sy >= source.height as i32 {
                        continue;
                    }
                    let si = sy as usize * source.width as usize + sx as usize;
                    if masks.target[si] > 0
                        || masks.protection[si] > 0
                        || source.pixels[si * 4 + 3] != 255
                    {
                        continue;
                    }
                    let q = 1.0 / (25.0 + m.error);
                    for (c, sum) in color.iter_mut().enumerate() {
                        *sum += q * source.pixels[si * 4 + c] as f32;
                    }
                    weight += q;
                }
            }
            if weight > 0.0 {
                for (c, value) in color.iter().enumerate() {
                    output.pixels[i * 4 + c] = (value / weight).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (RgbaImage, RgbaImage, InpaintMasks, RgbaImage) {
        let (w, h) = (64usize, 48usize);
        let mut pixels = vec![255; w * h * 4];
        let mut target = vec![0; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let v = if (x / 4 + y / 4) % 2 == 0 { 45 } else { 210 };
                pixels[i * 4..i * 4 + 3].fill(v);
                if (24..40).contains(&x) && (16..32).contains(&y) {
                    target[i] = 255;
                }
            }
        }
        let clean = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let mut source = clean.clone();
        let mut candidate = clean.clone();
        for (i, &t) in target.iter().enumerate() {
            if t > 0 {
                source.pixels[i * 4..i * 4 + 3].copy_from_slice(&[255, 0, 100]);
                candidate.pixels[i * 4..i * 4 + 3].fill(128);
            }
        }
        let masks = InpaintMasks {
            width: w as u32,
            height: h as u32,
            target,
            protection: vec![0; w * h],
        };
        (source, candidate, masks, clean)
    }

    #[test]
    fn repairs_repeated_structure_from_known_boundary_instead_of_blurry_model_fill() {
        let (source, candidate, masks, clean) = fixture();
        let out = refine(&source, &candidate, &masks).unwrap();
        let error = |image: &RgbaImage| -> f32 {
            masks
                .target
                .iter()
                .enumerate()
                .filter(|(_, t)| **t > 0)
                .map(|(i, _)| (image.pixels[i * 4] as f32 - clean.pixels[i * 4] as f32).abs())
                .sum::<f32>()
                / 256.0
        };
        assert!(
            error(&out) < error(&candidate) * 0.35,
            "refined error {} vs blur {}",
            error(&out),
            error(&candidate)
        );
    }

    #[test]
    fn erased_protected_and_transparent_rgb_are_never_refinement_evidence() {
        let (mut source, candidate, mut masks, _) = fixture();
        for y in 3..12 {
            for x in 3..12 {
                masks.protection[y * 64 + x] = 255;
            }
        }
        for y in 36..44 {
            for x in 3..14 {
                source.pixels[(y * 64 + x) * 4 + 3] = 0;
            }
        }
        let mut changed = source.clone();
        for i in 0..masks.target.len() {
            if masks.target[i] > 0 || masks.protection[i] > 0 || source.pixels[i * 4 + 3] == 0 {
                changed.pixels[i * 4..i * 4 + 3].copy_from_slice(&[1, 253, 50]);
            }
        }
        let a = refine(&source, &candidate, &masks).unwrap();
        let b = refine(&changed, &candidate, &masks).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn tiny_images_and_absence_of_known_opaque_patches_are_explicit() {
        let (source, candidate, mut masks, _) = fixture();
        masks.protection.fill(255);
        assert!(refine(&source, &candidate, &masks).is_err());
        let image = RgbaImage::new(1, 1, vec![1, 2, 3, 255]).unwrap();
        let masks = InpaintMasks {
            width: 1,
            height: 1,
            target: vec![255],
            protection: vec![0],
        };
        assert_eq!(refine(&image, &image, &masks).unwrap(), image);
    }
}
