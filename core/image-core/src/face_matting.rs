//! Germer et al., "Fast Multi-Level Foreground Estimation", arXiv:2006.14970.
//! Rust translation of the multi-scale 2×2 F/B updates in PyMatting's official
//! estimate_foreground_ml (nearest resize, 10 small / 2 large iterations,
//! regularization=1e-5, gradient_weight=1). Alpha estimation is not included.
//! Only opaque composite photographs satisfy this input model.

use crate::{engine::byte, RgbaImage};

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
