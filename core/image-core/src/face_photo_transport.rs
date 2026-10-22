//! Offline numerical candidate: transport the observed photograph with one map.
//!
//! Source alpha defines geometric support only; it is never composited into RGB.
//! This carries observed antialias/optical edges without assigning them to a
//! fabricated foreground. The outer band also moves background texture and may
//! distort missed accessories. It is not a physical-layer or inpainting model.
//! Multiplying an integrated inverse field by a taper is NOT a diffeomorphism
//! guarantee. Bilinear cells are C0; determinant diagnostics below are necessary
//! checks, not an aesthetic score or a proof of global injectivity.

use crate::RgbaImage;

#[derive(Clone, Copy, Debug)]
pub struct Crop {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct CellJacobian {
    /// Order: top-left, top-right, bottom-left, bottom-right.
    pub corners: [f64; 4],
    pub min: f64,
    pub max: f64,
    pub nonpositive_corners: usize,
}

#[derive(Clone, Debug)]
pub struct JacobianReport {
    /// Cells in row-major order, (crop.width-1) by (crop.height-1).
    pub cells: Vec<CellJacobian>,
    pub min: f64,
    pub max: f64,
    pub nonpositive_cells: usize,
    pub nonpositive_corners: usize,
}

pub struct PhotoTransport {
    pub image: RgbaImage,
    /// The actual tapered inverse offsets, including zeroed frozen pixels.
    pub inverse: Vec<[f64; 2]>,
    pub jacobian: JacobianReport,
    pub max_input_movement: f64,
    pub fade_width: f64,
    /// Must be zero to join the identity map outside this crop continuously.
    pub boundary_max_movement: f64,
    /// Image sampling clamps these coordinates. Jacobians describe the map
    /// before that clamp, so clamped cases require separate boundary review.
    pub clamped_samples: usize,
}

pub fn render(
    source: &RgbaImage,
    crop: Crop,
    inverse: &[[f32; 2]],
    color_alpha: &[f32],
    frozen: &[bool],
) -> Result<PhotoTransport, &'static str> {
    let (w, h) = (crop.width as usize, crop.height as usize);
    let count = w.checked_mul(h).ok_or("crop dimensions overflow")?;
    if w < 2
        || h < 2
        || inverse.len() != count
        || color_alpha.len() != count
        || frozen.len() != count
        || source.width == 0
        || source.height == 0
        || (source.width as usize)
            .checked_mul(source.height as usize)
            .and_then(|n| n.checked_mul(4))
            != Some(source.pixels.len())
        || crop
            .left
            .checked_add(crop.width)
            .is_none_or(|right| right > source.width)
        || crop
            .top
            .checked_add(crop.height)
            .is_none_or(|bottom| bottom > source.height)
        || inverse.iter().flatten().any(|x| !x.is_finite())
        || color_alpha
            .iter()
            .any(|a| !a.is_finite() || !(0.0..=1.0).contains(a))
    {
        return Err("invalid photo transport dimensions, masks, or field");
    }
    let max_input_movement = inverse
        .iter()
        .map(|v| (v[0] as f64).hypot(v[1] as f64))
        .fold(0.0, f64::max);
    let fade_width = (3.0 * max_input_movement).max(12.0);
    let distance = support_distance(color_alpha, w, h);
    let field: Vec<[f64; 2]> = inverse
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let weight = if frozen[i] {
                0.0
            } else {
                falloff(distance[i], fade_width)
            };
            [v[0] as f64 * weight, v[1] as f64 * weight]
        })
        .collect();
    let boundary_max_movement = field
        .iter()
        .enumerate()
        .filter(|(i, _)| i % w == 0 || i % w == w - 1 || i / w == 0 || i / w == h - 1)
        .map(|(_, v)| v[0].hypot(v[1]))
        .fold(0.0, f64::max);
    let jacobian = cell_jacobians(&field, w, h);
    let mut image = source.clone();
    let mut clamped_samples = 0;
    for (i, &[dx, dy]) in field.iter().enumerate() {
        // Preserve exact original bytes, including hidden RGB at zero alpha.
        if dx == 0.0 && dy == 0.0 {
            continue;
        }
        let x = crop.left as usize + i % w;
        let y = crop.top as usize + i / w;
        let (qx, qy) = (x as f64 + dx, y as f64 + dy);
        if qx < 0.0 || qy < 0.0 || qx > (source.width - 1) as f64 || qy > (source.height - 1) as f64
        {
            clamped_samples += 1;
        }
        let rgba = sample_photo(source, qx, qy);
        let at = (y * source.width as usize + x) * 4;
        image.pixels[at..at + 4].copy_from_slice(&rgba);
    }
    Ok(PhotoTransport {
        image,
        inverse: field,
        jacobian,
        max_input_movement,
        fade_width,
        boundary_max_movement,
        clamped_samples,
    })
}

fn sample_photo(source: &RgbaImage, x: f64, y: f64) -> [u8; 4] {
    let (w, h) = (source.width as usize, source.height as usize);
    let (x, y) = (x.clamp(0.0, (w - 1) as f64), y.clamp(0.0, (h - 1) as f64));
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f64, y - y0 as f64);
    let mut premul = [0.0; 3];
    let mut opacity = 0.0;
    for (xx, yy, weight) in [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x1, y0, tx * (1.0 - ty)),
        (x0, y1, (1.0 - tx) * ty),
        (x1, y1, tx * ty),
    ] {
        let rgba = &source.pixels[(yy * w + xx) * 4..(yy * w + xx + 1) * 4];
        let a = rgba[3] as f64 / 255.0 * weight;
        opacity += a;
        for c in 0..3 {
            premul[c] += rgba[c] as f64 * a;
        }
    }
    let mut out = [0; 4];
    out[3] = (opacity * 255.0).round().clamp(0.0, 255.0) as u8;
    if out[3] > 0 {
        for c in 0..3 {
            out[c] = (premul[c] / opacity).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn falloff(distance: f64, fade_width: f64) -> f64 {
    let t = ((distance - 3.0) / fade_width).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// Exact squared Euclidean distance transform, separable lower parabolas.
fn support_distance(alpha: &[f32], w: usize, h: usize) -> Vec<f64> {
    if !alpha.iter().any(|&a| a > 1e-6) {
        return vec![f64::INFINITY; w * h];
    }
    let mut intermediate = vec![0.0; w * h];
    let mut result = vec![0.0; w * h];
    for y in 0..h {
        let row: Vec<f64> = alpha[y * w..(y + 1) * w]
            .iter()
            .map(|&a| if a > 1e-6 { 0.0 } else { 1e20 })
            .collect();
        intermediate[y * w..(y + 1) * w].copy_from_slice(&distance_1d(&row));
    }
    for x in 0..w {
        let column: Vec<f64> = (0..h).map(|y| intermediate[y * w + x]).collect();
        for (y, d) in distance_1d(&column).into_iter().enumerate() {
            result[y * w + x] = d.sqrt();
        }
    }
    result
}

fn distance_1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut sites = vec![0; n];
    let mut crossings = vec![f64::INFINITY; n + 1];
    crossings[0] = f64::NEG_INFINITY;
    let mut k = 0;
    for q in 1..n {
        let mut s;
        loop {
            let p = sites[k];
            s = ((input[q] + (q as f64).powi(2)) - (input[p] + (p as f64).powi(2)))
                / (2.0 * (q - p) as f64);
            if s > crossings[k] {
                break;
            }
            k -= 1;
        }
        k += 1;
        sites[k] = q;
        crossings[k] = s;
        crossings[k + 1] = f64::INFINITY;
    }
    k = 0;
    (0..n)
        .map(|q| {
            while crossings[k + 1] < q as f64 {
                k += 1;
            }
            (q as f64 - sites[k] as f64).powi(2) + input[sites[k]]
        })
        .collect()
}

fn cell_jacobians(field: &[[f64; 2]], w: usize, h: usize) -> JacobianReport {
    let mut result = JacobianReport {
        cells: Vec::with_capacity((w - 1) * (h - 1)),
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        nonpositive_cells: 0,
        nonpositive_corners: 0,
    };
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let [p, q, r, s] = [
                field[y * w + x],
                field[y * w + x + 1],
                field[(y + 1) * w + x],
                field[(y + 1) * w + x + 1],
            ];
            let (a, b, c, d) = (q[0] - p[0], r[0] - p[0], q[1] - p[1], r[1] - p[1]);
            let (e, f) = (s[0] - q[0] - r[0] + p[0], s[1] - q[1] - r[1] + p[1]);
            let d00 = (1.0 + a) * (1.0 + d) - b * c;
            let du = (1.0 + a) * f - c * e;
            let dv = e * (1.0 + d) - b * f;
            // The u*v determinant terms cancel: extrema are at the corners.
            let corners = [d00, d00 + du, d00 + dv, d00 + du + dv];
            let min = corners.into_iter().fold(f64::INFINITY, f64::min);
            let max = corners.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let nonpositive_corners = corners.iter().filter(|&&v| v <= 0.0).count();
            result.min = result.min.min(min);
            result.max = result.max.max(max);
            result.nonpositive_cells += usize::from(nonpositive_corners > 0);
            result.nonpositive_corners += nonpositive_corners;
            result.cells.push(CellJacobian {
                corners,
                min,
                max,
                nonpositive_corners,
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: u32, h: u32) -> RgbaImage {
        RgbaImage::new(
            w,
            h,
            (0..w * h)
                .flat_map(|i| {
                    [
                        (i % w * 3) as u8,
                        (i / w * 5) as u8,
                        (i % 17 * 11) as u8,
                        255,
                    ]
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn real_pixels_move_by_the_supplied_inverse_offset_inside_support() {
        let source = textured(64, 64);
        let crop = Crop {
            left: 0,
            top: 0,
            width: 64,
            height: 64,
        };
        let mut alpha = vec![0.0; 64 * 64];
        alpha[32 * 64 + 32] = 1.0;
        let out = render(
            &source,
            crop,
            &vec![[1.0, 0.0]; 64 * 64],
            &alpha,
            &vec![false; 64 * 64],
        )
        .unwrap();
        assert_eq!(
            &out.image.pixels[(32 * 64 + 32) * 4..(32 * 64 + 33) * 4],
            &source.pixels[(32 * 64 + 33) * 4..(32 * 64 + 34) * 4]
        );
    }

    #[test]
    fn identity_preserves_every_rgba_byte_including_transparent_hidden_rgb() {
        let mut source = textured(48, 48);
        for (i, rgba) in source.pixels.chunks_exact_mut(4).enumerate() {
            rgba[3] = (i % 256) as u8;
        }
        let crop = Crop {
            left: 8,
            top: 8,
            width: 32,
            height: 32,
        };
        let mut support = vec![0.0; 32 * 32];
        support[16 * 32 + 16] = 1.0;
        let out = render(
            &source,
            crop,
            &vec![[0.0, 0.0]; 32 * 32],
            &support,
            &vec![false; 32 * 32],
        )
        .unwrap();
        assert_eq!(out.image, source);
        assert_eq!((out.jacobian.min, out.jacobian.max), (1.0, 1.0));
    }

    #[test]
    fn both_translation_directions_carry_antialias_and_dark_halo_without_old_line() {
        let (w, h) = (97usize, 49usize);
        let profile: Vec<u8> = (0..w)
            .map(|x| match x {
                29 | 67 => 96,
                30 | 66 => 115,
                31 | 65 => 170,
                32..=64 => 220,
                _ => 100,
            })
            .collect();
        let source = RgbaImage::new(
            w as u32,
            h as u32,
            (0..w * h)
                .flat_map(|i| [profile[i % w], profile[i % w], profile[i % w], 255])
                .collect(),
        )
        .unwrap();
        let crop = Crop {
            left: 0,
            top: 0,
            width: w as u32,
            height: h as u32,
        };
        let mut support = vec![0.0; w * h];
        for x in 30..=66 {
            support[24 * w + x] = if x == 30 || x == 66 { 0.125 } else { 1.0 };
        }
        for dx in [-2.0_f32, -0.5, 0.5, 2.0] {
            let out = render(
                &source,
                crop,
                &vec![[dx, 0.0]; w * h],
                &support,
                &vec![false; w * h],
            )
            .unwrap();
            for x in 3..w - 3 {
                // Independent 1D signal oracle: the encoded observed edge
                // (including its dark overshoot) translates as a whole.
                let q = x as f64 + dx as f64;
                let left = q.floor() as usize;
                let t = q - left as f64;
                let expected =
                    (profile[left] as f64 * (1.0 - t) + profile[left + 1] as f64 * t).round() as u8;
                assert_eq!(
                    out.image.pixels[(24 * w + x) * 4],
                    expected,
                    "dx={dx}, x={x}"
                );
            }
            assert_ne!(out.image, source);
            assert_eq!(out.boundary_max_movement, 0.0);
        }
    }

    #[test]
    fn frozen_pixels_and_every_pixel_outside_the_crop_are_exact() {
        let source = textured(96, 96);
        let crop = Crop {
            left: 16,
            top: 16,
            width: 64,
            height: 64,
        };
        let mut support = vec![0.0; 64 * 64];
        support[32 * 64 + 32] = 1.0;
        let mut frozen = vec![false; 64 * 64];
        frozen[32 * 64 + 32] = true;
        let out = render(&source, crop, &vec![[2.0, 1.0]; 64 * 64], &support, &frozen).unwrap();
        for y in 0..96 {
            for x in 0..96 {
                if !(16..80).contains(&x) || !(16..80).contains(&y) || (x == 48 && y == 48) {
                    let i = (y * 96 + x) * 4;
                    assert_eq!(&out.image.pixels[i..i + 4], &source.pixels[i..i + 4]);
                }
            }
        }
        assert_eq!(out.inverse[32 * 64 + 32], [0.0, 0.0]);
        assert_ne!(out.image, source);
    }

    #[test]
    fn nonempty_textured_field_reaches_identity_continuously_at_outer_band() {
        let source = textured(64, 64);
        let mut support = vec![0.0; 64 * 64];
        support[32 * 64 + 32] = 1.0;
        let out = render(
            &source,
            Crop {
                left: 0,
                top: 0,
                width: 64,
                height: 64,
            },
            &vec![[2.0, 0.0]; 64 * 64],
            &support,
            &vec![false; 64 * 64],
        )
        .unwrap();
        assert_eq!(out.inverse[32 * 64 + 35], [2.0, 0.0]); // first three exterior pixels
        assert_eq!(out.inverse[32 * 64 + 47], [0.0, 0.0]); // 3 + 12 pixels
        assert!(out.inverse[32 * 64 + 46][0] > 0.0 && out.inverse[32 * 64 + 46][0] < 0.05);
        assert_eq!(out.boundary_max_movement, 0.0);
        assert!(falloff(15.0 - 1e-4, 12.0) < 1e-8);
        assert_eq!(falloff(15.0 + 1e-4, 12.0), 0.0);
        assert_ne!(out.image, source);
        assert_eq!(out.jacobian.nonpositive_cells, 0);
    }

    #[test]
    fn support_distance_is_euclidean_including_diagonal_and_empty_rows() {
        let mut alpha = vec![0.0; 10 * 10];
        alpha[3 * 10 + 3] = 1.0;
        let d = support_distance(&alpha, 10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let expected = (x as f64 - 3.0).hypot(y as f64 - 3.0);
                assert!((d[y * 10 + x] - expected).abs() < 1e-12, "({x},{y})");
            }
        }
    }

    #[test]
    fn jacobian_corners_match_independent_edge_vectors_and_detect_folds() {
        let points = [[0.0, 0.0], [1.1, 0.1], [0.1, 0.9], [1.2, 1.3]];
        let field = [[0.0, 0.0], [0.1, 0.1], [0.1, -0.1], [0.2, 0.3]];
        let report = cell_jacobians(&field, 2, 2);
        for (corner, &(a, b, c, d)) in [(0, 1, 0, 2), (0, 1, 1, 3), (2, 3, 0, 2), (2, 3, 1, 3)]
            .iter()
            .enumerate()
        {
            let dx = [points[b][0] - points[a][0], points[b][1] - points[a][1]];
            let dy = [points[d][0] - points[c][0], points[d][1] - points[c][1]];
            let determinant = dx[0] * dy[1] - dx[1] * dy[0];
            assert!((report.cells[0].corners[corner] - determinant).abs() < 1e-12);
        }
        let folded: Vec<_> = (0..9).map(|i| [-2.0 * (i % 3) as f64, 0.0]).collect();
        let report = cell_jacobians(&folded, 3, 3);
        assert_eq!((report.min, report.max), (-1.0, -1.0));
        assert_eq!(report.nonpositive_cells, 4);
        assert_eq!(report.nonpositive_corners, 16);
    }

    #[test]
    fn rgba_interpolation_does_not_import_hidden_transparent_rgb() {
        let mut source = textured(64, 64);
        let at = (32 * 64 + 32) * 4;
        source.pixels[at..at + 4].copy_from_slice(&[255, 0, 0, 255]);
        source.pixels[at + 4..at + 8].copy_from_slice(&[0, 255, 0, 0]);
        let mut support = vec![0.0; 64 * 64];
        support[32 * 64 + 32] = 1.0;
        let out = render(
            &source,
            Crop {
                left: 0,
                top: 0,
                width: 64,
                height: 64,
            },
            &vec![[0.5, 0.0]; 64 * 64],
            &support,
            &vec![false; 64 * 64],
        )
        .unwrap();
        assert_eq!(&out.image.pixels[at..at + 4], &[255, 0, 0, 128]);
    }

    #[test]
    fn nonzero_crop_boundary_is_reported_instead_of_claiming_continuity() {
        let source = textured(8, 8);
        let out = render(
            &source,
            Crop {
                left: 0,
                top: 0,
                width: 8,
                height: 8,
            },
            &[[1.0, 0.0]; 64],
            &[1.0; 64],
            &[false; 64],
        )
        .unwrap();
        assert_eq!(out.boundary_max_movement, 1.0);
        assert_eq!(out.clamped_samples, 8);
    }
}
