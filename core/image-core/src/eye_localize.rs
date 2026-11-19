//! Source-only eyelid localization. Landmark curves are anatomical priors;
//! only observed side-sclera contrast can correct them. These curves are not
//! optical alpha, material labels, or permission to reveal hidden pixels.
use crate::eye_source::{gaussian_reflect, luma, median, sample_rgb, source_noise, Pchip};
use crate::{EyeSourceGeometry, ImageError, RgbaImage};
use clarabel::{algebra::CscMatrix, solver::*};

const BAND: f64 = 2.;
const STEP: f64 = 0.25;
const SMOOTH_LENGTH: f64 = 4.;

struct EyeSpec {
    name: &'static str,
    upper: [usize; 9],
    lower: [usize; 9],
    iris: usize,
    ring: [usize; 4],
}
const EYES: [EyeSpec; 2] = [
    EyeSpec {
        name: "leftEye",
        upper: [33, 246, 161, 160, 159, 158, 157, 173, 133],
        lower: [33, 7, 163, 144, 145, 153, 154, 155, 133],
        iris: 468,
        ring: [469, 470, 471, 472],
    },
    EyeSpec {
        name: "rightEye",
        upper: [362, 398, 384, 385, 386, 387, 388, 466, 263],
        lower: [362, 382, 381, 380, 374, 373, 390, 249, 263],
        iris: 473,
        ring: [474, 475, 476, 477],
    },
];

fn invalid(message: impl Into<String>) -> ImageError {
    ImageError::InvalidOperation(message.into())
}

fn redness(rgb: [f64; 3]) -> f64 {
    (rgb[0] - rgb[1]) / (luma(rgb) + 20.) * 100.
}

/// Localize both eyes once from original opaque RGBA and normalized detector
/// points. Detector coordinates describe physical image positions, while the
/// returned origins describe integer pixel indices. No target edit is input.
pub fn localize_eye_sources(
    source: &RgbaImage,
    normalized: &[[f64; 2]],
) -> Result<Vec<EyeSourceGeometry>, ImageError> {
    let w = source.width as usize;
    let h = source.height as usize;
    if w == 0
        || h == 0
        || w.checked_mul(h).and_then(|n| n.checked_mul(4)) != Some(source.pixels.len())
        || source.pixels.chunks_exact(4).any(|p| p[3] != 255)
    {
        return Err(invalid(
            "eye localization requires nonempty complete opaque RGBA",
        ));
    }
    if normalized.len() < 478 || normalized.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid(
            "eye localization requires at least 478 finite landmarks",
        ));
    }
    let mesh: Vec<[f64; 2]> = normalized
        .iter()
        .map(|p| [p[0] * w as f64, p[1] * h as f64])
        .collect();
    for spec in &EYES {
        for i in spec
            .upper
            .iter()
            .chain(&spec.lower)
            .chain(&spec.ring)
            .chain([&spec.iris])
        {
            let p = mesh[*i];
            if !p[0].is_finite()
                || !p[1].is_finite()
                || p[0] < 0.5
                || p[1] < 0.5
                || p[0] > w as f64 - 0.5
                || p[1] > h as f64 - 0.5
            {
                return Err(invalid(format!(
                    "{} source landmark {i} leaves pixel-center domain",
                    spec.name
                )));
            }
        }
    }
    let rgb: Vec<f64> = source
        .pixels
        .chunks_exact(4)
        .flat_map(|p| p[..3].iter().map(|v| *v as f64))
        .collect();
    EYES.iter()
        .map(|spec| localize_one(&rgb, w, h, &mesh, spec))
        .collect()
}

fn localize_one(
    rgb: &[f64],
    w: usize,
    h: usize,
    mesh: &[[f64; 2]],
    spec: &EyeSpec,
) -> Result<EyeSourceGeometry, ImageError> {
    let a = mesh[spec.upper[0]];
    let b = mesh[spec.upper[8]];
    let width = (b[0] - a[0]).hypot(b[1] - a[1]);
    if !width.is_finite() || width <= 2. {
        return Err(invalid("source eye frame has insufficient extent"));
    }
    let axis = [(b[0] - a[0]) / width, (b[1] - a[1]) / width];
    let to_local = |p: [f64; 2]| {
        let q = [p[0] - a[0], p[1] - a[1]];
        [
            q[0] * axis[0] + q[1] * axis[1],
            -q[0] * axis[1] + q[1] * axis[0],
        ]
    };
    let at = |u: f64, v: f64| {
        sample_rgb(
            rgb,
            w,
            h,
            [
                a[0] + u * axis[0] - v * axis[1] - 0.5,
                a[1] + u * axis[1] + v * axis[0] - 0.5,
            ],
        )
    };
    let prior = |ids: &[usize; 9]| -> Result<Pchip, ImageError> {
        let mut uv: Vec<_> = ids.iter().map(|i| to_local(mesh[*i])).collect();
        if uv.windows(2).any(|p| p[1][0] <= p[0][0]) {
            return Err(invalid(
                "ordered source eyelid points are not monotone along canthi",
            ));
        }
        uv[0] = [0., 0.];
        uv[8] = [width, 0.];
        Pchip::new(
            &uv.iter().map(|p| p[0]).collect::<Vec<_>>(),
            &uv.iter().map(|p| p[1]).collect::<Vec<_>>(),
        )
    };
    let upper_prior = prior(&spec.upper)?;
    let lower_prior = prior(&spec.lower)?;
    let n = width.ceil() as usize + 1;
    let mut grid: Vec<_> = (0..n).map(|i| width * i as f64 / (n - 1) as f64).collect();
    grid[0] = 0.;
    grid[n - 1] = width;
    let upper: Vec<_> = grid.iter().map(|u| upper_prior.at(*u)).collect();
    let lower: Vec<_> = grid.iter().map(|u| lower_prior.at(*u)).collect();
    if upper.iter().zip(&lower).any(|(t, b)| b - t < -1e-6) {
        return Err(invalid("source anatomical upper/lower priors cross"));
    }
    let center = to_local(mesh[spec.iris]);
    let mut radii = [0_f64; 2];
    for i in spec.ring {
        let p = to_local(mesh[i]);
        for c in 0..2 {
            radii[c] = radii[c].max((p[c] - center[c]).abs());
        }
    }
    if radii.iter().any(|r| !r.is_finite() || *r <= 0.) {
        return Err(invalid("source iris ring has insufficient extent"));
    }

    // Crop bounds retain physical coordinates. Applying the sampling -.5 here
    // would shift a different set of pixels into the source noise estimate.
    let points: Vec<_> = spec
        .upper
        .iter()
        .chain(&spec.lower)
        .map(|i| mesh[*i])
        .collect();
    let context = (width * 0.25).ceil().max(12.);
    let low = |c: usize| {
        (points
            .iter()
            .map(|p| p[c])
            .fold(f64::INFINITY, f64::min)
            .floor()
            - context)
            .max(0.) as usize
    };
    let high = |c: usize, limit: usize| {
        (points
            .iter()
            .map(|p| p[c])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            + context)
            .min(limit as f64) as usize
    };
    let crop = [low(0), low(1), high(0, w), high(1, h)];
    let [x0, y0, x1, y1] = crop;
    if x0 >= x1 || y0 >= y1 {
        return Err(invalid("source eye crop has no extent"));
    }
    let mut lum = Vec::with_capacity((x1 - x0) * (y1 - y0));
    let mut chroma = Vec::with_capacity(lum.capacity());
    for y in y0..y1 {
        for x in x0..x1 {
            let k = 3 * (y * w + x);
            let p = [rgb[k], rgb[k + 1], rgb[k + 2]];
            lum.push(luma(p));
            chroma.push(redness(p));
        }
    }
    let noise = source_noise(&lum, x1 - x0, y1 - y0);
    let blurred = gaussian_reflect(&chroma, x1 - x0, y1 - y0, 0.6);
    let mut errors: Vec<_> = chroma
        .iter()
        .zip(blurred)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let chroma_noise = (1.4826 * median(&mut errors)).max(0.5);
    let iris_rgb = at(center[0], center[1]);
    let mid: Vec<_> = upper
        .iter()
        .zip(&lower)
        .map(|(t, b)| (t + b) * 0.5)
        .collect();
    let evidence: Vec<_> = grid
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let top = at(*u, upper[i] - 4.);
            let bottom = at(*u, lower[i] + 4.);
            let external = std::array::from_fn(|c| (top[c] + bottom[c]) * 0.5);
            let candidate = at(*u, mid[i]);
            (*u - center[0]).abs() > radii[0] + 1.
                && *u > width * 0.12
                && *u < width * 0.88
                && lower[i] - upper[i] > 2. * BAND
                && luma(candidate) > luma(external) + 3. * noise
                && luma(candidate) > luma(iris_rgb) + 3. * noise
                && redness(candidate) < redness(external) - 3. * chroma_noise
        })
        .collect();
    let upper = fit_curve(&at, &grid, &upper, &mid, &evidence, noise, true)?;
    let lower = fit_curve(&at, &grid, &lower, &mid, &evidence, noise, false)?;
    let top = Pchip::new(&grid, &upper)?;
    let bottom = Pchip::new(&grid, &lower)?;
    let dense_steps = (width * 16.).ceil() as usize;
    for i in 0..=dense_steps {
        let u = width * i as f64 / dense_steps as f64;
        if bottom.at(u) - top.at(u) < -1e-7 {
            return Err(invalid(
                "fitted source curves cross; reject without clipping",
            ));
        }
    }
    Ok(EyeSourceGeometry {
        eye: spec.name.into(),
        origin: [a[0] - 0.5, a[1] - 0.5],
        axis,
        width,
        crop,
        knots: grid,
        upper,
        lower,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_curve(
    at: &impl Fn(f64, f64) -> [f64; 3],
    grid: &[f64],
    prior: &[f64],
    mid: &[f64],
    evidence: &[bool],
    noise: f64,
    upper: bool,
) -> Result<Vec<f64>, ImageError> {
    let n = grid.len();
    let inward = if upper { 1. } else { -1. };
    let mut anchors = Vec::new();
    for i in 1..n - 1 {
        if !evidence[i] {
            continue;
        }
        let mut gradients = [f64::NEG_INFINITY; 17];
        for (j, gradient) in gradients.iter_mut().enumerate() {
            let v = prior[i] - BAND + j as f64 * STEP;
            if (upper && v <= mid[i]) || (!upper && v >= mid[i]) {
                let inner = at(grid[i], v + inward * 0.75);
                let outer = at(grid[i], v - inward * 0.75);
                *gradient = luma(std::array::from_fn(|c| inner[c] - outer[c]));
            }
        }
        // Preserve the first maximum on ties, matching the source experiment.
        let mut best = 0;
        for j in 1..gradients.len() {
            if gradients[j] > gradients[best] {
                best = j;
            }
        }
        let maximum = gradients[best];
        if maximum <= 3. * noise || best == 0 || best == gradients.len() - 1 {
            continue;
        }
        let high: Vec<_> = (0..gradients.len())
            .filter(|j| gradients[*j] >= maximum * 0.8)
            .collect();
        if high.windows(2).any(|p| p[1] > p[0] + 1) {
            continue;
        }
        let peak_width = ((high[high.len() - 1] - high[0] + 1) as f64 * STEP).max(STEP);
        anchors.push((i, -BAND + best as f64 * STEP, (peak_width * 0.5).max(0.5)));
    }
    if anchors.is_empty() {
        return Ok(prior.to_vec());
    }
    solve_displacements(grid, prior, mid, &anchors, upper)
}

fn solve_displacements(
    grid: &[f64],
    prior: &[f64],
    mid: &[f64],
    anchors: &[(usize, f64, f64)],
    upper: bool,
) -> Result<Vec<f64>, ImageError> {
    let m = grid.len() - 2;
    // P=A^T A is symmetric pentadiagonal. Construct its upper triangle rather
    // than allocating a dense matrix; Clarabel solves 1/2 x^T P x + q^T x.
    let mut diagonals = vec![[0.; 3]; m];
    let mut q = vec![0.; m];
    for p in &mut diagonals {
        p[0] = 1. / BAND.powi(2);
    }
    let du = grid[1] - grid[0];
    let smooth = (SMOOTH_LENGTH.powi(2) / BAND) / du.powi(2);
    for i in 0..m {
        let mut row = Vec::with_capacity(3);
        for (j, coefficient) in [1., -2., 1.].iter().enumerate() {
            let full = i + j;
            if full > 0 && full <= m {
                row.push((full - 1, coefficient * smooth));
            }
        }
        for (a, (col_a, value_a)) in row.iter().enumerate() {
            for (col_b, value_b) in &row[a..] {
                diagonals[*col_a][col_b - col_a] += value_a * value_b;
            }
        }
    }
    for &(i, delta, sigma) in anchors {
        let weight = 1. / sigma.powi(2);
        diagonals[i - 1][0] += weight;
        q[i - 1] -= delta * weight;
    }
    let mut pr = Vec::new();
    let mut pc = Vec::new();
    let mut pv = Vec::new();
    let mut ar = Vec::with_capacity(2 * m);
    let mut ac = Vec::with_capacity(2 * m);
    let mut av = Vec::with_capacity(2 * m);
    let mut b = Vec::with_capacity(2 * m);
    for i in 0..m {
        for offset in 0..3 {
            if i + offset < m && diagonals[i][offset] != 0. {
                pr.push(i);
                pc.push(i + offset);
                pv.push(diagonals[i][offset]);
            }
        }
        let (low, high) = if upper {
            (-BAND, BAND.min(mid[i + 1] - prior[i + 1]))
        } else {
            ((-BAND).max(mid[i + 1] - prior[i + 1]), BAND)
        };
        if low > high {
            return Err(invalid("source eyelid displacement bounds are infeasible"));
        }
        ar.extend([2 * i, 2 * i + 1]);
        ac.extend([i, i]);
        av.extend([1., -1.]);
        b.extend([high, -low]);
    }
    let p = CscMatrix::new_from_triplets(m, m, pr, pc, pv);
    let a = CscMatrix::new_from_triplets(2 * m, m, ar, ac, av);
    let settings = DefaultSettings {
        verbose: false,
        max_iter: 150,
        tol_feas: 1e-10,
        tol_gap_abs: 1e-10,
        tol_gap_rel: 1e-10,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &[NonnegativeConeT(2 * m)], settings);
    solver.solve();
    if solver.solution.status != SolverStatus::Solved {
        return Err(invalid(format!(
            "source eyelid curve QP {:?}",
            solver.solution.status
        )));
    }
    let mut result = prior.to_vec();
    for (i, displacement) in solver.solution.x.iter().enumerate() {
        if !displacement.is_finite() || displacement.abs() > BAND + 1e-7 {
            return Err(invalid(
                "source eyelid displacement is nonfinite or outside frozen band",
            ));
        }
        result[i + 1] += displacement;
        if (upper && result[i + 1] > mid[i + 1] + 1e-7)
            || (!upper && result[i + 1] < mid[i + 1] - 1e-7)
        {
            return Err(invalid(
                "source eyelid displacement crossed the prior midpoint",
            ));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (RgbaImage, Vec<[f64; 2]>) {
        let (w, h) = (180_u32, 96_u32);
        let source = RgbaImage {
            width: w,
            height: h,
            pixels: [170, 120, 100, 255].repeat((w * h) as usize),
        };
        let mut normalized = vec![[0.5, 0.5]; 478];
        for (eye, spec) in EYES.iter().enumerate() {
            let x0 = 24.25 + eye as f64 * 86.;
            let y0 = 45.75;
            for (side, ids) in [(-1., &spec.upper), (1., &spec.lower)] {
                for (i, id) in ids.iter().enumerate() {
                    let t = i as f64 / 8.;
                    let v = if i == 0 || i == 8 {
                        0.
                    } else {
                        side * 6. * (std::f64::consts::PI * t).sin()
                    };
                    normalized[*id] = [(x0 + 40. * t) / w as f64, (y0 + v) / h as f64];
                }
            }
            normalized[spec.iris] = [(x0 + 20.) / w as f64, y0 / h as f64];
            for (id, (du, dv)) in spec
                .ring
                .iter()
                .zip([(4., 0.), (0., 4.), (-4., 0.), (0., -4.)])
            {
                normalized[*id] = [(x0 + 20. + du) / w as f64, (y0 + dv) / h as f64];
            }
        }
        (source, normalized)
    }

    #[test]
    fn flat_source_preserves_anatomical_prior_and_physical_crop_convention() {
        let (source, normalized) = fixture();
        let localized = localize_eye_sources(&source, &normalized).unwrap();
        assert_eq!(localized.len(), 2);
        let g = &localized[0];
        assert_eq!(g.origin, [23.75, 45.25]);
        assert_eq!(g.crop, [12, 27, 77, 64]);
        assert!((g.width - 40.).abs() < 1e-12);
        assert_eq!(g.upper[0], 0.);
        assert_eq!(g.upper[g.upper.len() - 1], 0.);
        for i in 0..9 {
            let index = i * 5;
            let expected = if i == 0 || i == 8 {
                0.
            } else {
                -6. * (std::f64::consts::PI * i as f64 / 8.).sin()
            };
            assert!((g.upper[index] - expected).abs() < 1e-12);
            assert!((g.lower[index] + expected).abs() < 1e-12);
        }
    }

    #[test]
    fn observed_edge_corrects_displacement_without_flattening_anatomy() {
        let grid: Vec<_> = (0..41).map(|i| i as f64).collect();
        let prior: Vec<_> = grid
            .iter()
            .map(|u| -6. * (std::f64::consts::PI * u / 40.).sin())
            .collect();
        let mid = vec![0.; grid.len()];
        let evidence: Vec<_> = grid.iter().map(|u| *u >= 8. && *u <= 32.).collect();
        // An independently specified visible inward RGB edge lies one pixel
        // below the anatomical prior. It supplies image evidence, never alpha.
        let at = |u: f64, v: f64| {
            let edge = -6. * (std::f64::consts::PI * u / 40.).sin() + 1.;
            let t = 1. / (1. + (-(v - edge) * 4.).exp());
            [80. + 150. * t; 3]
        };
        let result = fit_curve(&at, &grid, &prior, &mid, &evidence, 1., true).unwrap();
        assert_eq!(result[0], prior[0]);
        assert_eq!(result[40], prior[40]);
        assert!(result[20] - prior[20] > 0.8 && result[20] - prior[20] < 1.1);
        assert!(
            result[20] < -4.8,
            "correct displacement rather than smooth away the anatomical curve"
        );
        assert!(result
            .iter()
            .zip(&prior)
            .all(|(a, b)| (a - b).abs() <= BAND));
    }

    #[test]
    fn bad_source_or_unobserved_geometry_is_rejected_without_sampling() {
        let (mut source, mut points) = fixture();
        source.pixels.pop();
        assert!(localize_eye_sources(&source, &points).is_err());
        let (mut source, _) = fixture();
        source.pixels[3] = 0;
        assert!(localize_eye_sources(&source, &points).is_err());
        let (source, _) = fixture();
        points[398][0] = -0.1;
        assert!(localize_eye_sources(&source, &points).is_err());
        let (_, mut points) = fixture();
        points[33] = points[133];
        assert!(localize_eye_sources(&source, &points).is_err());
        let (_, mut points) = fixture();
        for id in EYES[0].ring {
            points[id] = points[468];
        }
        assert!(localize_eye_sources(&source, &points).is_err());
        let (_, mut points) = fixture();
        points[0][0] = f64::NAN;
        assert!(localize_eye_sources(&source, &points).is_err());
        assert!(localize_eye_sources(&source, &points[..477]).is_err());
    }
}
