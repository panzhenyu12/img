//! Source-only eye material preparation and once-only layer transport.
//! A prepared edit owns the decoded source. No target image or alpha is input.
use crate::{eye_material, ImageError, RgbaImage};
use serde::{Deserialize, Serialize};

fn invalid(message: &str) -> ImageError {
    ImageError::InvalidOperation(message.into())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EyeSourceGeometry {
    pub eye: String,
    /// Integer pixel-index coordinates; physical detector points subtract 0.5 once.
    pub origin: [f64; 2],
    pub axis: [f64; 2],
    pub width: f64,
    pub crop: [usize; 4],
    pub knots: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EyeMaterialPoint {
    pub x: f64,
    pub y: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EyeMaterialRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EyeMaterialStroke {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "shape", rename_all = "lowercase", deny_unknown_fields)]
pub enum EyeMaterialShape {
    Rect {
        region: EyeMaterialRect,
    },
    Polygon {
        points: Vec<EyeMaterialPoint>,
    },
    Brush {
        strokes: Vec<EyeMaterialStroke>,
    },
    /// A categorical capsule corridor in original pixel-index coordinates.
    /// Radius identifies material support; it is not opacity or feathering.
    Polyline {
        points: Vec<EyeMaterialPoint>,
        radius: f64,
    },
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EyeMaterialLabels {
    #[serde(default)]
    pub ocular_anchors: Vec<EyeMaterialShape>,
    #[serde(default)]
    pub moving_lash_support: Vec<EyeMaterialShape>,
    #[serde(default)]
    pub unknown: Vec<EyeMaterialShape>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeSourceReport {
    pub eye: String,
    pub source_crop: [usize; 4],
    pub source_noise: f64,
    pub ocular_anchor_pixels: usize,
    pub moving_support_pixels: usize,
    pub estimated_moving_pixels: usize,
    pub model_uncertain_pixels: usize,
    pub unknown_pixels: usize,
    pub source_reconstruction_max_error: f64,
}

/// Frozen source estimate. The private owned source prevents cross-image reuse.
pub struct PreparedEyeClosure {
    source: RgbaImage,
    residual: Vec<f64>,
    exposure: Vec<f64>,
    ocular: Vec<f64>,
    support: Vec<bool>,
    unknown: Vec<bool>,
    reports: Vec<EyeSourceReport>,
}

/// One source observation and one material preparation, reused by all strengths.
pub struct PreparedEyeEdit {
    layers: PreparedEyeClosure,
    landmarks: Vec<[f64; 2]>,
    geometries: Vec<EyeSourceGeometry>,
}

impl PreparedEyeEdit {
    pub fn reports(&self) -> &[EyeSourceReport] {
        self.layers.reports()
    }
    pub fn geometries(&self) -> &[EyeSourceGeometry] {
        &self.geometries
    }
    pub fn render(&self, amount: i32) -> Result<RgbaImage, ImageError> {
        Ok(self.render_diagnostic(amount)?.0)
    }
    /// The ordinary result and its actual field, for algorithm inspection.
    /// This never accepts cached fields or estimates source layers a second time.
    pub fn render_diagnostic(
        &self,
        amount: i32,
    ) -> Result<(RgbaImage, crate::EyeClosureField), ImageError> {
        let source = self.layers.source();
        let params = crate::eye_closure_params_from_landmarks(
            source.width,
            source.height,
            &self.landmarks,
            amount,
        )?;
        let field = crate::prepare_eye_closure_field(source, &params)?;
        let image = self.layers.render_field(&field)?;
        Ok((image, field))
    }
}

/// Prepare light closing from decoded original RGB and one full 478-point
/// MediaPipe observation. Normalized landmarks are physical image coordinates.
/// Material labels are optional explicit corrections, not automatic recognition.
pub fn prepare_eye_closure(
    source: &RgbaImage,
    landmarks: &[[f64; 2]],
    materials: &std::collections::BTreeMap<String, EyeMaterialLabels>,
) -> Result<PreparedEyeEdit, ImageError> {
    crate::eye_closure_params_from_landmarks(source.width, source.height, landmarks, 0)?;
    let geometries = crate::eye_localize::localize_eye_sources(source, landmarks)?;
    let layers = prepare_eye_closure_source(source, &geometries, materials)?;
    Ok(PreparedEyeEdit {
        layers,
        landmarks: landmarks.to_vec(),
        geometries,
    })
}

impl PreparedEyeClosure {
    pub fn reports(&self) -> &[EyeSourceReport] {
        &self.reports
    }
    pub fn source(&self) -> &RgbaImage {
        &self.source
    }

    /// Render an explicitly prepared closing field without another source fit.
    /// The eye ball stays at x; the premultiplied lid/lash moves exactly once.
    pub fn render_field(&self, field: &crate::EyeClosureField) -> Result<RgbaImage, ImageError> {
        let w = self.source.width as usize;
        let h = self.source.height as usize;
        if field.width != self.source.width
            || field.height != self.source.height
            || field.inverse.len() != w * h
            || field.stationary.len() != w * h
        {
            return Err(invalid("eye field and prepared source dimensions differ"));
        }
        let mut output = self.source.clone();
        for i in 0..w * h {
            let x = i % w;
            let y = i / w;
            let q = [field.inverse[i][0] as f64, field.inverse[i][1] as f64];
            if q.iter().any(|v| !v.is_finite())
                || q[0] < 0.
                || q[1] < 0.
                || q[0] > (w - 1) as f64
                || q[1] > (h - 1) as f64
            {
                return Err(invalid(
                    "eye inverse field leaves source pixel-center domain",
                ));
            }
            if field.stationary[i] || q == [x as f64, y as f64] {
                continue;
            }
            let footprint = bilinear_footprint(w, h, q);
            if self.unknown[i] || footprint.iter().any(|(j, a)| *a > 0. && self.unknown[*j]) {
                return Err(invalid("eye edit depends on unknown source material"));
            }
            let e: f64 = footprint.iter().map(|(j, a)| self.exposure[*j] * a).sum();
            if e > 1e-8 && !self.support[i] {
                return Err(invalid(
                    "eye transport requires unsupported ocular underlay",
                ));
            }
            for c in 0..3 {
                let value: f64 = footprint
                    .iter()
                    .map(|(j, a)| self.residual[3 * j + c] * a)
                    .sum::<f64>()
                    + e * self.ocular[3 * i + c];
                if !value.is_finite() || !(-1e-6..=255. + 1e-6).contains(&value) {
                    return Err(invalid(
                        "eye layer transport produces out-of-range RGB; no output clipping",
                    ));
                }
                output.pixels[4 * i + c] = value.round() as u8;
            }
        }
        Ok(output)
    }
}

/// Prepare source layers from localized source curves and categorical labels.
/// This is an intermediate algorithm API; curves carry geometry, never alpha.
pub fn prepare_eye_closure_source(
    source: &RgbaImage,
    geometries: &[EyeSourceGeometry],
    materials: &std::collections::BTreeMap<String, EyeMaterialLabels>,
) -> Result<PreparedEyeClosure, ImageError> {
    let (w, h) = (source.width as usize, source.height as usize);
    if w < 2
        || h < 2
        || w.checked_mul(h).and_then(|n| n.checked_mul(4)) != Some(source.pixels.len())
    {
        return Err(ImageError::InvalidDimensions);
    }
    if source.pixels.chunks_exact(4).any(|p| p[3] != 255) {
        return Err(invalid("eye closure requires an opaque source"));
    }
    let mut names = std::collections::BTreeSet::new();
    for g in geometries {
        if !["leftEye", "rightEye"].contains(&g.eye.as_str()) || !names.insert(g.eye.clone()) {
            return Err(invalid("eye source geometry requires distinct named eyes"));
        }
    }
    if geometries.is_empty() || materials.keys().any(|k| !names.contains(k)) {
        return Err(invalid("material eye has no matching source geometry"));
    }
    let mut rgb: Vec<_> = source
        .pixels
        .chunks_exact(4)
        .flat_map(|p| p[..3].iter().map(|c| *c as f64))
        .collect();
    let mut prepared = PreparedEyeClosure {
        source: source.clone(),
        residual: Vec::new(),
        exposure: vec![0.; w * h],
        ocular: vec![0.; w * h * 3],
        support: vec![false; w * h],
        unknown: vec![false; w * h],
        reports: Vec::new(),
    };
    let mut label_owner = vec![false; w * h];
    for g in geometries {
        validate_geometry(g, w, h)?;
        let labels = materials.get(&g.eye).cloned().unwrap_or_default();
        if materials.contains_key(&g.eye)
            && labels.ocular_anchors.is_empty()
            && labels.moving_lash_support.is_empty()
            && labels.unknown.is_empty()
        {
            return Err(invalid("explicit source material correction is empty"));
        }
        let full = [
            rasterize(&labels.ocular_anchors, w, h)?,
            rasterize(&labels.moving_lash_support, w, h)?,
            rasterize(&labels.unknown, w, h)?,
        ];
        let [x0, y0, x1, y1] = g.crop;
        for i in 0..w * h {
            let count = full.iter().filter(|mask| mask[i]).count();
            if count > 1 || (count > 0 && label_owner[i]) {
                return Err(invalid("eye material labels conflict"));
            }
            if count > 0 {
                if i % w < x0 || i % w >= x1 || i / w < y0 || i / w >= y1 {
                    return Err(invalid("eye material label leaves prepared source crop"));
                }
                label_owner[i] = true;
            }
        }
        let cw = x1 - x0;
        let ch = y1 - y0;
        let indices: Vec<_> = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| y * w + x))
            .collect();
        let crop: Vec<_> = indices
            .iter()
            .flat_map(|i| rgb[3 * i..3 * i + 3].iter().copied())
            .collect();
        let upper = Pchip::new(&g.knots, &g.upper)?;
        let lower = Pchip::new(&g.knots, &g.lower)?;
        let mut base = estimate_normal_layers(&crop, cw, ch, g, &upper, &lower);
        let local: Vec<Vec<bool>> = full
            .iter()
            .map(|mask| indices.iter().map(|i| mask[*i]).collect())
            .collect();
        let (anchors, moving, unknown) = (&local[0], &local[1], &local[2]);
        if (0..cw * ch).any(|i| anchors[i] && !base.domain[i]) {
            return Err(invalid(
                "ocular anchor leaves source eye aperture; correct source geometry first",
            ));
        }
        let mut unknown = unknown.clone();
        for i in 0..cw * ch {
            if anchors[i] {
                base.ocular[3 * i..3 * i + 3].copy_from_slice(&crop[3 * i..3 * i + 3]);
                base.exposure[i] = 1.;
                base.support[i] = true;
            }
        }
        let mut estimated_count = 0;
        let mut uncertain_count = 0;
        if moving.iter().any(|v| *v) {
            let layers =
                eye_material::estimate_explicit_materials(&eye_material::EyeMaterialInput {
                    width: cw,
                    height: ch,
                    source: &crop,
                    moving,
                    donors: anchors,
                    domain: &base.domain,
                    skin_prior: &base.skin,
                    noise: base.noise,
                    radius: 6.,
                })?;
            for i in 0..cw * ch {
                if layers.estimated[i] {
                    base.ocular[3 * i..3 * i + 3].copy_from_slice(&layers.ocular[3 * i..3 * i + 3]);
                    base.exposure[i] = layers.exposure[i];
                    base.support[i] = true;
                    estimated_count += 1;
                }
                uncertain_count += usize::from(layers.model_uncertain[i]);
                unknown[i] |= layers.unknown[i];
            }
        }
        let mut reconstruction_error = 0f64;
        for (i, global) in indices.iter().enumerate() {
            if unknown[i] {
                base.support[i] = false;
                prepared.unknown[*global] = true;
            }
            if !base.support[i] {
                continue;
            }
            if prepared.support[*global] {
                return Err(invalid("source eye material supports overlap"));
            }
            let e = base.exposure[i];
            for c in 0..3 {
                let b = base.ocular[3 * i + c];
                let residual = crop[3 * i + c] - e * b;
                if !b.is_finite()
                    || !(0.0..=255.0).contains(&b)
                    || !e.is_finite()
                    || !(0.0..=1.0).contains(&e)
                    || residual < -1e-6
                    || residual > 255. * (1. - e) + 1e-6
                {
                    return Err(invalid("invalid bounded source eye decomposition"));
                }
                reconstruction_error =
                    reconstruction_error.max((residual + e * b - crop[3 * i + c]).abs());
            }
            prepared.support[*global] = true;
            prepared.exposure[*global] = e;
            prepared.ocular[3 * global..3 * global + 3]
                .copy_from_slice(&base.ocular[3 * i..3 * i + 3]);
        }
        prepared.reports.push(EyeSourceReport {
            eye: g.eye.clone(),
            source_crop: g.crop,
            source_noise: base.noise,
            ocular_anchor_pixels: anchors.iter().filter(|v| **v).count(),
            moving_support_pixels: moving.iter().filter(|v| **v).count(),
            estimated_moving_pixels: estimated_count,
            model_uncertain_pixels: uncertain_count,
            unknown_pixels: unknown.iter().filter(|v| **v).count(),
            source_reconstruction_max_error: reconstruction_error,
        });
    }
    // R is source-only as well: retain it once instead of allocating and
    // decomposing the full photograph on every slider movement.
    for (i, c) in rgb.iter_mut().enumerate() {
        *c -= prepared.exposure[i / 3] * prepared.ocular[i];
    }
    prepared.residual = rgb;
    Ok(prepared)
}

struct SourceLayers {
    ocular: Vec<f64>,
    exposure: Vec<f64>,
    support: Vec<bool>,
    domain: Vec<bool>,
    skin: Vec<f64>,
    noise: f64,
}
fn estimate_normal_layers(
    rgb: &[f64],
    w: usize,
    h: usize,
    g: &EyeSourceGeometry,
    upper: &Pchip,
    lower: &Pchip,
) -> SourceLayers {
    let n = w * h;
    let band = 3.;
    let noise = source_noise(
        &rgb.chunks_exact(3)
            .map(|p| luma([p[0], p[1], p[2]]))
            .collect::<Vec<_>>(),
        w,
        h,
    );
    let mut ocular = vec![0.; 3 * n];
    let mut raw = vec![0.; 3 * n];
    let mut skin = vec![0.; 3 * n];
    let mut exposure = vec![0.; n];
    let mut support = vec![false; n];
    let mut domain = vec![false; n];
    let to_pixel = |u: f64, v: f64| {
        [
            g.origin[0] + g.axis[0] * u - g.axis[1] * v - g.crop[0] as f64,
            g.origin[1] + g.axis[1] * u + g.axis[0] * v - g.crop[1] as f64,
        ]
    };
    for i in 0..n {
        let p = [
            (i % w + g.crop[0]) as f64 - g.origin[0],
            (i / w + g.crop[1]) as f64 - g.origin[1],
        ];
        let u = p[0] * g.axis[0] + p[1] * g.axis[1];
        let v = -p[0] * g.axis[1] + p[1] * g.axis[0];
        let su = u.clamp(0., g.width);
        let top = upper.at(su);
        let bottom = lower.at(su);
        let gap = (bottom - top).max(0.);
        let inside = (0.0..=g.width).contains(&u);
        let upper_side = v - top <= bottom - v;
        let boundary = if upper_side { top } else { bottom };
        let inward = if upper_side { 1. } else { -1. };
        let depth = (v - top).min(bottom - v);
        domain[i] = inside && depth > 0.;
        let mut bs = [[0.; 3]; 3];
        let mut fs = [[0.; 3]; 3];
        for (k, offset) in [0., 0.5, 1.].iter().enumerate() {
            bs[k] = sample_rgb(
                rgb,
                w,
                h,
                to_pixel(
                    su,
                    boundary + inward * ((band + 1f64).min(gap * 0.45) + offset).min(gap * 0.48),
                ),
            );
            fs[k] = sample_rgb(rgb, w, h, to_pixel(su, boundary - inward * (band + offset)));
        }
        let b = std::array::from_fn(|c| median3(bs[0][c], bs[1][c], bs[2][c]));
        let f = std::array::from_fn(|c| median3(fs[0][c], fs[1][c], fs[2][c]));
        let mut lash = [0.; 3];
        let mut darkest = f64::INFINITY;
        for du in [-2., 0., 2.] {
            let au = (su + du).clamp(0., g.width);
            let av = if upper_side {
                upper.at(au)
            } else {
                lower.at(au)
            };
            for dv in [-1., 0., 1.] {
                let color = sample_rgb(rgb, w, h, to_pixel(au, av + inward * dv));
                let light = luma(color);
                if light < darkest {
                    darkest = light;
                    lash = color;
                }
            }
        }
        let observed = [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]];
        let prior = if inside {
            smoothstep((depth + band) / (2. * band))
        } else {
            0.
        };
        let mut e = eye_material::triangle_weights(observed, b, f, lash, prior, noise)[0];
        support[i] = inside && depth > -band;
        if !support[i] {
            e = 0.;
        }
        let mut background = b;
        if inside && depth >= band {
            e = 1.;
            background = observed;
        }
        for c in 0..3 {
            if background[c] > 0. {
                e = e.min(observed[c] / background[c]);
            }
            if background[c] < 255. {
                e = e.min((255. - observed[c]) / (255. - background[c]));
            }
        }
        exposure[i] = e;
        raw[3 * i..3 * i + 3].copy_from_slice(&b);
        skin[3 * i..3 * i + 3].copy_from_slice(&f);
        if support[i] {
            ocular[3 * i..3 * i + 3].copy_from_slice(&background);
        }
    }
    // A fixed source-only 3x3 footprint, before any requested field exists.
    let old = support.clone();
    for i in 0..n {
        let x = i % w;
        let y = i / w;
        if (y.saturating_sub(1)..=(y + 1).min(h - 1)).any(|yy| {
            (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| exposure[yy * w + xx] > 0.)
        }) {
            support[i] = true;
            if !old[i] {
                ocular[3 * i..3 * i + 3].copy_from_slice(&raw[3 * i..3 * i + 3]);
            }
        }
    }
    SourceLayers {
        ocular,
        exposure,
        support,
        domain,
        skin,
        noise,
    }
}

fn validate_geometry(g: &EyeSourceGeometry, w: usize, h: usize) -> Result<(), ImageError> {
    if g.origin.iter().chain(g.axis.iter()).any(|v| !v.is_finite())
        || !g.width.is_finite()
        || g.width <= 2.
        || g.width > (w as f64).hypot(h as f64)
        || (g.axis[0].hypot(g.axis[1]) - 1.).abs() > 1e-8
    {
        return Err(invalid(
            "finite bounded orthonormal source eye frame required",
        ));
    }
    let [x0, y0, x1, y1] = g.crop;
    if x0 >= x1
        || y0 >= y1
        || x1 > w
        || y1 > h
        || g.knots.first() != Some(&0.)
        || g.knots.last() != Some(&g.width)
    {
        return Err(invalid("invalid source eye crop or curve extent"));
    }
    let upper = Pchip::new(&g.knots, &g.upper)?;
    let lower = Pchip::new(&g.knots, &g.lower)?;
    let steps = (g.width * 16.).ceil() as usize;
    for i in 0..=steps {
        let u = g.width * i as f64 / steps as f64;
        let top = upper.at(u);
        let bottom = lower.at(u);
        if !top.is_finite() || !bottom.is_finite() || bottom - top < -1e-7 {
            return Err(invalid("source eyelid curves cross or are nonfinite"));
        }
        for v in [top, bottom, 0.] {
            let x = g.origin[0] + g.axis[0] * u - g.axis[1] * v;
            let y = g.origin[1] + g.axis[1] * u + g.axis[0] * v;
            if !x.is_finite()
                || !y.is_finite()
                || x < x0 as f64
                || y < y0 as f64
                || x > (x1 - 1) as f64
                || y > (y1 - 1) as f64
            {
                return Err(invalid("source eye curves leave the prepared source crop"));
            }
        }
    }
    Ok(())
}

pub(crate) struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
}
impl Pchip {
    pub(crate) fn new(x: &[f64], y: &[f64]) -> Result<Self, ImageError> {
        if x.len() < 2
            || x.len() != y.len()
            || x.iter().chain(y.iter()).any(|v| !v.is_finite())
            || x.windows(2).any(|a| a[0] >= a[1])
        {
            return Err(invalid("finite monotone eye curve knots required"));
        }
        let n = x.len();
        let h: Vec<_> = x.windows(2).map(|a| a[1] - a[0]).collect();
        let m: Vec<_> = y
            .windows(2)
            .zip(&h)
            .map(|(a, h)| (a[1] - a[0]) / h)
            .collect();
        let mut d = vec![0.; n];
        if n == 2 {
            d[0] = m[0];
            d[1] = m[0];
        } else {
            for i in 1..n - 1 {
                if m[i - 1] * m[i] > 0. {
                    let a = 2. * h[i] + h[i - 1];
                    let b = h[i] + 2. * h[i - 1];
                    d[i] = (a + b) / (a / m[i - 1] + b / m[i]);
                }
            }
            let end = |h0: f64, h1: f64, m0: f64, m1: f64| {
                let v = ((2. * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
                if v.signum() != m0.signum() {
                    0.
                } else if m0.signum() != m1.signum() && v.abs() > 3. * m0.abs() {
                    3. * m0
                } else {
                    v
                }
            };
            d[0] = end(h[0], h[1], m[0], m[1]);
            d[n - 1] = end(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
        }
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            d,
        })
    }
    pub(crate) fn at(&self, u: f64) -> f64 {
        let i = self
            .x
            .partition_point(|x| *x <= u)
            .saturating_sub(1)
            .min(self.x.len() - 2);
        let h = self.x[i + 1] - self.x[i];
        let t = (u - self.x[i]) / h;
        (2. * t * t * t - 3. * t * t + 1.) * self.y[i]
            + (t * t * t - 2. * t * t + t) * h * self.d[i]
            + (-2. * t * t * t + 3. * t * t) * self.y[i + 1]
            + (t * t * t - t * t) * h * self.d[i + 1]
    }
}

pub(crate) fn bilinear_footprint(w: usize, h: usize, q: [f64; 2]) -> [(usize, f64); 4] {
    let x = q[0].clamp(0., (w - 1) as f64);
    let y = q[1].clamp(0., (h - 1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    [
        (y0 * w + x0, (1. - dx) * (1. - dy)),
        (y0 * w + x1, dx * (1. - dy)),
        (y1 * w + x0, (1. - dx) * dy),
        (y1 * w + x1, dx * dy),
    ]
}
pub(crate) fn sample_rgb(rgb: &[f64], w: usize, h: usize, q: [f64; 2]) -> [f64; 3] {
    let footprint = bilinear_footprint(w, h, q);
    std::array::from_fn(|c| footprint.iter().map(|(i, a)| rgb[3 * i + c] * a).sum())
}
pub(crate) fn luma(p: [f64; 3]) -> f64 {
    p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722
}
fn smoothstep(v: f64) -> f64 {
    let v = v.clamp(0., 1.);
    v * v * (3. - 2. * v)
}
fn median3(a: f64, b: f64, c: f64) -> f64 {
    if a < b {
        if b < c {
            b
        } else {
            a.max(c)
        }
    } else if a < c {
        a
    } else {
        b.max(c)
    }
}
pub(crate) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    } else {
        values[n / 2]
    }
}
pub(crate) fn source_noise(values: &[f64], w: usize, h: usize) -> f64 {
    let blurred = gaussian_reflect(values, w, h, 0.6);
    let mut errors: Vec<_> = values
        .iter()
        .zip(blurred)
        .map(|(a, b)| (a - b).abs())
        .collect();
    (1.4826 * median(&mut errors)).max(1.)
}
pub(crate) fn gaussian_reflect(values: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let radius = (4. * sigma + 0.5).floor() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }
    let reflect = |mut x: isize, n: usize| {
        while x < 0 || x >= n as isize {
            x = if x < 0 {
                -x - 1
            } else {
                2 * n as isize - x - 1
            };
        }
        x as usize
    };
    let mut tmp = vec![0.; w * h];
    let mut out = vec![0.; w * h];
    for y in 0..h {
        for x in 0..w {
            for (k, a) in kernel.iter().enumerate() {
                tmp[y * w + x] += a * values[reflect(y as isize + k as isize - radius, h) * w + x];
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for (k, a) in kernel.iter().enumerate() {
                out[y * w + x] += a * tmp[y * w + reflect(x as isize + k as isize - radius, w)];
            }
        }
    }
    out
}

fn rasterize(shapes: &[EyeMaterialShape], w: usize, h: usize) -> Result<Vec<bool>, ImageError> {
    let mut mask = vec![false; w * h];
    for shape in shapes {
        let mut part = vec![false; w * h];
        match shape {
            EyeMaterialShape::Polyline { points, radius } => {
                if points.len() < 2
                    || !radius.is_finite()
                    || *radius <= 0.
                    || points.iter().any(|p| {
                        !p.x.is_finite()
                            || !p.y.is_finite()
                            || p.x - radius < 0.
                            || p.y - radius < 0.
                            || p.x + radius > (w - 1) as f64
                            || p.y + radius > (h - 1) as f64
                    })
                    || !points
                        .windows(2)
                        .any(|p| p[0].x != p[1].x || p[0].y != p[1].y)
                {
                    return Err(invalid(
                        "material polyline requires a nonempty bounded source corridor",
                    ));
                }
                for segment in points.windows(2) {
                    let (a, b) = (&segment[0], &segment[1]);
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    let length2 = dx * dx + dy * dy;
                    let x0 = (a.x.min(b.x) - radius).ceil() as usize;
                    let x1 = (a.x.max(b.x) + radius).floor() as usize;
                    let y0 = (a.y.min(b.y) - radius).ceil() as usize;
                    let y1 = (a.y.max(b.y) + radius).floor() as usize;
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let t = if length2 > 0. {
                                (((x as f64 - a.x) * dx + (y as f64 - a.y) * dy) / length2)
                                    .clamp(0., 1.)
                            } else {
                                0.
                            };
                            part[y * w + x] |= (x as f64 - a.x - t * dx).powi(2)
                                + (y as f64 - a.y - t * dy).powi(2)
                                <= radius * radius;
                        }
                    }
                }
            }
            EyeMaterialShape::Rect { region: r } => {
                if [r.x, r.y, r.width, r.height].iter().any(|v| !v.is_finite())
                    || r.x < 0.
                    || r.y < 0.
                    || r.width <= 0.
                    || r.height <= 0.
                    || r.x + r.width > w as f64
                    || r.y + r.height > h as f64
                {
                    return Err(invalid("material rectangle leaves source"));
                }
                for y in 0..h {
                    for x in 0..w {
                        part[y * w + x] = (x as f64) >= r.x
                            && (x as f64) < r.x + r.width
                            && (y as f64) >= r.y
                            && (y as f64) < r.y + r.height;
                    }
                }
            }
            EyeMaterialShape::Brush { strokes } => {
                if strokes.is_empty() {
                    return Err(invalid("material brush requires strokes"));
                }
                for s in strokes {
                    if [s.x, s.y, s.radius].iter().any(|v| !v.is_finite())
                        || s.radius <= 0.
                        || s.x - s.radius < 0.
                        || s.y - s.radius < 0.
                        || s.x + s.radius > (w - 1) as f64
                        || s.y + s.radius > (h - 1) as f64
                    {
                        return Err(invalid("material brush leaves source"));
                    }
                    for y in 0..h {
                        for x in 0..w {
                            part[y * w + x] |= (x as f64 - s.x).powi(2) + (y as f64 - s.y).powi(2)
                                <= s.radius * s.radius;
                        }
                    }
                }
            }
            EyeMaterialShape::Polygon { points } => {
                if points.len() < 3
                    || points.iter().any(|p| {
                        !p.x.is_finite()
                            || !p.y.is_finite()
                            || p.x < 0.
                            || p.y < 0.
                            || p.x > (w - 1) as f64
                            || p.y > (h - 1) as f64
                    })
                {
                    return Err(invalid("material polygon requires bounded source points"));
                }
                let area: f64 = (0..points.len())
                    .map(|i| {
                        let a = &points[i];
                        let b = &points[(i + 1) % points.len()];
                        a.x * b.y - a.y * b.x
                    })
                    .sum();
                if area.abs() < 1e-8 {
                    return Err(invalid("material polygon has no area"));
                }
                for y in 0..h {
                    for x in 0..w {
                        let (x, y) = (x as f64, y as f64);
                        let mut inside = false;
                        let mut boundary = false;
                        for i in 0..points.len() {
                            let a = &points[i];
                            let b = &points[(i + 1) % points.len()];
                            if (a.y > y) != (b.y > y)
                                && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x
                            {
                                inside = !inside;
                            }
                            let dx = b.x - a.x;
                            let dy = b.y - a.y;
                            let denom = dx * dx + dy * dy;
                            if denom > 0. {
                                let t = (((x - a.x) * dx + (y - a.y) * dy) / denom).clamp(0., 1.);
                                boundary |= (x - a.x - t * dx).powi(2) + (y - a.y - t * dy).powi(2)
                                    <= 1e-16;
                            }
                        }
                        part[y as usize * w + x as usize] = inside || boundary;
                    }
                }
            }
        }
        if !part.iter().any(|v| *v) {
            return Err(invalid("material shape selects no source pixels"));
        }
        for (a, b) in mask.iter_mut().zip(part) {
            *a |= b;
        }
    }
    Ok(mask)
}
