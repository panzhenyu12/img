//! Source-only layers for explicitly identified fine lashes and lid mixtures.
//!
//! Material labels are categorical, not optical alpha. This module receives
//! source RGB, visible ocular donors and a source-derived skin proposal. It
//! never receives edited pixels, an inverse field, true alpha or a hidden
//! ocular plate. Only `estimated` moving pixels contain usable inferred layers.
//! Color-model uncertainty does not revoke an explicitly supplied identity;
//! missing source support and invalid physical layers remain `unknown`.

use crate::ImageError;
use nalgebra::{DMatrix, Matrix3, Vector3};
use serde::Serialize;
use std::collections::VecDeque;

const HUBER_CONSTANT: f64 = 1.345;
const IRLS_UPDATES: usize = 8;
const MIN_DONORS: usize = 6;
const MAX_DESIGN_CONDITION: f64 = 1_000.0;
const PHYSICAL_TOLERANCE: f64 = 1e-10;

/// RGB arrays are interleaved, row-major f64 in 0..255. Masks contain one bool
/// per source pixel. `domain` licenses estimation, not donor visibility.
#[derive(Debug, Clone, Copy)]
pub struct EyeMaterialInput<'a> {
    pub width: usize,
    pub height: usize,
    pub source: &'a [f64],
    pub moving: &'a [bool],
    pub donors: &'a [bool],
    pub domain: &'a [bool],
    pub skin_prior: &'a [f64],
    pub noise: f64,
    pub radius: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFitDiagnostic {
    pub pixel: [usize; 2],
    pub donor_count: usize,
    pub residual_mad_scale: f64,
    pub prediction_scale: f64,
    pub prediction_leverage: f64,
    pub source_background: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PigmentDiagnostic {
    pub component: usize,
    pub pixels: usize,
    pub source_pigment_pixel: [usize; 2],
    pub source_pigment_rgb: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeMaterialDiagnostics {
    pub method: &'static str,
    pub radius_source_pixels: f64,
    pub source_noise: f64,
    pub huber_constant: f64,
    pub irls_iterations: usize,
    pub explicit_moving_pixels: usize,
    pub explicit_ocular_donor_pixels: usize,
    pub fitted_pixels: usize,
    pub estimated_pixels: usize,
    pub unknown_pixels: usize,
    pub outside_domain_pixels: usize,
    pub model_uncertain_pixels: usize,
    pub finite_physical_failure_pixels: usize,
    pub feasibility_adjusted_pixels: usize,
    pub source_reconstruction_max_error: f64,
    pub source_foreground_residual_minimum: Option<f64>,
    pub source_foreground_residual_upper_violation: Option<f64>,
    pub explicit_material_identity_required: bool,
    pub automatic_material_permission: bool,
    pub hidden_ocular_is_observed: bool,
    pub output_clipping: bool,
    pub source_surface_fit_records: Vec<SourceFitDiagnostic>,
    pub source_pigment_records: Vec<PigmentDiagnostic>,
}

/// Output arrays use the input layout. Only `estimated` moving pixels may
/// overwrite a prepared source layer. Other pixels are diagnostic proposals
/// or zero placeholders; they are not an automatically prepared whole eye.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeMaterialLayers {
    pub ocular: Vec<f64>,
    pub exposure: Vec<f64>,
    pub residual: Vec<f64>,
    pub estimated: Vec<bool>,
    pub unknown: Vec<bool>,
    pub model_uncertain: Vec<bool>,
    pub fit_error: Vec<f64>,
    pub prediction_scale: Vec<f64>,
    pub pigment: Vec<f64>,
    pub foreground_prior: Vec<f64>,
    /// Three coefficients per pixel in ocular / skin / lash order. These are
    /// inferred appearance coefficients, never accepted as material labels.
    pub material_weights: Vec<f64>,
    pub diagnostics: EyeMaterialDiagnostics,
}

fn invalid(message: &str) -> ImageError {
    ImageError::InvalidOperation(message.to_owned())
}

fn validate(input: &EyeMaterialInput<'_>) -> Result<usize, ImageError> {
    let count = input.width.checked_mul(input.height);
    let rgb_count = count.and_then(|n| n.checked_mul(3));
    let Some(count) = count.filter(|&n| n > 0) else {
        return Err(ImageError::InvalidDimensions);
    };
    if input.width == 0
        || input.height == 0
        || rgb_count != Some(input.source.len())
        || rgb_count != Some(input.skin_prior.len())
        || input.moving.len() != count
        || input.donors.len() != count
        || input.domain.len() != count
    {
        return Err(ImageError::InvalidDimensions);
    }
    if input
        .source
        .iter()
        .chain(input.skin_prior)
        .any(|v| !v.is_finite() || !(0.0..=255.0).contains(v))
    {
        return Err(invalid(
            "eye material source and skin proposal require finite RGB in 0..255",
        ));
    }
    if !input.noise.is_finite()
        || input.noise <= 0.0
        || !input.radius.is_finite()
        || input.radius <= 0.0
    {
        return Err(invalid(
            "eye material noise and radius must be positive and finite",
        ));
    }
    for i in 0..count {
        if input.moving[i] && input.donors[i] {
            return Err(invalid(
                "moving source material conflicts with visible ocular donors",
            ));
        }
        if input.donors[i] && !input.domain[i] {
            return Err(invalid(
                "visible ocular donors leave the permitted source domain",
            ));
        }
    }
    Ok(count)
}

fn rgb(data: &[f64], index: usize) -> [f64; 3] {
    [data[index * 3], data[index * 3 + 1], data[index * 3 + 2]]
}

fn put_rgb(data: &mut [f64], index: usize, value: [f64; 3]) {
    data[index * 3..index * 3 + 3].copy_from_slice(&value);
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

/// Weighted source-only affine RGB prediction. There is an initial fit and
/// exactly eight Huber reweightings, matching the frozen Python estimator.
fn fit_surface(input: &EyeMaterialInput<'_>, x: usize, y: usize) -> Option<SourceFitDiagnostic> {
    let x0 = ((x as f64 - input.radius).floor().max(0.0) as usize).min(input.width);
    let x1 = ((x as f64 + input.radius).ceil() as usize)
        .saturating_add(1)
        .min(input.width);
    let y0 = ((y as f64 - input.radius).floor().max(0.0) as usize).min(input.height);
    let y1 = ((y as f64 + input.radius).ceil() as usize)
        .saturating_add(1)
        .min(input.height);
    let mut rows = Vec::<[f64; 3]>::new();
    let mut colors = Vec::<[f64; 3]>::new();
    let mut spatial = Vec::new();
    for sy in y0..y1 {
        for sx in x0..x1 {
            let index = sy * input.width + sx;
            let dx = (sx as f64 - x as f64) / input.radius;
            let dy = (sy as f64 - y as f64) / input.radius;
            let r2 = dx * dx + dy * dy;
            if input.donors[index] && r2 < 1.0 {
                rows.push([1.0, dx, dy]);
                colors.push(rgb(input.source, index));
                spatial.push((1.0 - r2).powi(2));
            }
        }
    }
    let count = rows.len();
    if count < MIN_DONORS {
        return None;
    }
    let mut weights = spatial.clone();
    let mut sigma = input.noise;
    let mut coefficients = DMatrix::zeros(3, 3);
    for iteration in 0..=IRLS_UPDATES {
        let mut design = DMatrix::zeros(count, 3);
        let mut observed = DMatrix::zeros(count, 3);
        for i in 0..count {
            let root_weight = weights[i].sqrt();
            for c in 0..3 {
                design[(i, c)] = rows[i][c] * root_weight;
                observed[(i, c)] = colors[i][c] * root_weight;
            }
        }
        let svd = design.svd(true, true);
        let minimum = svd
            .singular_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let maximum = svd.singular_values.iter().copied().fold(0.0, f64::max);
        if !minimum.is_finite() || minimum <= 0.0 || maximum / minimum > MAX_DESIGN_CONDITION {
            return None;
        }
        let tolerance = f64::EPSILON * count.max(3) as f64 * maximum;
        coefficients = svd.solve(&observed, tolerance).ok()?;
        if coefficients.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut residuals = vec![[0.0; 3]; count];
        let mut channel_centers = [0.0; 3];
        for c in 0..3 {
            let mut channel = Vec::with_capacity(count);
            for i in 0..count {
                let predicted = (0..3)
                    .map(|k| rows[i][k] * coefficients[(k, c)])
                    .sum::<f64>();
                residuals[i][c] = colors[i][c] - predicted;
                channel.push(residuals[i][c]);
            }
            channel_centers[c] = median(&mut channel);
        }
        let mut deviations = Vec::with_capacity(count * 3);
        for residual in &residuals {
            for c in 0..3 {
                deviations.push((residual[c] - channel_centers[c]).abs());
            }
        }
        sigma = input.noise.max(1.4826 * median(&mut deviations));
        if iteration < IRLS_UPDATES {
            for i in 0..count {
                let rms = (residuals[i].iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
                weights[i] = spatial[i] * (HUBER_CONSTANT * sigma / rms.max(1e-20)).min(1.0);
            }
        }
    }
    let background = [
        coefficients[(0, 0)],
        coefficients[(0, 1)],
        coefficients[(0, 2)],
    ];
    if background
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=255.0).contains(v))
    {
        return None;
    }
    let mut gram = Matrix3::<f64>::zeros();
    for (row, weight) in rows.iter().zip(&weights) {
        let row = Vector3::from_row_slice(row);
        gram += (row * row.transpose()) * *weight;
    }
    let inverse = gram.try_inverse()?;
    let leverage = inverse[(0, 0)];
    let prediction_scale = sigma * (1.0 + leverage).sqrt();
    if !leverage.is_finite() || leverage < 0.0 || !prediction_scale.is_finite() {
        return None;
    }
    Some(SourceFitDiagnostic {
        pixel: [x, y],
        donor_count: count,
        residual_mad_scale: sigma,
        prediction_scale,
        prediction_leverage: leverage,
        source_background: background,
    })
}

fn source_pigments(input: &EyeMaterialInput<'_>, output: &mut [f64]) -> Vec<PigmentDiagnostic> {
    let mut visited = vec![false; input.moving.len()];
    let mut records = Vec::new();
    for seed in 0..visited.len() {
        if visited[seed] || !input.moving[seed] {
            continue;
        }
        visited[seed] = true;
        let mut queue = VecDeque::from([seed]);
        let mut members = Vec::new();
        while let Some(index) = queue.pop_front() {
            members.push(index);
            let x = index % input.width;
            let y = index / input.width;
            for sy in y.saturating_sub(1)..=(y + 1).min(input.height - 1) {
                for sx in x.saturating_sub(1)..=(x + 1).min(input.width - 1) {
                    let neighbor = sy * input.width + sx;
                    if input.moving[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        // scipy labels components in scan order, and np.argmin picks the first
        // scan-order pixel on a tie. BFS traversal order must not change that.
        members.sort_unstable();
        let mut darkest = members[0];
        let luminance = |i| {
            let c = rgb(input.source, i);
            0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        };
        let mut minimum = luminance(darkest);
        for &index in &members[1..] {
            let value = luminance(index);
            if value < minimum {
                darkest = index;
                minimum = value;
            }
        }
        let color = rgb(input.source, darkest);
        for &index in &members {
            put_rgb(output, index, color);
        }
        records.push(PigmentDiagnostic {
            component: records.len() + 1,
            pixels: members.len(),
            source_pigment_pixel: [darkest % input.width, darkest / input.width],
            source_pigment_rgb: color,
        });
    }
    records
}

/// Closest point on the seven faces of the ocular/skin/lash simplex. The
/// optional fourth coordinate is the V1 geometric prior, not an optical fact.
/// Explicit materials call this with prior=0 and noise=0 (pure RGB fit).
pub(crate) fn triangle_weights(
    color: [f64; 3],
    ocular: [f64; 3],
    skin: [f64; 3],
    lash: [f64; 3],
    prior: f64,
    noise: f64,
) -> [f64; 3] {
    let target = [color[0], color[1], color[2], noise * prior];
    let palette = [
        [ocular[0], ocular[1], ocular[2], noise],
        [skin[0], skin[1], skin[2], 0.0],
        [lash[0], lash[1], lash[2], 0.0],
    ];
    let dot = |a: [f64; 4], b: [f64; 4]| (0..4).map(|i| a[i] * b[i]).sum::<f64>();
    let subtract = |a: [f64; 4], b: [f64; 4]| -> [f64; 4] { std::array::from_fn(|i| a[i] - b[i]) };
    let mut choices = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let delta = subtract(palette[j], palette[i]);
        let fraction = (dot(subtract(target, palette[i]), delta) / dot(delta, delta).max(1e-20))
            .clamp(0.0, 1.0);
        let mut weights = [0.0; 3];
        weights[i] = 1.0 - fraction;
        weights[j] = fraction;
        choices.push(weights);
    }
    let a = subtract(palette[1], palette[0]);
    let d = subtract(palette[2], palette[0]);
    let z = subtract(target, palette[0]);
    let aa = dot(a, a);
    let dd = dot(d, d);
    let ad = dot(a, d);
    let az = dot(a, z);
    let dz = dot(d, z);
    let determinant = aa * dd - ad * ad;
    if determinant > 1e-12 {
        let skin_weight = (az * dd - dz * ad) / determinant;
        let lash_weight = (dz * aa - az * ad) / determinant;
        let weights = [1.0 - skin_weight - lash_weight, skin_weight, lash_weight];
        if weights.iter().all(|&v| v >= 0.0) {
            choices.push(weights);
        }
    }
    let mut best = choices[0];
    let mut best_cost = f64::INFINITY;
    for weights in choices {
        let cost = (0..4)
            .map(|c| {
                let value = (0..3).map(|k| weights[k] * palette[k][c]).sum::<f64>() - target[c];
                value * value
            })
            .sum::<f64>();
        if cost < best_cost {
            best = weights;
            best_cost = cost;
        }
    }
    best
}

/// Infer only explicitly marked moving source material. No target image or
/// requested edit is an input. The caller must preserve and enforce `unknown`.
pub fn estimate_explicit_materials(
    input: &EyeMaterialInput<'_>,
) -> Result<EyeMaterialLayers, ImageError> {
    let count = validate(input)?;
    let mut result = EyeMaterialLayers {
        ocular: vec![0.0; count * 3],
        exposure: vec![0.0; count],
        residual: input.source.to_vec(),
        estimated: vec![false; count],
        unknown: input.moving.to_vec(),
        model_uncertain: vec![false; count],
        fit_error: vec![0.0; count],
        prediction_scale: vec![input.noise; count],
        pigment: vec![0.0; count * 3],
        foreground_prior: vec![0.0; count * 3],
        material_weights: vec![0.0; count * 3],
        diagnostics: EyeMaterialDiagnostics {
            method: "explicit-source-ocular-skin-lash-robust-layers-v2",
            radius_source_pixels: input.radius,
            source_noise: input.noise,
            huber_constant: HUBER_CONSTANT,
            irls_iterations: IRLS_UPDATES,
            explicit_moving_pixels: input.moving.iter().filter(|&&v| v).count(),
            explicit_ocular_donor_pixels: input.donors.iter().filter(|&&v| v).count(),
            fitted_pixels: 0,
            estimated_pixels: 0,
            unknown_pixels: 0,
            outside_domain_pixels: (0..count)
                .filter(|&i| input.moving[i] && !input.domain[i])
                .count(),
            model_uncertain_pixels: 0,
            finite_physical_failure_pixels: 0,
            feasibility_adjusted_pixels: 0,
            source_reconstruction_max_error: 0.0,
            source_foreground_residual_minimum: None,
            source_foreground_residual_upper_violation: None,
            explicit_material_identity_required: true,
            automatic_material_permission: false,
            hidden_ocular_is_observed: false,
            output_clipping: false,
            source_surface_fit_records: Vec::new(),
            source_pigment_records: Vec::new(),
        },
    };
    result.diagnostics.source_pigment_records = source_pigments(input, &mut result.pigment);
    for index in 0..count {
        if !input.moving[index] || !input.domain[index] {
            continue;
        }
        let Some(fit) = fit_surface(input, index % input.width, index / input.width) else {
            continue;
        };
        let background = fit.source_background;
        result.prediction_scale[index] = fit.prediction_scale;
        result.diagnostics.source_surface_fit_records.push(fit);
        result.diagnostics.fitted_pixels += 1;
        put_rgb(&mut result.ocular, index, background);
        let observed = rgb(input.source, index);
        let skin = rgb(input.skin_prior, index);
        let lash = rgb(&result.pigment, index);
        let weights = triangle_weights(observed, background, skin, lash, 0.0, 0.0);
        put_rgb(&mut result.material_weights, index, weights);
        let initial_exposure = weights[0];
        let foreground = std::array::from_fn(|c| {
            if 1.0 - initial_exposure > 1e-12 {
                (weights[1] * skin[c] + weights[2] * lash[c]) / (1.0 - initial_exposure)
            } else {
                lash[c]
            }
        });
        put_rgb(&mut result.foreground_prior, index, foreground);
        let mut cap: f64 = 1.0;
        for c in 0..3 {
            if background[c] > 0.0 {
                cap = cap.min(observed[c] / background[c]);
            }
            if background[c] < 255.0 {
                cap = cap.min((255.0 - observed[c]) / (255.0 - background[c]));
            }
        }
        let exposure = initial_exposure.min(cap);
        let residual = std::array::from_fn(|c| observed[c] - exposure * background[c]);
        let reconstruction_error = (0..3)
            .map(|c| (residual[c] + exposure * background[c] - observed[c]).abs())
            .fold(0.0, f64::max);
        result.exposure[index] = exposure;
        put_rgb(&mut result.residual, index, residual);
        let finite = exposure.is_finite()
            && residual
                .iter()
                .chain(background.iter())
                .all(|v| v.is_finite());
        let bounded = (0.0..=1.0).contains(&exposure)
            && background.iter().all(|v| (0.0..=255.0).contains(v))
            && residual.iter().all(|&v| {
                v >= -PHYSICAL_TOLERANCE && v <= 255.0 * (1.0 - exposure) + PHYSICAL_TOLERANCE
            });
        let reconstructs = reconstruction_error <= PHYSICAL_TOLERANCE;
        if !(finite && bounded && reconstructs) {
            result.diagnostics.finite_physical_failure_pixels += 1;
            continue;
        }
        result.estimated[index] = true;
        result.unknown[index] = false;
        let error = ((0..3)
            .map(|c| {
                let model = exposure * background[c] + (1.0 - exposure) * foreground[c];
                (observed[c] - model).powi(2)
            })
            .sum::<f64>()
            / 3.0)
            .sqrt();
        result.fit_error[index] = error;
        result.model_uncertain[index] = error > result.prediction_scale[index];
        result.diagnostics.estimated_pixels += 1;
        result.diagnostics.model_uncertain_pixels += usize::from(result.model_uncertain[index]);
        result.diagnostics.feasibility_adjusted_pixels +=
            usize::from(initial_exposure > exposure + 1e-12);
        result.diagnostics.source_reconstruction_max_error = result
            .diagnostics
            .source_reconstruction_max_error
            .max(reconstruction_error);
        let minimum = residual.iter().copied().fold(f64::INFINITY, f64::min);
        let violation = residual
            .iter()
            .map(|&v| v - 255.0 * (1.0 - exposure))
            .fold(f64::NEG_INFINITY, f64::max);
        result.diagnostics.source_foreground_residual_minimum = Some(
            result
                .diagnostics
                .source_foreground_residual_minimum
                .unwrap_or(f64::INFINITY)
                .min(minimum),
        );
        result
            .diagnostics
            .source_foreground_residual_upper_violation = Some(
            result
                .diagnostics
                .source_foreground_residual_upper_violation
                .unwrap_or(f64::NEG_INFINITY)
                .max(violation),
        );
    }
    result.diagnostics.unknown_pixels = result.unknown.iter().filter(|&&v| v).count();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_sample(data: &[f64], width: usize, height: usize, point: [f64; 2]) -> f64 {
        let [x, y] = point;
        assert!(x >= 0.0 && x <= (width - 1) as f64 && y >= 0.0 && y <= (height - 1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        (1.0 - tx) * (1.0 - ty) * data[y0 * width + x0]
            + tx * (1.0 - ty) * data[y0 * width + x1]
            + (1.0 - tx) * ty * data[y1 * width + x0]
            + tx * ty * data[y1 * width + x1]
    }

    fn color_sample(data: &[f64], width: usize, height: usize, point: [f64; 2]) -> [f64; 3] {
        let [x, y] = point;
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        std::array::from_fn(|c| {
            (1.0 - tx) * (1.0 - ty) * data[(y0 * width + x0) * 3 + c]
                + tx * (1.0 - ty) * data[(y0 * width + x1) * 3 + c]
                + (1.0 - tx) * ty * data[(y1 * width + x0) * 3 + c]
                + tx * ty * data[(y1 * width + x1) * 3 + c]
        })
    }

    struct KnownLash {
        width: usize,
        height: usize,
        source: Vec<f64>,
        skin_proposal: Vec<f64>,
        moving: Vec<bool>,
        donors: Vec<bool>,
        domain: Vec<bool>,
        true_ocular: Vec<f64>,
        true_residual: Vec<f64>,
        true_exposure: Vec<f64>,
        true_aperture: Vec<f64>,
        true_lash: Vec<f64>,
        inverse: Vec<[f64; 2]>,
    }

    impl KnownLash {
        fn input(&self) -> EyeMaterialInput<'_> {
            EyeMaterialInput {
                width: self.width,
                height: self.height,
                source: &self.source,
                moving: &self.moving,
                donors: &self.donors,
                domain: &self.domain,
                skin_prior: &self.skin_proposal,
                noise: 1.0,
                radius: 6.0,
            }
        }

        fn expected(&self, field: &[[f64; 2]]) -> Vec<f64> {
            let mut output = Vec::with_capacity(self.source.len());
            for (i, &point) in field.iter().enumerate() {
                let residual = color_sample(&self.true_residual, self.width, self.height, point);
                let exposure = scalar_sample(&self.true_exposure, self.width, self.height, point);
                output.extend((0..3).map(|c| residual[c] + exposure * self.true_ocular[i * 3 + c]));
            }
            output
        }
    }

    /// Independent known layers generate source RGB and the test oracle.
    /// Estimation receives only that RGB, hand-declared discrete labels, and
    /// a skin proposal sampled from visible source row 8; no true layer input.
    fn known_lash() -> KnownLash {
        let (width, height) = (73, 45);
        let count = width * height;
        let mut f = KnownLash {
            width,
            height,
            source: vec![0.0; count * 3],
            skin_proposal: vec![0.0; count * 3],
            moving: vec![false; count],
            donors: vec![false; count],
            domain: vec![false; count],
            true_ocular: vec![0.0; count * 3],
            true_residual: vec![0.0; count * 3],
            true_exposure: vec![0.0; count],
            true_aperture: vec![0.0; count],
            true_lash: vec![0.0; count],
            inverse: vec![[0.0; 2]; count],
        };
        let upper = |u: f64| -7.25 * (std::f64::consts::PI * u.clamp(0.0, 52.0) / 52.0).sin();
        let lower = |u: f64| 6.75 * (std::f64::consts::PI * u.clamp(0.0, 52.0) / 52.0).sin();
        let a = [19.0, upper(19.0) - 0.6];
        let b = [27.8, -1.7];
        let vector = [b[0] - a[0], b[1] - a[1]];
        let length2 = vector[0] * vector[0] + vector[1] * vector[1];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let u = x as f64 - 10.0;
                let v = y as f64 - 22.0;
                let iris = ((u - 26.0) / 11.0).powi(2) + (v / 9.5).powi(2) < 1.0;
                let ocular = if iris {
                    let variation = (0.7 * v + 0.15 * u).sin();
                    [
                        65.0 + 1.3 * v + 10.0 * variation,
                        85.0 + 0.7 * v + 6.0 * variation,
                        105.0 + 0.8 * v + 8.0 * variation,
                    ]
                } else {
                    [205.0 + 0.35 * v, 218.0 + 0.35 * v, 223.0 + 0.35 * v]
                };
                let skin = [175.0 + 0.03 * u, 124.0 + 0.01 * u, 102.0 + 0.02 * u];
                let mut aperture = 0.0;
                let mut lash = 0.0;
                for sy in 0..8 {
                    for sx in 0..8 {
                        let qu = u + (sx as f64 + 0.5) / 8.0 - 0.5;
                        let qv = v + (sy as f64 + 0.5) / 8.0 - 0.5;
                        let open = qu > 0.0 && qu < 52.0 && qv > upper(qu) && qv < lower(qu);
                        let t = (((qu - a[0]) * vector[0] + (qv - a[1]) * vector[1]) / length2)
                            .clamp(0.0, 1.0);
                        let distance = (qu - a[0] - t * vector[0]).hypot(qv - a[1] - t * vector[1]);
                        aperture += if open { 1.0 / 64.0 } else { 0.0 };
                        lash += if open && distance < 0.65 {
                            1.0 / 64.0
                        } else {
                            0.0
                        };
                    }
                }
                let exposure = aperture - lash;
                let residual = std::array::from_fn(|c| {
                    (1.0 - aperture) * skin[c] + lash * [9.0, 10.0, 13.0][c]
                });
                put_rgb(&mut f.true_ocular, i, ocular);
                put_rgb(&mut f.true_residual, i, residual);
                put_rgb(
                    &mut f.source,
                    i,
                    std::array::from_fn(|c| residual[c] + exposure * ocular[c]),
                );
                f.true_exposure[i] = exposure;
                f.true_aperture[i] = aperture;
                f.true_lash[i] = lash;
                f.domain[i] = (0.0..=52.0).contains(&u) && v > upper(u) && v < lower(u);
                // Literal, deliberately coarse source brush; independent of
                // the generator's segment, pixel coverage and true alpha.
                let dx = x as f64 - 31.0;
                let dy = y as f64 - 16.0;
                let t = ((dx * 7.0 + dy * 4.5) / (49.0 + 20.25)).clamp(0.0, 1.0);
                f.moving[i] = (dx - t * 7.0).hypot(dy - t * 4.5) <= 1.4 && y >= 16;
                f.donors[i] = ((18..27).contains(&y) && (27..31).contains(&x))
                    || ((23..28).contains(&y) && (31..40).contains(&x))
                    || ((17..26).contains(&y) && (40..45).contains(&x))
                    || (y == 16 && (35..41).contains(&x));
                let window = if u > 0.0 && u < 52.0 {
                    (std::f64::consts::PI * u / 52.0).sin().powi(2)
                } else {
                    0.0
                };
                let taper = ((18.0 - v.abs()) / 6.0).clamp(0.0, 1.0);
                f.inverse[i] = [x as f64, y as f64 + 0.18 * window * taper * v];
            }
        }
        for y in 0..height {
            for x in 0..width {
                put_rgb(
                    &mut f.skin_proposal,
                    y * width + x,
                    rgb(&f.source, 8 * width + x),
                );
            }
        }
        f
    }

    #[test]
    fn explicitly_labeled_lash_is_inferred_and_moves_once_in_final_rgb() {
        let f = known_lash();
        let layers = estimate_explicit_materials(&f.input()).unwrap();
        assert_eq!(layers.diagnostics.explicit_moving_pixels, 27);
        assert_eq!(layers.diagnostics.explicit_ocular_donor_pixels, 132);
        assert_eq!(layers.diagnostics.unknown_pixels, 0);
        assert_eq!(layers.diagnostics.estimated_pixels, 27);
        // Isolate material correction from geometry: the wrong baseline holds
        // source ocular RGB stationary. Only the marked source layer is changed.
        let mut ocular = f.source.clone();
        let mut exposure = vec![1.0; f.width * f.height];
        for i in 0..exposure.len() {
            if layers.estimated[i] {
                put_rgb(&mut ocular, i, rgb(&layers.ocular, i));
                exposure[i] = layers.exposure[i];
            }
        }
        let residual: Vec<_> = f
            .source
            .iter()
            .enumerate()
            .map(|(k, &c)| c - exposure[k / 3] * ocular[k])
            .collect();
        let mut output = vec![0.0; f.source.len()];
        for (i, &point) in f.inverse.iter().enumerate() {
            let r = color_sample(&residual, f.width, f.height, point);
            let e = scalar_sample(&exposure, f.width, f.height, point);
            put_rgb(
                &mut output,
                i,
                std::array::from_fn(|c| r[c] + e * ocular[i * 3 + c]),
            );
        }
        let field_y: Vec<_> = f.inverse.iter().map(|p| p[1]).collect();
        let twice_field: Vec<_> = f
            .inverse
            .iter()
            .map(|&p| [p[0], scalar_sample(&field_y, f.width, f.height, p)])
            .collect();
        let expected = f.expected(&f.inverse);
        let twice = f.expected(&twice_field);
        let occupied: Vec<_> = (0..exposure.len())
            .map(|i| {
                f.true_lash[i] > 0.0
                    || scalar_sample(&f.true_lash, f.width, f.height, f.inverse[i]) > 0.0
                    || scalar_sample(&f.true_lash, f.width, f.height, twice_field[i]) > 0.0
            })
            .collect();
        let mut focus = Vec::new();
        for y in 0..f.height {
            for x in 0..f.width {
                let i = y * f.width + x;
                let u = x as f64 - 10.0;
                let v = y as f64 - 22.0;
                let nearby = (y.saturating_sub(1)..=(y + 1).min(f.height - 1)).any(|sy| {
                    (x.saturating_sub(1)..=(x + 1).min(f.width - 1))
                        .any(|sx| occupied[sy * f.width + sx])
                });
                if nearby
                    && ((u - 26.0) / 11.0).powi(2) + (v / 9.5).powi(2) < 1.0
                    && f.true_aperture[i] == 1.0
                    && scalar_sample(&f.true_aperture, f.width, f.height, f.inverse[i]) == 1.0
                    && scalar_sample(&f.true_aperture, f.width, f.height, twice_field[i]) == 1.0
                {
                    focus.push(i);
                }
            }
        }
        assert!(focus.len() > 8);
        let rms = |image: &[f64]| {
            let mut squared = 0.0;
            for &i in &focus {
                for c in 0..3 {
                    squared += (image[i * 3 + c] - expected[i * 3 + c]).powi(2);
                }
            }
            (squared / (focus.len() * 3) as f64).sqrt()
        };
        let reference = rms(&f.source).min(rms(&twice));
        assert!(reference > 5.0);
        assert!(
            rms(&output) < 0.4 * reference,
            "inferred lash RMSE {} vs fixed limit {}",
            rms(&output),
            0.4 * reference
        );
        let centroid = |image: &[f64]| {
            let mut mass = 0.0;
            let mut moment = 0.0;
            for y in 17..=22 {
                let i = y * f.width + 35;
                let b = rgb(&f.true_ocular, i);
                let c = rgb(image, i);
                let contrast: [f64; 3] = std::array::from_fn(|k| b[k] - [9.0, 10.0, 13.0][k]);
                let darkness = ((0..3).map(|k| (b[k] - c[k]) * contrast[k]).sum::<f64>()
                    / contrast.iter().map(|v| v * v).sum::<f64>())
                .clamp(0.0, 1.0);
                mass += darkness;
                moment += darkness * (y as f64 - 22.0);
            }
            assert!(mass > 1.0);
            moment / mass
        };
        assert!((centroid(&output) - centroid(&expected)).abs() < 0.25);
        assert!(layers.diagnostics.source_reconstruction_max_error <= 1e-10);
    }

    #[test]
    fn palette_uncertainty_does_not_revoke_explicit_identity_but_absent_donors_do() {
        let (width, height) = (9, 9);
        let count = width * height;
        let mut source = [100.0, 120.0, 140.0].repeat(count);
        let skin = [180.0, 160.0, 140.0].repeat(count);
        let mut moving = vec![false; count];
        moving[40] = true;
        moving[41] = true;
        put_rgb(&mut source, 40, [250.0, 10.0, 220.0]);
        put_rgb(&mut source, 41, [10.0, 10.0, 10.0]);
        let donors: Vec<_> = moving.iter().map(|v| !v).collect();
        let domain = vec![true; count];
        let mut input = EyeMaterialInput {
            width,
            height,
            source: &source,
            moving: &moving,
            donors: &donors,
            domain: &domain,
            skin_prior: &skin,
            noise: 1.0,
            radius: 6.0,
        };
        let layers = estimate_explicit_materials(&input).unwrap();
        assert!(layers.estimated[40] && !layers.unknown[40] && layers.model_uncertain[40]);
        for c in 0..3 {
            assert!(
                (layers.residual[120 + c] + layers.exposure[40] * layers.ocular[120 + c]
                    - source[120 + c])
                    .abs()
                    <= 1e-10
            );
        }
        let missing = vec![false; count];
        input.donors = &missing;
        let unsupported = estimate_explicit_materials(&input).unwrap();
        assert_eq!(unsupported.diagnostics.estimated_pixels, 0);
        assert_eq!(unsupported.diagnostics.unknown_pixels, 2);
    }

    #[test]
    fn rejects_material_conflicts_out_of_domain_donors_and_invalid_rgb() {
        let source = vec![120.0; 27];
        let mut moving = vec![false; 9];
        moving[4] = true;
        let donors = vec![true; 9];
        let domain = vec![true; 9];
        let input = EyeMaterialInput {
            width: 3,
            height: 3,
            source: &source,
            moving: &moving,
            donors: &donors,
            domain: &domain,
            skin_prior: &source,
            noise: 1.0,
            radius: 6.0,
        };
        assert!(estimate_explicit_materials(&input).is_err());
        let empty = vec![false; 9];
        let wrong_domain = vec![false; 9];
        assert!(estimate_explicit_materials(&EyeMaterialInput {
            moving: &empty,
            domain: &wrong_domain,
            ..input
        })
        .is_err());
        let mut invalid_rgb = source.clone();
        invalid_rgb[0] = 256.0;
        assert!(estimate_explicit_materials(&EyeMaterialInput {
            source: &invalid_rgb,
            donors: &empty,
            ..input
        })
        .is_err());
        assert!(estimate_explicit_materials(&EyeMaterialInput {
            width: usize::MAX,
            ..input
        })
        .is_err());
    }

    #[test]
    fn three_color_simplex_and_optional_prior_use_the_same_palette_contract() {
        let ocular = [20.0, 200.0, 80.0];
        let skin = [220.0, 150.0, 100.0];
        let lash = [5.0, 5.0, 10.0];
        let expected = [0.3, 0.25, 0.45];
        let color = std::array::from_fn(|c| {
            expected[0] * ocular[c] + expected[1] * skin[c] + expected[2] * lash[c]
        });
        let weights = triangle_weights(color, ocular, skin, lash, 0.0, 0.0);
        for i in 0..3 {
            assert!((weights[i] - expected[i]).abs() < 1e-12);
        }
        let same = [100.0; 3];
        let weights = triangle_weights(same, same, same, same, 0.7, 4.0);
        assert!((weights[0] - 0.7).abs() < 1e-12);
    }
}
