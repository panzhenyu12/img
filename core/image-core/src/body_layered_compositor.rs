//! Frozen stationary foreground over an independently transported rear layer.
//!
//! Source preparation supplies bounded floating-point colors, already paired
//! with this opaque photograph. This module performs neither segmentation nor
//! hidden-content generation, and does not infer anatomical motion from alpha.

use crate::{ImageError, RgbaImage};

/// A prepared two-material source, in the photograph's encoded RGB values.
/// RGB slices are interleaved, three f64 values per pixel in [0,255]. Coverage
/// slices contain one f64 value per pixel in [0,1]. No intermediate RGBA8 layer
/// is introduced: rounding a mixed-edge underlay would break source identity.
///
/// `rear_source_rgb` is the complete lower image before motion, including its
/// own stationary scene contribution. The observed source must reconstruct as
/// `front_alpha*front_rgb + (1-front_alpha)*rear_source_rgb`. The lower image
/// must also admit a bounded foreground at `rear_alpha` over `scene_rgb`.
pub struct BodyLayeredSource<'a> {
    pub width: u32,
    pub height: u32,
    pub front_alpha: &'a [f64],
    pub front_rgb: &'a [f64],
    pub rear_alpha: &'a [f64],
    pub rear_source_rgb: &'a [f64],
    pub scene_rgb: &'a [f64],
}

fn invalid(message: &str) -> ImageError {
    ImageError::InvalidOperation(message.into())
}

// Bounds cover only accumulated IEEE-754 arithmetic error in the displayed
// products/sums. They are many orders below one RGB code value, and never
// permit a mismatched source, adjust alpha, or repair a substantive RGB error.
fn arithmetic_error(scale: f64) -> f64 {
    (16.0 * f64::EPSILON / (1.0 - 16.0 * f64::EPSILON)) * scale
}

impl BodyLayeredSource<'_> {
    pub(crate) fn validate_source(&self, source: &RgbaImage) -> Result<usize, ImageError> {
        let count = (source.width as usize)
            .checked_mul(source.height as usize)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("layered body source dimensions are invalid"))?;
        let rgb_count = count
            .checked_mul(3)
            .ok_or_else(|| invalid("layered body source dimensions are invalid"))?;
        let rgba_count = count
            .checked_mul(4)
            .ok_or_else(|| invalid("layered body source dimensions are invalid"))?;
        if self.width != source.width
            || self.height != source.height
            || self.front_alpha.len() != count
            || self.rear_alpha.len() != count
            || self.front_rgb.len() != rgb_count
            || self.rear_source_rgb.len() != rgb_count
            || self.scene_rgb.len() != rgb_count
            || source.pixels.len() != rgba_count
        {
            return Err(invalid(
                "layered body arrays must match the source dimensions",
            ));
        }
        if source.pixels.chunks_exact(4).any(|p| p[3] != 255) {
            return Err(invalid(
                "layered body compositor requires an opaque source canvas",
            ));
        }
        if self
            .front_alpha
            .iter()
            .chain(self.rear_alpha)
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(invalid(
                "layered body coverage must be finite and in [0,1]",
            ));
        }
        if self
            .front_rgb
            .iter()
            .chain(self.rear_source_rgb)
            .chain(self.scene_rgb)
            .any(|v| !v.is_finite() || !(0.0..=255.0).contains(v))
        {
            return Err(invalid(
                "prepared layered body colors must be finite and in [0,255]",
            ));
        }
        for i in 0..count {
            let front = self.front_alpha[i];
            let rear = self.rear_alpha[i];
            for c in 0..3 {
                let k = i * 3 + c;
                let foreground = front * self.front_rgb[k];
                let lower = (1.0 - front) * self.rear_source_rgb[k];
                let observed = source.pixels[i * 4 + c] as f64;
                let reconstructed = foreground + lower;
                if (reconstructed - observed).abs()
                    > arithmetic_error(foreground.abs() + lower.abs() + observed.abs())
                {
                    return Err(invalid(
                        "prepared front and lower layers do not reconstruct this source",
                    ));
                }
                let background = (1.0 - rear) * self.scene_rgb[k];
                let residual = self.rear_source_rgb[k] - background;
                let maximum = 255.0 * rear;
                let error =
                    arithmetic_error(self.rear_source_rgb[k].abs() + background.abs() + maximum);
                if residual < -error || residual > maximum + error {
                    return Err(invalid(
                        "prepared rear source and scene do not admit bounded material colors",
                    ));
                }
            }
        }
        Ok(count)
    }
}

/// Composite a caller's fixed inverse displacement field without modifying it.
/// Coordinates are source-pixel centers: output (x,y) reads (x+dx,y+dy), clamped
/// to the source canvas, with bilinear residual and coverage transport. Scene
/// and front remain stationary. Zero field and opaque front pixels retain the
/// observed source exactly. Validation runs even for a zero request.
///
/// Callers remain responsible for the field's selection/protection policy and
/// for useful source material roles. This is not a new geometric waist driver.
pub fn composite_body_source_layers(
    source: &RgbaImage,
    layers: &BodyLayeredSource<'_>,
    inverse_displacement: &[[f32; 2]],
) -> Result<RgbaImage, ImageError> {
    let count = layers.validate_source(source)?;
    if inverse_displacement.len() != count
        || inverse_displacement.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(invalid("layered body inverse field must be finite and match the source dimensions"));
    }
    let width = source.width as usize;
    let height = source.height as usize;
    let mut result = source.clone();
    for i in 0..count {
        let d = inverse_displacement[i];
        if d == [0.0, 0.0] || layers.front_alpha[i] == 1.0 {
            continue;
        }
        let x = ((i % width) as f64 + d[0] as f64).clamp(0.0, (width - 1) as f64);
        let y = ((i / width) as f64 + d[1] as f64).clamp(0.0, (height - 1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        let samples = [
            (y0 * width + x0, (1.0 - ty) * (1.0 - tx)),
            (y0 * width + x1, (1.0 - ty) * tx),
            (y1 * width + x0, ty * (1.0 - tx)),
            (y1 * width + x1, ty * tx),
        ];
        let mut coverage = 0.0;
        let mut transported = [0.0; 3];
        let mut transported_error = [0.0; 3];
        for (at, weight) in samples {
            let a = layers.rear_alpha[at];
            coverage += weight * a;
            for (c, value) in transported.iter_mut().enumerate() {
                let k = at * 3 + c;
                let background = (1.0 - a) * layers.scene_rgb[k];
                let residual = layers.rear_source_rgb[k] - background;
                *value += weight * residual;
                // Source feasibility permits only its calculated arithmetic
                // error. Carry that SAME bound with the residual, since a
                // bright source sample may move onto a black destination.
                // Recomputing a smaller bound from the final near-zero RGB
                // would reject an already admitted source pair.
                transported_error[c] += weight
                    * arithmetic_error(
                        layers.rear_source_rgb[k].abs() + background.abs() + 255.0 * a,
                    );
            }
        }
        let front = layers.front_alpha[i];
        for (c, &value) in transported.iter().enumerate() {
            let k = i * 3 + c;
            let rear = value + (1.0 - coverage) * layers.scene_rgb[k];
            let front_term = front * layers.front_rgb[k];
            let rear_term = (1.0 - front) * rear;
            let color = front_term + rear_term;
            let error = arithmetic_error(front_term.abs() + rear_term.abs() + 255.0)
                + (1.0 - front) * transported_error[c];
            if !color.is_finite() || color < -error || color > 255.0 + error {
                return Err(invalid(
                    "layered body composition produced out-of-range RGB; output was not clipped",
                ));
            }
            // The bounded result (including only insignificant roundoff at
            // 0/255) rounds into the byte domain; no RGB clamp is applied.
            let rounded = (color + 0.5).floor();
            if !(0.0..=255.0).contains(&rounded) {
                return Err(invalid("layered body RGB cannot be represented as a byte"));
            }
            result.pixels[i * 4 + c] = rounded as u8;
        }
    }
    Ok(result)
}
