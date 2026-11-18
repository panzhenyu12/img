//! Physical source coverage is separate from anatomical motion selection.
//!
//! This single moving layer describes an opaque photograph, including soft
//! foreground/background coverage. It does not infer material identity or
//! independent occlusion order. Callers must supply same-source layer data
//! and retain other fixed foreground objects in explicit protection.

use crate::{ImageError, RgbaImage};
use serde_json::Value;

/// Already authorized source material, with explicit dimensions and f32
/// optical coverage. A provided background is immutable and must be a feasible
/// paired plate; algebraic validation cannot prove that it came from this photo.
pub struct SourceLayers<'a> {
    pub width: u32,
    pub height: u32,
    pub optical_alpha: &'a [f32],
    pub paired_background: Option<&'a RgbaImage>,
}

pub(super) struct PreparedSourceLayers<'a> {
    alpha: &'a [f32],
    background: RgbaImage,
}

fn fixed_plate_is_feasible_at_alpha_precision(
    source: &RgbaImage,
    alpha: &[f32],
    background: &RgbaImage,
) -> bool {
    alpha.iter().enumerate().all(|(i, &a)| {
        // A rounded coverage can be at most halfway to the next f32 above
        // it. Increasing alpha widens both feasible RGB limits. This is an
        // alpha-representation bound, not a fixed color tolerance; alpha=0
        // and alpha=1 keep their exact endpoint contracts.
        let rounding = if a > 0.0 && a < 1.0 {
            (f32::from_bits(a.to_bits() + 1) as f64 - a as f64) * 0.5
        } else {
            0.0
        };
        (0..3).all(|c| {
            let observed = source.pixels[i * 4 + c] as f64;
            let b = background.pixels[i * 4 + c] as f64;
            let p = (observed - b) + a as f64 * b;
            p >= -b * rounding && p <= 255.0 * a as f64 + (255.0 - b) * rounding
        })
    })
}

impl<'a> SourceLayers<'a> {
    pub(super) fn prepare(
        &self,
        source: &RgbaImage,
    ) -> Result<PreparedSourceLayers<'a>, ImageError> {
        let invalid = |message: &str| ImageError::InvalidOperation(message.into());
        let count = (source.width as usize)
            .checked_mul(source.height as usize)
            .filter(|&count| count > 0)
            .ok_or_else(|| invalid("body source image dimensions are invalid"))?;
        let pixel_bytes = count
            .checked_mul(4)
            .ok_or_else(|| invalid("body source image dimensions are invalid"))?;
        if self.width != source.width
            || self.height != source.height
            || self.optical_alpha.len() != count
        {
            return Err(invalid(
                "body source optical alpha must have the source image dimensions",
            ));
        }
        if self
            .optical_alpha
            .iter()
            .any(|a| !a.is_finite() || !(0.0..=1.0).contains(a))
        {
            return Err(invalid(
                "body source optical alpha must contain finite f32 coverage in [0,1]",
            ));
        }
        if source.pixels.len() != pixel_bytes || source.pixels.chunks_exact(4).any(|p| p[3] != 255)
        {
            return Err(invalid(
                "body source layers currently require an opaque RGBA source canvas",
            ));
        }
        let mut background = if let Some(fixed) = self.paired_background {
            fixed.clone()
        } else {
            // This is a source-only prior. It never depends on a slider,
            // target holes, the sign of motion or the eventual write domain.
            match super::body_layer_colors(source, self.optical_alpha) {
                Some((_, background)) => background,
                None if self.optical_alpha.iter().all(|&a| a == 0.0) => source.clone(),
                None => return Err(invalid("body source layers need a fixed background when source-only background estimation is unavailable")),
            }
        };
        crate::face::layer_pair::pair(source, self.optical_alpha, &mut background)
            .ok_or_else(|| invalid("body source background must match the opaque source canvas"))?;
        if let Some(fixed) = self.paired_background {
            if background != *fixed {
                if !fixed_plate_is_feasible_at_alpha_precision(source, self.optical_alpha, fixed) {
                    return Err(invalid(
                        "body fixed paired background is inconsistent with source optical coverage",
                    ));
                }
                // Only the roundoff exception above reaches this branch.
                // Residual transport preserves the exact source relation
                // without replacing the caller's fixed plate or alpha.
                background = fixed.clone();
            }
        }
        Ok(PreparedSourceLayers {
            alpha: self.optical_alpha,
            background,
        })
    }
}

pub(super) fn render(
    source: &RgbaImage,
    params: &Value,
    region: super::Region,
    warps: &[super::Warp],
    subject_mask: &Value,
    layers: &PreparedSourceLayers<'_>,
) -> Result<RgbaImage, ImageError> {
    let width = source.width as usize;
    let height = source.height as usize;
    let target = super::geometry_target(source, params, subject_mask);
    let controls = super::motion_controls(source, params, warps, subject_mask, &target);
    let mut result = source.clone();
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if controls.user_protected[i] {
                continue;
            }
            let d = super::contour_displacement(
                region,
                &controls.measured,
                (x as f32 + 0.5, y as f32 + 0.5),
                controls.selected[i],
                controls.protect_fade[i],
                width as f32,
                height as f32,
            );
            if d.0 == 0.0 && d.1 == 0.0 {
                continue;
            }
            // Geometric selection and explicit protection govern motion.
            // A semantic alpha=0, or a true source optical aperture, is not
            // an instruction to freeze this destination. Coverage comes
            // only from the transported physical source layer.
            let (pixel, membership) = crate::face::layer_pair::sample_residual(
                source,
                layers.alpha,
                &layers.background,
                (x as f32 + d.0).clamp(0.0, source.width as f32 - 1.0),
                (y as f32 + d.1).clamp(0.0, source.height as f32 - 1.0),
            );
            for c in 0..3 {
                result.pixels[i * 4 + c] = super::byte(
                    pixel[c] + (1.0 - membership) * layers.background.pixels[i * 4 + c] as f32,
                );
            }
            result.pixels[i * 4 + 3] = 255;
        }
    }
    super::finish_body_blend(source, &mut result, params);
    Ok(result)
}
