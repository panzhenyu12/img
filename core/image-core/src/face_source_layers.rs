//! In-memory source evidence for the continuous portrait material field.
//! No model inference, file loading or research-mode selection occurs here.

use crate::{ImageError, RgbaImage};
use serde_json::Value;

/// Half-open source-pixel rectangle. Outside it the existing source estimator
/// remains authoritative; zero alpha inside it is an actual observation.
#[derive(Clone, Copy, Debug)]
pub struct ObservationRegion {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

/// Frozen evidence for a single source photograph. Semantic masks and sliders
/// still use the normal face parameter contract. Optical coverage, motion
/// permission and the stationary scene have deliberately separate inputs.
pub struct SourceLayers<'a> {
    /// Exact decoded image used to prepare the evidence (including alpha).
    pub observed_source: &'a RgbaImage,
    /// Full-canvas, row-major f32 source coverage estimate in [0,1], without
    /// byte encoding. Material locks and source-scene feasibility can adjust
    /// the consumed coverage; this is not a final-alpha passthrough API.
    pub optical_alpha: &'a [f32],
    pub observation: ObservationRegion,
    /// Same-source scene estimate. It is an editing prior, not hidden truth.
    /// It is applied only within observed, nonzero, unfrozen source material;
    /// the existing estimator is retained elsewhere. The resulting plate is
    /// frozen for both residual sampling and final composition.
    pub scene: &'a RgbaImage,
    /// Optional source geometry: {mask: raster, contour: polygon}. The mask
    /// uses the existing 0..255 raster schema and the hair raster's resolution.
    /// It grants material motion; it never replaces optical coverage.
    pub motion_interface: Option<&'a Value>,
    /// Optional full-canvas binary 0/255 raster granting connected accessories
    /// motion with the ear. It cannot override user/feature/material protection.
    pub moving_attachment_support: Option<&'a Value>,
}

impl SourceLayers<'_> {
    pub(super) fn validate(&self, source: &RgbaImage) -> Result<(), ImageError> {
        let error = super::invalid;
        let count = (source.width as usize)
            .checked_mul(source.height as usize)
            .filter(|&n| n > 0 && n <= 32_000_000)
            .ok_or_else(|| error("face source dimensions are invalid"))?;
        if source != self.observed_source {
            return Err(error(
                "face source layers belong to a different decoded source image",
            ));
        }
        if [source, self.scene].iter().any(|image| {
            image.width != source.width
                || image.height != source.height
                || image.pixels.len() != count * 4
                || image.pixels.chunks_exact(4).any(|p| p[3] != 255)
        }) {
            return Err(error(
                "face source layers require matching opaque RGBA canvases",
            ));
        }
        if self.optical_alpha.len() != count
            || self
                .optical_alpha
                .iter()
                .any(|a| !a.is_finite() || !(0.0..=1.0).contains(a))
        {
            return Err(error(
                "face optical alpha must be a full-canvas finite f32 array in [0,1]",
            ));
        }
        let r = self.observation;
        if r.width == 0
            || r.height == 0
            || r.left.checked_add(r.width).is_none_or(|v| v > source.width)
            || r.top
                .checked_add(r.height)
                .is_none_or(|v| v > source.height)
        {
            return Err(error(
                "face source observation region is empty or outside the canvas",
            ));
        }
        if let Some(mask) = self.moving_attachment_support {
            if mask.get("shape").and_then(Value::as_str) != Some("raster")
                || mask.get("width").and_then(Value::as_u64) != Some(source.width as u64)
                || mask.get("height").and_then(Value::as_u64) != Some(source.height as u64)
                || !mask
                    .get("data")
                    .and_then(Value::as_array)
                    .is_some_and(|data| {
                        data.len() == count
                            && data.iter().all(|v| matches!(v.as_u64(), Some(0 | 255)))
                    })
            {
                return Err(error(
                    "face moving attachment support must be a full-source binary raster",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn contains(&self, x: usize, y: usize) -> bool {
        let r = self.observation;
        x >= r.left as usize
            && x < (r.left + r.width) as usize
            && y >= r.top as usize
            && y < (r.top + r.height) as usize
    }

    pub(super) fn apply_scene(
        &self,
        crop: &RgbaImage,
        alpha: &mut [f32],
        background: &mut RgbaImage,
        frozen: &[bool],
        origin: (usize, usize),
    ) -> Result<(), ImageError> {
        let (cw, ch) = (crop.width as usize, crop.height as usize);
        let mut pixels = Vec::with_capacity(cw * ch * 4);
        let mut weights = Vec::with_capacity(cw * ch);
        for y in origin.1..origin.1 + ch {
            let start = y * self.scene.width as usize + origin.0;
            pixels.extend_from_slice(&self.scene.pixels[4 * start..4 * (start + cw)]);
            weights.extend((origin.0..origin.0 + cw).map(|x| {
                if self.contains(x, y) {
                    1.0
                } else {
                    0.0
                }
            }));
        }
        let scene = RgbaImage::new(crop.width, crop.height, pixels)?;
        // Preserve the reviewed source-only order: initial pairing, fixed
        // scene prior, then the minimal coverage required by that scene. This
        // is a conditional coverage bound, never a claim of measured opacity.
        super::source_plate::apply(crop, alpha, background, &scene, &weights, frozen)
            .map_err(|e| super::invalid(&e))?;
        super::source_plate::repair_coverage(crop, alpha, background, &weights, frozen)
            .map_err(|e| super::invalid(&e))?;
        Ok(())
    }
}
