//! Deterministic, non-destructive image operations.
//!
//! The core deliberately keeps its pixel representation dependency-free. The
//! same typed operations can be compiled to native code or WebAssembly; a host
//! adapter translates the shared JSON operation protocol into these Rust types.

mod body;
mod color;
mod engine;
mod face;
pub(crate) mod face_matting;
mod geometry;
mod inpaint;
mod mask;
mod model_inpaint;
mod paint;
mod validation;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use engine::{apply_graph, apply_graph_json, EditGraph, GraphOperation};
pub use model_inpaint::{composite_model_inpaint, prepare_model_inpaint, InpaintMasks};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Parameters for the shared `color.basic` operation.
///
/// Values are normalized for direct use as UI slider values. Brightness and
/// saturation use -1..=1. Contrast uses -1..=1, with 0 unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAdjustParams {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

impl Default for ColorAdjustParams {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
        }
    }
}

/// Parameters for the shared `filter.film` operation. Grain is deterministic
/// for a given seed, keeping previews and exports reproducible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmFilterParams {
    pub grain: f32,
    pub vignette: f32,
    pub seed: u64,
}

impl Default for FilmFilterParams {
    fn default() -> Self {
        Self {
            grain: 0.0,
            vignette: 0.0,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperationType {
    Identity,
    ColorBasic(ColorAdjustParams),
    FilterFilm(FilmFilterParams),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub id: String,
    pub operation_type: OperationType,
    pub enabled: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ImageError {
    InvalidDimensions,
    PixelLengthMismatch,
    InvalidOperation(String),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => f.write_str("invalid image dimensions"),
            Self::PixelLengthMismatch => {
                f.write_str("RGBA buffer length does not match dimensions")
            }
            Self::InvalidOperation(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for ImageError {}

impl RgbaImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions);
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|length| length.checked_mul(4))
            .filter(|length| *length <= 128_000_000)
            .ok_or(ImageError::InvalidDimensions)?;
        if pixels.len() != expected {
            return Err(ImageError::PixelLengthMismatch);
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn identity(&self) -> Self {
        self.clone()
    }

    /// Apply brightness, contrast and saturation while preserving alpha.
    pub fn adjust_color(&self, params: ColorAdjustParams) -> Self {
        let brightness = finite_clamp(params.brightness, -1.0, 1.0, 0.0);
        let contrast = finite_clamp(params.contrast, -1.0, 1.0, 0.0);
        let saturation = finite_clamp(params.saturation, -1.0, 1.0, 0.0);
        let contrast_factor = 1.0 + contrast;
        let brightness_offset = brightness * 255.0;
        let saturation_factor = 1.0 + saturation;

        let mut pixels = self.pixels.clone();
        for rgba in pixels.chunks_exact_mut(4) {
            let mut r = rgba[0] as f32;
            let mut g = rgba[1] as f32;
            let mut b = rgba[2] as f32;

            r = (r + brightness_offset - 128.0) * contrast_factor + 128.0;
            g = (g + brightness_offset - 128.0) * contrast_factor + 128.0;
            b = (b + brightness_offset - 128.0) * contrast_factor + 128.0;

            // Rec.709 luma gives a useful neutral point for RGB photos and
            // keeps grayscale pixels unchanged when saturation changes.
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            r = luma + (r - luma) * saturation_factor;
            g = luma + (g - luma) * saturation_factor;
            b = luma + (b - luma) * saturation_factor;

            rgba[0] = to_u8(r);
            rgba[1] = to_u8(g);
            rgba[2] = to_u8(b);
        }
        Self {
            pixels,
            ..self.clone()
        }
    }

    /// Add deterministic monochrome grain and a radial vignette.
    pub fn apply_film(&self, params: FilmFilterParams) -> Self {
        let grain = finite_clamp(params.grain, 0.0, 1.0, 0.0);
        let vignette = finite_clamp(params.vignette, 0.0, 1.0, 0.0);
        let grain_amount = grain * 32.0;
        let mut pixels = self.pixels.clone();

        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_index = (y as usize * self.width as usize + x as usize) * 4;
                let nx = ((x as f32 + 0.5) / self.width as f32) * 2.0 - 1.0;
                let ny = ((y as f32 + 0.5) / self.height as f32) * 2.0 - 1.0;
                let radius = ((nx * nx + ny * ny) * 0.5).sqrt().min(1.0);
                let vignette_factor = 1.0 - vignette * radius * radius * 0.75;
                let noise = grain_amount * grain_noise(params.seed, pixel_index as u64);

                pixels[pixel_index] = to_u8(pixels[pixel_index] as f32 * vignette_factor + noise);
                pixels[pixel_index + 1] =
                    to_u8(pixels[pixel_index + 1] as f32 * vignette_factor + noise);
                pixels[pixel_index + 2] =
                    to_u8(pixels[pixel_index + 2] as f32 * vignette_factor + noise);
                // Alpha is deliberately untouched.
            }
        }
        Self {
            pixels,
            ..self.clone()
        }
    }
}

pub fn apply(image: &RgbaImage, operation: &Operation) -> RgbaImage {
    if !operation.enabled {
        return image.clone();
    }
    match operation.operation_type {
        OperationType::Identity => image.identity(),
        OperationType::ColorBasic(params) => image.adjust_color(params),
        OperationType::FilterFilm(params) => image.apply_film(params),
    }
}

fn finite_clamp(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

fn to_u8(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    value.round().clamp(0.0, 255.0) as u8
}

/// A tiny integer hash mapped to [-1, 1]. It avoids pulling a PRNG crate into
/// the WASM bundle while staying deterministic across platforms.
fn grain_noise(seed: u64, index: u64) -> f32 {
    let mut x = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = (x as u32) as f32 / u32::MAX as f32;
    unit * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_pixel_buffer() {
        assert_eq!(
            RgbaImage::new(1, 1, vec![0; 3]),
            Err(ImageError::PixelLengthMismatch)
        );
    }

    #[test]
    fn identity_preserves_dimensions_and_pixels() {
        let source = RgbaImage::new(2, 1, vec![1, 2, 3, 255, 4, 5, 6, 255]).unwrap();
        let operation = Operation {
            id: "op-1".into(),
            operation_type: OperationType::Identity,
            enabled: true,
        };
        assert_eq!(apply(&source, &operation), source);
    }

    #[test]
    fn disabled_operation_is_a_noop() {
        let source = RgbaImage::new(1, 1, vec![10, 20, 30, 255]).unwrap();
        let operation = Operation {
            id: "op-1".into(),
            operation_type: OperationType::ColorBasic(ColorAdjustParams {
                brightness: 1.0,
                ..Default::default()
            }),
            enabled: false,
        };
        assert_eq!(apply(&source, &operation), source);
    }

    fn image(width: u32, height: u32, rgba: [u8; 4]) -> RgbaImage {
        RgbaImage::new(width, height, rgba.repeat(width as usize * height as usize)).unwrap()
    }

    #[test]
    fn brightness_preserves_alpha_and_clamps_channels() {
        let source = image(1, 1, [10, 120, 250, 77]);
        let output = source.adjust_color(ColorAdjustParams {
            brightness: 0.5,
            ..Default::default()
        });
        assert_eq!(output.pixels, vec![138, 248, 255, 77]);
    }

    #[test]
    fn contrast_keeps_middle_gray_stable() {
        let source = image(1, 1, [128, 128, 128, 99]);
        let output = source.adjust_color(ColorAdjustParams {
            contrast: 1.0,
            ..Default::default()
        });
        assert_eq!(output.pixels, source.pixels);
    }

    #[test]
    fn saturation_zero_and_negative_are_predictable() {
        let source = image(1, 1, [200, 100, 50, 255]);
        assert_eq!(source.adjust_color(ColorAdjustParams::default()), source);
        let gray = source.adjust_color(ColorAdjustParams {
            saturation: -1.0,
            ..Default::default()
        });
        assert_eq!(gray.pixels[0], gray.pixels[1]);
        assert_eq!(gray.pixels[1], gray.pixels[2]);
        assert_eq!(gray.pixels[3], 255);
    }

    #[test]
    fn film_is_deterministic_and_only_changes_rgb() {
        let source = image(3, 3, [120, 120, 120, 61]);
        let params = FilmFilterParams {
            grain: 0.8,
            vignette: 0.5,
            seed: 42,
        };
        let first = source.apply_film(params);
        let second = source.apply_film(params);
        assert_eq!(first, second);
        assert!(first.pixels.chunks_exact(4).all(|pixel| pixel[3] == 61));
        assert_ne!(first.pixels, source.pixels);
    }

    #[test]
    fn operation_dispatches_color_and_film_variants() {
        let source = image(1, 1, [20, 30, 40, 255]);
        let color = Operation {
            id: "color".into(),
            operation_type: OperationType::ColorBasic(ColorAdjustParams {
                brightness: 0.2,
                ..Default::default()
            }),
            enabled: true,
        };
        assert_ne!(apply(&source, &color), source);
        let film = Operation {
            id: "film".into(),
            operation_type: OperationType::FilterFilm(FilmFilterParams {
                grain: 0.5,
                ..Default::default()
            }),
            enabled: true,
        };
        assert_ne!(apply(&source, &film), source);
    }
}
