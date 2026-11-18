//! Source-matte anchored foreground/background estimation.
//!
//! A numerical flow grid may grow without changing the photograph. Passing
//! that grid directly to the multilevel F/B estimator changes its pyramid and
//! initialization. Estimate on the fixed source support instead, then embed
//! the result into the caller's view. This provides view invariance when the
//! complete matte and observation halo are present, not improved hidden-color
//! accuracy or continuity when the source matte itself changes.

use crate::{face_matting, RgbaImage};

pub(super) fn estimate(source: &RgbaImage, alpha: &[f32]) -> Option<(RgbaImage, RgbaImage)> {
    let (width, height) = (source.width as usize, source.height as usize);
    if width == 0
        || height == 0
        || alpha.len() != width * height
        || alpha
            .iter()
            .any(|a| !a.is_finite() || !(0.0..=1.0).contains(a))
        || source.pixels.chunks_exact(4).any(|p| p[3] != 255)
    {
        return None;
    }

    let mut bounds = (width, height, 0, 0);
    for (i, &a) in alpha.iter().enumerate() {
        if a > 0.0 {
            bounds.0 = bounds.0.min(i % width);
            bounds.1 = bounds.1.min(i / width);
            bounds.2 = bounds.2.max(i % width + 1);
            bounds.3 = bounds.3.max(i / width + 1);
        }
    }
    if bounds.2 == 0 {
        return None;
    }

    // Existing local color evidence uses a 12px donor reach and 2px erosion.
    // This sets a source observation policy, not a theorem about the pyramid's
    // dependency radius. It is independent of target motion and flow padding.
    const OBSERVATION_HALO: usize = 14;
    let (left, top, right, bottom) = (
        bounds.0.saturating_sub(OBSERVATION_HALO),
        bounds.1.saturating_sub(OBSERVATION_HALO),
        (bounds.2 + OBSERVATION_HALO).min(width),
        (bounds.3 + OBSERVATION_HALO).min(height),
    );
    let (crop_width, crop_height) = (right - left, bottom - top);
    let mut pixels = Vec::with_capacity(crop_width * crop_height * 4);
    let mut crop_alpha = Vec::with_capacity(crop_width * crop_height);
    for y in top..bottom {
        pixels.extend_from_slice(&source.pixels[(y * width + left) * 4..(y * width + right) * 4]);
        crop_alpha.extend_from_slice(&alpha[y * width + left..y * width + right]);
    }
    let crop = RgbaImage {
        width: crop_width as u32,
        height: crop_height as u32,
        pixels,
    };
    let (foreground_crop, background_crop) = face_matting::estimate(&crop, &crop_alpha)?;
    let mut foreground = source.clone();
    let mut background = source.clone();
    for y in 0..crop_height {
        let dst = ((top + y) * width + left) * 4;
        let src = y * crop_width * 4;
        foreground.pixels[dst..dst + crop_width * 4]
            .copy_from_slice(&foreground_crop.pixels[src..src + crop_width * 4]);
        background.pixels[dst..dst + crop_width * 4]
            .copy_from_slice(&background_crop.pixels[src..src + crop_width * 4]);
    }
    // Outside this domain alpha is zero. Source RGB is the visible background;
    // the returned foreground there is only a placeholder, not inferred color.
    Some((foreground, background))
}

#[cfg(test)]
#[path = "face_optical_domain_regression.rs"]
mod regression;
