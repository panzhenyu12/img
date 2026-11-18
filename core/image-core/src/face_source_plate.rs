//! Experimental source-conditioned plate for encoded photographic contrast.
//!
//! A reviewed/generated scene prior can disagree with the convex F/B model
//! used by `layer_pair::pair` (for example at a sharpened boundary). Here the
//! caller explicitly assigns C-(1-a)B to selected moving material. It may be
//! signed or exceed 255*a. This is an editing hypothesis, not recovered
//! physical foreground. Use the SAME immutable plate in residual sampling
//! and destination composition. The function does not infer ownership.

use crate::{engine::byte, RgbaImage};

/// Load one immutable source-space prior for native algorithm experiments.
/// File I/O is test-only; the public algorithm accepts memory buffers.
#[cfg(test)]
pub(super) fn apply_cached(
    full_source: &RgbaImage,
    crop: &RgbaImage,
    alpha: &mut [f32],
    background: &mut RgbaImage,
    frozen: &[bool],
    origin: (usize, usize),
    descriptor: &std::path::Path,
) -> Result<usize, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Prior {
        source: std::path::PathBuf,
        // Python prior builders record the encoded file hash for their
        // manifest. Native replay binds the input more directly below by
        // comparing the entire decoded source (including alpha). This field
        // is metadata, never a substitute for that pixel equality check.
        #[serde(default, rename = "sourceSha256")]
        _source_sha256: Option<String>,
        image: std::path::PathBuf,
        weights: std::path::PathBuf,
        #[serde(default)]
        coverage_policy: CoveragePolicy,
    }
    #[derive(Default, serde::Deserialize)]
    #[serde(rename_all = "kebab-case")]
    enum CoveragePolicy {
        #[default]
        Signed,
        SceneLowerBound,
    }
    let evidence: Prior =
        serde_json::from_slice(&std::fs::read(descriptor).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let base = descriptor
        .parent()
        .ok_or("prior descriptor has no parent")?;
    let resolve = |p: &std::path::Path| {
        if p.is_absolute() {
            p.to_owned()
        } else {
            base.join(p)
        }
    };
    let read_image = |p: &std::path::Path| -> Result<RgbaImage, String> {
        let image = image::load_from_memory_with_format(
            &std::fs::read(resolve(p)).map_err(|e| e.to_string())?,
            image::ImageFormat::Png,
        )
        .map_err(|e| e.to_string())?
        .into_rgba8();
        RgbaImage::new(image.width(), image.height(), image.into_raw()).map_err(|e| e.to_string())
    };
    let observed = read_image(&evidence.source)?;
    if observed != *full_source {
        return Err("scene prior belongs to a different decoded source image".into());
    }
    let prior = read_image(&evidence.image)?;
    let data = std::fs::read(resolve(&evidence.weights)).map_err(|e| e.to_string())?;
    let count = full_source.width as usize * full_source.height as usize;
    if prior.width != full_source.width
        || prior.height != full_source.height
        || data.len() != 4 * count
    {
        return Err("source prior and f32 weights must match the full source canvas".into());
    }
    let weights: Vec<_> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if weights
        .iter()
        .any(|&w| !w.is_finite() || !(0.0..=1.0).contains(&w))
    {
        return Err("source prior weights must be finite in [0, 1]".into());
    }
    let (left, top) = origin;
    let (cw, ch) = (crop.width as usize, crop.height as usize);
    if left + cw > full_source.width as usize || top + ch > full_source.height as usize {
        return Err("source prior crop exceeds canvas".into());
    }
    let mut pixels = Vec::with_capacity(cw * ch * 4);
    let mut selected = Vec::with_capacity(cw * ch);
    for y in top..top + ch {
        let start = y * full_source.width as usize + left;
        pixels.extend_from_slice(&prior.pixels[4 * start..4 * (start + cw)]);
        selected.extend_from_slice(&weights[start..start + cw]);
    }
    let cropped = RgbaImage::new(crop.width, crop.height, pixels).map_err(|e| e.to_string())?;
    let changed = apply(crop, alpha, background, &cropped, &selected, frozen)?;
    if matches!(evidence.coverage_policy, CoveragePolicy::SceneLowerBound) {
        repair_coverage(crop, alpha, background, &selected, frozen)?;
    }
    Ok(changed)
}

pub(super) fn apply(
    source: &RgbaImage,
    alpha: &[f32],
    background: &mut RgbaImage,
    prior: &RgbaImage,
    weights: &[f32],
    frozen: &[bool],
) -> Result<usize, String> {
    let count = source.width as usize * source.height as usize;
    if source.width == 0
        || source.height == 0
        || [source, background, prior].iter().any(|image| {
            image.width != source.width
                || image.height != source.height
                || image.pixels.len() != 4 * count
                || image.pixels.chunks_exact(4).any(|p| p[3] != 255)
        })
        || alpha.len() != count
        || weights.len() != count
        || frozen.len() != count
        || alpha
            .iter()
            .chain(weights)
            .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err(
            "source plate requires matching opaque images, finite alpha/weights and fixed support"
                .into(),
        );
    }
    let mut changed = 0;
    for i in 0..count {
        // The existing renderer does not transport zero-coverage material.
        // Do not introduce a signed layer outside that support, or modify a
        // user/material lock by changing its lower layer behind its back.
        if frozen[i] || alpha[i] <= 1e-6 || weights[i] == 0.0 {
            continue;
        }
        let mut different = false;
        for c in 0..3 {
            let at = i * 4 + c;
            let value = byte(
                background.pixels[at] as f32 * (1.0 - weights[i])
                    + prior.pixels[at] as f32 * weights[i],
            );
            different |= value != background.pixels[at];
            background.pixels[at] = value;
        }
        changed += usize::from(different);
    }
    Ok(changed)
}

/// Minimal source coverage compatible with a chosen byte scene plate.
/// This is a conditional editing prior, not a measured physical opacity.
pub(super) fn repair_coverage(
    source: &RgbaImage,
    alpha: &mut [f32],
    background: &RgbaImage,
    weights: &[f32],
    frozen: &[bool],
) -> Result<usize, String> {
    let count = source.width as usize * source.height as usize;
    if count == 0
        || source.width != background.width
        || source.height != background.height
        || source.pixels.len() != count * 4
        || background.pixels.len() != count * 4
        || alpha.len() != count
        || weights.len() != count
        || frozen.len() != count
        || source
            .pixels
            .chunks_exact(4)
            .chain(background.pixels.chunks_exact(4))
            .any(|p| p[3] != 255)
        || alpha
            .iter()
            .chain(weights)
            .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err(
            "coverage repair requires matching opaque source layers and finite weights".into(),
        );
    }
    let mut changed = 0;
    for i in 0..count {
        if frozen[i] || weights[i] == 0.0 || alpha[i] <= 1e-6 {
            continue;
        }
        let mut required = alpha[i] as f64;
        for c in 0..3 {
            let observed = source.pixels[4 * i + c] as f64;
            let b = background.pixels[4 * i + c] as f64;
            // C=aF+(1-a)B with 0<=F<=255. Each inequality gives a
            // conditional lower bound on a. No output or movement is input.
            let lower = if observed < b {
                (b - observed) / b
            } else if observed > b {
                (observed - b) / (255.0 - b)
            } else {
                0.0
            };
            required = required.max(lower);
        }
        let mut coverage = required as f32;
        // Round the feasibility bound up by at most one f32 ulp. Rounding
        // down would reintroduce negative P before any image movement.
        if (coverage as f64) < required {
            coverage = f32::from_bits(coverage.to_bits() + 1);
        }
        changed += usize::from(coverage > alpha[i]);
        alpha[i] = coverage;
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::super::layer_pair;
    use super::*;

    #[test]
    fn cached_prior_accepts_builder_metadata_but_binds_the_complete_decoded_source() {
        use std::{fs, time::{SystemTime, UNIX_EPOCH}};
        let directory = std::env::temp_dir().join(format!(
            "image-source-prior-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
        }
        let _cleanup = Cleanup(directory.clone());
        let source = gray(&[120, 140]);
        let prior = gray(&[90, 90]);
        for (name, input) in [("source.png", &source), ("prior.png", &prior)] {
            image::save_buffer_with_format(directory.join(name), &input.pixels,
                input.width, input.height, image::ColorType::Rgba8,
                image::ImageFormat::Png).unwrap();
        }
        fs::write(directory.join("weights.f32"), [1.0_f32, 1.0].iter()
            .flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let descriptor = directory.join("prior.json");
        // The prior builders emit this metadata key. Pixel binding below
        // must still reject a different image even if its dimensions match.
        fs::write(&descriptor, serde_json::to_vec(&serde_json::json!({
            "source": "source.png", "sourceSha256": "manifest-metadata",
            "image": "prior.png", "weights": "weights.f32"
        })).unwrap()).unwrap();
        let mut alpha = vec![0.8; 2];
        let mut background = source.clone();
        assert_eq!(apply_cached(&source, &source, &mut alpha, &mut background,
            &[false; 2], (0, 0), &descriptor).unwrap(), 2);
        assert_eq!(background, prior);
        let unrelated = gray(&[120, 141]);
        let snapshot = background.clone();
        let error = apply_cached(&unrelated, &unrelated, &mut alpha, &mut background,
            &[false; 2], (0, 0), &descriptor).unwrap_err();
        assert!(error.contains("different decoded source"), "{error}");
        assert_eq!(background, snapshot);
    }

    fn gray(values: &[u8]) -> RgbaImage {
        RgbaImage::new(
            values.len() as u32,
            1,
            values.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        )
        .unwrap()
    }
    fn composite(
        source: &RgbaImage,
        alpha: &[f32],
        plate: &RgbaImage,
        at: usize,
        sample: f32,
    ) -> u8 {
        let (p, a) = layer_pair::sample_residual(source, alpha, plate, sample, 0.0);
        byte(p[0] + (1.0 - a) * plate.pixels[4 * at] as f32)
    }

    #[test]
    fn assigned_encoded_edge_leaves_its_source_and_preserves_identity() {
        // Independently specified acquisition: a 100-code object over a
        // 220-code scene has a .1-alpha edge at 208. A +28-code sharpening
        // component is assigned to that moving edge, giving source C=236.
        let source = gray(&[220, 220, 236, 100, 100, 220, 220]);
        let alpha = [0.0, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0];
        let known_scene = gray(&[220; 7]);
        let mut old = known_scene.clone();
        layer_pair::pair(&source, &alpha, &mut old).unwrap();
        assert_eq!(
            composite(&source, &alpha, &old, 2, 1.0),
            234,
            "physical box projection reproduces the old bright source ridge"
        );
        let mut plate = old.clone();
        apply(
            &source,
            &alpha,
            &mut plate,
            &known_scene,
            &[1.0; 7],
            &[false; 7],
        )
        .unwrap();
        for i in 0..7 {
            assert_eq!(
                composite(&source, &alpha, &plate, i, i as f32),
                source.pixels[4 * i]
            );
        }
        // A one-pixel rightward translation exposes background at the old
        // edge and carries the observed encoded edge to its new location.
        assert_eq!(composite(&source, &alpha, &plate, 2, 1.0), 220);
        assert_eq!(composite(&source, &alpha, &plate, 3, 2.0), 236);
        // Subpixel transport carries the full observed contrast once.
        assert_eq!(composite(&source, &alpha, &plate, 2, 1.5), 228);
    }

    #[test]
    fn fixed_scene_line_and_protection_are_not_reassigned_by_a_bad_prior() {
        let source = gray(&[220, 40, 160, 100, 220]);
        let alpha = [0.0, 0.0, 0.5, 1.0, 0.0];
        let mut background = gray(&[220, 40, 220, 220, 220]);
        let before = background.clone();
        let prior = gray(&[0; 5]);
        apply(
            &source,
            &alpha,
            &mut background,
            &prior,
            &[1.0; 5],
            &[false, false, true, true, false],
        )
        .unwrap();
        assert_eq!(background, before);
        assert_eq!(composite(&source, &alpha, &background, 1, 0.0), 40);
    }

    #[test]
    fn physical_soft_edge_still_matches_a_known_layer_without_signed_energy() {
        let source = gray(&[20, 60, 100, 180, 20]);
        let alpha = [0.0, 0.25, 0.5, 1.0, 0.0];
        let mut background = gray(&[20; 5]);
        apply(
            &source,
            &alpha,
            &mut background,
            &gray(&[20; 5]),
            &[1.0; 5],
            &[false; 5],
        )
        .unwrap();
        assert_eq!(composite(&source, &alpha, &background, 1, 1.5), 80);
        assert_eq!(composite(&source, &alpha, &background, 1, 0.0), 20);
    }

    #[test]
    fn underestimated_dark_material_transports_to_darker_scene_without_negative_color() {
        // Independent known layer: black material at .25 coverage over B=200
        // gives C=150. The model incorrectly estimates .01 coverage. Moving
        // this edge onto B=20 must yield 15, not a clipped negative residual.
        let source = gray(&[20, 150, 20]);
        let background = gray(&[20, 200, 20]);
        let mut alpha = [0.0, 0.01, 0.0];
        repair_coverage(&source, &mut alpha, &background, &[1.0; 3], &[false; 3]).unwrap();
        for at in [0, 2] {
            let (p, a) = layer_pair::sample_residual(&source, &alpha, &background, 1.0, 0.0);
            let result = p[0] + (1.0 - a) * background.pixels[4 * at] as f32;
            assert!(
                (result - 15.0).abs() < 1e-4,
                "unclipped known-layer result was {result}"
            );
        }
        for at in 0..3 {
            assert_eq!(
                composite(&source, &alpha, &background, at, at as f32),
                source.pixels[4 * at]
            );
        }
    }

    #[test]
    fn coverage_prior_keeps_locks_zero_support_and_already_feasible_layers() {
        let source = gray(&[20, 150, 150, 60]);
        let background = gray(&[200, 200, 200, 20]);
        let mut alpha = [0.0, 0.01, 0.01, 0.25];
        let before = alpha;
        repair_coverage(
            &source,
            &mut alpha,
            &background,
            &[1.0, 1.0, 0.0, 1.0],
            &[false, true, false, false],
        )
        .unwrap();
        assert_eq!(alpha, before);
    }
}
