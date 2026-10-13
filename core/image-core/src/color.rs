use crate::engine::{byte, number, percent, positive};
use crate::mask::Region;
use crate::{ImageError, RgbaImage};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;

fn luma(r: f32, g: f32, b: f32) -> f32 {
    r * 0.2126 + g * 0.7152 + b * 0.0722
}

fn smoothstep(start: f32, end: f32, value: f32) -> f32 {
    let t = ((value - start) / (end - start).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Reconstruct RGB from luminance and zero-luminance chroma. Compress chroma
/// at the gamut edge instead of clipping channels, preserving hue in bright
/// skin and colored makeup when luminance or warmth is adjusted.
fn luma_chroma_rgb(light: f32, chroma: [f32; 3]) -> [f32; 3] {
    let light = light.clamp(0.0, 255.0);
    let mut scale: f32 = 1.0;
    for value in chroma {
        if value > 0.0 {
            scale = scale.min((255.0 - light) / value);
        } else if value < 0.0 {
            scale = scale.min(light / -value);
        }
    }
    chroma.map(|value| light + value * scale.clamp(0.0, 1.0))
}

fn brighten_luma(rgb: [f32; 3], delta: f32) -> [f32; 3] {
    let light = luma(rgb[0], rgb[1], rgb[2]);
    luma_chroma_rgb(light + delta, rgb.map(|value| value - light))
}

fn srgb_to_linear(value: f32) -> f32 {
    let normalized = (value / 255.0).clamp(0.0, 1.0);
    if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        // Use the same software implementation on native and WASM. Platform
        // powf implementations can straddle a final 8-bit rounding boundary.
        libm::powf((normalized + 0.055) / 1.055, 2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    let linear = value.clamp(0.0, 1.0);
    let normalized = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * libm::powf(linear, 1.0 / 2.4) - 0.055
    };
    normalized * 255.0
}

/// Inner-mouth landmarks also enclose gums, tongue and dark mouth cavities.
/// Teeth whitening therefore needs both geometric coverage and a photometric
/// gate; bright, weakly saturated enamel receives most of the effect.
fn tooth_confidence(rgb: [f32; 3]) -> f32 {
    let light = luma(rgb[0], rgb[1], rgb[2]);
    let max = rgb.into_iter().fold(0.0, f32::max);
    let min = rgb.into_iter().fold(255.0, f32::min);
    let saturation = (max - min) / max.max(1.0);
    smoothstep(70.0, 150.0, light) * (1.0 - smoothstep(0.28, 0.50, saturation))
}

pub(crate) fn basic(source: &RgbaImage, p: &Value) -> RgbaImage {
    let exposure = libm::powf(2.0, percent(p, "exposure") * 2.0);
    let brightness = percent(p, "brightness") * 100.0;
    let contrast = 1.0 + percent(p, "contrast");
    let saturation = 1.0 + percent(p, "saturation");
    let temperature = percent(p, "temperature") * 32.0;
    let tint = percent(p, "tint") * 24.0;
    let highlights = percent(p, "highlights") * 70.0;
    let shadows = percent(p, "shadows") * 70.0;
    let whites = percent(p, "whites") * 70.0;
    let blacks = percent(p, "blacks") * 70.0;
    let vibrance = percent(p, "vibrance");
    let clarity = percent(p, "clarity");
    let sharpness = positive(p, "sharpness");
    let noise = positive(p, "noise");
    let texture = percent(p, "texture");
    let black_point = number(p, "blackPoint", 0.0) / 100.0 * 255.0;
    let white_point = number(p, "whitePoint", 100.0) / 100.0 * 255.0;
    let gamma = number(p, "gamma", 1.0).clamp(0.1, 5.0);
    let short_edge = source.width.min(source.height) as f32;
    let clarity_base = (clarity.abs() > 0.001)
        .then(|| local_blurred(source, p, (short_edge * 0.020).clamp(2.0, 32.0)));
    let sharpening_base = (sharpness > 0.001)
        .then(|| local_blurred(source, p, (short_edge * 0.001).clamp(0.55, 2.0)));
    let texture_base = (texture.abs() > 0.001)
        .then(|| local_blurred(source, p, (short_edge * 0.004).clamp(1.0, 8.0)));
    let denoised = (noise > 0.001).then(|| bilateral_denoise(source, p, noise));
    let dehazed = (percent(p, "dehaze").abs() > 0.001).then(|| atmospheric_dehaze(source, p));
    let mut out = source.clone();
    for (index, pixel) in out.pixels.chunks_exact_mut(4).enumerate() {
        let mut rgb = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let input_luma = luma(rgb[0], rgb[1], rgb[2]) / 255.0;
        let high = input_luma.powi(2);
        let low = (1.0 - input_luma).powi(2);
        let adjustment =
            highlights * high + shadows * low + whites * high.powi(2) + blacks * low.powi(2);
        for (c, value) in rgb.iter_mut().enumerate() {
            if let Some(dehazed) = &dehazed {
                *value = dehazed.pixels[index * 4 + c] as f32;
            }
            if let Some(denoised) = &denoised {
                *value += (denoised.pixels[index * 4 + c] as f32
                    - source.pixels[index * 4 + c] as f32)
                    * noise;
            }
            for (base, strength) in [
                (&clarity_base, clarity * 0.8),
                (&sharpening_base, sharpness * 1.2),
                (&texture_base, texture * 0.75),
            ] {
                if let Some(base) = base {
                    *value += (source.pixels[index * 4 + c] as f32
                        - base.pixels[index * 4 + c] as f32)
                        * strength;
                }
            }
            *value = libm::powf(
                ((*value - black_point) / (white_point - black_point).max(1.0)).clamp(0.0, 1.0),
                1.0 / gamma,
            ) * 255.0;
            // Exposure is an EV multiplier in linear light, rather than a
            // multiplication of gamma-encoded display bytes.
            if exposure != 1.0 {
                *value = linear_to_srgb(srgb_to_linear(*value) * exposure);
            }
            *value = ((*value + brightness + adjustment) - 128.0) * contrast + 128.0;
        }
        rgb[0] += temperature + tint * 0.3;
        rgb[1] -= tint;
        rgb[2] -= temperature - tint * 0.3;
        let gray = luma(rgb[0], rgb[1], rgb[2]);
        let max = rgb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = rgb.iter().copied().fold(f32::INFINITY, f32::min);
        let natural = 1.0 + vibrance * (1.0 - ((max - min) / 255.0).clamp(0.0, 1.0)) * 0.9;
        for c in 0..3 {
            pixel[c] = byte(gray + (rgb[c] - gray) * saturation * natural);
        }
        let monochrome = if p
            .get("blackWhite")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            1.0
        } else {
            positive(p, "blackWhite")
        };
        if monochrome > 0.0 {
            let gray = luma(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            for value in &mut pixel[..3] {
                *value = byte(*value as f32 + (gray - *value as f32) * monochrome);
            }
        }
    }
    out
}

pub(crate) fn curves(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let points = p
        .get("points")
        .and_then(Value::as_array)
        .filter(|v| v.len() >= 2)
        .ok_or_else(|| {
            ImageError::InvalidOperation("curves requires at least two points".into())
        })?;
    let mut values: Vec<(f32, f32)> = points
        .iter()
        .map(|p| {
            (
                number(p, "x", 0.0).clamp(0.0, 1.0),
                number(p, "y", 0.0).clamp(0.0, 1.0),
            )
        })
        .collect();
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    values.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    if values.len() < 2 {
        return Err(ImageError::InvalidOperation(
            "curves x coordinates must be distinct".into(),
        ));
    }
    let mut lookup = [0u8; 256];
    for (index, value) in lookup.iter_mut().enumerate() {
        let x = index as f32 / 255.0;
        let y = if x <= values[0].0 {
            values[0].1
        } else if x >= values[values.len() - 1].0 {
            values[values.len() - 1].1
        } else {
            let pair = values
                .windows(2)
                .find(|pair| x >= pair[0].0 && x <= pair[1].0)
                .unwrap();
            let t = (x - pair[0].0) / (pair[1].0 - pair[0].0);
            pair[0].1 * (1.0 - t) + pair[1].1 * t
        };
        *value = byte(y * 255.0);
    }
    let channel = p.get("channel").and_then(Value::as_str).unwrap_or("rgb");
    let mut out = source.clone();
    for pixel in out.pixels.chunks_exact_mut(4) {
        for (c, value) in pixel.iter_mut().enumerate().take(3) {
            if channel == "rgb"
                || channel == "master"
                || channel == ["red", "green", "blue"][c]
                || channel == ["r", "g", "b"][c]
            {
                *value = lookup[*value as usize];
            }
        }
    }
    Ok(out)
}

fn rgb_to_hsl(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0] / 255.0;
    let g = rgb[1] / 255.0;
    let b = rgb[2] / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let light = (max + min) * 0.5;
    if delta < 1e-6 {
        return (0.0, 0.0, light);
    }
    let sat = delta / (1.0 - (2.0 * light - 1.0).abs()).max(1e-6);
    let hue = if max == r {
        60.0 * ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    (hue, sat, light)
}

fn hsl_to_rgb(hue: f32, sat: f32, light: f32) -> [f32; 3] {
    let h = hue.rem_euclid(360.0) / 60.0;
    let s = sat.clamp(0.0, 1.0);
    let l = light.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (h.rem_euclid(2.0) - 1.0).abs());
    let m = l - c * 0.5;
    let [r, g, b] = match h as u32 {
        0 => [c, x, 0.0],
        1 => [x, c, 0.0],
        2 => [0.0, c, x],
        3 => [0.0, x, c],
        4 => [x, 0.0, c],
        _ => [c, 0.0, x],
    };
    [(r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0]
}

pub(crate) fn hsl(source: &RgbaImage, p: &Value) -> RgbaImage {
    let settings = p
        .get("hsl")
        .or_else(|| p.get("channels"))
        .and_then(Value::as_array);
    let mut out = source.clone();
    for pixel in out.pixels.chunks_exact_mut(4) {
        let (h, s, l) = rgb_to_hsl([pixel[0] as f32, pixel[1] as f32, pixel[2] as f32]);
        let mut dh = percent(p, "hue") * 180.0;
        let mut ds = percent(p, "saturation");
        let mut dl = percent(p, "lightness") * 0.35;
        if let Some(settings) = settings {
            for setting in settings {
                let center = number(setting, "hue", 0.0).rem_euclid(360.0);
                let range = number(setting, "range", 45.0).clamp(1.0, 180.0);
                let distance = (h - center + 180.0).rem_euclid(360.0) - 180.0;
                // Achromatic pixels have no defined hue. A red-range
                // lightness adjustment must not treat every gray as red.
                let weight = (1.0 - distance.abs() / range).max(0.0) * smoothstep(0.02, 0.12, s);
                dh += percent(setting, "shift") * 60.0 * weight;
                ds += percent(setting, "saturation") * weight;
                dl += percent(setting, "lightness") * 0.35 * weight;
            }
        }
        let rgb = hsl_to_rgb(h + dh, s * (1.0 + ds), l + dl);
        for c in 0..3 {
            pixel[c] = byte(rgb[c]);
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct LandmarkBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn landmark_box(params: &Value) -> Option<LandmarkBox> {
    let points = params.get("landmarks")?.as_array()?;
    let mut min_x: f32 = 1.0;
    let mut min_y: f32 = 1.0;
    let mut max_x: f32 = 0.0;
    let mut max_y: f32 = 0.0;
    let mut count = 0usize;
    for point in points {
        let (Some(x), Some(y)) = (
            point.get("x").and_then(Value::as_f64).map(|v| v as f32),
            point.get("y").and_then(Value::as_f64).map(|v| v as f32),
        ) else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        count += 1;
    }
    if count < 3 || max_x - min_x < 0.01 || max_y - min_y < 0.01 {
        return None;
    }
    Some(LandmarkBox {
        x: min_x,
        y: min_y,
        width: (max_x - min_x).max(0.001),
        height: (max_y - min_y).max(0.001),
    })
}

fn ellipse_weight(x: f32, y: f32, cx: f32, cy: f32, rx: f32, ry: f32) -> f32 {
    let distance = ((x - cx) / rx.max(1e-4)).powi(2) + ((y - cy) / ry.max(1e-4)).powi(2);
    if distance >= 1.0 {
        0.0
    } else {
        // A squared falloff avoids a hard edge on detector-derived regions.
        (1.0 - distance).powi(2)
    }
}

/// Infer semantic facial regions when a host sends only MediaPipe landmarks.
/// The landmarks' bounding box anchors the regions, so they follow a tilted or
/// differently sized face instead of using a canvas-wide fixed ellipse.
fn inferred_semantic_weight(bbox: LandmarkBox, name: &str, x: f32, y: f32) -> Option<f32> {
    let cx = bbox.x + bbox.width * 0.5;
    let top = bbox.y;
    let h = bbox.height;
    let w = bbox.width;
    let weight = match name {
        "face" | "faceOutline" | "skin" => {
            ellipse_weight(x, y, cx, top + h * 0.5, w * 0.52, h * 0.53)
        }
        "leftEye" => ellipse_weight(x, y, bbox.x + w * 0.31, top + h * 0.42, w * 0.19, h * 0.095),
        "rightEye" => ellipse_weight(x, y, bbox.x + w * 0.69, top + h * 0.42, w * 0.19, h * 0.095),
        "eyes" => inferred_semantic_weight(bbox, "leftEye", x, y)
            .unwrap_or(0.0)
            .max(inferred_semantic_weight(bbox, "rightEye", x, y).unwrap_or(0.0)),
        "leftBrow" => ellipse_weight(x, y, bbox.x + w * 0.31, top + h * 0.29, w * 0.20, h * 0.065),
        "rightBrow" => ellipse_weight(x, y, bbox.x + w * 0.69, top + h * 0.29, w * 0.20, h * 0.065),
        "brows" => inferred_semantic_weight(bbox, "leftBrow", x, y)
            .unwrap_or(0.0)
            .max(inferred_semantic_weight(bbox, "rightBrow", x, y).unwrap_or(0.0)),
        "leftUnderEye" => {
            ellipse_weight(x, y, bbox.x + w * 0.31, top + h * 0.53, w * 0.21, h * 0.075)
        }
        "rightUnderEye" => {
            ellipse_weight(x, y, bbox.x + w * 0.69, top + h * 0.53, w * 0.21, h * 0.075)
        }
        "underEyes" => inferred_semantic_weight(bbox, "leftUnderEye", x, y)
            .unwrap_or(0.0)
            .max(inferred_semantic_weight(bbox, "rightUnderEye", x, y).unwrap_or(0.0)),
        "lips" | "lip" => ellipse_weight(x, y, cx, top + h * 0.76, w * 0.27, h * 0.105),
        "teeth" => ellipse_weight(x, y, cx, top + h * 0.755, w * 0.19, h * 0.045),
        _ => return None,
    };
    Some(weight)
}

#[cfg(test)]
fn semantic_mask_value(mask: &Value, x: f32, y: f32, width: u32, height: u32) -> f32 {
    if mask.is_array() {
        // Compact array form is accepted for AI tools and native callers.
        let polygon = json!({"shape":"polygon", "points": mask});
        return crate::mask::coverage(&polygon, x, y, width, height);
    }
    if !mask.is_object() {
        return 0.0;
    }
    if mask.get("shape").is_none() && mask.get("kind").is_none() && mask.get("points").is_some() {
        let mut polygon = mask.clone();
        if let Some(object) = polygon.as_object_mut() {
            object.insert("shape".into(), Value::String("polygon".into()));
        }
        return crate::mask::coverage(&polygon, x, y, width, height);
    }
    crate::mask::coverage(mask, x, y, width, height)
}

fn semantic_aliases(semantic: &str) -> Option<&'static [&'static str]> {
    Some(match semantic {
        "face" => &["face", "faceOutline", "skin"],
        "skin" => &["skin", "face", "faceOutline"],
        "eyes" => &["eyes", "leftEye", "rightEye"],
        "brows" => &["brows", "leftBrow", "rightBrow"],
        "underEyes" => &["underEyes", "leftUnderEye", "rightUnderEye"],
        "lips" => &["lips", "lip"],
        "teeth" => &["teeth"],
        "hair" => &["hair"],
        "accessories" => &["accessories"],
        "ears" => &["ears"],
        "skinProtection" => &["skinProtection"],
        _ => return None,
    })
}

#[cfg(test)]
fn semantic_weight(
    params: &Value,
    bbox: Option<LandmarkBox>,
    semantic: &str,
    x: f32,
    y: f32,
    width: u32,
    height: u32,
) -> Option<f32> {
    // `skin` may be a pixel parsing result narrower than the face silhouette.
    // It is authoritative, including zero coverage; unioning it with `face`
    // would silently put hair, nostrils or exclusions back into the skin pass.
    if semantic == "skin" {
        if let Some(mask) = params
            .get("semanticMasks")
            .and_then(|masks| masks.get("skin"))
        {
            return Some(semantic_mask_value(mask, x, y, width, height));
        }
    }
    let aliases = semantic_aliases(semantic)?;
    if let Some(masks) = params.get("semanticMasks").and_then(Value::as_object) {
        let mut found = false;
        let mut result: f32 = 0.0;
        for name in aliases {
            if let Some(mask) = masks.get(*name) {
                found = true;
                result = result.max(semantic_mask_value(mask, x, y, width, height));
            }
        }
        if found {
            return Some(result);
        }
    }
    bbox.and_then(|box_| {
        aliases
            .iter()
            .filter_map(|name| inferred_semantic_weight(box_, name, x, y))
            .reduce(f32::max)
    })
}

struct PreparedSemantic<'a> {
    masks: Vec<crate::mask::PreparedMask<'a>>,
    aliases: &'static [&'static str],
    bbox: Option<LandmarkBox>,
}
impl<'a> PreparedSemantic<'a> {
    fn new(params: &'a Value, bbox: Option<LandmarkBox>, name: &str) -> Self {
        let mut aliases = semantic_aliases(name).unwrap_or(&[]);
        let masks = params.get("semanticMasks");
        if name == "skin" && masks.and_then(|m| m.get("skin")).is_some() {
            aliases = &["skin"];
        }
        let prepared = aliases
            .iter()
            .filter_map(|name| masks.and_then(|m| m.get(*name)))
            .map(crate::mask::PreparedMask::new)
            .collect();
        Self {
            masks: prepared,
            aliases,
            bbox,
        }
    }
    fn coverage(&self, x: f32, y: f32, width: u32, height: u32) -> Option<f32> {
        if !self.masks.is_empty() {
            return self
                .masks
                .iter()
                .map(|m| m.coverage(x, y, width, height))
                .reduce(f32::max);
        }
        self.bbox.and_then(|b| {
            self.aliases
                .iter()
                .filter_map(|name| inferred_semantic_weight(b, name, x, y))
                .reduce(f32::max)
        })
    }
}

/// Keep both edit coverage and eligible smoothing donors in the face ROI.
/// Protected pixels do not contribute to the low-frequency skin estimate,
/// even where a feathered protection boundary gives partial edit coverage.
struct LocalSelection {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    weights: Vec<(f32, f32)>,
}

impl LocalSelection {
    fn region(source: &RgbaImage, params: &Value) -> Self {
        let region = Region::read(params.get("region"));
        let x = (region.x * source.width as f32)
            .floor()
            .clamp(0.0, source.width as f32) as u32;
        let y = (region.y * source.height as f32)
            .floor()
            .clamp(0.0, source.height as f32) as u32;
        let right = ((region.x + region.width) * source.width as f32)
            .ceil()
            .clamp(0.0, source.width as f32) as u32;
        let bottom = ((region.y + region.height) * source.height as f32)
            .ceil()
            .clamp(0.0, source.height as f32) as u32;
        let width = right.saturating_sub(x);
        let height = bottom.saturating_sub(y);
        Self {
            x,
            y,
            width,
            height,
            weights: Vec::with_capacity(width as usize * height as usize),
        }
    }

    fn skin(source: &RgbaImage, params: &Value, bbox: Option<LandmarkBox>) -> Self {
        let region = Region::read(params.get("region"));
        let mut selection = Self::region(source, params);
        let semantic_enabled = bbox.is_some() || params.get("semanticMasks").is_some();
        let skin_region = PreparedSemantic::new(params, bbox, "skin");
        let exclusions: Vec<_> = [
            "eyes",
            "brows",
            "lips",
            "teeth",
            "hair",
            "accessories",
            "ears",
            "skinProtection",
        ]
        .into_iter()
        .map(|name| PreparedSemantic::new(params, bbox, name))
        .collect();
        let frozen_mask = params
            .get("protectMask")
            .map(crate::mask::PreparedMask::new);
        let selected_mask = params.get("mask").map(crate::mask::PreparedMask::new);
        for py in selection.y..selection.y + selection.height {
            for px in selection.x..selection.x + selection.width {
                let nx = (px as f32 + 0.5) / source.width as f32;
                let ny = (py as f32 + 0.5) / source.height as f32;
                if !region.contains(nx, ny) {
                    selection.weights.push((0.0, 0.0));
                    continue;
                }
                let skin = skin_region
                    .coverage(nx, ny, source.width, source.height)
                    .unwrap_or(1.0);
                if skin <= 1e-4 {
                    selection.weights.push((0.0, 0.0));
                    continue;
                }
                let protected = if semantic_enabled {
                    exclusions
                        .iter()
                        .filter_map(|mask| mask.coverage(nx, ny, source.width, source.height))
                        .fold(0.0, f32::max)
                } else {
                    0.0
                };
                let frozen = frozen_mask
                    .as_ref()
                    .map(|mask| mask.coverage(nx, ny, source.width, source.height))
                    .unwrap_or(0.0);
                let selected = selected_mask
                    .as_ref()
                    .map(|mask| mask.coverage(nx, ny, source.width, source.height))
                    .unwrap_or(1.0);
                let alpha = source.pixels
                    [(py as usize * source.width as usize + px as usize) * 4 + 3]
                    as f32
                    / 255.0;
                let donor = if protected <= 1e-4 && frozen <= 1e-4 {
                    skin * alpha * selected
                } else {
                    0.0
                };
                selection.weights.push((skin * (1.0 - protected), donor));
            }
        }
        selection
    }

    fn clothing(source: &RgbaImage, params: &Value) -> Self {
        let region = Region::read(params.get("region"));
        let mut selection = Self::region(source, params);
        for py in selection.y..selection.y + selection.height {
            for px in selection.x..selection.x + selection.width {
                let nx = (px as f32 + 0.5) / source.width as f32;
                let ny = (py as f32 + 0.5) / source.height as f32;
                let selected = if region.contains(nx, ny) {
                    params
                        .get("mask")
                        .map(|mask| {
                            crate::mask::coverage(mask, nx, ny, source.width, source.height)
                        })
                        .unwrap_or(1.0)
                } else {
                    0.0
                };
                let frozen = params
                    .get("protectMask")
                    .map(|mask| crate::mask::coverage(mask, nx, ny, source.width, source.height))
                    .unwrap_or(0.0);
                let alpha = source.pixels
                    [(py as usize * source.width as usize + px as usize) * 4 + 3]
                    as f32
                    / 255.0;
                // Even a feathered freeze edge must not donate its original
                // pigment into the selected garment. Final edit coverage is
                // applied once by the graph compositor.
                let donor = if frozen <= 1e-4 {
                    selected * alpha
                } else {
                    0.0
                };
                selection.weights.push((selected * (1.0 - frozen), donor));
            }
        }
        selection
    }
}

/// Normalized convolution reuses the image crate's Gaussian implementation.
/// Blur RGB multiplied by eligible donor weight and blur that weight in the
/// fourth channel, then divide at read time. This avoids pulling eye, brow,
/// hair or lip color into nearby skin and avoids dark seams at mask borders.
fn selection_low_frequency(
    source: &RgbaImage,
    selection: &LocalSelection,
    sigma: f32,
) -> image::Rgba32FImage {
    let weighted = image::Rgba32FImage::from_fn(selection.width, selection.height, |x, y| {
        let weight = selection.weights[(y * selection.width + x) as usize].1;
        let index =
            ((y + selection.y) as usize * source.width as usize + (x + selection.x) as usize) * 4;
        image::Rgba([
            source.pixels[index] as f32 * weight,
            source.pixels[index + 1] as f32 * weight,
            source.pixels[index + 2] as f32 * weight,
            weight,
        ])
    });
    image::imageops::blur(&weighted, sigma)
}

/// Normalized local convolution is shared by spatial color effects. Pixels
/// outside a selection and frozen/transparent pixels cannot lend RGB color.
fn local_blurred(source: &RgbaImage, params: &Value, sigma: f32) -> RgbaImage {
    let selection = LocalSelection::clothing(source, params);
    if selection.width == 0 || selection.height == 0 {
        return source.clone();
    }
    let low = selection_low_frequency(source, &selection, sigma);
    let mut output = source.clone();
    for (i, &(selected, _)) in selection.weights.iter().enumerate() {
        if selected <= 1e-4 {
            continue;
        }
        let x = i as u32 % selection.width;
        let y = i as u32 / selection.width;
        let base = low.get_pixel(x, y).0;
        if base[3] <= 1e-4 {
            continue;
        }
        let index =
            ((y + selection.y) as usize * source.width as usize + (x + selection.x) as usize) * 4;
        for (c, value) in output.pixels[index..index + 3].iter_mut().enumerate() {
            *value = byte(base[c] / base[3]);
        }
    }
    output
}

/// A bilateral neighborhood reduces small color noise while a range kernel
/// stops high-contrast silhouettes and facial features from crossing edges.
fn bilateral_denoise(source: &RgbaImage, params: &Value, amount: f32) -> RgbaImage {
    let selection = LocalSelection::clothing(source, params);
    let mut output = source.clone();
    let step = (source.width.min(source.height) as f32 / 1024.0)
        .ceil()
        .clamp(1.0, 4.0) as i32;
    let range_sigma = 7.0 + amount * 18.0;
    for (i, &(selected, _)) in selection.weights.iter().enumerate() {
        if selected <= 1e-4 {
            continue;
        }
        let x = i as u32 % selection.width;
        let y = i as u32 / selection.width;
        let index =
            ((y + selection.y) as usize * source.width as usize + (x + selection.x) as usize) * 4;
        let mut rgb = [0.0; 3];
        let mut total = 0.0;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let sx = x as i32 + dx * step;
                let sy = y as i32 + dy * step;
                if sx < 0 || sy < 0 || sx >= selection.width as i32 || sy >= selection.height as i32
                {
                    continue;
                }
                let donor =
                    selection.weights[sy as usize * selection.width as usize + sx as usize].1;
                if donor <= 1e-4 {
                    continue;
                }
                let sample_index = ((sy as u32 + selection.y) as usize * source.width as usize
                    + (sx as u32 + selection.x) as usize)
                    * 4;
                let distance: f32 = (0..3)
                    .map(|c| {
                        (source.pixels[sample_index + c] as f32 - source.pixels[index + c] as f32)
                            .powi(2)
                    })
                    .sum::<f32>()
                    / 3.0;
                let spatial = (dx * dx + dy * dy) as f32 / (2.0 * 1.25f32.powi(2));
                let range = distance / (2.0 * range_sigma.powi(2));
                let weight = (-spatial - range).exp() * donor;
                for (c, value) in rgb.iter_mut().enumerate() {
                    *value += source.pixels[sample_index + c] as f32 * weight;
                }
                total += weight;
            }
        }
        if total > 1e-4 {
            for (c, value) in output.pixels[index..index + 3].iter_mut().enumerate() {
                *value = byte(rgb[c] / total);
            }
        }
    }
    output
}

fn minimum_filter(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    use std::collections::VecDeque;
    let mut horizontal = vec![0.0; values.len()];
    let mut result = vec![0.0; values.len()];
    for y in 0..height {
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut next = 0;
        for x in 0..width {
            let right = (x + radius).min(width - 1);
            while next <= right {
                while queue
                    .back()
                    .is_some_and(|&j| values[y * width + j] >= values[y * width + next])
                {
                    queue.pop_back();
                }
                queue.push_back(next);
                next += 1;
            }
            while queue.front().is_some_and(|&j| j < x.saturating_sub(radius)) {
                queue.pop_front();
            }
            horizontal[y * width + x] = values[y * width + queue[0]];
        }
    }
    for x in 0..width {
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut next = 0;
        for y in 0..height {
            let bottom = (y + radius).min(height - 1);
            while next <= bottom {
                while queue
                    .back()
                    .is_some_and(|&j| horizontal[j * width + x] >= horizontal[next * width + x])
                {
                    queue.pop_back();
                }
                queue.push_back(next);
                next += 1;
            }
            while queue.front().is_some_and(|&j| j < y.saturating_sub(radius)) {
                queue.pop_front();
            }
            result[y * width + x] = horizontal[queue[0] * width + x];
        }
    }
    result
}

/// Weighted box statistics for the guided filter, accumulated in f64 with a
/// fixed traversal order. Unlike dynamic Gaussian kernels this needs no
/// platform exp function; ineligible mask/alpha texels do not dilute the mean.
fn guided_box_means(
    values: &[[f64; 4]],
    eligible: &[bool],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<[f64; 4]> {
    let stride = width + 1;
    let mut prefix = vec![[0.0; 5]; stride * (height + 1)];
    for y in 0..height {
        let mut row = [0.0; 5];
        for x in 0..width {
            let i = y * width + x;
            if eligible[i] {
                for (c, value) in values[i].iter().enumerate() {
                    row[c] += value;
                }
                row[4] += 1.0;
            }
            for (c, value) in row.iter().enumerate() {
                prefix[(y + 1) * stride + x + 1][c] = prefix[y * stride + x + 1][c] + value;
            }
        }
    }
    let mut output = vec![[0.0; 4]; values.len()];
    for y in 0..height {
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius + 1).min(height);
            let sum = |c: usize| {
                prefix[y1 * stride + x1][c]
                    - prefix[y0 * stride + x1][c]
                    - prefix[y1 * stride + x0][c]
                    + prefix[y0 * stride + x0][c]
            };
            let count = sum(4).max(1.0);
            for (c, value) in output[y * width + x].iter_mut().enumerate() {
                *value = sum(c) / count;
            }
        }
    }
    output
}

/// Dark-channel prior estimates atmospheric light and transmission; a
/// luminance-guided local linear filter refines transmission around edges.
/// Computation uses a bounded analysis image and linear-light RGB. This is a
/// classical dehazing model, not a generic contrast curve or learned model.
fn atmospheric_dehaze(source: &RgbaImage, params: &Value) -> RgbaImage {
    let amount = percent(params, "dehaze");
    let selection = LocalSelection::clothing(source, params);
    if selection.width == 0 || selection.height == 0 {
        return source.clone();
    }
    let scale = (256.0 / selection.width.max(selection.height) as f32).min(1.0);
    let width = ((selection.width as f32 * scale).round() as usize).max(1);
    let height = ((selection.height as f32 * scale).round() as usize).max(1);
    let mut colors = vec![[0.0; 3]; width * height];
    let mut eligible = vec![false; width * height];
    let mut darkest = vec![1.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let sx = ((x as f32 + 0.5) * selection.width as f32 / width as f32).floor() as u32;
            let sy = ((y as f32 + 0.5) * selection.height as f32 / height as f32).floor() as u32;
            let j = (sy * selection.width + sx) as usize;
            if selection.weights[j].1 <= 1e-4 {
                continue;
            }
            let i = y * width + x;
            let index = ((sy + selection.y) as usize * source.width as usize
                + (sx + selection.x) as usize)
                * 4;
            colors[i] = std::array::from_fn(|c| srgb_to_linear(source.pixels[index + c] as f32));
            darkest[i] = colors[i].into_iter().fold(1.0, f32::min);
            eligible[i] = true;
        }
    }
    let radius = ((width.min(height) as f32 * 0.035).round() as usize).clamp(1, 7);
    let dark_channel = minimum_filter(&darkest, width, height, radius);
    let mut candidates: Vec<usize> = (0..colors.len()).filter(|&i| eligible[i]).collect();
    if candidates.is_empty() {
        return source.clone();
    }
    let min_light = candidates
        .iter()
        .map(|&i| luma(colors[i][0], colors[i][1], colors[i][2]))
        .fold(1.0, f32::min);
    let max_light = candidates
        .iter()
        .map(|&i| luma(colors[i][0], colors[i][1], colors[i][2]))
        .fold(0.0, f32::max);
    // A single flat patch has no identifiable atmospheric veil. Preserve it
    // rather than inventing a scene contrast from an ill-conditioned estimate.
    if max_light - min_light < 1e-5 {
        return source.clone();
    }
    candidates.sort_unstable_by(|&a, &b| dark_channel[b].total_cmp(&dark_channel[a]));
    let top = candidates.len().div_ceil(1000).max(1);
    let atmospheric_index = *candidates[..top]
        .iter()
        .max_by(|&&a, &&b| {
            luma(colors[a][0], colors[a][1], colors[a][2]).total_cmp(&luma(
                colors[b][0],
                colors[b][1],
                colors[b][2],
            ))
        })
        .unwrap();
    let atmosphere = colors[atmospheric_index].map(|v| v.max(0.001));
    let normalized: Vec<f32> = colors
        .iter()
        .enumerate()
        .map(|(i, rgb)| {
            if eligible[i] {
                (0..3).map(|c| rgb[c] / atmosphere[c]).fold(1.0, f32::min)
            } else {
                1.0
            }
        })
        .collect();
    let local_dark = minimum_filter(&normalized, width, height, radius);
    let transmission: Vec<f32> = local_dark
        .iter()
        .map(|&v| (1.0 - 0.95 * v).clamp(0.10, 1.0))
        .collect();
    let guide: Vec<[f64; 4]> = (0..colors.len())
        .map(|i| {
            let light = luma(colors[i][0], colors[i][1], colors[i][2]) as f64;
            let transmission = transmission[i] as f64;
            [light, transmission, light * light, light * transmission]
        })
        .collect();
    let stats = guided_box_means(&guide, &eligible, width, height, 4);
    let coefficients: Vec<[f64; 4]> = stats
        .into_iter()
        .map(|m| {
            let a = (m[3] - m[0] * m[1]) / ((m[2] - m[0] * m[0]).max(0.0) + 0.001);
            [a, m[1] - a * m[0], 0.0, 0.0]
        })
        .collect();
    let mean_coefficients = guided_box_means(&coefficients, &eligible, width, height, 4);
    let refined = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_fn(
        width as u32,
        height as u32,
        |x, y| {
            let i = y as usize * width + x as usize;
            let m = mean_coefficients[i];
            let light = guide[i][0];
            image::Luma([(m[0] * light + m[1]).clamp(0.10, 1.0) as f32])
        },
    );
    let full = image::imageops::resize(
        &refined,
        selection.width,
        selection.height,
        image::imageops::FilterType::Triangle,
    );
    let mut output = source.clone();
    for (i, &(selected, _)) in selection.weights.iter().enumerate() {
        if selected <= 1e-4 {
            continue;
        }
        let x = i as u32 % selection.width;
        let y = i as u32 / selection.width;
        let index =
            ((y + selection.y) as usize * source.width as usize + (x + selection.x) as usize) * 4;
        for (c, value) in output.pixels[index..index + 3].iter_mut().enumerate() {
            let original = srgb_to_linear(source.pixels[index + c] as f32);
            let target = if amount >= 0.0 {
                let transmission = 1.0 - amount * (1.0 - full.get_pixel(x, y).0[0]);
                (original - atmosphere[c]) / transmission.max(0.10) + atmosphere[c]
            } else {
                let transmission = 1.0 + amount * 0.65;
                original * transmission + atmosphere[c] * (1.0 - transmission)
            };
            *value = byte(linear_to_srgb(target));
        }
    }
    output
}

/// Mask-normalized Fast Guided Filter (He & Sun, 2015). Fit RGB against a
/// common luminance guide on a bounded analysis raster, then interpolate the
/// linear coefficients and evaluate against the full-resolution guide. This
/// preserves broad shading and edges while smoothing small skin variation;
/// unlike a Gaussian residual gate, a dark speck is not automatically an edge.
struct SkinGuidedBase {
    width: usize,
    height: usize,
    rg: Vec<[f64; 4]>,
    b: Vec<[f64; 4]>,
}

impl SkinGuidedBase {
    fn sample(&self, x: u32, y: u32, selection: &LocalSelection, light: f32) -> [f32; 3] {
        let px = ((x as f64 + 0.5) * self.width as f64 / selection.width as f64 - 0.5)
            .clamp(0.0, (self.width - 1) as f64);
        let py = ((y as f64 + 0.5) * self.height as f64 / selection.height as f64 - 0.5)
            .clamp(0.0, (self.height - 1) as f64);
        let x0 = px.floor() as usize;
        let y0 = py.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = px - x0 as f64;
        let ty = py - y0 as f64;
        let sample = |values: &[[f64; 4]], c| {
            values[y0 * self.width + x0][c] * (1.0 - tx) * (1.0 - ty)
                + values[y0 * self.width + x1][c] * tx * (1.0 - ty)
                + values[y1 * self.width + x0][c] * (1.0 - tx) * ty
                + values[y1 * self.width + x1][c] * tx * ty
        };
        [
            (sample(&self.rg, 0) * light as f64 + sample(&self.rg, 1)) as f32,
            (sample(&self.rg, 2) * light as f64 + sample(&self.rg, 3)) as f32,
            (sample(&self.b, 0) * light as f64 + sample(&self.b, 1)) as f32,
        ]
    }
}

fn skin_guided_base(
    source: &RgbaImage,
    selection: &LocalSelection,
    smooth: f32,
    face_width: f32,
) -> SkinGuidedBase {
    let scale = (512.0 / selection.width.max(selection.height) as f32).min(1.0);
    let width = ((selection.width as f32 * scale).round() as usize).max(1);
    let height = ((selection.height as f32 * scale).round() as usize).max(1);
    // Area accumulation, rather than nearest-neighbor downsampling: protected
    // or transparent RGB never enters the statistics, even at reduced scale.
    let mut colors = vec![[0.0f64; 4]; width * height];
    for (i, &(_, weight)) in selection.weights.iter().enumerate() {
        if weight <= 1e-4 {
            continue;
        }
        let x = i % selection.width as usize;
        let y = i / selection.width as usize;
        let ax = x * width / selection.width as usize;
        let ay = y * height / selection.height as usize;
        let index =
            ((y + selection.y as usize) * source.width as usize + x + selection.x as usize) * 4;
        let cell = &mut colors[ay * width + ax];
        for (c, value) in cell.iter_mut().enumerate().take(3) {
            *value += source.pixels[index + c] as f64 * weight as f64;
        }
        cell[3] += weight as f64;
    }
    let eligible: Vec<bool> = colors.iter().map(|c| c[3] > 1e-4).collect();
    for color in &mut colors {
        if color[3] > 1e-4 {
            for c in 0..3 {
                color[c] /= color[3];
            }
        }
    }
    let guide: Vec<f64> = colors
        .iter()
        .map(|c| luma(c[0] as f32, c[1] as f32, c[2] as f32) as f64)
        .collect();
    let radius = ((face_width * scale * 0.035).round() as usize).clamp(1, 32);
    let moments_a: Vec<[f64; 4]> = colors
        .iter()
        .zip(&guide)
        .map(|(c, &l)| [l, l * l, c[0], c[1]])
        .collect();
    let moments_b: Vec<[f64; 4]> = colors
        .iter()
        .zip(&guide)
        .map(|(c, &l)| [c[2], l * c[0], l * c[1], l * c[2]])
        .collect();
    let means_a = guided_box_means(&moments_a, &eligible, width, height, radius);
    let means_b = guided_box_means(&moments_b, &eligible, width, height, radius);
    let epsilon = (4.0 + 10.0 * smooth as f64).powi(2);
    let mut rg = vec![[0.0; 4]; width * height];
    let mut b = vec![[0.0; 4]; width * height];
    for i in 0..rg.len() {
        let ma = means_a[i];
        let mb = means_b[i];
        let variance = (ma[1] - ma[0] * ma[0]).max(0.0);
        let ar = (mb[1] - ma[0] * ma[2]) / (variance + epsilon);
        let ag = (mb[2] - ma[0] * ma[3]) / (variance + epsilon);
        let ab = (mb[3] - ma[0] * mb[0]) / (variance + epsilon);
        rg[i] = [ar, ma[2] - ar * ma[0], ag, ma[3] - ag * ma[0]];
        b[i] = [ab, mb[0] - ab * ma[0], 0.0, 0.0];
    }
    SkinGuidedBase {
        width,
        height,
        rg: guided_box_means(&rg, &eligible, width, height, radius),
        b: guided_box_means(&b, &eligible, width, height, radius),
    }
}

/// Keep coherent skin grain separate from pixel noise. The guided plane models
/// lighting, the medium plane models local tone, and the fine plane separates
/// single-pixel noise. Compact high-contrast anomalies get their own clean
/// neighborhood estimate; texture retention does not undo that correction.
struct SkinDetailPlanes {
    fine: Vec<[f32; 3]>,
    medium: Vec<[f32; 3]>,
    structure: Vec<f32>,
    blemish: Vec<([f32; 3], f32)>,
}

struct SkinChromaField {
    width: usize,
    height: usize,
    red: Vec<f32>,
    yellow: Vec<f32>,
}

impl SkinChromaField {
    fn sample(&self, x: u32, y: u32, selection: &LocalSelection) -> f32 {
        self.sample_plane(&self.red, x, y, selection)
    }

    fn sample_yellow(&self, x: u32, y: u32, selection: &LocalSelection) -> f32 {
        self.sample_plane(&self.yellow, x, y, selection)
    }

    fn sample_plane(&self, plane: &[f32], x: u32, y: u32, selection: &LocalSelection) -> f32 {
        let px = ((x as f32 + 0.5) * self.width as f32 / selection.width as f32 - 0.5)
            .clamp(0.0, (self.width - 1) as f32);
        let py = ((y as f32 + 0.5) * self.height as f32 / selection.height as f32 - 0.5)
            .clamp(0.0, (self.height - 1) as f32);
        let x0 = px.floor() as usize;
        let y0 = py.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = px - x0 as f32;
        let ty = py - y0 as f32;
        plane[y0 * self.width + x0] * (1.0 - tx) * (1.0 - ty)
            + plane[y0 * self.width + x1] * tx * (1.0 - ty)
            + plane[y1 * self.width + x0] * (1.0 - tx) * ty
            + plane[y1 * self.width + x1] * tx * ty
    }
}

/// A bounded, mask-normalized local quantile of relative red chroma supplies
/// a healthy-neighborhood reference even when a blotch spans many pores.
/// Correct chroma, never the shading or high-frequency luminance texture.
/// This is a color statistic, not a learned diagnosis of acne or pigmentation.
fn skin_chroma_field(
    source: &RgbaImage,
    selection: &LocalSelection,
    face_width: f32,
) -> SkinChromaField {
    const BINS: usize = 64;
    let scale = (192.0 / selection.width.max(selection.height) as f32).min(1.0);
    let width = ((selection.width as f32 * scale).round() as usize).max(1);
    let height = ((selection.height as f32 * scale).round() as usize).max(1);
    let mut cells = vec![[0.0; 3]; width * height];
    for (i, &(_, weight)) in selection.weights.iter().enumerate() {
        if weight <= 1e-4 {
            continue;
        }
        let x = i % selection.width as usize;
        let y = i / selection.width as usize;
        let index =
            ((y + selection.y as usize) * source.width as usize + x + selection.x as usize) * 4;
        let rgb: [f32; 3] = std::array::from_fn(|c| source.pixels[index + c] as f32);
        let light = luma(rgb[0], rgb[1], rgb[2]);
        if !(25.0..=235.0).contains(&light) {
            continue;
        }
        let cell = &mut cells[(y * height / selection.height as usize) * width
            + x * width / selection.width as usize];
        cell[0] += (rgb[0] - rgb[1]) / (light + 20.0) * 100.0 * weight;
        cell[1] += (rgb[1] - rgb[2]) / (light + 20.0) * 100.0 * weight;
        cell[2] += weight;
    }
    let stride = width + 1;
    let mut prefix = vec![[0u32; BINS]; stride * (height + 1)];
    for y in 0..height {
        let mut row = [0u32; BINS];
        for x in 0..width {
            let cell = cells[y * width + x];
            if cell[2] > 1e-4 {
                let red = cell[0] / cell[2];
                let bin = ((red + 16.0) / 1.5).floor().clamp(0.0, 63.0) as usize;
                row[bin] += 1;
            }
            for (bin, &count) in row.iter().enumerate() {
                prefix[(y + 1) * stride + x + 1][bin] = prefix[y * stride + x + 1][bin] + count;
            }
        }
    }
    let radius = ((face_width * scale * 0.055).round() as usize).clamp(3, 20);
    let mut reference = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius + 1).min(height);
            let histogram: [u32; BINS] = std::array::from_fn(|bin| {
                // Each rectangle count is nonnegative; subtract in signed
                // arithmetic to avoid an intermediate unsigned underflow.
                (prefix[y1 * stride + x1][bin] as i64 + prefix[y0 * stride + x0][bin] as i64
                    - prefix[y0 * stride + x1][bin] as i64
                    - prefix[y1 * stride + x0][bin] as i64) as u32
            });
            let count: u32 = histogram.iter().sum();
            if count == 0 {
                continue;
            }
            let target = (count as f32 * 0.35).ceil().max(1.0) as u32;
            let mut cumulative = 0;
            for (bin, &count) in histogram.iter().enumerate() {
                cumulative += count;
                if cumulative >= target {
                    reference[y * width + x] = -16.0 + (bin as f32 + 0.5) * 1.5;
                    break;
                }
            }
        }
    }
    // Recover the complementary yellow chroma from locally ordinary skin.
    // Correcting only red/green turns a dark reddish spot grey/purple when its
    // green/blue chroma is also atypical. Eligible neighbours keep the local
    // skin hue without requiring a global reference skin colour.
    let eligible: Vec<bool> = cells
        .iter()
        .zip(&reference)
        .map(|(cell, &red)| cell[2] > 1e-4 && cell[0] / cell[2] <= red + 2.5)
        .collect();
    let yellow_samples: Vec<[f64; 4]> = cells
        .iter()
        .map(|cell| {
            [
                if cell[2] > 1e-4 {
                    (cell[1] / cell[2]) as f64
                } else {
                    0.0
                },
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect();
    let yellow = guided_box_means(&yellow_samples, &eligible, width, height, radius)
        .into_iter()
        .map(|cell| cell[0] as f32)
        .collect();
    SkinChromaField {
        width,
        height,
        red: reference,
        yellow,
    }
}

fn skin_detail_planes(
    source: &RgbaImage,
    selection: &LocalSelection,
    face_width: f32,
) -> SkinDetailPlanes {
    let width = selection.width as usize;
    let height = selection.height as usize;
    let normalized = |image: image::Rgba32FImage| {
        image
            .pixels()
            .enumerate()
            .map(|(i, p)| {
                if p[3] > 1e-4 {
                    [p[0] / p[3], p[1] / p[3], p[2] / p[3]]
                } else {
                    let index = ((i / width + selection.y as usize) * source.width as usize
                        + i % width
                        + selection.x as usize)
                        * 4;
                    std::array::from_fn(|c| source.pixels[index + c] as f32)
                }
            })
            .collect::<Vec<[f32; 3]>>()
    };
    let fine = normalized(selection_low_frequency(
        source,
        selection,
        (face_width * 0.0015).clamp(0.55, 0.9),
    ));
    let medium = normalized(selection_low_frequency(
        source,
        selection,
        (face_width * 0.012).clamp(1.8, 6.0),
    ));
    let lights: Vec<f32> = medium.iter().map(|c| luma(c[0], c[1], c[2])).collect();
    let mut structure = vec![1.0; width * height];
    let mut candidates = vec![false; width * height];
    let mut residuals = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if selection.weights[i].1 <= 1e-4 {
                continue;
            }
            let index =
                ((y + selection.y as usize) * source.width as usize + x + selection.x as usize) * 4;
            let original: [f32; 3] = std::array::from_fn(|c| source.pixels[index + c] as f32);
            let dx = (lights[y * width + (x + 1).min(width - 1)]
                - lights[y * width + x.saturating_sub(1)])
            .abs();
            let dy = (lights[(y + 1).min(height - 1) * width + x]
                - lights[y.saturating_sub(1) * width + x])
                .abs();
            // Wide lighting/material edges must not acquire detail halos.
            structure[i] = 1.0 - smoothstep(8.0, 25.0, dx.max(dy));
            let darkness = lights[i] - luma(original[0], original[1], original[2]);
            let excess_red = (original[0] - original[1]) - (medium[i][0] - medium[i][1]);
            residuals[i] = darkness.max(excess_red * 1.3);
            candidates[i] = residuals[i] > 7.0 && structure[i] > 0.6;
        }
    }
    let mut visited = vec![false; width * height];
    let mut blemish = vec![([0.0; 3], 0.0); width * height];
    let max_radius = (face_width * 0.028).clamp(4.0, 18.0);
    let max_area = (std::f32::consts::PI * max_radius.powi(2)) as usize;
    let padding = (face_width * 0.02).round().clamp(4.0, 12.0) as usize;
    for start in 0..candidates.len() {
        if !candidates[start] || visited[start] {
            continue;
        }
        let mut component = vec![start];
        visited[start] = true;
        let mut cursor = 0;
        let mut bounds = [start % width, start / width, start % width, start / width];
        let mut safe = true;
        while cursor < component.len() {
            let i = component[cursor];
            cursor += 1;
            let x = i % width;
            let y = i / width;
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let j = ny * width + nx;
                    safe &= selection.weights[j].1 > 0.5;
                    if candidates[j] && !visited[j] {
                        visited[j] = true;
                        component.push(j);
                    }
                }
            }
        }
        let span_x = bounds[2] - bounds[0] + 1;
        let span_y = bounds[3] - bounds[1] + 1;
        let peak = component.iter().map(|&i| residuals[i]).fold(0.0, f32::max);
        // A dark line, crease, beard edge or large shadow is not a spot.
        if !safe
            || (component.len() < 4 && peak < 25.0)
            || component.len() > max_area
            || span_x.max(span_y) as f32 > span_x.min(span_y) as f32 * 2.7
            || component.len() as f32 / ((span_x * span_y) as f32) < 0.3
        {
            continue;
        }
        let mut sum = [0.0; 3];
        let mut count = 0.0;
        for y in bounds[1].saturating_sub(padding)..=(bounds[3] + padding).min(height - 1) {
            for x in bounds[0].saturating_sub(padding)..=(bounds[2] + padding).min(width - 1) {
                let i = y * width + x;
                if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3])
                    || selection.weights[i].1 <= 0.5
                    || residuals[i].abs() > 8.0
                    || structure[i] < 0.8
                {
                    continue;
                }
                let index =
                    ((y + selection.y as usize) * source.width as usize + x + selection.x as usize)
                        * 4;
                let weight = selection.weights[i].1;
                for (c, value) in sum.iter_mut().enumerate() {
                    *value += source.pixels[index + c] as f32 * weight;
                }
                count += weight;
            }
        }
        if count < 8.0 {
            continue;
        }
        let tone = sum.map(|v| v / count);
        for &i in &component {
            let amount = smoothstep(2.0, 14.0, residuals[i]);
            if amount > blemish[i].1 {
                blemish[i] = (tone, amount);
            }
            // A short transition into the surrounding healthy skin avoids a
            // hard connected-component border. No protected donor is admitted.
            let x = i % width;
            let y = i / width;
            for ny in y.saturating_sub(2)..=(y + 2).min(height - 1) {
                for nx in x.saturating_sub(2)..=(x + 2).min(width - 1) {
                    let j = ny * width + nx;
                    if selection.weights[j].1 <= 0.5 || structure[j] < 0.8 {
                        continue;
                    }
                    let distance =
                        ((nx as f32 - x as f32).powi(2) + (ny as f32 - y as f32).powi(2)).sqrt();
                    let spread = amount
                        * (1.0 - distance / 3.0).max(0.0)
                        * smoothstep(2.0, 11.0, residuals[j]);
                    if spread > blemish[j].1 {
                        blemish[j] = (tone, spread);
                    }
                }
            }
        }
    }
    SkinDetailPlanes {
        fine,
        medium,
        structure,
        blemish,
    }
}

pub(crate) fn skin(source: &RgbaImage, p: &Value) -> RgbaImage {
    let values = p.get("adjustments").unwrap_or(p);
    let smooth = positive(values, "smooth");
    let texture = number(values, "texture", 50.0).clamp(0.0, 100.0) / 100.0;
    let whiten = positive(values, "whiten");
    let warmth = percent(values, "warmth");
    let redness = percent(values, "redness");
    let bbox = landmark_box(p);
    let selection = LocalSelection::skin(source, p, bbox);
    if selection.width == 0 || selection.height == 0 {
        return source.clone();
    }
    let face_width = bbox
        .map(|b| b.width * source.width as f32)
        .unwrap_or(selection.width as f32);
    let base = (smooth > 0.0).then(|| skin_guided_base(source, &selection, smooth, face_width));
    let detail = (smooth > 0.0).then(|| skin_detail_planes(source, &selection, face_width));
    // The negative redness control neutralizes local excess erythema even
    // without smoothing. The quantile reference uses eligible skin donors and
    // is independent of the pixel texture, so a red patch does not force all
    // surrounding healthy skin towards green/grey.
    let pigment = (smooth > 0.0 || redness < 0.0).then(|| {
        // An explicit de-red request can use a wider neighbourhood than
        // automatic smoothing so coherent inflammation does not set its own
        // healthy reference. The bounded analysis remains at 192px.
        let reference_width = face_width * if redness < 0.0 { 1.55 } else { 1.0 };
        skin_chroma_field(source, &selection, reference_width)
    });
    let mut out = source.clone();
    for (local_index, &(weight, _)) in selection.weights.iter().enumerate() {
        if weight <= 0.0001 {
            continue;
        }
        let x = local_index as u32 % selection.width + selection.x;
        let y = local_index as u32 / selection.width + selection.y;
        let index = (y as usize * source.width as usize + x as usize) * 4;
        if source.pixels[index + 3] == 0 {
            continue;
        }
        let pixel = &mut out.pixels[index..index + 4];
        let original = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let mut rgb = original;
        if let (Some(base), Some(detail)) = (&base, &detail) {
            let low = base.sample(
                x - selection.x,
                y - selection.y,
                &selection,
                luma(
                    detail.medium[local_index][0],
                    detail.medium[local_index][1],
                    detail.medium[local_index][2],
                ),
            );
            let edge = detail.structure[local_index];
            let (blemish_tone, blemish_amount) = detail.blemish[local_index];
            for c in 0..3 {
                let noise = original[c] - detail.fine[local_index][c];
                let uneven_tone = detail.medium[local_index][c] - low[c];
                let target = original[c]
                    - noise * (0.9 * (1.0 - texture * 0.96)) * edge
                    - uneven_tone * (0.65 * (1.0 - texture * 0.5)) * edge;
                let repaired = target + (blemish_tone[c] - target) * blemish_amount * 0.82;
                rgb[c] += (repaired - original[c]) * smooth * weight;
            }
            if let Some(pigment) = &pigment {
                let light = luma(rgb[0], rgb[1], rgb[2]);
                let reference = pigment.sample(x - selection.x, y - selection.y, &selection);
                let excess = ((rgb[0] - rgb[1]) / (light + 20.0) * 100.0 - reference).max(0.0);
                let correction = (excess - 2.0).max(0.0)
                    * smoothstep(3.0, 10.0, excess)
                    * 0.9
                    * smooth
                    * weight
                    * detail.structure[local_index]
                    * smoothstep(25.0, 80.0, light)
                    * (1.0 - smoothstep(205.0, 245.0, light))
                    * (light + 20.0)
                    / 100.0;
                rgb = luma_chroma_rgb(
                    light,
                    [rgb[0] - light - correction, rgb[1] - light, rgb[2] - light],
                );
            }
        }
        let light = luma(rgb[0], rgb[1], rgb[2]);
        // Lighten skin in the luminance plane. Protect deep shadows and bright
        // highlights so whitening does not wash out their contrast or turn
        // specular highlights into broad white patches.
        let highlight_protection = 1.0 - smoothstep(205.0, 250.0, light);
        let midtone = smoothstep(20.0, 80.0, light) * highlight_protection;
        let target_light = light + whiten * weight * 24.0 * midtone;
        let chroma_strength = weight * midtone;
        let redness_lift = redness.max(0.0);
        let (neutralization, yellow_recovery) = if redness < 0.0 {
            let pigment = pigment
                .as_ref()
                .expect("negative redness prepares a reference");
            let reference = pigment.sample(x - selection.x, y - selection.y, &selection);
            let relative_red = (rgb[0] - rgb[1]) / (light + 20.0) * 100.0;
            let excess = (relative_red - reference).max(0.0);
            // Keep a small uniform response for the colour slider, with most
            // of the correction reserved for neighbourhood outliers. Never
            // invert the red/green relation of the observed skin tone.
            let local =
                (excess - 1.5).max(0.0) * smoothstep(2.0, 7.0, excess) * (light + 20.0) / 100.0;
            let red_correction =
                (2.0 + local).min((rgb[0] - rgb[1]).max(0.0) * 0.65) * -redness * chroma_strength;
            let target_yellow = pigment.sample_yellow(x - selection.x, y - selection.y, &selection);
            let relative_yellow = (rgb[1] - rgb[2]) / (light + 20.0) * 100.0;
            let yellow_correction = ((target_yellow - relative_yellow) * (light + 20.0) / 100.0)
                .clamp(-10.0, 16.0)
                * smoothstep(2.0, 7.0, excess)
                * -redness
                * chroma_strength;
            (red_correction, yellow_correction)
        } else {
            (0.0, 0.0)
        };
        let tint = [
            (warmth * 12.0 + redness_lift * 10.0) * chroma_strength - neutralization,
            -redness_lift * 3.0 * chroma_strength,
            (-warmth * 12.0 - redness_lift * 3.0) * chroma_strength - yellow_recovery,
        ];
        let tint_light = luma(tint[0], tint[1], tint[2]);
        let chroma = std::array::from_fn(|c| rgb[c] - light + tint[c] - tint_light);
        rgb = luma_chroma_rgb(target_light, chroma);
        for c in 0..3 {
            pixel[c] = byte(rgb[c]);
        }
    }
    out
}

/// Clothing smoothing separates a blurred low-frequency plane from the
/// residual fabric detail. This keeps seams and weave visible while reducing
/// broad brightness folds; it is deterministic and works without a model.
fn clothing_retouch(source: &RgbaImage, p: &Value) -> RgbaImage {
    let values = p.get("adjustments").unwrap_or(p);
    let smooth = positive(values, "smooth");
    let texture = number(values, "texture", 70.0).clamp(0.0, 100.0) / 100.0;
    if smooth <= 0.0001 {
        return source.clone();
    }
    let selection = LocalSelection::clothing(source, p);
    if selection.width == 0 || selection.height == 0 {
        return source.clone();
    }
    let low = selection_low_frequency(source, &selection, 1.2 + smooth * 7.0);
    let mut out = source.clone();
    // Keep a minimum amount of high-frequency detail so a shirt does not turn
    // into a plastic surface even when the smooth slider is high.
    let detail_keep = 1.0 - smooth * (1.0 - (0.18 + texture * 0.82) * 0.65);
    for (local_index, &(selected, _)) in selection.weights.iter().enumerate() {
        if selected <= 1e-4 {
            continue;
        }
        let local_x = local_index as u32 % selection.width;
        let local_y = local_index as u32 / selection.width;
        let low = low.get_pixel(local_x, local_y).0;
        if low[3] <= 1e-4 {
            continue;
        }
        let index = ((local_y + selection.y) as usize * source.width as usize
            + (local_x + selection.x) as usize)
            * 4;
        let pixel = &mut out.pixels[index..index + 4];
        for (c, channel) in pixel.iter_mut().enumerate().take(3) {
            let original = source.pixels[index + c] as f32;
            let base = low[c] / low[3];
            let detail = original - base;
            *channel = byte(base + detail * detail_keep);
        }
    }
    out
}

/// Estimate eye-shadow tone from eligible nearby skin, rather than adding a
/// constant brightness to every pixel in an under-eye polygon. An affine
/// lighting plane preserves directional illumination, while transferring only
/// its low-frequency difference leaves pores and fine lower-lid detail intact.
struct UnderEyeCorrection {
    selection: LocalSelection,
    corrections: Vec<[f32; 3]>,
}

struct UnderEyeDonor {
    position: [f64; 3],
    color: [f32; 3],
    weight: f64,
}

/// The ring can cross cheek highlights, blush or a small residual mask error.
/// A robust plane rejects those local outliers instead of moving the healthy
/// reference toward the lightest / most saturated nearby patch.
fn under_eye_reference(donors: &[UnderEyeDonor]) -> Option<[f32; 3]> {
    if donors.len() < 12 {
        return None;
    }
    let mut plane = [[0.0f64; 3]; 3];
    for iteration in 0..3 {
        let mut matrix = [[0.0f64; 3]; 3];
        let mut rhs = [[0.0f64; 3]; 3];
        let mut used = 0;
        for donor in donors {
            let mut weight = donor.weight;
            if iteration > 0 {
                let predicted: [f32; 3] = std::array::from_fn(|c| {
                    (0..3).map(|a| plane[a][c] * donor.position[a]).sum::<f64>() as f32
                });
                let observed_light = luma(donor.color[0], donor.color[1], donor.color[2]);
                let predicted_light = luma(predicted[0], predicted[1], predicted[2]);
                let chroma_error = (0..3)
                    .map(|c| {
                        ((donor.color[c] - observed_light) - (predicted[c] - predicted_light)).abs()
                    })
                    .fold(0.0, f32::max);
                // Tukey's biweight has bounded influence. Smooth falloff
                // avoids a binary switch when a donor moves by one pixel.
                let residual =
                    ((observed_light - predicted_light).abs() / 20.0).max(chroma_error / 14.0);
                weight *= (1.0 - residual * residual).max(0.0).powi(2) as f64;
            }
            if weight <= 1e-4 {
                continue;
            }
            for a in 0..3 {
                for b in 0..3 {
                    matrix[a][b] += donor.position[a] * donor.position[b] * weight;
                    rhs[a][b] += donor.position[a] * donor.color[b] as f64 * weight;
                }
            }
            used += 1;
        }
        let m = matrix;
        let inverse_numerator = [
            [
                m[1][1] * m[2][2] - m[1][2] * m[2][1],
                m[0][2] * m[2][1] - m[0][1] * m[2][2],
                m[0][1] * m[1][2] - m[0][2] * m[1][1],
            ],
            [
                m[1][2] * m[2][0] - m[1][0] * m[2][2],
                m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][2] * m[1][0] - m[0][0] * m[1][2],
            ],
            [
                m[1][0] * m[2][1] - m[1][1] * m[2][0],
                m[0][1] * m[2][0] - m[0][0] * m[2][1],
                m[0][0] * m[1][1] - m[0][1] * m[1][0],
            ],
        ];
        let determinant = (0..3)
            .map(|c| m[0][c] * inverse_numerator[c][0])
            .sum::<f64>();
        // A line of visible donors cannot determine a lighting plane.
        if used < 12 || determinant < m[0][0].powi(3) * 0.002 {
            return None;
        }
        for a in 0..3 {
            for c in 0..3 {
                plane[a][c] = (0..3)
                    .map(|b| inverse_numerator[a][b] * rhs[b][c])
                    .sum::<f64>()
                    / determinant;
            }
        }
    }
    Some(plane[0].map(|v| v as f32))
}

impl UnderEyeCorrection {
    fn sample(&self, x: u32, y: u32) -> [f32; 3] {
        let s = &self.selection;
        if x < s.x || y < s.y || x >= s.x + s.width || y >= s.y + s.height {
            return [0.0; 3];
        }
        self.corrections[((y - s.y) * s.width + x - s.x) as usize]
    }
}

fn under_eye_correction(
    source: &RgbaImage,
    p: &Value,
    bbox: Option<LandmarkBox>,
) -> UnderEyeCorrection {
    let mut selection = LocalSelection::skin(source, p, bbox);
    let count = selection.width as usize * selection.height as usize;
    let mut corrections = vec![[0.0; 3]; count];
    if count == 0 {
        return UnderEyeCorrection {
            selection,
            corrections,
        };
    }
    let under_eyes = PreparedSemantic::new(p, bbox, "underEyes");
    let mut targets = vec![0.0; count];
    for (i, target) in targets.iter_mut().enumerate() {
        let x = i as u32 % selection.width + selection.x;
        let y = i as u32 / selection.width + selection.y;
        let weight = under_eyes
            .coverage(
                (x as f32 + 0.5) / source.width as f32,
                (y as f32 + 0.5) / source.height as f32,
                source.width,
                source.height,
            )
            .unwrap_or(0.0);
        let alpha = source.pixels[(y as usize * source.width as usize + x as usize) * 4 + 3];
        *target = if alpha == 0 {
            0.0
        } else {
            weight * selection.weights[i].0
        };
    }
    if targets.iter().all(|&weight| weight <= 1e-4) {
        return UnderEyeCorrection {
            selection,
            corrections,
        };
    }
    let face_width = bbox
        .map(|b| b.width * source.width as f32)
        .unwrap_or(selection.width as f32);
    let sigma = (face_width * 0.009).clamp(1.2, 4.0);
    let low = selection_low_frequency(source, &selection, sigma);
    // Even the feathered shadow boundary must not become a reference donor.
    for (i, weights) in selection.weights.iter_mut().enumerate() {
        if targets[i] > 1e-4 {
            weights.1 = 0.0;
        }
    }
    let reference_low = selection_low_frequency(source, &selection, sigma);
    let radius = (face_width * 0.13).clamp(4.0, 72.0);
    let offsets: Vec<_> = [0.65f32, 1.0, 1.4]
        .into_iter()
        .flat_map(|ring| {
            (0..24).map(move |step| {
                let angle = step as f32 * std::f32::consts::TAU / 24.0;
                (
                    libm::cosf(angle) * ring,
                    libm::sinf(angle) * ring,
                    1.0 / ring,
                )
            })
        })
        .collect();
    for (i, &weight) in targets.iter().enumerate() {
        if weight <= 1e-4 || low.as_raw()[i * 4 + 3] <= 1e-4 {
            continue;
        }
        let x = i % selection.width as usize;
        let y = i / selection.width as usize;
        let mut donors = Vec::with_capacity(offsets.len());
        let mut limits = [[f32::INFINITY, f32::NEG_INFINITY]; 3];
        for &(ox, oy, spatial_weight) in &offsets {
            let px = x as f32 + ox * radius;
            let py = y as f32 + oy * radius;
            if px < 0.0 || py < 0.0 || px >= selection.width as f32 || py >= selection.height as f32
            {
                continue;
            }
            let px = px.round().min(selection.width as f32 - 1.0) as usize;
            let py = py.round().min(selection.height as f32 - 1.0) as usize;
            let j = py * selection.width as usize + px;
            let donor = selection.weights[j].1;
            let color = reference_low.get_pixel(px as u32, py as u32).0;
            if donor <= 0.5 || color[3] <= 1e-4 {
                continue;
            }
            for a in 0..3 {
                let value = color[a] / color[3];
                limits[a][0] = limits[a][0].min(value);
                limits[a][1] = limits[a][1].max(value);
            }
            donors.push(UnderEyeDonor {
                position: [
                    1.0,
                    (px as f64 - x as f64) / radius as f64,
                    (py as f64 - y as f64) / radius as f64,
                ],
                color: std::array::from_fn(|c| color[c] / color[3]),
                weight: (donor * spatial_weight) as f64,
            });
        }
        let Some(predicted) = under_eye_reference(&donors) else {
            continue;
        };
        let reference: [f32; 3] = std::array::from_fn(|c| {
            predicted[c].clamp(
                (limits[c][0] - 8.0).max(0.0),
                (limits[c][1] + 8.0).min(255.0),
            )
        });
        let center = low.get_pixel(x as u32, y as u32).0;
        let local: [f32; 3] = std::array::from_fn(|c| center[c] / center[3]);
        let reference_light = luma(reference[0], reference[1], reference[2]);
        let local_light = luma(local[0], local[1], local[2]);
        let index =
            (((y as u32 + selection.y) * source.width) + x as u32 + selection.x) as usize * 4;
        let original_light = luma(
            source.pixels[index] as f32,
            source.pixels[index + 1] as f32,
            source.pixels[index + 2] as f32,
        );
        // A broad chromatic shadow is editable; a narrow dark contour may
        // instead be eyelashes, eyeglass frames or a lower-lid crease missed
        // by parsing. Do not turn those outlines into pale painted strokes.
        let contour_keep = 1.0 - smoothstep(12.0, 28.0, local_light - original_light);
        let deficit = reference_light - local_light;
        let confidence = smoothstep(1.5, 7.0, deficit)
            * smoothstep(18.0, 60.0, local_light)
            * (1.0 - smoothstep(205.0, 245.0, local_light));
        // The target is a subtle local fill rather than a flat white wash.
        // Lift a little more of a genuine broad shadow (the confidence gate
        // already excludes healthy skin and highlights), while keeping the
        // chroma transfer separate so blue/brown cast is corrected without
        // bleaching the surrounding texture.
        let lift = deficit.clamp(0.0, 38.0) * 0.88 * confidence;
        for c in 0..3 {
            // Only a dark region gets a subtle correction of its relative
            // blue/red cast; healthy skin and specular highlights stay put.
            let color_difference =
                ((reference[c] - reference_light) - (local[c] - local_light)).clamp(-12.0, 12.0);
            corrections[i][c] =
                (lift + color_difference * confidence * 0.58) * weight * contour_keep;
        }
    }
    UnderEyeCorrection {
        selection,
        corrections,
    }
}

pub(crate) fn local_color(source: &RgbaImage, p: &Value, clothing: bool) -> RgbaImage {
    let values = p.get("adjustments").unwrap_or(p);
    if [
        "smooth",
        "hue",
        "saturation",
        "browColor",
        "lipColor",
        "teethWhiten",
        "brightness",
        "lipBrightness",
        "lipSaturation",
        "darkCircle",
    ]
    .iter()
    .all(|key| number(values, key, 0.0).abs() <= 1e-6)
    {
        return source.clone();
    }
    let region = Region::read(p.get("region"));
    let bbox = landmark_box(p);
    let mut out = source.clone();
    if clothing {
        out = clothing_retouch(&out, p);
    } else if number(values, "smooth", 0.0) > 0.0 {
        out = skin(&out, p);
    }
    let hue = percent(values, "hue") * 180.0;
    let saturation = percent(values, "saturation");
    let brow = percent(values, "browColor");
    let lip = percent(values, "lipColor");
    let teeth = positive(values, "teethWhiten");
    let brightness = percent(values, "brightness") * 75.0;
    let lip_brightness = percent(values, "lipBrightness");
    let lip_saturation = percent(values, "lipSaturation");
    let dark_circle = positive(values, "darkCircle");
    let semantic_enabled = p.get("semanticMasks").is_some() || bbox.is_some();
    let adaptive_under_eye = (dark_circle > 0.0
        && (bbox.is_some() || p.get("semanticMasks").and_then(|m| m.get("skin")).is_some()))
    .then(|| under_eye_correction(source, p, bbox));
    if let Some(correction) = &adaptive_under_eye {
        if [
            "smooth",
            "hue",
            "saturation",
            "browColor",
            "lipColor",
            "teethWhiten",
            "brightness",
            "lipBrightness",
            "lipSaturation",
        ]
        .iter()
        .all(|key| number(values, key, 0.0).abs() <= 1e-6)
        {
            for (i, &delta) in correction.corrections.iter().enumerate() {
                let s = &correction.selection;
                let x = i as u32 % s.width + s.x;
                let y = i as u32 / s.width + s.y;
                let index = (y as usize * source.width as usize + x as usize) * 4;
                for (c, &amount) in delta.iter().enumerate() {
                    out.pixels[index + c] =
                        byte(source.pixels[index + c] as f32 + amount * dark_circle);
                }
            }
            return out;
        }
    }
    let skin_semantic = PreparedSemantic::new(p, bbox, "skin");
    let eye_semantic = PreparedSemantic::new(p, bbox, "eyes");
    let under_eye_semantic = PreparedSemantic::new(p, bbox, "underEyes");
    let brow_semantic = PreparedSemantic::new(p, bbox, "brows");
    let lip_semantic = PreparedSemantic::new(p, bbox, "lips");
    let teeth_semantic = PreparedSemantic::new(p, bbox, "teeth");
    // A brow polygon includes skin between strands. Estimate that skin from
    // eligible surrounding texels rather than darkening the entire polygon.
    let brow_reference = (brow.abs() > 1e-6
        && (bbox.is_some() || p.get("semanticMasks").and_then(|m| m.get("skin")).is_some()))
    .then(|| {
        let selection = LocalSelection::skin(source, p, bbox);
        let face_width = bbox.map_or(region.width, |b| b.width) * source.width as f32;
        let low =
            selection_low_frequency(source, &selection, (face_width * 0.045).clamp(3.0, 28.0));
        (selection, low)
    });
    for y in 0..source.height {
        for x in 0..source.width {
            let nx = (x as f32 + 0.5) / source.width as f32;
            let ny = (y as f32 + 0.5) / source.height as f32;
            if !region.contains(nx, ny) {
                continue;
            }
            let rx = (nx - region.x) / region.width;
            let ry = (ny - region.y) / region.height;
            let index = (y as usize * source.width as usize + x as usize) * 4;
            let mut rgb = [
                out.pixels[index] as f32,
                out.pixels[index + 1] as f32,
                out.pixels[index + 2] as f32,
            ];
            // Generic color controls stay scoped to detected skin when semantic
            // regions are present; old region behavior remains unchanged when
            // callers provide no detector data.
            let skin_weight = if semantic_enabled {
                skin_semantic
                    .coverage(nx, ny, source.width, source.height)
                    .unwrap_or(0.0)
            } else {
                1.0
            };
            for value in &mut rgb {
                *value += brightness * skin_weight;
            }
            let fallback_under_eye = (((rx - 0.32) / 0.20).powi(2) + ((ry - 0.46) / 0.09).powi(2))
                .min(((rx - 0.68) / 0.20).powi(2) + ((ry - 0.46) / 0.09).powi(2));
            let mut under_eye_weight = if semantic_enabled {
                under_eye_semantic
                    .coverage(nx, ny, source.width, source.height)
                    .unwrap_or(0.0)
            } else if fallback_under_eye < 1.0 {
                (1.0 - fallback_under_eye).powi(2)
            } else {
                0.0
            };
            if semantic_enabled {
                under_eye_weight *= 1.0
                    - eye_semantic
                        .coverage(nx, ny, source.width, source.height)
                        .unwrap_or(0.0)
                        .max(
                            brow_semantic
                                .coverage(nx, ny, source.width, source.height)
                                .unwrap_or(0.0),
                        );
            }
            if let Some(correction) = &adaptive_under_eye {
                let delta = correction.sample(x, y);
                for c in 0..3 {
                    rgb[c] += delta[c] * dark_circle;
                }
            } else if under_eye_weight > 0.0 {
                let light = luma(rgb[0], rgb[1], rgb[2]);
                let shadow_lift = (28.0 + (170.0 - light).max(0.0) * 0.08).min(42.0);
                rgb = brighten_luma(
                    rgb,
                    shadow_lift
                        * under_eye_weight
                        * dark_circle
                        * 0.35
                        * (1.0 - smoothstep(205.0, 250.0, light)),
                );
            }
            if hue.abs() > 0.001 || saturation.abs() > 0.001 {
                let (h, s, l) = rgb_to_hsl(rgb);
                rgb = hsl_to_rgb(
                    h + hue * skin_weight,
                    s * (1.0 + saturation * skin_weight),
                    l,
                );
            }
            let fallback_brow = (((rx - 0.32) / 0.21).powi(2) + ((ry - 0.28) / 0.07).powi(2))
                .min(((rx - 0.68) / 0.21).powi(2) + ((ry - 0.28) / 0.07).powi(2));
            let brow_weight = if semantic_enabled {
                brow_semantic
                    .coverage(nx, ny, source.width, source.height)
                    .unwrap_or(0.0)
            } else if fallback_brow < 1.0 {
                (1.0 - fallback_brow).powi(2)
            } else {
                0.0
            };
            if brow_weight > 0.0 && brow.abs() > 1e-6 {
                let reference = brow_reference.as_ref().and_then(|(s, low)| {
                    if x < s.x || y < s.y || x >= s.x + s.width || y >= s.y + s.height {
                        return None;
                    }
                    let color = low.get_pixel(x - s.x, y - s.y).0;
                    (color[3] >= 0.015)
                        .then(|| std::array::from_fn::<_, 3, _>(|c| color[c] / color[3]))
                });
                if let Some(reference) = reference {
                    let original =
                        std::array::from_fn::<_, 3, _>(|c| source.pixels[index + c] as f32);
                    let light = luma(original[0], original[1], original[2]);
                    let skin_light = luma(reference[0], reference[1], reference[2]);
                    let hair = smoothstep(0.04, 0.28, (skin_light - light) / skin_light.max(20.0));
                    let coverage = brow_weight * hair;
                    for c in 0..3 {
                        if brow > 0.0 {
                            rgb[c] = linear_to_srgb(
                                srgb_to_linear(rgb[c]) * (1.0 - brow * coverage * 0.52),
                            );
                        } else {
                            rgb[c] += (reference[c] - original[c]) * (-brow) * coverage * 0.32;
                        }
                    }
                } else {
                    // A standalone hand-drawn brow mask lacks a skin reference.
                    for value in &mut rgb {
                        *value *= 1.0 - brow * brow_weight * 0.5;
                    }
                }
            }
            let fallback_mouth = ((rx - 0.5) / 0.25).powi(2) + ((ry - 0.76) / 0.09).powi(2);
            let mouth_weight = if semantic_enabled {
                lip_semantic
                    .coverage(nx, ny, source.width, source.height)
                    .unwrap_or(0.0)
            } else if fallback_mouth < 1.0 {
                1.0 - fallback_mouth
            } else {
                0.0
            };
            let inner_mouth_weight = if semantic_enabled {
                teeth_semantic
                    .coverage(nx, ny, source.width, source.height)
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            // Outer lip polygons contain the inner mouth. Subtract that area
            // before applying lipstick, lip brightness or saturation; enamel
            // and mouth cavities must not receive the lip color pass.
            // A feathered teeth polygon is a hard exclusion for lipstick. A
            // linear subtraction leaves a colored halo inside enamel because
            // the polygon edge coverage is rarely exactly 1.0 at 8-bit size.
            let teeth_exclusion = (inner_mouth_weight > 1e-4) as u8 as f32;
            let lip_weight = mouth_weight * (1.0 - teeth_exclusion);
            if lip_weight > 0.0 {
                // Change diffuse pigment at fixed luminance. A neutral bright
                // reflection is lighting, and should not become a red stripe.
                let original = std::array::from_fn::<_, 3, _>(|c| source.pixels[index + c] as f32);
                let original_light = luma(original[0], original[1], original[2]);
                let high = original.into_iter().fold(0.0, f32::max);
                let low = original.into_iter().fold(255.0, f32::min);
                let saturation = (high - low) / high.max(1.0);
                let diffuse = 1.0
                    - smoothstep(185.0, 245.0, original_light)
                        * (1.0 - smoothstep(0.06, 0.22, saturation));
                let light = luma(rgb[0], rgb[1], rgb[2]);
                let tint = [35.0, -12.0, -4.0];
                let tint_light = luma(tint[0], tint[1], tint[2]);
                let chroma: [f32; 3] = std::array::from_fn(|c| {
                    (rgb[c] - light) * (1.0 + lip_saturation * lip_weight * diffuse)
                        + (tint[c] - tint_light) * lip * lip_weight * diffuse
                });
                let lift = if lip_brightness >= 0.0 {
                    (255.0 - light) * lip_brightness * 0.32
                } else {
                    light * lip_brightness * 0.32
                };
                rgb = luma_chroma_rgb(light + lift * lip_weight, chroma);
            }
            // This pass also works when the host provides only a teeth mask.
            // Gate from immutable input color so a lipstick pass cannot turn a
            // red pixel into a whitening candidate in the same operation.
            let original_rgb = [
                source.pixels[index] as f32,
                source.pixels[index + 1] as f32,
                source.pixels[index + 2] as f32,
            ];
            let teeth_region = if semantic_enabled {
                inner_mouth_weight
            } else {
                mouth_weight
            };
            let tooth_weight = teeth_region * teeth * tooth_confidence(original_rgb);
            if tooth_weight > 0.0 {
                let light = luma(rgb[0], rgb[1], rgb[2]);
                let target_light = light + (255.0 - light) * tooth_weight * 0.24;
                let chroma = rgb.map(|value| (value - light) * (1.0 - tooth_weight * 0.55));
                rgb = luma_chroma_rgb(target_light, chroma);
            }
            for (dst, value) in out.pixels[index..index + 3].iter_mut().zip(rgb) {
                *dst = byte(value);
            }
        }
    }
    out
}

pub(crate) fn film(source: &RgbaImage, p: &Value) -> RgbaImage {
    let preset = p.get("preset").and_then(Value::as_str).unwrap_or("neutral");
    let intensity = number(p, "intensity", 100.0).clamp(0.0, 100.0) / 100.0;
    let response = crate::film::FilmResponse::new(preset);
    let grain = positive(p, "grain");
    let vignette = positive(p, "vignette");
    let glow = positive(p, "glow");
    let halation = positive(p, "halation");
    let fade = positive(p, "fade");
    let seed = number(p, "seed", 42.0).max(0.0) as u64;
    let light = if glow > 0.0 {
        Some(local_blurred(
            source,
            p,
            (source.width.min(source.height) as f32 * 0.008).clamp(1.0, 24.0),
        ))
    } else {
        None
    };
    let halo = (halation > 0.0).then(|| {
        let mut emission = source.clone();
        for pixel in emission.pixels.chunks_exact_mut(4) {
            let light = luma(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            let gate = byte(smoothstep(180.0, 245.0, light) * 255.0);
            pixel[..3].fill(gate);
        }
        let spread = local_blurred(
            &emission,
            p,
            (source.width.min(source.height) as f32 * 0.006).clamp(1.0, 24.0),
        );
        (emission, spread)
    });
    let mut out = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let index = (y as usize * source.width as usize + x as usize) * 4;
            let original = [
                source.pixels[index] as f32,
                source.pixels[index + 1] as f32,
                source.pixels[index + 2] as f32,
            ];
            let gray = luma(original[0], original[1], original[2]);
            let mut rgb = original;
            match preset {
                "warm" => {
                    rgb[0] = original[0] * 1.035 + 10.0;
                    rgb[1] = original[1] * 1.01 + 3.0;
                    rgb[2] = original[2] * 0.94;
                }
                "cool" => {
                    rgb[0] = original[0] * 0.93;
                    rgb[1] = original[1] + 3.0;
                    rgb[2] = original[2] * 1.04 + 8.0;
                }
                "classic" => {
                    for c in 0..3 {
                        rgb[c] = gray + (original[c] - gray) * 0.78;
                    }
                    rgb[0] += 8.0;
                    rgb[2] -= 8.0;
                }
                "fade" | "pastel" => {
                    for c in 0..3 {
                        rgb[c] = 28.0 + original[c] * 0.82;
                    }
                    rgb[0] += 4.0;
                }
                "vivid" => {
                    for c in 0..3 {
                        rgb[c] = 128.0 + (gray + (original[c] - gray) * 1.25 - 128.0) * 1.10;
                    }
                }
                "mono" => {
                    rgb = [128.0 + (gray - 128.0) * 1.18; 3];
                }
                "cinema" | "night" => {
                    let shadow = 1.0 - gray / 255.0;
                    rgb[0] -= shadow * 13.0;
                    rgb[1] += shadow * 7.0;
                    rgb[2] += shadow * 12.0;
                    rgb[0] += (1.0 - shadow) * 14.0;
                    rgb[2] -= (1.0 - shadow) * 8.0;
                }
                "provia" | "astia" | "velvia" | "classicNegative" | "eterna" | "acros" => {
                    // These profiles use the linear-light response above;
                    // keeping them separate from the legacy presets avoids
                    // changing existing projects when the new response family
                    // is added to the preset list.
                    if let Some(response) = &response {
                        rgb = response.apply(original);
                    }
                }
                _ => {}
            }
            for value in &mut rgb {
                *value = *value * (1.0 - fade * 0.25) + fade * 30.0;
            }
            if let Some(light) = &light {
                let bright = luma(
                    light.pixels[index] as f32,
                    light.pixels[index + 1] as f32,
                    light.pixels[index + 2] as f32,
                );
                let highlight = ((bright - 125.0) / 130.0).clamp(0.0, 1.0);
                for value in &mut rgb {
                    *value += (255.0 - *value) * highlight * glow * 0.35;
                }
            }
            if let Some((emission, spread)) = &halo {
                // Light leaking through film spreads outside an emissive
                // highlight. Subtract its core so halation produces a red
                // rim rather than recoloring the entire bright object.
                let halo = ((spread.pixels[index] as f32 - emission.pixels[index] as f32) / 255.0)
                    .max(0.0)
                    * halation;
                rgb[0] += halo * 60.0;
                rgb[1] += halo * 10.0;
            }
            let nx = (x as f32 + 0.5) / source.width as f32 * 2.0 - 1.0;
            let ny = (y as f32 + 0.5) / source.height as f32 * 2.0 - 1.0;
            let falloff = 1.0 - vignette * (nx * nx + ny * ny).min(2.0) * 0.375;
            let noise = crate::grain_noise(seed, (index / 4) as u64) * grain * 25.0;
            for (c, (dst, value)) in out.pixels[index..index + 3].iter_mut().zip(rgb).enumerate() {
                let effect = value * falloff + noise;
                *dst = byte(original[c] + (effect - original[c]) * intensity);
            }
        }
    }
    out
}

pub(crate) fn lut(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let size = number(p, "size", 0.0) as usize;
    if !(2..=65).contains(&size) {
        return Err(ImageError::InvalidOperation(
            "LUT size must be 2..65".into(),
        ));
    }
    let values = p
        .get("data")
        .and_then(Value::as_array)
        .filter(|v| v.len() == size * size * size)
        .ok_or_else(|| {
            ImageError::InvalidOperation("LUT data must contain size^3 RGB points".into())
        })?;
    let mut entries = Vec::with_capacity(values.len());
    for value in values {
        let entry = value.as_array().filter(|v| v.len() == 3).ok_or_else(|| {
            ImageError::InvalidOperation("LUT point must contain three channels".into())
        })?;
        let mut rgb = [0.0; 3];
        for c in 0..3 {
            rgb[c] = entry[c].as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                ImageError::InvalidOperation("LUT point must contain finite numbers".into())
            })? as f32;
        }
        entries.push(rgb);
    }
    let domain = |name: &str, fallback: f32| -> Result<[f32; 3], ImageError> {
        if let Some(value) = p.get(name) {
            let arr = value.as_array().filter(|v| v.len() == 3).ok_or_else(|| {
                ImageError::InvalidOperation("LUT domain must have three numbers".into())
            })?;
            let mut out = [0.0; 3];
            for c in 0..3 {
                out[c] = arr[c].as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                    ImageError::InvalidOperation("LUT domain must contain finite numbers".into())
                })? as f32;
            }
            Ok(out)
        } else {
            Ok([fallback; 3])
        }
    };
    let min = domain("domainMin", 0.0)?;
    let max = domain("domainMax", 1.0)?;
    if (0..3).any(|c| max[c] <= min[c]) {
        return Err(ImageError::InvalidOperation(
            "LUT domainMax must be greater than domainMin".into(),
        ));
    }
    let amount = number(p, "amount", 100.0).clamp(0.0, 100.0) / 100.0;
    let mut out = source.clone();
    for pixel in out.pixels.chunks_exact_mut(4) {
        let mut coordinate = [0.0; 3];
        for c in 0..3 {
            coordinate[c] = ((pixel[c] as f32 / 255.0 - min[c]) / (max[c] - min[c]))
                .clamp(0.0, 1.0)
                * (size - 1) as f32;
        }
        let lows = coordinate.map(|v| v.floor() as usize);
        let highs = lows.map(|v| (v + 1).min(size - 1));
        let fractions = std::array::from_fn::<_, 3, _>(|c| coordinate[c] - lows[c] as f32);
        let mut color = [0.0; 3];
        for bz in 0..2 {
            for gy in 0..2 {
                for rx in 0..2 {
                    let coords = [
                        if rx == 0 { lows[0] } else { highs[0] },
                        if gy == 0 { lows[1] } else { highs[1] },
                        if bz == 0 { lows[2] } else { highs[2] },
                    ];
                    let weight = (if rx == 0 {
                        1.0 - fractions[0]
                    } else {
                        fractions[0]
                    }) * (if gy == 0 {
                        1.0 - fractions[1]
                    } else {
                        fractions[1]
                    }) * (if bz == 0 {
                        1.0 - fractions[2]
                    } else {
                        fractions[2]
                    });
                    let entry = entries[coords[2] * size * size + coords[1] * size + coords[0]];
                    for c in 0..3 {
                        color[c] += entry[c] * weight;
                    }
                }
            }
        }
        for c in 0..3 {
            pixel[c] = byte(pixel[c] as f32 * (1.0 - amount) + color[c] * 255.0 * amount);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pixel_light(image: &RgbaImage, index: usize) -> f32 {
        luma(
            image.pixels[index * 4] as f32,
            image.pixels[index * 4 + 1] as f32,
            image.pixels[index * 4 + 2] as f32,
        )
    }

    fn chroma_span(image: &RgbaImage, index: usize) -> f32 {
        let pixel = &image.pixels[index * 4..index * 4 + 3];
        *pixel.iter().max().unwrap() as f32 - *pixel.iter().min().unwrap() as f32
    }

    #[test]
    fn basic_neutral_settings_preserve_every_channel_and_alpha() {
        let source = RgbaImage::new(
            4,
            1,
            vec![
                0, 1, 255, 0, 31, 79, 203, 27, 128, 128, 128, 255, 255, 254, 253, 191,
            ],
        )
        .unwrap();
        let params = json!({"exposure":0,"brightness":0,"contrast":0,"highlights":0,"shadows":0,"whites":0,"blacks":0,"temperature":0,"tint":0,"saturation":0,"vibrance":0,"clarity":0,"sharpness":0,"texture":0,"dehaze":0,"noise":0,"blackWhite":0,"blackPoint":0,"whitePoint":100,"gamma":1});
        assert_eq!(basic(&source, &params), source);
    }

    #[test]
    fn software_srgb_transfer_round_trips_every_encoded_byte_monotonically() {
        let mut previous = -1.0;
        for encoded in 0..=255 {
            let linear = srgb_to_linear(encoded as f32);
            assert!(linear > previous);
            assert_eq!(byte(linear_to_srgb(linear)), encoded);
            previous = linear;
        }
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert_eq!(srgb_to_linear(255.0), 1.0);
    }

    #[test]
    fn guided_statistics_use_only_eligible_samples_and_preserve_constant_fields() {
        let samples = vec![[2.0, 3.0, 4.0, 5.0]; 9];
        let eligible = [true, false, true, false, true, false, true, false, true];
        let a = guided_box_means(&samples, &eligible, 3, 3, 1);
        assert!(a.iter().all(|mean| *mean == [2.0, 3.0, 4.0, 5.0]));
        let mutated: Vec<_> = samples
            .into_iter()
            .enumerate()
            .map(|(i, value)| if eligible[i] { value } else { [1e12; 4] })
            .collect();
        assert_eq!(a, guided_box_means(&mutated, &eligible, 3, 3, 1));
        let ramp: Vec<_> = (0..9).map(|i| [i as f64; 4]).collect();
        let means = guided_box_means(&ramp, &[true; 9], 3, 3, 1);
        assert_eq!(means[0], [2.0; 4]);
        assert_eq!(means[4], [4.0; 4]);
        assert_eq!(means[8], [6.0; 4]);
    }

    #[test]
    fn prepared_semantics_match_legacy_aliases_and_compact_polygon_masks() {
        let polygon =
            json!([{"x":0.2,"y":0.2},{"x":0.8,"y":0.2},{"x":0.7,"y":0.8},{"x":0.25,"y":0.75}]);
        let params = json!({"semanticMasks":{
            "face":{"shape":"ellipse","region":{"x":0,"y":0,"width":1,"height":1}},
            "skin":{"points":polygon,"feather":3},
            "leftEye":polygon,
            "rightEye":{"shape":"raster","width":2,"height":2,"data":[0,127,255,30]},
            "leftUnderEye":{"shape":"rect","region":{"x":0.1,"y":0.2,"width":0.3,"height":0.2}},
            "rightUnderEye":{"shape":"radial","region":{"x":0.5,"y":0.2,"width":0.3,"height":0.2}},
            "lips":{"shape":"polygon","points":polygon,"invert":true}
        }});
        let bbox = Some(LandmarkBox {
            x: 0.1,
            y: 0.1,
            width: 0.8,
            height: 0.8,
        });
        for name in [
            "skin",
            "face",
            "eyes",
            "brows",
            "underEyes",
            "lips",
            "teeth",
            "hair",
        ] {
            let prepared = PreparedSemantic::new(&params, bbox, name);
            for y in -2..66 {
                for x in -2..66 {
                    let nx = (x as f32 + 0.5) / 64.0;
                    let ny = (y as f32 + 0.5) / 64.0;
                    let old = semantic_weight(&params, bbox, name, nx, ny, 64, 64);
                    let new = prepared.coverage(nx, ny, 64, 64);
                    match (old, new) {
                        (Some(a), Some(b)) => {
                            assert!((a - b).abs() < 1e-6, "{name} at {x},{y}: {a} != {b}")
                        }
                        (None, None) => {}
                        _ => panic!("semantic existence changed: {name}"),
                    }
                }
            }
        }
    }

    #[test]
    fn exposure_obeys_linear_light_ev_instead_of_encoded_byte_multiplication() {
        let source =
            RgbaImage::new(3, 1, vec![64, 64, 64, 13, 128, 128, 128, 23, 0, 0, 0, 33]).unwrap();
        let plus_one = basic(&source, &json!({"exposure":50}));
        let minus_one = basic(&source, &json!({"exposure":-50}));
        // IEC sRGB reference values: encoded 64 at +1 EV is about 90;
        // encoded 128 at +1 EV is about 176, rather than 256.
        assert!((plus_one.pixels[0] as i32 - 90).abs() <= 1);
        assert!((plus_one.pixels[4] as i32 - 176).abs() <= 1);
        assert!((minus_one.pixels[0] as i32 - 44).abs() <= 1);
        assert_eq!(&plus_one.pixels[8..11], &[0, 0, 0]);
        assert_eq!(plus_one.pixels[3], 13);
        assert_eq!(plus_one.pixels[7], 23);
    }

    #[test]
    fn tonal_sliders_have_correct_direction_and_light_range_preference() {
        let source = RgbaImage::new(2, 1, vec![40, 40, 40, 101, 210, 210, 210, 102]).unwrap();
        for key in ["brightness", "highlights", "shadows", "whites", "blacks"] {
            let positive = basic(&source, &json!({key:30}));
            let negative = basic(&source, &json!({key:-30}));
            let low = positive.pixels[0] as i32 - 40;
            let high = positive.pixels[4] as i32 - 210;
            assert!(low >= 0 && high >= 0, "{key} positive must lift brightness");
            assert!(
                negative.pixels[0] <= 40 && negative.pixels[4] <= 210,
                "{key} negative must lower brightness"
            );
            match key {
                "highlights" | "whites" => assert!(high > low, "{key} must prefer highlights"),
                "shadows" | "blacks" => assert!(low > high, "{key} must prefer shadows"),
                _ => assert_eq!(low, high),
            }
            assert_eq!(positive.pixels[3], 101);
            assert_eq!(positive.pixels[7], 102);
        }
        let contrast = basic(&source, &json!({"contrast":30}));
        let flat = basic(&source, &json!({"contrast":-30}));
        assert!(contrast.pixels[0] < 40 && contrast.pixels[4] > 210);
        assert!(flat.pixels[0] > 40 && flat.pixels[4] < 210);
    }

    #[test]
    fn levels_and_gamma_have_stable_endpoints_and_monotone_midtone_behavior() {
        let source = RgbaImage::new(
            5,
            1,
            [0, 64, 128, 192, 255]
                .into_iter()
                .flat_map(|v| [v, v, v, 71])
                .collect(),
        )
        .unwrap();
        let black = basic(&source, &json!({"blackPoint":25}));
        let white = basic(&source, &json!({"whitePoint":75}));
        let gamma = basic(&source, &json!({"gamma":2}));
        let dark_gamma = basic(&source, &json!({"gamma":0.5}));
        assert_eq!(black.pixels[4], 0);
        assert!(black.pixels[8] < 128);
        assert_eq!(white.pixels[12], 255);
        assert!(white.pixels[8] > 128);
        assert!(gamma.pixels[8] > 128 && dark_gamma.pixels[8] < 128);
        for output in [&black, &white, &gamma, &dark_gamma] {
            assert_eq!(output.pixels[0], 0);
            assert_eq!(output.pixels[16], 255);
            assert!(output
                .pixels
                .chunks_exact(4)
                .map(|p| p[0])
                .collect::<Vec<_>>()
                .windows(2)
                .all(|w| w[0] <= w[1]));
            assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 71));
        }
    }

    #[test]
    fn temperature_tint_saturation_and_vibrance_match_their_color_axes() {
        let neutral = RgbaImage::new(1, 1, vec![128, 128, 128, 93]).unwrap();
        let warm = basic(&neutral, &json!({"temperature":50}));
        let cool = basic(&neutral, &json!({"temperature":-50}));
        let magenta = basic(&neutral, &json!({"tint":50}));
        let green = basic(&neutral, &json!({"tint":-50}));
        assert!(warm.pixels[0] > warm.pixels[2] && cool.pixels[2] > cool.pixels[0]);
        assert!(magenta.pixels[1] < magenta.pixels[0] && green.pixels[1] > green.pixels[0]);
        let colors = RgbaImage::new(
            3,
            1,
            vec![150, 130, 110, 93, 220, 80, 80, 93, 128, 128, 128, 93],
        )
        .unwrap();
        let vivid = basic(&colors, &json!({"saturation":50}));
        let desaturated = basic(&colors, &json!({"saturation":-100}));
        let natural = basic(&colors, &json!({"vibrance":50}));
        assert!(chroma_span(&vivid, 0) > chroma_span(&colors, 0));
        assert_eq!(chroma_span(&desaturated, 0), 0.0);
        assert_eq!(&natural.pixels[8..12], &colors.pixels[8..12]);
        let low_gain = chroma_span(&natural, 0) / chroma_span(&colors, 0);
        let high_gain = chroma_span(&natural, 1) / chroma_span(&colors, 1);
        assert!(
            low_gain > high_gain,
            "vibrance must preferentially boost weak saturation"
        );
        for output in [
            &warm,
            &cool,
            &magenta,
            &green,
            &vivid,
            &desaturated,
            &natural,
        ] {
            assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 93));
        }
    }

    #[test]
    fn black_white_is_a_progressive_mix_and_preserves_luminance() {
        let source = RgbaImage::new(1, 1, vec![190, 100, 50, 81]).unwrap();
        let quarter = basic(&source, &json!({"blackWhite":25}));
        let half = basic(&source, &json!({"blackWhite":50}));
        let full = basic(&source, &json!({"blackWhite":100}));
        assert!(chroma_span(&quarter, 0) > chroma_span(&half, 0));
        assert!(chroma_span(&half, 0) > 0.0);
        assert_eq!(chroma_span(&full, 0), 0.0);
        assert!((pixel_light(&quarter, 0) - pixel_light(&source, 0)).abs() < 1.0);
        assert_eq!(full.pixels[3], 81);
    }

    #[test]
    fn detail_controls_use_distinct_scales_and_signed_strengths() {
        let width = 256;
        let source = RgbaImage::new(
            width,
            8,
            (0..width * 8)
                .flat_map(|i| {
                    let x = (i % width) as f32;
                    let value =
                        (128.0 + 20.0 * (x * std::f32::consts::TAU / 32.0).sin()).round() as u8;
                    [value, value, value, 177]
                })
                .collect(),
        )
        .unwrap();
        let clarity = basic(&source, &json!({"clarity":100}));
        let sharpness = basic(&source, &json!({"sharpness":100}));
        let texture = basic(&source, &json!({"texture":100}));
        let softened = basic(&source, &json!({"clarity":-100}));
        assert_ne!(clarity, sharpness);
        assert_ne!(texture, sharpness);
        let amplitude = |image: &RgbaImage| {
            let values: Vec<_> = (32..224)
                .map(|x| image.pixels[(3 * width as usize + x) * 4])
                .collect();
            *values.iter().max().unwrap() as i32 - *values.iter().min().unwrap() as i32
        };
        assert!(amplitude(&clarity) > amplitude(&source));
        assert!(amplitude(&softened) < amplitude(&source));
        assert!(clarity.pixels.chunks_exact(4).all(|p| p[3] == 177));
        let fine = RgbaImage::new(
            24,
            24,
            (0..24 * 24)
                .flat_map(|i| {
                    let v = if (i % 24 + i / 24) % 2 == 0 { 145 } else { 111 };
                    [v, v, v, 177]
                })
                .collect(),
        )
        .unwrap();
        let reduced = basic(&fine, &json!({"texture":-100}));
        let enhanced = basic(&fine, &json!({"sharpness":100}));
        assert!(reduced.pixels[0] - reduced.pixels[4] < 34);
        assert!(enhanced.pixels[0] - enhanced.pixels[4] > 34);
    }

    #[test]
    fn bilateral_noise_reduction_suppresses_noise_without_smearing_a_silhouette() {
        let source = RgbaImage::new(
            32,
            16,
            (0..32 * 16)
                .flat_map(|i| {
                    let x = i % 32;
                    let y = i / 32;
                    let noise = if (x + y) % 2 == 0 { 8 } else { 0 };
                    let value = if x < 16 { 56 + noise } else { 192 + noise };
                    [value, value, value, 203]
                })
                .collect(),
        )
        .unwrap();
        let output = basic(&source, &json!({"noise":100}));
        let read = |image: &RgbaImage, x: usize, y: usize| image.pixels[(y * 32 + x) * 4] as i32;
        assert!((read(&output, 8, 8) - read(&output, 9, 8)).abs() < 4);
        assert!(read(&output, 16, 8) - read(&output, 15, 8) > 125);
        assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 203));
    }

    #[test]
    fn dark_channel_dehaze_reduces_error_on_a_known_atmospheric_veil() {
        let width = 96;
        let clean = RgbaImage::new(
            width,
            32,
            (0..width * 32)
                .flat_map(|i| {
                    let x = i % width;
                    let rgb = if x < 24 {
                        [5, 70, 110]
                    } else if x < 48 {
                        [40, 120, 180]
                    } else if x < 72 {
                        [5, 90, 150]
                    } else {
                        [255, 255, 255]
                    };
                    [rgb[0], rgb[1], rgb[2], 167]
                })
                .collect(),
        )
        .unwrap();
        let mut hazy = clean.clone();
        for pixel in hazy.pixels.chunks_exact_mut(4) {
            for value in &mut pixel[..3] {
                *value = byte(linear_to_srgb(srgb_to_linear(*value as f32) * 0.55 + 0.45));
            }
        }
        let restored = basic(&hazy, &json!({"dehaze":100}));
        let error = |image: &RgbaImage| -> u64 {
            let mut sum = 0;
            for x in 0..72 {
                for c in 0..3 {
                    let index = (16 * width as usize + x) * 4 + c;
                    sum += (image.pixels[index] as i32 - clean.pixels[index] as i32).unsigned_abs()
                        as u64;
                }
            }
            sum
        };
        assert!(
            error(&restored) < error(&hazy) / 2,
            "dehaze must restore the synthesized scene, not just alter contrast"
        );
        assert!(restored.pixels.chunks_exact(4).all(|p| p[3] == 167));
        let added = basic(&clean, &json!({"dehaze":-60}));
        assert!(
            pixel_light(&added, 16 * width as usize + 12)
                > pixel_light(&clean, 16 * width as usize + 12)
        );
        let flat = RgbaImage::new(8, 8, [170, 180, 190, 61].repeat(64)).unwrap();
        assert_eq!(basic(&flat, &json!({"dehaze":100})), flat);
    }

    #[test]
    fn spatial_color_algorithms_cannot_import_frozen_or_unselected_colors() {
        let make = |outside: [u8; 3]| {
            RgbaImage::new(
                48,
                24,
                (0..48 * 24)
                    .flat_map(|i| {
                        let x = i % 48;
                        let y = i / 48;
                        let rgb = if x >= 32 || (12..16).contains(&x) {
                            outside
                        } else {
                            let bright = if x >= 24 {
                                240
                            } else if (x + y) % 2 == 0 {
                                130
                            } else {
                                115
                            };
                            [bright, bright, bright]
                        };
                        [rgb[0], rgb[1], rgb[2], 197]
                    })
                    .collect(),
            )
            .unwrap()
        };
        for (is_film, key) in [
            (false, "clarity"),
            (false, "sharpness"),
            (false, "texture"),
            (false, "noise"),
            (false, "dehaze"),
            (true, "glow"),
            (true, "halation"),
        ] {
            let params = json!({key:100,"preset":"neutral","mask":{"shape":"rect","region":{"x":0,"y":0,"width":2.0/3.0,"height":1}},"protectMask":{"shape":"rect","region":{"x":0.25,"y":0,"width":1.0/12.0,"height":1}}});
            let a = make([130, 130, 130]);
            let b = make([255, 0, 0]);
            let ca = crate::mask::composite(
                &a,
                &if is_film {
                    film(&a, &params)
                } else {
                    basic(&a, &params)
                },
                &params,
            );
            let cb = crate::mask::composite(
                &b,
                &if is_film {
                    film(&b, &params)
                } else {
                    basic(&b, &params)
                },
                &params,
            );
            for y in 0..24 {
                for x in 0..48 {
                    let i = (y * 48 + x) * 4;
                    if x >= 32 || (12..16).contains(&x) {
                        assert_eq!(
                            &ca.pixels[i..i + 4],
                            &a.pixels[i..i + 4],
                            "{key} must preserve unselected/frozen output"
                        );
                        assert_eq!(&cb.pixels[i..i + 4], &b.pixels[i..i + 4]);
                    } else {
                        assert_eq!(
                            &ca.pixels[i..i + 4],
                            &cb.pixels[i..i + 4],
                            "{key} has an ineligible color donor"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn curves_interpolate_exactly_and_only_touch_the_selected_channel() {
        let source = RgbaImage::new(2, 1, vec![64, 128, 192, 45, 128, 64, 32, 51]).unwrap();
        for channel in ["red", "r", "green", "g", "blue", "b", "rgb", "master"] {
            let output=curves(&source,&json!({"channel":channel,"points":[{"x":1,"y":1},{"x":0.5,"y":0.75},{"x":0,"y":0}]})).unwrap();
            for (i, (before, after)) in source
                .pixels
                .chunks_exact(4)
                .zip(output.pixels.chunks_exact(4))
                .enumerate()
            {
                for c in 0..3 {
                    let selected = channel == "rgb"
                        || channel == "master"
                        || channel == ["red", "green", "blue"][c]
                        || channel == ["r", "g", "b"][c];
                    if selected {
                        assert!(
                            after[c] > before[c],
                            "{channel} curve must lift point {i} channel {c}"
                        );
                    } else {
                        assert_eq!(after[c], before[c]);
                    }
                }
                assert_eq!(after[3], before[3]);
            }
            if channel == "red" {
                assert!((output.pixels[4] as i32 - 192).abs() <= 1);
            }
        }
        assert!(curves(
            &source,
            &json!({"points":[{"x":0.5,"y":0},{"x":0.5,"y":1}]})
        )
        .is_err());
    }

    #[test]
    fn hsl_global_and_selective_adjustments_respect_hue_saturation_lightness() {
        let source = RgbaImage::new(
            3,
            1,
            vec![180, 100, 100, 82, 100, 100, 180, 83, 128, 128, 128, 84],
        )
        .unwrap();
        assert_eq!(
            hsl(&source, &json!({"hue":0,"saturation":0,"lightness":0})),
            source
        );
        let rotated = hsl(&source, &json!({"hue":100}));
        assert!(rotated.pixels[1] > rotated.pixels[0] && rotated.pixels[2] > rotated.pixels[0]);
        let saturated = hsl(&source, &json!({"saturation":50}));
        let gray = hsl(&source, &json!({"saturation":-100}));
        let bright = hsl(&source, &json!({"lightness":30}));
        assert!(chroma_span(&saturated, 0) > chroma_span(&source, 0));
        assert_eq!(chroma_span(&gray, 0), 0.0);
        assert!(pixel_light(&bright, 0) > pixel_light(&source, 0));
        for key in ["hsl", "channels"] {
            for adjustment in ["shift", "saturation", "lightness"] {
                let selected = hsl(&source, &json!({key:[{"hue":0,"range":45,adjustment:50}]}));
                assert_eq!(
                    &selected.pixels[4..],
                    &source.pixels[4..],
                    "{adjustment} red-range must preserve blue and neutral gray"
                );
                match adjustment {
                    "shift" => assert!(selected.pixels[1] > source.pixels[1]),
                    "saturation" => assert!(chroma_span(&selected, 0) > chroma_span(&source, 0)),
                    _ => assert!(pixel_light(&selected, 0) > pixel_light(&source, 0)),
                }
            }
        }
        let narrow = hsl(
            &source,
            &json!({"hsl":[{"hue":180,"range":30,"shift":100}]}),
        );
        assert_eq!(narrow, source);
        for output in [&rotated, &saturated, &gray, &bright] {
            assert_eq!(output.pixels[3], 82);
            assert_eq!(output.pixels[7], 83);
            assert_eq!(output.pixels[11], 84);
        }
    }

    #[test]
    fn lut_trilinear_interpolation_channel_order_amount_and_domain_are_correct() {
        let source = RgbaImage::new(1, 1, vec![64, 128, 192, 29]).unwrap();
        let data: Vec<_> = (0..2)
            .flat_map(|b| (0..2).flat_map(move |g| (0..2).map(move |r| vec![b, g, r])))
            .collect();
        let swapped = lut(&source, &json!({"size":2,"data":data})).unwrap();
        assert_eq!(swapped.pixels, vec![192, 128, 64, 29]);
        let half = lut(&source, &json!({"size":2,"data":data,"amount":50})).unwrap();
        assert_eq!(half.pixels, vec![128, 128, 128, 29]);
        assert_eq!(
            lut(&source, &json!({"size":2,"data":data,"amount":0})).unwrap(),
            source
        );
        let identity: Vec<_> = (0..2)
            .flat_map(|b| (0..2).flat_map(move |g| (0..2).map(move |r| vec![r, g, b])))
            .collect();
        let domain = lut(
            &source,
            &json!({"size":2,"data":identity,"domainMin":[0.25,0,0.5],"domainMax":[0.75,1,1]}),
        )
        .unwrap();
        assert!(domain.pixels[0] <= 1);
        assert_eq!(domain.pixels[1], 128);
        assert!((domain.pixels[2] as i32 - 129).abs() <= 1);
        assert!(lut(
            &source,
            &json!({"size":2,"data":identity,"domainMin":[1,0,0],"domainMax":[1,1,1]})
        )
        .is_err());
    }

    #[test]
    fn film_master_intensity_scales_every_effect_and_zero_is_identity() {
        let source = RgbaImage::new(
            32,
            16,
            (0..32 * 16)
                .flat_map(|i| {
                    let v = if i % 32 < 16 { 120 } else { 240 };
                    [v, v, v, 73]
                })
                .collect(),
        )
        .unwrap();
        let mut params = json!({"preset":"cool","grain":50,"vignette":50,"glow":50,"halation":50,"fade":50,"seed":19,"intensity":0});
        assert_eq!(film(&source, &params), source);
        params["intensity"] = json!(100);
        let full = film(&source, &params);
        params["intensity"] = json!(50);
        let half = film(&source, &params);
        for i in 0..source.pixels.len() {
            if i % 4 == 3 {
                assert_eq!(half.pixels[i], 73);
            } else {
                let expected = (source.pixels[i] as f32 + full.pixels[i] as f32) * 0.5;
                assert!(
                    (half.pixels[i] as f32 - expected).abs() <= 1.0,
                    "film midpoint differs at {i}"
                );
            }
        }
    }

    #[test]
    fn film_grain_is_repeatable_zero_mean_and_seeded_without_color_drift() {
        let source = RgbaImage::new(64, 64, [128, 128, 128, 141].repeat(4096)).unwrap();
        let params = json!({"preset":"neutral","grain":100,"seed":7});
        let first = film(&source, &params);
        assert_eq!(first, film(&source, &params));
        assert_ne!(
            first,
            film(&source, &json!({"preset":"neutral","grain":100,"seed":8}))
        );
        let mean = first
            .pixels
            .chunks_exact(4)
            .map(|p| p[0] as f32 - 128.0)
            .sum::<f32>()
            / 4096.0;
        assert!(mean.abs() < 1.0);
        let variance = first
            .pixels
            .chunks_exact(4)
            .map(|p| (p[0] as f32 - 128.0).powi(2))
            .sum::<f32>()
            / 4096.0;
        assert!(variance > 20.0);
        assert!(first
            .pixels
            .chunks_exact(4)
            .all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 141));
    }

    #[test]
    fn film_vignette_fade_glow_and_halation_have_distinct_spatial_effects() {
        let flat = RgbaImage::new(32, 32, [128, 128, 128, 101].repeat(1024)).unwrap();
        let vignette = film(&flat, &json!({"preset":"neutral","vignette":100}));
        assert!(pixel_light(&vignette, 0) < pixel_light(&vignette, 16 * 32 + 16) - 50.0);
        let tones = RgbaImage::new(2, 1, vec![0, 0, 0, 101, 255, 255, 255, 101]).unwrap();
        let fade = film(&tones, &json!({"preset":"neutral","fade":100}));
        assert!(fade.pixels[0] > 0 && fade.pixels[4] < 255);
        let source = RgbaImage::new(
            64,
            32,
            (0..64 * 32)
                .flat_map(|i| {
                    let x = i % 64;
                    let v = if (16..48).contains(&x) { 255 } else { 110 };
                    [v, v, v, 101]
                })
                .collect(),
        )
        .unwrap();
        let glow = film(&source, &json!({"preset":"neutral","glow":100}));
        let halo = film(&source, &json!({"preset":"neutral","halation":100}));
        let edge = (16 * 64 + 15) * 4;
        let center = (16 * 64 + 32) * 4;
        assert!(glow.pixels[edge] > source.pixels[edge]);
        assert_eq!(glow.pixels[edge], glow.pixels[edge + 1]);
        assert!(
            halo.pixels[edge] > halo.pixels[edge + 1]
                && halo.pixels[edge + 1] >= halo.pixels[edge + 2]
        );
        assert_eq!(
            &halo.pixels[center..center + 4],
            &source.pixels[center..center + 4],
            "halation must not recolor highlight cores"
        );
        assert_eq!(
            &halo.pixels[..4],
            &source.pixels[..4],
            "far dark regions must have no halo"
        );
    }

    #[test]
    fn each_film_preset_has_its_declared_color_direction() {
        let source = RgbaImage::new(2, 1, vec![140, 120, 100, 93, 40, 40, 40, 93]).unwrap();
        assert_eq!(film(&source, &json!({"preset":"neutral"})), source);
        let warm = film(&source, &json!({"preset":"warm"}));
        let cool = film(&source, &json!({"preset":"cool"}));
        assert!(warm.pixels[0] > 140 && warm.pixels[2] < 100);
        assert!(cool.pixels[0] < 140 && cool.pixels[2] > 100);
        let saturated = RgbaImage::new(1, 1, vec![220, 80, 40, 93]).unwrap();
        let classic = film(&saturated, &json!({"preset":"classic"}));
        assert!(chroma_span(&classic, 0) < chroma_span(&saturated, 0));
        let faded = film(&source, &json!({"preset":"fade"}));
        assert!(faded.pixels[4] > 40);
        let vivid = film(&source, &json!({"preset":"vivid"}));
        assert!(chroma_span(&vivid, 0) > chroma_span(&source, 0));
        let mono = film(&source, &json!({"preset":"mono"}));
        assert_eq!(chroma_span(&mono, 0), 0.0);
        let cinema = film(&source, &json!({"preset":"cinema"}));
        assert!(cinema.pixels[6] > cinema.pixels[4]);
        for output in [&warm, &cool, &classic, &faded, &vivid, &mono, &cinema] {
            assert_eq!(output.pixels[3], 93);
            assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 93));
        }
    }

    #[test]
    fn film_response_profiles_roll_highlights_and_keep_skin_safe_chroma() {
        let source = RgbaImage::new(
            4,
            1,
            vec![
                18, 24, 34, 255, // shadow
                160, 112, 92, 255, // skin-like midtone
                232, 184, 142, 255, // warm highlight
                255, 255, 255, 255,
            ],
        )
        .unwrap();
        for preset in [
            "provia",
            "astia",
            "velvia",
            "classicNegative",
            "eterna",
            "acros",
        ] {
            let output = film(&source, &json!({"preset":preset,"intensity":100}));
            assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 255));
            // The response must be monotone in luminance: a profile may
            // change hue, but it must not invert a highlight and a shadow.
            let dark = pixel_light(&output, 0);
            let mid = pixel_light(&output, 1);
            let bright = pixel_light(&output, 2);
            assert!(
                dark < mid && mid < bright,
                "{preset}: {dark} {mid} {bright}"
            );
        }
        let acros = film(&source, &json!({"preset":"acros"}));
        for pixel in acros.pixels.chunks_exact(4) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
        }
        let provia = film(&source, &json!({"preset":"provia"}));
        let eterna = film(&source, &json!({"preset":"eterna"}));
        assert!(chroma_span(&provia, 1) > chroma_span(&eterna, 1));
        assert!(pixel_light(&eterna, 2) < pixel_light(&source, 2));
    }

    #[test]
    fn skin_redness_is_chromatic_and_texture_preserves_detail_only_when_smoothing() {
        let source = RgbaImage::new(1, 1, vec![175, 135, 100, 19]).unwrap();
        let red = skin(&source, &json!({"redness":100}));
        let neutralized = skin(&source, &json!({"redness":-100}));
        assert!(red.pixels[0] - red.pixels[1] > source.pixels[0] - source.pixels[1]);
        assert!(
            neutralized.pixels[0] - neutralized.pixels[1] < source.pixels[0] - source.pixels[1]
        );
        assert!((pixel_light(&red, 0) - pixel_light(&source, 0)).abs() < 1.0);
        assert!((pixel_light(&neutralized, 0) - pixel_light(&source, 0)).abs() < 1.0);
        let detail = RgbaImage::new(
            24,
            24,
            (0..24 * 24)
                .flat_map(|i| {
                    let v = if (i % 24 + i / 24) % 2 == 0 { 150 } else { 130 };
                    [v, v, v, 79]
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(skin(&detail, &json!({"smooth":0,"texture":0})), detail);
        assert_eq!(skin(&detail, &json!({"smooth":0,"texture":100})), detail);
        let smoothed = skin(&detail, &json!({"smooth":100,"texture":0}));
        let retained = skin(&detail, &json!({"smooth":100,"texture":100}));
        let amplitude = |image: &RgbaImage| {
            (image.pixels[(12 * 24 + 12) * 4] as i32 - image.pixels[(12 * 24 + 13) * 4] as i32)
                .abs()
        };
        assert!(amplitude(&smoothed) < amplitude(&retained));
        assert!(amplitude(&retained) <= amplitude(&detail));
        assert!(retained.pixels.chunks_exact(4).all(|p| p[3] == 79));
    }

    #[test]
    fn redness_neutralization_prefers_local_red_patch_over_healthy_skin_and_keeps_texture() {
        let width = 160u32;
        let source = RgbaImage::new(
            width,
            96,
            (0..width * 96)
                .flat_map(|i| {
                    let x = i % width;
                    let y = i / width;
                    let bump =
                        (-((x as f32 - 104.0).powi(2) + (y as f32 - 48.0).powi(2)) / 80.0).exp();
                    let pore = if (x + y).is_multiple_of(2) { 2.0 } else { -2.0 };
                    let base = 138.0 + x as f32 * 0.12 + y as f32 * 0.06 + pore;
                    [
                        byte(base + 32.0 + bump * 20.0),
                        byte(base - bump * 6.0),
                        byte(base - 19.0 + bump * 8.0),
                        173,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let output = skin(&source, &json!({"redness":-100,"smooth":0}));
        let red_at = |image: &RgbaImage, x: usize, y: usize| {
            let i = (y * width as usize + x) * 4;
            image.pixels[i] as f32 - image.pixels[i + 1] as f32
        };
        let center = 48 * width as usize + 104;
        let healthy = 48 * width as usize + 40;
        assert!(red_at(&source, 104, 48) - red_at(&output, 104, 48) > 16.0);
        assert!((red_at(&source, 40, 48) - red_at(&output, 40, 48)).abs() <= 3.0);
        let center_pixel = &output.pixels[center * 4..center * 4 + 3];
        assert!(
            center_pixel[1] as i32 - center_pixel[2] as i32 >= 14,
            "de-red must recover local warm skin chroma rather than leave a purple spot"
        );
        for i in [center, healthy] {
            assert!((pixel_light(&output, i) - pixel_light(&source, i)).abs() < 1.0);
        }
        let original_grain = pixel_light(&source, healthy) - pixel_light(&source, healthy + 1);
        let edited_grain = pixel_light(&output, healthy) - pixel_light(&output, healthy + 1);
        assert!((edited_grain - original_grain).abs() < 1.0);
        assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 173));
    }

    #[test]
    fn redness_reference_cannot_import_frozen_or_semantically_protected_colour() {
        let make = |blocked: [u8; 3]| {
            RgbaImage::new(
                64,
                40,
                (0..64 * 40)
                    .flat_map(|i| {
                        let (x, y) = (i % 64, i / 64);
                        let rgb = if (26..38).contains(&x) && (12..28).contains(&y) {
                            blocked
                        } else {
                            [174 + (x % 3) as u8, 136, 110]
                        };
                        [rgb[0], rgb[1], rgb[2], 173]
                    })
                    .collect(),
            )
            .unwrap()
        };
        let first = make([255, 0, 0]);
        let second = make([0, 255, 255]);
        let protection = json!({"shape":"rect","region":{
            "x":26.0/64.0,"y":12.0/40.0,"width":12.0/64.0,"height":16.0/40.0}});
        for semantic in ["skinProtection", "protectMask"] {
            let mut params = json!({"redness":-100,"smooth":0,"semanticMasks":{
                "skin":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}}});
            if semantic == "protectMask" {
                params["protectMask"] = protection.clone();
            } else {
                params["semanticMasks"][semantic] = protection.clone();
            }
            let left = crate::mask::composite(&first, &skin(&first, &params), &params);
            let right = crate::mask::composite(&second, &skin(&second, &params), &params);
            for y in 0..40 {
                for x in 0..64 {
                    let i = (y * 64 + x) * 4;
                    if (26..38).contains(&x) && (12..28).contains(&y) {
                        assert_eq!(&left.pixels[i..i + 4], &first.pixels[i..i + 4]);
                        assert_eq!(&right.pixels[i..i + 4], &second.pixels[i..i + 4]);
                    } else {
                        assert_eq!(&left.pixels[i..i + 4], &right.pixels[i..i + 4]);
                    }
                }
            }
        }
    }

    #[test]
    fn guided_skin_reduces_small_dark_spots_without_flattening_shading() {
        let width = 128;
        let height = 80;
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let base = 130 + x / 4;
                let spot = x == 60 && y == 40;
                let value = if spot { base - 70 } else { base };
                pixels.extend_from_slice(&[
                    value as u8,
                    (value - 15) as u8,
                    (value - 30) as u8,
                    173,
                ]);
            }
        }
        let source = RgbaImage::new(width, height, pixels).unwrap();
        let output = skin(&source, &json!({"smooth":100,"texture":0}));
        let spot = (40 * width as usize + 60) * 4;
        assert!(
            output.pixels[spot] > source.pixels[spot] + 25,
            "a small dark spot must be softened, not mistaken for a silhouette edge"
        );
        for x in [20usize, 100] {
            let i = (20 * width as usize + x) * 4;
            assert!(
                (output.pixels[i] as i32 - source.pixels[i] as i32).abs() <= 1,
                "broad affine shading must survive skin smoothing"
            );
        }
        assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 173));
    }

    #[test]
    fn guided_skin_suppresses_fine_noise_and_preserves_strong_edges() {
        let width = 128u32;
        let height = 64u32;
        let source = RgbaImage::new(
            width,
            height,
            (0..height)
                .flat_map(|y| {
                    (0..width).flat_map(move |x| {
                        let base = if x < 64 { 105i32 } else { 205 };
                        let noise = if (x + y) % 2 == 0 { 8 } else { -8 };
                        [
                            (base + noise) as u8,
                            (base + noise - 10) as u8,
                            (base + noise - 20) as u8,
                            255,
                        ]
                    })
                })
                .collect(),
        )
        .unwrap();
        let output = skin(&source, &json!({"smooth":100,"texture":0}));
        let original_contrast = (source.pixels[(32 * 128 + 80) * 4] as i32
            - source.pixels[(32 * 128 + 20) * 4] as i32)
            .abs();
        let edited_contrast = (output.pixels[(32 * 128 + 80) * 4] as i32
            - output.pixels[(32 * 128 + 20) * 4] as i32)
            .abs();
        assert!(
            edited_contrast >= original_contrast - 2,
            "separate skin lighting regions retain contrast"
        );
        let before_noise = (source.pixels[(32 * 128 + 20) * 4] as i32
            - source.pixels[(32 * 128 + 21) * 4] as i32)
            .abs();
        let after_noise = (output.pixels[(32 * 128 + 20) * 4] as i32
            - output.pixels[(32 * 128 + 21) * 4] as i32)
            .abs();
        assert!(
            after_noise <= before_noise / 2,
            "fine color variation must actually decrease"
        );
        let boundary_contrast =
            output.pixels[(32 * 128 + 64) * 4] as i32 - output.pixels[(32 * 128 + 63) * 4] as i32;
        assert!(
            boundary_contrast >= 85,
            "the lighting edge must stay sharp without Gaussian bleeding"
        );
    }

    fn pore_and_blemish_fixture() -> RgbaImage {
        let width = 192u32;
        let height = 128u32;
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let base = 156.0 + x as f32 * 0.08;
                    // Coherent, low-contrast pore detail spans several pixels;
                    // it is different from alternating one-pixel sensor noise.
                    let mut pore = 0.0f32;
                    for cy in [32.0f32, 56.0, 80.0, 104.0] {
                        for cx in [32.0f32, 56.0, 80.0] {
                            let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                            pore += 10.0 * (-d2 / 2.0).exp();
                        }
                    }
                    let d2 = (x as f32 - 136.0).powi(2) + (y as f32 - 64.0).powi(2);
                    let blemish = 44.0 * (-d2 / 14.0).exp();
                    [
                        byte(base - pore - blemish * 0.55),
                        byte(base - 17.0 - pore - blemish),
                        byte(base - 29.0 - pore - blemish),
                        173,
                    ]
                })
            })
            .collect();
        RgbaImage::new(width, height, pixels).unwrap()
    }

    #[test]
    fn strong_skin_smoothing_keeps_coherent_pores_but_reduces_compact_blemishes() {
        let source = pore_and_blemish_fixture();
        let output = skin(&source, &json!({"smooth":100,"texture":0}));
        let light_at = |image: &RgbaImage, x: usize, y: usize| {
            pixel_light(image, y * image.width as usize + x)
        };
        let pore_before = light_at(&source, 32, 27) - light_at(&source, 32, 32);
        let pore_after = light_at(&output, 32, 27) - light_at(&output, 32, 32);
        assert!(
            pore_after >= pore_before * 0.55,
            "normal coherent pores must remain under strong smoothing: {pore_before} -> {pore_after}"
        );
        let blemish_before = light_at(&source, 136, 51) - light_at(&source, 136, 64);
        let blemish_after = light_at(&output, 136, 51) - light_at(&output, 136, 64);
        assert!(
            blemish_after <= blemish_before * 0.60,
            "larger high-contrast compact blemishes must improve: {blemish_before} -> {blemish_after}"
        );
        assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 173));
    }

    #[test]
    fn texture_retention_does_not_disable_compact_blemish_correction() {
        let source = pore_and_blemish_fixture();
        let output = skin(&source, &json!({"smooth":100,"texture":100}));
        let center = 64 * source.width as usize + 136;
        assert!(
            pixel_light(&output, center) >= pixel_light(&source, center) + 15.0,
            "texture slider must preserve healthy grain without preserving every blemish"
        );
        let pore = 32 * source.width as usize + 32;
        assert!(
            (pixel_light(&output, pore) - pixel_light(&source, pore)).abs() <= 3.0,
            "a texture-preserved pore must retain its local tone"
        );
    }

    #[test]
    fn texture_preserved_skin_reduces_coherent_red_patch_without_erasing_low_contrast_pores() {
        let source = RgbaImage::new(
            192,
            128,
            (0..128)
                .flat_map(|y| {
                    (0..192).flat_map(move |x| {
                        let d2 = (x as f32 - 96.0).powi(2) + (y as f32 - 64.0).powi(2);
                        let spot = (-d2 / 14.0).exp();
                        [
                            byte(180.0 + 6.0 * spot),
                            byte(150.0 - 9.0 * spot),
                            byte(130.0 - 5.0 * spot),
                            255,
                        ]
                    })
                })
                .collect(),
        )
        .unwrap();
        let output = skin(&source, &json!({"smooth":100,"texture":100}));
        let center = (64 * 192 + 96) * 4;
        let red_excess =
            |image: &RgbaImage| image.pixels[center] as i32 - image.pixels[center + 1] as i32 - 30;
        assert!(
            red_excess(&output) <= (red_excess(&source) as f32 * 0.65) as i32,
            "a spatially coherent red patch must be softened even with maximum texture: {} -> {}",
            red_excess(&source),
            red_excess(&output)
        );
    }

    #[test]
    fn wide_red_blotch_is_corrected_without_flattening_healthy_lighting() {
        let source = RgbaImage::new(
            192,
            128,
            (0..128)
                .flat_map(|y| {
                    (0..192).flat_map(move |x| {
                        let base = 170.0 + x as f32 * 0.15;
                        let d2 = (x as f32 - 96.0).powi(2) + (y as f32 - 64.0).powi(2);
                        let stain = (-d2 / 72.0).exp();
                        [
                            byte(base + 8.0 * stain),
                            byte(base - 30.0 - 14.0 * stain),
                            byte(base - 50.0 - 4.0 * stain),
                            173,
                        ]
                    })
                })
                .collect(),
        )
        .unwrap();
        let output = skin(&source, &json!({"smooth":100,"texture":100}));
        let center = (64 * 192 + 96) * 4;
        let red_excess =
            |image: &RgbaImage| image.pixels[center] as i32 - image.pixels[center + 1] as i32 - 30;
        assert!(
            red_excess(&output) <= 14,
            "a broad chromatic stain must improve without disabling texture: {} -> {}",
            red_excess(&source),
            red_excess(&output)
        );
        for x in [20usize, 160] {
            let i = 20 * 192 + x;
            assert!((pixel_light(&output, i) - pixel_light(&source, i)).abs() <= 1.0);
            for c in 0..3 {
                assert!(
                    (output.pixels[i * 4 + c] as i32 - source.pixels[i * 4 + c] as i32).abs() <= 1
                );
            }
        }
    }

    #[test]
    fn brow_tone_does_not_paint_between_hair_strands() {
        let (w, h) = (96_u32, 48_u32);
        let mut pixels = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let strand = (18..26).contains(&y) && (12..84).contains(&x) && x % 5 < 2;
                pixels.extend_from_slice(if strand {
                    &[54, 42, 31, 255]
                } else {
                    &[190, 155, 131, 255]
                });
            }
        }
        let source = RgbaImage::new(w, h, pixels).unwrap();
        let masks = json!({"skin":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},"brows":{"shape":"rect","region":{"x":0.125,"y":0.375,"width":0.75,"height":1.0/6.0}}});
        for amount in [-100, -60, 60, 100] {
            let output = local_color(
                &source,
                &json!({"semanticMasks":masks,"adjustments":{"browColor":amount}}),
                false,
            );
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    let strand = (18..26).contains(&y) && (12..84).contains(&x) && x % 5 < 2;
                    if !strand {
                        assert_eq!(
                            &output.pixels[i * 4..i * 4 + 4],
                            &source.pixels[i * 4..i * 4 + 4]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lipstick_keeps_teeth_polygon_bit_exact() {
        let source = RgbaImage::new(20, 8, [170, 90, 100, 255].repeat(160)).unwrap();
        let params = json!({"semanticMasks":{"lips":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},"teeth":{"shape":"rect","region":{"x":0.4,"y":0,"width":0.2,"height":1}}},"adjustments":{"lipColor":100}});
        let output = local_color(&source, &params, false);
        for y in 0..8 {
            for x in 8..12 {
                let i = (y * 20 + x) as usize;
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }

    #[test]
    fn every_makeup_color_parameter_has_its_own_direction_and_semantic_boundary() {
        let source = RgbaImage::new(
            4,
            1,
            vec![
                80, 60, 40, 29, 140, 60, 65, 29, 210, 195, 175, 29, 90, 80, 70, 29,
            ],
        )
        .unwrap();
        let masks = json!({
            "brows":{"shape":"rect","region":{"x":0,"y":0,"width":0.25,"height":1}},
            "lips":{"shape":"rect","region":{"x":0.25,"y":0,"width":0.25,"height":1}},
            "teeth":{"shape":"rect","region":{"x":0.5,"y":0,"width":0.25,"height":1}},
            "underEyes":{"shape":"rect","region":{"x":0.75,"y":0,"width":0.25,"height":1}}
        });
        for (key, target) in [
            ("browColor", 0),
            ("lipColor", 1),
            ("lipBrightness", 1),
            ("lipSaturation", 1),
            ("teethWhiten", 2),
            ("darkCircle", 3),
        ] {
            let params = json!({"adjustments":{key:70},"semanticMasks":masks});
            let output = local_color(&source, &params, false);
            for i in 0..4 {
                if i != target {
                    assert_eq!(
                        &output.pixels[i * 4..i * 4 + 4],
                        &source.pixels[i * 4..i * 4 + 4],
                        "{key} escaped its semantic target"
                    );
                }
                assert_eq!(output.pixels[i * 4 + 3], 29);
            }
            match key {
                "browColor" => assert!(pixel_light(&output, target) < pixel_light(&source, target)),
                "lipColor" => assert!(
                    output.pixels[target * 4] - output.pixels[target * 4 + 1]
                        > source.pixels[target * 4] - source.pixels[target * 4 + 1]
                ),
                "lipSaturation" => {
                    assert!(chroma_span(&output, target) > chroma_span(&source, target))
                }
                _ => assert!(pixel_light(&output, target) > pixel_light(&source, target)),
            }
            let frozen = json!({"adjustments":{key:70},"semanticMasks":masks,"protectMask":{"shape":"rect","region":{"x":target as f32/4.0,"y":0,"width":0.25,"height":1}}});
            assert_eq!(
                crate::mask::composite(&source, &local_color(&source, &frozen, false), &frozen),
                source
            );
            assert_eq!(
                local_color(
                    &source,
                    &json!({"adjustments":{key:0},"semanticMasks":masks}),
                    false
                ),
                source
            );
        }
    }

    fn under_eye_fixture(shadow: bool) -> (RgbaImage, Value) {
        let mut pixels = Vec::new();
        for y in 0..120 {
            for x in 0..160 {
                let light = 130.0 + x as f32 * 0.22 + y as f32 * 0.14;
                let distance =
                    ((x as f32 - 60.0) / 18.0).powi(2) + ((y as f32 - 55.0) / 8.0).powi(2);
                let shade = if shadow {
                    (1.0 - distance).max(0.0) * 30.0
                } else {
                    0.0
                };
                let pore = if (x / 2 + y / 2) % 2 == 0 { 2.0 } else { -2.0 };
                pixels.extend_from_slice(&[
                    byte(light + 25.0 + pore - shade),
                    byte(light + pore - shade),
                    byte(light - 16.0 + pore - shade * 0.75),
                    255,
                ]);
            }
        }
        let params = json!({"adjustments":{"darkCircle":100},"semanticMasks":{
            "skin":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
            "underEyes":{"shape":"rect","region":{"x":0.25,"y":0.375,"width":0.25,"height":0.17}}
        }});
        (RgbaImage::new(160, 120, pixels).unwrap(), params)
    }

    #[test]
    fn under_eye_plane_rejects_local_highlight_and_chroma_outliers() {
        let mut donors = Vec::new();
        for step in 0..36 {
            let angle = step as f32 * std::f32::consts::TAU / 36.0;
            let (x, y) = (libm::cosf(angle) as f64, libm::sinf(angle) as f64);
            let light = 145.0 + x * 18.0 + y * 8.0;
            let mut color = [light as f32 + 24.0, light as f32, light as f32 - 16.0];
            // A compact ring section passes over bright cheek color / blush.
            if (4..9).contains(&step) {
                color = [245.0, 200.0, 190.0];
            }
            donors.push(UnderEyeDonor {
                position: [1.0, x, y],
                color,
                weight: 1.0,
            });
        }
        let reference = under_eye_reference(&donors).unwrap();
        for (actual, expected) in reference.into_iter().zip([169.0, 145.0, 129.0]) {
            assert!(
                (actual - expected).abs() < 0.05,
                "outlier biased reference: {reference:?}"
            );
        }
        // All visible donors on a single line cannot support a plane.
        for donor in &mut donors {
            donor.position[2] = 0.0;
        }
        assert!(under_eye_reference(&donors).is_none());
    }

    #[test]
    fn under_eye_preserves_narrow_dark_contours_inside_the_shadow() {
        let (mut source, params) = under_eye_fixture(true);
        // An unparsed eyeglass / eyelash contour inside the requested region.
        for y in 50..61 {
            let i = (y * 160 + 60) * 4;
            source.pixels[i..i + 3].copy_from_slice(&[32, 28, 30]);
        }
        let output = local_color(&source, &params, false);
        for y in 50..61 {
            let i = (y * 160 + 60) * 4;
            assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
        }
        let nearby = 55 * 160 + 67;
        assert!(pixel_light(&output, nearby) > pixel_light(&source, nearby) + 5.0);
    }

    #[test]
    fn adaptive_under_eye_reduces_shadow_without_flattening_texture_or_affine_lighting() {
        let (healthy, params) = under_eye_fixture(false);
        let (source, _) = under_eye_fixture(true);
        let output = local_color(&source, &params, false);
        let center = 55 * 160 + 60;
        let before = pixel_light(&healthy, center) - pixel_light(&source, center);
        let after = pixel_light(&healthy, center) - pixel_light(&output, center);
        assert!(after < before * 0.55, "eye shadow: {before} -> {after}");
        let healthy_output = local_color(&healthy, &params, false);
        assert_eq!(
            healthy_output, healthy,
            "healthy lit skin must not be brightened"
        );
        // The pass transfers low frequency illumination; a two-pixel pore
        // remains instead of being replaced by a flat patch.
        let contrast =
            |image: &RgbaImage| pixel_light(image, center) - pixel_light(image, center + 2);
        assert!((contrast(&output) - contrast(&source)).abs() < 1.5);
        let stronger = local_color(&source, &params, false);
        let mut weaker_params = params.clone();
        weaker_params["adjustments"]["darkCircle"] = json!(40);
        let weaker = local_color(&source, &weaker_params, false);
        assert!(pixel_light(&weaker, center) > pixel_light(&source, center));
        assert!(pixel_light(&stronger, center) > pixel_light(&weaker, center));
        for (i, (a, b)) in source
            .pixels
            .chunks_exact(4)
            .zip(output.pixels.chunks_exact(4))
            .enumerate()
        {
            let (x, y) = (i % 160, i / 160);
            if !(40..80).contains(&x) || !(45..65).contains(&y) {
                assert_eq!(a, b);
            }
            assert_eq!(a[3], b[3]);
        }
    }

    #[test]
    fn adaptive_under_eye_excludes_protected_and_transparent_reference_colors() {
        let (mut a, mut params) = under_eye_fixture(true);
        params["semanticMasks"]["skinProtection"] =
            json!({"shape":"rect","region":{"x":0.35,"y":0.56,"width":0.3,"height":0.15}});
        let mut b = a.clone();
        for y in 0..120 {
            for x in 0..160 {
                let i = (y * 160 + x) * 4;
                if (56..104).contains(&x) && (67..85).contains(&y) {
                    a.pixels[i..i + 3].copy_from_slice(&[10, 240, 30]);
                    b.pixels[i..i + 3].copy_from_slice(&[240, 10, 230]);
                }
                if (82..98).contains(&x) && (40..65).contains(&y) {
                    a.pixels[i..i + 4].copy_from_slice(&[0, 255, 0, 0]);
                    b.pixels[i..i + 4].copy_from_slice(&[255, 0, 255, 0]);
                }
            }
        }
        let first = local_color(&a, &params, false);
        let second = local_color(&b, &params, false);
        for y in 45..65 {
            for x in 40..80 {
                let i = (y * 160 + x) * 4;
                assert_eq!(
                    &first.pixels[i..i + 4],
                    &second.pixels[i..i + 4],
                    "excluded donor leaked at {x},{y}"
                );
            }
        }
        assert_ne!(
            first.pixels[(55 * 160 + 60) * 4],
            a.pixels[(55 * 160 + 60) * 4]
        );
    }

    #[test]
    fn clothing_hue_saturation_brightness_and_texture_have_correct_local_semantics() {
        let source = RgbaImage::new(2, 1, vec![150, 110, 90, 49, 60, 70, 80, 53]).unwrap();
        for key in ["hue", "saturation", "brightness"] {
            let params =
                json!({"adjustments":{key:50},"region":{"x":0,"y":0,"width":0.5,"height":1}});
            let output =
                crate::mask::composite(&source, &local_color(&source, &params, true), &params);
            assert_eq!(&output.pixels[4..], &source.pixels[4..]);
            assert_eq!(output.pixels[3], 49);
            match key {
                "hue" => {
                    let before = rgb_to_hsl([150.0, 110.0, 90.0]).0;
                    let after = rgb_to_hsl([
                        output.pixels[0] as f32,
                        output.pixels[1] as f32,
                        output.pixels[2] as f32,
                    ])
                    .0;
                    assert!((after - before - 90.0).abs() < 1.0);
                }
                "saturation" => assert!(chroma_span(&output, 0) > chroma_span(&source, 0)),
                _ => assert!(pixel_light(&output, 0) > pixel_light(&source, 0) + 35.0),
            }
        }
        assert_eq!(
            clothing_retouch(&source, &json!({"smooth":0,"texture":0})),
            source
        );
        assert_eq!(
            clothing_retouch(&source, &json!({"smooth":0,"texture":100})),
            source
        );
    }

    #[test]
    fn clothing_smoothing_strength_is_continuous_and_monotone_from_zero() {
        let source = RgbaImage::new(
            24,
            24,
            (0..24 * 24)
                .flat_map(|i| {
                    let v = if (i % 24 + i / 24) % 2 == 0 { 160 } else { 100 };
                    [v, v, v, 91]
                })
                .collect(),
        )
        .unwrap();
        let error = |amount: f32| {
            let output = clothing_retouch(&source, &json!({"smooth":amount,"texture":0}));
            output
                .pixels
                .iter()
                .zip(&source.pixels)
                .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
                .sum::<u64>()
        };
        let zero = error(0.0);
        let low = error(1.0);
        let middle = error(50.0);
        let high = error(100.0);
        assert_eq!(zero, 0);
        assert!(low < middle && middle < high);
        assert!(
            low < high / 20,
            "a 1% slider change must not cause an immediate strong smoothing jump"
        );
    }

    #[test]
    fn every_public_basic_parameter_obeys_mask_freeze_opacity_and_alpha() {
        let source = RgbaImage::new(
            24,
            24,
            (0..24 * 24)
                .flat_map(|i| {
                    let x = i % 24;
                    let y = i / 24;
                    [
                        70 + (x * 4) as u8,
                        60 + (y * 4) as u8,
                        100 + ((x + y) % 2 * 12) as u8,
                        97,
                    ]
                })
                .collect(),
        )
        .unwrap();
        for (key, value) in [
            ("exposure", 30.0),
            ("brightness", 30.0),
            ("contrast", 30.0),
            ("highlights", 30.0),
            ("shadows", 30.0),
            ("whites", 30.0),
            ("blacks", 30.0),
            ("temperature", 30.0),
            ("tint", 30.0),
            ("saturation", 30.0),
            ("vibrance", 30.0),
            ("clarity", 30.0),
            ("sharpness", 30.0),
            ("texture", 30.0),
            ("dehaze", 30.0),
            ("noise", 30.0),
            ("blackWhite", 30.0),
            ("blackPoint", 10.0),
            ("whitePoint", 90.0),
            ("gamma", 1.5),
        ] {
            let params = json!({key:value,"mask":{"shape":"rect","region":{"x":0,"y":0,"width":0.5,"height":1}},"protectMask":{"shape":"rect","region":{"x":0,"y":0,"width":0.25,"height":1}}});
            let output = crate::mask::composite(&source, &basic(&source, &params), &params);
            for y in 0..24 {
                for x in 0..24 {
                    let i = (y * 24 + x) * 4;
                    if !(6..12).contains(&x) {
                        assert_eq!(
                            &output.pixels[i..i + 4],
                            &source.pixels[i..i + 4],
                            "{key} crossed mask/protection"
                        );
                    }
                    assert_eq!(output.pixels[i + 3], 97);
                }
            }
            let mut zero = params;
            zero["opacity"] = json!(0);
            assert_eq!(
                crate::mask::composite(&source, &basic(&source, &zero), &zero),
                source,
                "{key} ignored zero opacity"
            );
        }
    }
    #[test]
    fn curves_identity_and_lut_identity_preserve_alpha() {
        let image = RgbaImage::new(1, 1, vec![31, 100, 200, 47]).unwrap();
        assert_eq!(
            curves(&image, &json!({"points":[{"x":0,"y":0},{"x":1,"y":1}]})).unwrap(),
            image
        );
        let mut data = Vec::new();
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    data.push(vec![r, g, b]);
                }
            }
        }
        assert_eq!(lut(&image, &json!({"size":2,"data":data})).unwrap(), image);
    }
    #[test]
    fn skin_and_film_do_not_change_transparency() {
        let image = RgbaImage::new(1, 1, vec![120, 100, 90, 37]).unwrap();
        assert_eq!(
            skin(&image, &json!({"whiten":70,"smooth":50})).pixels[3],
            37
        );
        let params = json!({"preset":"classic","grain":80,"seed":7});
        assert_eq!(film(&image, &params), film(&image, &params));
        assert_eq!(film(&image, &params).pixels[3], 37);
    }
    #[test]
    fn clothing_smoothing_reduces_fold_contrast_and_keeps_fabric_detail() {
        let mut pixels = Vec::with_capacity(32 * 32 * 4);
        for y in 0..32 {
            for x in 0..32 {
                // A broad gradient represents a lighting fold; the alternating
                // channel adds a small weave detail that texture should retain.
                let fold = (x as f32 / 31.0 * 70.0) as u8;
                let weave = if (x + y) % 2 == 0 { 18 } else { 0 };
                pixels.extend_from_slice(&[80 + fold, 90 + weave, 100, 123]);
            }
        }
        let source = RgbaImage::new(32, 32, pixels).unwrap();
        let smooth = clothing_retouch(&source, &json!({"smooth":90,"texture":15}));
        let textured = clothing_retouch(&source, &json!({"smooth":90,"texture":100}));
        assert_ne!(smooth.pixels, source.pixels);
        assert_eq!(smooth.pixels[3], 123);
        let detail = |image: &RgbaImage| -> u32 {
            let a = image.pixels[(15 * 32 + 15) * 4 + 1] as i32;
            let b = image.pixels[(15 * 32 + 16) * 4 + 1] as i32;
            (a - b).unsigned_abs()
        };
        assert!(detail(&textured) >= detail(&smooth));
    }

    #[test]
    fn skin_smoothing_cannot_import_color_from_outside_the_selection() {
        let make = |outside: [u8; 3]| {
            let mut pixels = Vec::new();
            for y in 0..48 {
                for x in 0..48 {
                    let inside = (x as f32 + 0.5 - 24.0).powi(2) + (y as f32 + 0.5 - 24.0).powi(2)
                        <= 10.0f32.powi(2);
                    let rgb = if inside {
                        let detail = if (x + y) % 2 == 0 { 8 } else { 0 };
                        [180 + detail, 140 + detail, 110 + detail]
                    } else {
                        outside
                    };
                    pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                }
            }
            RgbaImage::new(48, 48, pixels).unwrap()
        };
        let params = json!({"smooth":100,"texture":0,
        "mask":{"shape":"ellipse","region":{
            "x":14.0/48.0,"y":14.0/48.0,"width":20.0/48.0,"height":20.0/48.0
        }}});
        let a = make([180, 140, 110]);
        let b = make([70, 40, 30]);
        let ca = crate::mask::composite(&a, &skin(&a, &params), &params);
        let cb = crate::mask::composite(&b, &skin(&b, &params), &params);
        let mut changed = 0;
        for y in 0..48 {
            for x in 0..48 {
                let i = (y * 48 + x) * 4;
                if a.pixels[i..i + 4] == b.pixels[i..i + 4] {
                    assert_eq!(&ca.pixels[i..i + 4], &cb.pixels[i..i + 4]);
                    changed += usize::from(ca.pixels[i..i + 4] != a.pixels[i..i + 4]);
                } else {
                    assert_eq!(&ca.pixels[i..i + 4], &a.pixels[i..i + 4]);
                    assert_eq!(&cb.pixels[i..i + 4], &b.pixels[i..i + 4]);
                }
            }
        }
        assert!(changed > 100, "selected skin must still be smoothed");
    }

    #[test]
    fn clothing_smoothing_excludes_unselected_and_frozen_color_donors() {
        // Test both a cropped region and an explicit mask. A frozen strip
        // inside the garment and pixels outside it carry different colors;
        // neither may alter the adjacent editable fabric's filtered result.
        for selection in [
            json!({"region":{"x":0,"y":0,"width":0.625,"height":1}}),
            json!({"mask":{"shape":"rect","region":{"x":0,"y":0,"width":0.625,"height":1}}}),
        ] {
            let make = |excluded: [u8; 3]| {
                let mut pixels = Vec::new();
                for y in 0..32 {
                    for x in 0..48 {
                        let rgb = if x >= 30 || (12..16).contains(&x) {
                            excluded
                        } else {
                            let weave = if (x + y) % 2 == 0 { 12 } else { 0 };
                            [180 + weave, 140 + weave, 110 + weave]
                        };
                        pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 173]);
                    }
                }
                RgbaImage::new(48, 32, pixels).unwrap()
            };
            let mut params = selection;
            params["adjustments"] = json!({"smooth":100,"texture":0});
            params["protectMask"] = json!({"shape":"rect","region":{
                "x":12.0/48.0,"y":0,"width":4.0/48.0,"height":1
            }});
            let a = make([180, 140, 110]);
            let b = make([255, 0, 0]);
            let ca = crate::mask::composite(&a, &clothing_retouch(&a, &params), &params);
            let cb = crate::mask::composite(&b, &clothing_retouch(&b, &params), &params);
            let mut changed = 0;
            for y in 0..32 {
                for x in 0..48 {
                    let i = (y * 48 + x) * 4;
                    if x >= 30 || (12..16).contains(&x) {
                        assert_eq!(&ca.pixels[i..i + 4], &a.pixels[i..i + 4]);
                        assert_eq!(&cb.pixels[i..i + 4], &b.pixels[i..i + 4]);
                    } else {
                        assert_eq!(&ca.pixels[i..i + 4], &cb.pixels[i..i + 4]);
                        assert_eq!(ca.pixels[i + 3], 173);
                        changed += usize::from(ca.pixels[i..i + 3] != a.pixels[i..i + 3]);
                    }
                }
            }
            assert!(changed > 100, "eligible fabric must still be softened");
        }
    }

    #[test]
    fn structural_eye_bag_and_tear_trough_are_not_faked_with_brightness() {
        let source = RgbaImage::new(24, 24, [120, 100, 90, 255].repeat(24 * 24)).unwrap();
        // Capability validation rejects these adjustments publicly. The
        // color implementation must also never counterfeit their geometry.
        assert_eq!(
            local_color(
                &source,
                &json!({"adjustments":{"eyeBag":100,"tearTrough":100}}),
                false
            ),
            source
        );
        assert_ne!(
            local_color(&source, &json!({"adjustments":{"darkCircle":100}}), false),
            source
        );
    }
    #[test]
    fn hsl_changes_selected_hue_without_changing_alpha() {
        let image = RgbaImage::new(1, 1, vec![255, 0, 0, 91]).unwrap();
        let result = hsl(&image, &json!({"hsl":[{"hue":0,"range":45,"shift":100}]}));
        assert!(result.pixels[1] > 200);
        assert_eq!(result.pixels[3], 91);
    }
    #[test]
    fn semantic_skin_mask_protects_features_and_background() {
        let mut pixels = [120, 100, 90, 255].repeat(24 * 24);
        // Bright red feature and dark eye patches are inside a face region.
        for y in 15..18 {
            for x in 9..15 {
                let index = (y * 24 + x) * 4;
                pixels[index..index + 4].copy_from_slice(&[210, 30, 40, 255]);
            }
        }
        let source = RgbaImage::new(24, 24, pixels).unwrap();
        let skin_mask = json!({"shape":"polygon","points":[
            {"x":0.20,"y":0.15},{"x":0.80,"y":0.15},{"x":0.80,"y":0.90},{"x":0.20,"y":0.90}
        ]});
        let lips = json!({"shape":"polygon","points":[
            {"x":0.35,"y":0.60},{"x":0.65,"y":0.60},{"x":0.65,"y":0.78},{"x":0.35,"y":0.78}
        ]});
        let result = skin(
            &source,
            &json!({"smooth":70,"whiten":100,
            "semanticMasks":{"skin":skin_mask,"lips":lips}}),
        );
        // Outside the semantic face mask remains byte-identical.
        assert_eq!(&result.pixels[..4], &source.pixels[..4]);
        // Lip pixels are protected from skin whitening.
        let lip = (16 * 24 + 12) * 4;
        assert_eq!(&result.pixels[lip..lip + 4], &source.pixels[lip..lip + 4]);
        // A regular skin pixel is changed by whitening.
        let cheek = (8 * 24 + 12) * 4;
        assert!(result.pixels[cheek] > source.pixels[cheek]);
    }

    #[test]
    fn local_makeup_uses_lip_polygon_instead_of_face_ellipse() {
        let source = RgbaImage::new(24, 24, [100, 100, 100, 255].repeat(24 * 24)).unwrap();
        let params = json!({
            "region":{"x":0,"y":0,"width":1,"height":1},
            "adjustments":{"lipColor":100,"browColor":0},
            "semanticMasks":{"lips":[
                {"x":0.40,"y":0.60},{"x":0.60,"y":0.60},{"x":0.60,"y":0.72},{"x":0.40,"y":0.72}
            ]}
        });
        let result = local_color(&source, &params, false);
        let inside = (16 * 24 + 12) * 4;
        let outside = (8 * 24 + 12) * 4;
        assert!(result.pixels[inside] > source.pixels[inside]);
        assert_eq!(
            &result.pixels[outside..outside + 4],
            &source.pixels[outside..outside + 4]
        );
    }

    #[test]
    fn skin_retouch_preserves_explicit_hair_and_skin_protection() {
        let source = RgbaImage::new(6, 2, [120, 100, 90, 77].repeat(12)).unwrap();
        let output = skin(
            &source,
            &json!({"whiten":100,"warmth":100,"redness":100,"smooth":70,
            "semanticMasks":{
                "skin":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
                "hair":{"shape":"rect","region":{"x":0,"y":0,"width":0.333333,"height":1}},
                "skinProtection":{"shape":"rect","region":{"x":0.333333,"y":0,"width":0.333334,"height":1}}
            }}),
        );
        for y in 0..2 {
            for x in 0..4 {
                let index = (y * 6 + x) * 4;
                assert_eq!(
                    &output.pixels[index..index + 4],
                    &source.pixels[index..index + 4],
                    "protected hair/skin pixel must remain identical"
                );
            }
        }
        assert_ne!(&output.pixels[16..19], &source.pixels[16..19]);
        assert!(output.pixels.chunks_exact(4).all(|pixel| pixel[3] == 77));
    }

    #[test]
    fn lip_adjustments_leave_inner_mouth_and_teeth_untinted() {
        let source = RgbaImage::new(
            4,
            1,
            vec![
                140, 60, 65, 31, 140, 60, 65, 31, 205, 195, 175, 31, 40, 30, 30, 31,
            ],
        )
        .unwrap();
        let output = local_color(
            &source,
            &json!({"adjustments":{"lipColor":100,"lipBrightness":60,"lipSaturation":60},
            "semanticMasks":{
                "lips":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
                "teeth":{"shape":"rect","region":{"x":0.5,"y":0,"width":0.5,"height":1}}
            }}),
            false,
        );
        assert_ne!(&output.pixels[..3], &source.pixels[..3]);
        // The inner-mouth polygon excludes both bright enamel and dark cavity.
        assert_eq!(&output.pixels[8..], &source.pixels[8..]);
        assert!(output.pixels.chunks_exact(4).all(|pixel| pixel[3] == 31));
    }

    #[test]
    fn teeth_whitening_runs_without_lip_mask_and_rejects_red_and_dark_pixels() {
        let source = RgbaImage::new(
            3,
            1,
            vec![210, 195, 160, 73, 210, 30, 40, 73, 40, 35, 35, 73],
        )
        .unwrap();
        let output = local_color(
            &source,
            &json!({"adjustments":{"teethWhiten":100},"semanticMasks":{
                "teeth":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}
            }}),
            false,
        );
        let before_light = luma(210.0, 195.0, 160.0);
        let after_light = luma(
            output.pixels[0] as f32,
            output.pixels[1] as f32,
            output.pixels[2] as f32,
        );
        assert!(after_light > before_light + 5.0);
        assert!(output.pixels[0] - output.pixels[2] < 210 - 160);
        assert_eq!(&output.pixels[4..], &source.pixels[4..]);
        assert_eq!(output.pixels[3], 73);
    }

    #[test]
    fn under_eye_controls_protect_eyes_and_brows_even_when_masks_overlap() {
        let source = RgbaImage::new(3, 1, [100, 90, 85, 55].repeat(3)).unwrap();
        let output = local_color(
            &source,
            &json!({"adjustments":{"eyeBag":100,"darkCircle":100,"tearTrough":100},
            "semanticMasks":{
                "underEyes":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
                "eyes":{"shape":"rect","region":{"x":0,"y":0,"width":0.333333,"height":1}},
                "brows":{"shape":"rect","region":{"x":0.333333,"y":0,"width":0.333334,"height":1}}
            }}),
            false,
        );
        assert_eq!(&output.pixels[..8], &source.pixels[..8]);
        assert!(output.pixels[8] > source.pixels[8]);
        assert!(output.pixels.chunks_exact(4).all(|pixel| pixel[3] == 55));
    }

    #[test]
    fn skin_tone_separates_luminance_and_chroma_and_protects_highlights() {
        let source = RgbaImage::new(1, 1, vec![175, 135, 100, 19]).unwrap();
        let warm = skin(&source, &json!({"warmth":100}));
        let cool = skin(&source, &json!({"warmth":-100}));
        let whitened = skin(&source, &json!({"whiten":100}));
        let pixel_light = |image: &RgbaImage| {
            luma(
                image.pixels[0] as f32,
                image.pixels[1] as f32,
                image.pixels[2] as f32,
            )
        };
        // Warm/cool changes are chromatic, preserving brightness within byte rounding.
        assert!((pixel_light(&warm) - pixel_light(&source)).abs() < 1.0);
        assert!((pixel_light(&cool) - pixel_light(&source)).abs() < 1.0);
        assert!(warm.pixels[0] > source.pixels[0]);
        assert!(cool.pixels[2] > source.pixels[2]);
        assert!(pixel_light(&whitened) > pixel_light(&source) + 20.0);
        assert_eq!(whitened.pixels[0] - whitened.pixels[1], 175 - 135);
        assert_eq!(whitened.pixels[1] - whitened.pixels[2], 135 - 100);

        let highlights = RgbaImage::new(2, 1, vec![248, 244, 238, 19, 255, 255, 255, 19]).unwrap();
        let adjusted = skin(
            &highlights,
            &json!({"whiten":100,"warmth":100,"redness":100}),
        );
        assert!(adjusted.pixels[..3].iter().all(|value| *value < 255));
        assert_eq!(&adjusted.pixels[4..], &highlights.pixels[4..]);
        assert!(adjusted.pixels.chunks_exact(4).all(|pixel| pixel[3] == 19));
    }

    #[test]
    fn explicit_skin_mask_is_authoritative_over_larger_face_mask() {
        let source = RgbaImage::new(4, 1, [150, 120, 100, 91].repeat(4)).unwrap();
        let output = skin(
            &source,
            &json!({"whiten":100,"warmth":70,"redness":50,"semanticMasks":{
                "face":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
                "skin":{"shape":"rect","region":{"x":0,"y":0,"width":0.5,"height":1}}
            }}),
        );
        assert_ne!(&output.pixels[..3], &source.pixels[..3]);
        assert_eq!(&output.pixels[8..], &source.pixels[8..]);
    }

    #[test]
    fn protected_feature_colors_cannot_contaminate_neighboring_skin_smoothing() {
        let mut pixels = Vec::new();
        for y in 0..20 {
            for x in 0..24 {
                let texture = if (x + y) % 2 == 0 { 5 } else { 0 };
                pixels.extend_from_slice(&[160 + texture, 130 + texture, 110 + texture, 201]);
            }
        }
        let mut first = RgbaImage::new(24, 20, pixels.clone()).unwrap();
        let mut second = RgbaImage::new(24, 20, pixels).unwrap();
        for y in 5..15 {
            for x in 10..14 {
                let index = (y * 24 + x) * 4;
                first.pixels[index..index + 3].copy_from_slice(&[220, 40, 40]);
                second.pixels[index..index + 3].copy_from_slice(&[80, 190, 240]);
            }
        }
        for semantic in [
            "eyes",
            "brows",
            "lips",
            "hair",
            "skinProtection",
            "protectMask",
        ] {
            let mut params = json!({"smooth":100,"texture":30,"semanticMasks":{
                "skin":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}
            }});
            let protection = json!({"shape":"rect","region":{
                "x":10.0/24.0,"y":0.25,"width":4.0/24.0,"height":0.5
            }});
            if semantic == "protectMask" {
                params["protectMask"] = protection;
            } else {
                params["semanticMasks"][semantic] = protection;
            }
            let left = skin(&first, &params);
            let right = skin(&second, &params);
            assert_ne!(
                &left.pixels[..3],
                &first.pixels[..3],
                "skin texture should still soften"
            );
            for y in 0..20 {
                for x in 0..24 {
                    if (10..14).contains(&x) && (5..15).contains(&y) {
                        continue;
                    }
                    let index = (y * 24 + x) * 4;
                    assert_eq!(
                        &left.pixels[index..index + 4],
                        &right.pixels[index..index + 4],
                        "{semantic} donor color leaked into skin at {x},{y}"
                    );
                }
            }
        }
    }

    #[test]
    fn skin_color_preserves_deep_shadows_and_specular_highlights() {
        let source = RgbaImage::new(
            3,
            1,
            vec![18, 14, 10, 87, 160, 130, 110, 87, 255, 255, 255, 87],
        )
        .unwrap();
        let output = skin(&source, &json!({"whiten":100,"warmth":100,"redness":100}));
        assert_eq!(&output.pixels[..4], &source.pixels[..4]);
        assert_ne!(&output.pixels[4..7], &source.pixels[4..7]);
        assert_eq!(&output.pixels[8..], &source.pixels[8..]);
    }

    #[test]
    fn lut_rejects_bad_point_count() {
        let image = RgbaImage::new(1, 1, vec![0; 4]).unwrap();
        assert!(lut(&image, &json!({"size":2,"data":[]})).is_err());
    }
}
