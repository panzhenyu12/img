//! Hand-tuned film response families. These are inspired looks, not measured
//! camera profiles. Linear-light tone/chroma separation keeps highlights and
//! saturated colors continuous, and lookup tables bound the browser cost.

struct Recipe {
    contrast: f32,
    saturation: f32,
    lift: f32,
    shadow: [f32; 3],
    highlight: [f32; 3],
    shoulder: f32,
    warm_compression: f32,
}

pub(crate) struct FilmResponse {
    recipe: Recipe,
    decode: [f32; 256],
    encode: Vec<f32>,
    tone: Vec<f32>,
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lookup(table: &[f32], x: f32) -> f32 {
    let position = x.clamp(0.0, 1.0) * (table.len() - 1) as f32;
    let i = (position as usize).min(table.len() - 2);
    table[i] + (table[i + 1] - table[i]) * (position - i as f32)
}

impl FilmResponse {
    pub(crate) fn new(preset: &str) -> Option<Self> {
        let (contrast, saturation, lift, shadow, highlight, shoulder) = match preset {
            "provia" => (
                1.06,
                1.06,
                0.0,
                [0.0, 0.004, 0.009],
                [0.010, 0.004, -0.003],
                0.08,
            ),
            "astia" => (
                0.96,
                1.03,
                0.003,
                [0.002, 0.001, 0.004],
                [0.012, 0.004, -0.004],
                0.10,
            ),
            "velvia" => (
                1.12,
                1.28,
                0.0,
                [-0.002, 0.004, 0.012],
                [0.014, 0.004, -0.008],
                0.12,
            ),
            "classicNegative" => (
                1.04,
                0.88,
                0.004,
                [0.0, 0.004, 0.012],
                [0.015, 0.004, -0.006],
                0.10,
            ),
            "eterna" => (
                0.90,
                0.78,
                0.006,
                [0.0, 0.006, 0.012],
                [0.010, 0.004, -0.001],
                0.15,
            ),
            "acros" => (1.12, 0.0, 0.0, [0.0; 3], [0.0; 3], 0.10),
            _ => return None,
        };
        let recipe = Recipe {
            contrast,
            saturation,
            lift,
            shadow,
            highlight,
            shoulder,
            warm_compression: if preset == "velvia" { 0.72 } else { 0.0 },
        };
        let decode = std::array::from_fn(|i| {
            let v = i as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                libm::powf((v + 0.055) / 1.055, 2.4)
            }
        });
        let encode = (0..=4096)
            .map(|i| {
                let v = i as f32 / 4096.0;
                255.0
                    * if v <= 0.0031308 {
                        v * 12.92
                    } else {
                        1.055 * libm::powf(v, 1.0 / 2.4) - 0.055
                    }
            })
            .collect();
        // Odds-power contrast pivots around 18% linear gray. Unlike a
        // straight-line contrast offset it never flattens a range to black.
        let k = libm::powf(0.18 / 0.82, recipe.contrast - 1.0);
        let tone = (0..=4096)
            .map(|i| {
                let y = i as f32 / 4096.0;
                let a = libm::powf(y, recipe.contrast);
                let b = libm::powf(1.0 - y, recipe.contrast);
                let t = a / (a + k * b);
                let lifted = t + recipe.lift * (1.0 - t).powi(2);
                lifted - recipe.shoulder * smoothstep(0.35, 1.0, lifted) * lifted
            })
            .collect();
        Some(Self {
            recipe,
            decode,
            encode,
            tone,
        })
    }

    pub(crate) fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let linear = rgb.map(|v| self.decode[v.clamp(0.0, 255.0).round() as usize]);
        let y = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
        let tone = lookup(&self.tone, y);
        // Chroma tracks the changed luminance so a lifted shadow does not
        // turn gray or a compressed highlight clip one color channel first.
        let ratio = if y > 1e-6 { tone / y } else { 1.0 };
        // A continuous warm-hue shoulder limits the *extra* colorfulness in
        // moderate red/yellow colors. This includes many skin colors but is
        // not a skin detector: wood and other similar colors also follow it.
        // Saturated reds/yellows and the green/blue landscape response retain
        // the recipe. The ramps avoid a visible threshold on cheek gradients.
        let warm = if self.recipe.warm_compression > 0.0 && rgb[0] > rgb[1] && rgb[1] > rgb[2] {
            let chroma = (rgb[0] - rgb[2]) / 255.0;
            let hue = (rgb[1] - rgb[2]) / (rgb[0] - rgb[2]);
            smoothstep(0.12, 0.28, hue)
                * (1.0 - smoothstep(0.62, 1.0, hue))
                * smoothstep(0.03, 0.10, chroma)
                * (1.0 - smoothstep(0.45, 0.70, chroma))
        } else {
            0.0
        } * self.recipe.warm_compression;
        let saturation = if warm > 0.0 {
            1.0 + (self.recipe.saturation - 1.0) * (1.0 - warm)
        } else {
            self.recipe.saturation
        } * ratio;
        let shadow = (1.0 - smoothstep(0.03, 0.38, y)) * smoothstep(0.0, 0.05, y);
        let highlight = smoothstep(0.28, 0.92, y);
        let mut chroma = linear.map(|v| (v - y) * saturation);
        for (c, value) in chroma.iter_mut().enumerate() {
            *value += (shadow * self.recipe.shadow[c] + highlight * self.recipe.highlight[c])
                * (1.0 - warm * 0.65);
        }
        let tint_luma = 0.2126 * chroma[0] + 0.7152 * chroma[1] + 0.0722 * chroma[2];
        for v in &mut chroma {
            *v -= tint_luma;
        }
        // Uniform gamut compression avoids hue shifts from per-channel clips.
        let mut scale = 1.0_f32;
        for c in chroma {
            if c > 0.0 {
                scale = scale.min((1.0 - tone) / c);
            } else if c < 0.0 {
                scale = scale.min(tone / -c);
            }
        }
        chroma.map(|c| lookup(&self.encode, tone + c * scale.clamp(0.0, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vivid_warm_gradients_remain_continuous_without_muting_landscape_colors() {
        let response = FilmResponse::new("velvia").unwrap();
        // Cheek-like ramps across the hue shoulder must not get a visible
        // classification seam. Sample integer inputs used by the 8bit core.
        for (r, b) in [(190.0, 125.0), (145.0, 80.0), (225.0, 160.0)] {
            let mut last = response.apply([r, b, b]);
            for g in b as u32 + 1..=r as u32 {
                let rgb = response.apply([r, g as f32, b]);
                let jump = rgb
                    .iter()
                    .zip(last)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(jump < 3.0, "warm hue seam at {r},{g},{b}: {jump}");
                last = rgb;
            }
        }
        let provia = FilmResponse::new("provia").unwrap();
        for source in [
            [60.0, 140.0, 80.0],
            [60.0, 120.0, 190.0],
            [205.0, 190.0, 60.0],
        ] {
            let chroma = |rgb: [f32; 3]| {
                rgb.into_iter().fold(f32::NEG_INFINITY, f32::max)
                    - rgb.into_iter().fold(f32::INFINITY, f32::min)
            };
            assert!(
                chroma(response.apply(source)) > chroma(provia.apply(source)),
                "vivid profile lost landscape color separation"
            );
        }
    }

    #[test]
    fn gray_ramps_keep_shadow_separation_and_are_monotone() {
        for name in [
            "provia",
            "astia",
            "velvia",
            "classicNegative",
            "eterna",
            "acros",
        ] {
            let response = FilmResponse::new(name).unwrap();
            let mut last = -1.0;
            for v in 0..=255 {
                let p = response.apply([v as f32; 3]);
                let y = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                assert!(y > last, "{name} flattens/reverses gray {v}: {last} -> {y}");
                last = y;
                assert!(p
                    .into_iter()
                    .all(|v| v.is_finite() && (0.0..=255.0).contains(&v)));
            }
        }
    }
}
