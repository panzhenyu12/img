use crate::engine::{byte, number};
use crate::RgbaImage;
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Region {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Region {
    pub fn read(value: Option<&Value>) -> Self {
        value
            .map(|v| Self {
                x: number(v, "x", 0.0),
                y: number(v, "y", 0.0),
                width: number(v, "width", 1.0).max(0.0001),
                height: number(v, "height", 1.0).max(0.0001),
            })
            .unwrap_or(Self {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            })
    }
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x <= self.x + self.width && y <= self.y + self.height
    }
}

fn smooth_step(edge0: f32, edge1: f32, value: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn segment_distance(x: f32, y: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy).max(1e-10);
    let t = t.clamp(0.0, 1.0);
    ((x - ax - t * dx).powi(2) + (y - ay - t * dy).powi(2)).sqrt()
}

/// Inline masks are shared by UI selections, AI operations and native export.
/// Feather is 0..100 and spans up to 20% of the shorter canvas dimension.
pub(crate) fn coverage(mask: &Value, x: f32, y: f32, width: u32, height: u32) -> f32 {
    let feather = number(mask, "feather", 0.0).clamp(0.0, 100.0) / 100.0 * 0.2;
    let region = Region::read(mask.get("region").or(Some(mask)));
    let shape = mask
        .get("shape")
        .or_else(|| mask.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("rect");
    let mut value = match shape {
        "raster" => raster_coverage(mask, x, y),
        "linear" => {
            let from = mask.get("from");
            let to = mask.get("to");
            let ax = from.map(|p| number(p, "x", region.x)).unwrap_or(region.x);
            let ay = from.map(|p| number(p, "y", region.y)).unwrap_or(region.y);
            let bx = to
                .map(|p| number(p, "x", region.x + region.width))
                .unwrap_or(region.x + region.width);
            let by = to
                .map(|p| number(p, "y", region.y + region.height))
                .unwrap_or(region.y + region.height);
            let dx = bx - ax;
            let dy = by - ay;
            smooth_step(
                0.0,
                1.0,
                ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy).max(1e-8),
            )
        }
        "radial" => {
            let nx = (x - region.x - region.width * 0.5) / (region.width * 0.5);
            let ny = (y - region.y - region.height * 0.5) / (region.height * 0.5);
            1.0 - smooth_step(0.0, 1.0, (nx * nx + ny * ny).sqrt())
        }
        "ellipse" => {
            let nx = (x - region.x - region.width * 0.5) / (region.width * 0.5);
            let ny = (y - region.y - region.height * 0.5) / (region.height * 0.5);
            let signed = (1.0 - (nx * nx + ny * ny).sqrt()) * region.width.min(region.height) * 0.5;
            if feather > 0.0 {
                smooth_step(-feather * 0.5, feather * 0.5, signed)
            } else if signed >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        "polygon" => {
            let points = mask.get("points").and_then(Value::as_array);
            if let Some(points) = points.filter(|p| p.len() >= 3) {
                let mut inside = false;
                let mut distance = f32::MAX;
                for i in 0..points.len() {
                    let a = &points[i];
                    let b = &points[(i + 1) % points.len()];
                    let ax = number(a, "x", 0.0);
                    let ay = number(a, "y", 0.0);
                    let bx = number(b, "x", 0.0);
                    let by = number(b, "y", 0.0);
                    if (ay > y) != (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax {
                        inside = !inside;
                    }
                    distance = distance.min(segment_distance(x, y, ax, ay, bx, by));
                }
                if feather > 0.0 {
                    smooth_step(
                        -feather * 0.5,
                        feather * 0.5,
                        if inside { distance } else { -distance },
                    )
                } else if inside {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        "brush" => 0.0,
        _ => {
            let signed = (x - region.x)
                .min(region.x + region.width - x)
                .min(y - region.y)
                .min(region.y + region.height - y);
            if feather > 0.0 {
                smooth_step(-feather * 0.5, feather * 0.5, signed)
            } else if signed >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
    };
    // Shape selections become editable masks without changing their JSON
    // shape. Add/erase stamps therefore work on every initial shape.
    if let Some(strokes) = mask.get("strokes").and_then(Value::as_array) {
        for stroke in strokes {
            let radius = number(stroke, "radius", 0.05).max(0.0001);
            let dx = (x - number(stroke, "x", 0.0)) * width as f32;
            let dy = (y - number(stroke, "y", 0.0)) * height as f32;
            let distance = (dx * dx + dy * dy).sqrt() / width.min(height) as f32;
            let softness = feather.max(radius * 0.12).min(radius);
            let brush = (1.0 - smooth_step(radius - softness, radius, distance))
                * number(stroke, "strength", 100.0).clamp(0.0, 100.0)
                / 100.0;
            if stroke
                .get("erase")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                value *= 1.0 - brush;
            } else {
                value = value.max(brush);
            }
        }
    }
    if mask.get("invert").and_then(Value::as_bool).unwrap_or(false) {
        1.0 - value
    } else {
        value
    }
}

/// Single-channel alpha in full-image coordinates, sampled at pixel centers.
/// Detector masks keep gaps between limbs without converting them to a hull.
fn raster_coverage(mask: &Value, x: f32, y: f32) -> f32 {
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        return 0.0;
    }
    let w = mask.get("width").and_then(Value::as_u64).unwrap_or(0) as usize;
    let h = mask.get("height").and_then(Value::as_u64).unwrap_or(0) as usize;
    let Some(data) = mask.get("data").and_then(Value::as_array) else {
        return 0.0;
    };
    if w == 0 || h == 0 || w.checked_mul(h) != Some(data.len()) {
        return 0.0;
    }
    let px = (x * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let py = (y * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = px.floor() as usize;
    let y0 = py.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let read = |ix: usize, iy: usize| data[iy * w + ix].as_f64().unwrap_or(0.0) as f32 / 255.0;
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;
    let top = read(x0, y0) * (1.0 - tx) + read(x1, y0) * tx;
    let bottom = read(x0, y1) * (1.0 - tx) + read(x1, y1) * tx;
    (top * (1.0 - ty) + bottom * ty).clamp(0.0, 1.0)
}

fn blend_channel(base: f32, edit: f32, mode: &str) -> f32 {
    match mode {
        "multiply" => base * edit / 255.0,
        "screen" => 255.0 - (255.0 - base) * (255.0 - edit) / 255.0,
        "overlay" => {
            if base < 128.0 {
                2.0 * base * edit / 255.0
            } else {
                255.0 - 2.0 * (255.0 - base) * (255.0 - edit) / 255.0
            }
        }
        "softLight" | "soft-light" => {
            let b = base / 255.0;
            let e = edit / 255.0;
            255.0 * ((1.0 - 2.0 * e) * b * b + 2.0 * e * b)
        }
        "darken" => base.min(edit),
        "lighten" => base.max(edit),
        _ => edit,
    }
}

pub(crate) fn composite(base: &RgbaImage, edit: &RgbaImage, params: &Value) -> RgbaImage {
    let mut output = edit.clone();
    let opacity = number(params, "opacity", 100.0).clamp(0.0, 100.0) / 100.0;
    let region = Region::read(params.get("region"));
    let mode = params
        .get("blendMode")
        .and_then(Value::as_str)
        .unwrap_or("normal");
    for y in 0..base.height {
        for x in 0..base.width {
            let nx = (x as f32 + 0.5) / base.width as f32;
            let ny = (y as f32 + 0.5) / base.height as f32;
            let mut weight = if region.contains(nx, ny) {
                opacity
            } else {
                0.0
            };
            if let Some(selection) = params.get("mask").filter(|v| v.is_object()) {
                weight *= coverage(selection, nx, ny, base.width, base.height);
            }
            if let Some(protection) = params.get("protectMask").filter(|v| v.is_object()) {
                weight *= 1.0 - coverage(protection, nx, ny, base.width, base.height);
            }
            let index = (y as usize * base.width as usize + x as usize) * 4;
            if weight < 1e-8 {
                output.pixels[index..index + 4].copy_from_slice(&base.pixels[index..index + 4]);
                continue;
            }
            let base_alpha = base.pixels[index + 3] as f32 / 255.0;
            let edit_alpha = edit.pixels[index + 3] as f32 / 255.0;
            let alpha = base_alpha * (1.0 - weight) + edit_alpha * weight;
            for channel in 0..3 {
                let original = base.pixels[index + channel] as f32;
                let changed = blend_channel(original, edit.pixels[index + channel] as f32, mode);
                output.pixels[index + channel] = if alpha > 1e-8 {
                    byte(
                        (original * base_alpha * (1.0 - weight) + changed * edit_alpha * weight)
                            / alpha,
                    )
                } else {
                    byte(original + (changed - original) * weight)
                };
            }
            output.pixels[index + 3] = byte(alpha * 255.0);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn raster_alpha_preserves_holes_and_interpolates_at_pixel_centers() {
        let mask = json!({"shape":"raster","width":3,"height":2,"data":[255,0,255,255,0,255]});
        assert_eq!(coverage(&mask, 0.5, 0.25, 300, 200), 0.0);
        assert_eq!(coverage(&mask, 1.0 / 6.0, 0.25, 300, 200), 1.0);
        assert!((coverage(&mask, 1.0 / 3.0, 0.25, 300, 200) - 0.5).abs() < 1e-6);
        assert_eq!(coverage(&mask, -0.1, 0.5, 300, 200), 0.0);
    }
    #[test]
    fn protect_mask_keeps_background_exact() {
        let base = RgbaImage::new(2, 1, vec![10, 20, 30, 40, 50, 60, 70, 80]).unwrap();
        let edit = RgbaImage::new(2, 1, vec![200, 200, 200, 40, 200, 200, 200, 80]).unwrap();
        let params = json!({"protectMask":{"shape":"rect","region":{"x":0.0,"y":0.0,"width":0.5,"height":1.0}}});
        let out = composite(&base, &edit, &params);
        assert_eq!(&out.pixels[..4], &base.pixels[..4]);
        assert_eq!(&out.pixels[4..], &edit.pixels[4..]);
    }
    #[test]
    fn brush_erase_and_invert_have_explicit_semantics() {
        let mask = json!({"shape":"brush","strokes":[{"x":0.5,"y":0.5,"radius":0.2},{"x":0.5,"y":0.5,"radius":0.1,"erase":true}]});
        assert_eq!(coverage(&mask, 0.5, 0.5, 100, 100), 0.0);
        let mask = json!({"shape":"ellipse","region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},"invert":true});
        assert_eq!(coverage(&mask, 0.5, 0.5, 100, 100), 0.0);
        assert_eq!(coverage(&mask, 0.0, 0.0, 100, 100), 1.0);
    }
    #[test]
    fn rectangular_selection_can_be_erased_with_a_mask_brush() {
        let mask = json!({"shape":"rect","region":{"x":0.1,"y":0.1,"width":0.8,"height":0.8},"strokes":[{"x":0.5,"y":0.5,"radius":0.1,"erase":true}]});
        assert_eq!(coverage(&mask, 0.5, 0.5, 100, 100), 0.0);
        assert_eq!(coverage(&mask, 0.2, 0.2, 100, 100), 1.0);
    }
}
