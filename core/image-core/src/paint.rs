use crate::engine::{byte, number};
use crate::{ImageError, RgbaImage};
use serde_json::Value;

/// Paint into a temporary transparent layer. Erase changes that layer's alpha
/// and can never delete source pixels. The graph's common mask/protection and
/// opacity pass is applied after this layer has been rendered.
pub(crate) fn render(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let color = p
        .get("color")
        .and_then(Value::as_array)
        .filter(|v| v.len() == 3)
        .ok_or_else(|| ImageError::InvalidOperation("paint requires RGB color".into()))?;
    let color = [
        color[0].as_f64().unwrap_or(0.0) as f32,
        color[1].as_f64().unwrap_or(0.0) as f32,
        color[2].as_f64().unwrap_or(0.0) as f32,
    ];
    let Some(strokes) = p.get("strokes").and_then(Value::as_array) else {
        return Ok(source.clone());
    };
    let mut alpha = vec![0.0f32; source.width as usize * source.height as usize];
    for stroke in strokes {
        let x = number(stroke, "x", 0.5) * source.width as f32;
        let y = number(stroke, "y", 0.5) * source.height as f32;
        let radius = number(stroke, "radius", 0.04).clamp(0.001, 1.0)
            * source.width.min(source.height) as f32;
        let strength = number(stroke, "strength", 100.0).clamp(0.0, 100.0) / 100.0;
        let erase = stroke
            .get("erase")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let xmin = (x - radius).floor().max(0.0) as u32;
        let xmax = (x + radius).ceil().min(source.width as f32) as u32;
        let ymin = (y - radius).floor().max(0.0) as u32;
        let ymax = (y + radius).ceil().min(source.height as f32) as u32;
        for py in ymin..ymax {
            for px in xmin..xmax {
                let distance = ((px as f32 - x).powi(2) + (py as f32 - y).powi(2)).sqrt() / radius;
                if distance >= 1.0 {
                    continue;
                }
                let weight = if distance < 0.75 {
                    strength
                } else {
                    let t = (1.0 - distance) / 0.25;
                    t * t * (3.0 - 2.0 * t) * strength
                };
                let index = py as usize * source.width as usize + px as usize;
                if erase {
                    alpha[index] *= 1.0 - weight;
                } else {
                    alpha[index] = alpha[index] + weight * (1.0 - alpha[index]);
                }
            }
        }
    }
    let mut out = source.clone();
    for (index, brush_alpha) in alpha.into_iter().enumerate() {
        let source_alpha = source.pixels[index * 4 + 3] as f32 / 255.0;
        let final_alpha = brush_alpha + source_alpha * (1.0 - brush_alpha);
        if brush_alpha < 1e-8 {
            continue;
        }
        for (c, value) in color.iter().enumerate() {
            out.pixels[index * 4 + c] = byte(
                (*value * brush_alpha
                    + source.pixels[index * 4 + c] as f32 * source_alpha * (1.0 - brush_alpha))
                    / final_alpha.max(1e-8),
            );
        }
        out.pixels[index * 4 + 3] = byte(final_alpha * 255.0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply_graph_json;
    use serde_json::json;
    #[test]
    fn paint_erase_restores_source_and_protect_mask_freezes_pixels() {
        let source = RgbaImage::new(20, 20, [10, 20, 30, 255].repeat(400)).unwrap();
        let params = json!({"color":[255,0,0],"strokes":[{"x":0.5,"y":0.5,"radius":0.3,"strength":100},{"x":0.5,"y":0.5,"radius":0.1,"strength":100,"erase":true}],"protectMask":{"shape":"rect","region":{"x":0,"y":0,"width":0.5,"height":1}}});
        let graph = json!({"operations":[{"type":"paint.brush","params":params}]});
        let result = apply_graph_json(&source, &graph.to_string()).unwrap();
        let center = (10 * 20 + 10) * 4;
        assert_eq!(
            &result.pixels[center..center + 4],
            &source.pixels[center..center + 4]
        );
        let painted = (10 * 20 + 13) * 4;
        assert_eq!(&result.pixels[painted..painted + 4], &[255, 0, 0, 255]);
        let protected = (10 * 20 + 8) * 4;
        assert_eq!(
            &result.pixels[protected..protected + 4],
            &source.pixels[protected..protected + 4]
        );
    }
    #[test]
    fn paint_on_transparent_source_keeps_color_under_partial_opacity() {
        let source = RgbaImage::new(20, 20, [0, 0, 0, 0].repeat(400)).unwrap();
        let graph = json!({"operations":[{"type":"paint.brush","params":{"color":[200,50,20],"opacity":50,"strokes":[{"x":0.5,"y":0.5,"radius":0.2,"strength":100}]}}]});
        let result = apply_graph_json(&source, &graph.to_string()).unwrap();
        let center = (10 * 20 + 10) * 4;
        assert_eq!(&result.pixels[center..center + 4], &[200, 50, 20, 128]);
    }
}
