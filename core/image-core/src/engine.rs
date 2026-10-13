use crate::{color, geometry, mask, ImageError, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphOperation {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "empty_params")]
    pub params: Value,
    #[serde(default)]
    pub control_points: Vec<Value>,
}

fn default_enabled() -> bool {
    true
}
fn empty_params() -> Value {
    json!({})
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EditGraph {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub source_asset_id: String,
    #[serde(default)]
    pub operations: Vec<GraphOperation>,
}

pub fn apply_graph_json(source: &RgbaImage, graph_json: &str) -> Result<RgbaImage, ImageError> {
    let graph: EditGraph = serde_json::from_str(graph_json)
        .map_err(|error| ImageError::InvalidOperation(format!("invalid EditGraph: {error}")))?;
    apply_graph(source, &graph)
}

/// The only pixel execution path used by both the native CLI and WASM binding.
/// Input pixels are immutable; operations execute sequentially in graph order.
pub fn apply_graph(source: &RgbaImage, graph: &EditGraph) -> Result<RgbaImage, ImageError> {
    if graph.operations.len() > 256 {
        return Err(ImageError::InvalidOperation(
            "EditGraph supports at most 256 operations".into(),
        ));
    }
    let mut current = source.clone();
    for op in &graph.operations {
        crate::validation::operation(op)?;
        if !op.enabled {
            continue;
        }
        if !op.params.is_object() {
            return Err(ImageError::InvalidOperation(format!(
                "{} params must be an object",
                op.kind
            )));
        }
        let processed = match op.kind.as_str() {
            "identity" | "core.identity" => current.clone(),
            "crop" => geometry::crop(&current, &op.params)?,
            "rotate" => geometry::rotate(&current, &op.params)?,
            "flip" => geometry::flip(&current, &op.params)?,
            "resize" => geometry::resize(&current, &op.params)?,
            "perspective" => geometry::perspective(&current, &op.params)?,
            "warp.grid" => geometry::grid(&current, &op.control_points, &op.params)?,
            "warp.liquify" => geometry::liquify(&current, &op.params)?,
            "face.adjust" => crate::face::adjust(&current, &op.params)?,
            "feature.adjust" | "clothing.adjust" => {
                geometry::local_adjust_kind(&current, &op.params, &op.kind)?
            }
            "body.adjust" => crate::body::adjust(&current, &op.params)?,
            "skin.retouch" => color::skin(&current, &op.params),
            "repair.clone" => geometry::repair(&current, &op.params, false)?,
            "repair.heal" => geometry::repair(&current, &op.params, true)?,
            "repair.inpaint" => {
                if op.params.get("method").and_then(Value::as_str) == Some("lama") {
                    return Err(ImageError::InvalidOperation("LaMa requires the local model worker; use the Web/Go renderer or method=exemplar".into()));
                }
                crate::inpaint::inpaint(&current, &op.params)?
            }
            "paint.brush" => crate::paint::render(&current, &op.params)?,
            "color.basic" => color::basic(&current, &op.params),
            "color.curves" => color::curves(&current, &op.params)?,
            "color.hsl" => color::hsl(&current, &op.params),
            "filter.film" => color::film(&current, &op.params),
            "filter.lut" => color::lut(&current, &op.params)?,
            other => {
                return Err(ImageError::InvalidOperation(format!(
                    "unsupported operation: {other}"
                )))
            }
        };
        // Global geometry changes dimensions and has no original pixel at the
        // new coordinates to blend against. Pixel effects and local warps share
        // one selection/protection/opacity/blending boundary.
        current = if op.kind == "face.adjust"
            || (op.kind == "body.adjust"
                && op.params.get("subjectMask").is_some_and(Value::is_object))
        {
            // The portrait compositor separates foreground membership from the
            // user support mask and handles ROI/protection/blending itself.
            // Reapplying the original contour would restore the old edge.
            processed
        } else if processed.width == current.width && processed.height == current.height {
            mask::composite(&current, &processed, &op.params)
        } else {
            if op.params.get("mask").is_some()
                || op.params.get("protectMask").is_some()
                || number(&op.params, "opacity", 100.0) != 100.0
                || op
                    .params
                    .get("blendMode")
                    .and_then(Value::as_str)
                    .is_some_and(|m| m != "normal")
            {
                return Err(ImageError::InvalidOperation(
                    "dimension-changing geometry cannot use local masks or blending".into(),
                ));
            }
            processed
        };
    }
    Ok(current)
}

pub(crate) fn number(value: &Value, key: &str, fallback: f32) -> f32 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .map(|n| n as f32)
        .unwrap_or(fallback)
}

pub(crate) fn percent(value: &Value, key: &str) -> f32 {
    number(value, key, 0.0).clamp(-100.0, 100.0) / 100.0
}

pub(crate) fn positive(value: &Value, key: &str) -> f32 {
    percent(value, key).max(0.0)
}

pub(crate) fn byte(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

pub(crate) fn as_native(source: &RgbaImage) -> image::RgbaImage {
    image::RgbaImage::from_raw(source.width, source.height, source.pixels.clone()).unwrap()
}

pub(crate) fn from_native(source: image::RgbaImage) -> RgbaImage {
    RgbaImage {
        width: source.width(),
        height: source.height(),
        pixels: source.into_raw(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_valid_operations_do_not_change_pixels() {
        let source = RgbaImage::new(1, 1, vec![1, 2, 3, 45]).unwrap();
        let output = apply_graph_json(
            &source,
            r#"{"operations":[{"type":"color.basic","enabled":false,"params":{"exposure":50}}] }"#,
        )
        .unwrap();
        assert_eq!(source, output);
    }

    #[test]
    fn rejects_unknown_enabled_operation_instead_of_silently_exporting_source() {
        let source = RgbaImage::new(1, 1, vec![1, 2, 3, 45]).unwrap();
        assert!(apply_graph_json(&source, r#"{"operations":[{"type":"future.ai"}]}"#).is_err());
    }

    #[test]
    fn graph_keeps_original_immutable_and_preserves_alpha() {
        let source = RgbaImage::new(1, 1, vec![60, 100, 130, 45]).unwrap();
        let original = source.clone();
        let output = apply_graph_json(&source, r#"{"operations":[{"type":"color.basic","params":{"exposure":30}},{"type":"filter.film","params":{"preset":"warm","intensity":70}}]}"#).unwrap();
        assert_eq!(source, original);
        assert_ne!(&output.pixels[..3], &source.pixels[..3]);
        assert_eq!(output.pixels[3], 45);
    }

    #[test]
    fn rejects_unknown_parameter_and_wrong_numeric_type() {
        let source = RgbaImage::new(1, 1, vec![1, 2, 3, 255]).unwrap();
        for graph in [
            r#"{"operations":[{"type":"color.basic","params":{"saturaton":40}}]}"#,
            r#"{"operations":[{"type":"color.basic","params":{"exposure":"bright"}}]}"#,
            r#"{"operations":[{"type":"face.adjust","params":{"adjustments":{"unknownNose":10}}}]}"#,
            r#"{"operations":[{"type":"color.basic","enabled":false,"params":{"imaginary":20}}]}"#,
        ] {
            assert!(apply_graph_json(&source, graph).is_err());
        }
    }

    #[test]
    fn selected_inpaint_changes_marked_area_and_preserves_protection() {
        let mut pixels = [20, 190, 60, 77].repeat(20 * 20);
        for y in 6..14 {
            for x in 6..14 {
                pixels[(y * 20 + x) * 4..(y * 20 + x) * 4 + 3].copy_from_slice(&[220, 20, 20]);
            }
        }
        let source = RgbaImage::new(20, 20, pixels).unwrap();
        let graph = serde_json::json!({"operations":[{"type":"repair.inpaint","params":{
            "mask":{"shape":"rect","region":{"x":0.3,"y":0.3,"width":0.4,"height":0.4}},
            "protectMask":{"shape":"rect","region":{"x":0.3,"y":0.3,"width":0.1,"height":0.4}},
            "radius":0.1,"iterations":4}}]});
        let out = apply_graph_json(&source, &graph.to_string()).unwrap();
        assert_ne!(
            &out.pixels[(10 * 20 + 11) * 4..(10 * 20 + 11) * 4 + 3],
            &source.pixels[(10 * 20 + 11) * 4..(10 * 20 + 11) * 4 + 3]
        );
        assert_eq!(
            &out.pixels[(10 * 20 + 6) * 4..(10 * 20 + 6) * 4 + 4],
            &source.pixels[(10 * 20 + 6) * 4..(10 * 20 + 6) * 4 + 4]
        );
        assert_eq!(&out.pixels[..4], &source.pixels[..4]);
        assert!(out.pixels.chunks_exact(4).all(|p| p[3] == 77));
    }

    #[test]
    fn polygon_inpaint_removes_selected_object_without_bleeding_protected_subject() {
        let mut pixels = Vec::new();
        for y in 0..24u8 {
            for x in 0..24u8 {
                let mut rgb = [x.saturating_mul(5), y.saturating_mul(5), 90];
                if (7..17).contains(&x) && (7..17).contains(&y) {
                    rgb = [230, 24, 24];
                }
                pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 211]);
            }
        }
        let source = RgbaImage::new(24, 24, pixels).unwrap();
        let graph = serde_json::json!({"operations":[{"type":"repair.inpaint","params":{
            "mask":{"shape":"polygon","points":[
                {"x":0.29,"y":0.29},{"x":0.71,"y":0.29},
                {"x":0.71,"y":0.71},{"x":0.29,"y":0.71}],"feather":0},
            "protectMask":{"shape":"polygon","points":[
                {"x":0.29,"y":0.29},{"x":0.40,"y":0.29},
                {"x":0.40,"y":0.71},{"x":0.29,"y":0.71}],"feather":0},
            "radius":0.08,"iterations":4}}]});
        let out = apply_graph_json(&source, &graph.to_string()).unwrap();
        let protected = (12 * 24 + 8) * 4;
        let removed = (12 * 24 + 14) * 4;
        assert_eq!(
            &out.pixels[protected..protected + 4],
            &source.pixels[protected..protected + 4]
        );
        assert_ne!(
            &out.pixels[removed..removed + 3],
            &source.pixels[removed..removed + 3]
        );
        assert_eq!(&out.pixels[..4], &source.pixels[..4]);
        assert!(out.pixels.chunks_exact(4).all(|pixel| pixel[3] == 211));
    }

    #[test]
    fn semantic_body_operation_moves_the_contour_and_keeps_the_background() {
        let mut pixels = [20, 20, 80, 255].repeat(32 * 32);
        for y in 0..32 {
            for x in 10..22 {
                let i = (y * 32 + x) * 4;
                pixels[i..i + 4].copy_from_slice(&[200, 100, 80, 255]);
            }
        }
        let source = RgbaImage::new(32, 32, pixels).unwrap();
        let matte: Vec<u8> = (0..32 * 32)
            .map(|i| if (10..22).contains(&(i % 32)) { 255 } else { 0 })
            .collect();
        let graph = serde_json::json!({"operations":[{"type":"body.adjust","params":{
            "region":{"x":0.25,"y":0.15,"width":0.5,"height":0.7},
            "bodyPart":"waist","adjustments":{"waist":70},
            "subjectMask":{"shape":"raster","width":32,"height":32,"data":matte},
            "landmarks":[{"x":0.5,"y":0.15},{"x":0.5,"y":0.5},{"x":0.5,"y":0.85}]
        }}]});
        let output = apply_graph_json(&source, &graph.to_string()).unwrap();
        assert_eq!(&output.pixels[..4 * 4], &source.pixels[..4 * 4]);
        assert_ne!(output.pixels, source.pixels);
        let subject_count =
            |image: &RgbaImage| image.pixels.chunks_exact(4).filter(|p| p[0] > 150).count();
        assert!(
            subject_count(&output) < subject_count(&source),
            "the visible subject must actually get narrower"
        );
        for (i, &coverage) in matte.iter().enumerate() {
            if coverage == 0 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "background changed at {i}"
                );
            }
        }
    }

    #[test]
    fn source_free_heal_removes_a_uniform_background_blemish() {
        let mut pixels = [180, 180, 180, 99].repeat(40 * 40);
        for y in 18..22 {
            for x in 18..22 {
                pixels[(y * 40 + x) * 4..(y * 40 + x) * 4 + 3].copy_from_slice(&[240, 20, 20]);
            }
        }
        let source = RgbaImage::new(40, 40, pixels).unwrap();
        let out=apply_graph_json(&source,r#"{"operations":[{"type":"repair.heal","params":{"strokes":[{"x":0.5,"y":0.5,"radius":0.12,"strength":100}]}}]}"#).unwrap();
        assert_eq!(
            &out.pixels[(20 * 40 + 20) * 4..(20 * 40 + 20) * 4 + 4],
            &[180, 180, 180, 99]
        );
    }
}
