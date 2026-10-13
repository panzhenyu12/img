use crate::{
    apply_graph_json, composite_model_inpaint, prepare_model_inpaint, GraphOperation, RgbaImage,
};
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Browser contract: render_rgba(width, height, Uint8Array, JSON.stringify(graph))
/// returns { width, height, pixels: Uint8Array }. JS and Go execute the exact
/// same apply_graph implementation; errors throw before the source is changed.
#[wasm_bindgen]
pub fn render_rgba(
    width: u32,
    height: u32,
    pixels: &[u8],
    graph_json: &str,
) -> Result<JsValue, JsValue> {
    let source = RgbaImage::new(width, height, pixels.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result =
        apply_graph_json(&source, graph_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = Object::new();
    Reflect::set(
        &output,
        &JsValue::from_str("width"),
        &JsValue::from_f64(result.width as f64),
    )?;
    Reflect::set(
        &output,
        &JsValue::from_str("height"),
        &JsValue::from_f64(result.height as f64),
    )?;
    Reflect::set(
        &output,
        &JsValue::from_str("pixels"),
        &Uint8Array::from(result.pixels.as_slice()),
    )?;
    Ok(output.into())
}

#[wasm_bindgen]
pub fn engine_version() -> String {
    format!("image-core-{}", env!("CARGO_PKG_VERSION"))
}

#[wasm_bindgen]
pub fn inpaint_masks(
    width: u32,
    height: u32,
    pixels: &[u8],
    operation_json: &str,
) -> Result<JsValue, JsValue> {
    let source = RgbaImage::new(width, height, pixels.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let op: GraphOperation =
        serde_json::from_str(operation_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let masks =
        prepare_model_inpaint(&source, &op).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = Object::new();
    Reflect::set(
        &output,
        &JsValue::from_str("target"),
        &Uint8Array::from(masks.target.as_slice()),
    )?;
    Reflect::set(
        &output,
        &JsValue::from_str("protection"),
        &Uint8Array::from(masks.protection.as_slice()),
    )?;
    Ok(output.into())
}

#[wasm_bindgen]
pub fn composite_inpaint(
    width: u32,
    height: u32,
    pixels: &[u8],
    candidate: &[u8],
    operation_json: &str,
) -> Result<Uint8Array, JsValue> {
    let source = RgbaImage::new(width, height, pixels.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let candidate = RgbaImage::new(width, height, candidate.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let op: GraphOperation =
        serde_json::from_str(operation_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = composite_model_inpaint(&source, &candidate, &op)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(Uint8Array::from(output.pixels.as_slice()))
}
