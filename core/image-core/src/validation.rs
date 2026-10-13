use crate::{GraphOperation, ImageError};
use serde_json::Value;

const COMMON: &[&str] = &["region", "mask", "protectMask", "opacity", "blendMode"];
const SEMANTIC_MASK_NAMES: &[&str] = &[
    "face",
    "faceOutline",
    "faceContour",
    "faceContourMask",
    "skin",
    "neck",
    "accessories",
    "ears",
    "skinProtection",
    "eyes",
    "leftEye",
    "rightEye",
    "brows",
    "leftBrow",
    "rightBrow",
    "underEyes",
    "leftUnderEye",
    "rightUnderEye",
    "lips",
    "lip",
    "teeth",
    "hair",
    "nose",
    "noseBridge",
    "noseWings",
    "noseTip",
    "noseRoot",
    "noseBase",
    "leftNostril",
    "rightNostril",
];
const FACE: &[&str] = &[
    "forehead",
    "temples",
    "cheekbones",
    "jaw",
    "chin",
    "chinWidth",
    "faceWidth",
    "faceHeight",
    "midface",
    "doubleChin",
    "symmetry",
    "foreheadWidth",
    "chinPoint",
];
const FEATURE: &[&str] = &[
    "eyeSize",
    "eyeWidth",
    "eyeHeight",
    "eyeSpacing",
    "eyeTilt",
    "eyeCorner",
    "eyelid",
    "browLength",
    "browWidth",
    "browThickness",
    "browPeak",
    "browTail",
    "browColor",
    "noseBridge",
    "noseWings",
    "noseLength",
    "noseTip",
    "mouthWidth",
    "mouthHeight",
    "upperLip",
    "lowerLip",
    "lipColor",
    "philtrum",
    "teethWhiten",
    "eyeInnerCorner",
    "eyeOuterCorner",
    "eyeBag",
    "darkCircle",
    "tearTrough",
    "browSpacing",
    "noseHeight",
    "noseStraight",
    "nostril",
    "noseRoot",
    "noseBase",
    "noseLipAngle",
    "lipPeak",
    "lipCorner",
    "lipBrightness",
    "lipSaturation",
    "symmetry",
];
const BODY: &[&str] = &[
    "neck",
    "shoulder",
    "waist",
    "belly",
    "arms",
    "legs",
    "hips",
    "height",
    "legLength",
    "neckLength",
    "trapezius",
    "shoulderLevel",
    "collarbone",
    "forearms",
    "thighs",
    "calves",
    "ankles",
    "posture",
    "headRatio",
];
const CLOTHING: &[&str] = &[
    "smooth",
    "texture",
    "hue",
    "saturation",
    "fit",
    "edges",
    "sleeves",
    "skirt",
    "brightness",
    "collar",
    "cuffs",
    "sleeveLength",
];

fn error(message: String) -> ImageError {
    ImageError::InvalidOperation(message)
}
fn object<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>, ImageError> {
    value
        .as_object()
        .ok_or_else(|| error(format!("{label} must be an object")))
}
fn keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), ImageError> {
    for key in object(value, label)?.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(error(format!("unknown {label} parameter: {key}")));
        }
    }
    Ok(())
}
fn number(value: &Value, key: &str, min: f64, max: f64, required: bool) -> Result<(), ImageError> {
    if let Some(value) = value.get(key) {
        if !value
            .as_f64()
            .is_some_and(|v| v.is_finite() && v >= min && v <= max)
        {
            return Err(error(format!("{key} must be a number in {min}..{max}")));
        }
    } else if required {
        return Err(error(format!("missing numeric {key}")));
    }
    Ok(())
}
fn region(value: &Value) -> Result<(), ImageError> {
    keys(value, &["x", "y", "width", "height"], "region")?;
    for key in ["x", "y"] {
        number(value, key, 0.0, 1.0, true)?;
    }
    for key in ["width", "height"] {
        number(value, key, 0.00001, 1.0, true)?;
    }
    Ok(())
}
fn point(value: &Value) -> Result<(), ImageError> {
    keys(value, &["x", "y"], "point")?;
    number(value, "x", -1.0, 2.0, true)?;
    number(value, "y", -1.0, 2.0, true)
}
fn landmark(value: &Value) -> Result<(), ImageError> {
    keys(value, &["x", "y", "toX", "toY", "weight"], "landmark")?;
    number(value, "x", 0.0, 1.0, true)?;
    number(value, "y", 0.0, 1.0, true)?;
    number(value, "toX", 0.0, 1.0, false)?;
    number(value, "toY", 0.0, 1.0, false)?;
    if value.get("toX").is_some() != value.get("toY").is_some() {
        return Err(error(
            "landmark toX and toY must be provided together".into(),
        ));
    }
    number(value, "weight", 0.01, 10.0, false)
}
fn array(value: &Value, key: &str, max: usize) -> Result<(), ImageError> {
    if let Some(value) = value.get(key) {
        if value.as_array().is_none_or(|v| v.len() > max) {
            return Err(error(format!(
                "{key} must be an array with at most {max} entries"
            )));
        }
    }
    Ok(())
}
fn boolean(value: &Value, key: &str) -> Result<(), ImageError> {
    if value.get(key).is_some_and(|v| !v.is_boolean()) {
        return Err(error(format!("{key} must be boolean")));
    }
    Ok(())
}

fn mask(value: &Value) -> Result<(), ImageError> {
    keys(
        value,
        &[
            "shape", "kind", "region", "x", "y", "width", "height", "points", "strokes", "invert",
            "feather", "from", "to", "data",
        ],
        "mask",
    )?;
    let shape = value
        .get("shape")
        .or_else(|| value.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("rect");
    if ![
        "rect", "ellipse", "polygon", "brush", "linear", "radial", "raster",
    ]
    .contains(&shape)
    {
        return Err(error("unsupported mask shape".into()));
    }
    if shape == "raster" {
        for key in ["width", "height"] {
            if !value
                .get(key)
                .and_then(Value::as_u64)
                .is_some_and(|n| (1..=512).contains(&n))
            {
                return Err(error(
                    "raster mask dimensions must be integers in 1..512".into(),
                ));
            }
        }
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| error("raster mask requires byte data".into()))?;
        let length = value["width"].as_u64().unwrap() * value["height"].as_u64().unwrap();
        if data.len() as u64 != length || data.iter().any(|n| n.as_u64().is_none_or(|v| v > 255)) {
            return Err(error(
                "raster data must have width * height integer bytes".into(),
            ));
        }
    } else if value.get("data").is_some() {
        return Err(error("mask data is only supported for raster masks".into()));
    }
    if let Some(region_value) = value.get("region") {
        region(region_value)?;
    }
    for key in ["from", "to"] {
        if let Some(value) = value.get(key) {
            point(value)?;
        }
    }
    number(value, "feather", 0.0, 100.0, false)?;
    boolean(value, "invert")?;
    array(value, "points", 1024)?;
    array(value, "strokes", 4096)?;
    if let Some(points) = value.get("points").and_then(Value::as_array) {
        for p in points {
            point(p)?;
        }
    }
    if shape == "polygon"
        && value
            .get("points")
            .and_then(Value::as_array)
            .is_none_or(|p| p.len() < 3)
    {
        return Err(error("polygon mask requires at least three points".into()));
    }
    if let Some(strokes) = value.get("strokes").and_then(Value::as_array) {
        for s in strokes {
            keys(s, &["x", "y", "radius", "strength", "erase"], "mask stroke")?;
            for key in ["x", "y"] {
                number(s, key, 0.0, 1.0, true)?;
            }
            number(s, "radius", 0.00001, 1.0, true)?;
            number(s, "strength", 0.0, 100.0, false)?;
            boolean(s, "erase")?;
        }
    }
    Ok(())
}

fn semantic_masks(value: &Value) -> Result<(), ImageError> {
    let object = object(value, "semanticMasks")?;
    if object.len() > SEMANTIC_MASK_NAMES.len() {
        return Err(error(
            "semanticMasks contains too many named regions".into(),
        ));
    }
    for (name, mask_value) in object {
        if !SEMANTIC_MASK_NAMES.contains(&name.as_str()) {
            return Err(error(format!("unknown semantic mask: {name}")));
        }
        if let Some(points) = mask_value.as_array() {
            if points.len() < 3 {
                return Err(error(format!(
                    "semantic mask {name} requires at least three points"
                )));
            }
            if points.len() > 256 {
                return Err(error(format!(
                    "semantic mask {name} supports at most 256 points"
                )));
            }
            for point_value in points {
                point(point_value)?;
            }
        } else {
            mask(mask_value)?;
            let shape = mask_value
                .get("shape")
                .or_else(|| mask_value.get("kind"))
                .and_then(Value::as_str)
                .unwrap_or("polygon");
            if shape == "polygon"
                && mask_value
                    .get("points")
                    .and_then(Value::as_array)
                    .is_none_or(|points| points.len() < 3)
            {
                return Err(error(format!(
                    "semantic mask {name} polygon requires at least three points"
                )));
            }
        }
    }
    Ok(())
}

fn landmarks(value: &Value) -> Result<(), ImageError> {
    let points = value
        .as_array()
        .ok_or_else(|| error("landmarks must be an array".into()))?;
    if points.len() > 128 {
        return Err(error("landmarks must contain at most 128 points".into()));
    }
    for point in points {
        landmark(point)?;
    }
    Ok(())
}

pub(crate) fn operation(op: &GraphOperation) -> Result<(), ImageError> {
    let allowed: &[&str] = match op.kind.as_str() {
        "identity" | "core.identity" => &[],
        "crop" => &["x", "y", "width", "height"],
        "rotate" => &["degrees", "expand"],
        "flip" => &["axis"],
        "resize" => &["width", "height"],
        "perspective" => &["corners", "points"],
        "warp.grid" => &["points", "controlPoints"],
        "warp.liquify" => &["strokes"],
        "face.adjust" | "feature.adjust" | "body.adjust" | "clothing.adjust" => &[
            "adjustments",
            "landmarks",
            "bodyPart",
            "semanticMasks",
            "subjectMask",
            "partMask",
        ],
        "skin.retouch" => &[
            "smooth",
            "texture",
            "whiten",
            "warmth",
            "redness",
            "landmarks",
            "semanticMasks",
        ],
        "repair.clone" | "repair.heal" => &["strokes"],
        "repair.inpaint" => &[
            "radius",
            "iterations",
            "method",
            "execution",
            "maxSide",
            "contextPadding",
            "maskDilation",
            "seamBlend",
            "contextMode",
        ],
        "paint.brush" => &["color", "strokes"],
        "color.basic" => &[
            "exposure",
            "brightness",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "temperature",
            "tint",
            "saturation",
            "vibrance",
            "clarity",
            "sharpness",
            "noise",
            "blackWhite",
            "texture",
            "dehaze",
            "blackPoint",
            "whitePoint",
            "gamma",
        ],
        "color.curves" => &["points", "channel"],
        "color.hsl" => &["hsl", "channels", "hue", "saturation", "lightness"],
        "filter.film" => &[
            "preset",
            "intensity",
            "grain",
            "vignette",
            "glow",
            "halation",
            "fade",
            "seed",
        ],
        "filter.lut" => &["size", "data", "domainMin", "domainMax", "amount"],
        _ => return Err(error(format!("unsupported operation: {}", op.kind))),
    };
    for (key, value) in object(&op.params, "operation params")? {
        if !allowed.contains(&key.as_str()) && !COMMON.contains(&key.as_str()) {
            return Err(error(format!("unknown {} parameter: {key}", op.kind)));
        }
        if [
            "exposure",
            "brightness",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "temperature",
            "tint",
            "saturation",
            "vibrance",
            "clarity",
            "warmth",
            "redness",
            "hue",
            "lightness",
            "dehaze",
        ]
        .contains(&key.as_str())
        {
            number(&op.params, key, -100.0, 100.0, false)?;
        }
        if [
            "smooth",
            "whiten",
            "sharpness",
            "noise",
            "intensity",
            "grain",
            "vignette",
            "glow",
            "halation",
            "fade",
            "amount",
            "opacity",
        ]
        .contains(&key.as_str())
        {
            number(&op.params, key, 0.0, 100.0, false)?;
        }
        if key == "texture" {
            number(
                &op.params,
                key,
                if op.kind == "color.basic" {
                    -100.0
                } else {
                    0.0
                },
                100.0,
                false,
            )?;
        }
        match key.as_str() {
            "region" => region(value)?,
            "mask" | "protectMask" => mask(value)?,
            "subjectMask" => {
                if op.kind != "face.adjust"
                    && (op.kind != "body.adjust" || op.params.get("bodyPart").is_none())
                {
                    return Err(error(
                        "subjectMask requires face.adjust or a semantic body.adjust operation"
                            .into(),
                    ));
                }
                mask(value)?;
            }
            "partMask" => {
                if op.kind != "body.adjust" || op.params.get("bodyPart").is_none() {
                    return Err(error(
                        "partMask requires a semantic body.adjust operation".into(),
                    ));
                }
                mask(value)?;
            }
            "method" => {
                if !value
                    .as_str()
                    .is_some_and(|m| ["exemplar", "lama"].contains(&m))
                {
                    return Err(error("inpaint method must be exemplar or lama".into()));
                }
            }
            "semanticMasks" => semantic_masks(value)?,
            "blendMode" => {
                if !value.as_str().is_some_and(|s| {
                    [
                        "normal",
                        "multiply",
                        "screen",
                        "overlay",
                        "softLight",
                        "soft-light",
                        "darken",
                        "lighten",
                    ]
                    .contains(&s)
                }) {
                    return Err(error("unsupported blendMode".into()));
                }
            }
            "bodyPart" => {
                if op.kind != "body.adjust"
                    || !value.as_str().is_some_and(|part| {
                        [
                            "full-body",
                            "torso",
                            "waist",
                            "hips",
                            "left-upper-arm",
                            "right-upper-arm",
                            "left-forearm",
                            "right-forearm",
                            "left-thigh",
                            "right-thigh",
                            "left-calf",
                            "right-calf",
                        ]
                        .contains(&part)
                    })
                {
                    return Err(error(
                        "bodyPart must be a supported body semantic region".into(),
                    ));
                }
            }
            "axis" => {
                if !value
                    .as_str()
                    .is_some_and(|s| ["horizontal", "vertical"].contains(&s))
                {
                    return Err(error("flip axis must be horizontal or vertical".into()));
                }
            }
            "expand" => boolean(&op.params, key)?,
            "preset" => {
                if !value.as_str().is_some_and(|s| {
                    [
                        "neutral",
                        "classic",
                        "warm",
                        "cool",
                        "fade",
                        "vivid",
                        "pastel",
                        "mono",
                        "night",
                        "cinema",
                        "provia",
                        "astia",
                        "velvia",
                        "classicNegative",
                        "eterna",
                        "acros",
                    ]
                    .contains(&s)
                }) {
                    return Err(error("unknown film preset".into()));
                }
            }
            "channel" => {
                if !value.as_str().is_some_and(|s| {
                    ["rgb", "master", "red", "green", "blue", "r", "g", "b"].contains(&s)
                }) {
                    return Err(error("unknown curve channel".into()));
                }
            }
            "blackWhite" => {
                if !value.is_boolean() {
                    number(&op.params, key, 0.0, 100.0, false)?;
                }
            }
            _ => {}
        }
    }
    // Parameters remain readable in older projects, but unfinished semantic
    // algorithms cannot silently execute an unrelated template warp.
    let adjustments = op.params.get("adjustments").unwrap_or(&op.params);
    let pending: &[&str] = match op.kind.as_str() {
        "face.adjust" => &["faceHeight", "midface", "doubleChin"],
        "feature.adjust" => &["eyeBag", "tearTrough", "noseHeight", "noseLipAngle"],
        "body.adjust" => &[
            "neck",
            "height",
            "legLength",
            "neckLength",
            "trapezius",
            "shoulderLevel",
            "collarbone",
            "ankles",
            "posture",
            "headRatio",
        ],
        "clothing.adjust" => &[
            "fit",
            "edges",
            "sleeves",
            "skirt",
            "collar",
            "cuffs",
            "sleeveLength",
        ],
        _ => &[],
    };
    for key in pending {
        if adjustments
            .get(key)
            .and_then(Value::as_f64)
            .is_some_and(|v| v.abs() > 1e-6)
        {
            return Err(error(format!(
                "{}.{} has no validated algorithm yet",
                op.kind, key
            )));
        }
    }
    let active = adjustments.as_object().is_some_and(|values| {
        values.iter().any(|(key, value)| {
            !["texture", "region", "opacity"].contains(&key.as_str())
                && value.as_f64().is_some_and(|v| v.abs() > 1e-6)
        })
    });
    if active
        && op.kind == "body.adjust"
        && (op.params.get("bodyPart").is_none() || op.params.get("subjectMask").is_none())
    {
        return Err(error("body contour edits require bodyPart and a subjectMask; use manual liquify for freeform edits".into()));
    }
    if active && op.kind == "body.adjust" {
        let part = op.params["bodyPart"].as_str().unwrap_or("");
        let applicable: &[&str] = match part {
            "waist" => &["waist", "belly"],
            "hips" => &["hips"],
            "torso" => &["shoulder", "waist", "belly", "hips"],
            "left-upper-arm" | "right-upper-arm" => &["arms"],
            "left-forearm" | "right-forearm" => &["arms", "forearms"],
            "left-thigh" | "right-thigh" => &["legs", "thighs"],
            "left-calf" | "right-calf" => &["legs", "calves"],
            "full-body" => &[
                "shoulder", "waist", "belly", "hips", "arms", "forearms", "legs", "thighs",
                "calves",
            ],
            _ => &[],
        };
        for (key, value) in adjustments.as_object().into_iter().flatten() {
            if BODY.contains(&key.as_str())
                && value.as_f64().is_some_and(|v| v.abs() > 1e-6)
                && !applicable.contains(&key.as_str())
            {
                return Err(error(format!("body.adjust.{key} does not apply to {part}")));
            }
        }
        let required = match part {
            "full-body" => 29,
            "torso" => 4,
            "waist" => 3,
            _ => 2,
        };
        if op
            .params
            .get("landmarks")
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
            < required
        {
            return Err(error(format!(
                "body.adjust.{part} requires {required} anatomical landmarks"
            )));
        }
    }
    if active
        && matches!(
            op.kind.as_str(),
            "feature.adjust" | "skin.retouch" | "clothing.adjust"
        )
    {
        let region = op
            .params
            .get("region")
            .map(|v| crate::mask::Region::read(Some(v)));
        let explicit_roi = region.is_some_and(|r| r.width < 0.95 || r.height < 0.95)
            || op.params.get("mask").is_some();
        if op.params.get("semanticMasks").is_none() && !explicit_roi {
            return Err(error(format!(
                "{} requires detection or an explicit local selection",
                op.kind
            )));
        }
    }
    if op.control_points.len() > 1024 {
        return Err(error("at most 1024 controlPoints are supported".into()));
    }
    for point in &op.control_points {
        control_point(point)?;
    }
    for key in [
        "points",
        "corners",
        "controlPoints",
        "strokes",
        "hsl",
        "channels",
    ] {
        array(&op.params, key, if key == "strokes" { 4096 } else { 1024 })?;
    }
    match op.kind.as_str() {
        "crop" => {
            for key in ["x", "y"] {
                number(&op.params, key, 0.0, 1_000_000.0, true)?;
            }
            for key in ["width", "height"] {
                number(&op.params, key, 1.0, 1_000_000.0, true)?;
            }
        }
        "resize" => {
            for key in ["width", "height"] {
                number(&op.params, key, 1.0, 100_000.0, true)?;
            }
        }
        "rotate" => number(&op.params, "degrees", -3600.0, 3600.0, true)?,
        "color.basic" => {
            number(&op.params, "blackPoint", 0.0, 99.9, false)?;
            number(&op.params, "whitePoint", 0.1, 100.0, false)?;
            number(&op.params, "gamma", 0.1, 5.0, false)?;
            if op
                .params
                .get("blackPoint")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                >= op
                    .params
                    .get("whitePoint")
                    .and_then(Value::as_f64)
                    .unwrap_or(100.0)
            {
                return Err(error("blackPoint must be lower than whitePoint".into()));
            }
        }
        "repair.inpaint" => {
            number(&op.params, "radius", 0.00001, 1.0, false)?;
            number(&op.params, "iterations", 1.0, 64.0, false)?;
            for key in [
                "execution",
                "maxSide",
                "contextPadding",
                "maskDilation",
                "seamBlend",
                "contextMode",
            ] {
                if op.params.get(key).is_some()
                    && op.params.get("method").and_then(Value::as_str) != Some("lama")
                {
                    return Err(error(format!("{key} requires inpaint method=lama")));
                }
            }
            number(&op.params, "maxSide", 256.0, 2048.0, false)?;
            number(&op.params, "contextPadding", 64.0, 1024.0, false)?;
            number(&op.params, "maskDilation", 0.0, 32.0, false)?;
            if let Some(value) = op.params.get("execution") {
                if !value
                    .as_str()
                    .is_some_and(|v| ["client", "server"].contains(&v))
                {
                    return Err(error("execution must be client or server".into()));
                }
            }
            if op.params.get("execution").and_then(Value::as_str) == Some("client") {
                if op
                    .params
                    .get("maxSide")
                    .and_then(Value::as_f64)
                    .unwrap_or(512.0)
                    > 512.0
                {
                    return Err(error("client LaMa supports maxSide up to 512".into()));
                }
                if op
                    .params
                    .get("seamBlend")
                    .and_then(Value::as_str)
                    .unwrap_or("none")
                    != "none"
                {
                    return Err(error(
                        "client LaMa currently supports seamBlend=none".into(),
                    ));
                }
            }
            for key in ["maxSide", "contextPadding", "maskDilation"] {
                if op
                    .params
                    .get(key)
                    .is_some_and(|v| v.as_f64().is_some_and(|n| n.fract() != 0.0))
                {
                    return Err(error(format!("{key} must be an integer")));
                }
            }
            if let Some(value) = op.params.get("contextMode") {
                if !value
                    .as_str()
                    .is_some_and(|v| ["full", "crop"].contains(&v))
                {
                    return Err(error("contextMode must be full or crop".into()));
                }
            }
            if let Some(value) = op.params.get("seamBlend") {
                if !value
                    .as_str()
                    .is_some_and(|v| ["none", "poisson"].contains(&v))
                {
                    return Err(error("seamBlend must be none or poisson".into()));
                }
            }
        }
        "paint.brush" => {
            let color = op
                .params
                .get("color")
                .and_then(Value::as_array)
                .filter(|v| v.len() == 3)
                .ok_or_else(|| error("paint color must contain three RGB channels".into()))?;
            if color
                .iter()
                .any(|v| v.as_f64().is_none_or(|c| !(0.0..=255.0).contains(&c)))
            {
                return Err(error(
                    "paint color channels must be numbers in 0..255".into(),
                ));
            }
            if let Some(strokes) = op.params.get("strokes").and_then(Value::as_array) {
                for s in strokes {
                    keys(
                        s,
                        &["x", "y", "radius", "strength", "erase"],
                        "paint stroke",
                    )?;
                    for key in ["x", "y"] {
                        number(s, key, 0.0, 1.0, true)?;
                    }
                    number(s, "radius", 0.00001, 1.0, true)?;
                    number(s, "strength", 0.0, 100.0, false)?;
                    boolean(s, "erase")?;
                }
            }
        }
        "perspective" => {
            let points = op
                .params
                .get("corners")
                .or_else(|| op.params.get("points"))
                .and_then(Value::as_array)
                .filter(|p| p.len() == 4)
                .ok_or_else(|| error("perspective requires four corners".into()))?;
            for p in points {
                point(p)?;
            }
        }
        "warp.grid" => {
            for key in ["points", "controlPoints"] {
                if let Some(points) = op.params.get(key).and_then(Value::as_array) {
                    for p in points {
                        control_point(p)?;
                    }
                }
            }
        }
        "color.curves" => {
            if let Some(points) = op.params.get("points").and_then(Value::as_array) {
                for p in points {
                    point(p)?;
                }
            }
        }
        "skin.retouch" => {
            if let Some(landmark_values) = op.params.get("landmarks") {
                landmarks(landmark_values)?;
            }
            if let Some(mask_values) = op.params.get("semanticMasks") {
                semantic_masks(mask_values)?;
            }
        }
        "face.adjust" | "feature.adjust" | "body.adjust" | "clothing.adjust" => {
            let allowed = match op.kind.as_str() {
                "face.adjust" => FACE,
                "feature.adjust" => FEATURE,
                "body.adjust" => BODY,
                _ => CLOTHING,
            };
            let values = op
                .params
                .get("adjustments")
                .ok_or_else(|| error("local adjust requires adjustments object".into()))?;
            keys(values, allowed, "adjustments")?;
            for key in object(values, "adjustments")?.keys() {
                number(values, key, -100.0, 100.0, false)?;
            }
            if op.kind == "body.adjust"
                && op.enabled
                && op
                    .params
                    .get("opacity")
                    .and_then(Value::as_f64)
                    .unwrap_or(100.0)
                    > 0.0
                && object(values, "adjustments")?
                    .values()
                    .any(|v| v.as_f64().unwrap_or(0.0) != 0.0)
                && op.params.get("subjectMask").is_none()
            {
                return Err(error("body reshaping requires a subjectMask describing the true subject silhouette; a rectangular ROI cannot isolate the background".into()));
            }
            if let Some(landmark_values) = op.params.get("landmarks") {
                landmarks(landmark_values)?;
            }
        }
        "warp.liquify" | "repair.clone" | "repair.heal" => {
            if let Some(strokes) = op.params.get("strokes").and_then(Value::as_array) {
                for s in strokes {
                    keys(
                        s,
                        if op.kind == "warp.liquify" {
                            &[
                                "x",
                                "y",
                                "toX",
                                "toY",
                                "radius",
                                "strength",
                                "mode",
                                "direction",
                            ]
                        } else {
                            &["x", "y", "sourceX", "sourceY", "radius", "strength"]
                        },
                        "stroke",
                    )?;
                    for key in ["x", "y"] {
                        number(s, key, 0.0, 1.0, true)?;
                    }
                    for key in ["toX", "toY"] {
                        number(s, key, -1.0, 2.0, false)?;
                    }
                    if op.kind == "repair.clone" {
                        for key in ["sourceX", "sourceY"] {
                            number(s, key, 0.0, 1.0, true)?;
                        }
                    } else {
                        for key in ["sourceX", "sourceY"] {
                            number(s, key, 0.0, 1.0, false)?;
                        }
                    }
                    number(s, "radius", 0.00001, 1.0, true)?;
                    number(s, "strength", 0.0, 100.0, false)?;
                    if s.get("mode").is_some_and(|v| {
                        !v.as_str().is_some_and(|s| {
                            ["push", "pinch", "bloat", "twirl", "reconstruct"].contains(&s)
                        })
                    }) {
                        return Err(error("unknown liquify mode".into()));
                    }
                    if s.get("direction")
                        .is_some_and(|v| v.as_f64() != Some(-1.0) && v.as_f64() != Some(1.0))
                    {
                        return Err(error("liquify direction must be -1 or 1".into()));
                    }
                }
            }
        }
        "color.hsl" => {
            for key in ["hsl", "channels"] {
                if let Some(entries) = op.params.get(key).and_then(Value::as_array) {
                    for p in entries {
                        keys(
                            p,
                            &["hue", "range", "shift", "saturation", "lightness"],
                            "hsl channel",
                        )?;
                        number(p, "hue", 0.0, 360.0, true)?;
                        number(p, "range", 1.0, 180.0, false)?;
                        for key in ["shift", "saturation", "lightness"] {
                            number(p, key, -100.0, 100.0, false)?;
                        }
                    }
                }
            }
        }
        "filter.film" => number(&op.params, "seed", 0.0, u32::MAX as f64, false)?,
        "filter.lut" => {
            number(&op.params, "size", 2.0, 65.0, true)?;
            if op.params["size"].as_f64().is_some_and(|n| n.fract() != 0.0) {
                return Err(error("LUT size must be an integer".into()));
            }
        }
        _ => {}
    }
    Ok(())
}

fn control_point(value: &Value) -> Result<(), ImageError> {
    keys(
        value,
        &["x", "y", "inX", "inY", "outX", "outY"],
        "controlPoint",
    )?;
    for key in ["x", "y", "inX", "inY", "outX", "outY"] {
        number(value, key, -1.0, 2.0, false)?;
    }
    if (value.get("inX").is_none() && value.get("x").is_none())
        || (value.get("inY").is_none() && value.get("y").is_none())
    {
        return Err(error("controlPoint requires inX/inY or x/y".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{apply_graph_json, RgbaImage};
    use serde_json::json;
    #[test]
    fn inpaint_execution_options_are_explicit_and_validated_even_when_disabled() {
        let make = |params: Value, enabled| {
            serde_json::from_value::<GraphOperation>(
                json!({"type":"repair.inpaint","enabled":enabled,"params":params}),
            )
            .unwrap()
        };
        for params in [
            json!({"method":"lama","execution":"client","maxSide":512}),
            json!({"method":"lama","execution":"server","maxSide":2048,"seamBlend":"poisson"}),
            json!({"method":"exemplar"}),
        ] {
            assert!(operation(&make(params, true)).is_ok());
        }
        for params in [
            json!({"method":"exemplar","execution":"client"}),
            json!({"method":"lama","execution":"auto"}),
            json!({"method":"lama","execution":"client","maxSide":1024}),
            json!({"method":"lama","execution":"client","seamBlend":"poisson"}),
            json!({"method":"lama","maxSide":511.5}),
        ] {
            assert!(operation(&make(params, false)).is_err());
        }
    }
    #[test]
    fn legacy_pending_parameters_are_rejected_instead_of_a_fake_pixel_effect() {
        for (kind, keys) in [
            ("face.adjust", vec!["faceHeight", "midface", "doubleChin"]),
            (
                "feature.adjust",
                vec!["eyeBag", "tearTrough", "noseHeight", "noseLipAngle"],
            ),
            (
                "body.adjust",
                vec![
                    "neck",
                    "height",
                    "legLength",
                    "neckLength",
                    "trapezius",
                    "shoulderLevel",
                    "collarbone",
                    "ankles",
                    "posture",
                    "headRatio",
                ],
            ),
            (
                "clothing.adjust",
                vec![
                    "fit",
                    "edges",
                    "sleeves",
                    "skirt",
                    "collar",
                    "cuffs",
                    "sleeveLength",
                ],
            ),
        ] {
            for key in keys {
                let graph = json!({"operations":[{"type":kind,"params":{"adjustments":{key:75}}}]});
                let source = RgbaImage::new(1, 1, vec![20, 40, 60, 255]).unwrap();
                let error = apply_graph_json(&source, &graph.to_string()).unwrap_err();
                assert!(
                    error.to_string().contains("no validated algorithm"),
                    "{kind}.{key}: {error}"
                );
            }
        }
    }

    #[test]
    fn landmark_pairs_are_bounded_and_strictly_validated() {
        let valid = GraphOperation {
            id: "face-landmarks".into(),
            kind: "face.adjust".into(),
            enabled: true,
            params: json!({
                "adjustments": {"faceWidth": 60},
                "landmarks": [{"x":0.2,"y":0.5,"toX":0.25,"toY":0.5,"weight":1.5}]
            }),
            control_points: vec![],
        };
        assert!(operation(&valid).is_ok());

        let mut unknown = valid.clone();
        unknown.params["landmarks"][0]["oops"] = json!(1);
        assert!(operation(&unknown).is_err());

        let mut too_many = valid;
        too_many.params["landmarks"] = json!(vec![json!({"x":0.5,"y":0.5}); 129]);
        assert!(operation(&too_many).is_err());
    }

    #[test]
    fn semantic_masks_are_strict_and_bounded() {
        let valid = GraphOperation {
            id: "skin-regions".into(),
            kind: "skin.retouch".into(),
            enabled: true,
            params: json!({
                "smooth": 70,
                "landmarks": [{"x":0.2,"y":0.2},{"x":0.8,"y":0.2},{"x":0.5,"y":0.8}],
                "semanticMasks": {
                    "skin": [{"x":0.2,"y":0.2},{"x":0.8,"y":0.2},{"x":0.8,"y":0.8},{"x":0.2,"y":0.8}],
                    "lips": {"shape":"polygon","points":[{"x":0.4,"y":0.6},{"x":0.6,"y":0.6},{"x":0.5,"y":0.7}]}
                }
            }),
            control_points: vec![],
        };
        assert!(operation(&valid).is_ok());
        let mut unknown = valid.clone();
        unknown.params["semanticMasks"]["beard"] = json!([]);
        assert!(operation(&unknown).is_err());
        let mut too_short = valid;
        too_short.params["semanticMasks"]["lips"] = json!([{"x":0.4,"y":0.6}]);
        assert!(operation(&too_short).is_err());
    }

    #[test]
    fn body_part_is_a_strict_semantic_region_key() {
        let mut op = GraphOperation {
            id: "body-waist".into(),
            kind: "body.adjust".into(),
            enabled: true,
            params: json!({"bodyPart":"waist", "adjustments":{"waist":35}, "landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.5},{"x":0.5,"y":0.8}], "subjectMask":{"shape":"rect","region":{"x":0.3,"y":0.1,"width":0.4,"height":0.8}}}),
            control_points: vec![],
        };
        assert!(operation(&op).is_ok());
        op.params["bodyPart"] = json!("background");
        assert!(operation(&op).is_err());
    }
    #[test]
    fn body_parameters_cannot_silently_target_an_unrelated_part_or_missing_axis() {
        let mut op = GraphOperation {
            id: "body".into(),
            kind: "body.adjust".into(),
            enabled: true,
            params: json!({"bodyPart":"left-upper-arm","subjectMask":{"shape":"rect"},"landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.8}],"adjustments":{"arms":40}}),
            control_points: vec![],
        };
        assert!(operation(&op).is_ok());
        op.params["adjustments"] = json!({"waist":40});
        assert!(operation(&op)
            .unwrap_err()
            .to_string()
            .contains("does not apply"));
        op.params["adjustments"] = json!({"arms":40});
        op.params["landmarks"] = json!([]);
        assert!(operation(&op)
            .unwrap_err()
            .to_string()
            .contains("anatomical landmarks"));
    }

    #[test]
    fn rejects_invalid_raster_alpha_and_subject_mask_on_other_tools() {
        let mut op = GraphOperation {
            id: "body".into(),
            kind: "body.adjust".into(),
            enabled: true,
            params: json!({"bodyPart":"waist", "adjustments":{}, "subjectMask":{"shape":"raster","width":2,"height":1,"data":[255,0]}}),
            control_points: vec![],
        };
        assert!(operation(&op).is_ok());
        for mask in [
            json!({"shape":"raster","width":2,"height":1,"data":[255]}),
            json!({"shape":"raster","width":1,"height":1,"data":[0.5]}),
            json!({"shape":"raster","width":513,"height":1,"data":[]}),
        ] {
            op.params["subjectMask"] = mask;
            assert!(operation(&op).is_err());
        }
        op.kind = "feature.adjust".into();
        op.params = json!({"subjectMask":{"shape":"raster","width":1,"height":1,"data":[255]}});
        assert!(operation(&op).is_err());
    }
}
