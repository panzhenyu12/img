//! Replay frozen source evidence through the ordinary image_core rlib API.
//! No research modules, private graph switches, model inference, or resizing.

use clap::Parser;
use image_core::{
    adjust_face_with_source_cache, adjust_face_with_source_layers, EditGraph,
    FaceObservationRegion, FaceSourceCache, FaceSourceLayers, RgbaImage,
};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Map, Value};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    /// Existing source-domain-continuous replay specification.
    spec: PathBuf,
    /// Override the specification's outputDir; the directory must not exist.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Reuse the source background solve across requests for each case.
    /// The first nonzero request populates the cache; all motion is recomputed.
    #[arg(long)]
    cache_source: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Spec {
    output_dir: PathBuf,
    candidate: Option<String>,
    #[serde(default)]
    trace: bool,
    #[serde(default)]
    probe_points: Vec<[f32; 2]>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Case {
    id: String,
    source: PathBuf,
    graph: PathBuf,
    amounts: Vec<f32>,
    #[serde(default = "default_adjustment")]
    adjustment: String,
    #[serde(default)]
    extra_adjustments: Map<String, Value>,
    physical_alpha: PathBuf,
    source_plate_prior: PathBuf,
    source_motion_interface: Option<PathBuf>,
    source_moving_attachment_support: Option<PathBuf>,
}

fn default_adjustment() -> String {
    "faceWidth".into()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AlphaRaster {
    shape: String,
    width: u32,
    height: u32,
    data: Vec<f64>,
    source_region: NormalizedRegion,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NormalizedRegion {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SceneDescriptor {
    source: PathBuf,
    source_sha256: Option<String>,
    image: PathBuf,
    weights: PathBuf,
    coverage_policy: String,
}

struct Prepared {
    source: RgbaImage,
    observed_source: RgbaImage,
    params: Value,
    alpha: Vec<f32>,
    observation: FaceObservationRegion,
    scene: RgbaImage,
    motion_interface: Option<Value>,
    moving_attachment_support: Option<Value>,
    provenance: Value,
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_png(path: &Path) -> Result<RgbaImage, String> {
    let decoded = image::load_from_memory_with_format(
        &fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
        image::ImageFormat::Png,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?
    .into_rgba8();
    RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw()).map_err(|e| e.to_string())
}

fn reject_private(value: &Value, path: &str) -> Result<(), String> {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let location = format!("{path}/{key}");
                if key.starts_with("__") || key.to_ascii_lowercase().starts_with("experimental") {
                    return Err(format!(
                        "private algorithm field is not accepted: {location}"
                    ));
                }
                reject_private(child, &location)?;
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                if child.is_object() || child.is_array() {
                    reject_private(child, &format!("{path}/{i}"))?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn integer_coordinate(value: f64, extent: u32, name: &str) -> Result<u32, String> {
    let pixel = value * extent as f64;
    // This only accommodates JSON normalization round-off, not fractional ROI
    // expansion. No image threshold or source permission is changed.
    if !value.is_finite() || !(0.0..=1.0).contains(&value) || (pixel - pixel.round()).abs() > 1e-7 {
        return Err(format!(
            "sourceRegion.{name} does not describe an integer source coordinate"
        ));
    }
    Ok(pixel.round() as u32)
}

fn prepare(case: &Case, base: &Path) -> Result<Prepared, String> {
    let source_path = resolve(base, &case.source);
    let source = read_png(&source_path)?;
    let graph_path = resolve(base, &case.graph);
    let graph_json: Value = read_json(&graph_path)?;
    reject_private(&graph_json, "graph")?;
    if graph_json.get("source") != Some(&json!({"width":source.width,"height":source.height})) {
        return Err("graph source dimensions differ from the source PNG".into());
    }
    let graph: EditGraph = serde_json::from_value(graph_json).map_err(|e| e.to_string())?;
    let mut enabled = graph.operations.into_iter().filter(|op| op.enabled);
    let op = enabled
        .next()
        .ok_or("graph has no enabled face operation")?;
    if op.kind != "face.adjust" || enabled.next().is_some() || !op.params.is_object() {
        return Err("graph must have exactly one enabled face.adjust with object params".into());
    }
    let alpha_path = resolve(base, &case.physical_alpha);
    let raster: AlphaRaster = read_json(&alpha_path)?;
    let count = source.width as usize * source.height as usize;
    if raster.shape != "raster"
        || raster.width != source.width
        || raster.height != source.height
        || raster.data.len() != count
        || raster
            .data
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=255.0).contains(v))
    {
        return Err("physicalAlpha must be a full-canvas finite 0..255 raster".into());
    }
    // Preserve PreparedMask's JSON number -> f32 -> /255 conversion exactly.
    let alpha = raster
        .data
        .into_iter()
        .map(|v| v as f32 / 255.0_f32)
        .collect();
    let r = raster.source_region;
    let observation = FaceObservationRegion {
        left: integer_coordinate(r.x, source.width, "x")?,
        top: integer_coordinate(r.y, source.height, "y")?,
        width: integer_coordinate(r.width, source.width, "width")?,
        height: integer_coordinate(r.height, source.height, "height")?,
    };
    if observation.width == 0
        || observation.height == 0
        || observation
            .left
            .checked_add(observation.width)
            .is_none_or(|r| r > source.width)
        || observation
            .top
            .checked_add(observation.height)
            .is_none_or(|b| b > source.height)
    {
        return Err("source observation region is empty or exceeds the source canvas".into());
    }
    let descriptor_path = resolve(base, &case.source_plate_prior);
    let descriptor: SceneDescriptor = read_json(&descriptor_path)?;
    if descriptor.coverage_policy != "scene-lower-bound" {
        return Err("sourcePlatePrior coveragePolicy must be scene-lower-bound".into());
    }
    let descriptor_base = descriptor_path
        .parent()
        .ok_or("scene descriptor has no parent")?;
    let observed_path = resolve(descriptor_base, &descriptor.source);
    let observed_source = read_png(&observed_path)?;
    if observed_source != source {
        return Err("scene descriptor belongs to a different decoded source RGBA image".into());
    }
    let scene_path = resolve(descriptor_base, &descriptor.image);
    let scene = read_png(&scene_path)?;
    let weights_path = resolve(descriptor_base, &descriptor.weights);
    let weights = fs::read(&weights_path).map_err(|e| e.to_string())?;
    if scene.width != source.width || scene.height != source.height || weights.len() != count * 4 {
        return Err("scene and weights must match the full source canvas".into());
    }
    for (i, bytes) in weights.chunks_exact(4).enumerate() {
        let (x, y) = (
            (i % source.width as usize) as u32,
            (i / source.width as usize) as u32,
        );
        let inside = x >= observation.left
            && x < observation.left + observation.width
            && y >= observation.top
            && y < observation.top + observation.height;
        let expected = if inside { 1.0 } else { 0.0 };
        if f32::from_le_bytes(bytes.try_into().unwrap()) != expected {
            return Err(format!(
                "scene weight at ({x},{y}) must be {expected} for the exact observation region"
            ));
        }
    }
    let optional_json = |path: &Option<PathBuf>| {
        path.as_ref()
            .map(|p| read_json(&resolve(base, p)))
            .transpose()
    };
    let motion_interface = optional_json(&case.source_motion_interface)?;
    let moving_attachment_support = optional_json(&case.source_moving_attachment_support)?;
    let provenance = json!({
        "source":source_path,"graph":graph_path,"physicalAlpha":alpha_path,
        "sourcePlatePrior":descriptor_path,"observedSource":observed_path,"scene":scene_path,
        "weights":weights_path,"sourceRgbaBinding":"exact decoded full-canvas equality",
        "upstreamSourceSha256":descriptor.source_sha256,"encodedSha256IndependentlyVerified":false,
        "alphaDecoder":"JSON f64 -> f32 / 255.0f32, no u8 rounding",
        "observation":{"left":observation.left,"top":observation.top,"width":observation.width,"height":observation.height},
        "weightsVerified":"exactly 1 inside observation, 0 outside","coveragePolicy":"scene-lower-bound",
        "sourceMotionInterface":case.source_motion_interface.as_ref().map(|p| resolve(base,p)),
        "sourceMovingAttachmentSupport":case.source_moving_attachment_support.as_ref().map(|p| resolve(base,p)),
    });
    Ok(Prepared {
        source,
        observed_source,
        params: op.params,
        alpha,
        observation,
        scene,
        motion_interface,
        moving_attachment_support,
        provenance,
    })
}

fn run(args: Args) -> Result<(), String> {
    let spec_path = fs::canonicalize(&args.spec).map_err(|e| e.to_string())?;
    let base = spec_path.parent().ok_or("spec has no parent")?;
    let spec: Spec = read_json(&spec_path)?;
    if spec
        .candidate
        .as_deref()
        .is_some_and(|c| c != "source-domain-continuous-v1")
        || spec.trace
        || !spec.probe_points.is_empty()
    {
        return Err("ordinary rlib replay accepts only source-domain-continuous-v1, trace:false, and no probePoints".into());
    }
    let output = args
        .output_dir
        .unwrap_or_else(|| resolve(base, &spec.output_dir));
    let output = if output.is_absolute() {
        output
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(output)
    };
    if fs::symlink_metadata(&output).is_ok() {
        return Err(format!("output already exists: {}", output.display()));
    }
    let mut ids = HashSet::new();
    if spec.cases.is_empty() {
        return Err("replay requires at least one case".into());
    }
    for case in &spec.cases {
        if case.id.is_empty()
            || !case
                .id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
            || !ids.insert(&case.id)
            || case.amounts.is_empty()
        {
            return Err(
                "case IDs must be unique filename-safe strings and amounts must be nonempty".into(),
            );
        }
        if case
            .amounts
            .iter()
            .any(|v| !v.is_finite() || !(-100.0..=100.0).contains(v))
        {
            return Err(format!("{} has an amount outside -100..=100", case.id));
        }
        reject_private(
            &Value::Object(case.extra_adjustments.clone()),
            "extraAdjustments",
        )?;
    }
    // create_dir is atomic against reusing an existing evidence directory.
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(&output).map_err(|e| e.to_string())?;
    let mut report = json!({"api":if args.cache_source {"image_core::adjust_face_with_source_cache"} else {"image_core::adjust_face_with_source_layers"},"cfgTest":false,
        "spec":spec_path,"outputDir":output,"imageCoreVersion":env!("CARGO_PKG_VERSION"),
        "timingScope":"native algorithm call only; excludes input decoding and PNG encoding",
        "sourceCache":args.cache_source,"cachePreparation":"lazy; included in the first nonzero render, not separate model preparation",
        "resized":false,"modelInference":false,"aestheticAcceptanceEstablished":false,"cases":[]});
    let mut failures = 0;
    for case in &spec.cases {
        let prepared = match prepare(case, base) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures += 1;
                report["cases"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"id":case.id,"status":"input-error","error":error}));
                continue;
            }
        };
        let layers = FaceSourceLayers {
            observed_source: &prepared.observed_source,
            optical_alpha: &prepared.alpha,
            observation: prepared.observation,
            scene: &prepared.scene,
            motion_interface: prepared.motion_interface.as_ref(),
            moving_attachment_support: prepared.moving_attachment_support.as_ref(),
        };
        let mut outputs = Vec::new();
        let mut source_cache = FaceSourceCache::default();
        for (index, &amount) in case.amounts.iter().enumerate() {
            let mut params = prepared.params.clone();
            params["adjustments"] = Value::Object(case.extra_adjustments.clone());
            params["adjustments"][&case.adjustment] = json!(amount);
            let start = Instant::now();
            let result = if args.cache_source {
                adjust_face_with_source_cache(&prepared.source, &params, &layers, &mut source_cache)
            } else {
                adjust_face_with_source_layers(&prepared.source, &params, &layers)
            };
            let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
            let mut item = json!({"amount":amount,"adjustments":params["adjustments"],"milliseconds":milliseconds});
            if args.cache_source {
                item["sourceCacheStats"] = json!(source_cache.stats());
            }
            let saved = result.map_err(|e| e.to_string()).and_then(|image| {
                if image.width != prepared.source.width || image.height != prepared.source.height {
                    return Err("algorithm changed source dimensions".into());
                }
                let stem = format!(
                    "{}-{index}-{}-candidate.png",
                    case.id,
                    amount.to_string().replace('-', "minus").replace('.', "p")
                );
                let path = output.join(stem);
                image::save_buffer_with_format(
                    &path,
                    &image.pixels,
                    image.width,
                    image.height,
                    image::ColorType::Rgba8,
                    image::ImageFormat::Png,
                )
                .map_err(|e| e.to_string())?;
                item["output"] = json!(path);
                item["dimensions"] = json!([image.width, image.height]);
                item["changedPixels"] = json!(image
                    .pixels
                    .chunks_exact(4)
                    .zip(prepared.source.pixels.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count());
                Ok(())
            });
            match saved {
                Ok(()) => item["status"] = json!("ok"),
                Err(error) => {
                    failures += 1;
                    item["status"] = json!("error");
                    item["error"] = json!(error);
                }
            }
            outputs.push(item);
        }
        report["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":case.id,"input":prepared.provenance,"outputs":outputs}));
    }
    report["failures"] = json!(failures);
    let manifest = output.join("manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", manifest.display());
    if failures > 0 {
        return Err(format!("{failures} replay failures retained in manifest"));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("face-source-replay: {error}");
        std::process::exit(1);
    }
}
