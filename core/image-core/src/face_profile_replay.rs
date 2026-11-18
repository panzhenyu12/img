//! Opt-in, native test replay of cached PNGs and operation graphs.
//! This is execution/permission evidence, not an aesthetic acceptance test.

use super::*;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReplaySpec {
    output_dir: PathBuf,
    #[serde(default = "default_candidate")]
    candidate: String,
    #[serde(default)]
    probe_points: Vec<[f32; 2]>,
    #[serde(default = "default_trace")]
    trace: bool,
    cases: Vec<ReplayCase>,
}

fn default_candidate() -> String {
    "segmented-lateral-v1".into()
}
fn default_trace() -> bool {
    true
}
fn default_adjustment() -> String {
    "faceWidth".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayCase {
    id: String,
    source: PathBuf,
    graph: PathBuf,
    amounts: Vec<f32>,
    #[serde(default = "default_adjustment")]
    adjustment: String,
    #[serde(default, rename = "extraAdjustments")]
    extra_adjustments: serde_json::Map<String, Value>,
    /// Explicitly reviewed depth annotation; never inferred from hair class.
    #[serde(default, rename = "rearHairMask")]
    rear_hair_mask: Option<PathBuf>,
    #[serde(default, rename = "physicalAlpha")]
    physical_alpha: Option<PathBuf>,
    /// Source-only, explicitly reviewed scene prior. Experimental native replay
    /// only; not a public edit-graph parameter or automatic ownership decision.
    #[serde(default, rename = "sourcePlatePrior")]
    source_plate_prior: Option<PathBuf>,
    /// Source-only admission geometry; injected into the research candidate
    /// after validating the unchanged public graph, never into its baseline.
    #[serde(default, rename = "sourceMotionInterface")]
    source_motion_interface: Option<PathBuf>,
    /// Source-only attachment permission shared with the independent field
    /// and source cache. Never changes physical alpha or the public baseline.
    #[serde(default, rename = "sourceMovingAttachmentSupport")]
    source_moving_attachment_support: Option<PathBuf>,
    #[serde(default, rename = "sharedSubjectAdmission")]
    shared_subject_admission: bool,
    #[serde(default, rename = "outerSubjectOccupancy")]
    outer_subject_occupancy: bool,
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

fn load_case(case: &ReplayCase, base: &Path) -> Result<(RgbaImage, crate::GraphOperation), String> {
    let source_path = resolve(base, &case.source);
    // These are already decoded/oriented caches. Never decode the original
    // JPEG again, reorient, resize, or copy a source image into the output.
    let decoded = image::load_from_memory_with_format(
        &fs::read(&source_path).map_err(|e| format!("read source: {e}"))?,
        image::ImageFormat::Png,
    )
    .map_err(|e| format!("decode cached PNG: {e}"))?
    .into_rgba8();
    let source = RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw())
        .map_err(|e| e.to_string())?;
    let graph: crate::EditGraph = serde_json::from_slice(
        &fs::read(resolve(base, &case.graph)).map_err(|e| format!("read graph: {e}"))?,
    )
    .map_err(|e| format!("decode graph: {e}"))?;
    let mut enabled = graph.operations.into_iter().filter(|op| op.enabled);
    let operation = enabled.next().ok_or("graph has no enabled operation")?;
    if operation.kind != "face.adjust" || enabled.next().is_some() {
        return Err("graph must contain exactly one enabled operation, of type face.adjust".into());
    }
    if !operation.params.is_object() || operation.params.get("experimentalFaceProfile").is_some() {
        return Err("baseline params must be an object without experimentalFaceProfile".into());
    }
    crate::validation::operation(&operation).map_err(|e| format!("invalid cached graph: {e}"))?;
    Ok((source, operation))
}

fn rgba_at(image: &RgbaImage, x: i32, y: i32) -> Option<[u8; 4]> {
    if x < 0 || y < 0 || x >= image.width as i32 || y >= image.height as i32 {
        return None;
    }
    let at = (y as usize * image.width as usize + x as usize) * 4;
    Some(image.pixels[at..at + 4].try_into().unwrap())
}

/// Fixed coordinate samples are diagnostics, not feature displacement estimates.
fn feature_diagnostic(source: &RgbaImage, output: &RgbaImage, p: Point) -> Value {
    let (cx, cy) = (p.0.floor() as i32, p.1.floor() as i32);
    json!({"sourceCenterPx":p,"sampledTexel":[cx,cy],
        "beforeRgba":rgba_at(source,cx,cy),"afterRgba":rgba_at(output,cx,cy)})
}

fn polygon_mask(value: Option<&Value>) -> Option<&Value> {
    value.filter(|m| m.get("shape").and_then(Value::as_str) == Some("polygon"))
}

fn polygon_area(polygon: &[Point]) -> f64 {
    polygon
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let b = polygon[(i + 1) % polygon.len()];
            a.0 as f64 * b.1 as f64 - b.0 as f64 * a.1 as f64
        })
        .sum::<f64>()
        .abs()
        * 0.5
}

fn contour_metrics(source: &[Point], target: &[Point]) -> Value {
    if source.len() != target.len() || source.is_empty() {
        return json!({"error":"source/target contour point count mismatch or empty contour"});
    }
    let (total, maximum) =
        source
            .iter()
            .zip(target)
            .fold((0.0_f64, 0.0_f64), |(sum, max), (a, b)| {
                let distance =
                    ((b.0 as f64 - a.0 as f64).powi(2) + (b.1 as f64 - a.1 as f64).powi(2)).sqrt();
                (sum + distance, max.max(distance))
            });
    let source_area = polygon_area(source);
    let target_area = polygon_area(target);
    json!({"points":source.len(),"meanRequestedDisplacementPx":total / source.len() as f64,
        "maxRequestedDisplacementPx":maximum,"sourceAreaPx2":source_area,
        "targetAreaPx2":target_area,"areaDeltaPx2":target_area-source_area,
        "areaChangePercent":(source_area > 0.0).then(||(target_area-source_area)/source_area*100.0)})
}

fn image_metrics(source: &RgbaImage, output: &RgbaImage, params: &Value) -> Result<Value, String> {
    if source.width != output.width
        || source.height != output.height
        || source.pixels.len() != output.pixels.len()
    {
        return Err("output dimensions differ from cached source".into());
    }
    let region = Region::read(params.get("region"));
    let protect = params.get("protectMask").map(PreparedMask::new);
    let (mut changed, mut outside, mut protected, mut protected_changes) = (0, 0, 0, 0);
    let mut maximum = 0_u8;
    for (i, (a, b)) in source
        .pixels
        .chunks_exact(4)
        .zip(output.pixels.chunks_exact(4))
        .enumerate()
    {
        let (x, y) = (
            ((i % source.width as usize) as f32 + 0.5) / source.width as f32,
            ((i / source.width as usize) as f32 + 0.5) / source.height as f32,
        );
        let is_protected = protect
            .as_ref()
            .is_some_and(|mask| mask.coverage(x, y, source.width, source.height) == 1.0);
        protected += usize::from(is_protected);
        if a != b {
            changed += 1;
            outside += usize::from(!region.contains(x, y));
            protected_changes += usize::from(is_protected);
            maximum = maximum.max((0..4).map(|c| a[c].abs_diff(b[c])).max().unwrap());
        }
    }
    let mut features = serde_json::Map::new();
    for key in [
        "leftEye",
        "rightEye",
        "leftBrow",
        "rightBrow",
        "nose",
        "lips",
        "teeth",
    ] {
        let diagnostic = params
            .get("semanticMasks")
            .and_then(|m| m.get(key))
            .and_then(|mask| center(&points(mask, source.width as f32, source.height as f32)))
            .map(|p| feature_diagnostic(source, output, p))
            .unwrap_or_else(|| json!({"unavailable":"no polygon feature evidence"}));
        features.insert(key.to_owned(), diagnostic);
    }
    Ok(
        json!({"changedPixels":changed,"outsideDeclaredRegionChanges":outside,
        "explicitFullyProtectedPixels":protected,"explicitFullyProtectedChanges":protected_changes,
        "maxRgbaDelta":maximum,"featureCenters":features}),
    )
}

fn curve_diagnostic(source: &RgbaImage, params: &Value, mode: &str) -> Result<Value, String> {
    let (w, h) = (source.width as f32, source.height as f32);
    let semantics = params.get("semanticMasks");
    let outline = polygon_mask(semantics.and_then(|m| m.get("faceContour")))
        .or_else(|| {
            polygon_mask(semantics.and_then(|m| m.get("faceOutline").or_else(|| m.get("face"))))
        })
        .or_else(|| polygon_mask(params.get("mask")))
        .ok_or("no polygon contour for curve diagnostic")?;
    let polygon = points(outline, w, h);
    if polygon.len() < 3 {
        return Err("incomplete contour".into());
    }
    let face_is_raster = semantics
        .and_then(|m| m.get("face"))
        .is_some_and(|m| m.get("shape").and_then(Value::as_str) == Some("raster"));
    let oval = if face_is_raster {
        params
            .get("landmarks")
            .map(|p| points(&json!({"points":p}), w, h))
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let frame = Frame::new(
        if oval.len() >= 12 { &oval } else { &polygon },
        semantics,
        w,
        h,
    );
    let values = params.get("adjustments").unwrap_or(params);
    let baseline = target_contour(&polygon, frame, values);
    let candidate = if mode != "segmented-lateral-v1" {
        Ok(baseline.clone())
    } else {
        crate::face_profile::target_contour(
            &polygon,
            (
                frame.origin,
                frame.across,
                frame.down,
                frame.left,
                frame.right,
                frame.top,
                frame.bottom,
            ),
            semantics,
            w,
            h,
            values,
        )
    };
    let candidate_metrics = match &candidate {
        Ok(target) => contour_metrics(&polygon, target),
        Err(error) => json!({"error":error}),
    };
    let (candidate_target, candidate_error) = match candidate {
        Ok(target) => (Some(target), None),
        Err(error) => (None, Some(error)),
    };
    Ok(
        json!({"coordinateConvention":"image pixels; requested contour targets before rendering",
        "sourceContour":polygon,"baselineTarget":baseline,
        "baselineMetrics":contour_metrics(&polygon,&baseline),"candidateMetrics":candidate_metrics,
        "candidateTarget":candidate_target,"candidateTargetError":candidate_error}),
    )
}

fn write_report(path: &Path, report: &Value) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write metrics: {e}"))?;
    fs::rename(temporary, path).map_err(|e| format!("save metrics: {e}"))
}

fn run_variant(source: &RgbaImage, params: &Value, path: &Path) -> Value {
    let started = Instant::now();
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value, String> {
            let mut params = params.clone();
            if params.get("__traceField").and_then(Value::as_bool) == Some(true) {
                params["__faceFieldTrace"] = json!(path.with_extension("trace"));
            }
            let output = adjust(source, &params).map_err(|e| e.to_string())?;
            if let Some(trace) = params.get("__faceFieldTrace").and_then(Value::as_str) {
                let metadata_path = Path::new(trace).join("metadata.json");
                if metadata_path.is_file() {
                    let mut metadata: Value = serde_json::from_slice(
                        &fs::read(&metadata_path).map_err(|e| e.to_string())?,
                    )
                    .map_err(|e| e.to_string())?;
                    metadata["image"] = json!(path);
                    fs::write(metadata_path, serde_json::to_vec(&metadata).unwrap())
                        .map_err(|e| e.to_string())?;
                } else if output != *source {
                    return Err("nonidentity renderer omitted the requested field trace".into());
                }
                // Exact no-op edits return before a field exists. Preserve
                // their identity output instead of treating absent metadata
                // as a rendering error; never synthesize a geometric trace.
            }
            let metrics = image_metrics(source, &output, &params)?;
            image::save_buffer_with_format(
                path,
                &output.pixels,
                output.width,
                output.height,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )
            .map_err(|e| format!("save output: {e}"))?;
            Ok(metrics)
        }));
    match result {
        Ok(Ok(metrics)) => {
            json!({"status":"rendered","output":path,"elapsedMs":started.elapsed().as_millis(),"metrics":metrics})
        }
        Ok(Err(error)) => {
            json!({"status":"error","elapsedMs":started.elapsed().as_millis(),"error":error})
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".into());
            json!({"status":"error","elapsedMs":started.elapsed().as_millis(),"error":format!("render panicked: {message}")})
        }
    }
}

fn execute(spec_path: &Path) -> Result<(), String> {
    let spec: ReplaySpec = serde_json::from_slice(&fs::read(spec_path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid replay spec: {e}"))?;
    if ![
        "segmented-lateral-v1",
        "source-fringe-v1",
        "feature-distance-v1",
        "supported-rbf-v1",
        "positive-field-v1",
        "positive-boundary-v2",
        "hairline-interior-v1",
        "upper-shared-v1",
        "rear-hair-v1",
        "physical-matte-v1",
        "physical-matte-rgb-v2",
        "layered-curve-v1",
        "shared-cheek-v1",
        "shared-cheek-curve-v4",
        "shared-cheek-ear-v1",
        "shared-cheek-ear-periodic-v1",
        "source-domain-v1",
        "source-domain-continuous-v1",
    ]
    .contains(&spec.candidate.as_str())
    {
        return Err("unsupported research candidate".into());
    }
    let base = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let current_baseline = [
        "shared-cheek-v1",
        "shared-cheek-curve-v4",
        "shared-cheek-ear-v1",
        "shared-cheek-ear-periodic-v1",
    ]
    .contains(&spec.candidate.as_str());
    let directory = resolve(base, &spec.output_dir);
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let report_path = directory.join("metrics.json");
    if report_path.exists() {
        return Err("outputDir already contains metrics.json; use a fresh directory".into());
    }
    let mut report = json!({"candidate":spec.candidate,"spec":spec_path,
        "baselineMeaning":if current_baseline {"same current default; candidate enables a shared skin/hair domain; curve-v4 adds source-relative displacement projection; ear-v1 additionally permits connected observed ear material"} else if ["rear-hair-v1","physical-matte-v1","physical-matte-rgb-v2","layered-curve-v1"].contains(&spec.candidate.as_str()) {"current default without experimental source evidence"} else if spec.candidate == "hairline-interior-v1" { "default positive field before interior hairline transport" } else if spec.candidate == "source-fringe-v1" { "historical default before inferred-fringe recovery" } else { "historical RBF field, including inferred-fringe recovery" },
        "aestheticApproval":false,"qualityScore":null,"completed":false,
        "notes":["No source or graph is copied; cached PNGs are not resized or reoriented.",
            "Outside-region counts refer to the original declared region, before any renderer support expansion.",
            "Feature colors are sampled at original coordinates, not measured output feature locations.",
            "Contour displacement/area refer to requested targets before rendering; smaller requests can reduce artifacts without improving the algorithm.",
            "Rendered status and low permission-change counts do not establish naturalness."],"cases":[]});
    let mut failures = 0;
    let mut ids = HashSet::new();
    for case in spec.cases {
        let case_index = report["cases"].as_array().unwrap().len();
        report["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":case.id,
            "source":resolve(base,&case.source),"graph":resolve(base,&case.graph),"extraAdjustments":case.extra_adjustments,"runs":[]}));
        let input = if case.id.is_empty()
            || !case
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            || !ids.insert(case.id.clone())
        {
            Err(
                "case id must be unique and contain only letters, digits, hyphens or underscores"
                    .into(),
            )
        } else if !CONTOUR_KEYS.contains(&case.adjustment.as_str()) {
            Err("unsupported contour adjustment".into())
        } else if case.amounts.is_empty() {
            Err("case amounts must not be empty".into())
        } else {
            load_case(&case, base)
        };
        let (source, operation) = match input {
            Ok(input) => input,
            Err(error) => {
                failures += 1;
                report["cases"][case_index]["error"] = json!(error);
                for amount in &case.amounts {
                    report["cases"][case_index]["runs"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"amount":amount,
                        "baseline":{"status":"error","error":"case preparation failed"},
                        "candidate":{"status":"error","error":"case preparation failed"}}));
                }
                write_report(&report_path, &report)?;
                continue;
            }
        };
        report["cases"][case_index]["dimensions"] = json!([source.width, source.height]);
        for (amount_index, amount) in case.amounts.iter().copied().enumerate() {
            let mut params = operation.params.clone();
            params["adjustments"] = Value::Object(case.extra_adjustments.clone());
            params["adjustments"][&case.adjustment] = json!(amount);
            if !spec.probe_points.is_empty() {
                params["__faceProbePoints"] = json!(spec.probe_points);
            }
            let stem = format!(
                "{}-{}-{}",
                case.id,
                amount_index,
                amount.to_string().replace('-', "minus").replace('.', "p")
            );
            let valid = if !amount.is_finite() || !(-100.0..=100.0).contains(&amount) {
                Err("faceWidth amount must be finite and in -100..=100".to_owned())
            } else {
                let mut request = operation.clone();
                request.params = params.clone();
                request
                    .params
                    .as_object_mut()
                    .unwrap()
                    .remove("__faceProbePoints");
                crate::validation::operation(&request).map_err(|e| e.to_string())
            };
            let (baseline, candidate, contours) = match valid {
                Err(error) => (
                    json!({"status":"error","error":error}),
                    json!({"status":"error","error":error}),
                    json!({"error":"request validation failed"}),
                ),
                Ok(()) => {
                    let curves = curve_diagnostic(&source, &params, &spec.candidate)
                        .unwrap_or_else(|error| json!({"error":error}));
                    let curve_path = directory.join(format!("{stem}-curves.json"));
                    let mut contours = json!({"baseline":curves.get("baselineMetrics"),"candidate":curves.get("candidateMetrics"),
                        "error":curves.get("error"),"output":curve_path});
                    if let Err(error) = serde_json::to_vec_pretty(&curves)
                        .map_err(|e| e.to_string())
                        .and_then(|bytes| fs::write(&curve_path, bytes).map_err(|e| e.to_string()))
                    {
                        contours["writeError"] = json!(error);
                        failures += 1;
                    }
                    if spec.candidate == "source-fringe-v1" {
                        // Historical comparison: current default already includes recovery.
                        params["__legacyInferredFringe"] = json!(true);
                    }
                    if spec.trace
                        && [
                            "feature-distance-v1",
                            "supported-rbf-v1",
                            "positive-field-v1",
                            "positive-boundary-v2",
                            "hairline-interior-v1",
                            "rear-hair-v1",
                            "physical-matte-v1",
                            "physical-matte-rgb-v2",
                            "layered-curve-v1",
                            "shared-cheek-v1",
                            "shared-cheek-curve-v4",
                            "shared-cheek-ear-v1",
                            "shared-cheek-ear-periodic-v1",
                            "source-domain-v1",
                            "source-domain-continuous-v1",
                        ]
                        .contains(&spec.candidate.as_str())
                    {
                        params["__traceField"] = json!(true);
                    }
                    if !current_baseline
                        && ![
                            "hairline-interior-v1",
                            "rear-hair-v1",
                            "physical-matte-v1",
                            "physical-matte-rgb-v2",
                            "layered-curve-v1",
                        ]
                        .contains(&spec.candidate.as_str())
                    {
                        params["__legacyFaceField"] = json!(true);
                    }
                    if !current_baseline {
                        params["__legacyHairline"] = json!(true);
                    }
                    // The default renderer now owns the evidence-bounded
                    // upper shared domain. Keep the historical baseline
                    // genuinely pre-integration when comparing a candidate.
                    if !current_baseline {
                        params["__legacyUpperShared"] = json!(true);
                    }
                    let baseline = run_variant(
                        &source,
                        &params,
                        &directory.join(format!("{stem}-baseline.png")),
                    );
                    params
                        .as_object_mut()
                        .unwrap()
                        .remove("__legacyInferredFringe");
                    if spec.candidate != "source-fringe-v1" {
                        params.as_object_mut().unwrap().remove("__legacyFaceField");
                    }
                    if ["hairline-interior-v1", "source-domain-continuous-v1"]
                        .contains(&spec.candidate.as_str())
                    {
                        params.as_object_mut().unwrap().remove("__legacyHairline");
                    }
                    params
                        .as_object_mut()
                        .unwrap()
                        .remove("__legacyUpperShared");
                    params["experimentalFaceProfile"] = json!(spec.candidate);
                    if let Some(path) = &case.source_motion_interface {
                        params["__sourceMotionInterface"] = serde_json::from_slice(
                            &fs::read(resolve(base, path)).map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    if let Some(path) = &case.source_moving_attachment_support {
                        params["__sourceMovingAttachmentSupport"] = serde_json::from_slice(
                            &fs::read(resolve(base, path)).map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    if case.shared_subject_admission {
                        params["__experimentalSharedSubjectAdmission"] = json!(true);
                    }
                    if case.outer_subject_occupancy {
                        params["__experimentalOuterSubjectOccupancy"] = json!(true);
                    }
                    if ["rear-hair-v1", "layered-curve-v1"].contains(&spec.candidate.as_str()) {
                        let mask_path = case
                            .rear_hair_mask
                            .as_ref()
                            .ok_or("rear-hair-v1 requires a reviewed rearHairMask file")?;
                        params["__rearHairMask"] = serde_json::from_slice(
                            &fs::read(resolve(base, mask_path)).map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    if ["physical-matte-v1", "physical-matte-rgb-v2"]
                        .contains(&spec.candidate.as_str())
                        // An explicitly supplied source observation must not
                        // disappear when selecting another research field.
                        || case.physical_alpha.is_some()
                    {
                        let mask_path = case
                            .physical_alpha
                            .as_ref()
                            .ok_or("physical-matte-v1 requires a source physicalAlpha file")?;
                        params["__physicalAlpha"] = serde_json::from_slice(
                            &fs::read(resolve(base, mask_path)).map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                        params["__refinePhysicalAlpha"] =
                            json!(spec.candidate == "physical-matte-rgb-v2");
                    }
                    if let Some(path) = &case.source_plate_prior {
                        params["__sourcePlatePrior"] = json!(resolve(base, path));
                    }
                    let candidate = run_variant(
                        &source,
                        &params,
                        &directory.join(format!("{stem}-candidate.png")),
                    );
                    (baseline, candidate, contours)
                }
            };
            failures += usize::from(baseline["status"] != "rendered")
                + usize::from(candidate["status"] != "rendered");
            report["cases"][case_index]["runs"].as_array_mut().unwrap().push(json!({"adjustment":case.adjustment,"amount":amount,"baseline":baseline,"candidate":candidate,"requestedContours":contours}));
            write_report(&report_path, &report)?;
        }
    }
    if report["cases"].as_array().unwrap().is_empty() {
        failures += 1;
    }
    report["completed"] = json!(true);
    report["executionFailures"] = json!(failures);
    write_report(&report_path, &report)?;
    if failures == 0 {
        Ok(())
    } else {
        Err(format!(
            "{failures} replay failures; all results retained in {}",
            report_path.display()
        ))
    }
}

#[test]
#[ignore = "manual cached-photo replay; set FACE_PROFILE_REPLAY_SPEC to a local JSON spec"]
fn replay_cached_profile() {
    let path = std::env::var_os("FACE_PROFILE_REPLAY_SPEC")
        .expect("FACE_PROFILE_REPLAY_SPEC is required for the explicitly requested replay");
    if let Err(error) = execute(Path::new(&path)) {
        panic!("{error}");
    }
}
