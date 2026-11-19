//! Ordinary native light eyelid closure from one source observation.
//! No Python, prepared layer files, historical fields or test binary are read.
use clap::Parser;
use image::{DynamicImage, ImageDecoder};
use image_core::{prepare_eye_closure, EyeMaterialLabels, RgbaImage};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs, io::Cursor, path::{Path, PathBuf}, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(about = "Prepare one source eye edit and render candidate integer light closing strengths in [-30,0]")]
struct Args {
    #[arg(long)] source: PathBuf,
    #[arg(long)] observations: PathBuf,
    #[arg(long)] materials: Option<PathBuf>,
    #[arg(long)] output_dir: PathBuf,
    /// Required when the full observation contains multiple faces.
    #[arg(long)] face_index: Option<usize>,
    /// Explicitly accept an observation without an adjacent SHA256 binding report.
    /// A present but invalid binding report is always rejected.
    #[arg(long)] allow_unbound_observations: bool,
    /// Save generated diagnostic fields; never reads a field or layer input.
    #[arg(long)] save_fields: bool,
    /// Candidate integer strengths. Default preserves the original three-level replay.
    #[arg(long, value_delimiter=',', allow_hyphen_values=true, default_value="0,-15,-30")]
    amounts: Vec<i32>,
}

fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut bytes=serde_json::to_vec_pretty(value)?;bytes.push(b'\n');
    fs::write(path,bytes)?;Ok(())
}

fn save_png(path: &Path, source: &RgbaImage) -> Result<()> {
    image::save_buffer_with_format(path,&source.pixels,source.width,source.height,
        image::ColorType::Rgba8,image::ImageFormat::Png)?;
    Ok(())
}

fn decode_source(bytes: &[u8]) -> Result<RgbaImage> {
    let reader=image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut decoder=reader.into_decoder()?;
    if decoder.orientation()? != image::metadata::Orientation::NoTransforms {
        return Err("source EXIF orientation requires a transform; normalize the original and regenerate its observations".into());
    }
    let (width,height)=decoder.dimensions();
    if width<=1 || height<=1 || width as u64*height as u64>32_000_000 {
        return Err("source dimensions are outside the native eye experiment limit".into());
    }
    let decoded=DynamicImage::from_decoder(decoder)?.into_rgba8();
    if decoded.pixels().any(|pixel|pixel.0[3]!=255) {
        return Err("eye closure requires an opaque source; transparency is not silently flattened".into());
    }
    Ok(RgbaImage::new(width,height,decoded.into_raw())?)
}

fn normalized_observation(raw:&[u8],width:u32,height:u32,face_index:Option<usize>)
    ->Result<(Vec<[f64;2]>,Value)>
{
    let document:Value=serde_json::from_slice(raw)?;
    if document["dimensions"]["width"].as_u64()!=Some(width as u64)
        || document["dimensions"]["height"].as_u64()!=Some(height as u64) {
        return Err("observation dimensions do not match the decoded source".into());
    }
    let rows=document["rows"].as_array().ok_or("observations require rows")?;
    let full:Vec<_>=rows.iter().filter(|row|row["id"].as_str()==Some("full")).collect();
    let full_box=full.first().and_then(|row|row["box"].as_array());
    let box_matches=full_box.is_some_and(|bounds| bounds.len()==4 && bounds.iter()
        .zip([0.0,0.0,width as f64,height as f64])
        .all(|(coordinate,expected)|coordinate.as_f64()==Some(expected)));
    if full.len()!=1 || !box_matches {
        return Err("exactly one full-image observation with matching box is required".into());
    }
    let faces=full[0]["sourceLandmarks"].as_array().ok_or("full observation has no sourceLandmarks")?;
    if faces.is_empty() || full[0]["faceCount"].as_u64()!=Some(faces.len() as u64) {
        return Err("full observation face count is absent or inconsistent".into());
    }
    let index=match face_index {
        Some(index) if index<faces.len()=>index,
        Some(_)=>return Err("selected face index does not exist".into()),
        None if faces.len()==1=>0,
        None=>return Err("multiple observed faces require an explicit --face-index".into()),
    };
    let points=faces[index].as_array().filter(|points|points.len()>=478)
        .ok_or("complete face and iris observation requires at least 478 landmarks")?;
    let mut normalized=Vec::with_capacity(points.len());
    for point in points {
        let x=point["x"].as_f64().filter(|v|v.is_finite()).ok_or("source landmark x must be finite")?;
        let y=point["y"].as_f64().filter(|v|v.is_finite()).ok_or("source landmark y must be finite")?;
        normalized.push([x,y]);
    }
    Ok((normalized,json!({"faceIndex":index,"observedFaceCount":faces.len(),
        "landmarkCount":points.len(),"coordinateContract":"MediaPipe normalized physical image XY; core adapter subtracts .5 exactly once for native feature points"})))
}

fn verify_observation_binding(path:&Path,source_sha:&str,observation_sha:&str,allow_unbound:bool)->Result<Value> {
    let report_path=path.with_file_name("report.json");
    if !report_path.try_exists()? {
        if !allow_unbound {
            return Err("observation has no adjacent report.json SHA256 binding; supply --allow-unbound-observations explicitly to use it".into());
        }
        return Ok(json!({"available":false,"verified":false,"explicitUnboundPermission":true}));
    }
    let raw=fs::read(&report_path)?;
    let binding:Value=serde_json::from_slice(&raw)?;
    if binding["sourceSHA256"].as_str()!=Some(source_sha)
        || binding["observationsSHA256"].as_str()!=Some(observation_sha) {
        return Err("observation binding report does not match the source and observation SHA256".into());
    }
    Ok(json!({"available":true,"verified":true,"path":fs::canonicalize(report_path)?,"sha256":sha(&raw)}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MaterialSource { sha256:String,width:u32,height:u32,coordinates:String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MaterialDocument {
    version:String,source:MaterialSource,eyes:BTreeMap<String,EyeMaterialLabels>,
    #[serde(default)] note:Option<Value>,
}

fn materials(raw:&[u8],source_sha:&str,width:u32,height:u32)->Result<BTreeMap<String,EyeMaterialLabels>> {
    let mut value:Value=serde_json::from_slice(raw)?;
    // The Python source annotation schema permits explicitly neutral feather
    // and inversion fields. Validate and remove only those optional defaults
    // before the core's strict categorical shape deserializer; never accept
    // opacity, fitted alpha, arbitrary raster or hidden-layer inputs.
    if let Some(eyes)=value.get_mut("eyes").and_then(Value::as_object_mut) {
        for eye in eyes.values_mut() {
            if let Some(labels)=eye.as_object_mut() {
                for regions in labels.values_mut() {
                    if let Some(regions)=regions.as_array_mut() {
                        for region in regions {
                            if let Some(shape)=region.as_object_mut() {
                                if let Some(feather)=shape.remove("feather") {
                                    if feather.as_f64()!=Some(0.0) {
                                        return Err("source material labels do not accept feathered alpha".into());
                                    }
                                }
                                if let Some(invert)=shape.remove("invert") {
                                    if invert.as_bool()!=Some(false) {
                                        return Err("source material labels do not accept inverted roles".into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let document:MaterialDocument=serde_json::from_value(value)?;
    if document.version!="1.0" || document.source.coordinates!="pixel-index" {
        return Err("source materials require version 1.0 and original pixel-index coordinates".into());
    }
    if document.source.sha256!=source_sha || document.source.width!=width || document.source.height!=height {
        return Err("material source SHA256 or dimensions do not match the original".into());
    }
    if document.eyes.is_empty() || document.eyes.keys().any(|name|name!="leftEye"&&name!="rightEye") {
        return Err("materials must explicitly name leftEye or rightEye".into());
    }
    for labels in document.eyes.values() {
        if labels.ocular_anchors.is_empty()&&labels.moving_lash_support.is_empty()&&labels.unknown.is_empty() {
            return Err("source material correction is empty".into());
        }
    }
    let _=document.note;
    Ok(document.eyes)
}

fn main()->Result<()> {
    let args=Args::parse();
    let unique:std::collections::BTreeSet<_>=args.amounts.iter().copied().collect();
    if args.amounts.is_empty() || unique.len()!=args.amounts.len()
        || args.amounts.iter().any(|amount|!(-30..=0).contains(amount)) {
        return Err("--amounts must contain unique integer light-closing strengths in [-30,0]".into());
    }
    match fs::symlink_metadata(&args.output_dir) {
        Ok(_)=>return Err("output directory already exists; choose a new directory".into()),
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>{},
        Err(error)=>return Err(error.into()),
    }
    let source_bytes=fs::read(&args.source)?;let source_sha=sha(&source_bytes);
    let source=decode_source(&source_bytes)?;
    let observation_bytes=fs::read(&args.observations)?;
    let observation_sha=sha(&observation_bytes);
    let binding=verify_observation_binding(&args.observations,&source_sha,&observation_sha,args.allow_unbound_observations)?;
    let (normalized,observation_report)=normalized_observation(&observation_bytes,source.width,source.height,args.face_index)?;
    let material_bytes=args.materials.as_ref().map(fs::read).transpose()?;
    let labels=material_bytes.as_ref().map(|raw|materials(raw,&source_sha,source.width,source.height))
        .transpose()?.unwrap_or_default();
    let preparation_start=Instant::now();
    let prepared=prepare_eye_closure(&source,&normalized,&labels)?;
    let preparation_seconds=preparation_start.elapsed().as_secs_f64();
    // mkdir is exclusive; a concurrent creator cannot cause silent overwrite.
    if let Some(parent)=args.output_dir.parent().filter(|path|!path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&args.output_dir)?;
    let source_output=args.output_dir.join("source.png");save_png(&source_output,&source)?;
    fs::write(args.output_dir.join("observations.json"),&observation_bytes)?;
    if let Some(raw)=&material_bytes { fs::write(args.output_dir.join("materials.json"),raw)?; }
    let mut report=json!({
        "status":"rendering", "api":"prepare_eye_closure / PreparedEyeEdit::render",
        "source":fs::canonicalize(&args.source)?, "sourceSHA256":source_sha,
        "sourceDecodedPngSHA256":sha(&fs::read(&source_output)?),
        "sourceDecoder":"Rust image 0.25 ImageReader/DynamicImage::from_decoder to RGBA8; source and all edits share this exact decode; no Python/native JPEG decoder comparison",
        "dimensions":{"width":source.width,"height":source.height},
        "opaqueSource":true,"exifOrientation":"NoTransforms",
        "observations":fs::canonicalize(&args.observations)?,"observationsSHA256":observation_sha,
        "observationBinding":binding,"observation":observation_report,
        "materialsSHA256":material_bytes.as_ref().map(|raw|sha(raw)),
        "sourcePreparationCalls":1,"sourcePreparationSeconds":preparation_seconds,
        "sourcePreparationCompletedBeforeFields":true,"sourceOnly":true,
        "geometries":prepared.geometries(),"sourceReports":prepared.reports(),
        "requiresHistoricalTestBinary":false,"readsPreparedLayersOrFields":false,
        "savesDiagnosticFields":args.save_fields,
        "publicGenericEyeHeightEnabled":false,"opticalTruth":false,
        "fieldCoordinates":"absolute source pixel-index XY; core landmark adapter applies the only -.5 shift; physical normalized region stays unchanged",
        "strengths":args.amounts,"rows":[],
    });
    let report_path=args.output_dir.join("report.json");write_json(&report_path,&report)?;
    let mut failures=Vec::new();
    for &amount in &args.amounts {
        let name=if amount==0 {"zero".to_owned()} else {format!("minus{}",amount.abs())};
        let start=Instant::now();
        let result=prepared.render_diagnostic(amount);
        let row=match result {
            Ok((output,_)) if amount==0 && output!=source=>{
                failures.push("zero changed source RGBA".to_owned());
                json!({"amount":amount,"status":"rejected","error":"zero changed source RGBA"})
            },
            Ok((output,field))=>{
                let render_seconds=start.elapsed().as_secs_f64();
                let path=args.output_dir.join(format!("candidate-{name}.png"));
                save_png(&path,&output)?;
                let mut row=json!({"amount":amount,"status":"rendered; independent visual review required",
                    "seconds":render_seconds,"sourceRefitted":false,
                    "output":fs::canonicalize(&path)?,"outputSHA256":sha(&fs::read(path)?),
                    "sourceIdentityRGBA":output==source,"integrationSteps":field.integration_steps});
                if args.save_fields {
                    let inverse:Vec<u8>=field.inverse.iter().flatten().flat_map(|v|v.to_le_bytes()).collect();
                    let stationary:Vec<u8>=field.stationary.iter().map(|&v|u8::from(v)).collect();
                    let inverse_path=args.output_dir.join(format!("candidate-{name}.inverse.f32"));
                    let stationary_path=args.output_dir.join(format!("candidate-{name}.stationary.u8"));
                    fs::write(&inverse_path,&inverse)?;fs::write(&stationary_path,&stationary)?;
                    row["field"]=json!(fs::canonicalize(inverse_path)?);
                    row["stationary"]=json!(fs::canonicalize(stationary_path)?);
                    row["fieldSHA256"]=json!(sha(&inverse));
                    row["stationarySHA256"]=json!(sha(&stationary));
                    row["fieldPixels"]=json!(field.inverse.len());
                    row["fieldEncoding"]=json!("little-endian interleaved absolute source pixel-index f32 XY; full row-major u8 stationary mask");
                }
                row
            },
            Err(error)=>{
                failures.push(format!("{amount}: {error}"));
                json!({"amount":amount,"status":"rejected","error":error.to_string(),
                    "seconds":start.elapsed().as_secs_f64(),"sourceRefitted":false})
            },
        };
        report["rows"].as_array_mut().unwrap().push(row);write_json(&report_path,&report)?;
    }
    report["status"]=json!(if failures.is_empty(){"rendered; independent visual review required"}else{"failed"});
    write_json(&report_path,&report)?;
    println!("{}",json!({"report":fs::canonicalize(report_path)?,"status":report["status"],
        "sourcePreparationCalls":1,"requiresHistoricalTestBinary":false}));
    if !failures.is_empty() { return Err(failures.join("; ").into()); }
    Ok(())
}
