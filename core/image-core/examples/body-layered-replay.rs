//! Replay prepared floating source layers through the ordinary Rust library.
use clap::Parser;
use image_core::{
    adjust_body_with_layered_source, composite_body_source_layers,
    prepare_body_inverse_displacement, BodyLayeredSource, EditGraph, RgbaImage,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    /// Folder with five source-bound, little-endian f64 arrays.
    #[arg(long)]
    prepared: PathBuf,
    /// Source-sized little-endian f32 interleaved inverse dx/dy. Omit for zero.
    #[arg(long)]
    field: Option<PathBuf>,
    /// One normal body.adjust operation. Computes current geometry directly.
    #[arg(long, conflicts_with = "field")]
    graph: Option<PathBuf>,
    /// Save the ordinary computed field, for inspection without private tracers.
    #[arg(long, requires = "graph")]
    save_field: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

fn doubles(path: &Path, count: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != count.checked_mul(8).ok_or("invalid source dimensions")? {
        return Err(format!("array length mismatch: {}", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() || args.output.is_symlink() {
        return Err("output already exists".into());
    }
    if args
        .save_field
        .as_ref()
        .is_some_and(|p| p.exists() || p.is_symlink())
    {
        return Err("field output already exists".into());
    }
    let image = image::open(&args.source)?.into_rgba8();
    let source = RgbaImage::new(image.width(), image.height(), image.into_raw())?;
    let count = source.width as usize * source.height as usize;
    let front_alpha = doubles(&args.prepared.join("front-alpha.f64"), count)?;
    let front = doubles(&args.prepared.join("front-rgb.f64"), count * 3)?;
    let rear_alpha = doubles(&args.prepared.join("rear-alpha.f64"), count)?;
    let rear = doubles(&args.prepared.join("rear-source-rgb.f64"), count * 3)?;
    let scene = doubles(&args.prepared.join("scene-rgb.f64"), count * 3)?;
    let field = if let Some(path) = args.field {
        let bytes = fs::read(path)?;
        if bytes.len() != count * 8 {
            return Err("inverse field dimensions mismatch".into());
        }
        bytes
            .chunks_exact(8)
            .map(|b| {
                [
                    f32::from_le_bytes(b[..4].try_into().unwrap()),
                    f32::from_le_bytes(b[4..].try_into().unwrap()),
                ]
            })
            .collect()
    } else {
        vec![[0.0, 0.0]; count]
    };
    let layers = BodyLayeredSource {
        width: source.width,
        height: source.height,
        front_alpha: &front_alpha,
        front_rgb: &front,
        rear_alpha: &rear_alpha,
        rear_source_rgb: &rear,
        scene_rgb: &scene,
    };
    let start = std::time::Instant::now();
    let (output, api) = if let Some(path) = args.graph {
        let graph: EditGraph = serde_json::from_slice(&fs::read(path)?)?;
        if graph.operations.len() != 1
            || graph.operations[0].kind != "body.adjust"
            || !graph.operations[0].enabled
        {
            return Err("graph must contain exactly one enabled body.adjust operation".into());
        }
        if let Some(path) = args.save_field {
            let field = prepare_body_inverse_displacement(&source, &graph.operations[0].params)?;
            fs::write(
                path,
                field
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        (
            adjust_body_with_layered_source(&source, &graph.operations[0].params, &layers)?,
            "adjust_body_with_layered_source",
        )
    } else {
        (
            composite_body_source_layers(&source, &layers, &field)?,
            "composite_body_source_layers",
        )
    };
    let seconds = start.elapsed().as_secs_f64();
    image::save_buffer_with_format(
        &args.output,
        &output.pixels,
        output.width,
        output.height,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )?;
    println!(
        "{}",
        serde_json::json!({"api":api, "algorithmSeconds":seconds,
        "sourceIdentity":source==output,"output":args.output,"fieldModified":false})
    );
    Ok(())
}
