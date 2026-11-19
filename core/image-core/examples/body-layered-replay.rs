//! Replay prepared floating source layers through the ordinary Rust library.
use clap::{Parser, ValueEnum};
use image_core::{
    adjust_body_with_layered_source, adjust_body_with_layered_source_with_reconstruction,
    composite_body_source_layers, composite_body_source_layers_with_reconstruction,
    prepare_body_inverse_displacement, BodyLayeredSource, EditGraph, RgbaImage, SourceReconstruction,
};

#[derive(Clone, Copy, ValueEnum)]
enum ReconstructionArg {
    Exact,
    NearestByte,
}

impl ReconstructionArg {
    fn label(self) -> &'static str {
        match self { Self::Exact => "exact", Self::NearestByte => "nearest-byte" }
    }
}
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    /// Folder with five source-bound, little-endian f64 arrays.
    #[arg(long, required_unless_present = "field_only")]
    prepared: Option<PathBuf>,
    /// Source-sized little-endian f32 interleaved inverse dx/dy. Omit for zero.
    #[arg(long)]
    field: Option<PathBuf>,
    /// One normal body.adjust operation. Computes current geometry directly.
    #[arg(long, conflicts_with = "field")]
    graph: Option<PathBuf>,
    /// Save the ordinary computed field, for inspection without private tracers.
    #[arg(long, requires = "graph")]
    save_field: Option<PathBuf>,
    /// Compute and save geometry only, before reading any prepared layers.
    #[arg(long, requires_all = ["graph", "save_field"],
        conflicts_with_all = ["field", "prepared", "output", "source_reconstruction"])]
    field_only: bool,
    /// Render-only source contract. Default: exact; never inferred from files.
    #[arg(long, value_enum)]
    source_reconstruction: Option<ReconstructionArg>,
    #[arg(long, required_unless_present = "field_only")]
    output: Option<PathBuf>,
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
    if args.output.as_ref().is_some_and(|p| p.exists() || p.is_symlink()) {
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
    let graph = args.graph.as_ref().map(|path| -> Result<EditGraph, Box<dyn std::error::Error>> {
        let graph: EditGraph = serde_json::from_slice(&fs::read(path)?)?;
        if graph.operations.len() != 1
            || graph.operations[0].kind != "body.adjust"
            || !graph.operations[0].enabled
        {
            return Err("graph must contain exactly one enabled body.adjust operation".into());
        }
        Ok(graph)
    }).transpose()?;
    if args.field_only {
        let start = std::time::Instant::now();
        let graph = graph.as_ref().ok_or("field-only requires graph")?;
        let path = args.save_field.as_ref().ok_or("field-only requires save-field")?;
        // Includes normal body.adjust schema validation. This early return
        // must precede every layer read: geometry grants no crop coverage.
        let field = prepare_body_inverse_displacement(&source, &graph.operations[0].params)?;
        fs::write(path, field.iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        println!("{}", serde_json::json!({"api":"prepare_body_inverse_displacement",
            "fieldOnly":true,"algorithmSeconds":start.elapsed().as_secs_f64(),
            "fieldPixels":field.len(),"outputField":path}));
        return Ok(());
    }
    let prepared = args.prepared.as_ref().ok_or("rendering requires prepared layers")?;
    let output_path = args.output.as_ref().ok_or("rendering requires output")?;
    let reconstruction = args.source_reconstruction.unwrap_or(ReconstructionArg::Exact);
    let front_alpha = doubles(&prepared.join("front-alpha.f64"), count)?;
    let front = doubles(&prepared.join("front-rgb.f64"), count * 3)?;
    let rear_alpha = doubles(&prepared.join("rear-alpha.f64"), count)?;
    let rear = doubles(&prepared.join("rear-source-rgb.f64"), count * 3)?;
    let scene = doubles(&prepared.join("scene-rgb.f64"), count * 3)?;
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
    let (output, api) = if let Some(graph) = graph {
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
        match reconstruction {
            ReconstructionArg::Exact => (
                adjust_body_with_layered_source(&source, &graph.operations[0].params, &layers)?,
                "adjust_body_with_layered_source",
            ),
            ReconstructionArg::NearestByte => (
                adjust_body_with_layered_source_with_reconstruction(&source,
                    &graph.operations[0].params, &layers, SourceReconstruction::NearestByte)?,
                "adjust_body_with_layered_source_with_reconstruction",
            ),
        }
    } else {
        match reconstruction {
            ReconstructionArg::Exact => (
                composite_body_source_layers(&source, &layers, &field)?,
                "composite_body_source_layers",
            ),
            ReconstructionArg::NearestByte => (
                composite_body_source_layers_with_reconstruction(&source, &layers,
                    &field, SourceReconstruction::NearestByte)?,
                "composite_body_source_layers_with_reconstruction",
            ),
        }
    };
    let seconds = start.elapsed().as_secs_f64();
    image::save_buffer_with_format(
        output_path,
        &output.pixels,
        output.width,
        output.height,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )?;
    println!(
        "{}",
        serde_json::json!({"api":api, "algorithmSeconds":seconds,
        "sourceReconstruction":reconstruction.label(),
        "sourceIdentity":source==output,"output":output_path,"fieldModified":false})
    );
    Ok(())
}
