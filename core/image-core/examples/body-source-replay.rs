//! Ordinary-rust body material replay. Detection/model preparation is external.
use clap::Parser;
use image_core::{adjust_body_with_source_layers, BodySourceLayers, EditGraph, RgbaImage};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    graph: PathBuf,
    /// Full source-sized little-endian f32 optical coverage in [0,1].
    #[arg(long)]
    alpha: PathBuf,
    /// Frozen paired background from the same decoded source.
    #[arg(long)]
    background: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, allow_hyphen_values = true)]
    amount: f32,
    #[arg(long, default_value = "waist")]
    adjustment: String,
}

fn read(path: &PathBuf) -> Result<RgbaImage, Box<dyn std::error::Error>> {
    let decoded = image::open(path)?.into_rgba8();
    Ok(RgbaImage::new(
        decoded.width(),
        decoded.height(),
        decoded.into_raw(),
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() || args.output.is_symlink() {
        return Err("output already exists".into());
    }
    let source = read(&args.source)?;
    let graph: EditGraph = serde_json::from_slice(&fs::read(&args.graph)?)?;
    if graph.operations.len() != 1
        || !graph.operations[0].enabled
        || graph.operations[0].kind != "body.adjust"
    {
        return Err("expected one enabled body.adjust operation".into());
    }
    if !args.amount.is_finite() || !(-100.0..=100.0).contains(&args.amount) {
        return Err("amount must be finite in [-100,100]".into());
    }
    let bytes = fs::read(&args.alpha)?;
    if bytes.len() != source.width as usize * source.height as usize * 4 {
        return Err("alpha must match source dimensions".into());
    }
    let alpha: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let background = args.background.as_ref().map(read).transpose()?;
    let mut params = graph.operations[0].params.clone();
    params["adjustments"] = serde_json::json!({args.adjustment:args.amount});
    let start = std::time::Instant::now();
    let output = adjust_body_with_source_layers(
        &source,
        &params,
        &BodySourceLayers {
            width: source.width,
            height: source.height,
            optical_alpha: &alpha,
            paired_background: background.as_ref(),
        },
    )?;
    let elapsed = start.elapsed().as_secs_f64();
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
        serde_json::json!({"api":"adjust_body_with_source_layers",
        "algorithmSeconds":elapsed,"identityExact":output==source,
        "output":args.output,"background":"caller supplied fixed pair or source-only estimate"})
    );
    Ok(())
}
