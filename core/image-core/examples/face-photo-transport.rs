use clap::Parser;
use image::{DynamicImage, ImageDecoder};
use image_core::RgbaImage;
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[path = "../src/face_photo_transport.rs"]
mod face_photo_transport;

#[derive(Parser)]
#[command(about = "Offline photo-transport candidate; never enables a production path")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    trace: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Defaults to OUTPUT with extension jacobian.json. Contains every cell.
    #[arg(long)]
    jacobian: Option<PathBuf>,
}

fn floats(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("truncated float32 trace".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output {
        return Err("output must not overwrite the source".into());
    }
    let reader = image::ImageReader::open(&args.input)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut decoded = DynamicImage::from_decoder(decoder)?;
    decoded.apply_orientation(orientation);
    let decoded = decoded.into_rgba8();
    let source = RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw())?;
    let metadata: Value = serde_json::from_slice(&fs::read(args.trace.join("metadata.json"))?)?;
    let number = |key: &str| -> Result<u32, Box<dyn std::error::Error>> {
        Ok(u32::try_from(
            metadata[key]
                .as_u64()
                .ok_or("missing trace crop metadata")?,
        )?)
    };
    let crop = face_photo_transport::Crop {
        left: number("left")?,
        top: number("top")?,
        width: number("width")?,
        height: number("height")?,
    };
    let vector_data = floats(&args.trace.join("inverse.f32"))?;
    if vector_data.len() % 2 != 0 {
        return Err("truncated inverse vector".into());
    }
    let inverse: Vec<[f32; 2]> = vector_data.chunks_exact(2).map(|v| [v[0], v[1]]).collect();
    let alpha = floats(&args.trace.join("color.f32"))?;
    let frozen = image::open(args.trace.join("frozen.png"))?.into_luma8();
    if frozen.dimensions() != (crop.width, crop.height) {
        return Err("frozen crop mismatch".into());
    }
    let frozen: Vec<bool> = frozen.into_raw().iter().map(|&v| v != 0).collect();
    let started = Instant::now();
    let result = face_photo_transport::render(&source, crop, &inverse, &alpha, &frozen)?;
    let elapsed = started.elapsed().as_secs_f64();
    let report_path = args
        .jacobian
        .unwrap_or_else(|| args.output.with_extension("jacobian.json"));
    let summary = json!({
        "candidate": "continuous-photo-transport", "production_enabled": false,
        "input": args.input, "trace": args.trace, "output": args.output,
        "crop": {"left":crop.left,"top":crop.top,"width":crop.width,"height":crop.height},
        "inner_full_weight_distance":3, "fade_width":result.fade_width,
        "max_input_movement":result.max_input_movement,
        "boundary_max_movement":result.boundary_max_movement,
        "boundary_continuous":result.boundary_max_movement == 0.0,
        "clamped_samples":result.clamped_samples,
        "active_pixels":result.inverse.iter().filter(|v| v[0] != 0.0 || v[1] != 0.0).count(),
        "jacobian_min":result.jacobian.min,"jacobian_max":result.jacobian.max,
        "nonpositive_cells":result.jacobian.nonpositive_cells,
        "nonpositive_corners":result.jacobian.nonpositive_corners,
        "cell_count":result.jacobian.cells.len(),"report":report_path,
        "algorithm_seconds":elapsed,
        "limitation":"No diffeomorphism or visual-quality guarantee; background and missed accessories may move."
    });
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    image::RgbaImage::from_raw(result.image.width, result.image.height, result.image.pixels)
        .ok_or("invalid rendered image")?
        .save(&args.output)?;
    let mut writer = BufWriter::new(fs::File::create(&report_path)?);
    writer.write_all(b"{\"summary\":")?;
    serde_json::to_writer(&mut writer, &summary)?;
    writer.write_all(b",\"cell_columns\":[\"top_left\",\"top_right\",\"bottom_left\",\"bottom_right\",\"min\",\"max\",\"nonpositive_corners\"],\"cells\":[")?;
    for (i, cell) in result.jacobian.cells.iter().enumerate() {
        if i > 0 {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(
            &mut writer,
            &(
                cell.corners[0],
                cell.corners[1],
                cell.corners[2],
                cell.corners[3],
                cell.min,
                cell.max,
                cell.nonpositive_corners,
            ),
        )?;
    }
    writer.write_all(b"]}\n")?;
    writer.flush()?;
    println!("{}", summary);
    Ok(())
}
