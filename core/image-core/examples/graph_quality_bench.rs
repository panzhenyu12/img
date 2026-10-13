//! Whole graph first/hot timings without decode/encode; optional exact output.
use image_core::{apply_graph_json, RgbaImage};
use std::{env, fs, hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let decoded = image::open(&args[1])?.into_rgba8();
    let source = RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw())?;
    let graph = fs::read_to_string(&args[2])?;
    let mut times = Vec::new();
    let mut result = source.clone();
    for _ in 0..6 {
        let start = Instant::now();
        result = black_box(apply_graph_json(&source, &graph)?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    if let Some(path) = args.get(3) {
        image::save_buffer(
            path,
            &result.pixels,
            result.width,
            result.height,
            image::ColorType::Rgba8,
        )?;
    }
    let cold = times.remove(0);
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"width":source.width,"height":source.height,
        "scope":"same-process graph parse, validation, edit and composite; excludes I/O/detection",
        "first_run_ms":cold,"hot_median_ms":times[2],"hot_samples_ms":times})
    );
    Ok(())
}
