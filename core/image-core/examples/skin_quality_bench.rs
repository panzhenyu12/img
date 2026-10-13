//! Decode outside the measured scope; report graph+edit first and repeated
//! same-process runs separately from CLI process/decode/encode timings.
use image_core::{apply_graph_json, RgbaImage};
use std::{env, fs, hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let decoded = image::open(&args[1])?.into_rgba8();
    let source = RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw())?;
    let graph = fs::read_to_string(&args[2])?;
    let mut times = Vec::new();
    for _ in 0..6 {
        let start = Instant::now();
        black_box(apply_graph_json(&source, &graph)?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let cold = times.remove(0);
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "width":source.width,"height":source.height,
            "scope":"same-process graph parse, validation, edit and final composite; excludes I/O and detection",
            "first_run_ms":cold,"hot_median_ms":times[2],"hot_samples_ms":times
        })
    );
    Ok(())
}
