#[cfg(not(target_arch = "wasm32"))]
mod native {
    use clap::Parser;
    use image::{codecs::jpeg::JpegEncoder, DynamicImage, ImageDecoder, ImageFormat};
    use image_core::{
        apply_graph_json, composite_model_inpaint, prepare_model_inpaint, EditGraph, RgbaImage,
    };
    use std::{
        fs,
        io::{BufWriter, Write},
        path::PathBuf,
    };

    #[derive(Parser)]
    #[command(
        name = "image-render",
        about = "Render an immutable source through a shared EditGraph"
    )]
    struct Args {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        graph: PathBuf,
        #[arg(long, default_value = "png")]
        format: String,
        #[arg(long, default_value_t = 92)]
        quality: u8,
        /// Export canonical grayscale masks and oriented source into --output directory.
        #[arg(long, conflicts_with = "candidate")]
        inpaint_masks: bool,
        /// Composite a model candidate with the single LaMa operation in --graph.
        #[arg(long)]
        candidate: Option<PathBuf>,
    }

    fn decode(path: &PathBuf) -> Result<RgbaImage, Box<dyn std::error::Error>> {
        let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
        let mut limits = image::Limits::default();
        limits.max_alloc = Some(128_000_000);
        reader.limits(limits);
        let mut decoder = reader.into_decoder()?;
        let orientation = decoder.orientation()?;
        let mut decoded = DynamicImage::from_decoder(decoder)?;
        decoded.apply_orientation(orientation);
        let decoded = decoded.into_rgba8();
        Ok(RgbaImage::new(
            decoded.width(),
            decoded.height(),
            decoded.into_raw(),
        )?)
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse();
        let source = decode(&args.input)?;
        let graph = fs::read_to_string(&args.graph)?;
        let result = if args.inpaint_masks || args.candidate.is_some() {
            let model_graph: EditGraph = serde_json::from_str(&graph)?;
            if model_graph.operations.len() != 1 {
                return Err("model mask/composite mode requires exactly one operation".into());
            }
            let operation = &model_graph.operations[0];
            if args.inpaint_masks {
                let masks = prepare_model_inpaint(&source, operation)?;
                fs::create_dir_all(&args.output)?;
                let target = image::GrayImage::from_raw(masks.width, masks.height, masks.target)
                    .ok_or("invalid target mask")?;
                let protection =
                    image::GrayImage::from_raw(masks.width, masks.height, masks.protection)
                        .ok_or("invalid protection mask")?;
                target.save(args.output.join("target.png"))?;
                protection.save(args.output.join("protection.png"))?;
                image::RgbaImage::from_raw(source.width, source.height, source.pixels.clone())
                    .ok_or("invalid oriented source")?
                    .save(args.output.join("source.png"))?;
                println!(
                    "{}",
                    serde_json::json!({
                        "width":source.width,"height":source.height,"format":"masks",
                        "selectedPixels":target.pixels().filter(|p|p[0]>0).count(),
                        "output":args.output
                    })
                );
                return Ok(());
            }
            let candidate = decode(args.candidate.as_ref().ok_or("candidate is required")?)?;
            composite_model_inpaint(&source, &candidate, operation)?
        } else {
            apply_graph_json(&source, &graph)?
        };
        let width = result.width;
        let height = result.height;
        let rendered = image::RgbaImage::from_raw(width, height, result.pixels)
            .ok_or("invalid rendered image")?;
        let file = fs::File::create(&args.output)?;
        let mut writer = BufWriter::new(file);
        let format = match args.format.as_str() {
            "jpeg" | "jpg" => {
                // JPEG has no alpha. Flatten on white rather than turning
                // transparent rotated image corners black.
                let rgb = image::RgbImage::from_fn(width, height, |x, y| {
                    let p = rendered.get_pixel(x, y);
                    let alpha = p[3] as f32 / 255.0;
                    image::Rgb(std::array::from_fn(|c| {
                        (p[c] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8
                    }))
                });
                JpegEncoder::new_with_quality(&mut writer, args.quality.clamp(1, 100))
                    .encode_image(&rgb)?;
                "jpeg"
            }
            "png" => {
                DynamicImage::ImageRgba8(rendered).write_to(&mut writer, ImageFormat::Png)?;
                "png"
            }
            _ => return Err("format must be png or jpeg".into()),
        };
        writer.flush()?;
        println!(
            "{}",
            serde_json::json!({"width":width,"height":height,"format":format,"output":args.output})
        );
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = native::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
