use crate::engine::{as_native, byte, from_native, number, percent};
use crate::mask::Region;
use crate::{ImageError, RgbaImage};
use serde_json::Value;

fn invalid(message: &str) -> ImageError {
    ImageError::InvalidOperation(message.to_owned())
}
fn empty(width: u32, height: u32) -> Result<RgbaImage, ImageError> {
    if width == 0 || height == 0 || width as u64 * height as u64 > 32_000_000 {
        return Err(ImageError::InvalidDimensions);
    }
    RgbaImage::new(width, height, vec![0; width as usize * height as usize * 4])
}

/// Bilinear inverse sampling with premultiplied alpha prevents dark halos
/// around transparent pixels. `clamp_edges` is used for local deformation;
/// geometry expansion leaves out-of-bounds pixels transparent.
fn sample(image: &RgbaImage, x: f32, y: f32, clamp_edges: bool) -> [u8; 4] {
    let x = if clamp_edges {
        x.clamp(0.0, image.width as f32 - 1.0)
    } else {
        x
    };
    let y = if clamp_edges {
        y.clamp(0.0, image.height as f32 - 1.0)
    } else {
        y
    };
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let mut channels = [0.0; 4];
    for (px, py, weight) in [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ] {
        if px < 0 || py < 0 || px >= image.width as i32 || py >= image.height as i32 {
            continue;
        }
        let index = (py as usize * image.width as usize + px as usize) * 4;
        let alpha = image.pixels[index + 3] as f32 / 255.0;
        for (c, channel) in channels.iter_mut().enumerate().take(3) {
            *channel += image.pixels[index + c] as f32 * alpha * weight;
        }
        channels[3] += alpha * weight;
    }
    if channels[3] <= 1e-6 {
        return [0, 0, 0, 0];
    }
    [
        byte(channels[0] / channels[3]),
        byte(channels[1] / channels[3]),
        byte(channels[2] / channels[3]),
        byte(channels[3] * 255.0),
    ]
}

fn remap<F: Fn(f32, f32) -> (f32, f32)>(
    source: &RgbaImage,
    width: u32,
    height: u32,
    clamp_edges: bool,
    mapping: F,
) -> Result<RgbaImage, ImageError> {
    let mut out = empty(width, height)?;
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = mapping(x as f32, y as f32);
            let index = (y as usize * width as usize + x as usize) * 4;
            out.pixels[index..index + 4].copy_from_slice(&sample(source, sx, sy, clamp_edges));
        }
    }
    Ok(out)
}

pub(crate) fn crop(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let x = number(p, "x", 0.0).round().max(0.0) as u32;
    let y = number(p, "y", 0.0).round().max(0.0) as u32;
    let width = number(p, "width", source.width as f32).round() as u32;
    let height = number(p, "height", source.height as f32).round() as u32;
    if x >= source.width || y >= source.height || width == 0 || height == 0 {
        return Err(invalid("crop is outside the image"));
    }
    let width = width.min(source.width - x);
    let height = height.min(source.height - y);
    Ok(from_native(
        image::imageops::crop_imm(&as_native(source), x, y, width, height).to_image(),
    ))
}

pub(crate) fn rotate(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let angle = number(p, "degrees", 0.0).rem_euclid(360.0);
    if angle.abs() < 1e-6 {
        return Ok(source.clone());
    }
    let expand = p.get("expand").and_then(Value::as_bool).unwrap_or(true);
    let image = as_native(source);
    if expand {
        if (angle - 90.0).abs() < 1e-6 {
            return Ok(from_native(image::imageops::rotate90(&image)));
        }
        if (angle - 180.0).abs() < 1e-6 {
            return Ok(from_native(image::imageops::rotate180(&image)));
        }
        if (angle - 270.0).abs() < 1e-6 {
            return Ok(from_native(image::imageops::rotate270(&image)));
        }
    }
    let radians = angle.to_radians();
    let c = radians.cos();
    let s = radians.sin();
    let (width, height) = if expand {
        (
            (source.width as f32 * c.abs() + source.height as f32 * s.abs()).ceil() as u32,
            (source.width as f32 * s.abs() + source.height as f32 * c.abs()).ceil() as u32,
        )
    } else {
        (source.width, source.height)
    };
    let cx = (source.width as f32 - 1.0) / 2.0;
    let cy = (source.height as f32 - 1.0) / 2.0;
    let ox = (width as f32 - 1.0) / 2.0;
    let oy = (height as f32 - 1.0) / 2.0;
    remap(source, width, height, false, |x, y| {
        let dx = x - ox;
        let dy = y - oy;
        (c * dx + s * dy + cx, -s * dx + c * dy + cy)
    })
}

pub(crate) fn flip(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let image = as_native(source);
    match p
        .get("axis")
        .and_then(Value::as_str)
        .unwrap_or("horizontal")
    {
        "horizontal" => Ok(from_native(image::imageops::flip_horizontal(&image))),
        "vertical" => Ok(from_native(image::imageops::flip_vertical(&image))),
        _ => Err(invalid("flip axis must be horizontal or vertical")),
    }
}

pub(crate) fn resize(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let width = number(p, "width", source.width as f32).round() as u32;
    let height = number(p, "height", source.height as f32).round() as u32;
    if width == 0 || height == 0 || width as u64 * height as u64 > 32_000_000 {
        return Err(ImageError::InvalidDimensions);
    }
    Ok(from_native(image::imageops::resize(
        &as_native(source),
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    )))
}

/// Solve the eight projective coefficients using pivoted Gaussian elimination.
fn homography(from: &[(f32, f32); 4], to: &[(f32, f32); 4]) -> Option<[f32; 8]> {
    let mut rows = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = from[i];
        let (u, v) = to[i];
        rows[i * 2] = [
            x as f64,
            y as f64,
            1.0,
            0.0,
            0.0,
            0.0,
            -(u * x) as f64,
            -(u * y) as f64,
            u as f64,
        ];
        rows[i * 2 + 1] = [
            0.0,
            0.0,
            0.0,
            x as f64,
            y as f64,
            1.0,
            -(v * x) as f64,
            -(v * y) as f64,
            v as f64,
        ];
    }
    for col in 0..8 {
        let pivot = (col..8).max_by(|a, b| rows[*a][col].abs().total_cmp(&rows[*b][col].abs()))?;
        if rows[pivot][col].abs() < 1e-10 {
            return None;
        }
        rows.swap(col, pivot);
        let divisor = rows[col][col];
        for value in rows[col].iter_mut().skip(col) {
            *value /= divisor;
        }
        for r in 0..8 {
            if r != col {
                let factor = rows[r][col];
                let pivot_row = rows[col];
                for (value, pivot_value) in
                    rows[r].iter_mut().skip(col).zip(pivot_row.iter().skip(col))
                {
                    *value -= factor * *pivot_value;
                }
            }
        }
    }
    Some(std::array::from_fn(|i| rows[i][8] as f32))
}

pub(crate) fn perspective(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let points = p
        .get("corners")
        .or_else(|| p.get("points"))
        .and_then(Value::as_array)
        .filter(|points| points.len() == 4)
        .ok_or_else(|| invalid("perspective requires four corners TL, TR, BR, BL"))?;
    let source_corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    if points.iter().any(|point| {
        !point.get("x").is_some_and(Value::is_number)
            || !point.get("y").is_some_and(Value::is_number)
    }) {
        return Err(invalid("perspective corners require numeric x and y"));
    }
    let destination = std::array::from_fn(|i| {
        (
            number(&points[i], "x", source_corners[i].0),
            number(&points[i], "y", source_corners[i].1),
        )
    });
    let coefficients = homography(&destination, &source_corners)
        .ok_or_else(|| invalid("perspective corners are degenerate"))?;
    remap(source, source.width, source.height, false, |x, y| {
        let nx = x / (source.width - 1).max(1) as f32;
        let ny = y / (source.height - 1).max(1) as f32;
        let divisor = coefficients[6] * nx + coefficients[7] * ny + 1.0;
        if divisor.abs() < 1e-6 {
            return (-1000.0, -1000.0);
        }
        (
            (coefficients[0] * nx + coefficients[1] * ny + coefficients[2]) / divisor
                * (source.width - 1) as f32,
            (coefficients[3] * nx + coefficients[4] * ny + coefficients[5]) / divisor
                * (source.height - 1) as f32,
        )
    })
}

pub(crate) fn grid(
    source: &RgbaImage,
    control_points: &[Value],
    p: &Value,
) -> Result<RgbaImage, ImageError> {
    let points = p
        .get("points")
        .or_else(|| p.get("controlPoints"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(control_points);
    if points.is_empty() {
        return Ok(source.clone());
    }
    // Inverse distance interpolation includes every grid anchor. Unmoved
    // anchors therefore hold the image boundary while moved anchors displace
    // only their surrounding cells.
    remap(source, source.width, source.height, true, |x, y| {
        let nx = x / (source.width - 1).max(1) as f32;
        let ny = y / (source.height - 1).max(1) as f32;
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut sum = 0.0;
        for point in points {
            let ix = number(point, "inX", number(point, "x", 0.5));
            let iy = number(point, "inY", number(point, "y", 0.5));
            let ox = number(point, "outX", ix);
            let oy = number(point, "outY", iy);
            let distance = ((nx - ox) * source.width as f32).powi(2)
                + ((ny - oy) * source.height as f32).powi(2);
            if distance < 1e-12 {
                return (
                    ix * (source.width - 1) as f32,
                    iy * (source.height - 1) as f32,
                );
            }
            let weight = 1.0 / (distance * distance).max(1e-8);
            dx += (ix - ox) * weight;
            dy += (iy - oy) * weight;
            sum += weight;
        }
        (
            (nx + dx / sum) * (source.width - 1) as f32,
            (ny + dy / sum) * (source.height - 1) as f32,
        )
    })
}

pub(crate) fn liquify(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    let Some(strokes) = p.get("strokes").and_then(Value::as_array) else {
        return Ok(source.clone());
    };
    let mut result = source.clone();
    for stroke in strokes {
        let cx = number(stroke, "x", 0.5) * source.width as f32;
        let cy = number(stroke, "y", 0.5) * source.height as f32;
        let radius = number(stroke, "radius", 0.08).clamp(0.001, 1.0)
            * source.width.min(source.height) as f32;
        let strength = number(stroke, "strength", 100.0).clamp(0.0, 100.0) / 100.0;
        let tx = number(stroke, "toX", number(stroke, "x", 0.5)) * source.width as f32;
        let ty = number(stroke, "toY", number(stroke, "y", 0.5)) * source.height as f32;
        let mode = stroke.get("mode").and_then(Value::as_str).unwrap_or("push");
        let direction = number(stroke, "direction", 1.0);
        if mode == "reconstruct" {
            let xmin = (cx - radius).floor().max(0.0) as u32;
            let xmax = (cx + radius).ceil().min(source.width as f32) as u32;
            let ymin = (cy - radius).floor().max(0.0) as u32;
            let ymax = (cy + radius).ceil().min(source.height as f32) as u32;
            for y in ymin..ymax {
                for x in xmin..xmax {
                    let distance =
                        ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / radius;
                    if distance >= 1.0 {
                        continue;
                    }
                    let weight = (1.0 - distance * distance).powi(2) * strength;
                    let index = (y as usize * source.width as usize + x as usize) * 4;
                    let a = result.pixels[index + 3] as f32 / 255.0;
                    let b = source.pixels[index + 3] as f32 / 255.0;
                    let alpha = a * (1.0 - weight) + b * weight;
                    for channel in 0..3 {
                        result.pixels[index + channel] = if alpha > 1e-8 {
                            byte(
                                (result.pixels[index + channel] as f32 * a * (1.0 - weight)
                                    + source.pixels[index + channel] as f32 * b * weight)
                                    / alpha,
                            )
                        } else {
                            source.pixels[index + channel]
                        };
                    }
                    result.pixels[index + 3] = byte(alpha * 255.0);
                }
            }
            continue;
        }
        result = remap(&result, source.width, source.height, true, |x, y| {
            let dx = x - cx;
            let dy = y - cy;
            let d = (dx * dx + dy * dy).sqrt() / radius;
            if d >= 1.0 {
                return (x, y);
            }
            let nx = (x + 0.5) / source.width as f32;
            let ny = (y + 0.5) / source.height as f32;
            let region = Region::read(p.get("region"));
            if !region.contains(nx, ny) {
                return (x, y);
            }
            let selection = p
                .get("mask")
                .map(|m| crate::mask::coverage(m, nx, ny, source.width, source.height))
                .unwrap_or(1.0);
            let protection = p
                .get("protectMask")
                .map(|m| crate::mask::coverage(m, nx, ny, source.width, source.height))
                .unwrap_or(0.0);
            let weight = (1.0 - d * d).powi(2) * strength * selection * (1.0 - protection);
            match mode {
                "pinch" => (
                    cx + dx * (1.0 + weight * 0.4),
                    cy + dy * (1.0 + weight * 0.4),
                ),
                "bloat" => (
                    cx + dx * (1.0 - weight * 0.4),
                    cy + dy * (1.0 - weight * 0.4),
                ),
                "twirl" => {
                    let angle = -weight * 0.8 * direction;
                    (
                        cx + dx * angle.cos() - dy * angle.sin(),
                        cy + dx * angle.sin() + dy * angle.cos(),
                    )
                }
                _ => (x - (tx - cx) * weight, y - (ty - cy) * weight),
            }
        })?;
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct SymmetryWarp {
    source: (f32, f32),
    target: (f32, f32),
    radius: (f32, f32),
}

/// Symmetry needs actual feature contours, not a face rectangle. Pixel-space
/// geometry keeps the estimated facial axis correct on non-square images.
fn symmetry_feature(p: &Value, name: &str, source: &RgbaImage) -> Option<Vec<(f32, f32)>> {
    let masks = p.get("semanticMasks")?.as_object()?;
    let mask = masks
        .get(name)
        .or_else(|| (name == "lips").then(|| masks.get("lip")).flatten())?;
    let points = mask.as_array().or_else(|| mask.get("points")?.as_array())?;
    let points: Vec<_> = points
        .iter()
        .filter_map(|point| {
            let x = point.get("x")?.as_f64()?;
            let y = point.get("y")?.as_f64()?;
            (x.is_finite() && y.is_finite() && (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y))
                .then_some((
                    x as f32 * source.width as f32,
                    y as f32 * source.height as f32,
                ))
        })
        .collect();
    (points.len() >= 3).then_some(points)
}

fn symmetry_bounds(points: &[(f32, f32)]) -> Region {
    let mut bounds = Region {
        x: f32::INFINITY,
        y: f32::INFINITY,
        width: f32::NEG_INFINITY,
        height: f32::NEG_INFINITY,
    };
    for &(x, y) in points {
        bounds.x = bounds.x.min(x);
        bounds.y = bounds.y.min(y);
        bounds.width = bounds.width.max(x);
        bounds.height = bounds.height.max(y);
    }
    bounds.width -= bounds.x;
    bounds.height -= bounds.y;
    bounds
}

fn symmetry_center(bounds: Region) -> (f32, f32) {
    (
        bounds.x + bounds.width * 0.5,
        bounds.y + bounds.height * 0.5,
    )
}

/// Move each side's own texture toward the reflected position of its partner.
/// The eye midpoint and lip center estimate an in-plane, roll-aware midline;
/// this does not estimate 3D pose or remove perspective asymmetry in profiles.
fn read_symmetry_warps(p: &Value, source: &RgbaImage, amount: f32) -> Vec<SymmetryWarp> {
    if amount <= 1e-6 {
        return Vec::new();
    }
    let Some(features) = ["leftEye", "rightEye", "leftBrow", "rightBrow", "lips"]
        .iter()
        .map(|name| symmetry_feature(p, name, source))
        .collect::<Option<Vec<_>>>()
    else {
        return Vec::new();
    };
    let bounds: Vec<_> = features
        .iter()
        .map(|points| symmetry_bounds(points))
        .collect();
    if bounds
        .iter()
        .any(|bounds| bounds.width < 2.0 || bounds.height < 1.0)
    {
        return Vec::new();
    }
    let centers: Vec<_> = bounds
        .iter()
        .map(|bounds| symmetry_center(*bounds))
        .collect();
    let eye_distance = (centers[1].0 - centers[0].0).hypot(centers[1].1 - centers[0].1);
    let origin = (
        (centers[0].0 + centers[1].0) * 0.5,
        (centers[0].1 + centers[1].1) * 0.5,
    );
    let axis_delta = (centers[4].0 - origin.0, centers[4].1 - origin.1);
    let axis_length = axis_delta.0.hypot(axis_delta.1);
    // Reject degenerate/strongly foreshortened inputs instead of inventing a
    // symmetric frontal face from missing or implausible semantic geometry.
    if eye_distance < bounds[0].width.max(bounds[1].width) * 1.1
        || !(0.4..=2.5).contains(&(bounds[0].width / bounds[1].width))
        || axis_length < eye_distance * 0.3
        || axis_length > eye_distance * 2.0
        || axis_delta.1 / axis_length < 0.6
    {
        return Vec::new();
    }
    let axis = (axis_delta.0 / axis_length, axis_delta.1 / axis_length);
    let across = (axis.1, -axis.0);
    let project = |point: (f32, f32)| {
        let delta = (point.0 - origin.0, point.1 - origin.1);
        (
            delta.0 * across.0 + delta.1 * across.1,
            delta.0 * axis.0 + delta.1 * axis.1,
        )
    };
    let mut warps = Vec::new();
    let mut add_pair = |a: (f32, f32), b: (f32, f32), sizes: [(f32, f32); 2]| {
        let ap = project(a);
        let bp = project(b);
        let half_separation = (bp.0 - ap.0) * 0.5;
        let along = (ap.1 + bp.1) * 0.5;
        for (point, across_target, size) in [
            (a, -half_separation, sizes[0]),
            (b, half_separation, sizes[1]),
        ] {
            let desired = (
                origin.0 + across.0 * across_target + axis.0 * along,
                origin.1 + across.1 * across_target + axis.1 * along,
            );
            let mut delta = (
                (desired.0 - point.0) * amount,
                (desired.1 - point.1) * amount,
            );
            // Keep this a conservative cosmetic correction. Very large
            // asymmetries need a pose/3D model rather than a large 2D warp.
            let travel = delta.0.hypot(delta.1);
            let limit = eye_distance * 0.12;
            if travel > limit {
                delta.0 *= limit / travel;
                delta.1 *= limit / travel;
            }
            if travel <= 1e-4 {
                continue;
            }
            warps.push(SymmetryWarp {
                source: point,
                target: (point.0 + delta.0, point.1 + delta.1),
                radius: (
                    (size.0 * 0.9).max(delta.0.abs() * 3.0).max(3.0),
                    (size.1 * 1.5).max(delta.1.abs() * 3.0).max(3.0),
                ),
            });
        }
    };
    for [a, b] in [[0, 1], [2, 3]] {
        add_pair(
            centers[a],
            centers[b],
            [
                (bounds[a].width, bounds[a].height),
                (bounds[b].width, bounds[b].height),
            ],
        );
    }
    // Outer lip corners form another bilateral pair. Average points close to
    // each extreme so a single noisy vertex cannot dictate the mouth warp.
    let lip_projection: Vec<_> = features[4].iter().map(|&point| project(point).0).collect();
    let min = lip_projection.iter().copied().fold(f32::INFINITY, f32::min);
    let max = lip_projection
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let lip_corner = |right: bool| {
        let mut sum = (0.0, 0.0);
        let mut count = 0.0;
        for (&point, &projection) in features[4].iter().zip(&lip_projection) {
            if if right {
                projection >= max - (max - min) * 0.05
            } else {
                projection <= min + (max - min) * 0.05
            } {
                sum.0 += point.0;
                sum.1 += point.1;
                count += 1.0;
            }
        }
        (sum.0 / count, sum.1 / count)
    };
    let lip_size = (bounds[4].width * 0.45, bounds[4].height);
    add_pair(lip_corner(false), lip_corner(true), [lip_size; 2]);
    warps
}

fn symmetry_inverse(x: f32, y: f32, warps: &[SymmetryWarp]) -> (f32, f32) {
    let mut displacement = (0.0, 0.0);
    let mut total_weight = 0.0;
    for warp in warps {
        let dx = (x - warp.target.0) / warp.radius.0;
        let dy = (y - warp.target.1) / warp.radius.1;
        let distance = dx.hypot(dy);
        if distance >= 1.0 {
            continue;
        }
        // Translate the feature center intact, then taper the surrounding
        // skin to zero. Compact support leaves remote background untouched.
        let weight = 1.0 - smoothstep(0.35, 1.0, distance);
        displacement.0 += (warp.target.0 - warp.source.0) * weight;
        displacement.1 += (warp.target.1 - warp.source.1) * weight;
        total_weight += weight;
    }
    let denominator = total_weight.max(1.0);
    (
        x - displacement.0 / denominator,
        y - displacement.1 / denominator,
    )
}

#[derive(Clone, Copy)]
struct FeatureFrame {
    center: (f32, f32),
    across: (f32, f32),
    width: f32,
    height: f32,
}

/// Use the detected eye line as a pixel-space basis. Width, height, rotation
/// and translations then have the same meaning on tilted and tall portraits.
fn feature_frame(p: &Value, name: &str, source: &RgbaImage) -> Option<FeatureFrame> {
    let points = symmetry_feature(p, name, source)?;
    let across = match (
        symmetry_feature(p, "leftEye", source),
        symmetry_feature(p, "rightEye", source),
    ) {
        (Some(a), Some(b)) => {
            let a = symmetry_center(symmetry_bounds(&a));
            let b = symmetry_center(symmetry_bounds(&b));
            let delta = (b.0 - a.0, b.1 - a.1);
            let length = delta.0.hypot(delta.1);
            if length < 1.0 {
                (1.0, 0.0)
            } else {
                (delta.0 / length, delta.1 / length)
            }
        }
        _ => (1.0, 0.0),
    };
    let down = (-across.1, across.0);
    let mut limits = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for &(x, y) in &points {
        let u = x * across.0 + y * across.1;
        let v = x * down.0 + y * down.1;
        limits[0] = limits[0].min(u);
        limits[2] = limits[2].max(u);
        limits[1] = limits[1].min(v);
        limits[3] = limits[3].max(v);
    }
    let u = (limits[0] + limits[2]) * 0.5;
    let v = (limits[1] + limits[3]) * 0.5;
    Some(FeatureFrame {
        center: (u * across.0 + v * down.0, u * across.1 + v * down.1),
        across,
        width: (limits[2] - limits[0]).max(1.0),
        height: (limits[3] - limits[1]).max(1.0),
    })
}

#[derive(Clone, Copy)]
struct LocalWarp {
    center: (f32, f32),
    across: (f32, f32),
    radius: (f32, f32),
    scale: (f32, f32),
    translation: (f32, f32),
    angle: f32,
}

fn nose_alignment(p: &Value, source: &RgbaImage, amount: f32) -> Option<SymmetryWarp> {
    if amount.abs() < 1e-6 {
        return None;
    }
    let root = symmetry_feature(p, "noseRoot", source)?;
    let tip = symmetry_feature(p, "noseTip", source)?;
    let frame = feature_frame(p, "nose", source)?;
    let root = symmetry_center(symmetry_bounds(&root));
    let tip = symmetry_center(symmetry_bounds(&tip));
    let delta = (tip.0 - root.0, tip.1 - root.1);
    let u = delta.0 * frame.across.0 + delta.1 * frame.across.1;
    let v = -delta.0 * frame.across.1 + delta.1 * frame.across.0;
    if v < frame.height * 0.15 {
        return None;
    }
    // Align the detected tip to the face's down-axis through the bridge.
    // A straight nose has zero correction. The root remains outside support.
    let shift = (u * amount).clamp(-frame.width * 0.25, frame.width * 0.25);
    Some(SymmetryWarp {
        source: tip,
        target: (
            tip.0 - frame.across.0 * shift,
            tip.1 - frame.across.1 * shift,
        ),
        radius: (
            (frame.width * 0.7).max(shift.abs() * 2.5).max(2.0),
            (v.abs() * 0.8).max(2.0),
        ),
    })
}

// Each adjustment becomes a compact local inverse warp expressed inside the
// user-selected region. Semantic polygons anchor facial feature geometry to
// the actual detector result; relative presets remain a manual fallback.
#[cfg(test)]
fn local_adjust(source: &RgbaImage, p: &Value) -> Result<RgbaImage, ImageError> {
    local_adjust_kind(source, p, "feature.adjust")
}

/// Apply a local semantic adjustment. The operation kind is kept separate
/// from the public parameter object so clothing can use a frequency-aware
/// wrinkle pass while face/body operations continue using their shape pass.
pub(crate) fn local_adjust_kind(
    source: &RgbaImage,
    p: &Value,
    kind: &str,
) -> Result<RgbaImage, ImageError> {
    let region = Region::read(p.get("region"));
    let values = p.get("adjustments").unwrap_or(p);
    let semantic_features = kind == "feature.adjust" && p.get("semanticMasks").is_some();
    let mut symmetry_warps = if matches!(kind, "face.adjust" | "feature.adjust") {
        read_symmetry_warps(
            p,
            source,
            number(values, "symmetry", 0.0).clamp(0.0, 100.0) / 100.0,
        )
    } else {
        Vec::new()
    };
    let mut changes = Vec::<LocalWarp>::new();
    if kind == "feature.adjust" {
        if let Some(warp) = nose_alignment(p, source, percent(values, "noseStraight")) {
            symmetry_warps.push(warp);
        }
    }
    let mut add = |key: &str,
                   cx: f32,
                   cy: f32,
                   rx: f32,
                   ry: f32,
                   sx: f32,
                   sy: f32,
                   mx: f32,
                   my: f32,
                   angle: f32| {
        let amount = percent(values, key);
        if amount.abs() > 1e-6 {
            // Reposition and resize the canonical template around the feature
            // polygon's actual bounds. The modest support beyond the polygon
            // moves surrounding skin smoothly instead of cutting at its edge.
            // A supplied semantic set is authoritative: an absent eye/brow
            // cannot create a phantom warp at the old fixed face coordinates.
            let feature = if key.starts_with("eye") {
                Some((
                    if cx < 0.5 { "leftEye" } else { "rightEye" },
                    if cx < 0.5 { 0.32 } else { 0.68 },
                    0.39,
                    0.26,
                    0.10,
                ))
            } else if key.starts_with("brow") {
                Some((
                    if cx < 0.5 { "leftBrow" } else { "rightBrow" },
                    if cx < 0.5 { 0.32 } else { 0.68 },
                    0.28,
                    0.28,
                    0.06,
                ))
            } else if key.starts_with("mouth")
                || key.starts_with("lip")
                || matches!(key, "upperLip" | "lowerLip" | "philtrum")
            {
                Some(("lips", 0.5, 0.76, 0.40, 0.13))
            } else if key.starts_with("nose") || key == "nostril" {
                let name = match key {
                    "noseBridge" => "noseBridge",
                    "noseWings" => "noseWings",
                    "noseTip" => "noseTip",
                    "noseRoot" => "noseRoot",
                    "noseBase" => "noseBase",
                    "nostril" => {
                        if cx < 0.5 {
                            "leftNostril"
                        } else {
                            "rightNostril"
                        }
                    }
                    _ => "nose",
                };
                let dedicated = name != "nose";
                Some((
                    name,
                    if dedicated { cx } else { 0.5 },
                    if dedicated { cy } else { 0.54 },
                    if dedicated { rx / 0.85 } else { 0.32 },
                    if dedicated { ry / 0.85 } else { 0.32 },
                ))
            } else {
                None
            };
            let fallback = FeatureFrame {
                center: (
                    (region.x + cx * region.width) * source.width as f32,
                    (region.y + cy * region.height) * source.height as f32,
                ),
                across: (1.0, 0.0),
                width: region.width * source.width as f32,
                height: region.height * source.height as f32,
            };
            let (frame, radius, translation) = if semantic_features {
                if let Some((name, origin_x, origin_y, width, height)) = feature {
                    let Some(mut frame) = feature_frame(p, name, source) else {
                        return;
                    };
                    let scale_x = frame.width / width;
                    let scale_y = frame.height / height;
                    let du = (cx - origin_x) * scale_x;
                    let dv = (cy - origin_y) * scale_y;
                    frame.center.0 += du * frame.across.0 - dv * frame.across.1;
                    frame.center.1 += du * frame.across.1 + dv * frame.across.0;
                    (
                        frame,
                        ((rx * scale_x).max(1.0), (ry * scale_y).max(1.0)),
                        (mx * scale_x, my * scale_y),
                    )
                } else {
                    (
                        fallback,
                        (rx * fallback.width, ry * fallback.height),
                        (mx * fallback.width, my * fallback.height),
                    )
                }
            } else {
                (
                    fallback,
                    (rx * fallback.width, ry * fallback.height),
                    (mx * fallback.width, my * fallback.height),
                )
            };
            changes.push(LocalWarp {
                center: frame.center,
                across: frame.across,
                radius,
                scale: (sx * amount, sy * amount),
                translation: (translation.0 * amount, translation.1 * amount),
                angle: angle * amount,
            });
        }
    };
    for (cx, sign) in [(0.32, -1.0), (0.68, 1.0)] {
        add("eyeSize", cx, 0.39, 0.20, 0.13, -0.3, -0.3, 0.0, 0.0, 0.0);
        add("eyeWidth", cx, 0.39, 0.2, 0.13, -0.3, 0.0, 0.0, 0.0, 0.0);
        add("eyeHeight", cx, 0.39, 0.2, 0.13, 0.0, -0.3, 0.0, 0.0, 0.0);
        add(
            "eyeSpacing",
            cx,
            0.39,
            0.2,
            0.13,
            0.0,
            0.0,
            -sign * 0.08,
            0.0,
            0.0,
        );
        add(
            "eyeTilt",
            cx,
            0.39,
            0.2,
            0.13,
            0.0,
            0.0,
            0.0,
            0.0,
            sign * 0.3,
        );
        add(
            "eyeCorner",
            cx + sign * 0.08,
            0.39,
            0.12,
            0.1,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
        );
        add("eyelid", cx, 0.35, 0.18, 0.08, 0.0, -0.2, 0.0, 0.02, 0.0);
        add(
            "browLength",
            cx,
            0.28,
            0.21,
            0.10,
            -0.25,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        add("browWidth", cx, 0.28, 0.21, 0.10, -0.15, 0.0, 0.0, 0.0, 0.0);
        add(
            "browThickness",
            cx,
            0.28,
            0.21,
            0.09,
            0.0,
            -0.3,
            0.0,
            0.0,
            0.0,
        );
        add("browPeak", cx, 0.26, 0.13, 0.09, 0.0, 0.0, 0.0, 0.04, 0.0);
        add(
            "browTail",
            cx + sign * 0.08,
            0.29,
            0.12,
            0.08,
            0.0,
            0.0,
            -sign * 0.03,
            0.03,
            0.0,
        );
    }
    add(
        "noseBridge",
        0.5,
        0.49,
        0.16,
        0.19,
        -0.24,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    add("noseWings", 0.5, 0.6, 0.18, 0.12, -0.26, 0.0, 0.0, 0.0, 0.0);
    add(
        "noseLength",
        0.5,
        0.54,
        0.18,
        0.23,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.0,
    );
    add("noseTip", 0.5, 0.62, 0.13, 0.12, -0.2, -0.2, 0.0, 0.02, 0.0);
    add(
        "mouthWidth",
        0.5,
        0.76,
        0.28,
        0.15,
        -0.24,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    add(
        "mouthHeight",
        0.5,
        0.76,
        0.28,
        0.15,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.0,
    );
    add("upperLip", 0.5, 0.73, 0.26, 0.07, 0.0, -0.3, 0.0, 0.0, 0.0);
    add("lowerLip", 0.5, 0.79, 0.26, 0.07, 0.0, -0.3, 0.0, 0.0, 0.0);
    add("philtrum", 0.5, 0.68, 0.2, 0.11, 0.0, -0.2, 0.0, 0.0, 0.0);

    for (cx, sign) in [(0.32, -1.0), (0.68, 1.0)] {
        add(
            "eyeInnerCorner",
            cx - sign * 0.08,
            0.39,
            0.10,
            0.09,
            0.0,
            0.0,
            -sign * 0.04,
            0.02,
            0.0,
        );
        add(
            "eyeOuterCorner",
            cx + sign * 0.08,
            0.39,
            0.10,
            0.09,
            0.0,
            0.0,
            sign * 0.04,
            0.03,
            0.0,
        );
        add(
            "browSpacing",
            cx,
            0.28,
            0.21,
            0.09,
            0.0,
            0.0,
            -sign * 0.07,
            0.0,
            0.0,
        );
        add(
            "nostril",
            0.5 + sign * 0.06,
            0.63,
            0.07,
            0.07,
            -0.2,
            -0.15,
            0.0,
            0.0,
            0.0,
        );
        add(
            "lipPeak",
            0.5 + sign * 0.05,
            0.73,
            0.07,
            0.07,
            0.0,
            -0.2,
            0.0,
            0.03,
            0.0,
        );
        add(
            "lipCorner",
            0.5 + sign * 0.18,
            0.76,
            0.10,
            0.09,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
        );
    }
    add(
        "noseHeight",
        0.5,
        0.56,
        0.15,
        0.22,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.0,
    );
    add(
        "noseRoot", 0.5, 0.42, 0.14, 0.11, -0.2, 0.0, 0.0, 0.025, 0.0,
    );
    add("noseBase", 0.5, 0.64, 0.18, 0.09, 0.0, -0.2, 0.0, 0.02, 0.0);
    add(
        "noseLipAngle",
        0.5,
        0.65,
        0.16,
        0.15,
        0.0,
        -0.1,
        0.0,
        0.04,
        0.0,
    );

    // A polygon/ellipse selection is also part of the displacement field.
    // Applying it only in `mask::composite` leaves the hidden rectangular
    // render free to pull background texels into the face edge.  Scaling the
    // local displacement by the same coverage keeps the field continuous at
    // the selection boundary and makes the fallback safe for manual masks.
    let selection_mask = p.get("mask").filter(|value| value.is_object());
    // Facial makeup is defined in the input feature coordinates. Color it
    // before sampling geometry so an edited lip/eyebrow carries its makeup
    // with it instead of leaving pigment at the pre-warp polygon location.
    let mut facial_input = matches!(kind, "face.adjust" | "feature.adjust")
        .then(|| crate::color::local_color(source, p, false));
    if let Some(input) = &mut facial_input {
        if selection_mask.is_some() || p.get("protectMask").is_some() {
            for y in 0..source.height {
                for x in 0..source.width {
                    let nx = (x as f32 + 0.5) / source.width as f32;
                    let ny = (y as f32 + 0.5) / source.height as f32;
                    let excluded = selection_mask.is_some_and(|mask| {
                        crate::mask::coverage(mask, nx, ny, source.width, source.height) <= 0.0
                    });
                    let frozen = p.get("protectMask").is_some_and(|mask| {
                        crate::mask::coverage(mask, nx, ny, source.width, source.height) >= 1.0
                    });
                    if excluded || frozen {
                        let index = (y as usize * source.width as usize + x as usize) * 4;
                        input.pixels[index..index + 4]
                            .copy_from_slice(&source.pixels[index..index + 4]);
                    }
                }
            }
        }
    }
    let sampling_source = facial_input.as_ref().unwrap_or(source);
    let mut result = remap(
        sampling_source,
        source.width,
        source.height,
        true,
        |x, y| {
            let nx = x / source.width as f32;
            let ny = y / source.height as f32;
            if !region.contains(nx, ny) {
                return (x, y);
            }
            let rx = (nx - region.x) / region.width;
            let ry = (ny - region.y) / region.height;
            let mut selection_weight = selection_mask
                .map(|mask| crate::mask::coverage(mask, nx, ny, source.width, source.height))
                .unwrap_or(1.0);
            if let Some(protection) = p.get("protectMask") {
                selection_weight *=
                    1.0 - crate::mask::coverage(protection, nx, ny, source.width, source.height);
            }
            let mut ux = rx;
            let mut uy = ry;

            for warp in &changes {
                let delta = (x - warp.center.0, y - warp.center.1);
                let u = delta.0 * warp.across.0 + delta.1 * warp.across.1;
                let v = -delta.0 * warp.across.1 + delta.1 * warp.across.0;
                let dx = u / warp.radius.0;
                let dy = v / warp.radius.1;
                let d = dx * dx + dy * dy;
                if d >= 1.0 {
                    continue;
                }
                let weight = (1.0 - d).powi(2) * selection_weight;
                let a = warp.angle * weight;
                let du = (u * warp.scale.0 + warp.translation.0) * weight + u * (a.cos() - 1.0)
                    - v * a.sin();
                let dv = (v * warp.scale.1 + warp.translation.1) * weight
                    + u * a.sin()
                    + v * (a.cos() - 1.0);
                ux += (du * warp.across.0 - dv * warp.across.1)
                    / (source.width as f32 * region.width);
                uy += (du * warp.across.1 + dv * warp.across.0)
                    / (source.height as f32 * region.height);
            }
            if !symmetry_warps.is_empty() {
                let (sx, sy) = symmetry_inverse(x, y, &symmetry_warps);
                let edge_distance = rx.min(1.0 - rx).min(ry).min(1.0 - ry);
                let weight = smoothstep(0.0, 0.06, edge_distance) * selection_weight;
                ux += (sx - x) / (source.width as f32 * region.width) * weight;
                uy += (sy - y) / (source.height as f32 * region.height) * weight;
            }
            (
                (region.x + ux * region.width) * source.width as f32,
                (region.y + uy * region.height) * source.height as f32,
            )
        },
    )?;
    // Clothing's frequency-based wrinkle/color pass retains its existing
    // post-geometry behavior; it does not use facial feature coordinates.
    if facial_input.is_none() {
        result = crate::color::local_color(&result, p, kind == "clothing.adjust");
    }
    Ok(result)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn clean_sample(source: &RgbaImage, x: f32, y: f32, radius: f32) -> (f32, f32) {
    let mut boundary = [0.0; 3];
    for i in 0..16 {
        let a = i as f32 * std::f32::consts::TAU / 16.0;
        let pixel = sample(
            source,
            x + a.cos() * radius * 1.1,
            y + a.sin() * radius * 1.1,
            true,
        );
        for c in 0..3 {
            boundary[c] += pixel[c] as f32 / 16.0;
        }
    }
    let mut best = (x - radius * 2.5, y);
    let mut best_score = f32::MAX;
    for i in 0..16 {
        let angle = i as f32 * std::f32::consts::TAU / 16.0;
        let sx = (x + angle.cos() * radius * 2.5).clamp(0.0, source.width as f32 - 1.0);
        let sy = (y + angle.sin() * radius * 2.5).clamp(0.0, source.height as f32 - 1.0);
        if (sx - x).hypot(sy - y) < radius * 1.6 {
            continue;
        }
        let center = sample(source, sx, sy, true);
        let mut score = 0.0;
        for c in 0..3 {
            score += (center[c] as f32 - boundary[c]).powi(2);
        }
        for j in 0..8 {
            let a = j as f32 * std::f32::consts::TAU / 8.0;
            let pixel = sample(
                source,
                sx + a.cos() * radius * 0.7,
                sy + a.sin() * radius * 0.7,
                true,
            );
            for c in 0..3 {
                score += (pixel[c] as f32 - center[c] as f32).powi(2) * 0.35;
            }
        }
        if score < best_score {
            best_score = score;
            best = (sx, sy);
        }
    }
    best
}

/// Traditional masked fill for the demo: propagate surrounding clean colors
/// into selected pixels, then relax the fill at its boundary. This is a local
/// non-generative algorithm, so it cannot invent a missing semantic object.
pub(crate) fn repair(source: &RgbaImage, p: &Value, heal: bool) -> Result<RgbaImage, ImageError> {
    let Some(strokes) = p.get("strokes").and_then(Value::as_array) else {
        return Ok(source.clone());
    };
    let mut result = source.clone();
    for stroke in strokes {
        let x = number(stroke, "x", 0.5) * source.width as f32;
        let y = number(stroke, "y", 0.5) * source.height as f32;
        let radius = number(stroke, "radius", 0.04).clamp(0.001, 1.0)
            * source.width.min(source.height) as f32;
        let (sx, sy) =
            if heal && (stroke.get("sourceX").is_none() || stroke.get("sourceY").is_none()) {
                clean_sample(source, x, y, radius)
            } else {
                (
                    number(stroke, "sourceX", 0.25) * source.width as f32,
                    number(stroke, "sourceY", 0.25) * source.height as f32,
                )
            };
        let strength = number(stroke, "strength", 100.0).clamp(0.0, 100.0) / 100.0;
        let mut correction = [0.0; 3];
        if heal {
            let mut count = 0.0;
            for i in 0..32 {
                let angle = i as f32 * std::f32::consts::TAU / 32.0;
                let (dx, dy) = (angle.cos() * radius * 0.85, angle.sin() * radius * 0.85);
                let dest = sample(&result, x + dx, y + dy, true);
                let src = sample(source, sx + dx, sy + dy, true);
                for c in 0..3 {
                    correction[c] += dest[c] as f32 - src[c] as f32;
                }
                count += 1.0;
            }
            for value in &mut correction {
                *value /= count;
            }
        }
        let xmin = (x - radius).floor().max(0.0) as u32;
        let xmax = (x + radius).ceil().min(source.width as f32) as u32;
        let ymin = (y - radius).floor().max(0.0) as u32;
        let ymax = (y + radius).ceil().min(source.height as f32) as u32;
        for py in ymin..ymax {
            for px in xmin..xmax {
                let dx = px as f32 - x;
                let dy = py as f32 - y;
                let d = (dx * dx + dy * dy).sqrt() / radius;
                if d >= 1.0 {
                    continue;
                }
                let falloff = if d < 0.65 {
                    1.0
                } else {
                    let t = (1.0 - d) / 0.35;
                    t * t * (3.0 - 2.0 * t)
                };
                let weight = falloff * strength;
                let cloned = sample(source, sx + dx, sy + dy, true);
                let index = (py as usize * source.width as usize + px as usize) * 4;
                for c in 0..3 {
                    let target = cloned[c] as f32 + correction[c];
                    result.pixels[index + c] =
                        byte(result.pixels[index + c] as f32 * (1.0 - weight) + target * weight);
                }
                // Repairing color does not change the original transparency.
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn source() -> RgbaImage {
        RgbaImage::new(
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 200, 0, 0, 255, 100, 255, 255, 255, 0,
            ],
        )
        .unwrap()
    }
    #[test]
    fn crop_flip_and_quarter_turn_have_exact_pixel_coordinates() {
        let image = source();
        assert_eq!(
            crop(&image, &json!({"x":1,"y":0,"width":1,"height":2}))
                .unwrap()
                .pixels,
            vec![0, 255, 0, 200, 255, 255, 255, 0]
        );
        assert_eq!(
            &flip(&image, &json!({"axis":"horizontal"})).unwrap().pixels[..4],
            &image.pixels[4..8]
        );
        let rotated = rotate(&image, &json!({"degrees":90})).unwrap();
        assert_eq!(&rotated.pixels[..4], &image.pixels[8..12]);
    }
    #[test]
    fn arbitrary_rotation_expands_and_keeps_transparent_corners() {
        let result = rotate(&source(), &json!({"degrees":45})).unwrap();
        assert_eq!((result.width, result.height), (3, 3));
        assert!(result.pixels.chunks_exact(4).any(|p| p[3] < 255));
    }
    #[test]
    fn perspective_identity_preserves_opaque_pixels_and_rejects_degenerate_quad() {
        let image = RgbaImage::new(
            2,
            2,
            vec![
                20, 30, 40, 255, 40, 50, 60, 255, 60, 70, 80, 255, 80, 90, 100, 255,
            ],
        )
        .unwrap();
        let points = json!({"corners":[{"x":0,"y":0},{"x":1,"y":0},{"x":1,"y":1},{"x":0,"y":1}]});
        assert_eq!(perspective(&image, &points).unwrap(), image);
        assert!(perspective(&image, &json!({"corners":[{},{},{},{}]})).is_err());
    }
    #[test]
    fn clone_copies_center_color_and_preserves_alpha() {
        let mut pixels = vec![10; 20 * 20 * 4];
        for p in pixels.chunks_exact_mut(4) {
            p[3] = 77;
        }
        let idx = (5 * 20 + 5) * 4;
        pixels[idx..idx + 3].copy_from_slice(&[200, 50, 20]);
        let image = RgbaImage::new(20, 20, pixels).unwrap();
        let out=repair(&image,&json!({"strokes":[{"x":0.5,"y":0.5,"sourceX":0.25,"sourceY":0.25,"radius":0.2,"strength":100}]}),false).unwrap();
        assert_eq!(
            &out.pixels[(10 * 20 + 10) * 4..(10 * 20 + 10) * 4 + 4],
            &[200, 50, 20, 77]
        );
        assert_eq!(&out.pixels[..4], &image.pixels[..4]);
    }

    fn feature_fixture() -> RgbaImage {
        let mut pixels = Vec::new();
        for y in 0..192 {
            for x in 0..192 {
                pixels.extend_from_slice(&[
                    ((x * 11 + y * 3) % 256) as u8,
                    ((x * 7 + y * 13) % 256) as u8,
                    ((x * 19 + y * 17) % 256) as u8,
                    211,
                ]);
            }
        }
        RgbaImage::new(192, 192, pixels).unwrap()
    }

    #[test]
    fn feature_geometry_follows_semantic_polygons_instead_of_fixed_face_anchors() {
        let source = feature_fixture();
        for (key, semantic, bounds, legacy) in [
            ("eyeSize", "leftEye", (0.66, 0.65, 0.18, 0.07), (0.36, 0.40)),
            (
                "browThickness",
                "rightBrow",
                (0.15, 0.69, 0.20, 0.05),
                (0.67, 0.32),
            ),
            ("mouthWidth", "lips", (0.16, 0.23, 0.22, 0.08), (0.53, 0.71)),
        ] {
            let (x, y, width, height) = bounds;
            let params = json!({
                "region":{"x":0.1,"y":0.1,"width":0.8,"height":0.8},
                "adjustments":{key:100},
                "semanticMasks":{semantic:{"shape":"polygon","points":[
                    {"x":x,"y":y},{"x":x+width,"y":y},
                    {"x":x+width,"y":y+height},{"x":x,"y":y+height}
                ]}}
            });
            let result = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            let changed = (0..192 * 192)
                .filter(|index| {
                    let px = (*index % 192) as f32 / 192.0;
                    let py = (*index / 192) as f32 / 192.0;
                    px > x
                        && px < x + width
                        && py > y
                        && py < y + height
                        && result.pixels[*index * 4..*index * 4 + 3]
                            != source.pixels[*index * 4..*index * 4 + 3]
                })
                .count();
            assert!(
                changed > 10,
                "{key} must reshape the actual {semantic} polygon"
            );
            let index = ((legacy.1 * 192.0) as usize * 192 + (legacy.0 * 192.0) as usize) * 4;
            assert_eq!(
                &result.pixels[index..index + 4],
                &source.pixels[index..index + 4],
                "{key} must leave the old template location untouched"
            );
            assert!(result.pixels.chunks_exact(4).all(|pixel| pixel[3] == 211));
        }
    }

    #[test]
    fn feature_shape_field_respects_polygon_and_frozen_area_before_compositing() {
        let source = feature_fixture();
        let params = json!({
            "adjustments":{"eyeSize":100,"browLength":80},
            "mask":{"shape":"polygon","points":[
                {"x":0.18,"y":0.2},{"x":0.48,"y":0.2},{"x":0.18,"y":0.62}
            ]},
            "protectMask":{"shape":"rect","region":{"x":0.25,"y":0.3,"width":0.08,"height":0.14}}
        });
        let result = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
        let mut changed = 0;
        for y in 0..192 {
            for x in 0..192 {
                let nx = x as f32 / 192.0;
                let ny = y as f32 / 192.0;
                let index = (y * 192 + x) * 4;
                let outside = crate::mask::coverage(&params["mask"], nx, ny, 192, 192) == 0.0;
                let frozen = crate::mask::coverage(&params["protectMask"], nx, ny, 192, 192) == 1.0;
                if outside || frozen {
                    assert_eq!(
                        &result.pixels[index..index + 4],
                        &source.pixels[index..index + 4]
                    );
                } else if result.pixels[index..index + 4] != source.pixels[index..index + 4] {
                    changed += 1;
                }
            }
        }
        assert!(changed > 10);
        let fallback = local_adjust_kind(
            &source,
            &json!({"adjustments":{"eyeSize":100}}),
            "feature.adjust",
        )
        .unwrap();
        assert_ne!(
            fallback.pixels, source.pixels,
            "manual operations without semantic masks keep their compatibility templates"
        );
    }

    fn symmetry_polygon(cx: f32, cy: f32, width: f32, height: f32) -> Value {
        json!({"shape":"polygon","points":[
            {"x":(cx-width*0.5)/256.0,"y":(cy-height*0.5)/256.0},
            {"x":(cx+width*0.5)/256.0,"y":(cy-height*0.5)/256.0},
            {"x":(cx+width*0.5)/256.0,"y":(cy+height*0.5)/256.0},
            {"x":(cx-width*0.5)/256.0,"y":(cy+height*0.5)/256.0}
        ]})
    }

    fn symmetry_fixture() -> (RgbaImage, Value) {
        let mut pixels = Vec::new();
        for y in 0i32..256 {
            for x in 0i32..256 {
                let mut color = if x < 128 {
                    [65, 40, 30, 211]
                } else {
                    [30, 50, 80, 211]
                };
                for (cx, cy, marker) in
                    [(80, 88, [230, 40, 20, 211]), (176, 100, [20, 50, 230, 211])]
                {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= 9 {
                        color = marker;
                    }
                }
                pixels.extend_from_slice(&color);
            }
        }
        let masks = json!({
            "leftEye":symmetry_polygon(80.0,88.0,28.0,12.0),
            "rightEye":symmetry_polygon(176.0,100.0,28.0,12.0),
            "leftBrow":symmetry_polygon(78.0,62.0,30.0,8.0),
            "rightBrow":symmetry_polygon(172.0,66.0,30.0,8.0),
            "lips":symmetry_polygon(128.0,172.0,54.0,20.0)
        });
        (
            RgbaImage::new(256, 256, pixels).unwrap(),
            json!({"adjustments":{"symmetry":100},"semanticMasks":masks}),
        )
    }

    fn eye_marker_center(image: &RgbaImage, right: bool) -> (f32, f32) {
        let mut sum = (0.0, 0.0);
        let mut count = 0.0;
        for y in 0..image.height as usize {
            for x in 0..image.width as usize {
                let index = (y * image.width as usize + x) * 4;
                if image.pixels[index + if right { 2 } else { 0 }] > 150 {
                    sum.0 += x as f32;
                    sum.1 += y as f32;
                    count += 1.0;
                }
            }
        }
        assert!(count > 10.0);
        (sum.0 / count, sum.1 / count)
    }

    #[test]
    fn symmetry_aligns_feature_positions_without_mixing_opposite_side_colors() {
        let (source, params) = symmetry_fixture();
        let result = local_adjust(&source, &params).unwrap();
        let left = eye_marker_center(&result, false);
        let right = eye_marker_center(&result, true);
        assert!(
            (left.1 - right.1).abs() < 1.0,
            "eye centers must approach the same axial position"
        );
        assert!((left.0 - 80.0).abs() < 1.0 && (right.0 - 176.0).abs() < 1.0);
        for (x, color) in [(80, [230, 40, 20, 211]), (176, [20, 50, 230, 211])] {
            let index = (94 * 256 + x) * 4;
            assert_eq!(&result.pixels[index..index + 4], &color);
        }
        // Lighting remains different on the two sides. A reflection/color
        // average would change these pixels even though no feature is there.
        for (x, y) in [(64, 148), (192, 148), (15, 94), (128, 12)] {
            let index = (y * 256 + x) * 4;
            assert_eq!(
                &result.pixels[index..index + 4],
                &source.pixels[index..index + 4]
            );
        }
        assert!(result.pixels.chunks_exact(4).all(|pixel| pixel[3] == 211));
    }

    #[test]
    fn symmetry_uses_rolled_axis_in_pixel_space_on_wide_images() {
        let (source, mut params) = symmetry_fixture();
        let angle = 0.35f32;
        let transform = |x: f32, y: f32| {
            let dx = x - 128.0;
            let dy = y - 94.0;
            (
                256.0 + dx * angle.cos() - dy * angle.sin(),
                96.0 + dx * angle.sin() + dy * angle.cos(),
            )
        };
        let rolled = remap(&source, 512, 256, false, |x, y| {
            let dx = x - 256.0;
            let dy = y - 96.0;
            (
                128.0 + dx * angle.cos() + dy * angle.sin(),
                94.0 - dx * angle.sin() + dy * angle.cos(),
            )
        })
        .unwrap();
        for mask in params["semanticMasks"]
            .as_object_mut()
            .unwrap()
            .values_mut()
        {
            for point in mask["points"].as_array_mut().unwrap() {
                let (x, y) = transform(
                    point["x"].as_f64().unwrap() as f32 * 256.0,
                    point["y"].as_f64().unwrap() as f32 * 256.0,
                );
                point["x"] = json!(x / 512.0);
                point["y"] = json!(y / 256.0);
            }
        }
        let aligned = local_adjust(&rolled, &params).unwrap();
        for (right, original_x) in [(false, 80.0), (true, 176.0)] {
            let observed = eye_marker_center(&aligned, right);
            let expected = transform(original_x, 94.0);
            assert!(
                (observed.0 - expected.0).hypot(observed.1 - expected.1) < 1.2,
                "feature should align about the rolled facial axis, preserving head tilt"
            );
        }
        params["adjustments"]["symmetry"] = json!(50);
        let halfway = local_adjust(&rolled, &params).unwrap();
        let expected = transform(80.0, 91.0);
        let observed = eye_marker_center(&halfway, false);
        assert!(
            (observed.0 - expected.0).hypot(observed.1 - expected.1) < 1.2,
            "half strength should move halfway toward the symmetric location"
        );
        assert_eq!(&aligned.pixels[..4], &rolled.pixels[..4]);
    }

    #[test]
    fn symmetry_corrects_lip_corner_height_using_each_corners_own_texture() {
        let (mut source, mut params) = symmetry_fixture();
        params["semanticMasks"]["lips"] = json!({"shape":"polygon","points":[
            {"x":101.0/256.0,"y":174.0/256.0},{"x":128.0/256.0,"y":164.0/256.0},
            {"x":155.0/256.0,"y":166.0/256.0},{"x":128.0/256.0,"y":180.0/256.0}
        ]});
        for (cx, cy, color) in [
            (101i32, 174i32, [230, 40, 20, 211]),
            (155, 166, [20, 50, 230, 211]),
        ] {
            for y in cy - 3..=cy + 3 {
                for x in cx - 3..=cx + 3 {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= 9 {
                        let index = (y as usize * 256 + x as usize) * 4;
                        source.pixels[index..index + 4].copy_from_slice(&color);
                    }
                }
            }
        }
        let aligned = local_adjust(&source, &params).unwrap();
        for (x, color) in [(101, [230, 40, 20, 211]), (155, [20, 50, 230, 211])] {
            let index = (170 * 256 + x) * 4;
            assert_eq!(&aligned.pixels[index..index + 4], &color);
        }
    }

    #[test]
    fn symmetry_obeys_selection_and_frozen_pixels_before_compositing() {
        let (source, mut params) = symmetry_fixture();
        params["mask"] = json!({"shape":"polygon","points":[
            {"x":0.16,"y":0.1},{"x":0.84,"y":0.1},{"x":0.84,"y":0.53},{"x":0.16,"y":0.53}
        ]});
        params["protectMask"] =
            json!({"shape":"rect","region":{"x":0.26,"y":0.29,"width":0.13,"height":0.15}});
        let result = local_adjust(&source, &params).unwrap();
        let mut changed = 0;
        for y in 0..256 {
            for x in 0..256 {
                let nx = x as f32 / 256.0;
                let ny = y as f32 / 256.0;
                let index = (y * 256 + x) * 4;
                let outside = crate::mask::coverage(&params["mask"], nx, ny, 256, 256) == 0.0;
                let frozen = crate::mask::coverage(&params["protectMask"], nx, ny, 256, 256) == 1.0;
                if outside || frozen {
                    assert_eq!(
                        &result.pixels[index..index + 4],
                        &source.pixels[index..index + 4]
                    );
                } else if result.pixels[index..index + 4] != source.pixels[index..index + 4] {
                    changed += 1;
                }
            }
        }
        assert!(changed > 20);
        assert!((eye_marker_center(&result, false).1 - 88.0).abs() < 0.5);
        assert!((eye_marker_center(&result, true).1 - 94.0).abs() < 1.0);
    }

    #[test]
    fn symmetry_without_complete_reliable_features_keeps_source_exact() {
        let (source, mut params) = symmetry_fixture();
        assert_eq!(
            local_adjust(&source, &json!({"adjustments":{"symmetry":100}})).unwrap(),
            source
        );
        params["semanticMasks"]
            .as_object_mut()
            .unwrap()
            .remove("rightEye");
        assert_eq!(local_adjust(&source, &params).unwrap(), source);
        let (_, mut profile) = symmetry_fixture();
        profile["semanticMasks"]["rightEye"] = symmetry_polygon(176.0, 100.0, 6.0, 12.0);
        assert_eq!(local_adjust(&source, &profile).unwrap(), source);
    }

    #[test]
    fn widened_lips_carry_makeup_beyond_original_semantic_polygon() {
        let mut pixels = Vec::new();
        for y in 0..256 {
            for x in 0..256 {
                pixels.extend_from_slice(if (80..160).contains(&x) && (128..152).contains(&y) {
                    &[175, 60, 75, 255]
                } else {
                    &[150, 120, 100, 255]
                });
            }
        }
        let source = RgbaImage::new(256, 256, pixels).unwrap();
        let shape = json!({"adjustments":{"mouthWidth":100},"semanticMasks":{"lips":symmetry_polygon(120.0,140.0,80.0,24.0)}});
        let mut combined = shape.clone();
        combined["adjustments"]["lipColor"] = json!(100);
        let enlarged = local_adjust_kind(&source, &shape, "feature.adjust").unwrap();
        let made_up = local_adjust_kind(&source, &combined, "feature.adjust").unwrap();
        let carried_tint = (161..175).any(|x| {
            let index = (140 * 256 + x) * 4;
            enlarged.pixels[index] > 160 && made_up.pixels[index] > enlarged.pixels[index] + 5
        });
        assert!(
            carried_tint,
            "makeup must follow the visible lip edge outside its old polygon"
        );
        let unedited = (140 * 256 + 180) * 4;
        assert_eq!(
            &made_up.pixels[unedited..unedited + 4],
            &source.pixels[unedited..unedited + 4]
        );
    }

    fn gradient() -> RgbaImage {
        let mut pixels = Vec::new();
        for y in 0..64 {
            for x in 0..64 {
                pixels.extend_from_slice(&[(x * 3) as u8, (y * 3) as u8, 80, 255]);
            }
        }
        RgbaImage::new(64, 64, pixels).unwrap()
    }
    #[test]
    fn reconstruct_restores_only_current_liquify_input_under_earlier_color() {
        let source = gradient();
        let earlier = crate::color::basic(&source, &json!({"brightness":20}));
        let push =
            json!({"x":0.5,"y":0.5,"toX":0.65,"toY":0.5,"radius":0.3,"strength":100,"mode":"push"});
        let moved = liquify(&earlier, &json!({"strokes":[push.clone()]})).unwrap();
        let restored=liquify(&earlier,&json!({"strokes":[push,{"x":0.5,"y":0.5,"radius":0.1,"strength":100,"mode":"reconstruct"}]})).unwrap();
        let center = (32 * 64 + 32) * 4;
        let outside = (32 * 64 + 42) * 4;
        assert_ne!(
            &moved.pixels[center..center + 4],
            &earlier.pixels[center..center + 4]
        );
        assert_eq!(
            &restored.pixels[center..center + 4],
            &earlier.pixels[center..center + 4]
        );
        assert_ne!(
            &restored.pixels[center..center + 4],
            &source.pixels[center..center + 4]
        );
        assert_eq!(
            &restored.pixels[outside..outside + 4],
            &moved.pixels[outside..outside + 4]
        );
    }
    #[test]
    fn opposite_twirl_directions_move_pixels_oppositely_and_keep_protection() {
        let source = gradient();
        let index = (32 * 64 + 40) * 4;
        let first=liquify(&source,&json!({"strokes":[{"x":0.5,"y":0.5,"radius":0.3,"strength":100,"mode":"twirl","direction":1}]})).unwrap();
        let second=liquify(&source,&json!({"strokes":[{"x":0.5,"y":0.5,"radius":0.3,"strength":100,"mode":"twirl","direction":-1}]})).unwrap();
        assert!(first.pixels[index + 1] < source.pixels[index + 1]);
        assert!(second.pixels[index + 1] > source.pixels[index + 1]);
        let graph = json!({"operations":[{"type":"warp.liquify","params":{"strokes":[{"x":0.5,"y":0.5,"radius":0.3,"strength":100,"mode":"twirl","direction":-1}],"protectMask":{"shape":"rect","region":{"x":0.5,"y":0,"width":0.5,"height":1}}}}]});
        let protected = crate::apply_graph_json(&source, &graph.to_string()).unwrap();
        for y in 0..64 {
            for x in 32..64 {
                let i = (y * 64 + x) * 4;
                assert_eq!(&protected.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
    }
}
