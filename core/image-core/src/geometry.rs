use crate::engine::{as_native, byte, from_native, number, percent};
use crate::mask::{PreparedMask, Region};
use crate::{ImageError, RgbaImage};
use serde_json::Value;

#[path = "eye_closure_field.rs"]
pub(crate) mod eye_closure_field;

#[cfg(test)]
#[path = "eye_material_admission.rs"]
mod eye_material_admission;

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
            // Identity pixels retain their exact RGBA, including hidden RGB at
            // alpha zero. Only genuinely moved pixels need alpha interpolation.
            // Expanded geometry canvases can extend beyond the source bounds.
            if sx == x as f32 && sy == y as f32 && x < source.width && y < source.height {
                let source_index = (y as usize * source.width as usize + x as usize) * 4;
                out.pixels[index..index + 4]
                    .copy_from_slice(&source.pixels[source_index..source_index + 4]);
            } else {
                out.pixels[index..index + 4].copy_from_slice(&sample(source, sx, sy, clamp_edges));
            }
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

impl FeatureFrame {
    /// Distance to the enclosing feature bounds in the eye-line basis. The
    /// whole detected eye is fixed for eyebrow edits; adjacent skin approaches
    /// that constraint smoothly instead of receiving a final hard pixel cut.
    fn surrounding_motion(&self, x: f32, y: f32) -> f32 {
        let delta = (x - self.center.0, y - self.center.1);
        let u = delta.0 * self.across.0 + delta.1 * self.across.1;
        let v = -delta.0 * self.across.1 + delta.1 * self.across.0;
        // The detector eye polygon encloses the aperture, not the eyelid
        // crease and eyeliner. Include that surrounding anatomy in the fixed
        // envelope; otherwise preserving the white of the eye still bends
        // its upper lid into an arch when the brow moves.
        let lid_margin = (self.height * 0.75).clamp(2.0, 24.0);
        let du = (u.abs() - self.width * 0.5 - lid_margin).max(0.0);
        let dv = (v.abs() - self.height * 0.5 - lid_margin).max(0.0);
        smoothstep(2.0, (self.width * 0.28).clamp(4.0, 24.0), du.hypot(dv))
    }
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

#[derive(Clone)]
struct FeatureBrowAnchor {
    points: Vec<(f32, f32)>,
    bounds: [f32; 4],
}
impl FeatureBrowAnchor {
    fn new(points: Vec<(f32, f32)>) -> Self {
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for &(x, y) in &points {
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
        }
        Self { points, bounds }
    }
    fn surrounding_motion(&self, point: (f32, f32)) -> f32 {
        let (x, y) = point;
        if x < self.bounds[0] - 3.5
            || y < self.bounds[1] - 3.5
            || x > self.bounds[2] + 3.5
            || y > self.bounds[3] + 3.5
        {
            return 1.0;
        }
        let mut inside = false;
        let mut distance = f32::INFINITY;
        for (a, b) in self
            .points
            .iter()
            .zip(self.points.iter().cycle().skip(1))
            .take(self.points.len())
        {
            if (a.1 > y) != (b.1 > y) && x < (b.0 - a.0) * (y - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
            let edge = (b.0 - a.0, b.1 - a.1);
            let length = edge.0 * edge.0 + edge.1 * edge.1;
            let t = if length > 0.0 {
                ((x - a.0) * edge.0 + (y - a.1) * edge.1) / length
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            distance = distance.min((x - a.0 - edge.0 * t).hypot(y - a.1 - edge.1 * t));
        }
        if inside {
            0.0
        } else {
            smoothstep(1.5, 3.5, distance)
        }
    }
}

const EYE_WIDTH_LOG_SCALE: f32 = 0.3;

const EYE_SIZE_LOG_SCALE: f32 = 0.3;
const EYE_TILT_RADIANS: f32 = 0.3;

// A cos(t) + B sin(t) has absolute maxima at phase + k*pi.
// Include interior extrema, not just the endpoints of the requested sweep.
fn angular_component_extent(a: f32, b: f32, angle: f32) -> f32 {
    let lo = angle.min(0.0);
    let hi = angle.max(0.0);
    let at = |t: f32| (a * t.cos() + b * t.sin()).abs();
    let phase = b.atan2(a);
    let period = std::f32::consts::PI;
    let critical = phase + ((lo - phase) / period).ceil() * period;
    let mut result = at(lo).max(at(hi));
    if critical <= hi {
        result = result.max(a.hypot(b));
    }
    result
}

// For an anisotropic ellipse, squared normalized radius is
// A + B*cos(2t) + C*sin(2t). Its maxima have period pi.
fn rotated_eye_extent(point: (f32, f32), radius: (f32, f32), angle: f32) -> f32 {
    let (u, v) = point;
    let ix = 1.0 / radius.0.powi(2);
    let iy = 1.0 / radius.1.powi(2);
    let b = 0.5 * (u * u - v * v) * (ix - iy);
    let c = u * v * (iy - ix);
    let lo = angle.min(0.0);
    let hi = angle.max(0.0);
    let at = |t: f32| {
        let (sn, cs) = t.sin_cos();
        ((u * cs - v * sn) / radius.0).powi(2) + ((u * sn + v * cs) / radius.1).powi(2)
    };
    let phase = 0.5 * c.atan2(b);
    let period = std::f32::consts::PI;
    let critical = phase + ((lo - phase) / period).ceil() * period;
    let mut maximum = at(lo).max(at(hi));
    if critical <= hi {
        maximum = maximum.max(at(critical));
    }
    maximum.sqrt()
}

#[derive(Clone)]
struct EyeWidthMaterial {
    center: (f32, f32),
    inverse: [f32; 4],
    minimum_radius: f32,
    aperture: Vec<(f32, f32)>,
    support_inner: f32,
    transition_end: f32,
}

fn feature_polygon_signed_distance(points: &[(f32, f32)], point: (f32, f32)) -> f32 {
    let (x, y) = point;
    let mut inside = false;
    let mut distance = f32::INFINITY;
    for (a, b) in points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
    {
        if (a.1 > y) != (b.1 > y) && x < (b.0 - a.0) * (y - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
        let edge = (b.0 - a.0, b.1 - a.1);
        let length = edge.0 * edge.0 + edge.1 * edge.1;
        let t = if length > 0.0 {
            ((x - a.0) * edge.0 + (y - a.1) * edge.1) / length
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        distance = distance.min((x - a.0 - edge.0 * t).hypot(y - a.1 - edge.1 * t));
    }
    if inside {
        -distance
    } else {
        distance
    }
}
impl EyeWidthMaterial {
    fn ellipse_coordinate(&self, p: (f32, f32)) -> (f32, f32) {
        let d = (p.0 - self.center.0, p.1 - self.center.1);
        (
            self.inverse[0] * d.0 + self.inverse[1] * d.1,
            self.inverse[2] * d.0 + self.inverse[3] * d.1,
        )
    }
    fn ellipse_distance(&self, p: (f32, f32)) -> f32 {
        let q = self.ellipse_coordinate(p);
        (q.0.hypot(q.1) - 1.0) * self.minimum_radius
    }
    fn surrounding_motion(&self, p: (f32, f32)) -> f32 {
        let ellipse = self.ellipse_distance(p);
        if ellipse >= self.transition_end {
            return 1.0;
        }
        // The observation is the iris projection intersected with the
        // actually visible source aperture. Hidden top/bottom arcs do not
        // become protected eyelid material. The 1.5px halo is sampling guard.
        let distance = ellipse.max(feature_polygon_signed_distance(&self.aperture, p));
        smoothstep(1.5, self.transition_end, distance)
    }
}

fn eye_width_material(
    p: &Value,
    name: &str,
    source: &RgbaImage,
    frame: FeatureFrame,
    radius: (f32, f32),
    forward_scale: f32,
    axis_vertical: bool,
) -> Result<Option<(EyeWidthMaterial, (f32, f32))>, String> {
    let Some(raw) = p
        .get("eyeGeometry")
        .and_then(|g| g.get(name))
        .and_then(|g| g.get("iris"))
    else {
        return Ok(None);
    };
    let point = |v: &Value| -> Option<(f32, f32)> {
        let x = v.get("x")?.as_f64()?;
        let y = v.get("y")?.as_f64()?;
        (x.is_finite() && y.is_finite() && (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y))
            .then_some((
                x as f32 * source.width as f32,
                y as f32 * source.height as f32,
            ))
    };
    let center =
        point(&raw["center"]).ok_or_else(|| format!("{name}: missing measured iris center"))?;
    let ring = raw["boundary"]
        .as_array()
        .ok_or_else(|| format!("{name}: iris boundary must contain four model points"))?;
    if ring.len() != 4 {
        return Err(format!(
            "{name}: iris boundary must contain four model points"
        ));
    }
    let points: Vec<_> = ring
        .iter()
        .map(point)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("{name}: invalid iris boundary coordinate"))?;
    let u = (
        (points[0].0 - points[2].0) * 0.5,
        (points[0].1 - points[2].1) * 0.5,
    );
    let v = (
        (points[3].0 - points[1].0) * 0.5,
        (points[3].1 - points[1].1) * 0.5,
    );
    let det = u.0 * v.1 - u.1 * v.0;
    let trace = u.0 * u.0 + u.1 * u.1 + v.0 * v.0 + v.1 * v.1;
    let minimum_radius = ((trace - (trace * trace - 4.0 * det * det).max(0.0).sqrt()) * 0.5)
        .max(0.0)
        .sqrt();
    if !minimum_radius.is_finite() || minimum_radius < 0.5 || det.abs() < 0.01 {
        return Err(format!("{name}: degenerate iris projection"));
    }
    let aperture = symmetry_feature(p, name, source)
        .ok_or_else(|| format!("{name}: missing source aperture"))?;
    let mut material = EyeWidthMaterial {
        center,
        inverse: [v.1 / det, -v.0 / det, -u.1 / det, u.0 / det],
        minimum_radius,
        aperture,
        support_inner: 0.0,
        transition_end: 0.0,
    };
    let aligned = |point: (f32, f32)| {
        let d = (point.0 - frame.center.0, point.1 - frame.center.1);
        (
            d.0 * frame.across.0 + d.1 * frame.across.1,
            -d.0 * frame.across.1 + d.1 * frame.across.0,
        )
    };
    let to_world = |point: (f32, f32)| {
        (
            frame.center.0 + point.0 * frame.across.0 - point.1 * frame.across.1,
            frame.center.1 + point.0 * frame.across.1 + point.1 * frame.across.0,
        )
    };
    let points: Vec<_> = material.aperture.iter().copied().map(aligned).collect();
    let source_extent = points
        .iter()
        .map(|&(x, y)| (x / radius.0).hypot(y / radius.1))
        .fold(0.0f32, f32::max);
    let expansion = forward_scale.exp();
    let requested_extent = points
        .iter()
        .map(|&(x, y)| {
            let (x, y) = if axis_vertical { (x, y * expansion) } else { (x * expansion, y) };
            (x / radius.0).hypot(y / radius.1)
        })
        .fold(source_extent, f32::max);
    let outer = requested_extent + (1.0 - source_extent);
    material.support_inner = (requested_extent / outer).powi(2);
    let mut clearance = f32::INFINITY;
    for side in [-1.0, 1.0] {
        let &canthus = points
            .iter()
            .max_by(|a, b| {
                let av = if axis_vertical { side * a.1 } else { side * a.0 };
                let bv = if axis_vertical { side * b.1 } else { side * b.0 };
                av.total_cmp(&bv)
            })
            .unwrap();
        let from = to_world(canthus);
        let to = if axis_vertical {
            to_world((canthus.0, canthus.1 * expansion))
        } else {
            to_world((canthus.0 * expansion, canthus.1))
        };
        let a = material.ellipse_coordinate(from);
        let b = material.ellipse_coordinate(to);
        let d = (b.0 - a.0, b.1 - a.1);
        let length = d.0 * d.0 + d.1 * d.1;
        let t = if length > 0.0 {
            -(a.0 * d.0 + a.1 * d.1) / length
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        clearance = clearance.min(((a.0 + t * d.0).hypot(a.1 + t * d.1) - 1.0) * minimum_radius);
    }
    if clearance <= 1.75 {
        return Err(format!("{name}: requested canthus trajectory conflicts with visible iris sampling guard; requires occlusion reconstruction"));
    }
    material.transition_end = 1.5 + (clearance - 1.5) * 0.5;
    Ok(Some((material, (radius.0 * outer, radius.1 * outer))))
}

#[derive(Clone)]
struct LocalWarp {
    center: (f32, f32),
    across: (f32, f32),
    radius: (f32, f32),
    scale: (f32, f32),
    translation: (f32, f32),
    angle: f32,
    preserve_eyes: bool,
    whole_eye_size: bool,
    whole_eye_tilt: bool,
    whole_eye_spacing: bool,
    eye_width_material: Option<std::sync::Arc<EyeWidthMaterial>>,
    whole_eye_inner: f32,
    eye_size_brows: Option<std::sync::Arc<Vec<FeatureBrowAnchor>>>,
}

/// A continuous pixel-space motion domain. Fixed obstruction texels and zero
/// selection/protection/region pixels get a two-pixel guard for bilinear donors.
/// A distance transition makes hard boundaries part of the velocity rather than
/// cutting the integrated mapping afterward. The field is bilinear, not a proof
/// of a global no-fold bound; the final f32 mapper is checked by regressions.
struct FeatureMotionDomain {
    width: usize,
    height: usize,
    weights: Vec<f32>,
    gradient: f32,
}

impl FeatureMotionDomain {
    fn new(source: &RgbaImage, stationary: &[bool], coverage: impl Fn(f32, f32) -> f32) -> Self {
        let width = source.width as usize;
        let height = source.height as usize;
        let mut weights = Vec::with_capacity(width * height);
        let mut distance = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let weight = coverage(x as f32, y as f32).clamp(0.0, 1.0);
                let fixed = weight <= 0.0
                    || stationary.get(i).copied().unwrap_or(false)
                    || x == 0
                    || y == 0
                    || x + 1 == width
                    || y + 1 == height;
                weights.push(if fixed { 0.0 } else { weight });
                distance.push(if fixed { 0.0 } else { width.max(height) as f32 });
            }
        }
        // Eight-neighbour chamfer distances are sufficient for the conservative
        // donor guard; interpolation makes the resulting motion field continuous.
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                if x > 0 {
                    distance[i] = distance[i].min(distance[i - 1] + 1.0);
                }
                if y > 0 {
                    distance[i] = distance[i].min(distance[i - width] + 1.0);
                    if x > 0 {
                        distance[i] =
                            distance[i].min(distance[i - width - 1] + std::f32::consts::SQRT_2);
                    }
                    if x + 1 < width {
                        distance[i] =
                            distance[i].min(distance[i - width + 1] + std::f32::consts::SQRT_2);
                    }
                }
            }
        }
        for y in (0..height).rev() {
            for x in (0..width).rev() {
                let i = y * width + x;
                if x + 1 < width {
                    distance[i] = distance[i].min(distance[i + 1] + 1.0);
                }
                if y + 1 < height {
                    distance[i] = distance[i].min(distance[i + width] + 1.0);
                    if x > 0 {
                        distance[i] =
                            distance[i].min(distance[i + width - 1] + std::f32::consts::SQRT_2);
                    }
                    if x + 1 < width {
                        distance[i] =
                            distance[i].min(distance[i + width + 1] + std::f32::consts::SQRT_2);
                    }
                }
                weights[i] *= smoothstep(2.0, 10.0, distance[i]);
            }
        }
        let mut gradient = 0.0f32;
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let dx = if x + 1 < width {
                    (weights[i + 1] - weights[i]).abs()
                } else {
                    0.0
                };
                let dy = if y + 1 < height {
                    (weights[i + width] - weights[i]).abs()
                } else {
                    0.0
                };
                gradient = gradient.max(dx.hypot(dy));
            }
        }
        Self {
            width,
            height,
            weights,
            gradient,
        }
    }

    fn coverage(&self, x: f32, y: f32) -> f32 {
        if x < 0.0 || y < 0.0 || x >= (self.width - 1) as f32 || y >= (self.height - 1) as f32 {
            return 0.0;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let tx = x - x.floor();
        let ty = y - y.floor();
        let i = y0 * self.width + x0;
        (self.weights[i] * (1.0 - tx) + self.weights[i + 1] * tx) * (1.0 - ty)
            + (self.weights[i + self.width] * (1.0 - tx) + self.weights[i + self.width + 1] * tx)
                * ty
    }
}

/// Interpret the compact feature edits as a stationary velocity field. A
/// direct x + v(x) warp can fold where the brow field approaches fixed eyes;
/// integrating the same requested field changes the geometry continuously.
#[derive(Clone, Copy, PartialEq)]
enum FeatureFlowSubset {
    All,
    WidthMaterial,
    Other,
    EyeSpacing,
}

fn feature_flow_velocity(
    subset: FeatureFlowSubset,
    point: (f32, f32),
    changes: &[LocalWarp],
    fixed_eyes: &[FeatureFrame],
    post_warps: &[SymmetryWarp],
    coverage: &impl Fn(f32, f32) -> f32,
) -> (f32, f32) {
    let mut velocity = (0.0, 0.0);
    let selection = coverage(point.0, point.1);
    if selection <= 0.0 {
        return velocity;
    }
    for warp in changes {
        if subset == FeatureFlowSubset::WidthMaterial && warp.eye_width_material.is_none()
            || subset == FeatureFlowSubset::EyeSpacing && !warp.whole_eye_spacing
            || subset == FeatureFlowSubset::Other
                && (warp.eye_width_material.is_some() || warp.whole_eye_spacing)
        {
            continue;
        }
        let delta = (point.0 - warp.center.0, point.1 - warp.center.1);
        let u = delta.0 * warp.across.0 + delta.1 * warp.across.1;
        let v = -delta.0 * warp.across.1 + delta.1 * warp.across.0;
        let d = (u / warp.radius.0).powi(2) + (v / warp.radius.1).powi(2);
        if d >= 1.0 {
            continue;
        }
        let eye_motion = if warp.preserve_eyes {
            fixed_eyes.iter().fold(1.0, |weight, eye| {
                weight * eye.surrounding_motion(point.0, point.1)
            })
        } else {
            1.0
        };
        // Whole-eye size, tilt and spacing have constant generators over the
        // detected organ and its requested trajectory. Other feature fields
        // retain their existing compact support within their ordered stage.
        let support = if let Some(material) = &warp.eye_width_material {
            1.0 - smoothstep(material.support_inner, 1.0, d)
        } else if warp.whole_eye_size || warp.whole_eye_tilt || warp.whole_eye_spacing {
            1.0 - smoothstep(warp.whole_eye_inner, 1.0, d)
        } else {
            (1.0 - d).powi(2)
        };
        let brow_motion = warp
            .eye_size_brows
            .as_ref()
            .map(|anchors| {
                anchors.iter().fold(1.0, |weight, anchor| {
                    weight * anchor.surrounding_motion(point)
                })
            })
            .unwrap_or(1.0);
        let iris_motion = warp
            .eye_width_material
            .as_ref()
            .map(|m| m.surrounding_motion(point))
            .unwrap_or(1.0);
        let weight = support * selection * eye_motion * brow_motion * iris_motion;
        let a = warp.angle * weight;
        // A stationary field needs the infinitesimal rotation generator.
        // Finite rotation displacement adds radial contraction on integration.
        let du = (u * warp.scale.0 + warp.translation.0) * weight - v * a;
        let dv = (v * warp.scale.1 + warp.translation.1) * weight + u * a;
        velocity.0 += du * warp.across.0 - dv * warp.across.1;
        velocity.1 += du * warp.across.1 + dv * warp.across.0;
    }
    let post = if matches!(
        subset,
        FeatureFlowSubset::WidthMaterial | FeatureFlowSubset::EyeSpacing
    ) {
        point
    } else {
        symmetry_inverse(point.0, point.1, post_warps)
    };
    velocity.0 += (post.0 - point.0) * selection;
    velocity.1 += (post.1 - point.1) * selection;
    velocity
}

fn feature_flow_steps(
    changes: &[LocalWarp],
    fixed_eyes: &[FeatureFrame],
    post_warps: &[SymmetryWarp],
    coverage_gradient: f32,
) -> usize {
    let eye_gradient: f32 = fixed_eyes
        .iter()
        .map(|eye| 1.5 / ((eye.width * 0.28).clamp(4.0, 24.0) - 2.0))
        .sum();
    let gradient_bound: f32 = changes
        .iter()
        .map(|warp| {
            let speed = (warp.radius.0 * warp.scale.0.abs() + warp.translation.0.abs())
                .hypot(warp.radius.1 * warp.scale.1.abs() + warp.translation.1.abs())
                + warp.radius.0.max(warp.radius.1) * warp.angle.abs();
            let support_gradient = (if let Some(material) = &warp.eye_width_material {
                3.0 / (1.0 - material.support_inner)
            } else if warp.whole_eye_size || warp.whole_eye_tilt || warp.whole_eye_spacing {
                3.0 / (1.0 - warp.whole_eye_inner)
            } else {
                1.54
            }) / warp.radius.0.min(warp.radius.1);
            warp.scale.0.abs().max(warp.scale.1.abs())
                + warp.angle.abs()
                + speed
                    * (support_gradient
                        + coverage_gradient
                        + warp
                            .eye_width_material
                            .as_ref()
                            .map(|m| 1.5 / (m.transition_end - 1.5))
                            .unwrap_or(0.0)
                        + warp
                            .eye_size_brows
                            .as_ref()
                            .map(|anchors| anchors.len() as f32 * 0.75)
                            .unwrap_or(0.0)
                        + if warp.preserve_eyes {
                            eye_gradient
                        } else {
                            0.0
                        })
        })
        .sum();
    let gradient_bound = gradient_bound
        + post_warps
            .iter()
            .map(|warp| {
                let speed = (warp.target.0 - warp.source.0).hypot(warp.target.1 - warp.source.1);
                speed * (2.4 / warp.radius.0.min(warp.radius.1) + coverage_gradient)
            })
            .sum::<f32>();
    // At least 32 RK4 steps, increasing when the compact field is steep.
    // This is a numerical budget, not a reduction of the requested slider.
    (gradient_bound * 4.0).ceil().max(32.0) as usize
}

fn integrate_feature_flow(
    point: (f32, f32),
    changes: &[LocalWarp],
    fixed_eyes: &[FeatureFrame],
    post_warps: &[SymmetryWarp],
    coverage: &impl Fn(f32, f32) -> f32,
    steps: usize,
    direction: f32,
) -> (f32, f32) {
    if !changes
        .iter()
        .any(|w| w.eye_width_material.is_some() || w.whole_eye_spacing)
    {
        return integrate_feature_flow_stage(
            FeatureFlowSubset::All,
            point,
            changes,
            fixed_eyes,
            post_warps,
            coverage,
            steps,
            direction,
        );
    }
    // Fixed organ order: width reshapes source aperture, size/pose/other
    // fields transport it, then spacing translates the resulting whole eye.
    // Reverse sampling undoes those operators in reverse order and samples
    // the image once. Every stage tends continuously to identity at zero.
    let order = if direction < 0.0 {
        [
            FeatureFlowSubset::WidthMaterial,
            FeatureFlowSubset::Other,
            FeatureFlowSubset::EyeSpacing,
        ]
    } else {
        [
            FeatureFlowSubset::EyeSpacing,
            FeatureFlowSubset::Other,
            FeatureFlowSubset::WidthMaterial,
        ]
    };
    let mut result = point;
    for subset in order {
        result = integrate_feature_flow_stage(
            subset, result, changes, fixed_eyes, post_warps, coverage, steps, direction,
        );
    }
    result
}

fn integrate_feature_flow_stage(
    subset: FeatureFlowSubset,
    point: (f32, f32),
    changes: &[LocalWarp],
    fixed_eyes: &[FeatureFrame],
    post_warps: &[SymmetryWarp],
    coverage: &impl Fn(f32, f32) -> f32,
    steps: usize,
    direction: f32,
) -> (f32, f32) {
    let velocity =
        |point| feature_flow_velocity(subset, point, changes, fixed_eyes, post_warps, coverage);
    let initial = velocity(point);
    if initial.0 == 0.0 && initial.1 == 0.0 {
        return point;
    }
    let dt = direction / steps as f32;
    let advance = |point: (f32, f32), delta: (f32, f32), time: f32| {
        (point.0 + delta.0 * time, point.1 + delta.1 * time)
    };
    let mut current = point;
    // Photographic coordinates can be hundreds or thousands of pixels while
    // each RK step is only hundredths of a pixel. Carry the low part lost
    // by f32 addition instead of accumulating one rounding error per step.
    let mut carry = (0.0f32, 0.0f32);
    for _ in 0..steps {
        let k1 = velocity(current);
        let k2 = velocity(advance(current, k1, dt * 0.5));
        let k3 = velocity(advance(current, k2, dt * 0.5));
        let k4 = velocity(advance(current, k3, dt));
        let step = (
            dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
            dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
        );
        let corrected = (step.0 - carry.0, step.1 - carry.1);
        let next = (current.0 + corrected.0, current.1 + corrected.1);
        carry = (
            (next.0 - current.0) - corrected.0,
            (next.1 - current.1) - corrected.1,
        );
        current = next;
    }
    current
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
    let stationary_masks: Vec<_> = if matches!(kind, "face.adjust" | "feature.adjust") {
        ["hair", "ears", "accessories"]
            .iter()
            .filter_map(|name| p.get("semanticMasks").and_then(|m| m.get(name)))
            .map(PreparedMask::new)
            .collect()
    } else {
        Vec::new()
    };
    let mut symmetry_warps = if matches!(kind, "face.adjust" | "feature.adjust") {
        read_symmetry_warps(
            p,
            source,
            number(values, "symmetry", 0.0).clamp(0.0, 100.0) / 100.0,
        )
    } else {
        Vec::new()
    };
    let eye_size_brows = std::sync::Arc::new(if semantic_features {
        ["leftBrow", "rightBrow"]
            .iter()
            .filter_map(|name| symmetry_feature(p, name, source))
            .map(FeatureBrowAnchor::new)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    });
    let mut eye_width_error: Option<String> = None;
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
            let (mut frame, radius, translation) = if semantic_features {
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
            let mut radius = radius;
            let whole_eye_inner = if semantic_features && key == "eyeSpacing" {
                let name = if cx < 0.5 { "leftEye" } else { "rightEye" };
                let points = symmetry_feature(p, name, source).unwrap();
                let aligned: Vec<_> = points
                    .iter()
                    .map(|&(x, y)| {
                        let dx = x - frame.center.0;
                        let dy = y - frame.center.1;
                        (
                            dx * frame.across.0 + dy * frame.across.1,
                            -dx * frame.across.1 + dy * frame.across.0,
                        )
                    })
                    .collect();
                let source_extent = aligned
                    .iter()
                    .map(|&(u, v)| (u / radius.0).hypot(v / radius.1))
                    .fold(0.0f32, f32::max);
                // Enclose the preceding organ shape, size and rotation.
                // Independent coordinate extrema give a conservative bound
                // for the rotated aperture plus the translation sweep.
                let width_scale = (EYE_WIDTH_LOG_SCALE * percent(values, "eyeWidth")).exp();
                let size_scale = (EYE_SIZE_LOG_SCALE * percent(values, "eyeSize"))
                    .max(0.0)
                    .exp();
                let sign = if cx < 0.5 { -1.0 } else { 1.0 };
                let theta = -sign * EYE_TILT_RADIANS * percent(values, "eyeTilt");
                let forward = (-translation.0 * amount, -translation.1 * amount);
                let mut requested_extent = source_extent;
                for &(u, v) in &aligned {
                    for width in [1.0, width_scale] {
                        let x = angular_component_extent(u * width, -v, theta) * size_scale
                            + forward.0.abs() * 0.5;
                        let y = angular_component_extent(v, u * width, theta) * size_scale
                            + forward.1.abs() * 0.5;
                        requested_extent = requested_extent.max((x / radius.0).hypot(y / radius.1));
                    }
                }
                // The plateau encloses both endpoints of the requested
                // translation, hence their whole straight swept segment.
                // Center its support at the trajectory midpoint while the
                // constant translation generator keeps the original request.
                frame.center.0 += (forward.0 * frame.across.0 - forward.1 * frame.across.1) * 0.5;
                frame.center.1 += (forward.0 * frame.across.1 + forward.1 * frame.across.0) * 0.5;
                let outer = requested_extent + (1.0 - source_extent);
                radius.0 *= outer;
                radius.1 *= outer;
                (requested_extent / outer).powi(2)
            } else if semantic_features && matches!(key, "eyeSize" | "eyeTilt") {
                let name = if cx < 0.5 { "leftEye" } else { "rightEye" };
                let points = symmetry_feature(p, name, source).unwrap();
                let aligned: Vec<_> = points
                    .iter()
                    .map(|&(x, y)| {
                        let dx = x - frame.center.0;
                        let dy = y - frame.center.1;
                        (
                            dx * frame.across.0 + dy * frame.across.1,
                            -dx * frame.across.1 + dy * frame.across.0,
                        )
                    })
                    .collect();
                let source_extent = aligned
                    .iter()
                    .map(|&(u, v)| ((u / radius.0).powi(2) + (v / radius.1).powi(2)).sqrt())
                    .fold(0.0f32, f32::max);
                let prior_width = if p
                    .get("eyeGeometry")
                    .and_then(|g| g.get(name))
                    .and_then(|g| g.get("iris"))
                    .is_some()
                {
                    (EYE_WIDTH_LOG_SCALE * percent(values, "eyeWidth")).exp()
                } else {
                    1.0
                };
                let sign = if cx < 0.5 { -1.0 } else { 1.0 };
                let theta = -sign * EYE_TILT_RADIANS * percent(values, "eyeTilt");
                let scale = (EYE_SIZE_LOG_SCALE * percent(values, "eyeSize"))
                    .max(0.0)
                    .exp();
                let mut swept_extent = source_extent;
                for &(u, v) in &aligned {
                    for width in [1.0, prior_width] {
                        swept_extent =
                            swept_extent.max(rotated_eye_extent((u * width, v), radius, theta));
                    }
                }
                // Size and tilt share one source-organ platform, covering
                // the full requested similarity trajectory. Preserve the
                // original transition width outside that analytic envelope.
                let requested_extent = swept_extent * scale;
                let outer = requested_extent + (1.0 - source_extent);
                radius.0 *= outer;
                radius.1 *= outer;
                (requested_extent / outer).powi(2)
            } else {
                0.0
            };
            let eye_width_material_opt = if semantic_features && key == "eyeWidth" {
                let name = if cx < 0.5 { "leftEye" } else { "rightEye" };
                match eye_width_material(p, name, source, frame, radius, -sx * amount, false) {
                    Ok(Some((material, expanded))) => {
                        radius = expanded;
                        Some(std::sync::Arc::new(material))
                    }
                    Ok(None) => None,
                    Err(error) => {
                        eye_width_error = Some(error);
                        return;
                    }
                }
            } else {
                None
            };
            let eye_height_material = if semantic_features && key == "eyeHeight" {
                let name = if cx < 0.5 { "leftEye" } else { "rightEye" };
                match eye_width_material(p, name, source, frame, radius, -sy * amount, true) {
                    Ok(Some((material, expanded))) => {
                        radius = expanded;
                        Some(std::sync::Arc::new(material))
                    }
                    Ok(None) => None,
                    Err(error) => {
                        eye_width_error = Some(error);
                        return;
                    }
                }
            } else {
                None
            };
            let eye_material = eye_width_material_opt.or(eye_height_material);
            let protect_width_brows = eye_material.is_some();
            changes.push(LocalWarp {
                center: frame.center,
                across: frame.across,
                radius,
                scale: (sx * amount, sy * amount),
                translation: (translation.0 * amount, translation.1 * amount),
                angle: angle * amount,
                preserve_eyes: key.starts_with("brow"),
                whole_eye_size: semantic_features && key == "eyeSize",
                whole_eye_tilt: semantic_features && key == "eyeTilt",
                whole_eye_spacing: semantic_features && key == "eyeSpacing",
                eye_width_material: eye_material,
                whole_eye_inner,
                eye_size_brows: (semantic_features
                    && (matches!(key, "eyeSize" | "eyeTilt" | "eyeSpacing")
                        || protect_width_brows))
                    .then(|| eye_size_brows.clone()),
            });
        }
    };
    for (cx, sign) in [(0.32, -1.0), (0.68, 1.0)] {
        add(
            "eyeSize",
            cx,
            0.39,
            0.20,
            0.13,
            -EYE_SIZE_LOG_SCALE,
            -EYE_SIZE_LOG_SCALE,
            0.0,
            0.0,
            0.0,
        );
        add(
            "eyeWidth",
            cx,
            0.39,
            0.2,
            0.13,
            -EYE_WIDTH_LOG_SCALE,
            0.0,
            0.0,
            0.0,
            0.0,
        );
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
            sign * EYE_TILT_RADIANS,
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
            // Nostrils are a small, high-contrast target.  The previous
            // coefficients produced only sub-pixel contour travel on the
            // photographic masks, so the control appeared inert even though
            // the field was non-zero.  Keep the support local while giving
            // the bilateral opening a visible, reversible response.
            -0.4,
            -0.25,
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

    if let Some(error) = eye_width_error {
        return Err(ImageError::InvalidOperation(error));
    }

    // Color-only edits do not need a full-image frozen raster or inverse
    // sampling. Evaluate semantic protection only where local_color changed
    // bytes; the graph still applies opacity/selection blending once.
    if changes.is_empty() && symmetry_warps.is_empty() {
        let mut output = crate::color::local_color(source, p, kind == "clothing.adjust");
        if matches!(kind, "face.adjust" | "feature.adjust") {
            let selection = p
                .get("mask")
                .filter(|v| v.is_object())
                .map(PreparedMask::new);
            let protection = p
                .get("protectMask")
                .filter(|v| v.is_object())
                .map(PreparedMask::new);
            for (i, (edited, original)) in output
                .pixels
                .chunks_exact_mut(4)
                .zip(source.pixels.chunks_exact(4))
                .enumerate()
            {
                if edited == original {
                    continue;
                }
                let nx = ((i % source.width as usize) as f32 + 0.5) / source.width as f32;
                let ny = ((i / source.width as usize) as f32 + 0.5) / source.height as f32;
                let frozen = stationary_masks
                    .iter()
                    .any(|mask| mask.coverage(nx, ny, source.width, source.height) >= 0.5)
                    || selection.as_ref().is_some_and(|mask| {
                        mask.coverage(nx, ny, source.width, source.height) <= 0.0
                    })
                    || protection.as_ref().is_some_and(|mask| {
                        mask.coverage(nx, ny, source.width, source.height) >= 1.0
                    });
                if frozen {
                    edited.copy_from_slice(original);
                }
            }
        }
        return Ok(output);
    }

    let eyebrow_eye_anchors: Vec<_> =
        if semantic_features && changes.iter().any(|warp| warp.preserve_eyes) {
            ["leftEye", "rightEye"]
                .iter()
                .filter_map(|name| feature_frame(p, name, source))
                .collect()
        } else {
            Vec::new()
        };

    let stationary: Vec<bool> = if stationary_masks.is_empty() {
        Vec::new()
    } else {
        (0..source.width as usize * source.height as usize)
            .map(|i| {
                let x = (i % source.width as usize) as f32 + 0.5;
                let y = (i / source.width as usize) as f32 + 0.5;
                stationary_masks.iter().any(|m| {
                    m.coverage(
                        x / source.width as f32,
                        y / source.height as f32,
                        source.width,
                        source.height,
                    ) >= 0.5
                })
            })
            .collect()
    };
    // A polygon/ellipse selection is also part of the displacement field.
    // Applying it only in `mask::composite` leaves the hidden rectangular
    // render free to pull background texels into the face edge.  Scaling the
    // local displacement by the same coverage keeps the field continuous at
    // the selection boundary and makes the fallback safe for manual masks.
    let selection_mask = p.get("mask").filter(|value| value.is_object());
    let flow_selection = selection_mask.map(PreparedMask::new);
    let flow_protection = p
        .get("protectMask")
        .filter(|value| value.is_object())
        .map(PreparedMask::new);
    // Use one geometry method at zero and nonzero brow values, so adding
    // a tiny brow edit cannot switch an existing eye/nose/mouth adjustment.
    let use_feature_flow = semantic_features;
    let raw_flow_coverage = |x: f32, y: f32| {
        let nx = x / source.width as f32;
        let ny = y / source.height as f32;
        if !region.contains(nx, ny) {
            return 0.0;
        }
        flow_selection
            .as_ref()
            .map(|mask| mask.coverage(nx, ny, source.width, source.height))
            .unwrap_or(1.0)
            * (1.0
                - flow_protection
                    .as_ref()
                    .map(|mask| mask.coverage(nx, ny, source.width, source.height))
                    .unwrap_or(0.0))
    };
    let flow_domain =
        use_feature_flow.then(|| FeatureMotionDomain::new(source, &stationary, raw_flow_coverage));
    let flow_coverage = |x: f32, y: f32| {
        flow_domain
            .as_ref()
            .map(|domain| domain.coverage(x, y))
            .unwrap_or(1.0)
    };
    let flow_steps = feature_flow_steps(
        &changes,
        &eyebrow_eye_anchors,
        &symmetry_warps,
        flow_domain
            .as_ref()
            .map(|domain| domain.gradient)
            .unwrap_or(0.0),
    );
    // Facial makeup is defined in the input feature coordinates. Color it
    // before sampling geometry so an edited lip/eyebrow carries its makeup
    // with it instead of leaving pigment at the pre-warp polygon location.
    let mut facial_input = matches!(kind, "face.adjust" | "feature.adjust")
        .then(|| crate::color::local_color(source, p, false));
    if let Some(input) = &mut facial_input {
        for (i, &frozen) in stationary.iter().enumerate() {
            if frozen {
                input.pixels[i * 4..i * 4 + 4].copy_from_slice(&source.pixels[i * 4..i * 4 + 4]);
            }
        }
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
    let feature_mapping = |x: f32, y: f32| {
        let destination = y as usize * source.width as usize + x as usize;
        if stationary.get(destination).copied().unwrap_or(false) {
            return (x, y);
        }
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

        if use_feature_flow {
            return integrate_feature_flow(
                (x, y),
                &changes,
                &eyebrow_eye_anchors,
                &symmetry_warps,
                &flow_coverage,
                flow_steps,
                1.0,
            );
        }

        for warp in &changes {
            if use_feature_flow {
                break;
            }
            let delta = (x - warp.center.0, y - warp.center.1);
            let u = delta.0 * warp.across.0 + delta.1 * warp.across.1;
            let v = -delta.0 * warp.across.1 + delta.1 * warp.across.0;
            let dx = u / warp.radius.0;
            let dy = v / warp.radius.1;
            let d = dx * dx + dy * dy;
            if d >= 1.0 {
                continue;
            }
            let eye_motion = if warp.preserve_eyes {
                eyebrow_eye_anchors
                    .iter()
                    .fold(1.0, |weight, eye| weight * eye.surrounding_motion(x, y))
            } else {
                1.0
            };
            let weight = (1.0 - d).powi(2) * selection_weight * eye_motion;
            let a = warp.angle * weight;
            let du = (u * warp.scale.0 + warp.translation.0) * weight + u * (a.cos() - 1.0)
                - v * a.sin();
            let dv = (v * warp.scale.1 + warp.translation.1) * weight
                + u * a.sin()
                + v * (a.cos() - 1.0);
            ux += (du * warp.across.0 - dv * warp.across.1) / (source.width as f32 * region.width);
            uy +=
                (du * warp.across.1 + dv * warp.across.0) / (source.height as f32 * region.height);
        }
        if !symmetry_warps.is_empty() {
            let (sx, sy) = symmetry_inverse(x, y, &symmetry_warps);
            let edge_distance = rx.min(1.0 - rx).min(ry).min(1.0 - ry);
            let weight = smoothstep(0.0, 0.06, edge_distance) * selection_weight;
            ux += (sx - x) / (source.width as f32 * region.width) * weight;
            uy += (sy - y) / (source.height as f32 * region.height) * weight;
        }
        let sx = (region.x + ux * region.width) * source.width as f32;
        let sy = (region.y + uy * region.height) * source.height as f32;
        // A stationary obstruction must not be cloned into neighboring
        // skin. Check all nonzero bilinear contributors, not only the
        // nearest sampling pixel or the frozen output location.
        if !stationary.is_empty() && ((sx - x).abs() + (sy - y).abs()) > 1e-5 {
            let bx = sx.clamp(0.0, source.width as f32 - 1.0);
            let by = sy.clamp(0.0, source.height as f32 - 1.0);
            let (x0, y0) = (bx.floor() as usize, by.floor() as usize);
            let (tx, ty) = (bx - bx.floor(), by - by.floor());
            for (xx, yy, weight) in [
                (x0, y0, (1.0 - tx) * (1.0 - ty)),
                ((x0 + 1).min(source.width as usize - 1), y0, tx * (1.0 - ty)),
                (
                    x0,
                    (y0 + 1).min(source.height as usize - 1),
                    (1.0 - tx) * ty,
                ),
                (
                    (x0 + 1).min(source.width as usize - 1),
                    (y0 + 1).min(source.height as usize - 1),
                    tx * ty,
                ),
            ] {
                if weight > 1e-6 && stationary[yy * source.width as usize + xx] {
                    return (x, y);
                }
            }
        }
        (sx, sy)
    };
    #[cfg(test)]
    if use_feature_flow {
        tests::audit_photographic_feature_flow(
            &feature_mapping,
            &changes,
            &eyebrow_eye_anchors,
            &symmetry_warps,
            &flow_coverage,
            flow_steps,
            source,
            p,
            &stationary,
        );
    }
    let mut result = remap(
        sampling_source,
        source.width,
        source.height,
        true,
        feature_mapping,
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

#[derive(Clone, Copy)]
struct RepairStroke {
    x: f32,
    y: f32,
    radius: f32,
    strength: f32,
    donor: Option<(f32, f32)>,
}

/// Source eligibility is separate from the output selection. A brush selection
/// can describe only the defect; automatic donors must remain outside all
/// marked defects and protected pixels. An explicitly chosen source may read
/// protected pixels: protection still freezes the output in the compositor.
/// No full-image eligibility raster is needed.
struct RepairSources<'a> {
    image: &'a RgbaImage,
    protection: Option<PreparedMask<'a>>,
    strokes: &'a [RepairStroke],
}

impl RepairSources<'_> {
    fn visible_sample(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        if x < 0.0
            || y < 0.0
            || x > self.image.width as f32 - 1.0
            || y > self.image.height as f32 - 1.0
        {
            return None;
        }
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (tx, ty) = (x - x.floor(), y - y.floor());
        for (px, py, weight) in [
            (x0, y0, (1.0 - tx) * (1.0 - ty)),
            ((x0 + 1).min(self.image.width - 1), y0, tx * (1.0 - ty)),
            (x0, (y0 + 1).min(self.image.height - 1), (1.0 - tx) * ty),
            (
                (x0 + 1).min(self.image.width - 1),
                (y0 + 1).min(self.image.height - 1),
                tx * ty,
            ),
        ] {
            if weight <= 1e-6 {
                continue;
            }
            let index = (py as usize * self.image.width as usize + px as usize) * 4;
            if self.image.pixels[index + 3] == 0
                || self.protection.as_ref().is_some_and(|mask| {
                    mask.coverage(
                        (px as f32 + 0.5) / self.image.width as f32,
                        (py as f32 + 0.5) / self.image.height as f32,
                        self.image.width,
                        self.image.height,
                    ) > 1e-6
                })
            {
                return None;
            }
        }
        let value = sample(self.image, x, y, false);
        Some([value[0] as f32, value[1] as f32, value[2] as f32])
    }

    fn donor_sample(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        if self.strokes.iter().any(|stroke| {
            stroke.strength > 0.0 && (x - stroke.x).hypot(y - stroke.y) < stroke.radius * 1.1 + 1.0
        }) {
            return None;
        }
        self.visible_sample(x, y)
    }
}

#[derive(Clone, Copy)]
struct RepairContext {
    dx: f32,
    dy: f32,
    rgb: [f32; 3],
}

/// Least-squares plane used only for ranking donors. Subtracting a lighting
/// plane before comparing the surrounding rings avoids selecting a smooth,
/// wrong-side-of-the-face patch just because its mean color happens to match.
fn repair_plane(context: &[RepairContext], colors: &[[f32; 3]], radius: f32) -> [[f32; 3]; 3] {
    let mut matrix = [[0.0; 3]; 3];
    let mut rhs = [[0.0; 3]; 3];
    for (point, color) in context.iter().zip(colors) {
        let v = [1.0, point.dx / radius, point.dy / radius];
        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] += v[i] * v[j];
                rhs[i][j] += v[i] * color[j];
            }
        }
    }
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] += 1e-4;
    }
    for i in 0..3 {
        let pivot = (i..3)
            .max_by(|a, b| matrix[*a][i].abs().total_cmp(&matrix[*b][i].abs()))
            .unwrap();
        matrix.swap(i, pivot);
        rhs.swap(i, pivot);
        let divisor = matrix[i][i];
        for j in 0..3 {
            matrix[i][j] /= divisor;
            rhs[i][j] /= divisor;
        }
        for k in 0..3 {
            if k == i {
                continue;
            }
            let factor = matrix[k][i];
            for j in 0..3 {
                matrix[k][j] -= matrix[i][j] * factor;
                rhs[k][j] -= rhs[i][j] * factor;
            }
        }
    }
    rhs
}

fn donor_score(
    sources: &RepairSources<'_>,
    context: &[RepairContext],
    x: f32,
    y: f32,
    radius: f32,
    origin: (f32, f32),
) -> Option<f32> {
    let mut colors = Vec::with_capacity(context.len());
    let mut difference = Vec::with_capacity(context.len());
    for point in context {
        let rgb = sources.donor_sample(x + point.dx, y + point.dy)?;
        difference.push(std::array::from_fn(|c| point.rgb[c] - rgb[c]));
        colors.push(rgb);
    }
    let adjustment = repair_plane(context, &difference, radius);
    let donor_plane = repair_plane(context, &colors, radius);
    let mut error = 0.0;
    for (point, difference) in context.iter().zip(&difference) {
        for c in 0..3 {
            let predicted = adjustment[0][c]
                + adjustment[1][c] * point.dx / radius
                + adjustment[2][c] * point.dy / radius;
            // Limit a single specular pore's influence without suppressing its
            // texture. Broad mismatch still raises the mean ring residual.
            error += (difference[c] - predicted).powi(2).min(1600.0);
        }
    }
    error /= context.len() as f32 * 3.0;
    // A second blemish can match the ring while hiding at its center. Compare
    // coarse inner-disk averages with the donor's own fitted lighting, rather
    // than rewarding zero pixel variance (which would erase healthy pores).
    for distance in [0.0, 0.35, 0.65] {
        let count = if distance == 0.0 { 1 } else { 16 };
        let mut coarse = [0.0; 3];
        for i in 0..count {
            let angle = i as f32 * std::f32::consts::TAU / count as f32;
            let (dx, dy) = (
                angle.cos() * radius * distance,
                angle.sin() * radius * distance,
            );
            let rgb = sources.donor_sample(x + dx, y + dy)?;
            for c in 0..3 {
                coarse[c] += (rgb[c]
                    - donor_plane[0][c]
                    - donor_plane[1][c] * dx / radius
                    - donor_plane[2][c] * dy / radius)
                    / count as f32;
            }
        }
        for value in coarse {
            error +=
                (value.abs() - 5.0).max(0.0).powi(2) * if distance == 0.0 { 0.05 } else { 0.25 };
        }
    }
    for ((mean, dx), dy) in adjustment[0].iter().zip(&adjustment[1]).zip(&adjustment[2]) {
        error += mean.powi(2) * 0.015 + (dx.powi(2) + dy.powi(2)) * 0.02;
    }
    // Small deterministic distance preference resolves equal texture matches.
    error += ((x - origin.0).hypot(y - origin.1) / radius) * 0.02;
    Some(error)
}

fn clean_sample(sources: &RepairSources<'_>, x: f32, y: f32, radius: f32) -> Option<(f32, f32)> {
    let mut context = Vec::with_capacity(96);
    for distance in [1.05, 1.3] {
        for i in 0..48 {
            let angle = i as f32 * std::f32::consts::TAU / 48.0;
            let (dx, dy) = (
                angle.cos() * radius * distance,
                angle.sin() * radius * distance,
            );
            if let Some(rgb) = sources.visible_sample(x + dx, y + dy) {
                context.push(RepairContext { dx, dy, rgb });
            }
        }
    }
    if context.len() < 24 {
        return None;
    }
    let mut candidates = Vec::with_capacity(264);
    // Search a bounded local neighborhood, not the full photograph on every
    // pointer stamp. Integer offsets preserve source texture without repeated
    // resampling. Four rings also avoid the old fixed-distance 16-way choice.
    for ring in [2.6, 3.5, 4.6, 6.0] {
        for i in 0..48 {
            let angle = i as f32 * std::f32::consts::TAU / 48.0;
            let sx = x + (angle.cos() * radius * ring).round();
            let sy = y + (angle.sin() * radius * ring).round();
            if let Some(score) = donor_score(sources, &context, sx, sy, radius, (x, y)) {
                candidates.push((score, sx, sy));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let finalists: Vec<_> = candidates.iter().take(8).copied().collect();
    let step = (radius * 0.3).round().max(1.0);
    for (_, bx, by) in finalists {
        for dy in [-step, 0.0, step] {
            for dx in [-step, 0.0, step] {
                let (sx, sy) = (bx + dx, by + dy);
                if let Some(score) = donor_score(sources, &context, sx, sy, radius, (x, y)) {
                    candidates.push((score, sx, sy));
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let xmin = (x - radius).floor().max(0.0) as u32;
    let xmax = (x + radius).ceil().min(sources.image.width as f32) as u32;
    let ymin = (y - radius).floor().max(0.0) as u32;
    let ymax = (y + radius).ceil().min(sources.image.height as f32) as u32;
    // Sparse ranking is followed by exhaustive validation of the actual patch.
    // A one-pixel alpha hole or marked neighboring blemish must not be copied
    // through gaps in the search probes.
    candidates.into_iter().find_map(|(_, sx, sy)| {
        for py in ymin..ymax {
            for px in xmin..xmax {
                let (dx, dy) = (px as f32 - x, py as f32 - y);
                if dx.hypot(dy) < radius && sources.donor_sample(sx + dx, sy + dy).is_none() {
                    return None;
                }
            }
        }
        Some((sx, sy))
    })
}

/// On a circular patch, Poisson cloning is donor texture plus a harmonic
/// correction whose boundary is destination minus donor. The Fourier solution
/// is analytic: it preserves donor detail, transfers local lighting gradients
/// and avoids an iterative full-image solver in the browser. Eight modes keep
/// individual pores/specular points from becoming low-frequency color bands.
struct HealLighting {
    mean: [f32; 3],
    cosine: [[f32; 3]; 8],
    sine: [[f32; 3]; 8],
    radius: f32,
}
impl HealLighting {
    fn new(
        sources: &RepairSources<'_>,
        target: &RgbaImage,
        stroke: RepairStroke,
        donor: (f32, f32),
    ) -> Option<Self> {
        let mut lighting = Self {
            mean: [0.0; 3],
            cosine: [[0.0; 3]; 8],
            sine: [[0.0; 3]; 8],
            radius: stroke.radius * 1.05,
        };
        let mut context = Vec::with_capacity(64);
        let mut difference = Vec::with_capacity(64);
        for i in 0..64 {
            let angle = i as f32 * std::f32::consts::TAU / 64.0;
            let (dx, dy) = (angle.cos() * lighting.radius, angle.sin() * lighting.radius);
            if sources
                .visible_sample(stroke.x + dx, stroke.y + dy)
                .is_none()
            {
                continue;
            }
            let rgb = sources.visible_sample(donor.0 + dx, donor.1 + dy)?;
            let dest = sample(target, stroke.x + dx, stroke.y + dy, false);
            context.push(RepairContext {
                dx,
                dy,
                rgb: std::array::from_fn(|c| dest[c] as f32),
            });
            difference.push(std::array::from_fn(|c| dest[c] as f32 - rgb[c]));
        }
        if context.len() < 24 {
            return None;
        }
        if context.len() != 64 {
            // Cropped image edges/protection interrupt the circle: a fitted
            // plane is stable here; uneven samples are not a Fourier basis.
            let plane = repair_plane(&context, &difference, lighting.radius);
            lighting.mean = plane[0];
            lighting.cosine[0] = plane[1];
            lighting.sine[0] = plane[2];
            return Some(lighting);
        }
        for (i, values) in difference.iter().enumerate() {
            let angle = i as f32 * std::f32::consts::TAU / 64.0;
            for (c, value) in values.iter().enumerate() {
                lighting.mean[c] += value / 64.0;
                for mode in 0..8 {
                    let a = angle * (mode + 1) as f32;
                    lighting.cosine[mode][c] += value * a.cos() / 32.0;
                    lighting.sine[mode][c] += value * a.sin() / 32.0;
                }
            }
        }
        Some(lighting)
    }
    fn correction(&self, dx: f32, dy: f32) -> [f32; 3] {
        let (x, y) = (dx / self.radius, dy / self.radius);
        let (mut real, mut imaginary) = (1.0, 0.0);
        let mut value = self.mean;
        for mode in 0..8 {
            (real, imaginary) = (real * x - imaginary * y, real * y + imaginary * x);
            for (c, value) in value.iter_mut().enumerate() {
                *value += self.cosine[mode][c] * real + self.sine[mode][c] * imaginary;
            }
        }
        value
    }
}

/// Local texture repair with optional manual source. Automatic healing ranks
/// clean surrounding patches and transfers donor gradients into the selected
/// defect. Alpha and hidden RGB stay byte-identical; output masks are applied
/// by the shared graph compositor exactly once.
pub(crate) fn repair(source: &RgbaImage, p: &Value, heal: bool) -> Result<RgbaImage, ImageError> {
    if number(p, "opacity", 100.0).clamp(0.0, 100.0) / 100.0 < 1e-8 {
        return Ok(source.clone());
    }
    let Some(values) = p.get("strokes").and_then(Value::as_array) else {
        return Ok(source.clone());
    };
    let strokes: Vec<RepairStroke> = values
        .iter()
        .map(|stroke| RepairStroke {
            x: number(stroke, "x", 0.5) * source.width as f32,
            y: number(stroke, "y", 0.5) * source.height as f32,
            radius: number(stroke, "radius", 0.04).clamp(0.001, 1.0)
                * source.width.min(source.height) as f32,
            strength: number(stroke, "strength", 100.0).clamp(0.0, 100.0) / 100.0,
            donor: if stroke.get("sourceX").is_some() && stroke.get("sourceY").is_some() {
                Some((
                    number(stroke, "sourceX", 0.25) * source.width as f32,
                    number(stroke, "sourceY", 0.25) * source.height as f32,
                ))
            } else {
                None
            },
        })
        .collect();
    let sources = RepairSources {
        image: source,
        protection: p
            .get("protectMask")
            .filter(|v| v.is_object())
            .map(PreparedMask::new),
        strokes: &strokes,
    };
    let manual_sources = RepairSources {
        image: source,
        protection: None,
        strokes: &strokes,
    };
    let selection = p
        .get("mask")
        .filter(|v| v.is_object())
        .map(PreparedMask::new);
    let region = Region::read(p.get("region"));
    let constrained =
        selection.is_some() || sources.protection.is_some() || p.get("region").is_some();
    let mut result = source.clone();
    for stroke in &strokes {
        if stroke.strength <= 0.0 {
            continue;
        }
        let xmin = (stroke.x - stroke.radius).floor().max(0.0) as u32;
        let xmax = (stroke.x + stroke.radius).ceil().min(source.width as f32) as u32;
        let ymin = (stroke.y - stroke.radius).floor().max(0.0) as u32;
        let ymax = (stroke.y + stroke.radius).ceil().min(source.height as f32) as u32;
        // An empty selection or a completely frozen brush is a no-op, even
        // when there is no usable donor. Decide before patch search so the
        // invisible layer cannot block later edits in the same graph.
        if constrained
            && !(ymin..ymax).any(|py| {
                (xmin..xmax).any(|px| {
                    if (px as f32 - stroke.x).hypot(py as f32 - stroke.y) >= stroke.radius {
                        return false;
                    }
                    if source.pixels[(py as usize * source.width as usize + px as usize) * 4 + 3]
                        == 0
                    {
                        return false;
                    }
                    let nx = (px as f32 + 0.5) / source.width as f32;
                    let ny = (py as f32 + 0.5) / source.height as f32;
                    let selected = selection.as_ref().map_or(1.0, |mask| {
                        mask.coverage(nx, ny, source.width, source.height)
                    });
                    let frozen = sources.protection.as_ref().map_or(0.0, |mask| {
                        mask.coverage(nx, ny, source.width, source.height)
                    });
                    region.contains(nx, ny) && selected * (1.0 - frozen) > 1e-8
                })
            })
        {
            continue;
        }
        let donor = match stroke.donor {
            Some(donor) => donor,
            None if heal => {
                clean_sample(&sources, stroke.x, stroke.y, stroke.radius).ok_or_else(|| {
                    invalid(
                        "healing has no valid nearby donor; use a smaller brush or choose a source",
                    )
                })?
            }
            None => (source.width as f32 * 0.25, source.height as f32 * 0.25),
        };
        // The selected source is an explicit instruction to copy this texture.
        // Keep automatic blemish search exclusions, but permit manual copying
        // from a frozen area without modifying that area. Both paths still
        // reject transparent/out-of-bounds source texels.
        let patch_sources = if stroke.donor.is_some() {
            &manual_sources
        } else {
            &sources
        };
        let lighting = if heal {
            Some(HealLighting::new(patch_sources, &result, *stroke, donor)
            .ok_or_else(|| invalid("healing source or surrounding context is transparent/protected; choose another source"))?)
        } else {
            None
        };
        for py in ymin..ymax {
            for px in xmin..xmax {
                let (dx, dy) = (px as f32 - stroke.x, py as f32 - stroke.y);
                let d = dx.hypot(dy) / stroke.radius;
                let index = (py as usize * source.width as usize + px as usize) * 4;
                if d >= 1.0 || source.pixels[index + 3] == 0 {
                    continue;
                }
                let Some(cloned) = patch_sources.visible_sample(donor.0 + dx, donor.1 + dy) else {
                    continue;
                };
                let correction = lighting
                    .as_ref()
                    .map(|v| v.correction(dx, dy))
                    .unwrap_or([0.0; 3]);
                let weight =
                    (1.0 - smoothstep(if heal { 0.8 } else { 0.65 }, 1.0, d)) * stroke.strength;
                for c in 0..3 {
                    result.pixels[index + c] = byte(
                        result.pixels[index + c] as f32 * (1.0 - weight)
                            + (cloned[c] + correction[c]) * weight,
                    );
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eye_size_scales_aperture_and_inner_texture_together_at_common_settings() {
        assert_eye_size_similarity_for_polygon(false);
    }
    #[test]
    fn eye_size_oblique_polygon_expands_support_instead_of_clipping_request() {
        assert_eye_size_similarity_for_polygon(true);
    }
    fn assert_eye_size_similarity_for_polygon(slanted: bool) {
        let width = 256;
        let height = 160;
        let pixels = (0..width * height)
            .flat_map(|i| {
                [
                    (i % 251) as u8,
                    ((i * 7) % 251) as u8,
                    ((i * 13) % 251) as u8,
                    255,
                ]
            })
            .collect();
        let source = RgbaImage::new(width, height, pixels).unwrap();
        let polygon = |cx: f32, cy: f32, rx: f32, ry: f32| {
            let points: Vec<_> = (0..16)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 16.0;
                    json!({"x":(cx+rx*a.cos())/width as f32,"y":(cy+ry*a.sin())/height as f32})
                })
                .collect();
            let points = if slanted && (rx - 24.0).abs() < 0.1 {
                [(-24.0, -9.0), (-4.0, -9.0), (24.0, 9.0), (4.0, 9.0)]
                    .iter()
                    .map(|&(x, y)| json!({"x":(cx+x)/width as f32,"y":(cy+y)/height as f32}))
                    .collect()
            } else {
                points
            };
            json!({"shape":"polygon","points":points,"feather":0})
        };
        for amount in [-100.0f32, -30.0, 10.0, 30.0, 100.0] {
            let params = json!({"adjustments":{"eyeSize":amount},"semanticMasks":{
                "leftEye":polygon(80.0,76.0,24.0,9.0),"rightEye":polygon(176.0,76.0,24.0,9.0),
                "leftBrow":polygon(80.0,43.0,31.0,5.0),"rightBrow":polygon(176.0,43.0,31.0,5.0)}});
            RECORD_MAPPING.set(true);
            RECORD_FLOW.set(true);
            let result = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            RECORD_FLOW.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            let report = LAST_FLOW_REPORT.take().unwrap();
            let actual_inverse = |x: f32, y: f32| {
                let x0 = x.floor() as usize;
                let y0 = y.floor() as usize;
                let tx = x - x.floor();
                let ty = y - y.floor();
                let i = y0 * width as usize + x0;
                let p = [
                    mapping[i],
                    mapping[i + 1],
                    mapping[i + width as usize],
                    mapping[i + width as usize + 1],
                ];
                (
                    p[0].0 * (1.0 - tx) * (1.0 - ty)
                        + p[1].0 * tx * (1.0 - ty)
                        + p[2].0 * (1.0 - tx) * ty
                        + p[3].0 * tx * ty,
                    p[0].1 * (1.0 - tx) * (1.0 - ty)
                        + p[1].1 * tx * (1.0 - ty)
                        + p[2].1 * (1.0 - tx) * ty
                        + p[3].1 * tx * ty,
                )
            };
            let scale = (0.3 * amount / 100.0).exp();
            for cx in [80.0, 176.0] {
                // Independent behavioral oracle: aperture, pupil/iris texture,
                // and intermediate points receive the same requested scale.
                for (rx, ry) in [(24.0, 9.0), (12.0, 6.0), (4.0, 4.0)] {
                    for i in 0..32 {
                        let angle = i as f32 * std::f32::consts::TAU / 32.0;
                        let point = (cx + rx * angle.cos(), 76.0 + ry * angle.sin());
                        let target = (cx + (point.0 - cx) * scale, 76.0 + (point.1 - 76.0) * scale);
                        let back = actual_inverse(target.0, target.1);
                        let error = (back.0 - point.0).hypot(back.1 - point.1);
                        assert!(error<0.025,"amount {amount}, source {point:?}, expected target {target:?}, actual inverse {back:?}, error {error}");
                    }
                }
            }
            for (name, cx) in [("leftEye", 80.0), ("rightEye", 176.0)] {
                let points = symmetry_feature(&params, name, &source).unwrap();
                for (a, b) in points
                    .iter()
                    .zip(points.iter().cycle().skip(1))
                    .take(points.len())
                {
                    // The oblique extremal vertices are the counterexample:
                    // checking only an ellipse ring would miss these corners.
                    for t in [0.0, 0.25, 0.5, 0.75] {
                        let point = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                        let target = (cx + (point.0 - cx) * scale, 76.0 + (point.1 - 76.0) * scale);
                        let back = actual_inverse(target.0, target.1);
                        let error = (back.0 - point.0).hypot(back.1 - point.1);
                        assert!(error<0.025,"actual polygon amount {amount}, source {point:?}, expected target {target:?}, actual inverse {back:?}, error {error}");
                    }
                }
            }
            assert_eq!(report["foldPixels"], 0);
            assert!(report["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01);
            // The anatomical eyebrow texture stays put. Expanded support may
            // legitimately move nearby skin outside the detected brow polygon.
            let brows: Vec<_> = ["leftBrow", "rightBrow"]
                .iter()
                .map(|name| PreparedMask::new(&params["semanticMasks"][name]))
                .collect();
            for y in 35..51 {
                for x in 45..210 {
                    if brows.iter().any(|mask| {
                        mask.coverage(
                            (x as f32 + 0.5) / width as f32,
                            (y as f32 + 0.5) / height as f32,
                            width,
                            height,
                        ) >= 0.5
                    }) {
                        let i = ((y * width + x) * 4) as usize;
                        assert_eq!(&source.pixels[i..i + 4], &result.pixels[i..i + 4]);
                    }
                }
            }
        }
    }

    #[test]
    fn eye_size_composes_with_existing_eye_fields_without_zero_switch() {
        let source = RgbaImage::new(
            256,
            160,
            (0..256 * 160)
                .flat_map(|i| {
                    [
                        (i % 251) as u8,
                        ((i * 7) % 251) as u8,
                        ((i * 13) % 251) as u8,
                        255,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let eye = |cx: f32| {
            json!({"shape":"polygon","feather":0,"points":[
            {"x":(cx-24.0)/256.0,"y":76.0/160.0},{"x":cx/256.0,"y":67.0/160.0},
            {"x":(cx+24.0)/256.0,"y":76.0/160.0},{"x":cx/256.0,"y":85.0/160.0}]})
        };
        for key in ["eyeWidth", "eyeTilt", "browPeak"] {
            let mut previous: Option<(Vec<(f32, f32)>, RgbaImage)> = None;
            for extra in [0.0, 0.001, 20.0] {
                let params = json!({"adjustments":{"eyeSize":30,key:extra},"semanticMasks":{"leftEye":eye(80.0),"rightEye":eye(176.0),"leftBrow":{"shape":"polygon","points":[{"x":0.20,"y":0.25},{"x":0.32,"y":0.21},{"x":0.44,"y":0.25},{"x":0.32,"y":0.26}]},"rightBrow":{"shape":"polygon","points":[{"x":0.56,"y":0.25},{"x":0.68,"y":0.21},{"x":0.80,"y":0.25},{"x":0.68,"y":0.26}]}}});
                RECORD_MAPPING.set(true);
                RECORD_FLOW.set(true);
                let result = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                RECORD_FLOW.set(false);
                let mapping = LAST_MAPPING.take().unwrap();
                let report = LAST_FLOW_REPORT.take().unwrap();
                assert_eq!(report["foldPixels"], 0, "{key}={extra}: {report}");
                assert!(
                    report["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01,
                    "{key}={extra}: {report}"
                );
                if extra == 0.001 {
                    let (old, old_image) = previous.as_ref().unwrap();
                    let max = mapping
                        .iter()
                        .zip(old)
                        .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                        .fold(0.0f32, f32::max);
                    assert!(max < 0.01, "{key}: 0 -> 0.001 map discontinuity {max}");
                    assert!(
                        result
                            .pixels
                            .iter()
                            .zip(&old_image.pixels)
                            .all(|(a, b)| a.abs_diff(*b) <= 1),
                        "{key}: pixels jump at zero"
                    );
                }
                if extra == 20.0 {
                    let (old, _) = previous.as_ref().unwrap();
                    let max = mapping
                        .iter()
                        .zip(old)
                        .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                        .fold(0.0f32, f32::max);
                    assert!(
                        max > 0.1,
                        "{key}: combination must actually respond, max {max}"
                    );
                }
                previous = Some((mapping, result));
            }
        }
    }

    #[test]
    fn eye_size_manual_fallback_keeps_legacy_rgba_output() {
        let (width, height) = (72, 60);
        let source = RgbaImage::new(
            width,
            height,
            (0..width * height)
                .flat_map(|i| {
                    let x = i % width;
                    let y = i / width;
                    [
                        ((x * 11 + y * 3) % 251) as u8,
                        ((x * 7 + y * 13) % 251) as u8,
                        ((x * 19 + y * 17) % 251) as u8,
                        (100 + (x + y) % 156) as u8,
                    ]
                })
                .collect(),
        )
        .unwrap();
        // Full-RGBA FNV-1a goldens from the unchanged legacy mapper,
        // captured before this candidate; no semantic masks are supplied.
        for (amount, expected) in [
            (-100, 0x7bc22786458aa97eu64),
            (30, 0x6d454437f5ad82b4u64),
            (100, 0x43b3bd9dbe721be3u64),
        ] {
            let params = json!({"adjustments":{"eyeSize":amount},"region":{"x":0.1,"y":0.1,"width":0.8,"height":0.8}});
            let result = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            let hash = result.pixels.iter().fold(14695981039346656037u64, |h, b| {
                (h ^ (*b as u64)).wrapping_mul(1099511628211)
            });
            assert_eq!(hash, expected, "manual fallback eyeSize {amount} changed");
        }
    }

    #[test]
    fn eye_width_moves_canthi_without_stretching_observed_iris() {
        let (width, height) = (256, 160);
        let source = RgbaImage::new(
            width,
            height,
            (0..width * height)
                .flat_map(|i| {
                    [
                        (i % 251) as u8,
                        ((i * 7) % 251) as u8,
                        ((i * 13) % 251) as u8,
                        255,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let eye = |cx: f32| {
            let points: Vec<_> = (0..32)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 32.0;
                    json!({"x":(cx+24.0*a.cos())/width as f32,"y":(76.0+6.0*a.sin())/height as f32})
                })
                .collect();
            json!({"shape":"polygon","points":points,"feather":0})
        };
        let iris = |cx: f32| json!({"iris":{"center":{"x":cx/256.0,"y":76.0/160.0},"boundary":[{"x":(cx+9.0)/256.0,"y":76.0/160.0},{"x":cx/256.0,"y":67.0/160.0},{"x":(cx-9.0)/256.0,"y":76.0/160.0},{"x":cx/256.0,"y":85.0/160.0}]}});
        for amount in [-30.0f32, 30.0] {
            let params = json!({"adjustments":{"eyeWidth":amount},"semanticMasks":{"leftEye":eye(80.0),"rightEye":eye(176.0)},"eyeGeometry":{"leftEye":iris(75.0),"rightEye":iris(181.0)}});
            RECORD_MAPPING.set(true);
            RECORD_FLOW.set(true);
            let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            RECORD_FLOW.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            let report = LAST_FLOW_REPORT.take().unwrap();
            let interpolate = |x: f32, y: f32| {
                let x0 = x.floor() as usize;
                let y0 = y.floor() as usize;
                let tx = x - x.floor();
                let ty = y - y.floor();
                let i = y0 * 256 + x0;
                let v = [
                    mapping[i],
                    mapping[i + 1],
                    mapping[i + 256],
                    mapping[i + 257],
                ];
                (
                    v[0].0 * (1.0 - tx) * (1.0 - ty)
                        + v[1].0 * tx * (1.0 - ty)
                        + v[2].0 * (1.0 - tx) * ty
                        + v[3].0 * tx * ty,
                    v[0].1 * (1.0 - tx) * (1.0 - ty)
                        + v[1].1 * tx * (1.0 - ty)
                        + v[2].1 * (1.0 - tx) * ty
                        + v[3].1 * tx * ty,
                )
            };
            for (cx, iris_cx) in [(80.0, 75.0), (176.0, 181.0)] {
                for sign in [-1.0, 1.0] {
                    let original = (cx + sign * 24.0, 76.0);
                    let target = (cx + sign * 24.0 * (amount * 0.3 / 100.0).exp(), 76.0);
                    let back = interpolate(target.0, target.1);
                    assert!((back.0-original.0).hypot(back.1-original.1)<0.025,"eyeWidth must move the actual canthus to its requested target: {amount}, {original:?}, {back:?}");
                }
                for y in 70..=82 {
                    for x in (iris_cx - 9.0) as u32..=(iris_cx + 9.0) as u32 {
                        let dx = x as f32 - iris_cx;
                        let dy = y as f32 - 76.0;
                        if dx * dx + dy * dy <= 81.0
                            && ((x as f32 - cx) / 24.0).powi(2) + (dy / 6.0).powi(2) <= 1.0
                        {
                            let i = (y * width + x) as usize;
                            assert_eq!(
                                mapping[i],
                                (x as f32, y as f32),
                                "visible iris material must remain stationary"
                            );
                            assert_eq!(
                                &output.pixels[i * 4..i * 4 + 4],
                                &source.pixels[i * 4..i * 4 + 4]
                            );
                        }
                    }
                }
            }
            assert_eq!(report["foldPixels"], 0);
            assert_eq!(report["fixedDonorImports"], 0);
            assert!(report["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01);
        }
    }

    fn measured_iris_feature_fixture() -> (RgbaImage, Value) {
        let source = RgbaImage::new(
            256,
            160,
            (0..256 * 160)
                .flat_map(|i| {
                    [
                        (i % 251) as u8,
                        ((i * 7) % 251) as u8,
                        ((i * 13) % 251) as u8,
                        255,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let eye = |cx: f32| {
            let points: Vec<_> = (0..32)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 32.0;
                    json!({"x":(cx+24.0*a.cos())/256.0,"y":(76.0+6.0*a.sin())/160.0})
                })
                .collect();
            json!({"shape":"polygon","points":points,"feather":0})
        };
        let iris = |cx: f32| json!({"iris":{"center":{"x":cx/256.0,"y":76.0/160.0},"boundary":[{"x":(cx+9.0)/256.0,"y":76.0/160.0},{"x":cx/256.0,"y":67.0/160.0},{"x":(cx-9.0)/256.0,"y":76.0/160.0},{"x":cx/256.0,"y":85.0/160.0}]}});
        (
            source,
            json!({"adjustments":{"eyeWidth":30},"semanticMasks":{"leftEye":eye(80.0),"rightEye":eye(176.0)},"eyeGeometry":{"leftEye":iris(75.0),"rightEye":iris(181.0)}}),
        )
    }

    #[test]
    fn eye_width_does_not_treat_hidden_iris_projection_as_observed_eyelid() {
        let (source, params) = measured_iris_feature_fixture();
        RECORD_MAPPING.set(true);
        local_adjust_kind(&source, &params, "feature.adjust").unwrap();
        RECORD_MAPPING.set(false);
        let mapping = LAST_MAPPING.take().unwrap();
        // These points are inside the predicted iris ellipse, but above the
        // actually observed aperture and outside the 1.5px sampling halo.
        for (x, y) in [(79usize, 68usize), (177, 68)] {
            let mapped = mapping[y * 256 + x];
            assert!(
                (mapped.0 - x as f32).hypot(mapped.1 - y as f32) > 0.001,
                "hidden predicted iris must not freeze eyelid texture: {mapped:?}"
            );
        }
    }

    #[test]
    fn eye_width_material_only_receives_the_existing_size_and_pose_targets() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeSize", "eyeSpacing", "eyeTilt"] {
            let mut params = base.clone();
            params["adjustments"] = json!({"eyeWidth":30,key:30});
            RECORD_WIDTH_MATERIAL.set(true);
            RECORD_FLOW.set(true);
            RECORD_MAPPING.set(true);
            local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_WIDTH_MATERIAL.set(false);
            RECORD_FLOW.set(false);
            RECORD_MAPPING.set(false);
            let report = LAST_WIDTH_MATERIAL_REPORT.take().unwrap();
            let flow = LAST_FLOW_REPORT.take().unwrap();
            let mapping = LAST_MAPPING.take().unwrap();
            for eye in report.as_array().unwrap() {
                assert!(eye["observedIrisSamples"].as_u64().unwrap() > 50);
                assert_eq!(eye["maxAdditionalMotionFromWidthPx"], 0.0, "{key}: {eye}");
                assert_eq!(eye["maxAdditionalJacobianColumnNorm"], 0.0, "{key}: {eye}");
            }
            for x in [75usize, 181] {
                let mapped = mapping[76 * 256 + x];
                assert!(
                    (mapped.0 - x as f32).hypot(mapped.1 - 76.0) > 0.01,
                    "{key} must still move/resize the iris"
                );
            }
            assert_eq!(flow["foldPixels"], 0, "{key}: {flow}");
            assert!(
                flow["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01,
                "{key}: {flow}"
            );
        }
    }

    #[test]
    fn eye_width_ordered_composition_is_continuous_when_either_control_crosses_zero() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeSize", "eyeSpacing", "eyeTilt"] {
            for fixed in [-30.0, 30.0] {
                for changing_width in [false, true] {
                    let mut previous: Option<(Vec<(f32, f32)>, RgbaImage)> = None;
                    for tiny in [-0.001, 0.0, 0.001] {
                        let mut params = base.clone();
                        params["adjustments"] = if changing_width {
                            json!({"eyeWidth":tiny,key:fixed})
                        } else {
                            json!({"eyeWidth":fixed,key:tiny})
                        };
                        RECORD_MAPPING.set(true);
                        let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                        RECORD_MAPPING.set(false);
                        let mapping = LAST_MAPPING.take().unwrap();
                        if let Some((old, old_pixels)) = previous {
                            let max = mapping
                                .iter()
                                .zip(old)
                                .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                                .fold(0.0f32, f32::max);
                            assert!(max<0.01,"{key}, fixed={fixed}, changing_width={changing_width}, tiny={tiny}: map jump {max}");
                            assert!(output.pixels.iter().zip(old_pixels.pixels).all(|(a,b)|a.abs_diff(b)<=1),"{key}, fixed={fixed}, changing_width={changing_width}, tiny={tiny}: pixels jumped");
                        }
                        previous = Some((mapping, output));
                    }
                }
            }
        }
    }

    #[test]
    fn eye_width_zero_preserves_existing_size_and_pose_output_exactly() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeSize", "eyeSpacing", "eyeTilt"] {
            for amount in [-30.0, 30.0] {
                let mut params = base.clone();
                params["adjustments"] = json!({"eyeWidth":0,key:amount});
                RECORD_MAPPING.set(true);
                let measured = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                let measured_mapping = LAST_MAPPING.take().unwrap();
                params.as_object_mut().unwrap().remove("eyeGeometry");
                params["adjustments"]
                    .as_object_mut()
                    .unwrap()
                    .remove("eyeWidth");
                RECORD_MAPPING.set(true);
                let existing = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                assert_eq!(
                    measured_mapping,
                    LAST_MAPPING.take().unwrap(),
                    "{key} {amount}: zero-width must not alter source-coordinate mapping"
                );
                assert_eq!(
                    measured.pixels, existing.pixels,
                    "{key} {amount}: zero-width must not alter output pixels"
                );
            }
        }
    }

    #[test]
    fn eye_width_size_composition_moves_canthi_to_full_requested_targets() {
        let (source, base) = measured_iris_feature_fixture();
        for width in [-60.0f32, 60.0] {
            for size in [-60.0f32, 60.0] {
                let mut params = base.clone();
                params["adjustments"] = json!({"eyeWidth":width,"eyeSize":size});
                RECORD_MAPPING.set(true);
                let _output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                let mapping = LAST_MAPPING.take().unwrap();
                for cx in [80.0, 176.0] {
                    for sign in [-1.0, 1.0] {
                        let original = (cx + sign * 24.0, 76.0);
                        let target = (
                            cx + sign * 24.0 * ((width + size) * 0.3 / 100.0).exp(),
                            76.0,
                        );
                        let x0 = target.0.floor() as usize;
                        let tx = target.0 - target.0.floor();
                        let a = mapping[76 * 256 + x0];
                        let b = mapping[76 * 256 + x0 + 1];
                        let back = (a.0 * (1.0 - tx) + b.0 * tx, a.1 * (1.0 - tx) + b.1 * tx);
                        let error = (back.0 - original.0).hypot(back.1 - original.1);
                        assert!(error<0.025,"width={width} size={size}: full canthus target missed: original={original:?} target={target:?} sampled_inverse={back:?} error={error}");
                    }
                }
            }
        }
    }

    #[test]
    fn eye_spacing_moves_entire_detected_eye_by_requested_translation() {
        let (source, base) = measured_iris_feature_fixture();
        for amount in [-30.0f32, 30.0] {
            let mut params = base.clone();
            params["adjustments"] = json!({"eyeSpacing":amount});
            RECORD_MAPPING.set(true);
            RECORD_FLOW.set(true);
            local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            RECORD_FLOW.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            let report = LAST_FLOW_REPORT.take().unwrap();
            for (cx, sign) in [(80.0, -1.0), (176.0, 1.0)] {
                let shift = sign * 0.08 * (48.0 / 0.26) * (amount / 100.0);
                for (rx, ry) in [(24.0, 6.0), (12.0, 4.0), (4.0, 3.0)] {
                    for i in 0..32 {
                        let a = i as f32 * std::f32::consts::TAU / 32.0;
                        let original = (cx + rx * a.cos(), 76.0 + ry * a.sin());
                        let target = (original.0 + shift, original.1);
                        let x0 = target.0.floor() as usize;
                        let y0 = target.1.floor() as usize;
                        let tx = target.0 - target.0.floor();
                        let ty = target.1 - target.1.floor();
                        let index = y0 * 256 + x0;
                        let v = [
                            mapping[index],
                            mapping[index + 1],
                            mapping[index + 256],
                            mapping[index + 257],
                        ];
                        let back = (
                            v[0].0 * (1.0 - tx) * (1.0 - ty)
                                + v[1].0 * tx * (1.0 - ty)
                                + v[2].0 * (1.0 - tx) * ty
                                + v[3].0 * tx * ty,
                            v[0].1 * (1.0 - tx) * (1.0 - ty)
                                + v[1].1 * tx * (1.0 - ty)
                                + v[2].1 * (1.0 - tx) * ty
                                + v[3].1 * tx * ty,
                        );
                        let error = (back.0 - original.0).hypot(back.1 - original.1);
                        assert!(error<0.025,"spacing={amount}: every aperture/iris point must get requested shift {shift}, source={original:?}, actual_inverse={back:?}, error={error}");
                    }
                }
            }
            assert_eq!(report["foldPixels"], 0);
            assert_eq!(report["fixedDonorImports"], 0);
            assert!(report["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01);
        }
    }

    #[test]
    fn eye_spacing_carries_width_size_results_without_changing_their_targets() {
        let (source, base) = measured_iris_feature_fixture();
        for (width, size) in [
            (0.0, 30.0),
            (30.0, 0.0),
            (30.0, 30.0),
            (-30.0, -30.0),
            (60.0, 60.0),
            (60.0, -60.0),
        ] {
            for spacing in [-30.0, 30.0] {
                let mut params = base.clone();
                params["adjustments"] =
                    json!({"eyeWidth":width,"eyeSize":size,"eyeSpacing":spacing});
                RECORD_SPACING_MATERIAL.set(true);
                RECORD_FLOW.set(true);
                local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_SPACING_MATERIAL.set(false);
                RECORD_FLOW.set(false);
                let report = LAST_SPACING_MATERIAL_REPORT.take().unwrap();
                let flow = LAST_FLOW_REPORT.take().unwrap();
                for eye in report.as_array().unwrap() {
                    assert!(
                        eye["maxAdditionalShiftTargetErrorPx"].as_f64().unwrap() < 0.025,
                        "width={width} size={size} spacing={spacing}: {eye}"
                    );
                    assert!(
                        eye["maxFinalInverseErrorPx"].as_f64().unwrap() < 0.01,
                        "{eye}"
                    );
                }
                assert_eq!(flow["foldPixels"], 0);
                assert_eq!(flow["fixedDonorImports"], 0);
                assert!(
                    flow["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01,
                    "{flow}"
                );
            }
        }
    }

    #[test]
    fn eye_spacing_preserves_brow_material_while_brow_control_still_works() {
        let (source, mut base) = measured_iris_feature_fixture();
        let brow = |cx: f32| json!({"shape":"polygon","points":[{"x":(cx-30.0)/256.0,"y":48.0/160.0},{"x":cx/256.0,"y":39.0/160.0},{"x":(cx+30.0)/256.0,"y":48.0/160.0},{"x":cx/256.0,"y":46.0/160.0}],"feather":0});
        base["semanticMasks"]["leftBrow"] = brow(80.0);
        base["semanticMasks"]["rightBrow"] = brow(176.0);
        for peak in [-30.0, 30.0] {
            for spacing in [-30.0, 30.0] {
                let mut params = base.clone();
                params["adjustments"] = json!({"browPeak":peak,"eyeSpacing":spacing});
                RECORD_SPACING_MATERIAL.set(true);
                local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_SPACING_MATERIAL.set(false);
                let report = LAST_SPACING_MATERIAL_REPORT.take().unwrap();
                for row in report.as_array().unwrap() {
                    if row["feature"].as_str().unwrap().ends_with("Brow") {
                        assert!(
                            row["maxPriorMovementPx"].as_f64().unwrap() > 0.1,
                            "browPeak must remain active: {row}"
                        );
                        assert!(
                            row["maxAdditionalShiftTargetErrorPx"].as_f64().unwrap() < 0.025,
                            "browPeak={peak}, spacing={spacing}: {row}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn eye_spacing_composition_is_continuous_from_both_sides_of_zero() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeSize", "eyeWidth", "eyeTilt"] {
            for fixed in [-30.0, 30.0] {
                for varying_spacing in [false, true] {
                    let mut previous: Option<(Vec<(f32, f32)>, RgbaImage)> = None;
                    for tiny in [-0.001, 0.0, 0.001] {
                        let mut params = base.clone();
                        params["adjustments"] = if varying_spacing {
                            json!({"eyeSpacing":tiny,key:fixed})
                        } else {
                            json!({"eyeSpacing":fixed,key:tiny})
                        };
                        RECORD_MAPPING.set(true);
                        let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                        RECORD_MAPPING.set(false);
                        let mapping = LAST_MAPPING.take().unwrap();
                        if let Some((old, pixels)) = previous {
                            let jump = mapping
                                .iter()
                                .zip(old)
                                .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                                .fold(0.0f32, f32::max);
                            assert!(jump<0.01,"{key}, fixed={fixed}, varying_spacing={varying_spacing}: jump={jump}");
                            assert!(output
                                .pixels
                                .iter()
                                .zip(pixels.pixels)
                                .all(|(a, b)| a.abs_diff(b) <= 1));
                        }
                        previous = Some((mapping, output));
                    }
                }
            }
        }
    }

    #[test]
    fn eye_spacing_preserves_the_existing_nonzero_tilt_result() {
        let (source, base) = measured_iris_feature_fixture();
        for tilt in [-30.0, 30.0] {
            for spacing in [-30.0, 30.0] {
                let mut params = base.clone();
                params["adjustments"] = json!({"eyeTilt":tilt,"eyeSpacing":spacing});
                RECORD_SPACING_MATERIAL.set(true);
                local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_SPACING_MATERIAL.set(false);
                let report = LAST_SPACING_MATERIAL_REPORT.take().unwrap();
                for row in report.as_array().unwrap() {
                    assert!(
                        row["maxPriorMovementPx"].as_f64().unwrap() > 0.1,
                        "tilt must remain active: {row}"
                    );
                    assert!(
                        row["maxAdditionalShiftTargetErrorPx"].as_f64().unwrap() < 0.005,
                        "tilt={tilt}, spacing={spacing}: {row}"
                    );
                    assert!(
                        row["maxFinalInverseErrorPx"].as_f64().unwrap() < 0.01,
                        "{row}"
                    );
                }
            }
        }
    }

    #[test]
    fn eye_spacing_zero_keeps_existing_width_size_tilt_output_exactly() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeWidth", "eyeSize", "eyeTilt"] {
            for amount in [-30.0, 30.0] {
                let mut params = base.clone();
                params["adjustments"] = json!({"eyeSpacing":0,key:amount});
                RECORD_MAPPING.set(true);
                let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                let mapping = LAST_MAPPING.take().unwrap();
                params["adjustments"]
                    .as_object_mut()
                    .unwrap()
                    .remove("eyeSpacing");
                RECORD_MAPPING.set(true);
                let reference = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                assert_eq!(mapping, LAST_MAPPING.take().unwrap(), "{key} {amount}");
                assert_eq!(output.pixels, reference.pixels, "{key} {amount}");
            }
        }
    }

    #[test]
    fn eye_tilt_rotates_aperture_and_iris_by_the_full_requested_angle() {
        let (source, base) = measured_iris_feature_fixture();
        for amount in [-30.0f32, 30.0] {
            let mut params = base.clone();
            params["adjustments"] = json!({"eyeTilt":amount});
            RECORD_MAPPING.set(true);
            RECORD_FLOW.set(true);
            local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            RECORD_FLOW.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            let flow = LAST_FLOW_REPORT.take().unwrap();
            for (cx, iris_cx, sign) in [(80.0, 75.0, -1.0), (176.0, 181.0, 1.0)] {
                let theta = -sign * 0.3 * amount / 100.0;
                let (sn, cs) = theta.sin_cos();
                for (px, rx, ry) in [(cx, 24.0, 6.0), (cx, 12.0, 4.0), (iris_cx, 4.0, 3.0)] {
                    for i in 0..32 {
                        let a = i as f32 * std::f32::consts::TAU / 32.0;
                        let original = (px + rx * a.cos(), 76.0 + ry * a.sin());
                        let d = (original.0 - cx, original.1 - 76.0);
                        let target = (cx + cs * d.0 - sn * d.1, 76.0 + sn * d.0 + cs * d.1);
                        let x0 = target.0.floor() as usize;
                        let y0 = target.1.floor() as usize;
                        let tx = target.0 - target.0.floor();
                        let ty = target.1 - target.1.floor();
                        let index = y0 * 256 + x0;
                        let v = [
                            mapping[index],
                            mapping[index + 1],
                            mapping[index + 256],
                            mapping[index + 257],
                        ];
                        let back = (
                            v[0].0 * (1.0 - tx) * (1.0 - ty)
                                + v[1].0 * tx * (1.0 - ty)
                                + v[2].0 * (1.0 - tx) * ty
                                + v[3].0 * tx * ty,
                            v[0].1 * (1.0 - tx) * (1.0 - ty)
                                + v[1].1 * tx * (1.0 - ty)
                                + v[2].1 * (1.0 - tx) * ty
                                + v[3].1 * tx * ty,
                        );
                        let error = (back.0 - original.0).hypot(back.1 - original.1);
                        assert!(error<0.025,"tilt={amount}: whole eye must rotate {theta} radians, source={original:?}, actual_inverse={back:?}, error={error}");
                    }
                }
            }
            assert_eq!(flow["foldPixels"], 0);
            assert_eq!(flow["fixedDonorImports"], 0);
            assert!(flow["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.01);
        }
    }

    #[test]
    fn eye_tilt_sweep_includes_interior_angular_extrema() {
        let a = 10.0f32.to_radians();
        for sign in [-1.0, 1.0] {
            let p = (sign * 10.0 * a.sin(), 10.0 * a.cos());
            let sweep = sign * 20.0f32.to_radians();
            // The vector becomes vertical halfway through the sweep. Neither
            // endpoint attains this known maximum normalized radius of one.
            assert!((rotated_eye_extent(p, (30.0, 10.0), sweep) - 1.0).abs() < 1e-6);
            assert!((angular_component_extent(p.1, p.0, sweep) - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn eye_tilt_size_width_spacing_obey_one_requested_organ_transform() {
        let (source, base) = measured_iris_feature_fixture();
        for (width, size, tilt, spacing) in [
            (0.0, 30.0, 30.0, 0.0),
            (0.0, -30.0, -30.0, 0.0),
            (0.0, 30.0, -30.0, 30.0),
            (0.0, -30.0, 30.0, -30.0),
            (30.0, 30.0, 30.0, 30.0),
            (-30.0, 30.0, -30.0, -30.0),
            (30.0, -30.0, -30.0, 30.0),
            (-30.0, -30.0, 30.0, -30.0),
        ] {
            let mut params = base.clone();
            params["adjustments"] =
                json!({"eyeWidth":width,"eyeSize":size,"eyeTilt":tilt,"eyeSpacing":spacing});
            RECORD_MAPPING.set(true);
            local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            for (cx, iris_cx, sign) in [(80.0f32, 75.0f32, -1.0f32), (176.0, 181.0, 1.0)] {
                let theta = -sign * 0.3 * tilt / 100.0;
                let (sn, cs) = theta.sin_cos();
                let scale = (0.3f32 * size / 100.0).exp();
                let width_scale = (0.3f32 * width / 100.0).exp();
                let shift = sign * 0.08 * (48.0 / 0.26) * spacing / 100.0;
                let mut points = vec![(cx - 24.0, 76.0, true), (cx + 24.0, 76.0, true)];
                for i in 0..32 {
                    let a = i as f32 * std::f32::consts::TAU / 32.0;
                    points.push((iris_cx + 4.0 * a.cos(), 76.0 + 3.0 * a.sin(), false));
                    if width == 0.0 {
                        points.push((cx + 24.0 * a.cos(), 76.0 + 6.0 * a.sin(), false));
                    }
                }
                for (x, y, canthus) in points {
                    let u = (x - cx) * if canthus { width_scale } else { 1.0 };
                    let v = y - 76.0;
                    let target = (
                        cx + scale * (cs * u - sn * v) + shift,
                        76.0 + scale * (sn * u + cs * v),
                    );
                    let x0 = target.0.floor() as usize;
                    let y0 = target.1.floor() as usize;
                    let tx = target.0 - target.0.floor();
                    let ty = target.1 - target.1.floor();
                    let i = y0 * 256 + x0;
                    let q = [
                        mapping[i],
                        mapping[i + 1],
                        mapping[i + 256],
                        mapping[i + 257],
                    ];
                    let back = (
                        q[0].0 * (1.0 - tx) * (1.0 - ty)
                            + q[1].0 * tx * (1.0 - ty)
                            + q[2].0 * (1.0 - tx) * ty
                            + q[3].0 * tx * ty,
                        q[0].1 * (1.0 - tx) * (1.0 - ty)
                            + q[1].1 * tx * (1.0 - ty)
                            + q[2].1 * (1.0 - tx) * ty
                            + q[3].1 * tx * ty,
                    );
                    let error = (back.0 - x).hypot(back.1 - y);
                    assert!(error<0.025,"w={width} s={size} t={tilt} spacing={spacing}, source=({x},{y}), target={target:?}, actual_inverse={back:?}, error={error}");
                }
            }
        }
    }

    #[test]
    fn eye_tilt_preserves_source_brows_and_keeps_brow_controls_active() {
        let (source, mut base) = measured_iris_feature_fixture();
        let brow = |cx: f32| json!({"shape":"polygon","points":[{"x":(cx-30.0)/256.0,"y":58.0/160.0},{"x":cx/256.0,"y":49.0/160.0},{"x":(cx+30.0)/256.0,"y":58.0/160.0},{"x":cx/256.0,"y":56.0/160.0}],"feather":0});
        base["semanticMasks"]["leftBrow"] = brow(80.0);
        base["semanticMasks"]["rightBrow"] = brow(176.0);
        for tilt in [-30.0, 30.0] {
            let mut params = base.clone();
            params["adjustments"] = json!({"eyeTilt":tilt});
            RECORD_MAPPING.set(true);
            let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            let mut brow_pixels = Vec::new();
            for name in ["leftBrow", "rightBrow"] {
                let p = symmetry_feature(&params, name, &source).unwrap();
                for y in 49..=58 {
                    for x in 45..=210 {
                        if feature_polygon_signed_distance(&p, (x as f32, y as f32)) <= 0.0 {
                            let i = y * 256 + x;
                            brow_pixels.push(i);
                            assert_eq!(mapping[i], (x as f32, y as f32));
                            assert_eq!(
                                &output.pixels[i * 4..i * 4 + 4],
                                &source.pixels[i * 4..i * 4 + 4]
                            );
                        }
                    }
                }
            }
            params["adjustments"]["browPeak"] = json!(30);
            RECORD_MAPPING.set(true);
            local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            RECORD_MAPPING.set(false);
            let changed = LAST_MAPPING.take().unwrap();
            assert!(
                brow_pixels
                    .iter()
                    .any(|&i| (changed[i].0 - (i % 256) as f32)
                        .hypot(changed[i].1 - (i / 256) as f32)
                        > 0.1),
                "browPeak must not be globally frozen"
            );
        }
    }

    #[test]
    fn eye_tilt_composition_is_continuous_from_both_sides_of_zero() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeSize", "eyeWidth", "eyeSpacing"] {
            for fixed in [-30.0, 30.0] {
                for changing_tilt in [false, true] {
                    let mut previous: Option<(Vec<(f32, f32)>, RgbaImage)> = None;
                    for tiny in [-0.001, 0.0, 0.001] {
                        let mut params = base.clone();
                        params["adjustments"] = if changing_tilt {
                            json!({"eyeTilt":tiny,key:fixed})
                        } else {
                            json!({"eyeTilt":fixed,key:tiny})
                        };
                        RECORD_MAPPING.set(true);
                        let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                        RECORD_MAPPING.set(false);
                        let mapping = LAST_MAPPING.take().unwrap();
                        if let Some((old, pixels)) = previous {
                            let jump = mapping
                                .iter()
                                .zip(old)
                                .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                                .fold(0.0f32, f32::max);
                            assert!(
                                jump < 0.01,
                                "{key}, fixed={fixed}, changing_tilt={changing_tilt}: {jump}"
                            );
                            assert!(output
                                .pixels
                                .iter()
                                .zip(pixels.pixels)
                                .all(|(a, b)| a.abs_diff(b) <= 1));
                        }
                        previous = Some((mapping, output));
                    }
                }
            }
        }
    }

    #[test]
    fn eye_tilt_zero_keeps_other_eye_output_exactly() {
        let (source, base) = measured_iris_feature_fixture();
        for key in ["eyeSize", "eyeWidth", "eyeSpacing"] {
            for amount in [-30.0, 30.0] {
                let mut params = base.clone();
                params["adjustments"] = json!({"eyeTilt":0,key:amount});
                RECORD_MAPPING.set(true);
                let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                let mapping = LAST_MAPPING.take().unwrap();
                params["adjustments"]
                    .as_object_mut()
                    .unwrap()
                    .remove("eyeTilt");
                RECORD_MAPPING.set(true);
                let reference = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                RECORD_MAPPING.set(false);
                assert_eq!(mapping, LAST_MAPPING.take().unwrap());
                assert_eq!(output.pixels, reference.pixels);
            }
        }
    }

    #[test]
    fn eye_tilt_flow_keeps_small_increments_at_photographic_coordinates() {
        let center = (400.0f32, 280.0f32);
        let point = (423.0f32, 278.0f32);
        let coverage = |_: f32, _: f32| 1.0;
        for (scale, angle, shift) in [
            (0.0f32, 0.0f32, (4.25f32, 1.375f32)),
            (0.0, 0.09, (0.0, 0.0)),
            (0.09, 0.09, (0.0, 0.0)),
        ] {
            let warp = LocalWarp {
                center,
                across: (1.0, 0.0),
                radius: (1000.0, 1000.0),
                scale: (scale, scale),
                translation: shift,
                angle,
                preserve_eyes: false,
                whole_eye_size: true,
                whole_eye_tilt: true,
                whole_eye_spacing: false,
                eye_width_material: None,
                whole_eye_inner: 0.9,
                eye_size_brows: None,
            };
            let actual =
                integrate_feature_flow(point, &[warp.clone()], &[], &[], &coverage, 267, 1.0);
            let expected = if shift.0 != 0.0 {
                (
                    point.0 as f64 + shift.0 as f64,
                    point.1 as f64 + shift.1 as f64,
                )
            } else {
                let a = angle as f64;
                let e = (scale as f64).exp();
                let x = (point.0 - center.0) as f64;
                let y = (point.1 - center.1) as f64;
                (
                    center.0 as f64 + e * (a.cos() * x - a.sin() * y),
                    center.1 as f64 + e * (a.sin() * x + a.cos() * y),
                )
            };
            assert!((actual.0 as f64-expected.0).hypot(actual.1 as f64-expected.1)<0.0001,"scale={scale}, angle={angle}, shift={shift:?}, actual={actual:?}, expected={expected:?}");
            let reverse = integrate_feature_flow(actual, &[warp], &[], &[], &coverage, 267, -1.0);
            assert!((reverse.0 - point.0).hypot(reverse.1 - point.1) < 0.0001);
        }
    }

    // Photo-derived landmarks stay local. Loading at runtime keeps ordinary
    // builds independent of private test data while retaining opt-in replay.
    fn local_photo_feature_fixture() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/user-brow-geometry-20261014.json");
        let data = std::fs::read_to_string(&path)
            .expect("local photo fixture is required; see doc/qa/local-evidence-policy.md");
        serde_json::from_str(&data).expect("local photo fixture must be valid JSON")
    }

    thread_local! {
        static RECORD_MAPPING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static RECORD_SPACING_MATERIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static LAST_SPACING_MATERIAL_REPORT: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
        static RECORD_WIDTH_MATERIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static LAST_WIDTH_MATERIAL_REPORT: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
        static LAST_MAPPING: std::cell::RefCell<Option<Vec<(f32, f32)>>> = const { std::cell::RefCell::new(None) };
        static RECORD_FLOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static LAST_FLOW_REPORT: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
        static RECORD_ROTATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static LAST_ROTATION_REPORT: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
    }

    /// Optional photographic diagnostics execute the exact final Rust mapper,
    /// including fixed-donor rejection, region/selection, and post fields.
    /// It is test-only; release builds contain no environment or QA hooks.
    // Keep every captured pipeline input explicit in this test-only observer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn audit_photographic_feature_flow(
        mapping: &impl Fn(f32, f32) -> (f32, f32),
        changes: &[LocalWarp],
        eyes: &[FeatureFrame],
        post_warps: &[SymmetryWarp],
        coverage: &impl Fn(f32, f32) -> f32,
        steps: usize,
        source: &RgbaImage,
        params: &Value,
        stationary: &[bool],
    ) {
        if RECORD_MAPPING.get() {
            LAST_MAPPING.set(Some(
                (0..source.width * source.height)
                    .map(|i| mapping((i % source.width) as f32, (i / source.width) as f32))
                    .collect(),
            ));
        }
        if RECORD_ROTATION.get() {
            let polygon_area = |points: &[(f32, f32)]| {
                points
                    .iter()
                    .zip(points.iter().cycle().skip(1))
                    .take(points.len())
                    .map(|(a, b)| a.0 as f64 * b.1 as f64 - a.1 as f64 * b.0 as f64)
                    .sum::<f64>()
                    .abs()
                    * 0.5
            };
            let mut rows = Vec::new();
            for name in ["leftEye", "rightEye"] {
                let eye = feature_frame(params, name, source).unwrap();
                let (x, y) = eye.center;
                let h = 0.1;
                let xp = mapping(x + h, y);
                let xm = mapping(x - h, y);
                let yp = mapping(x, y + h);
                let ym = mapping(x, y - h);
                let jacobian =
                    ((xp.0 - xm.0) * (yp.1 - ym.1) - (xp.1 - xm.1) * (yp.0 - ym.0)) / (4.0 * h * h);
                let angle = (xp.1 - xm.1).atan2(xp.0 - xm.0);
                let warp = changes
                    .iter()
                    .min_by(|a, b| {
                        let distance = |w: &LocalWarp| (w.center.0 - x).hypot(w.center.1 - y);
                        distance(a).total_cmp(&distance(b))
                    })
                    .unwrap();
                let original = symmetry_feature(params, name, source).unwrap();
                let forward: Vec<_> = original
                    .iter()
                    .map(|&point| {
                        integrate_feature_flow(
                            point, changes, eyes, post_warps, coverage, steps, -1.0,
                        )
                    })
                    .collect();
                rows.push(json!({"name":name,"center":[x,y],"mappedCenter":mapping(x,y),
                    "centerInverseJacobian":jacobian,"centerForwardAreaRatio":1.0/jacobian,
                    "centerAngleRadians":angle,"requestedAngleRadians":warp.angle,
                    "detectedPolygonForwardAreaRatio":polygon_area(&forward)/polygon_area(&original)}));
            }
            LAST_ROTATION_REPORT.set(Some(json!(rows)));
        }

        let spacing_path = std::env::var("BROW_FLOW_DIAGNOSTICS_OUT").ok();
        if spacing_path.is_some() || RECORD_SPACING_MATERIAL.get() {
            let without: Vec<_> = changes
                .iter()
                .filter(|w| !w.whole_eye_spacing)
                .cloned()
                .collect();
            let full =
                |p| integrate_feature_flow(p, changes, eyes, post_warps, coverage, steps, -1.0);
            let prior =
                |p| integrate_feature_flow(p, &without, eyes, post_warps, coverage, steps, -1.0);
            let mut records = Vec::new();
            for name in ["leftEye", "rightEye", "leftBrow", "rightBrow"] {
                let Some(points) = symmetry_feature(params, name, source) else {
                    continue;
                };
                let eye_name = if name.starts_with("left") {
                    "leftEye"
                } else {
                    "rightEye"
                };
                let frame = feature_frame(params, eye_name, source).unwrap();
                let sign = if name.starts_with("left") { -1.0 } else { 1.0 };
                let expected = if name.ends_with("Eye") {
                    let shift = sign
                        * 0.08
                        * (frame.width / 0.26)
                        * percent(params.get("adjustments").unwrap(), "eyeSpacing");
                    (frame.across.0 * shift, frame.across.1 * shift)
                } else {
                    (0.0, 0.0)
                };
                let mut max_error = 0.0f32;
                let mut reference_movement = 0.0f32;
                let mut reverse_error = 0.0f32;
                let mut count = 0;
                for (a, b) in points
                    .iter()
                    .zip(points.iter().cycle().skip(1))
                    .take(points.len())
                {
                    for t in [0.0, 0.25, 0.5, 0.75] {
                        let p = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                        let target = full(p);
                        let reference = prior(p);
                        let back = mapping(target.0, target.1);
                        max_error = max_error.max(
                            (target.0 - reference.0 - expected.0)
                                .hypot(target.1 - reference.1 - expected.1),
                        );
                        reference_movement =
                            reference_movement.max((reference.0 - p.0).hypot(reference.1 - p.1));
                        reverse_error = reverse_error.max((back.0 - p.0).hypot(back.1 - p.1));
                        count += 1;
                    }
                }
                records.push(json!({"feature":name,"sourceBoundarySamples":count,"maxAdditionalShiftTargetErrorPx":max_error,"maxPriorMovementPx":reference_movement,"maxFinalInverseErrorPx":reverse_error,"scope":"actual final forward flow compared with same prior fields after removing spacing; expected eye constant translation or brow zero additional motion"}));
            }
            if let Some(path) = spacing_path {
                std::fs::write(
                    format!("{}.spacing.json", path),
                    serde_json::to_vec_pretty(&records).unwrap(),
                )
                .unwrap();
            }
            if RECORD_SPACING_MATERIAL.get() {
                LAST_SPACING_MATERIAL_REPORT.set(Some(json!(records)));
            }
        }

        let material_path = std::env::var("BROW_FLOW_DIAGNOSTICS_OUT").ok();
        if material_path.is_some() || RECORD_WIDTH_MATERIAL.get() {
            let other: Vec<_> = changes
                .iter()
                .filter(|w| w.eye_width_material.is_none())
                .cloned()
                .collect();
            let mut records = Vec::new();
            for warp in changes {
                let Some(material) = &warp.eye_width_material else {
                    continue;
                };
                let mut b = [
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ];
                for &(x, y) in &material.aperture {
                    b[0] = b[0].min(x);
                    b[1] = b[1].min(y);
                    b[2] = b[2].max(x);
                    b[3] = b[3].max(y);
                }
                let mut n = 0;
                let mut max_error = 0.0f32;
                let mut sum = 0.0f64;
                let mut max_at = (0.0, 0.0);
                let mut derivative_error = 0.0f32;
                let full =
                    |p| integrate_feature_flow(p, changes, eyes, post_warps, coverage, steps, -1.0);
                let reference =
                    |p| integrate_feature_flow(p, &other, eyes, post_warps, coverage, steps, -1.0);
                for y in b[1].floor() as i32..=b[3].ceil() as i32 {
                    for x in b[0].floor() as i32..=b[2].ceil() as i32 {
                        let p = (x as f32 + 0.5, y as f32 + 0.5);
                        let q = material.ellipse_coordinate(p);
                        if q.0 * q.0 + q.1 * q.1 > 1.0
                            || feature_polygon_signed_distance(&material.aperture, p) > 0.0
                        {
                            continue;
                        }
                        let a = full(p);
                        let r = reference(p);
                        let error = (a.0 - r.0).hypot(a.1 - r.1);
                        n += 1;
                        sum += (error as f64).powi(2);
                        if error > max_error {
                            max_error = error;
                            max_at = p;
                        }
                        for h in [(0.25, 0.0), (0.0, 0.25)] {
                            let ap = full((p.0 + h.0, p.1 + h.1));
                            let am = full((p.0 - h.0, p.1 - h.1));
                            let rp = reference((p.0 + h.0, p.1 + h.1));
                            let rm = reference((p.0 - h.0, p.1 - h.1));
                            derivative_error = derivative_error.max(
                                ((ap.0 - am.0) - (rp.0 - rm.0))
                                    .hypot((ap.1 - am.1) - (rp.1 - rm.1))
                                    / 0.5,
                            );
                        }
                    }
                }
                records.push(json!({"observedIrisSamples":n,"maxAdditionalMotionFromWidthPx":max_error,"rmsAdditionalMotionFromWidthPx":(sum/n as f64).sqrt(),"maximumAt":max_at,"maxAdditionalJacobianColumnNorm":derivative_error,"comparison":"actual full mapper forward flow versus same other fields with eyeWidth removed; observed iris projection intersect source aperture"}));
            }
            if !records.is_empty() {
                if let Some(path) = material_path {
                    std::fs::write(
                        format!("{}.material.json", path),
                        serde_json::to_vec_pretty(&records).unwrap(),
                    )
                    .unwrap();
                }
                if RECORD_WIDTH_MATERIAL.get() {
                    LAST_WIDTH_MATERIAL_REPORT.set(Some(json!(records)));
                }
            }
        }

        let path = std::env::var_os("BROW_FLOW_DIAGNOSTICS_OUT");
        if path.is_none() && !RECORD_FLOW.get() {
            return;
        }
        let x0 = changes
            .iter()
            .map(|w| w.center.0 - w.radius.0.max(w.radius.1) - 3.0)
            .chain(
                post_warps
                    .iter()
                    .map(|w| w.target.0 - w.radius.0.max(w.radius.1) - 3.0),
            )
            .fold(source.width as f32, f32::min)
            .floor()
            .max(1.0) as u32;
        let y0 = changes
            .iter()
            .map(|w| w.center.1 - w.radius.0.max(w.radius.1) - 3.0)
            .chain(
                post_warps
                    .iter()
                    .map(|w| w.target.1 - w.radius.0.max(w.radius.1) - 3.0),
            )
            .fold(source.height as f32, f32::min)
            .floor()
            .max(1.0) as u32;
        let x1 = changes
            .iter()
            .map(|w| w.center.0 + w.radius.0.max(w.radius.1) + 3.0)
            .chain(
                post_warps
                    .iter()
                    .map(|w| w.target.0 + w.radius.0.max(w.radius.1) + 3.0),
            )
            .fold(0.0, f32::max)
            .ceil()
            .min(source.width as f32 - 2.0) as u32;
        let y1 = changes
            .iter()
            .map(|w| w.center.1 + w.radius.0.max(w.radius.1) + 3.0)
            .chain(
                post_warps
                    .iter()
                    .map(|w| w.target.1 + w.radius.0.max(w.radius.1) + 3.0),
            )
            .fold(0.0, f32::max)
            .ceil()
            .min(source.height as f32 - 2.0) as u32;
        let eye_masks: Vec<_> = ["leftEye", "rightEye"]
            .iter()
            .filter_map(|name| params.get("semanticMasks").and_then(|m| m.get(name)))
            .map(PreparedMask::new)
            .collect();
        let eye_pixel = |x: f32, y: f32| {
            eye_masks.iter().any(|m| {
                m.coverage(
                    (x + 0.5) / source.width as f32,
                    (y + 0.5) / source.height as f32,
                    source.width,
                    source.height,
                ) >= 0.5
            })
        };
        let other: Vec<_> = changes
            .iter()
            .cloned()
            .filter(|w| !w.preserve_eyes)
            .collect();
        let mut minimum = f32::INFINITY;
        let mut minimum_at = (0, 0);
        let mut folds = 0usize;
        let mut eye_donors = 0usize;
        let mut fixed_donors = 0usize;
        let mut fixed_movement = 0.0f32;
        let mut reverse_at = (0, 0);
        let mut maximum_eye_deviation = 0.0f32;
        let mut reverse_error = 0.0f32;
        let mut maximum_displacement = 0.0f32;
        let derivative = 0.25;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let point = (x as f32, y as f32);
                let mapped = mapping(point.0, point.1);
                let xp = mapping(point.0 + derivative, point.1);
                let xm = mapping(point.0 - derivative, point.1);
                let yp = mapping(point.0, point.1 + derivative);
                let ym = mapping(point.0, point.1 - derivative);
                let jacobian = ((xp.0 - xm.0) * (yp.1 - ym.1) - (xp.1 - xm.1) * (yp.0 - ym.0))
                    / (4.0 * derivative * derivative);
                if jacobian < minimum {
                    minimum = jacobian;
                    minimum_at = (x, y);
                }
                folds += usize::from(jacobian <= 0.0);
                let displacement = (mapped.0 - point.0).hypot(mapped.1 - point.1);
                maximum_displacement = maximum_displacement.max(displacement);
                let reversed = integrate_feature_flow(
                    mapped, changes, eyes, post_warps, coverage, steps, -1.0,
                );
                let error = (reversed.0 - point.0).hypot(reversed.1 - point.1);
                if error > reverse_error {
                    reverse_error = error;
                    reverse_at = (x, y);
                }
                if stationary
                    .get(y as usize * source.width as usize + x as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    fixed_movement = fixed_movement.max(displacement);
                }
                if displacement > 0.1 {
                    let bx = mapped.0.floor();
                    let by = mapped.1.floor();
                    let tx = mapped.0 - bx;
                    let ty = mapped.1 - by;
                    fixed_donors += usize::from(
                        [
                            (0.0, 0.0, (1.0 - tx) * (1.0 - ty)),
                            (1.0, 0.0, tx * (1.0 - ty)),
                            (0.0, 1.0, (1.0 - tx) * ty),
                            (1.0, 1.0, tx * ty),
                        ]
                        .iter()
                        .any(|&(dx, dy, w)| {
                            let xx = (bx + dx).clamp(0.0, source.width as f32 - 1.0) as usize;
                            let yy = (by + dy).clamp(0.0, source.height as f32 - 1.0) as usize;
                            w > 1e-6
                                && stationary
                                    .get(yy * source.width as usize + xx)
                                    .copied()
                                    .unwrap_or(false)
                        }),
                    );
                }
                if eyes
                    .iter()
                    .any(|eye| eye.surrounding_motion(point.0, point.1) == 0.0)
                {
                    let expected = integrate_feature_flow(
                        point, &other, eyes, post_warps, coverage, steps, 1.0,
                    );
                    maximum_eye_deviation = maximum_eye_deviation
                        .max((mapped.0 - expected.0).hypot(mapped.1 - expected.1));
                }
                if displacement > 0.1 && !eye_pixel(point.0, point.1) {
                    let bx = mapped.0.floor();
                    let by = mapped.1.floor();
                    let tx = mapped.0 - bx;
                    let ty = mapped.1 - by;
                    eye_donors += usize::from(
                        [
                            (0.0, 0.0, (1.0 - tx) * (1.0 - ty)),
                            (1.0, 0.0, tx * (1.0 - ty)),
                            (0.0, 1.0, (1.0 - tx) * ty),
                            (1.0, 1.0, tx * ty),
                        ]
                        .iter()
                        .any(|&(dx, dy, w)| w > 1e-6 && eye_pixel(bx + dx, by + dy)),
                    );
                }
            }
        }
        let report = json!({
            "scope":"Actual final Rust f32 feature_mapping; native pixel sampling coordinates; central derivative h=0.25px; full active support bounding rectangle. Other post fields are included.",
            "region":[x0,y0,x1,y1],"steps":steps,"minimumJacobian":minimum,"minimumJacobianXY":[minimum_at.0,minimum_at.1],"foldPixels":folds,
            "maxEyeEnvelopeDeviationFromOtherEditsPx":maximum_eye_deviation,"eyeDonorImportsOutsideOriginalEyes":eye_donors,
            "maxInverseFlowReverseErrorPx":reverse_error,"maxInverseFlowReverseErrorXY":[reverse_at.0,reverse_at.1],"maxDisplacementPx":maximum_displacement,
            "fixedDonorImports":fixed_donors,"maxFixedPixelMovementPx":fixed_movement,
            "adjustments":params.get("adjustments")
        });
        if let Some(path) = path {
            std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
        if RECORD_FLOW.get() {
            LAST_FLOW_REPORT.set(Some(report));
        }
    }

    #[test]
    fn stationary_rotation_generator_preserves_radius_area_and_requested_angle() {
        let coverage = |_: f32, _: f32| 1.0;
        for angle in [-0.8, -0.3, 0.3, 0.8] {
            let warp = LocalWarp {
                center: (0.0, 0.0),
                across: (1.0, 0.0),
                radius: (32.0, 32.0),
                scale: (0.0, 0.0),
                translation: (0.0, 0.0),
                angle,
                preserve_eyes: false,
                whole_eye_size: false,
                whole_eye_tilt: false,
                whole_eye_spacing: false,
                eye_width_material: None,
                whole_eye_inner: 0.0,
                eye_size_brows: None,
            };
            let flow = |point| {
                integrate_feature_flow(point, &[warp.clone()], &[], &[], &coverage, 64, 1.0)
            };
            let point = flow((10.0, 0.0));
            let expected_angle = angle * (1.0f32 - (10.0f32 / 32.0).powi(2)).powi(2);
            assert!(
                (point.0.hypot(point.1) - 10.0).abs() < 0.00005,
                "angle={angle}: unintended radial scaling {point:?}"
            );
            assert!(
                (point.1.atan2(point.0) - expected_angle).abs() < 0.00001,
                "angle={angle}: requested angular velocity was altered"
            );
            let h = 0.001;
            let xp = flow((h, 0.0));
            let xm = flow((-h, 0.0));
            let yp = flow((0.0, h));
            let ym = flow((0.0, -h));
            let det =
                ((xp.0 - xm.0) * (yp.1 - ym.1) - (xp.1 - xm.1) * (yp.0 - ym.0)) / (4.0 * h * h);
            assert!(
                (det - 1.0).abs() < 0.00001,
                "angle={angle}: center area {det}"
            );
        }
    }

    #[test]
    #[ignore = "requires local photo-derived landmarks; run make rust-photo-test"]
    fn local_photo_detected_eye_tilt_keeps_center_area_and_angle_independent_of_eye_size() {
        let fixture = local_photo_feature_fixture();
        let image = feature_fixture();
        for amount in [-100, 100] {
            let mut params = fixture["params"].clone();
            params["adjustments"] = json!({"eyeTilt":amount});
            RECORD_ROTATION.set(true);
            local_adjust_kind(&image, &params, "feature.adjust").unwrap();
            RECORD_ROTATION.set(false);
            let report = LAST_ROTATION_REPORT.take().unwrap();
            eprintln!("eyeTilt={amount}: {report}");
            for row in report.as_array().unwrap() {
                assert!(
                    (row["centerInverseJacobian"].as_f64().unwrap() - 1.0).abs() < 0.003,
                    "eyeTilt={amount}: {row}"
                );
                assert!(
                    (row["centerAngleRadians"].as_f64().unwrap()
                        - row["requestedAngleRadians"].as_f64().unwrap())
                    .abs()
                        < 0.002,
                    "eyeTilt={amount}: {row}"
                );
            }
        }
    }

    #[test]
    fn public_semantic_graph_keeps_identity_rgba_exact_with_nonuniform_alpha() {
        let width = 96;
        let height = 96;
        let pixels = (0..width * height)
            .flat_map(|i| {
                let x = i % width;
                let y = i / width;
                [
                    ((x * 11 + y * 3) % 256) as u8,
                    ((x * 7 + y * 13) % 256) as u8,
                    ((x * 19 + y * 17) % 256) as u8,
                    if (x + y) % 3 == 0 {
                        0
                    } else if (x + y) % 3 == 1 {
                        1
                    } else {
                        ((x * 5 + y * 17) % 256) as u8
                    },
                ]
            })
            .collect();
        let image = RgbaImage::new(width, height, pixels).unwrap();
        let eye = |x| {
            json!({"shape":"polygon","points":[
            {"x":x,"y":0.3},{"x":x+0.2,"y":0.3},
            {"x":x+0.2,"y":0.4},{"x":x,"y":0.4}]})
        };
        let frozen = json!({"shape":"rect","region":{"x":0.79,"y":0.79,"width":0.1,"height":0.1}});
        for constraint in ["accessories", "hair", "ears", "protection", "selection"] {
            let mut params = json!({"adjustments":{"eyeSize":100},"semanticMasks":{
                "leftEye":eye(0.25),"rightEye":eye(0.65)}});
            match constraint {
                "protection" => params["protectMask"] = frozen.clone(),
                "selection" => {
                    params["mask"] = json!({"shape":"rect","region":{"x":0.15,"y":0.15,"width":0.75,"height":0.45}})
                }
                name => params["semanticMasks"][name] = frozen.clone(),
            }
            let graph = json!({"operations":[{"type":"feature.adjust","params":params}]});
            RECORD_MAPPING.set(true);
            let output = crate::apply_graph_json(&image, &graph.to_string()).unwrap();
            RECORD_MAPPING.set(false);
            let mapping = LAST_MAPPING.take().unwrap();
            let mut identities = 0;
            let mut hidden = 0;
            let mut moving = 0;
            for (i, &mapped) in mapping.iter().enumerate() {
                let point = ((i as u32 % width) as f32, (i as u32 / width) as f32);
                if mapped == point {
                    identities += 1;
                    hidden += usize::from(image.pixels[i * 4 + 3] == 0);
                    assert_eq!(
                        &output.pixels[i * 4..i * 4 + 4],
                        &image.pixels[i * 4..i * 4 + 4],
                        "{constraint}: identity RGBA at {point:?}"
                    );
                } else if output.pixels[i * 4..i * 4 + 4] != image.pixels[i * 4..i * 4 + 4] {
                    moving += 1;
                }
            }
            assert!(
                identities > 500 && hidden > 100 && moving > 10,
                "{constraint}: must exercise exact identities and real edits"
            );
        }
    }

    #[test]
    fn identity_remap_copies_hidden_rgb_and_checks_expanded_source_bounds() {
        let image = source();
        let output = remap(&image, 3, 3, false, |x, y| (x, y)).unwrap();
        for y in 0..3 {
            for x in 0..3 {
                let index = (y * 3 + x) * 4;
                if x < 2 && y < 2 {
                    let original = (y * 2 + x) * 4;
                    assert_eq!(
                        &output.pixels[index..index + 4],
                        &image.pixels[original..original + 4]
                    );
                } else {
                    assert_eq!(&output.pixels[index..index + 4], &[0, 0, 0, 0]);
                }
            }
        }
    }

    #[test]
    #[ignore = "Photographic QA requires BROW_FLOW_CASES with immutable input/graph paths"]
    fn photographic_brow_flow_diagnostics() {
        let cases: Value = serde_json::from_slice(
            &std::fs::read(std::env::var("BROW_FLOW_CASES").unwrap()).unwrap(),
        )
        .unwrap();
        let mut failures = Vec::new();
        for case in cases.as_array().unwrap() {
            let source = from_native(
                image::open(case["input"].as_str().unwrap())
                    .unwrap()
                    .to_rgba8(),
            );
            let graph: Value =
                serde_json::from_slice(&std::fs::read(case["graph"].as_str().unwrap()).unwrap())
                    .unwrap();
            let params = &graph["operations"][0]["params"];
            std::env::set_var(
                "BROW_FLOW_DIAGNOSTICS_OUT",
                case["diagnostics"].as_str().unwrap(),
            );
            let result = local_adjust_kind(&source, params, "feature.adjust").unwrap();
            std::env::remove_var("BROW_FLOW_DIAGNOSTICS_OUT");
            as_native(&result)
                .save(case["output"].as_str().unwrap())
                .unwrap();
            let report: Value = serde_json::from_slice(
                &std::fs::read(case["diagnostics"].as_str().unwrap()).unwrap(),
            )
            .unwrap();
            eprintln!("{} {}", case["label"].as_str().unwrap(), report);
            if report["fixedDonorImports"] != 0 {
                failures.push(format!("{}: fixed donor imported", case["label"]));
            }
            if report["maxInverseFlowReverseErrorPx"].as_f64().unwrap() >= 0.05 {
                failures.push(format!("{}: inverse error", case["label"]));
            }
            if report["foldPixels"] != 0 {
                failures.push(format!("{}: final mapping folded", case["label"]));
            }
            if report["maxEyeEnvelopeDeviationFromOtherEditsPx"]
                .as_f64()
                .unwrap()
                >= 0.05
            {
                failures.push(format!(
                    "{}: non-target eye envelope changed",
                    case["label"]
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
    /// Real detector coordinates are kept locally, separate from source control.
    #[test]
    #[ignore = "requires local photo-derived landmarks; run make rust-photo-test"]
    fn local_photo_detected_brow_final_mapping_stays_invertible_at_extremes_and_boundaries() {
        let fixture = local_photo_feature_fixture();
        let base = fixture["params"].clone();
        let mut cases = Vec::new();
        for key in [
            "browPeak",
            "browTail",
            "browLength",
            "browWidth",
            "browThickness",
            "browSpacing",
            "eyeSize",
            "eyeWidth",
            "eyeHeight",
            "eyeSpacing",
            "eyeTilt",
            "eyeCorner",
            "eyelid",
            "eyeInnerCorner",
            "eyeOuterCorner",
            "noseBridge",
            "noseWings",
            "noseLength",
            "noseStraight",
            "noseTip",
            "noseRoot",
            "noseBase",
            "nostril",
            "mouthWidth",
            "mouthHeight",
            "upperLip",
            "lowerLip",
            "philtrum",
            "lipPeak",
            "lipCorner",
        ] {
            for value in [-100, 100] {
                let mut p = base.clone();
                p["adjustments"] = json!({key:value});
                cases.push((
                    format!("{key}={value}"),
                    p,
                    256,
                    256,
                    key.starts_with("brow"),
                ));
            }
        }
        for key in ["eyeSize", "eyeSpacing", "noseStraight"] {
            for value in [-100, 100] {
                let mut p = base.clone();
                p["adjustments"] = json!({"browPeak":value,"browTail":value,key:value});
                cases.push((
                    format!("brow+{key}={value}"),
                    p,
                    256,
                    256,
                    key == "noseStraight",
                ));
            }
        }
        for (label, adjustments) in [
            ("symmetry100", json!({"symmetry":100})),
            ("symmetry100-eye100", json!({"symmetry":100,"eyeSize":100})),
            (
                "symmetry100-brow100",
                json!({"symmetry":100,"browPeak":100,"browTail":100}),
            ),
            (
                "symmetry100-mixed",
                json!({"symmetry":100,"browPeak":100,"browTail":-100,"eyeSize":-100,"eyeSpacing":100}),
            ),
        ] {
            let mut p = base.clone();
            p["adjustments"] = adjustments;
            cases.push((label.to_owned(), p, 256, 256, false));
        }
        for feather in [0, 8] {
            for kind in ["selection", "protection", "region", "obstruction"] {
                let mut p = base.clone();
                p["adjustments"] = json!({"browPeak":100,"browTail":100});
                let mask = json!({"shape":"rect","region":{"x":0.56,"y":0.25,"width":0.08,"height":0.095},"feather":feather});
                match kind {
                    "selection" => p["mask"] = mask,
                    "protection" => p["protectMask"] = mask,
                    "region" => {
                        p["region"] = mask["region"].clone();
                        p.as_object_mut().unwrap().remove("mask");
                    }
                    _ => p["semanticMasks"]["accessories"] = mask,
                }
                cases.push((format!("{kind}-feather={feather}"), p, 256, 256, true));
            }
        }
        for (label, width, height, angle) in [
            ("roll30", 256, 256, std::f32::consts::PI / 6.0),
            ("tall", 256, 384, 0.0),
        ] {
            for key in ["browPeak", "browTail"] {
                for value in [-100, 100] {
                    let mut p = base.clone();
                    p["adjustments"] = json!({key:value});
                    let transform = |point: &mut Value| {
                        let x = point["x"].as_f64().unwrap() as f32 * 256.0 - 128.0;
                        let y = point["y"].as_f64().unwrap() as f32 * 256.0 - 128.0;
                        point["x"] = json!(
                            (x * angle.cos() - y * angle.sin() + width as f32 * 0.5) / width as f32
                        );
                        point["y"] = json!(
                            (x * angle.sin() + y * angle.cos() + height as f32 * 0.5)
                                / height as f32
                        );
                    };
                    for mask in p["semanticMasks"].as_object_mut().unwrap().values_mut() {
                        for point in mask["points"].as_array_mut().unwrap() {
                            transform(point);
                        }
                    }
                    if let Some(points) = p["landmarks"].as_array_mut() {
                        for point in points {
                            transform(point);
                        }
                    }
                    for point in p["mask"]["points"].as_array_mut().unwrap() {
                        transform(point);
                    }
                    // A rotated rectangle's axis-aligned enclosing region must
                    // contain the transformed authoritative polygon selection.
                    let points = p["mask"]["points"].as_array().unwrap();
                    let min_x = points
                        .iter()
                        .map(|v| v["x"].as_f64().unwrap())
                        .fold(f64::INFINITY, f64::min);
                    let max_x = points
                        .iter()
                        .map(|v| v["x"].as_f64().unwrap())
                        .fold(f64::NEG_INFINITY, f64::max);
                    let min_y = points
                        .iter()
                        .map(|v| v["y"].as_f64().unwrap())
                        .fold(f64::INFINITY, f64::min);
                    let max_y = points
                        .iter()
                        .map(|v| v["y"].as_f64().unwrap())
                        .fold(f64::NEG_INFINITY, f64::max);
                    p["region"] =
                        json!({"x":min_x,"y":min_y,"width":max_x-min_x,"height":max_y-min_y});
                    cases.push((format!("{label}-{key}={value}"), p, width, height, true));
                }
            }
        }
        for (label, params, width, height, brow_only) in cases {
            let pixels = (0..width * height)
                .flat_map(|i| {
                    let x = i % width;
                    let y = i / width;
                    [
                        ((x * 11 + y * 3) % 256) as u8,
                        ((x * 7 + y * 13) % 256) as u8,
                        ((x * 19 + y * 17) % 256) as u8,
                        211,
                    ]
                })
                .collect();
            let image = RgbaImage::new(width, height, pixels).unwrap();
            LAST_FLOW_REPORT.set(None);
            RECORD_FLOW.set(true);
            let output = local_adjust_kind(&image, &params, "feature.adjust").unwrap();
            RECORD_FLOW.set(false);
            let report = LAST_FLOW_REPORT
                .take()
                .expect("actual final mapper diagnostic must run");
            assert_eq!(report["foldPixels"], 0, "{label}: {report}");
            assert_eq!(report["fixedDonorImports"], 0, "{label}: {report}");
            assert_eq!(report["maxFixedPixelMovementPx"], 0.0, "{label}: {report}");
            assert!(
                report["minimumJacobian"].as_f64().unwrap() > 0.01,
                "{label}: {report}"
            );
            assert!(
                report["maxInverseFlowReverseErrorPx"].as_f64().unwrap() < 0.05,
                "{label}: {report}"
            );
            assert!(
                report["maxEyeEnvelopeDeviationFromOtherEditsPx"]
                    .as_f64()
                    .unwrap()
                    < 0.05,
                "{label}: {report}"
            );
            if brow_only {
                assert_eq!(
                    report["eyeDonorImportsOutsideOriginalEyes"], 0,
                    "{label}: {report}"
                );
            }
            assert!(
                report["maxDisplacementPx"].as_f64().unwrap() > 0.1,
                "{label}: target edit must remain visible"
            );
            assert!(
                output
                    .pixels
                    .chunks_exact(4)
                    .zip(image.pixels.chunks_exact(4))
                    .any(|(a, b)| a != b),
                "{label}: actual pixels must change"
            );
            assert!(
                output.pixels.chunks_exact(4).all(|p| p[3] == 211),
                "{label}: alpha must remain exact"
            );
        }
    }

    #[test]
    #[ignore = "requires local photo-derived landmarks; run make rust-photo-test"]
    fn local_photo_tiny_feature_family_edits_are_continuous_in_final_mapping_and_pixels() {
        let fixture = local_photo_feature_fixture();
        for (active, value, introduced) in [
            ("eyeSize", 100, "browPeak"),
            ("eyeSize", 30, "browPeak"),
            ("eyeSize", 30, "noseBridge"),
            ("browPeak", 30, "eyeSize"),
            ("eyeSize", 100, "symmetry"),
            ("browPeak", 100, "symmetry"),
            ("symmetry", 100, "browPeak"),
            ("symmetry", 100, "eyeSize"),
            ("noseStraight", 100, "symmetry"),
        ] {
            let mut params = fixture["params"].clone();
            params["adjustments"] = json!({active:value,introduced:0.0});
            params["semanticMasks"]["accessories"] =
                json!({"shape":"rect","region":{"x":0.55,"y":0.30,"width":0.02,"height":0.08}});
            let source = feature_fixture();
            RECORD_MAPPING.set(true);
            let before = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            let before_mapping = LAST_MAPPING.take().unwrap();
            params["adjustments"][introduced] = json!(0.001);
            let after = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
            let after_mapping = LAST_MAPPING.take().unwrap();
            RECORD_MAPPING.set(false);
            let max_motion_change = before_mapping
                .iter()
                .zip(&after_mapping)
                .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                .fold(0.0f32, f32::max);
            let max_byte_change = before
                .pixels
                .iter()
                .zip(&after.pixels)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(max_motion_change < 0.01, "{active}={value} + {introduced} 0→0.001 switched the final mapper by {max_motion_change}px");
            assert!(
                max_byte_change <= 1,
                "{active}={value} + {introduced} 0→0.001 switched RGBA by {max_byte_change}/255"
            );
        }
    }

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

    #[test]
    fn color_only_feature_path_respects_semantic_freezing_and_graph_masks() {
        let mut pixels = Vec::new();
        for y in 0..32 {
            for x in 0..64 {
                pixels.extend_from_slice(&[110, 85, 75, if (x + y) % 19 == 0 { 0 } else { 123 }]);
            }
        }
        let source = RgbaImage::new(64, 32, pixels).unwrap();
        let p = json!({"adjustments":{"browColor":100},
        "mask":{"shape":"rect","region":{"x":0.0625,"y":0,"width":0.875,"height":1}},
        "protectMask":{"shape":"rect","region":{"x":0.5,"y":0,"width":0.0625,"height":1}},
        "semanticMasks":{
            "leftBrow":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
            "hair":{"shape":"rect","region":{"x":0.125,"y":0,"width":0.125,"height":1}},
            "ears":{"shape":"rect","region":{"x":0.25,"y":0,"width":0.0625,"height":1}},
            "accessories":{"shape":"rect","region":{"x":0.375,"y":0,"width":0.0625,"height":1}}
        }});
        let processed = local_adjust_kind(&source, &p, "feature.adjust").unwrap();
        let output = crate::mask::composite(&source, &processed, &p);
        let color = crate::color::local_color(&source, &p, false);
        let mut changed = 0;
        for y in 0..32 {
            for x in 0..64 {
                let i = (y * 64 + x) * 4;
                let frozen = !(4..60).contains(&x)
                    || (8..20).contains(&x)
                    || (24..28).contains(&x)
                    || (32..36).contains(&x);
                if frozen {
                    assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
                } else {
                    assert_eq!(&output.pixels[i..i + 4], &color.pixels[i..i + 4]);
                    changed += usize::from(output.pixels[i..i + 4] != source.pixels[i..i + 4]);
                }
                assert_eq!(output.pixels[i + 3], source.pixels[i + 3]);
            }
        }
        assert!(
            changed > 100,
            "valid brow pixels must remain editable in the fast path"
        );
    }

    fn heal_texture_fixture() -> RgbaImage {
        let mut pixels = Vec::with_capacity(128 * 128 * 4);
        for y in 0..128 {
            for x in 0..128 {
                let dx = x as f32 - 64.0;
                let dy = y as f32 - 64.0;
                // A harmonic lighting field gives a known clean reference.
                // The donor's illumination slope differs from the target's.
                let light = 0.03 * (dx * dx - dy * dy);
                let pore = if (x + y * 3) % 8 < 2 { -5.0 } else { 2.0 };
                pixels.extend_from_slice(&[
                    byte(140.0 + light + pore),
                    byte(116.0 + light + pore),
                    byte(103.0 + light + pore),
                    77 + ((x + y) % 7) as u8,
                ]);
            }
        }
        RgbaImage::new(128, 128, pixels).unwrap()
    }

    fn add_heal_blemish(image: &mut RgbaImage) {
        for y in 60..69 {
            for x in 60..69 {
                if (x as f32 - 64.0).hypot(y as f32 - 64.0) <= 4.0 {
                    let i = (y * 128 + x) * 4;
                    image.pixels[i..i + 3].copy_from_slice(&[245, 38, 29]);
                }
            }
        }
    }

    #[test]
    fn heal_poisson_recovers_reference_lighting_and_keeps_donor_pores() {
        let clean = heal_texture_fixture();
        let mut damaged = clean.clone();
        add_heal_blemish(&mut damaged);
        let params = json!({"strokes":[{"x":0.5,"y":0.5,"sourceX":0.25,"sourceY":0.5,
            "radius":0.078125,"strength":100}]});
        let out = repair(&damaged, &params, true).unwrap();
        let mut error = 0.0;
        let mut count = 0;
        for y in 0..128 {
            for x in 0..128 {
                let i = (y * 128 + x) * 4;
                let d = (x as f32 - 64.0).hypot(y as f32 - 64.0);
                assert_eq!(out.pixels[i + 3], damaged.pixels[i + 3]);
                if d >= 10.0 {
                    assert_eq!(&out.pixels[i..i + 4], &damaged.pixels[i..i + 4]);
                }
                if d <= 7.0 {
                    for c in 0..3 {
                        error += (out.pixels[i + c] as f32 - clean.pixels[i + c] as f32).powi(2);
                        count += 1;
                    }
                }
            }
        }
        let mse = error / count as f32;
        assert!(
            mse < 1.8,
            "clean-reference MSE {mse}; the old flat correction loses the lighting gradient"
        );
        // Two adjacent known pore phases retain their original contrast.
        let a = (64 * 128 + 64) * 4;
        let b = (64 * 128 + 66) * 4;
        assert_eq!(
            out.pixels[b] as i16 - out.pixels[a] as i16,
            clean.pixels[b] as i16 - clean.pixels[a] as i16
        );
    }

    #[test]
    fn automatic_heal_removes_defect_without_flattening_clean_texture() {
        let clean = heal_texture_fixture();
        let mut damaged = clean.clone();
        add_heal_blemish(&mut damaged);
        let out = repair(
            &damaged,
            &json!({"strokes":[{"x":0.5,"y":0.5,
            "radius":0.078125,"strength":100}]}),
            true,
        )
        .unwrap();
        let mut error = 0.0;
        let mut count = 0;
        let mut residuals = Vec::new();
        for y in 59..70 {
            for x in 59..70 {
                if (x as f32 - 64.0).hypot(y as f32 - 64.0) > 5.0 {
                    continue;
                }
                let i = (y * 128 + x) * 4;
                for c in 0..3 {
                    error += (out.pixels[i + c] as f32 - clean.pixels[i + c] as f32).powi(2);
                    count += 1;
                }
                residuals.push(
                    out.pixels[i] as f32
                        - 140.0
                        - 0.03 * ((x as f32 - 64.0).powi(2) - (y as f32 - 64.0).powi(2)),
                );
            }
        }
        let mse = error / count as f32;
        assert!(
            mse < 16.0,
            "automatic heal must restore known normal skin, MSE {mse}"
        );
        let mean = residuals.iter().sum::<f32>() / residuals.len() as f32;
        let variance =
            residuals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / residuals.len() as f32;
        assert!(
            variance > 5.0,
            "donor pore variance {variance}; a flat fill is not texture repair"
        );
    }

    #[test]
    fn automatic_heal_ignores_hidden_rgb_and_preserves_transparent_destination() {
        let mut a = heal_texture_fixture();
        add_heal_blemish(&mut a);
        for y in 0..128 {
            for x in 0..128 {
                if x > 77 || (x == 64 && y == 64) {
                    let i = (y * 128 + x) * 4;
                    a.pixels[i..i + 4].copy_from_slice(&[255, 0, 220, 0]);
                }
            }
        }
        let mut b = a.clone();
        for pixel in b.pixels.chunks_exact_mut(4) {
            if pixel[3] == 0 {
                pixel[..3].copy_from_slice(&[0, 255, 0]);
            }
        }
        let params = json!({"strokes":[{"x":0.5,"y":0.5,"radius":0.078125,"strength":100}]});
        let oa = repair(&a, &params, true).unwrap();
        let ob = repair(&b, &params, true).unwrap();
        for i in (0..a.pixels.len()).step_by(4) {
            assert_eq!(oa.pixels[i + 3], a.pixels[i + 3]);
            assert_eq!(ob.pixels[i + 3], b.pixels[i + 3]);
            if a.pixels[i + 3] == 0 {
                assert_eq!(&oa.pixels[i..i + 4], &a.pixels[i..i + 4]);
                assert_eq!(&ob.pixels[i..i + 4], &b.pixels[i..i + 4]);
            } else {
                assert_eq!(&oa.pixels[i..i + 4], &ob.pixels[i..i + 4]);
            }
        }
        let i = (64 * 128 + 65) * 4;
        assert!(
            oa.pixels[i + 1] > 95,
            "visible neighboring blemish should still be healed"
        );
    }

    #[test]
    fn automatic_heal_excludes_protected_source_texels() {
        let mut a = heal_texture_fixture();
        add_heal_blemish(&mut a);
        let mut b = a.clone();
        for y in 0..128 {
            for x in 0..49 {
                let i = (y * 128 + x) * 4;
                b.pixels[i..i + 3].copy_from_slice(&[0, 255, 0]);
            }
        }
        let params = json!({"protectMask":{"shape":"rect","region":{"x":0.0,"y":0.0,"width":0.38,"height":1.0}},
            "strokes":[{"x":0.5,"y":0.5,"radius":0.078125,"strength":100}]});
        let oa = repair(&a, &params, true).unwrap();
        let ob = repair(&b, &params, true).unwrap();
        for y in 0..128 {
            for x in 49..128 {
                let i = (y * 128 + x) * 4;
                assert_eq!(
                    &oa.pixels[i..i + 4],
                    &ob.pixels[i..i + 4],
                    "protected source contamination at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn manual_repair_reads_selected_frozen_source_without_writing_to_it() {
        let mut image = heal_texture_fixture();
        add_heal_blemish(&mut image);
        let center = (64 * 128 + 64) * 4;
        for kind in ["repair.clone", "repair.heal"] {
            let graph = json!({"operations":[{"type":kind,"params":{
                "protectMask":{"shape":"rect","region":{"x":0,"y":0,"width":0.4,"height":1}},
                "strokes":[{"x":0.5,"y":0.5,"sourceX":0.25,"sourceY":0.5,"radius":0.078125,"strength":100}]
            }}]});
            let output = crate::apply_graph_json(&image, &graph.to_string()).unwrap();
            assert_ne!(
                &output.pixels[center..center + 3],
                &image.pixels[center..center + 3],
                "{kind} must apply selected source"
            );
            for y in 0..128 {
                for x in 0..51 {
                    let i = (y * 128 + x) * 4;
                    assert_eq!(
                        &output.pixels[i..i + 4],
                        &image.pixels[i..i + 4],
                        "{kind}: frozen {x},{y}"
                    );
                }
            }
            for (before, after) in image
                .pixels
                .chunks_exact(4)
                .zip(output.pixels.chunks_exact(4))
            {
                assert_eq!(before[3], after[3]);
            }
        }
    }

    #[test]
    fn invisible_heal_cannot_fail_or_block_a_later_visible_edit() {
        let image = RgbaImage::new(32, 32, [90, 80, 70, 255].repeat(32 * 32)).unwrap();
        let bad_source =
            json!({"strokes":[{"x":0.5,"y":0.5,"sourceX":0.01,"sourceY":0.01,"radius":0.2}]});
        let expected = crate::apply_graph_json(
            &image,
            r#"{"operations":[{"type":"color.basic","params":{"exposure":30}}]}"#,
        )
        .unwrap();
        let restrictions = [
            json!({"opacity":0}),
            json!({"protectMask":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}}),
            json!({"mask":{"shape":"raster","width":1,"height":1,"data":[0]}}),
            json!({"region":{"x":0,"y":0,"width":0.1,"height":0.1}}),
        ];
        for restriction in restrictions {
            let mut params = bad_source.clone();
            params
                .as_object_mut()
                .unwrap()
                .extend(restriction.as_object().unwrap().clone());
            let noop = json!({"operations":[{"type":"repair.heal","params":params.clone()}]});
            assert_eq!(
                crate::apply_graph_json(&image, &noop.to_string()).unwrap(),
                image
            );
            let chain = json!({"operations":[{"type":"repair.heal","params":params},{"type":"color.basic","params":{"exposure":30}}]});
            assert_eq!(
                crate::apply_graph_json(&image, &chain.to_string()).unwrap(),
                expected
            );
        }
        let visible = json!({"operations":[{"type":"repair.heal","params":bad_source}]});
        assert!(
            crate::apply_graph_json(&image, &visible.to_string()).is_err(),
            "visible invalid donor still needs an error"
        );
    }

    #[test]
    fn heal_reports_missing_donor_instead_of_painting_transparent_black() {
        let image = RgbaImage::new(32, 32, [200, 100, 50, 0].repeat(32 * 32)).unwrap();
        assert!(repair(
            &image,
            &json!({"strokes":[{"x":0.5,"y":0.5,"radius":0.1}]}),
            true
        )
        .is_err());
        assert_eq!(
            repair(
                &image,
                &json!({"strokes":[{"x":0.5,"y":0.5,"radius":0.1,"strength":0}]}),
                true
            )
            .unwrap(),
            image
        );
    }

    #[test]
    fn clone_does_not_import_hidden_rgb_or_change_hidden_destination() {
        let mut image = heal_texture_fixture();
        let donor = (64 * 128 + 32) * 4;
        image.pixels[donor..donor + 4].copy_from_slice(&[255, 0, 255, 0]);
        let hidden_target = (64 * 128 + 66) * 4;
        image.pixels[hidden_target..hidden_target + 4].copy_from_slice(&[77, 88, 99, 0]);
        let out = repair(
            &image,
            &json!({"strokes":[{"x":0.5,"y":0.5,"sourceX":0.25,"sourceY":0.5,
            "radius":0.078125,"strength":100}]}),
            false,
        )
        .unwrap();
        let center = (64 * 128 + 64) * 4;
        assert_eq!(
            &out.pixels[center..center + 4],
            &image.pixels[center..center + 4]
        );
        assert_eq!(
            &out.pixels[hidden_target..hidden_target + 4],
            &image.pixels[hidden_target..hidden_target + 4]
        );
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
    fn eyebrow_reshaping_moves_brows_without_bending_adjacent_eyes() {
        let source = feature_fixture();
        let polygon = |cx: f32, cy: f32, width: f32, height: f32| {
            symmetry_polygon(cx * 256.0, cy * 256.0, width * 256.0, height * 256.0)
        };
        let semantics = json!({
            "leftEye": polygon(0.31, 0.49, 0.18, 0.08),
            "rightEye": polygon(0.69, 0.49, 0.18, 0.08),
            "leftBrow": polygon(0.31, 0.38, 0.25, 0.09),
            "rightBrow": polygon(0.69, 0.38, 0.25, 0.09)
        });
        for key in [
            "browPeak",
            "browTail",
            "browThickness",
            "browLength",
            "browSpacing",
        ] {
            for amount in [-100, 100] {
                let params = json!({"adjustments":{key:amount},"semanticMasks":semantics});
                let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                let mut brow_changes = 0;
                for y in 0..source.height {
                    for x in 0..source.width {
                        let i = (y as usize * source.width as usize + x as usize) * 4;
                        let nx = (x as f32 + 0.5) / source.width as f32;
                        let ny = (y as f32 + 0.5) / source.height as f32;
                        let inside = |name: &str| {
                            crate::mask::coverage(
                                &semantics[name],
                                nx,
                                ny,
                                source.width,
                                source.height,
                            ) >= 0.5
                        };
                        if inside("leftEye") || inside("rightEye") {
                            assert_eq!(
                                &output.pixels[i..i + 4],
                                &source.pixels[i..i + 4],
                                "{key}={amount} bent non-target eye at ({x},{y})"
                            );
                        }
                        if inside("leftBrow") || inside("rightBrow") {
                            brow_changes +=
                                usize::from(output.pixels[i..i + 4] != source.pixels[i..i + 4]);
                        }
                        assert_eq!(output.pixels[i + 3], source.pixels[i + 3]);
                    }
                }
                assert!(
                    brow_changes > 10,
                    "{key}={amount} must still reshape the eyebrows"
                );
            }
        }
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
    fn semantic_stationary_layers_neither_move_nor_get_sampled_into_neighboring_skin() {
        let (mut source, mut params) = symmetry_fixture();
        let mut frozen = vec![0_u8; source.width as usize * source.height as usize];
        // An occluding accessory sits inside the moving feature's support,
        // so output-only freezing would still duplicate it in nearby skin.
        for y in 84..94 {
            for x in 68..74 {
                let i = y * source.width as usize + x;
                frozen[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[0, 255, 0, 255]);
            }
        }
        for key in ["accessories", "ears", "hair"] {
            params["semanticMasks"][key] = serde_json::json!({"shape":"raster","width":source.width,"height":source.height,"data":frozen});
            for adjustments in [
                serde_json::json!({"symmetry":100}),
                serde_json::json!({"eyeSize":100}),
            ] {
                params["adjustments"] = adjustments;
                let output = local_adjust_kind(&source, &params, "feature.adjust").unwrap();
                assert_ne!(
                    output.pixels, source.pixels,
                    "feature must still move outside {key}"
                );
                for (i, &protected) in frozen.iter().enumerate() {
                    if protected > 0 {
                        assert_eq!(
                            &output.pixels[i * 4..i * 4 + 4],
                            &source.pixels[i * 4..i * 4 + 4],
                            "frozen {key} {i}"
                        );
                    } else {
                        assert!(
                            !(output.pixels[i * 4] < 40
                                && output.pixels[i * 4 + 1] > 240
                                && output.pixels[i * 4 + 2] < 40),
                            "{key} was copied into skin pixel {i}"
                        );
                    }
                }
            }
            params["semanticMasks"].as_object_mut().unwrap().remove(key);
        }
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
