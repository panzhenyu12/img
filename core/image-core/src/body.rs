//! Pose-aware local body contour edits. Axes and radii use image pixels so a
//! tall portrait and a square preview have the same geometric meaning.
//! Detector polygons remain approximate selections; they are not parsing.

use crate::engine::{byte, percent};
use crate::mask::{coverage, Region};
use crate::{ImageError, RgbaImage};
use serde_json::Value;

type Point = (f32, f32);

fn sample(image: &RgbaImage, x: f32, y: f32) -> [u8; 4] {
    sample_with_alpha(image, x, y).0
}
fn sample_with_alpha(image: &RgbaImage, x: f32, y: f32) -> ([u8; 4], f32) {
    let x = x.clamp(0.0, image.width as f32 - 1.0);
    let y = y.clamp(0.0, image.height as f32 - 1.0);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let mut color = [0.0f32; 4];
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
        for (channel, value) in color[..3].iter_mut().enumerate() {
            *value += image.pixels[index + channel] as f32 * alpha * weight;
        }
        color[3] += alpha * weight;
    }
    if color[3] <= 1e-6 {
        return ([0, 0, 0, 0], 0.0);
    }
    (
        [
            byte(color[0] / color[3]),
            byte(color[1] / color[3]),
            byte(color[2] / color[3]),
            byte(color[3] * 255.0),
        ],
        color[3],
    )
}
fn sample_coverage(values: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, width as f32 - 1.0);
    let y = y.clamp(0.0, height as f32 - 1.0);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    values[y0 * width + x0] * (1.0 - tx) * (1.0 - ty)
        + values[y0 * width + x1] * tx * (1.0 - ty)
        + values[y1 * width + x0] * (1.0 - tx) * ty
        + values[y1 * width + x1] * tx * ty
}
fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn read_points(value: Option<&Value>, width: f32, height: f32) -> Vec<Point> {
    value
        .and_then(Value::as_array)
        .map(|points| {
            points
                .iter()
                .take(128)
                .filter_map(|p| {
                    let x = p.get("x")?.as_f64()? as f32;
                    let y = p.get("y")?.as_f64()? as f32;
                    if x.is_finite()
                        && y.is_finite()
                        && (0.0..=1.0).contains(&x)
                        && (0.0..=1.0).contains(&y)
                    {
                        Some((x * width, y * height))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}
fn mid(a: Point, b: Point) -> Point {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

#[derive(Clone, Copy, Debug)]
struct Axis {
    origin: Point,
    direction: Point,
    length: f32,
    radius: f32,
}
impl Axis {
    fn new(a: Point, b: Point, radius: f32) -> Option<Self> {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let length = dx.hypot(dy);
        (length >= 1.0).then_some(Self {
            origin: a,
            direction: (dx / length.max(1.0), dy / length.max(1.0)),
            length,
            radius: radius.max(1.0),
        })
    }
    fn coordinates(self, p: Point) -> (f32, f32) {
        let dx = p.0 - self.origin.0;
        let dy = p.1 - self.origin.1;
        (
            (dx * self.direction.0 + dy * self.direction.1) / self.length,
            dx * -self.direction.1 + dy * self.direction.0,
        )
    }
    fn normal(self) -> Point {
        (-self.direction.1, self.direction.0)
    }
}
#[derive(Clone, Copy)]
struct Warp {
    axis: Axis,
    amount: f32,
    center: f32,
    extent: f32,
}
impl Warp {
    fn displacement(self, p: Point) -> Point {
        self.displacement_for(p, false)
    }
    fn displacement_for(self, p: Point, separated_foreground: bool) -> Point {
        let (t, across) = self.axis.coordinates(p);
        let side = (across / self.axis.radius).abs();
        // The unsegmented compatibility path anchors the selection edge. A
        // separated subject instead moves its silhouette: only the wider
        // selection/ROI boundary is anchored, not the foreground boundary.
        let outer = if separated_foreground {
            1.0 - smoothstep(0.95, 1.30, side)
        } else {
            1.0 - smoothstep(0.86, 1.0, side)
        };
        let transverse = smoothstep(0.16, 0.38, side) * outer;
        let endpoint = smoothstep(0.0, 0.12, t) * (1.0 - smoothstep(0.88, 1.0, t));
        let distance = ((t - self.center) / self.extent).abs();
        let profile = if distance < 1.0 {
            (1.0 - distance * distance).powi(2)
        } else {
            0.0
        };
        let shift = across.signum()
            * self.amount
            * self.axis.radius
            * 0.30
            * transverse
            * endpoint
            * profile;
        let normal = self.axis.normal();
        (normal.0 * shift, normal.1 * shift)
    }
}
fn add(warps: &mut Vec<Warp>, axis: Axis, values: &Value, key: &str, center: f32, extent: f32) {
    let amount = percent(values, key);
    if amount.abs() > 1e-6 {
        warps.push(Warp {
            axis,
            amount,
            center,
            extent,
        });
    }
}
fn extent_radius(axis: Axis, polygon: &[Point]) -> f32 {
    polygon
        .iter()
        .map(|&p| axis.coordinates(p).1.abs())
        .fold(0.0, f32::max)
        .max(1.0)
}
fn local_axis(
    part: &str,
    region: Region,
    points: &[Point],
    polygon: &[Point],
    w: f32,
    h: f32,
) -> Axis {
    let center = (
        region.x * w + region.width * w * 0.5,
        region.y * h + region.height * h * 0.5,
    );
    let top = region.y * h;
    let bottom = (region.y + region.height) * h;
    let fallback = Axis::new((center.0, top), (center.0, bottom), region.width * w * 0.5)
        .expect("nonempty region");
    let limb = part.contains("arm") || part.contains("thigh") || part.contains("calf");
    let mut axis = if limb && points.len() >= 2 {
        Axis::new(points[0], points[1], fallback.radius).unwrap_or(fallback)
    } else if part == "waist" && points.len() >= 3 {
        // The region covers only the waist, whereas supplied endpoints
        // cover shoulder→hip. Keep its orientation but use the region ROI.
        let dir = Axis::new(points[0], points[2], fallback.radius)
            .unwrap_or(fallback)
            .direction;
        let length = (region.height * h).max(1.0);
        Axis::new(
            (
                points[1].0 - dir.0 * length * 0.5,
                points[1].1 - dir.1 * length * 0.5,
            ),
            (
                points[1].0 + dir.0 * length * 0.5,
                points[1].1 + dir.1 * length * 0.5,
            ),
            fallback.radius,
        )
        .unwrap_or(fallback)
    } else if part == "hips" && points.len() >= 2 {
        // Left/right hips describe a cross-section, not a length axis.
        // Build the perpendicular through their midpoint over the ROI.
        let m = mid(points[0], points[1]);
        let dx = points[1].0 - points[0].0;
        let dy = points[1].1 - points[0].1;
        let len = dx.hypot(dy).max(1.0);
        let mut dir = (-dy / len, dx / len);
        if dir.1 < 0.0 {
            dir = (-dir.0, -dir.1);
        }
        let length = (region.height * h).max(1.0);
        Axis::new(
            (m.0 - dir.0 * length * 0.5, m.1 - dir.1 * length * 0.5),
            (m.0 + dir.0 * length * 0.5, m.1 + dir.1 * length * 0.5),
            fallback.radius,
        )
        .unwrap_or(fallback)
    } else if part == "torso" && points.len() >= 4 {
        Axis::new(
            mid(points[0], points[1]),
            mid(points[2], points[3]),
            fallback.radius,
        )
        .unwrap_or(fallback)
    } else {
        fallback
    };
    // Project the actual selected polygon into the axis frame. This matches
    // tilted limb masks and their true width instead of guessing from length.
    if polygon.len() >= 3 {
        axis.radius = extent_radius(axis, polygon);
    }
    axis
}
fn limb_warps(warps: &mut Vec<Warp>, axis: Axis, part: &str, values: &Value) {
    let keys: &[&str] = match part {
        "left-upper-arm" | "right-upper-arm" => &["arms"],
        "left-forearm" | "right-forearm" => &["arms", "forearms"],
        "left-thigh" | "right-thigh" => &["legs", "thighs"],
        "left-calf" | "right-calf" => &["legs", "calves"],
        _ => &[],
    };
    for key in keys {
        add(warps, axis, values, key, 0.5, 0.5);
    }
}
fn torso_warps(warps: &mut Vec<Warp>, axis: Axis, values: &Value) {
    add(warps, axis, values, "shoulder", 0.15, 0.20);
    add(warps, axis, values, "waist", 0.60, 0.28);
    add(warps, axis, values, "belly", 0.75, 0.24);
    add(warps, axis, values, "hips", 0.90, 0.22);
}
fn full_body_warps(warps: &mut Vec<Warp>, points: &[Point], values: &Value) {
    if points.len() < 29 {
        return;
    }
    let shoulder = mid(points[11], points[12]);
    let hip = mid(points[23], points[24]);
    let torso_radius = ((points[11].0 - points[12].0).hypot(points[11].1 - points[12].1) * 0.5)
        .max((points[23].0 - points[24].0).hypot(points[23].1 - points[24].1) * 0.58);
    if let Some(axis) = Axis::new(shoulder, hip, torso_radius) {
        torso_warps(warps, axis, values);
    }
    for (a, b, ratio, part) in [
        (11, 13, 0.17, "left-upper-arm"),
        (12, 14, 0.17, "right-upper-arm"),
        (13, 15, 0.14, "left-forearm"),
        (14, 16, 0.14, "right-forearm"),
        (23, 25, 0.21, "left-thigh"),
        (24, 26, 0.21, "right-thigh"),
        (25, 27, 0.17, "left-calf"),
        (26, 28, 0.17, "right-calf"),
    ] {
        let length = (points[b].0 - points[a].0).hypot(points[b].1 - points[a].1);
        if let Some(axis) = Axis::new(points[a], points[b], length * ratio) {
            limb_warps(warps, axis, part, values);
        }
    }
}

fn displacement(
    source: &RgbaImage,
    params: &Value,
    region: Region,
    warps: &[Warp],
    p: Point,
    separated_foreground: bool,
) -> Point {
    let w = source.width as f32;
    let h = source.height as f32;
    let (nx, ny) = (p.0 / w, p.1 / h);
    if !region.contains(nx, ny) {
        return (0.0, 0.0);
    }
    let selected = params
        .get("mask")
        .map(|m| coverage(m, nx, ny, source.width, source.height))
        .unwrap_or(1.0);
    let protection = params
        .get("protectMask")
        .map(|m| coverage(m, nx, ny, source.width, source.height))
        .unwrap_or(0.0);
    if selected <= 1e-6 || protection > 1e-6 {
        return (0.0, 0.0);
    }
    let edge = (p.0 - region.x * w)
        .min((region.x + region.width) * w - p.0)
        .min(p.1 - region.y * h)
        .min((region.y + region.height) * h - p.1);
    let mut feather = smoothstep(0.0, 3.0, edge);
    if separated_foreground {
        feather *= selected;
    }
    let (mut dx, mut dy) = (0.0, 0.0);
    for warp in warps {
        let d = if separated_foreground {
            warp.displacement_for(p, true)
        } else {
            warp.displacement(p)
        };
        dx += d.0;
        dy += d.1;
    }
    // Bound the combined field, including overlapping torso/limb controls.
    let limit = (region.width * w).min(region.height * h) * 0.20;
    (
        dx.clamp(-limit, limit) * feather,
        dy.clamp(-limit, limit) * feather,
    )
}

fn neighbors(index: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let x = index % width;
    let y = index / width;
    [
        (x > 0).then(|| index - 1),
        (x + 1 < width).then(|| index + 1),
        (y > 0).then(|| index - width),
        (y + 1 < height).then(|| index + width),
    ]
    .into_iter()
    .flatten()
}

/// Check before filling that each revealed component can be reached from
/// actual background. Interior foreground and protected pixels cannot seed
/// the fill, including when the selected person touches the canvas edge.
fn check_background_donors(
    holes: &[bool],
    protected: &[bool],
    width: usize,
    height: usize,
) -> Result<(), ImageError> {
    let reached = reachable_background_targets(holes, protected, width, height);
    if holes
        .iter()
        .zip(reached)
        .any(|(&hole, reached)| hole && !reached)
    {
        return Err(missing_background());
    }
    Ok(())
}

fn missing_background() -> ImageError {
    ImageError::InvalidOperation(
        "body contour reconstruction requires unprotected background next to the revealed silhouette"
            .into(),
    )
}

fn reachable_background_targets(
    targets: &[bool],
    protected: &[bool],
    width: usize,
    height: usize,
) -> Vec<bool> {
    use std::collections::VecDeque;
    let mut reached = vec![false; targets.len()];
    let mut queue = VecDeque::new();
    for (i, &target) in targets.iter().enumerate() {
        if target && neighbors(i, width, height).any(|n| !targets[n] && !protected[n]) {
            reached[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for n in neighbors(i, width, height) {
            if targets[n] && !reached[n] {
                reached[n] = true;
                queue.push_back(n);
            }
        }
    }
    reached
}

fn reconstruction_targets(
    holes: &[bool],
    subject: &[f32],
    user_protected: &[bool],
    width: usize,
    height: usize,
) -> Result<Vec<bool>, ImageError> {
    // A coarse segmentation mask's soft fringe can extend past the selected
    // anatomical ROI. That fringe contains foreground, so it cannot donate
    // background, but the hidden background may be reconstructed through it.
    // Reconstructing this layer does not edit the unselected output pixels.
    let candidates: Vec<bool> = subject
        .iter()
        .enumerate()
        .map(|(i, &alpha)| {
            (holes[i] || (alpha > 1.0 / 255.0 && alpha < 1.0 - 1e-6)) && !user_protected[i]
        })
        .collect();
    let protected: Vec<bool> = subject
        .iter()
        .enumerate()
        .map(|(i, &alpha)| (alpha > 1.0 / 255.0 && !candidates[i]) || user_protected[i])
        .collect();
    let reached = reachable_background_targets(&candidates, &protected, width, height);
    // Only requested revealed pixels are mandatory. Isolated soft confidence
    // islands inside opaque foreground are excluded, rather than becoming
    // either erroneous donor colors or unrelated fill failures.
    if holes
        .iter()
        .zip(&reached)
        .any(|(&hole, &reached)| hole && !reached)
    {
        return Err(missing_background());
    }
    Ok(reached)
}

/// Estimate a segment's actual half-width from contiguous subject spans.
/// Stop at the first background crossing so a nearby leg/arm is not counted
/// as part of the edited segment. Pose polygon width is only the fallback.
fn subject_radius(axis: Axis, subject: &[f32], width: usize, height: usize, center: f32) -> f32 {
    let read = |p: Point| {
        if p.0 < 0.0 || p.1 < 0.0 || p.0 >= width as f32 || p.1 >= height as f32 {
            return 0.0;
        }
        subject[p.1.floor() as usize * width + p.0.floor() as usize]
    };
    let normal = axis.normal();
    let mut spans = Vec::new();
    let limit = (width.max(height) as f32 * 0.5).ceil() as usize;
    for offset in [-0.10, -0.05, 0.0, 0.05, 0.10] {
        let t = (center + offset).clamp(0.15, 0.85);
        let point = (
            axis.origin.0 + axis.direction.0 * axis.length * t,
            axis.origin.1 + axis.direction.1 * axis.length * t,
        );
        if read(point) < 0.5 {
            continue;
        }
        let mut near_side = f32::INFINITY;
        for sign in [-1.0, 1.0] {
            for distance in 1..=limit {
                let d = distance as f32;
                if read((point.0 + normal.0 * sign * d, point.1 + normal.1 * sign * d)) < 0.5 {
                    near_side = near_side.min(d);
                    break;
                }
            }
        }
        // A limb touching the torso has one short free-side ray and one long
        // ray through the torso. Pooling both sides makes the upper median
        // select the torso thickness and erases the visible arm's influence.
        if near_side.is_finite() {
            spans.push(near_side);
        }
    }
    spans.sort_by(f32::total_cmp);
    spans
        .get(spans.len() / 2)
        .copied()
        .unwrap_or(axis.radius)
        .max(1.0)
}

// A revealed silhouette strip needs only local background continuation.
// Keep this diffusion separate from public exemplar object removal: narrow
// strips can have transparent donors or no complete clean exemplar patch.
fn reconstruct_narrow_strip(
    source: &RgbaImage,
    targets: &[bool],
    protected: &[bool],
) -> Result<RgbaImage, ImageError> {
    use std::collections::VecDeque;
    let width = source.width as usize;
    let height = source.height as usize;
    let mut known: Vec<bool> = targets.iter().map(|v| !v).collect();
    let mut queue = VecDeque::new();
    for (i, &target) in targets.iter().enumerate() {
        if target && neighbors(i, width, height).any(|n| known[n] && !protected[n]) {
            queue.push_back(i);
        }
    }
    let mut result = source.clone();
    while let Some(i) = queue.pop_front() {
        if known[i] {
            continue;
        }
        let donors: Vec<usize> = neighbors(i, width, height)
            .filter(|&n| known[n] && !protected[n])
            .collect();
        if donors.is_empty() {
            continue;
        }
        for c in 0..3 {
            result.pixels[i * 4 + c] = (donors
                .iter()
                .map(|&n| result.pixels[n * 4 + c] as u32)
                .sum::<u32>()
                / donors.len() as u32) as u8;
        }
        known[i] = true;
        for n in neighbors(i, width, height) {
            if !known[n] {
                queue.push_back(n);
            }
        }
    }
    if known.iter().any(|v| !v) {
        return Err(missing_background());
    }
    for _ in 0..8 {
        let previous = result.pixels.clone();
        for (i, &target) in targets.iter().enumerate() {
            if !target {
                continue;
            }
            let donors: Vec<usize> = neighbors(i, width, height)
                .filter(|&n| !protected[n])
                .collect();
            if donors.is_empty() {
                continue;
            }
            for c in 0..3 {
                let mean = donors
                    .iter()
                    .map(|&n| previous[n * 4 + c] as u32)
                    .sum::<u32>() as f32
                    / donors.len() as f32;
                result.pixels[i * 4 + c] = byte(previous[i * 4 + c] as f32 * 0.25 + mean * 0.75);
            }
        }
    }
    Ok(result)
}

fn fill_revealed_background(
    source: &RgbaImage,
    holes: &[bool],
    protected: &[bool],
) -> Result<RgbaImage, ImageError> {
    let width = source.width as usize;
    let height = source.height as usize;
    let Some(first) = holes.iter().position(|v| *v) else {
        return Ok(source.clone());
    };
    let (mut min_x, mut max_x) = (first % width, first % width);
    let (mut min_y, mut max_y) = (first / width, first / width);
    for (i, &hole) in holes.iter().enumerate() {
        if hole {
            min_x = min_x.min(i % width);
            max_x = max_x.max(i % width);
            min_y = min_y.min(i / width);
            max_y = max_y.max(i / width);
        }
    }
    // Inpaint's farthest donor step is twelve pixels. Retaining that margin
    // keeps surrounding donors available without serializing full-size JSON
    // masks for a small waist/limb correction on a high-resolution photo.
    let left = min_x.saturating_sub(12);
    let top = min_y.saturating_sub(12);
    let right = (max_x + 13).min(width);
    let bottom = (max_y + 13).min(height);
    let crop_width = right - left;
    let crop_height = bottom - top;
    let mut pixels = Vec::with_capacity(crop_width * crop_height * 4);
    let mut targets = Vec::with_capacity(crop_width * crop_height);
    let mut protection = Vec::with_capacity(crop_width * crop_height);
    for y in top..bottom {
        for x in left..right {
            let i = y * width + x;
            pixels.extend_from_slice(&source.pixels[i * 4..i * 4 + 4]);
            targets.push(holes[i]);
            protection.push(protected[i]);
        }
    }
    check_background_donors(&targets, &protection, crop_width, crop_height)?;
    let crop = RgbaImage::new(crop_width as u32, crop_height as u32, pixels)?;
    let mut filled = reconstruct_narrow_strip(&crop, &targets, &protection)?;
    // Inpaint preserves source alpha for object-removal operations. A body
    // contour instead reveals a different layer, so its alpha must also
    // come from clean background donors (e.g. a transparent PNG backdrop).
    use std::collections::VecDeque;
    let mut known: Vec<bool> = targets.iter().map(|v| !v).collect();
    let mut queue = VecDeque::new();
    for (i, &target) in targets.iter().enumerate() {
        if target && neighbors(i, crop_width, crop_height).any(|n| known[n] && !protection[n]) {
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        if known[i] {
            continue;
        }
        let donors: Vec<usize> = neighbors(i, crop_width, crop_height)
            .filter(|&n| known[n] && !protection[n])
            .collect();
        if donors.is_empty() {
            continue;
        }
        filled.pixels[i * 4 + 3] = (donors
            .iter()
            .map(|&n| filled.pixels[n * 4 + 3] as u32)
            .sum::<u32>()
            / donors.len() as u32) as u8;
        known[i] = true;
        for n in neighbors(i, crop_width, crop_height) {
            if !known[n] {
                queue.push_back(n);
            }
        }
    }
    let mut result = source.clone();
    for (i, &target) in targets.iter().enumerate() {
        if target {
            let j = (top + i / crop_width) * width + left + i % crop_width;
            result.pixels[j * 4..j * 4 + 4].copy_from_slice(&filled.pixels[i * 4..i * 4 + 4]);
        }
    }
    Ok(result)
}

fn separated_adjust(
    source: &RgbaImage,
    params: &Value,
    region: Region,
    warps: &[Warp],
    subject_mask: &Value,
) -> Result<RgbaImage, ImageError> {
    let width = source.width as usize;
    let height = source.height as usize;
    let count = width * height;
    let mut subject = vec![0.0; count];
    let mut foreground = source.clone();
    let mut user_protected = vec![false; count];
    for y in 0..height {
        for x in 0..width {
            let (nx, ny) = (
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let i = y * width + x;
            subject[i] = coverage(subject_mask, nx, ny, source.width, source.height);
            // RGB is sampled only with this alpha. A zero-alpha background
            // texel therefore contributes no color to the warped foreground.
            foreground.pixels[i * 4 + 3] = byte(source.pixels[i * 4 + 3] as f32 * subject[i]);
            user_protected[i] = params
                .get("protectMask")
                .is_some_and(|m| coverage(m, nx, ny, source.width, source.height) > 1e-6);
        }
    }
    let mut measured = warps.to_vec();
    for warp in &mut measured {
        warp.axis.radius = subject_radius(warp.axis, &subject, width, height, warp.center);
    }
    let mut warped = foreground.clone();
    let mut warped_alpha = vec![0.0; count];
    let mut warped_subject = subject.clone();
    let mut changed = vec![false; count];
    let mut holes = vec![false; count];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let d = displacement(
                source,
                params,
                region,
                &measured,
                (x as f32 + 0.5, y as f32 + 0.5),
                true,
            );
            if d.0.abs() + d.1.abs() < 1e-6 {
                continue;
            }
            let (pixel, alpha) = sample_with_alpha(&foreground, x as f32 + d.0, y as f32 + d.1);
            // Identity samples stay bit-exact, especially in the untouched
            // background gaps inside a whole-body pose hull.
            if pixel == foreground.pixels[i * 4..i * 4 + 4] {
                continue;
            }
            warped.pixels[i * 4..i * 4 + 4].copy_from_slice(&pixel);
            warped_alpha[i] = alpha;
            warped_subject[i] =
                sample_coverage(&subject, width, height, x as f32 + d.0, y as f32 + d.1);
            changed[i] = true;
            // Only the newly uncovered contour strip requires background.
            // Pixels fully covered by the new foreground need no fill.
            holes[i] =
                subject[i] > 1.0 / 255.0 && warped_subject[i] < 1.0 - 1e-6 && !user_protected[i];
        }
    }
    let mut background = source.clone();
    if holes.iter().any(|v| *v) {
        let targets = reconstruction_targets(&holes, &subject, &user_protected, width, height)?;
        // Remove the revealed strip from donor protection while retaining all
        // remaining original foreground and every user-protected pixel.
        let donor_protected: Vec<bool> = subject
            .iter()
            .enumerate()
            .map(|(i, &alpha)| (alpha > 1.0 / 255.0 && !targets[i]) || user_protected[i])
            .collect();
        background = fill_revealed_background(source, &targets, &donor_protected)?;
    }
    let mut result = source.clone();
    for i in 0..count {
        if !changed[i] || user_protected[i] {
            continue;
        }
        let fg_alpha = warped_alpha[i];
        if fg_alpha <= 1e-6 && !holes[i] {
            continue;
        }
        // A soft detector fringe contains background pixels. A contraction
        // cannot repaint those original outside pixels with foreground RGB;
        // an expansion may cover them, but still never moves their texture.
        if subject[i] < 0.5 && warped_subject[i] <= subject[i] {
            continue;
        }
        // The background is never displaced. Only reconstructed contour
        // strips differ from the input background; expanded silhouettes may
        // cover it without moving its texture.
        let bg_alpha = background.pixels[i * 4 + 3] as f32 / 255.0;
        // Segmentation coverage partitions one source layer; RGBA alpha is
        // not that coverage. Using (1-fg_alpha) here would incorrectly make
        // the same half-transparent photo more opaque after a contour edit.
        let uncovered = 1.0 - warped_subject[i];
        let alpha = fg_alpha + bg_alpha * uncovered;
        for c in 0..3 {
            result.pixels[i * 4 + c] = if alpha > 1e-6 {
                byte(
                    (warped.pixels[i * 4 + c] as f32 * fg_alpha
                        + background.pixels[i * 4 + c] as f32 * bg_alpha * uncovered)
                        / alpha,
                )
            } else {
                0
            };
        }
        result.pixels[i * 4 + 3] = byte(alpha * 255.0);
    }
    Ok(result)
}

/// Width adjustments act only on their semantic region and add independently.
/// Height/legLength are deliberately not interpreted as width controls.
pub(crate) fn adjust(source: &RgbaImage, params: &Value) -> Result<RgbaImage, ImageError> {
    let region = Region::read(params.get("region"));
    let values = params.get("adjustments").unwrap_or(params);
    let part = params
        .get("bodyPart")
        .and_then(Value::as_str)
        .unwrap_or("full-body");
    let w = source.width as f32;
    let h = source.height as f32;
    let points = read_points(params.get("landmarks"), w, h);
    let mask = params.get("mask");
    let polygon = read_points(mask.and_then(|m| m.get("points")), w, h);
    let mut warps = Vec::new();
    if part == "full-body" {
        full_body_warps(&mut warps, &points, values);
    } else {
        let axis = local_axis(part, region, &points, &polygon, w, h);
        match part {
            "waist" => {
                add(&mut warps, axis, values, "waist", 0.50, 0.55);
                add(&mut warps, axis, values, "belly", 0.65, 0.40);
            }
            "hips" => add(&mut warps, axis, values, "hips", 0.50, 0.55),
            "torso" => torso_warps(&mut warps, axis, values),
            _ => limb_warps(&mut warps, axis, part, values),
        }
    }
    if warps.is_empty() {
        return Ok(source.clone());
    }
    if let Some(subject_mask) = params.get("subjectMask").filter(|mask| mask.is_object()) {
        return separated_adjust(source, params, region, &warps, subject_mask);
    }
    let mut output = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let (dx, dy) = displacement(source, params, region, &warps, p, false);
            if dx.abs() + dy.abs() < 1e-6 {
                continue;
            }
            let index = (y as usize * source.width as usize + x as usize) * 4;
            output.pixels[index..index + 4].copy_from_slice(&sample(
                source,
                x as f32 + dx,
                y as f32 + dy,
            ));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn arm_touching_torso_uses_its_free_side_radius_and_moves_visible_edge() {
        let width = 120usize;
        let height = 160usize;
        let mut subject = vec![0.0; width * height];
        let mut pixels = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let foreground = (36..100).contains(&x) && (24..136).contains(&y);
                subject[y * width + x] = if foreground { 1.0 } else { 0.0 };
                pixels[(y * width + x) * 4..(y * width + x) * 4 + 4].copy_from_slice(
                    if foreground {
                        &[210, 110, 70, 255]
                    } else {
                        &[20, 40, 110, 255]
                    },
                );
            }
        }
        let axis = Axis::new((45.0, 24.0), (45.0, 136.0), 20.0).unwrap();
        let radius = subject_radius(axis, &subject, width, height, 0.5);
        assert!(
            (radius - 10.0).abs() <= 1.0,
            "must not pick torso's 55px span: {radius}"
        );
        let source = RgbaImage::new(width as u32, height as u32, pixels).unwrap();
        let alpha: Vec<u8> = subject.iter().map(|&v| byte(v * 255.0)).collect();
        let params = json!({"bodyPart":"left-upper-arm","region":{"x":0.23,"y":0.05,"width":0.3,"height":0.9},"landmarks":[{"x":0.375,"y":0.15},{"x":0.375,"y":0.85}],"subjectMask":{"shape":"raster","width":width,"height":height,"data":alpha},"adjustments":{"arms":80}});
        let output = adjust(&source, &params).unwrap();
        let start = |im: &RgbaImage| {
            (0..width)
                .find(|&x| im.pixels[(80 * width + x) * 4] > 150)
                .unwrap()
        };
        assert!(
            start(&output) > start(&source),
            "visible arm edge must move inward"
        );
        assert_eq!(
            &output.pixels[(80 * width + 80) * 4..(80 * width + 80) * 4 + 4],
            &source.pixels[(80 * width + 80) * 4..(80 * width + 80) * 4 + 4]
        );
    }
    fn image(w: u32, h: u32) -> RgbaImage {
        let mut pixels = Vec::new();
        for y in 0..h {
            for x in 0..w {
                pixels.extend_from_slice(&[
                    (x * 255 / w) as u8,
                    (y * 255 / h) as u8,
                    ((x + y) % 200) as u8,
                    255,
                ]);
            }
        }
        RgbaImage::new(w, h, pixels).unwrap()
    }
    fn hips_params(value: i32) -> Value {
        json!({"bodyPart":"hips","region":{"x":0.2,"y":0.3,"width":0.6,"height":0.4},"mask":{"shape":"polygon","points":[{"x":0.2,"y":0.3},{"x":0.8,"y":0.3},{"x":0.8,"y":0.7},{"x":0.2,"y":0.7}],"feather":0},"landmarks":[{"x":0.35,"y":0.5},{"x":0.65,"y":0.5},{"x":0.5,"y":0.5}],"adjustments":{"hips":value}})
    }
    #[test]
    fn level_hips_use_roi_height_and_opposite_directions() {
        let source = image(100, 160);
        let narrow = adjust(&source, &hips_params(70)).unwrap();
        let wide = adjust(&source, &hips_params(-70)).unwrap();
        let i = (80 * 100 + 70) * 4;
        assert!(narrow.pixels[i] > source.pixels[i]);
        assert!(wide.pixels[i] < source.pixels[i]);
        let centre = (80 * 100 + 50) * 4;
        assert_eq!(
            &narrow.pixels[centre..centre + 4],
            &source.pixels[centre..centre + 4]
        );
        for y in 0..160 {
            for x in 0..100 {
                if !(20..80).contains(&x) || !(48..112).contains(&y) {
                    let i = (y * 100 + x) * 4;
                    assert_eq!(&narrow.pixels[i..i + 4], &source.pixels[i..i + 4]);
                }
            }
        }
    }
    #[test]
    fn tilted_limb_radius_matches_polygon_and_axis_stays_fixed() {
        let source = image(200, 100);
        let params = json!({"bodyPart":"left-thigh","region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},"landmarks":[{"x":0.3,"y":0.3},{"x":0.7,"y":0.7}],"mask":{"shape":"polygon","points":[{"x":0.25,"y":0.4},{"x":0.65,"y":0.8},{"x":0.75,"y":0.6},{"x":0.35,"y":0.2}],"feather":0},"adjustments":{"thighs":60}});
        let output = adjust(&source, &params).unwrap();
        assert_ne!(output.pixels, source.pixels);
        let points = read_points(params.get("landmarks"), 200.0, 100.0);
        let polygon = read_points(params["mask"].get("points"), 200.0, 100.0);
        let axis = local_axis(
            "left-thigh",
            Region::read(params.get("region")),
            &points,
            &polygon,
            200.0,
            100.0,
        );
        assert!((axis.radius - 13.4164).abs() < 0.01);
        let d = Warp {
            axis,
            amount: 0.7,
            center: 0.5,
            extent: 0.5,
        }
        .displacement(mid(points[0], points[1]));
        assert_eq!(d, (0.0, 0.0));
    }
    #[test]
    fn generic_and_specific_limb_controls_both_contribute() {
        let source = image(100, 160);
        let mut p = json!({"bodyPart":"left-calf","region":{"x":0.3,"y":0.2,"width":0.4,"height":0.6},"landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.8}],"mask":{"shape":"polygon","points":[{"x":0.3,"y":0.2},{"x":0.7,"y":0.2},{"x":0.7,"y":0.8},{"x":0.3,"y":0.8}],"feather":0},"adjustments":{"legs":30}});
        let first = adjust(&source, &p).unwrap();
        p["adjustments"]["calves"] = json!(30);
        let both = adjust(&source, &p).unwrap();
        assert_ne!(first.pixels, both.pixels);
        let i = (80 * 100 + 61) * 4;
        assert!(both.pixels[i] > first.pixels[i]);
    }
    #[test]
    fn full_body_never_uses_length_as_width_and_combines_distinct_parts() {
        let source = image(100, 160);
        let mut points = vec![json!({"x":0.5,"y":0.1}); 33];
        for (i, x, y) in [
            (11, 0.3, 0.2),
            (12, 0.7, 0.2),
            (13, 0.22, 0.4),
            (14, 0.78, 0.4),
            (15, 0.2, 0.55),
            (16, 0.8, 0.55),
            (23, 0.35, 0.5),
            (24, 0.65, 0.5),
            (25, 0.36, 0.7),
            (26, 0.64, 0.7),
            (27, 0.36, 0.9),
            (28, 0.64, 0.9),
        ] {
            points[i] = json!({"x":x,"y":y});
        }
        let mut p = json!({"bodyPart":"full-body","region":{"x":0.1,"y":0.1,"width":0.8,"height":0.85},"landmarks":points,"adjustments":{"height":75,"legLength":60}});
        assert_eq!(adjust(&source, &p).unwrap(), source);
        p["adjustments"] = json!({"waist":40});
        let waist = adjust(&source, &p).unwrap();
        p["adjustments"]["calves"] = json!(60);
        let both = adjust(&source, &p).unwrap();
        assert_ne!(waist.pixels, both.pixels);
    }

    #[test]
    fn every_semantic_width_region_has_local_effect_without_cross_part_controls() {
        let source = image(100, 160);
        for (part, key) in [
            ("waist", "waist"),
            ("hips", "hips"),
            ("left-upper-arm", "arms"),
            ("right-upper-arm", "arms"),
            ("left-forearm", "forearms"),
            ("right-forearm", "forearms"),
            ("left-thigh", "thighs"),
            ("right-thigh", "thighs"),
            ("left-calf", "calves"),
            ("right-calf", "calves"),
        ] {
            let mut params = json!({
                "bodyPart":part,
                "region":{"x":0.3,"y":0.2,"width":0.4,"height":0.6},
                "landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.8}],
                "mask":{"shape":"polygon","points":[{"x":0.3,"y":0.2},{"x":0.7,"y":0.2},{"x":0.7,"y":0.8},{"x":0.3,"y":0.8}],"feather":0},
                "adjustments":{}
            });
            params["adjustments"][key] = json!(60);
            let result = adjust(&source, &params).unwrap();
            assert_ne!(result, source, "{part}.{key} must have a local effect");
            // Choose an unrelated parameter: each semantic path must ignore
            // it, rather than selecting the first nonzero global value.
            params["adjustments"] = json!({"height":80,"legLength":70,"headRatio":50});
            assert_eq!(
                adjust(&source, &params).unwrap(),
                source,
                "{part} must ignore structural controls"
            );
        }
    }

    fn segmented_person(amount: i32) -> (RgbaImage, Value) {
        let (w, h) = (160, 200);
        let mut pixels = Vec::new();
        let mut alpha = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let person = (60..100).contains(&x) && (30..170).contains(&y);
                let rgb = if person {
                    [230, 40, 60]
                } else {
                    // A grid gives many visible reference edges. None may
                    // move as the person's silhouette changes.
                    [0, if x % 13 == 0 || y % 17 == 0 { 220 } else { 80 }, 190]
                };
                pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                alpha.push(if person { 255u8 } else { 0u8 });
            }
        }
        (
            RgbaImage::new(w, h, pixels).unwrap(),
            json!({
                "bodyPart":"waist",
                "region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},
                "landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.5},{"x":0.5,"y":0.8}],
                "mask":{"shape":"polygon","feather":0,"points":[{"x":0.2,"y":0.2},{"x":0.8,"y":0.2},{"x":0.8,"y":0.8},{"x":0.2,"y":0.8}]},
                "subjectMask":{"shape":"raster","width":w,"height":h,"data":alpha},
                "adjustments":{"waist":amount}
            }),
        )
    }

    #[test]
    fn segmented_body_changes_silhouette_in_both_directions_without_warping_background() {
        let (source, mut params) = segmented_person(80);
        let narrow = adjust(&source, &params).unwrap();
        params["adjustments"]["waist"] = json!(-80);
        let wide = adjust(&source, &params).unwrap();
        let count_subject = |image: &RgbaImage| {
            (80..120)
                .flat_map(|y| (0..160).map(move |x| (y * 160 + x) * 4))
                .filter(|&i| image.pixels[i..i + 3] == [230, 40, 60])
                .count()
        };
        assert!(count_subject(&narrow) < count_subject(&source));
        assert!(count_subject(&wide) > count_subject(&source));
        let right_contour = (100 * 160 + 99) * 4;
        assert_eq!(
            narrow.pixels[right_contour], 0,
            "revealed strip must come from background donors"
        );
        let expanded_contour = (100 * 160 + 101) * 4;
        assert_eq!(
            &wide.pixels[expanded_contour..expanded_contour + 3],
            &[230, 40, 60]
        );
        for y in 0..200 {
            for x in 0..160 {
                let i = (y * 160 + x) * 4;
                if source.pixels[i] == 0 {
                    assert_eq!(
                        &narrow.pixels[i..i + 4],
                        &source.pixels[i..i + 4],
                        "background grid moved at {x},{y}"
                    );
                }
                if !(53..=107).contains(&x) || !(40..160).contains(&y) {
                    assert_eq!(
                        &wide.pixels[i..i + 4],
                        &source.pixels[i..i + 4],
                        "background outside expanded silhouette moved at {x},{y}"
                    );
                }
            }
        }
    }

    #[test]
    fn segmented_foreground_has_no_background_color_and_respects_protection() {
        let (source, mut params) = segmented_person(-90);
        params["protectMask"] = json!({
            "shape":"rect","region":{"x":0.5,"y":0.0,"width":0.5,"height":1.0}
        });
        let output = adjust(&source, &params).unwrap();
        for y in 0..200 {
            for x in 80..160 {
                let i = (y * 160 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
        let expanded_left = (100 * 160 + 58) * 4;
        assert_eq!(
            &output.pixels[expanded_left..expanded_left + 4],
            &[230, 40, 60, 255]
        );
        // Every fully opaque foreground sample retains its original pure
        // color; bilinear interpolation may blend alpha only at its edge.
        for pixel in output.pixels.chunks_exact(4) {
            if pixel[0] == 230 {
                assert_eq!(pixel, &[230, 40, 60, 255]);
            }
        }
    }

    #[test]
    fn revealed_component_without_clean_background_returns_an_explicit_error() {
        let holes = vec![false, true, false, false];
        let protected = vec![true, false, true, true];
        let error = check_background_donors(&holes, &protected, 2, 2).unwrap_err();
        assert!(error.to_string().contains("unprotected background"));
    }

    #[test]
    fn separated_layers_do_not_double_the_alpha_of_a_half_transparent_photo() {
        let (mut source, mut params) = segmented_person(80);
        for pixel in source.pixels.chunks_exact_mut(4) {
            pixel[3] = 128;
        }
        for amount in [80, -80] {
            params["adjustments"]["waist"] = json!(amount);
            let output = adjust(&source, &params).unwrap();
            assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 128));
        }
    }

    #[test]
    fn transparent_backdrop_is_revealed_and_expansion_keeps_source_subject_alpha() {
        let (mut source, mut params) = segmented_person(80);
        for pixel in source.pixels.chunks_exact_mut(4) {
            pixel[3] = if pixel[0] == 230 { 128 } else { 0 };
        }
        let narrowed = adjust(&source, &params).unwrap();
        let revealed = (100 * 160 + 99) * 4;
        assert_eq!(narrowed.pixels[revealed + 3], 0);
        let center = (100 * 160 + 80) * 4;
        assert_eq!(narrowed.pixels[center + 3], 128);
        params["adjustments"]["waist"] = json!(-80);
        let expanded = adjust(&source, &params).unwrap();
        let outer = (100 * 160 + 101) * 4;
        assert_eq!(&expanded.pixels[outer..outer + 4], &[230, 40, 60, 128]);
        let far_background = (100 * 160 + 110) * 4;
        assert_eq!(
            &expanded.pixels[far_background..far_background + 4],
            &source.pixels[far_background..far_background + 4]
        );
    }

    #[test]
    fn coarse_soft_subject_fringe_outside_roi_can_supply_hidden_background() {
        let source = image(128, 128);
        let alpha: Vec<u8> = (0..8)
            .flat_map(|y| {
                (0..8).map(move |x| {
                    if (2..=5).contains(&x) && (1..=6).contains(&y) {
                        255
                    } else {
                        0
                    }
                })
            })
            .collect();
        let params = json!({
            "bodyPart":"waist","region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},
            "landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.5},{"x":0.5,"y":0.8}],
            "subjectMask":{"shape":"raster","width":8,"height":8,"data":alpha},
            "adjustments":{"waist":20}
        });
        let output = adjust(&source, &params).unwrap();
        assert_ne!(output, source);
        for y in 0..128 {
            for x in 0..128 {
                let i = (y * 128 + x) * 4;
                if !(26..102).contains(&x) || !(26..102).contains(&y) {
                    assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
                }
                assert_eq!(output.pixels[i + 3], 255);
            }
        }
    }
}
