//! Pose-aware local body contour edits. Axes and radii use image pixels so a
//! tall portrait and a square preview have the same geometric meaning.
//! Detector polygons remain approximate selections; they are not parsing.

use crate::engine::{byte, number, percent};
use crate::mask::{coverage, Region};
use crate::{ImageError, RgbaImage};
use serde_json::Value;

type Point = (f32, f32);

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
    limb: bool,
}
fn add(warps: &mut Vec<Warp>, axis: Axis, values: &Value, key: &str, center: f32, extent: f32) {
    let amount = percent(values, key);
    if amount.abs() > 1e-6 {
        warps.push(Warp {
            axis,
            amount,
            center,
            extent,
            limb: matches!(key, "arms" | "forearms" | "legs" | "thighs" | "calves"),
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
        // A bounding rectangle grows vertically when a wide torso tilts.
        // Waist length follows the shoulder→hip axis, independently of that
        // rectangle, and agrees with the measured support in the browser.
        let bone = Axis::new(points[0], points[2], fallback.radius).unwrap_or(fallback);
        let dir = bone.direction;
        let length = (bone.length * 0.65).max(1.0);
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
        let length = if polygon.len() >= 3 {
            let projections: Vec<f32> = polygon
                .iter()
                .map(|p| (p.0 - m.0) * dir.0 + (p.1 - m.1) * dir.1)
                .collect();
            projections
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                - projections.iter().copied().fold(f32::INFINITY, f32::min)
        } else {
            region.height * h
        }
        .max(1.0);
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
    let Some(first) = holes.iter().position(|&hole| hole) else {
        return Ok(vec![false; holes.len()]);
    };
    let (mut left, mut right, mut top, mut bottom) =
        (first % width, first % width, first / width, first / width);
    for (i, &hole) in holes.iter().enumerate() {
        if hole {
            left = left.min(i % width);
            right = right.max(i % width);
            top = top.min(i / width);
            bottom = bottom.max(i / width);
        }
    }
    // Connect a coarse alpha fringe to nearby clean donors, without filling
    // unrelated confidence edges around the person's head and feet.
    let margin = (width.max(height) / 32).max(8);
    left = left.saturating_sub(margin);
    top = top.saturating_sub(margin);
    right = (right + margin).min(width - 1);
    bottom = (bottom + margin).min(height - 1);
    // A coarse segmentation mask's soft fringe can extend past the selected
    // anatomical ROI. That fringe contains foreground, so it cannot donate
    // background, but the hidden background may be reconstructed through it.
    // Reconstructing this layer does not edit the unselected output pixels.
    let candidates: Vec<bool> = subject
        .iter()
        .enumerate()
        .map(|(i, &alpha)| {
            let nearby =
                (left..=right).contains(&(i % width)) && (top..=bottom).contains(&(i / width));
            (holes[i] || (nearby && alpha > 1.0 / 255.0 && alpha < 1.0 - 1e-6))
                && !user_protected[i]
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
#[cfg(test)]
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

/// A cross-section stores the real left/right exterior edges independently.
/// One guessed half-width freezes a wide shirt edge or sends a touching arm's
/// displacement through the torso. The subject layer supplies contour data;
/// image background never participates in the deformation.
#[derive(Clone)]
struct ContourProfile {
    warp: Warp,
    sections: Vec<(f32, f32)>,
    fringe: f32,
}
impl ContourProfile {
    fn new(
        warp: Warp,
        alpha: &[f32],
        width: usize,
        height: usize,
        limb: bool,
        fringe: f32,
    ) -> Self {
        let axis = warp.axis;
        let normal = axis.normal();
        let read = |p: Point| {
            if p.0 < 0.0 || p.1 < 0.0 || p.0 >= width as f32 || p.1 >= height as f32 {
                0.0
            } else {
                sample_coverage(alpha, width, height, p.0 - 0.5, p.1 - 0.5)
            }
        };
        let mut sections = Vec::with_capacity(65);
        for i in 0..=64 {
            let t = i as f32 / 64.0;
            let center = (
                axis.origin.0 + axis.direction.0 * axis.length * t,
                axis.origin.1 + axis.direction.1 * axis.length * t,
            );
            let mut sides = [0.0f32; 2];
            if read(center) >= 0.5 {
                for (side, sign) in [-1.0f32, 1.0].into_iter().enumerate() {
                    let limit = (axis.radius * 2.0 + 8.0)
                        .min(width.max(height) as f32 * 0.6)
                        .ceil() as usize;
                    let mut previous = read(center);
                    for d in 1..=limit {
                        let value = read((
                            center.0 + normal.0 * sign * d as f32,
                            center.1 + normal.1 * sign * d as f32,
                        ));
                        if value < 0.5 {
                            let fraction = (previous - 0.5) / (previous - value).max(1e-6);
                            sides[side] = d as f32 - 1.0 + fraction;
                            break;
                        }
                        previous = value;
                    }
                }
                if limb {
                    // An attached arm's inner ray may run through the whole
                    // torso. Freeze that hidden side instead of reshaping a
                    // second anatomical part. A pixel part mask removes this
                    // ambiguity and needs no such conservative side gate.
                    let near = sides[0].min(sides[1]);
                    for side in &mut sides {
                        if near > 0.0 && *side > near * 3.0 && *side > axis.length * 0.2 {
                            *side = 0.0;
                        }
                    }
                }
            }
            sections.push((sides[0], sides[1]));
        }
        // A short local low-pass regularises contour confidence noise while
        // retaining genuine asymmetric widths and anchored occlusion gaps.
        let original = sections.clone();
        for (i, section) in sections.iter_mut().enumerate() {
            let window = &original[i.saturating_sub(2)..(i + 3).min(original.len())];
            for side in 0..2 {
                let current = if side == 0 { section.0 } else { section.1 };
                if current <= 0.0 {
                    continue;
                }
                let values: Vec<f32> = window
                    .iter()
                    .map(|p| if side == 0 { p.0 } else { p.1 })
                    .filter(|r| *r > 0.0)
                    .collect();
                let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
                if side == 0 {
                    section.0 = current * 0.5 + mean * 0.5;
                } else {
                    section.1 = current * 0.5 + mean * 0.5;
                }
            }
        }
        Self {
            warp,
            sections,
            fringe,
        }
    }
    fn displacement(&self, p: Point) -> Point {
        let (t, across) = self.warp.axis.coordinates(p);
        if !(0.0..=1.0).contains(&t) {
            return (0.0, 0.0);
        }
        let slot = t * (self.sections.len() - 1) as f32;
        let first = slot.floor() as usize;
        let last = (first + 1).min(self.sections.len() - 1);
        let side = |pair: (f32, f32)| if across < 0.0 { pair.0 } else { pair.1 };
        let r0 = side(self.sections[first]);
        let r1 = side(self.sections[last]);
        if r0 <= 0.0 || r1 <= 0.0 {
            return (0.0, 0.0);
        }
        let radius = r0 + (r1 - r0) * (slot - first as f32);
        let distance = ((t - self.warp.center) / self.warp.extent).abs();
        if distance >= 1.0 {
            return (0.0, 0.0);
        }
        let endpoint = smoothstep(0.0, 0.12, t) * (1.0 - smoothstep(0.88, 1.0, t));
        let profile = (1.0 - distance * distance).powi(2) * endpoint;
        let target = radius * (1.0 - self.warp.amount * 0.26 * profile);
        let core = radius * 0.30;
        let output = across.abs();
        if output <= core || output > radius.max(target) + self.fringe {
            return (0.0, 0.0);
        }
        // Piecewise harmonic extension between fixed bone-core pixels and
        // the prescribed silhouette. This is a monotone inverse map: no
        // folds, no global photo stretch, exact control of contour shrinkage.
        let input = if output <= target {
            core + (output - core) * (radius - core) / (target - core).max(1.0)
        } else {
            radius + output - target
        };
        let shift = (input - output) * across.signum();
        let n = self.warp.axis.normal();
        (n.0 * shift, n.1 * shift)
    }
}

fn contour_displacement(
    region: Region,
    profiles: &[ContourProfile],
    p: Point,
    selected: f32,
    protection_fade: f32,
    w: f32,
    h: f32,
) -> Point {
    if protection_fade <= 1e-6 || selected <= 1e-6 || !region.contains(p.0 / w, p.1 / h) {
        return (0.0, 0.0);
    }
    let edge = (p.0 - region.x * w)
        .min((region.x + region.width) * w - p.0)
        .min(p.1 - region.y * h)
        .min((region.y + region.height) * h - p.1);
    let feather = smoothstep(0.0, 3.0, edge) * selected * protection_fade;
    let mut result = (0.0, 0.0);
    for profile in profiles {
        let d = profile.displacement(p);
        result.0 += d.0;
        result.1 += d.1;
    }
    let limit = (region.width * w).min(region.height * h) * 0.20;
    (
        result.0.clamp(-limit, limit) * feather,
        result.1.clamp(-limit, limit) * feather,
    )
}

/// A hard jump from a ten-pixel displacement to a frozen sleeve creates a
/// tear even when both endpoint pixels are legal. Use a distance field to
/// anchor the displacement continuously before reaching frozen pixels.
fn protection_fade(protected: &[bool], width: usize, height: usize, falloff: f32) -> Vec<f32> {
    if !protected.iter().any(|p| *p) {
        return vec![1.0; protected.len()];
    }
    let mut distance: Vec<f32> = protected
        .iter()
        .map(|p| if *p { 0.0 } else { f32::INFINITY })
        .collect();
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if x > 0 {
                distance[i] = distance[i].min(distance[i - 1] + 1.0);
            }
            if y > 0 {
                distance[i] = distance[i].min(distance[i - width] + 1.0);
            }
            if x > 0 && y > 0 {
                distance[i] = distance[i].min(distance[i - width - 1] + std::f32::consts::SQRT_2);
            }
            if x + 1 < width && y > 0 {
                distance[i] = distance[i].min(distance[i - width + 1] + std::f32::consts::SQRT_2);
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
            }
            if x + 1 < width && y + 1 < height {
                distance[i] = distance[i].min(distance[i + width + 1] + std::f32::consts::SQRT_2);
            }
            if x > 0 && y + 1 < height {
                distance[i] = distance[i].min(distance[i + width - 1] + std::f32::consts::SQRT_2);
            }
        }
    }
    distance
        .into_iter()
        .map(|d| smoothstep(0.0, falloff, d))
        .collect()
}

/// Rasterize once per edit. Brush stamps have a finite pixel footprint;
/// evaluating every JSON stamp for every output pixel made protected arms
/// needlessly expensive in WebAssembly. Composition order matches coverage.
fn rasterize_mask(mask: &Value, width: u32, height: u32) -> Vec<f32> {
    let mut base = mask.clone();
    let invert = base.get("invert").and_then(Value::as_bool).unwrap_or(false);
    if let Some(object) = base.as_object_mut() {
        object.remove("strokes");
        object.remove("invert");
    }
    let w = width as usize;
    let h = height as usize;
    let mut values: Vec<f32> = (0..w * h)
        .map(|i| {
            coverage(
                &base,
                (i % w) as f32 / width as f32 + 0.5 / width as f32,
                (i / w) as f32 / height as f32 + 0.5 / height as f32,
                width,
                height,
            )
        })
        .collect();
    let read = |value: &Value, key: &str, default: f32| {
        value
            .get(key)
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(default)
    };
    let feather = read(mask, "feather", 0.0).clamp(0.0, 100.0) / 100.0 * 0.2;
    if let Some(strokes) = mask.get("strokes").and_then(Value::as_array) {
        for stroke in strokes {
            let radius = read(stroke, "radius", 0.05).max(0.0001);
            let cx = read(stroke, "x", 0.0) * width as f32;
            let cy = read(stroke, "y", 0.0) * height as f32;
            let footprint = radius * width.min(height) as f32;
            let softness = feather.max(radius * 0.12).min(radius);
            let strength = read(stroke, "strength", 100.0).clamp(0.0, 100.0) / 100.0;
            let erase = stroke
                .get("erase")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let left = (cx - footprint - 1.0).floor().max(0.0) as usize;
            let right = ((cx + footprint + 1.0).ceil() as usize).min(w);
            let top = (cy - footprint - 1.0).floor().max(0.0) as usize;
            let bottom = ((cy + footprint + 1.0).ceil() as usize).min(h);
            for y in top..bottom {
                for x in left..right {
                    let distance =
                        (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) / width.min(height) as f32;
                    let brush = (1.0 - smoothstep(radius - softness, radius, distance)) * strength;
                    let value = &mut values[y * w + x];
                    if erase {
                        *value *= 1.0 - brush;
                    } else {
                        *value = value.max(brush);
                    }
                }
            }
        }
    }
    if invert {
        for value in &mut values {
            *value = 1.0 - *value;
        }
    }
    values
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
    let subject = rasterize_mask(subject_mask, source.width, source.height);
    let has_part_mask = params.get("partMask").is_some_and(Value::is_object);
    let mut target = params
        .get("partMask")
        .filter(|m| m.is_object())
        .map(|m| rasterize_mask(m, source.width, source.height))
        .unwrap_or_else(|| subject.clone());
    for (i, alpha) in target.iter_mut().enumerate() {
        *alpha = alpha.min(subject[i]);
    }
    let selected = params
        .get("mask")
        .map(|m| rasterize_mask(m, source.width, source.height))
        .unwrap_or_else(|| vec![1.0; count]);
    let user_protected: Vec<bool> = params
        .get("protectMask")
        .map(|m| {
            rasterize_mask(m, source.width, source.height)
                .iter()
                .map(|v| *v > 1e-6)
                .collect()
        })
        .unwrap_or_else(|| vec![false; count]);
    let matte_layers = crate::face_matting::estimate(source, &target);
    let mut foreground = matte_layers
        .as_ref()
        .map(|layers| layers.0.clone())
        .unwrap_or_else(|| source.clone());
    for (i, &alpha) in target.iter().enumerate() {
        // Only the selected semantic part is sampled. The underlying torso,
        // the other leg and the backdrop remain a fixed context layer.
        if alpha >= 1.0 - 1e-6 {
            foreground.pixels[i * 4..i * 4 + 3].copy_from_slice(&source.pixels[i * 4..i * 4 + 3]);
        }
        foreground.pixels[i * 4 + 3] = byte(source.pixels[i * 4 + 3] as f32 * alpha);
    }
    let alpha_mask = params
        .get("partMask")
        .filter(|m| m.is_object())
        .unwrap_or(subject_mask);
    let fringe = if alpha_mask.get("shape").and_then(Value::as_str) == Some("raster") {
        let mw = alpha_mask
            .get("width")
            .and_then(Value::as_u64)
            .unwrap_or(source.width as u64)
            .max(1) as f32;
        let mh = alpha_mask
            .get("height")
            .and_then(Value::as_u64)
            .unwrap_or(source.height as u64)
            .max(1) as f32;
        (source.width as f32 / mw).max(source.height as f32 / mh) * 4.0
    } else {
        8.0
    }
    .max(3.0);
    let measured: Vec<ContourProfile> = warps
        .iter()
        .map(|&warp| {
            ContourProfile::new(
                warp,
                &target,
                width,
                height,
                warp.limb && !has_part_mask,
                fringe,
            )
        })
        .collect();
    let max_shift = measured
        .iter()
        .map(|profile| {
            profile
                .sections
                .iter()
                .map(|&(a, b)| a.max(b))
                .fold(0.0, f32::max)
                * profile.warp.amount.abs()
                * 0.26
        })
        .sum::<f32>();
    let protect_fade = protection_fade(
        &user_protected,
        width,
        height,
        (max_shift * 1.8 + 3.0).max(8.0),
    );
    let mut warped = foreground.clone();
    let mut warped_alpha = vec![0.0; count];
    let mut warped_subject = target.clone();
    let mut changed = vec![false; count];
    let mut holes = vec![false; count];
    let pure_contraction = warps.iter().all(|warp| warp.amount >= 0.0);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            // Shrinking cannot cover original known background. Noisy alpha
            // islands and smoothed cross-sections can otherwise make a
            // sample's alpha slightly larger in gaps outside the old edge.
            // The true soft foreground (> .01) still moves in its entirety.
            if pure_contraction && target[i] <= 0.01 {
                continue;
            }
            let d = contour_displacement(
                region,
                &measured,
                (x as f32 + 0.5, y as f32 + 0.5),
                selected[i],
                protect_fade[i],
                width as f32,
                height as f32,
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
                sample_coverage(&target, width, height, x as f32 + d.0, y as f32 + d.1);
            changed[i] = true;
            // Only the newly uncovered contour strip requires background.
            // Pixels fully covered by the new foreground need no fill.
            holes[i] =
                target[i] > 1.0 / 255.0 && warped_subject[i] < 1.0 - 1e-6 && !user_protected[i];
        }
    }
    let mut background = matte_layers
        .as_ref()
        .map(|layers| layers.1.clone())
        .unwrap_or_else(|| source.clone());
    for (i, &alpha) in target.iter().enumerate() {
        if alpha <= 1.0 / 255.0 {
            background.pixels[i * 4..i * 4 + 4].copy_from_slice(&source.pixels[i * 4..i * 4 + 4]);
        }
    }
    if holes.iter().any(|v| *v) {
        let targets = reconstruction_targets(&holes, &target, &user_protected, width, height)?;
        // Remove the revealed strip from donor protection while retaining all
        // remaining original foreground and every user-protected pixel.
        let donor_protected: Vec<bool> = target
            .iter()
            .enumerate()
            .map(|(i, &alpha)| (alpha > 1.0 / 255.0 && !targets[i]) || user_protected[i])
            .collect();
        background = fill_revealed_background(&background, &targets, &donor_protected)?;
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
        // The complete soft foreground fringe must move with the contour.
        // Freezing its alpha<.5 half leaves the old edge as a dotted ghost.
        // Only known background is immutable during contraction.
        if target[i] <= 0.01 && warped_subject[i] <= target[i] {
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
    let opacity = number(params, "opacity", 100.0);
    let blend = params
        .get("blendMode")
        .and_then(Value::as_str)
        .unwrap_or("normal");
    if opacity != 100.0 || blend != "normal" {
        let mut settings = params.clone();
        if let Some(object) = settings.as_object_mut() {
            object.remove("mask");
            object.remove("protectMask");
        }
        let composed = crate::mask::composite(source, &result, &settings);
        // Non-normal blend modes must not change the untouched context just
        // because it lies inside the support rectangle.
        for i in 0..count {
            if result.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4] {
                result.pixels[i * 4..i * 4 + 4].copy_from_slice(&composed.pixels[i * 4..i * 4 + 4]);
            }
        }
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
    let subject_mask=params.get("subjectMask").filter(|mask|mask.is_object()).ok_or_else(||ImageError::InvalidOperation("body contour edits require a real subjectMask; detect the person or draw the foreground silhouette manually".into()))?;
    separated_adjust(source, params, region, &warps, subject_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn a_nonzero_body_edit_without_a_real_subject_mask_is_rejected() {
        let (source, mut params) = segmented_person(60);
        params.as_object_mut().unwrap().remove("subjectMask");
        let error = adjust(&source, &params).unwrap_err();
        assert!(error.to_string().contains("real subjectMask"));
        params["adjustments"]["waist"] = json!(0);
        assert_eq!(adjust(&source, &params).unwrap(), source);
    }
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
    #[test]
    fn asymmetric_waist_moves_both_real_edges_and_keeps_the_bone_core() {
        let (width, height) = (160usize, 200usize);
        let mut pixels = vec![0u8; width * height * 4];
        let mut alpha = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let person = (20..110).contains(&x) && (30..170).contains(&y);
                alpha[y * width + x] = if person { 255 } else { 0 };
                pixels[(y * width + x) * 4..(y * width + x) * 4 + 4].copy_from_slice(if person {
                    &[230, 40, 60, 255]
                } else {
                    &[10, 70, 180, 255]
                });
            }
        }
        let source = RgbaImage::new(width as u32, height as u32, pixels).unwrap();
        let params = json!({"bodyPart":"waist","region":{"x":0.02,"y":0.1,"width":0.9,"height":0.8},"landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.5},{"x":0.5,"y":0.8}],"subjectMask":{"shape":"raster","width":width,"height":height,"data":alpha},"adjustments":{"waist":80}});
        let output = adjust(&source, &params).unwrap();
        let row = 100 * width;
        let edges = |image: &RgbaImage| {
            let xs: Vec<usize> = (0..width)
                .filter(|&x| image.pixels[(row + x) * 4] > 150)
                .collect();
            (*xs.first().unwrap(), *xs.last().unwrap())
        };
        let (before_left, before_right) = edges(&source);
        let (after_left, after_right) = edges(&output);
        assert!(
            after_left > before_left + 6,
            "wide side must not remain fixed: {after_left}"
        );
        assert!(
            after_right < before_right - 3,
            "short side must also contract: {after_right}"
        );
        for x in 75..85 {
            assert_eq!(
                &output.pixels[(row + x) * 4..(row + x) * 4 + 4],
                &source.pixels[(row + x) * 4..(row + x) * 4 + 4]
            );
        }
        for (i, &coverage) in alpha.iter().enumerate() {
            if coverage == 0 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }
    #[test]
    fn part_mask_reveals_the_underlying_torso_without_moving_it() {
        let (width, height) = (120usize, 160usize);
        let mut pixels = vec![0u8; width * height * 4];
        let mut subject = vec![0u8; width * height];
        let mut arm = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let visible = (24..136).contains(&y);
                let target = visible && (36..56).contains(&x);
                let torso = visible && (56..100).contains(&x);
                let i = y * width + x;
                subject[i] = if target || torso { 255 } else { 0 };
                arm[i] = if target { 255 } else { 0 };
                pixels[i * 4..i * 4 + 4].copy_from_slice(if target {
                    &[210, 40, 70, 255]
                } else if torso {
                    &[30, 210, 70, 255]
                } else {
                    &[20, 40, 110, 255]
                });
            }
        }
        let source = RgbaImage::new(width as u32, height as u32, pixels).unwrap();
        let params = json!({"bodyPart":"left-upper-arm","region":{"x":0.23,"y":0.05,"width":0.3,"height":0.9},"landmarks":[{"x":0.375,"y":0.15},{"x":0.375,"y":0.85}],"subjectMask":{"shape":"raster","width":width,"height":height,"data":subject},"partMask":{"shape":"raster","width":width,"height":height,"data":arm},"adjustments":{"arms":80}});
        let output = adjust(&source, &params).unwrap();
        assert_eq!(
            &output.pixels[(80 * width + 54) * 4..(80 * width + 54) * 4 + 4],
            &[30, 210, 70, 255],
            "newly revealed inner strip must continue the torso context"
        );
        assert_eq!(
            &output.pixels[(80 * width + 36) * 4..(80 * width + 36) * 4 + 4],
            &[20, 40, 110, 255],
            "outer strip must continue the background context"
        );
        for (i, &coverage) in arm.iter().enumerate() {
            if coverage == 0 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "non-target pixel changed at {i}"
                );
            }
        }
    }
    #[test]
    fn precomputed_brush_mask_preserves_stamp_order_feather_and_inversion() {
        let mask = json!({"shape":"rect","region":{"x":0.2,"y":0.3,"width":0.3,"height":0.4},"feather":12,"invert":true,"strokes":[{"x":0.4,"y":0.5,"radius":0.14,"strength":60},{"x":0.45,"y":0.55,"radius":0.09,"strength":80,"erase":true},{"x":0.7,"y":0.6,"radius":0.11,"strength":70}]});
        let (w, h) = (89u32, 133u32);
        let values = rasterize_mask(&mask, w, h);
        for y in 0..h {
            for x in 0..w {
                let expected = coverage(
                    &mask,
                    (x as f32 + 0.5) / w as f32,
                    (y as f32 + 0.5) / h as f32,
                    w,
                    h,
                );
                assert!(
                    (values[(y * w + x) as usize] - expected).abs() < 1e-5,
                    "mask differs at {x},{y}"
                );
            }
        }
    }
    #[test]
    fn freezing_an_occluding_sleeve_does_not_tear_the_adjacent_fabric() {
        let (width, height) = (160usize, 200usize);
        let mut pixels = vec![0u8; width * height * 4];
        let mut alpha = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let person = (30..130).contains(&x) && (30..170).contains(&y);
                alpha[y * width + x] = if person { 255 } else { 0 };
                let pixel = if person {
                    [x as u8, 40, 60, 255]
                } else {
                    [10, 70, 180, 255]
                };
                pixels[(y * width + x) * 4..(y * width + x) * 4 + 4].copy_from_slice(&pixel);
            }
        }
        let source = RgbaImage::new(width as u32, height as u32, pixels).unwrap();
        let params = json!({"bodyPart":"waist","region":{"x":0.02,"y":0.1,"width":0.96,"height":0.8},"landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.5},{"x":0.5,"y":0.8}],"subjectMask":{"shape":"raster","width":width,"height":height,"data":alpha},"protectMask":{"shape":"rect","region":{"x":0.6875,"y":0.3,"width":0.0625,"height":0.4}},"adjustments":{"waist":80}});
        let output = adjust(&source, &params).unwrap();
        for x in 92..114 {
            let a = output.pixels[(100 * width + x) * 4] as i32;
            let b = output.pixels[(100 * width + x + 1) * 4] as i32;
            assert!(
                (0..=4).contains(&(b - a)),
                "fabric continuity broke at {x}: {a}->{b}"
            );
        }
        for x in 110..120 {
            assert_eq!(
                &output.pixels[(100 * width + x) * 4..(100 * width + x) * 4 + 4],
                &source.pixels[(100 * width + x) * 4..(100 * width + x) * 4 + 4]
            );
        }
    }
    #[test]
    fn segmented_body_applies_opacity_once_and_blending_keeps_context_fixed() {
        let (source, mut params) = segmented_person(80);
        let full = adjust(&source, &params).unwrap();
        params["opacity"] = json!(50);
        let half = adjust(&source, &params).unwrap();
        for i in 0..source.pixels.len() {
            assert!(
                (half.pixels[i] as i32
                    - ((full.pixels[i] as i32 + source.pixels[i] as i32 + 1) / 2))
                    .abs()
                    <= 1
            );
        }
        params["opacity"] = json!(0);
        assert_eq!(adjust(&source, &params).unwrap(), source);
        params["opacity"] = json!(100);
        params["blendMode"] = json!("multiply");
        let blended = adjust(&source, &params).unwrap();
        for i in 0..(source.width * source.height) as usize {
            if full.pixels[i * 4..i * 4 + 4] == source.pixels[i * 4..i * 4 + 4] {
                assert_eq!(
                    &blended.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }
    #[test]
    fn soft_foreground_fringe_does_not_leave_the_old_contour_as_a_ghost() {
        let (source, mut params) = segmented_person(80);
        let (width, height) = (source.width as usize, source.height as usize);
        let alpha: Vec<u8> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                if !(30..170).contains(&y) || x < 60 {
                    0
                } else if x < 95 {
                    255
                } else {
                    byte(((102 - x.min(102)) as f32 / 8.0).clamp(0.0, 1.0) * 255.0)
                }
            })
            .collect();
        params["subjectMask"] =
            json!({"shape":"raster","width":width,"height":height,"data":alpha});
        let output = adjust(&source, &params).unwrap();
        assert!(
            output.pixels[(100 * width + 99) * 4] < 20,
            "the alpha<.5 original contour must be reconstructed rather than frozen"
        );
        for y in 0..height {
            for x in 110..width {
                let i = (y * width + x) * 4;
                assert_eq!(
                    &output.pixels[i..i + 4],
                    &source.pixels[i..i + 4],
                    "known far background changed"
                );
            }
        }
    }
    #[test]
    fn contraction_keeps_known_background_fixed_with_noisy_soft_contours() {
        let (width, height) = (160usize, 200usize);
        let mut pixels = vec![0; width * height * 4];
        let mut alpha = vec![0u8; width * height];
        for y in 0..height {
            let radius = 31.0 + ((y as f32 * 0.31).sin() * 3.0);
            for x in 0..width {
                let i = y * width + x;
                let across = (x as f32 + 0.5 - 80.0).abs();
                let coverage = if (25..175).contains(&y) {
                    if across > radius + 3.0 && across < radius + 8.0 && y % 4 == 0 {
                        // Confidence islands outside a corrugated silhouette
                        // can be sampled into adjacent known-background gaps.
                        0.20
                    } else {
                        ((radius + 2.0 - across) / 4.0).clamp(0.0, 1.0)
                    }
                } else {
                    0.0
                };
                alpha[i] = byte(coverage * 255.0);
                pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    byte(20.0 + coverage * 190.0),
                    byte(40.0 + coverage * 30.0),
                    byte(110.0 - coverage * 50.0),
                    255,
                ]);
            }
        }
        let source = RgbaImage::new(width as u32, height as u32, pixels).unwrap();
        let params = json!({"bodyPart":"waist","region":{"x":0.1,"y":0.08,"width":0.8,"height":0.84},"landmarks":[{"x":0.5,"y":0.1},{"x":0.5,"y":0.5},{"x":0.5,"y":0.9}],"subjectMask":{"shape":"raster","width":width,"height":height,"data":alpha},"adjustments":{"waist":80}});
        let output = adjust(&source, &params).unwrap();
        assert_ne!(output, source, "the true foreground must still change");
        for (i, &coverage) in alpha.iter().enumerate() {
            if coverage <= 2 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "contraction overwrote known background at {i}"
                );
            }
        }
    }
    fn image(w: u32, h: u32) -> RgbaImage {
        let mut pixels = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let nx = (x as f32 + 0.5) / w as f32;
                let ny = (y as f32 + 0.5) / h as f32;
                let subject = (0.25..0.75).contains(&nx) && (0.15..0.92).contains(&ny);
                pixels.extend_from_slice(&if subject {
                    [(x * 255 / w) as u8, (y * 255 / h) as u8, 70, 255]
                } else {
                    [0, 100, 180, 255]
                });
            }
        }
        RgbaImage::new(w, h, pixels).unwrap()
    }
    fn test_subject_mask() -> Value {
        json!({"shape":"rect","region":{"x":0.25,"y":0.15,"width":0.5,"height":0.77}})
    }
    fn hips_params(value: i32) -> Value {
        json!({"bodyPart":"hips","region":{"x":0.2,"y":0.3,"width":0.6,"height":0.4},"subjectMask":test_subject_mask(),"mask":{"shape":"polygon","points":[{"x":0.2,"y":0.3},{"x":0.8,"y":0.3},{"x":0.8,"y":0.7},{"x":0.2,"y":0.7}],"feather":0},"landmarks":[{"x":0.35,"y":0.5},{"x":0.65,"y":0.5},{"x":0.5,"y":0.5}],"adjustments":{"hips":value}})
    }
    #[test]
    fn level_hips_use_roi_height_and_opposite_directions() {
        let source = image(100, 160);
        let narrow = adjust(&source, &hips_params(70)).unwrap();
        let wide = adjust(&source, &hips_params(-70)).unwrap();
        let foreground_width = |im: &RgbaImage| {
            (0..100)
                .filter(|&x| im.pixels[(80 * 100 + x) * 4] > 50)
                .count()
        };
        assert!(foreground_width(&narrow) < foreground_width(&source));
        assert!(foreground_width(&wide) > foreground_width(&source));
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
        let params = json!({"bodyPart":"left-thigh","subjectMask":test_subject_mask(),"region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},"landmarks":[{"x":0.3,"y":0.3},{"x":0.7,"y":0.7}],"mask":{"shape":"polygon","points":[{"x":0.25,"y":0.4},{"x":0.65,"y":0.8},{"x":0.75,"y":0.6},{"x":0.35,"y":0.2}],"feather":0},"adjustments":{"thighs":60}});
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
        let warp = Warp {
            axis,
            amount: 0.7,
            center: 0.5,
            extent: 0.5,
            limb: true,
        };
        let alpha = rasterize_mask(&params["subjectMask"], source.width, source.height);
        let d = ContourProfile::new(
            warp,
            &alpha,
            source.width as usize,
            source.height as usize,
            true,
            3.0,
        )
        .displacement(mid(points[0], points[1]));
        assert_eq!(d, (0.0, 0.0));
    }
    #[test]
    fn generic_and_specific_limb_controls_both_contribute() {
        let source = image(100, 160);
        let mut p = json!({"bodyPart":"left-calf","subjectMask":test_subject_mask(),"region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},"landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.8}],"mask":{"shape":"polygon","points":[{"x":0.2,"y":0.2},{"x":0.8,"y":0.2},{"x":0.8,"y":0.8},{"x":0.2,"y":0.8}],"feather":0},"adjustments":{"legs":30}});
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
        let mut p = json!({"bodyPart":"full-body","subjectMask":test_subject_mask(),"region":{"x":0.1,"y":0.1,"width":0.8,"height":0.85},"landmarks":points,"adjustments":{"height":75,"legLength":60}});
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
                "subjectMask":test_subject_mask(),
                "region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},
                "landmarks":[{"x":0.5,"y":0.2},{"x":0.5,"y":0.8}],
                "mask":{"shape":"polygon","points":[{"x":0.2,"y":0.2},{"x":0.8,"y":0.2},{"x":0.8,"y":0.8},{"x":0.2,"y":0.8}],"feather":0},
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
