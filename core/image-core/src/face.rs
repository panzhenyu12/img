//! Face contour editing separates the detected face foreground from its
//! backdrop. The original oval is a source alpha, never an output clipping
//! mask. Only the narrow strip exposed by a moved contour is reconstructed.
//! Pixel parsing supplies both the head alpha and the skin/hair face boundary.
//! A manual or legacy mesh contour remains a geometric approximation.

use crate::engine::{byte, percent};
use crate::mask::{coverage, PreparedMask, Region};
use crate::{ImageError, RgbaImage};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::VecDeque;

#[path = "face_mls.rs"]
mod mls;

type Point = (f32, f32);
const CONTOUR_KEYS: &[&str] = &[
    "faceWidth",
    "forehead",
    "foreheadWidth",
    "temples",
    "cheekbones",
    "jaw",
    "chin",
    "chinWidth",
    "chinPoint",
];

fn invalid(message: &str) -> ImageError {
    ImageError::InvalidOperation(message.into())
}
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn profile(y: f32, start: f32, peak: f32, end: f32) -> f32 {
    if y < peak {
        smoothstep(start, peak, y)
    } else {
        1.0 - smoothstep(peak, end, y)
    }
}
fn points(mask: &Value, w: f32, h: f32) -> Vec<Point> {
    mask.get("points")
        .and_then(Value::as_array)
        .map(|p| {
            p.iter()
                .filter_map(|p| {
                    let (x, y) = (p.get("x")?.as_f64()? as f32, p.get("y")?.as_f64()? as f32);
                    (x.is_finite() && y.is_finite()).then_some((x * w, y * h))
                })
                .collect()
        })
        .unwrap_or_default()
}
fn dot(a: Point, b: Point) -> f32 {
    a.0 * b.0 + a.1 * b.1
}
fn sub(a: Point, b: Point) -> Point {
    (a.0 - b.0, a.1 - b.1)
}
fn add(a: Point, b: Point) -> Point {
    (a.0 + b.0, a.1 + b.1)
}
fn scale(p: Point, a: f32) -> Point {
    (p.0 * a, p.1 * a)
}
fn center(points: &[Point]) -> Option<Point> {
    (!points.is_empty()).then(|| {
        scale(
            points.iter().copied().fold((0.0, 0.0), add),
            1.0 / points.len() as f32,
        )
    })
}
fn closest(p: Point, a: Point, b: Point) -> (f32, f32) {
    let segment = sub(b, a);
    let t = (dot(sub(p, a), segment) / dot(segment, segment).max(1e-6)).clamp(0.0, 1.0);
    let d = sub(p, add(a, scale(segment, t)));
    (dot(d, d).sqrt(), t)
}
fn inside(p: Point, polygon: &[Point]) -> bool {
    let mut inside = false;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
    }
    inside
}
/// Signed pixel distance gives a consistent one-pixel antialias fringe on
/// both square and tall images; normalized polygon feather does not.
fn alpha(p: Point, polygon: &[Point]) -> f32 {
    let mut inside = false;
    let mut distance = f32::MAX;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
        distance = distance.min(closest(p, a, b).0);
    }
    smoothstep(-0.5, 0.5, if inside { distance } else { -distance })
}

#[derive(Clone, Copy)]
struct Frame {
    origin: Point,
    across: Point,
    down: Point,
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
}
impl Frame {
    fn new(polygon: &[Point], semantics: Option<&Value>, w: f32, h: f32) -> Self {
        let semantic_center = |key: &str| {
            semantics
                .and_then(|m| m.get(key))
                .and_then(|m| center(&points(m, w, h)))
        };
        let eyes = semantic_center("leftEye").zip(semantic_center("rightEye"));
        let mouth = semantic_center("lips");
        let (origin, across) = if let Some((left, right)) = eyes {
            let mut direction = sub(right, left);
            if direction.0 < 0.0 {
                direction = scale(direction, -1.0);
            }
            let length = dot(direction, direction).sqrt();
            (
                scale(add(left, right), 0.5),
                scale(direction, 1.0 / length.max(1.0)),
            )
        } else {
            (center(polygon).unwrap_or((w * 0.5, h * 0.5)), (1.0, 0.0))
        };
        let mut down = (-across.1, across.0);
        if mouth.is_some_and(|mouth| dot(sub(mouth, origin), down) < 0.0) {
            down = scale(down, -1.0);
        }
        let mut frame = Self {
            origin,
            across,
            down,
            left: f32::MAX,
            right: f32::MIN,
            top: f32::MAX,
            bottom: f32::MIN,
        };
        for &p in polygon {
            let (u, v) = frame.coordinates(p);
            frame.left = frame.left.min(u);
            frame.right = frame.right.max(u);
            frame.top = frame.top.min(v);
            frame.bottom = frame.bottom.max(v);
        }
        frame
    }
    fn coordinates(self, p: Point) -> (f32, f32) {
        let p = sub(p, self.origin);
        (dot(p, self.across), dot(p, self.down))
    }
    fn position(self, u: f32, v: f32) -> Point {
        add(self.origin, add(scale(self.across, u), scale(self.down, v)))
    }
    fn width(self) -> f32 {
        self.right - self.left
    }
    fn height(self) -> f32 {
        self.bottom - self.top
    }
}

#[derive(Clone)]
struct Anchor {
    center: Point,
    across: Point,
    down: Point,
    rx: f32,
    ry: f32,
    polygon: Vec<Point>,
}
impl Anchor {
    fn freeze(&self, p: Point) -> f32 {
        let delta = sub(p, self.center);
        let r = (dot(delta, self.across) / self.rx).hypot(dot(delta, self.down) / self.ry);
        if !self.polygon.is_empty() {
            if r > 1.8 {
                return 0.0;
            }
            if inside(p, &self.polygon) {
                return 1.0;
            }
            let distance = self
                .polygon
                .iter()
                .enumerate()
                .map(|(i, &a)| closest(p, a, self.polygon[(i + 1) % self.polygon.len()]).0)
                .fold(f32::MAX, f32::min);
            // The old expanded enclosing ellipses froze a wide wedge of
            // cheek beside an eye: the contour stalled there and moved fully
            // one row lower, creating a visible inward notch. Freeze the
            // observed feature itself, with a two-pixel transition. MLS fixed
            // handles distribute the surrounding skin motion continuously.
            return 1.0 - smoothstep(0.0, 2.0, distance);
        }
        1.0 - smoothstep(1.0, 1.7, r)
    }
}
fn anchors(semantics: Option<&Value>, frame: Frame, w: f32, h: f32) -> Vec<Anchor> {
    let mut result = Vec::new();
    for key in [
        "leftEye",
        "rightEye",
        "leftBrow",
        "rightBrow",
        "lips",
        "teeth",
        "nose",
        "noseBridge",
        "noseWings",
    ] {
        let Some(polygon) = semantics.and_then(|m| m.get(key)).map(|m| points(m, w, h)) else {
            continue;
        };
        let Some(c) = center(&polygon) else { continue };
        let (mut rx, mut ry): (f32, f32) = (0.0, 0.0);
        for &p in &polygon {
            rx = rx.max(dot(sub(p, c), frame.across).abs());
            ry = ry.max(dot(sub(p, c), frame.down).abs());
        }
        // An enclosing ellipse protects the entire feature, including its
        // corners. The surrounding transition moves skin continuously.
        result.push(Anchor {
            center: c,
            across: frame.across,
            down: frame.down,
            rx: (rx * 1.45 + 1.0).max(1.0),
            ry: (ry * 1.45 + 1.0).max(1.0),
            polygon,
        });
    }
    // A detector without a nose polygon still fixes the central nose bridge
    // and tip. This conservative anchor is omitted for explicit hand contours.
    if semantics.is_some_and(|m| m.get("nose").is_none()) {
        result.push(Anchor {
            center: frame.position(
                (frame.left + frame.right) * 0.5,
                frame.top + frame.height() * 0.54,
            ),
            across: frame.across,
            down: frame.down,
            rx: frame.width() * 0.105,
            ry: frame.height() * 0.15,
            polygon: Vec::new(),
        });
    }
    result
}

fn target_contour(polygon: &[Point], frame: Frame, values: &Value) -> Vec<Point> {
    polygon
        .iter()
        .map(|&p| {
            let (u, v) = frame.coordinates(p);
            let y = (v - frame.top) / frame.height().max(1.0);
            let center = (frame.left + frame.right) * 0.5;
            let side = (u - center).signum();
            let lateral = smoothstep(0.08, 0.28, ((u - center) / (frame.width() * 0.5)).abs());
            // The widest cheek edge moves fully. Falloff is anatomical/vertical,
            // never tied to the silhouette's own bounding-box boundary.
            let cheek_band = smoothstep(0.17, 0.34, y) * (1.0 - smoothstep(0.88, 1.0, y));
            let mut inward = percent(values, "faceWidth") * 0.065 * cheek_band;
            inward += percent(values, "temples") * 0.050 * profile(y, 0.12, 0.28, 0.46);
            inward += percent(values, "cheekbones") * 0.055 * profile(y, 0.25, 0.48, 0.69);
            inward += percent(values, "jaw") * 0.065 * profile(y, 0.50, 0.76, 0.97);
            inward -= percent(values, "foreheadWidth") * 0.060 * profile(y, 0.0, 0.16, 0.36);
            inward -= percent(values, "chinWidth") * 0.050 * profile(y, 0.69, 0.89, 1.04);
            inward += percent(values, "chinPoint") * 0.055 * profile(y, 0.72, 0.93, 1.02);
            let du = -side * frame.width() * inward.clamp(-0.11, 0.11) * lateral;
            let dv = frame.height()
                * (percent(values, "chin") * 0.075 * smoothstep(0.74, 1.0, y)
                    - percent(values, "forehead") * 0.060 * (1.0 - smoothstep(0.0, 0.30, y)));
            frame.position(u + du, v + dv)
        })
        .collect()
}

struct ContourField<'a> {
    source: &'a [Point],
    target: &'a [Point],
    anchors: &'a [Anchor],
    support: f32,
    mls: mls::RigidMls,
}
impl ContourField<'_> {
    fn inverse_unanchored(&self, p: Point) -> (Point, f32) {
        let mut distance = f32::MAX;
        let mut movement = (0.0, 0.0);
        for i in 0..self.target.len() {
            let next = (i + 1) % self.target.len();
            let (d, t) = closest(p, self.target[i], self.target[next]);
            if d < distance {
                distance = d;
                let a = sub(self.source[i], self.target[i]);
                let b = sub(self.source[next], self.target[next]);
                movement = add(scale(a, 1.0 - t), scale(b, t));
            }
        }
        let rigid = sub(self.mls.map(p), p);
        // Exact contour-segment inverse sampling is a portrait adaptation:
        // enforce the requested raster boundary in a narrow three-pixel band,
        // while rigid MLS distributes interior motion with fixed features.
        movement = add(
            scale(movement, 1.0 - smoothstep(0.0, 3.0, distance)),
            scale(rigid, smoothstep(0.0, 3.0, distance)),
        );
        let weight = 1.0 - smoothstep(0.0, self.support, distance);
        (scale(movement, weight), distance)
    }
    fn anchor_weight(&self, p: Point) -> f32 {
        let mut weight = 1.0;
        for anchor in self.anchors {
            weight *= 1.0 - anchor.freeze(p);
        }
        weight
    }
}

/// A four-pixel sampling grid avoids solving all MLS handles for every
/// interior pixel; grid caching is an implementation optimization.
/// The actual contour band stays exact, and feature protection is evaluated
/// at the true pixel rather than interpolated across protected features.
struct ContourGrid {
    origin: Point,
    step: f32,
    width: usize,
    height: usize,
    samples: Vec<Option<(Point, f32)>>,
}
impl ContourGrid {
    fn new(origin: Point, w: usize, h: usize) -> Self {
        let step = 4.0;
        let width = w.div_ceil(step as usize) + 1;
        let height = h.div_ceil(step as usize) + 1;
        let samples = vec![None; width * height];
        Self {
            origin,
            step,
            width,
            height,
            samples,
        }
    }
    fn inverse(&mut self, field: &ContourField<'_>, p: Point) -> Point {
        let x = ((p.0 - self.origin.0) / self.step).clamp(0.0, (self.width - 1) as f32);
        let y = ((p.1 - self.origin.1) / self.step).clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let mut movement = (0.0, 0.0);
        let mut distance = 0.0;
        for (x, y, weight) in [
            (x0, y0, (1.0 - tx) * (1.0 - ty)),
            (x1, y0, tx * (1.0 - ty)),
            (x0, y1, (1.0 - tx) * ty),
            (x1, y1, tx * ty),
        ] {
            let index = y * self.width + x;
            let sample = *self.samples[index].get_or_insert_with(|| {
                field.inverse_unanchored((
                    self.origin.0 + x as f32 * self.step,
                    self.origin.1 + y as f32 * self.step,
                ))
            });
            movement = add(movement, scale(sample.0, weight));
            distance += sample.1 * weight;
        }
        if distance < 3.0 + self.step * 2.0_f32.sqrt() {
            movement = field.inverse_unanchored(p).0;
        }
        scale(movement, field.anchor_weight(p))
    }
}

fn neighbors(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % w, i / w);
    [
        (x > 0).then(|| i - 1),
        (x + 1 < w).then(|| i + 1),
        (y > 0).then(|| i - w),
        (y + 1 < h).then(|| i + w),
    ]
    .into_iter()
    .flatten()
}

/// Propagate only clean exterior donors into the newly exposed band. The
/// surviving original foreground and frozen pixels are never donors. This
/// deterministic local fill cannot reconstruct complex hidden scene content.
fn reconstruct(
    source: &RgbaImage,
    holes: &[bool],
    subject: &[f32],
    frozen: &[bool],
    donor_excluded: &[bool],
    known_layer: &[u8],
    estimated_background: Option<&RgbaImage>,
) -> Result<RgbaImage, ImageError> {
    let w = source.width as usize;
    let h = source.height as usize;
    let count = w * h;
    let target: Vec<bool> = (0..count)
        .map(|i| {
            // Non-neck person pixels can bridge a hole to the actual backdrop.
            // Reconstruct this hidden layer without editing those output pixels
            // or copying their colors. Explicit clean neck pixels remain known
            // donors for the separate underlying neck layer.
            (holes[i]
                || (subject[i] > 0.0 && subject[i] < 1.0)
                || (donor_excluded[i] && subject[i] <= 1e-6))
                && !frozen[i]
                && known_layer[i] == 0
        })
        .collect();
    let protected: Vec<bool> = (0..count)
        .map(|i| {
            ((subject[i] > 0.0 || donor_excluded[i]) && !target[i] && known_layer[i] == 0)
                || frozen[i]
        })
        .collect();
    // The lower jaw can uncover neck rather than scene background. Partition
    // the hidden layer by the nearest accessible clean exterior donor. Neck,
    // visible hair and backdrop donors never mix across their layer join.
    // `known_layer` is 0 for scene, 1 for neck and 2 for fixed visible hair.
    let mut layer: Vec<u8> = known_layer.to_vec();
    let mut classified: Vec<bool> = target.iter().map(|t| !t).collect();
    let mut front = VecDeque::new();
    for i in 0..count {
        if !target[i] && !protected[i] && neighbors(i, w, h).any(|n| target[n]) {
            front.push_back(i);
        }
    }
    while let Some(i) = front.pop_front() {
        for n in neighbors(i, w, h) {
            if target[n] && !classified[n] {
                layer[n] = layer[i];
                classified[n] = true;
                front.push_back(n);
            }
        }
    }
    let mut known: Vec<bool> = target.iter().map(|t| !t).collect();
    let mut queue = VecDeque::new();
    for (i, &is_target) in target.iter().enumerate() {
        if is_target
            && neighbors(i, w, h).any(|n| known[n] && !protected[n] && layer[n] == layer[i])
        {
            queue.push_back(i);
        }
    }
    let mut result = source.clone();
    while let Some(i) = queue.pop_front() {
        if known[i] {
            continue;
        }
        let mut donors = [0_usize; 4];
        let mut donor_count = 0;
        for n in neighbors(i, w, h).filter(|&n| known[n] && !protected[n] && layer[n] == layer[i]) {
            donors[donor_count] = n;
            donor_count += 1;
        }
        let donors = &donors[..donor_count];
        if donor_count == 0 {
            continue;
        }
        // Premultiplied averaging also propagates transparent backdrops.
        let sum_alpha: f32 = donors
            .iter()
            .map(|&n| result.pixels[n * 4 + 3] as f32)
            .sum();
        for c in 0..3 {
            result.pixels[i * 4 + c] = if sum_alpha > 1e-6 {
                byte(
                    donors
                        .iter()
                        .map(|&n| result.pixels[n * 4 + c] as f32 * result.pixels[n * 4 + 3] as f32)
                        .sum::<f32>()
                        / sum_alpha,
                )
            } else {
                0
            };
        }
        result.pixels[i * 4 + 3] = byte(sum_alpha / donors.len() as f32);
        known[i] = true;
        for n in neighbors(i, w, h) {
            if target[n] && !known[n] {
                queue.push_back(n);
            }
        }
    }
    for (i, &hole) in holes.iter().enumerate() {
        if hole && !known[i] {
            if let Some(background) = estimated_background {
                // An uncertain interior alpha island has no direct exterior
                // donor. The paper's F/B estimator supplies a regularized
                // background estimate; it is not recovered hidden texture.
                result.pixels[i * 4..i * 4 + 4]
                    .copy_from_slice(&background.pixels[i * 4..i * 4 + 4]);
                known[i] = true;
            } else {
                return Err(invalid(
                    "face contour requires clean, unprotected background beside the revealed edge",
                ));
            }
        }
    }
    // Harmonic color extrapolation in float eliminates nearest-donor Voronoi
    // stripes. Eight byte-quantized iterations leave visible vertical blocks
    // on a shortened chin; solve only actual exposed pixels to convergence.
    // The neck and scene layers never mix.
    let hole_indices: Vec<usize> = (0..count).filter(|&i| holes[i] && known[i]).collect();
    let mut colors: Vec<[f32; 4]> = result
        .pixels
        .chunks_exact(4)
        .map(|p| {
            let a = p[3] as f32 / 255.0;
            [
                p[0] as f32 * a,
                p[1] as f32 * a,
                p[2] as f32 * a,
                p[3] as f32,
            ]
        })
        .collect();
    for _ in 0..128 {
        let mut residual = 0.0_f32;
        for parity in 0..2 {
            for &i in &hole_indices {
                if ((i % w + i / w) & 1) != parity {
                    continue;
                }
                let mut sum = [0.0; 4];
                let mut donor_count = 0;
                for n in neighbors(i, w, h)
                    .filter(|&n| !protected[n] && known[n] && layer[n] == layer[i])
                {
                    for c in 0..4 {
                        sum[c] += colors[n][c];
                    }
                    donor_count += 1;
                }
                if donor_count == 0 {
                    continue;
                }
                for (channel, total) in colors[i].iter_mut().zip(sum) {
                    let value = total / donor_count as f32;
                    residual = residual.max((value - *channel).abs());
                    *channel = value;
                }
            }
        }
        if residual < 0.05 {
            break;
        }
    }
    for i in hole_indices {
        let a = colors[i][3] / 255.0;
        for (c, &channel) in colors[i].iter().take(3).enumerate() {
            result.pixels[i * 4 + c] = if a > 1e-6 { byte(channel / a) } else { 0 };
        }
        result.pixels[i * 4 + 3] = byte(colors[i][3]);
    }
    Ok(result)
}

/// Bilinear sampling of RGB premultiplied by both original RGBA alpha and
/// source face membership prevents backdrop colors leaking into the face.
fn sample(source: &RgbaImage, subject: &[f32], x: f32, y: f32) -> ([f32; 4], f32) {
    let w = source.width as usize;
    let h = source.height as usize;
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (tx, ty) = (x - x.floor(), y - y.floor());
    let mut pixel = [0.0; 4];
    let mut membership = 0.0;
    for (x, y, weight) in [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ] {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            continue;
        }
        let i = y as usize * w + x as usize;
        let face = subject[i];
        let a = source.pixels[i * 4 + 3] as f32 / 255.0 * face * weight;
        membership += face * weight;
        for (c, value) in pixel.iter_mut().enumerate().take(3) {
            *value += source.pixels[i * 4 + c] as f32 * a;
        }
        pixel[3] += a;
    }
    (pixel, membership.clamp(0.0, 1.0))
}

/// Move the observed neck illumination with the jaw for newly uncovered
/// pixels. Harmonic fill has no boundary shadow and averages away neck detail.
/// A corresponding clean neck donor retains that real surface and shading;
/// unsupported/occluded donors keep the separate-layer reconstruction above.
/// Existing neck output and scene pixels are never warped by this function.
#[cfg(test)]
fn restore_exposed_neck_legacy(
    background: &mut RgbaImage,
    crop: &RgbaImage,
    holes: &[bool],
    known_neck: &[bool],
    field: &ContourField<'_>,
    origin: Point,
) {
    if !known_neck.iter().any(|&v| v) {
        return;
    }
    let clean: Vec<f32> = known_neck
        .iter()
        .map(|&v| if v { 1.0 } else { 0.0 })
        .collect();
    let w = crop.width as usize;
    for (i, &hole) in holes.iter().enumerate() {
        if !hole {
            continue;
        }
        let p = (
            origin.0 + (i % w) as f32 + 0.5,
            origin.1 + (i / w) as f32 + 0.5,
        );
        if inside(p, field.target) {
            continue;
        }
        let mut best = (f32::MAX, 0, 0.0);
        for j in 0..field.target.len() {
            let (distance, t) = closest(
                p,
                field.target[j],
                field.target[(j + 1) % field.target.len()],
            );
            if distance < best.0 {
                best = (distance, j, t);
            }
        }
        let (distance, j, t) = best;
        let next = (j + 1) % field.source.len();
        let boundary = add(
            scale(field.source[j], 1.0 - t),
            scale(field.source[next], t),
        );
        let target_boundary = add(
            scale(field.target[j], 1.0 - t),
            scale(field.target[next], t),
        );
        let tangent = sub(field.source[next], field.source[j]);
        let length = tangent.0.hypot(tangent.1);
        if length < 1e-5 {
            continue;
        }
        let mut normal = (-tangent.1 / length, tangent.0 / length);
        if inside(add(boundary, scale(normal, 2.0)), field.source) {
            normal = scale(normal, -1.0);
        }
        // The candidate must meet the unchanged observed neck at the old
        // boundary. Direct donor copying makes a second hard jaw-shaped ring.
        // Taper to the reconstructed layer across the actual exposed width.
        let exposed_width = dot(sub(boundary, target_boundary), normal);
        if exposed_width <= 1e-5 {
            continue;
        }
        let weight = smoothstep(
            0.0,
            1.0,
            ((exposed_width - distance) / exposed_width).clamp(0.0, 1.0),
        );
        if weight <= 1e-6 {
            continue;
        }
        for offset in [0.0, 1.0, 2.0, 3.0, 4.0] {
            let donor = add(boundary, scale(normal, distance + offset));
            let (pixel, membership) = sample(
                crop,
                &clean,
                donor.0 - origin.0 - 0.5,
                donor.1 - origin.1 - 0.5,
            );
            if membership < 0.999 || pixel[3] <= 1e-6 {
                continue;
            }
            for (c, &value) in pixel.iter().enumerate().take(3) {
                background.pixels[i * 4 + c] = byte(
                    background.pixels[i * 4 + c] as f32 * (1.0 - weight)
                        + value / pixel[3] * weight,
                );
            }
            background.pixels[i * 4 + 3] = byte(
                background.pixels[i * 4 + 3] as f32 * (1.0 - weight) + pixel[3] * 255.0 * weight,
            );
            break;
        }
    }
}

/// Arc-distance tangent avoids treating a one-pixel marching-squares zigzag
/// as an anatomical 90-degree jaw turn. Geometry still uses the exact curve;
/// only the neck illumination correspondence uses this local normal.
fn neck_normal(polygon: &[Point], edge: usize, t: f32, reach: f32) -> Point {
    let n = polygon.len();
    let boundary = add(
        scale(polygon[edge], 1.0 - t),
        scale(polygon[(edge + 1) % n], t),
    );
    let walk = |forward: bool| {
        let mut current = boundary;
        let mut distance = 0.0;
        let mut j = if forward { (edge + 1) % n } else { edge };
        for _ in 0..n {
            let next = polygon[j];
            let segment = sub(next, current);
            let length = segment.0.hypot(segment.1);
            if distance + length >= reach {
                return add(
                    current,
                    scale(segment, (reach - distance) / length.max(1e-6)),
                );
            }
            current = next;
            distance += length;
            j = if forward {
                (j + 1) % n
            } else {
                (j + n - 1) % n
            };
        }
        current
    };
    let tangent = sub(walk(true), walk(false));
    let length = tangent.0.hypot(tangent.1);
    if length < 1e-5 {
        return (0.0, 0.0);
    }
    let mut normal = (-tangent.1 / length, tangent.0 / length);
    if inside(add(boundary, scale(normal, 2.0)), polygon) {
        normal = scale(normal, -1.0);
    }
    normal
}

/// Reposition only observed neck illumination after jaw contraction. Neck
/// texture stays at its source coordinates: RGB_out = RGB_source + L(q)-L(p).
/// Newly exposed neck receives its observed donor at the corresponding distance
/// from the old jaw. Explicit protection excludes both output and donor pixels.
fn restore_neck_lighting(
    background: &mut RgbaImage,
    crop: &RgbaImage,
    holes: &[bool],
    known_neck: &[bool],
    frozen: &[bool],
    field: &ContourField<'_>,
    origin: Point,
) -> Vec<bool> {
    let (w, h) = (crop.width as usize, crop.height as usize);
    let mut changed = vec![false; w * h];
    let clean: Vec<f32> = known_neck
        .iter()
        .zip(frozen)
        .map(|(&known, &fixed)| if known && !fixed { 1.0 } else { 0.0 })
        .collect();
    if !clean.iter().any(|&v| v > 0.0) {
        return changed;
    }
    let extent = field.source.iter().map(|p| p.0).fold(f32::MIN, f32::max)
        - field.source.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let radius = (extent * 0.022).round().clamp(2.0, 8.0) as usize;
    let stride = w + 1;
    let mut integral = vec![[0.0_f64; 4]; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = [0.0_f64; 4];
        for x in 0..w {
            let i = y * w + x;
            row[3] += clean[i] as f64;
            for (c, value) in row.iter_mut().enumerate().take(3) {
                *value += crop.pixels[i * 4 + c] as f64 * clean[i] as f64;
            }
            for c in 0..4 {
                integral[(y + 1) * stride + x + 1][c] = integral[y * stride + x + 1][c] + row[c];
            }
        }
    }
    let mut low = crop.clone();
    let mut reliable = vec![0.0; w * h];
    let mut seam = vec![None; w * h];
    let mut bridge_valid = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let (x0, y0, x1, y1) = (
                x.saturating_sub(radius),
                y.saturating_sub(radius),
                (x + radius + 1).min(w),
                (y + radius + 1).min(h),
            );
            let sum: [f64; 4] = std::array::from_fn(|c| {
                integral[y1 * stride + x1][c]
                    - integral[y0 * stride + x1][c]
                    - integral[y1 * stride + x0][c]
                    + integral[y0 * stride + x0][c]
            });
            if sum[3] < ((x1 - x0) * (y1 - y0)) as f64 * 0.15 {
                continue;
            }
            let i = y * w + x;
            reliable[i] = 1.0;
            for c in 0..3 {
                low.pixels[i * 4 + c] = byte((sum[c] / sum[3]) as f32);
            }
        }
    }
    for i in 0..w * h {
        if frozen[i] || (!holes[i] && clean[i] <= 0.0) {
            continue;
        }
        let p = (
            origin.0 + (i % w) as f32 + 0.5,
            origin.1 + (i / w) as f32 + 0.5,
        );
        if inside(p, field.target) {
            continue;
        }
        let mut best = (f32::MAX, 0, 0.0);
        for j in 0..field.target.len() {
            let (d, t) = closest(
                p,
                field.target[j],
                field.target[(j + 1) % field.target.len()],
            );
            if d < best.0 {
                best = (d, j, t);
            }
        }
        let (distance, j, t) = best;
        let next = (j + 1) % field.source.len();
        let old = add(
            scale(field.source[j], 1.0 - t),
            scale(field.source[next], t),
        );
        let new = add(
            scale(field.target[j], 1.0 - t),
            scale(field.target[next], t),
        );
        let normal = neck_normal(field.source, j, t, (radius as f32).max(3.0));
        let movement = sub(old, new);
        let exposed = dot(movement, normal);
        if exposed <= 0.05 {
            continue;
        }
        let old_distance = field
            .source
            .iter()
            .enumerate()
            .map(|(j, &a)| closest(p, a, field.source[(j + 1) % field.source.len()]).0)
            .fold(f32::MAX, f32::min);
        // A semantic class join is not stationary neck texture. Only a narrow
        // old-contour band with space on both sides of the newly moved jaw can
        // participate in the join correction below.
        if old_distance < 3.0 && distance > 4.0 {
            seam[i] = Some((normal, old_distance));
        }
        let falloff = 1.0
            - smoothstep(
                exposed + radius as f32,
                exposed + (exposed * 2.0).max(radius as f32 * 4.0),
                distance,
            );
        if falloff <= 1e-5 {
            continue;
        }
        let mut donor = add(p, scale(movement, falloff));
        let mut observed = None;
        for offset in [0.0, 1.0, 2.0, 3.0, 4.0] {
            let q = add(donor, scale(normal, offset));
            let sample = sample(crop, &clean, q.0 - origin.0 - 0.5, q.1 - origin.1 - 0.5);
            if sample.1 >= 0.999 && sample.0[3] > 0.999 {
                donor = q;
                observed = Some(sample.0);
                break;
            }
        }
        let Some(observed) = observed else {
            continue;
        };
        bridge_valid[i] = 1.0;
        if holes[i] {
            // The target contour is the zero-width join. Taper the sampled
            // neck donor from the target into the old neck boundary instead of
            // writing a second opaque jaw arc into harmonic background.
            let blend = smoothstep(0.0, exposed.max(1.0), distance);
            for (c, &v) in observed.iter().take(3).enumerate() {
                let donor = v / observed[3];
                let current = background.pixels[i * 4 + c] as f32;
                background.pixels[i * 4 + c] = byte(current * (1.0 - blend) + donor * blend);
            }
            background.pixels[i * 4 + 3] = crop.pixels[i * 4 + 3];
        } else {
            if reliable[i] <= 0.0 {
                continue;
            }
            let (mapped, reliability) = sample(
                &low,
                &reliable,
                donor.0 - origin.0 - 0.5,
                donor.1 - origin.1 - 0.5,
            );
            if reliability < 0.999 || mapped[3] <= 1e-6 {
                continue;
            }
            for (c, &v) in mapped.iter().take(3).enumerate() {
                background.pixels[i * 4 + c] = byte(
                    crop.pixels[i * 4 + c] as f32 + v / mapped[3] - low.pixels[i * 4 + c] as f32,
                );
            }
        }
        // The old subpixel jaw edge is a mixed face/neck observation, not
        // stationary neck microtexture. Relighting its retained high frequency
        // would retain bright fragments of the removed skin. Clean only the
        // Confident parsing can put the physical mixed RGB edge a few pixels
        // beyond its alpha .5 contour. A six-pixel taper removes that offset
        // skin fragment; farther visible neck keeps its high frequency above.
        if clean[i] > 0.0 && !holes[i] {
            let demix = 1.0 - smoothstep(1.5, 6.0, old_distance);
            if demix > 0.0 {
                for (c, &v) in observed.iter().take(3).enumerate() {
                    let current = background.pixels[i * 4 + c] as f32;
                    let donor = v / observed[3];
                    let delta = donor - current;
                    // A skin/neck mixed pixel must be removed decisively when
                    // it is brighter than the observed neck donor. Conversely,
                    // a dark submental shadow mapped to a lighter donor cannot
                    // be injected as a one-pixel white arc: cap only positive
                    // relighting at the original contour and let the low-pass
                    // field take over away from that edge.
                    let bounded = if delta > 0.0 { delta.min(20.0) } else { delta };
                    background.pixels[i * 4 + c] = byte(current + bounded * demix);
                }
            }
        }
        changed[i] = background.pixels[i * 4..i * 4 + 4] != crop.pixels[i * 4..i * 4 + 4];
    }
    // Couple the reconstructed strip and the relit observed neck across their
    // semantic join. A one-dimensional harmonic bridge removes an old-jaw
    // impulse without smoothing along the jaw or across the new face edge.
    // Its endpoints must both already belong to this revealed/observed neck
    // layer; protected pixels and scene samples are not allowed as donors.
    let bridge_source = background.clone();
    for (i, &join) in seam.iter().enumerate() {
        let Some((normal, distance)) = join else {
            continue;
        };
        if bridge_valid[i] <= 0.0 || !changed[i] {
            continue;
        }
        let p = ((i % w) as f32, (i / w) as f32);
        let (before, before_valid) = sample(
            &bridge_source,
            &bridge_valid,
            p.0 - normal.0 * 3.0,
            p.1 - normal.1 * 3.0,
        );
        let (after, after_valid) = sample(
            &bridge_source,
            &bridge_valid,
            p.0 + normal.0 * 3.0,
            p.1 + normal.1 * 3.0,
        );
        if before_valid < 0.999 || after_valid < 0.999 || before[3] <= 1e-6 || after[3] <= 1e-6 {
            continue;
        }
        let weight = 1.0 - smoothstep(0.75, 3.0, distance);
        for c in 0..3 {
            let bridge = (before[c] / before[3] + after[c] / after[3]) * 0.5;
            background.pixels[i * 4 + c] =
                byte(bridge_source.pixels[i * 4 + c] as f32 * (1.0 - weight) + bridge * weight);
        }
        changed[i] = background.pixels[i * 4..i * 4 + 4] != crop.pixels[i * 4..i * 4 + 4];
    }

    // The parser's neck class is intentionally conservative: on a soft jaw
    // edge it often marks every third or fourth pixel as a high-confidence
    // neck texel.  The lighting correspondence above is then evaluated on
    // those isolated texels and can leave a dotted bright/dark arc under a
    // shortened chin.  Smooth only this old-contour join, along its tangent;
    // the surrounding neck keeps its original high-frequency texture and the
    // face, hair and scene are never sampled by this pass.  A robust median
    // is used instead of a box blur so a necklace or a sharp shadow does not
    // bleed across the join.
    let seam_source = background.clone();
    for (i, &join) in seam.iter().enumerate() {
        let Some((normal, distance)) = join else {
            continue;
        };
        if !changed[i] || frozen[i] {
            continue;
        }
        let tangent = (-normal.1, normal.0);
        let mut values: Vec<[f32; 3]> = Vec::new();
        let px = (i % w) as f32;
        let py = (i / w) as f32;
        for step in -6..=6 {
            if step == 0 {
                continue;
            }
            let x = (px + tangent.0 * step as f32).round() as isize;
            let y = (py + tangent.1 * step as f32).round() as isize;
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                continue;
            }
            let j = y as usize * w + x as usize;
            // Include the corresponding join even if its local donor was not
            // reliable enough to mark `changed`; its observed neck value is a
            // better robust statistic than a sparse bright donor.  A bridge
            // pixel is already known to belong to the same exposed layer.
            if seam[j].is_none() && bridge_valid[j] <= 0.0 {
                continue;
            }
            let alpha = seam_source.pixels[j * 4 + 3] as f32 / 255.0;
            if alpha <= 1e-3 {
                continue;
            }
            values.push([
                seam_source.pixels[j * 4] as f32,
                seam_source.pixels[j * 4 + 1] as f32,
                seam_source.pixels[j * 4 + 2] as f32,
            ]);
        }
        if values.len() < 2 {
            continue;
        }
        let mut median = [0.0; 3];
        for c in 0..3 {
            let mut channel: Vec<f32> = values.iter().map(|v| v[c]).collect();
            channel.sort_by(|a, b| a.total_cmp(b));
            median[c] = channel[channel.len() / 2];
        }
        let weight = 0.62 * (1.0 - smoothstep(0.75, 3.0, distance));
        if weight <= 1e-4 {
            continue;
        }
        for (c, value) in median.into_iter().enumerate() {
            let current = seam_source.pixels[i * 4 + c] as f32;
            background.pixels[i * 4 + c] = byte(current * (1.0 - weight) + value * weight);
        }
    }
    changed
}

/// A probability matte can cross .5 gradually. Requiring a .25/.75 jump in
/// three pixels incorrectly rejects a genuine marching-squares contour.
/// Check the level set itself and a local sign change instead of edge contrast.
fn validate_isoline(
    matte: &Value,
    polygon: &[Point],
    source: &RgbaImage,
    name: &str,
) -> Result<(), ImageError> {
    let (w, h) = (source.width as f32, source.height as f32);
    let mw = matte.get("width").and_then(Value::as_u64).unwrap_or(1) as f32;
    let mh = matte.get("height").and_then(Value::as_u64).unwrap_or(1) as f32;
    let tolerance = (w / mw).max(h / mh).max(1.0) * 1.5;
    let read = |p: Point| coverage(matte, p.0 / w, p.1 / h, source.width, source.height);
    let matches = polygon
        .iter()
        .filter(|&&p| {
            // Polygonized pixel-center alpha can round a geometric vertex by
            // a fraction of a pixel. Require the same level set within that
            // fraction, rather than accepting arbitrary low/high probabilities.
            let mut nearby_minimum = read(p);
            let mut nearby_maximum = nearby_minimum;
            for dx in [-0.75 * w / mw, 0.0, 0.75 * w / mw] {
                for dy in [-0.75 * h / mh, 0.0, 0.75 * h / mh] {
                    let value = read((p.0 + dx, p.1 + dy));
                    nearby_minimum = nearby_minimum.min(value);
                    nearby_maximum = nearby_maximum.max(value);
                }
            }
            if nearby_minimum > 0.5 || nearby_maximum < 0.5 {
                return false;
            }
            let mut minimum = 1.0_f32;
            let mut maximum = 0.0_f32;
            for dx in [-tolerance, 0.0, tolerance] {
                for dy in [-tolerance, 0.0, tolerance] {
                    let value = read((p.0 + dx, p.1 + dy));
                    minimum = minimum.min(value);
                    maximum = maximum.max(value);
                }
            }
            minimum < 0.5 - 1e-5 && maximum > 0.5 + 1e-5
        })
        .count();
    if matches * 10 < polygon.len() * 9 {
        return Err(invalid(&format!("{name} does not follow the supplied pixel face alpha; extract its alpha 0.5 silhouette instead of using landmark oval points")));
    }
    Ok(())
}

pub(crate) fn adjust(source: &RgbaImage, params: &Value) -> Result<RgbaImage, ImageError> {
    #[cfg(not(target_arch = "wasm32"))]
    let mut profile_start = std::time::Instant::now();
    macro_rules! mark {
        ($label:expr) => {
            #[cfg(not(target_arch = "wasm32"))]
            if std::env::var_os("FACE_CORE_PROFILE").is_some() {
                eprintln!(
                    "face core {} {:.1} ms",
                    $label,
                    profile_start.elapsed().as_secs_f64() * 1000.0
                );
                profile_start = std::time::Instant::now();
            }
        };
    }
    let values = params.get("adjustments").unwrap_or(params);
    for key in ["faceHeight", "midface", "doubleChin"] {
        if percent(values, key).abs() > 1e-6 {
            return Err(invalid(&format!(
                "{key} requires an anatomical reconstruction algorithm and is not implemented"
            )));
        }
    }
    let contour_active = CONTOUR_KEYS
        .iter()
        .any(|key| percent(values, key).abs() > 1e-6);
    let symmetry = percent(values, "symmetry").max(0.0);
    if !contour_active {
        if symmetry <= 1e-6 {
            return Ok(source.clone());
        }
        let rendered = crate::geometry::local_adjust_kind(source, params, "feature.adjust")?;
        return Ok(crate::mask::composite(source, &rendered, params));
    }
    let (w, h) = (source.width as f32, source.height as f32);
    let semantics = params.get("semanticMasks");
    let face_matte = semantics
        .and_then(|m| m.get("face"))
        .filter(|m| m.get("shape").and_then(Value::as_str) == Some("raster"));
    let head_outline = semantics
        .and_then(|m| m.get("faceOutline").or_else(|| m.get("face")))
        .filter(|m| m.get("shape").and_then(Value::as_str) == Some("polygon"));
    let face_contour = semantics
        .and_then(|m| m.get("faceContour"))
        .filter(|m| m.get("shape").and_then(Value::as_str) == Some("polygon"));
    let contour_matte = semantics
        .and_then(|m| m.get("faceContourMask"))
        .filter(|m| m.get("shape").and_then(Value::as_str) == Some("raster"));
    if face_contour.is_some() && (face_matte.is_none() || contour_matte.is_none()) {
        return Err(invalid(
            "pixel faceContour requires both face foreground alpha and a matching faceContourMask",
        ));
    }
    let semantic_outline = face_contour.or(head_outline);
    let manual_outline = params
        .get("mask")
        .filter(|m| m.get("shape").and_then(Value::as_str) == Some("polygon"));
    if face_matte.is_some() && head_outline.is_none() {
        return Err(invalid("pixel face alpha requires a matching faceOutline polygon extracted from its silhouette"));
    }
    let outline = semantic_outline.or(manual_outline).ok_or_else(|| {
        invalid("face contour edits require face detection or a manually drawn face polygon")
    })?;
    let polygon = points(outline, w, h);
    if polygon.len() < 3 {
        return Err(invalid(
            "face contour requires at least three perimeter points",
        ));
    }
    let area = polygon
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let b = polygon[(i + 1) % polygon.len()];
            a.0 * b.1 - b.0 * a.1
        })
        .sum::<f32>()
        .abs()
        * 0.5;
    if area >= w * h * 0.92 {
        return Err(invalid(
            "draw the actual face perimeter; a whole-image polygon is not a face contour",
        ));
    }
    // Hair/ears can extend beyond the anatomical facial oval, especially for
    // long hairstyles. The ordered detector oval supplies anatomical scale;
    // the fitted pixel outline supplies the actual moving alpha boundary.
    let landmark_oval = face_matte
        .and_then(|_| params.get("landmarks"))
        .map(|p| points(&json!({"points":p}), w, h))
        .unwrap_or_default();
    let frame = Frame::new(
        if landmark_oval.len() >= 12 {
            &landmark_oval
        } else {
            &polygon
        },
        semantics,
        w,
        h,
    );
    if frame.width() < 4.0 || frame.height() < 4.0 {
        return Err(invalid("face contour is too small to reshape"));
    }
    if let Some(matte) = face_matte {
        validate_isoline(
            matte,
            &points(head_outline.expect("checked outline"), w, h),
            source,
            "faceOutline",
        )?;
    }
    if let Some(matte) = contour_matte.filter(|_| face_contour.is_some()) {
        validate_isoline(matte, &polygon, source, "faceContour")?;
    }
    let target = target_contour(&polygon, frame, values);
    // A purely shrinking operation must not fill concave notches which were
    // outside the original foreground. Moving every x toward the center is
    // insufficient to prove polygon containment for parsed concave contours.
    let contraction_only = [
        ("faceWidth", 1.0),
        ("temples", 1.0),
        ("cheekbones", 1.0),
        ("jaw", 1.0),
        ("chinPoint", 1.0),
        ("foreheadWidth", -1.0),
        ("chinWidth", -1.0),
        ("forehead", -1.0),
        ("chin", -1.0),
    ]
    .iter()
    .all(|&(key, sign)| percent(values, key) * sign >= -1e-6);
    let target_bounds = target.iter().fold(
        (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
        |(l, t, r, b), p| (l.min(p.0), t.min(p.1), r.max(p.0), b.max(p.1)),
    );
    let anchors = anchors(semantics, frame, w, h);
    let mut handles = Vec::new();
    let mut lengths = vec![0.0];
    for i in 0..polygon.len() {
        let length = sub(polygon[(i + 1) % polygon.len()], polygon[i]);
        lengths.push(lengths.last().copied().unwrap_or(0.0) + length.0.hypot(length.1));
    }
    let perimeter = lengths.last().copied().unwrap_or(1.0);
    for k in 0..64 {
        let at = perimeter * k as f32 / 64.0;
        let edge = lengths
            .partition_point(|&d| d <= at)
            .saturating_sub(1)
            .min(polygon.len() - 1);
        let t = (at - lengths[edge]) / (lengths[edge + 1] - lengths[edge]).max(1e-6);
        let next = (edge + 1) % polygon.len();
        handles.push((
            add(scale(target[edge], 1.0 - t), scale(target[next], t)),
            add(scale(polygon[edge], 1.0 - t), scale(polygon[next], t)),
        ));
    }
    for anchor in &anchors {
        if !anchor.polygon.is_empty() {
            handles.push((anchor.center, anchor.center));
            // Fix handles on the observed feature rather than its enclosing
            // ellipse: an ellipse extreme can sit in editable cheek skin and
            // become an identity island outside the requested target contour.
            for k in 0..8 {
                let p = anchor.polygon[k * anchor.polygon.len() / 8];
                handles.push((p, p));
            }
            continue;
        }
        for offset in [
            (0.0, 0.0),
            (-anchor.rx, 0.0),
            (anchor.rx, 0.0),
            (0.0, -anchor.ry),
            (0.0, anchor.ry),
        ] {
            let p = add(
                anchor.center,
                add(scale(anchor.across, offset.0), scale(anchor.down, offset.1)),
            );
            handles.push((p, p));
        }
    }
    let field = ContourField {
        source: &polygon,
        target: &target,
        anchors: &anchors,
        support: (frame.width() * 0.28).max(3.0),
        mls: mls::RigidMls::new(handles),
    };
    let manual = semantic_outline.is_none();
    // Older detector projects stored the same oval twice. Interpret that
    // redundant mask as a source contour, just like a manual face tracing.
    let redundant_outline = params
        .get("mask")
        .is_some_and(|m| m.get("points") == outline.get("points"));
    let mut support_params = Cow::Borrowed(params);
    if manual || redundant_outline {
        support_params
            .to_mut()
            .as_object_mut()
            .expect("validated params")
            .remove("mask");
        let selected = Region::read(params.get("region"));
        if polygon.iter().all(|p| selected.contains(p.0 / w, p.1 / h)) {
            let pad = frame.width() * 0.18;
            let left = polygon.iter().map(|p| p.0).fold(f32::MAX, f32::min) - pad;
            let top = polygon.iter().map(|p| p.1).fold(f32::MAX, f32::min) - pad;
            let right = polygon.iter().map(|p| p.0).fold(f32::MIN, f32::max) + pad;
            let bottom = polygon.iter().map(|p| p.1).fold(f32::MIN, f32::max) + pad;
            support_params.to_mut()["region"] = json!({"x":left.max(0.0)/w,"y":top.max(0.0)/h,"width":(right.min(w)-left.max(0.0))/w,"height":(bottom.min(h)-top.max(0.0))/h});
        }
    }
    let region = Region::read(support_params.get("region"));
    let context_outline = head_outline.map(|m| points(m, w, h)).unwrap_or_default();
    let pad = field.support + frame.width() * 0.11 + 4.0;
    let left = (polygon
        .iter()
        .chain(&target)
        .chain(&context_outline)
        .map(|p| p.0)
        .fold(f32::MAX, f32::min)
        - pad)
        .floor()
        .max(0.0) as usize;
    let top = (polygon
        .iter()
        .chain(&target)
        .chain(&context_outline)
        .map(|p| p.1)
        .fold(f32::MAX, f32::min)
        - pad)
        .floor()
        .max(0.0) as usize;
    let right = (polygon
        .iter()
        .chain(&target)
        .chain(&context_outline)
        .map(|p| p.0)
        .fold(f32::MIN, f32::max)
        + pad)
        .ceil()
        .min(w) as usize;
    let bottom = (polygon
        .iter()
        .chain(&target)
        .chain(&context_outline)
        .map(|p| p.1)
        .fold(f32::MIN, f32::max)
        + pad)
        .ceil()
        .min(h) as usize;
    let (cw, ch) = (right - left, bottom - top);
    let count = cw * ch;
    let mut grid = ContourGrid::new((left as f32 + 0.5, top as f32 + 0.5), cw, ch);
    let source_matte = face_matte.map(PreparedMask::new);
    let contour_mask = contour_matte.map(PreparedMask::new);
    let person_mask = params.get("subjectMask").map(PreparedMask::new);
    let protect_mask = params.get("protectMask").map(PreparedMask::new);
    let neck_mask = semantics.and_then(|m| m.get("neck")).map(PreparedMask::new);
    let hair_mask = semantics.and_then(|m| m.get("hair")).map(PreparedMask::new);
    let output_mask = support_params.get("mask").map(PreparedMask::new);
    let fixed_masks: Vec<_> = ["accessories", "ears", "hair"]
        .iter()
        .filter_map(|key| semantics.and_then(|m| m.get(key)))
        .map(PreparedMask::new)
        .collect();
    let mut input = source.clone();
    if symmetry > 1e-6 {
        let mut symmetry_params = params.clone();
        symmetry_params["adjustments"] = json!({"symmetry":symmetry*100.0});
        symmetry_params["opacity"] = json!(100);
        let corrected =
            crate::geometry::local_adjust_kind(source, &symmetry_params, "feature.adjust")?;
        input = crate::mask::composite(source, &corrected, &symmetry_params);
    }
    let mut crop_pixels = Vec::with_capacity(count * 4);
    let mut subject = Vec::with_capacity(count);
    let mut face_membership = Vec::with_capacity(count);
    let mut frozen = Vec::with_capacity(count);
    let mut donor_frozen = Vec::with_capacity(count);
    let mut protection = Vec::with_capacity(count);
    let mut donor_excluded = Vec::with_capacity(count);
    let mut known_layer = Vec::with_capacity(count);
    let mut neck_lighting_known = Vec::with_capacity(count);
    mark!("prepare");
    for y in top..bottom {
        for x in left..right {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let i = (y * source.width as usize + x) * 4;
            crop_pixels.extend_from_slice(&input.pixels[i..i + 4]);
            subject.push(
                source_matte
                    .as_ref()
                    .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                    .unwrap_or_else(|| alpha(p, &polygon)),
            );
            face_membership.push(
                contour_mask
                    .as_ref()
                    .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                    .unwrap_or_else(|| alpha(p, &polygon)),
            );
            let person_coverage = person_mask
                .as_ref()
                .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                .unwrap_or(0.0);
            donor_excluded.push(person_coverage > 1.0 / 255.0);
            let protected = protect_mask
                .as_ref()
                .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                .unwrap_or(0.0);
            let accessory = fixed_masks
                .iter()
                .any(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5);
            let hair = hair_mask
                .as_ref()
                .is_some_and(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5);
            // Record the observed fixed lower layer independently of the
            // semantic face confidence. The physical matte below decides
            // which of these texels are foreground and must be reconstructed.
            known_layer.push(
                if neck_mask.as_ref().is_some_and(|m| {
                    m.coverage(p.0 / w, p.1 / h, source.width, source.height) > 0.8
                }) {
                    1
                } else if hair {
                    2
                } else {
                    0
                },
            );
            // A confidently classified non-neck person pixel is not a soft
            // facial fringe, even if approximate head alpha overlaps it.
            // Keep other people, clothing and undetected ears fixed.
            let other_person = person_coverage >= 0.8
                // The face and neck probabilities can both fall below .5 in
                // the one-pixel class transition. Freezing that uncertainty
                // leaves dotted copies of the old chin after extension. Only
                // a pixel wholly outside both facial supports is evidence of
                // another person/clothing; hair/ears/accessories have their
                // own authoritative protection masks above.
                && subject.last().copied().unwrap_or(1.0) <= 1.0 / 255.0
                && face_membership.last().copied().unwrap_or(1.0) <= 1.0 / 255.0
                && !neck_mask.as_ref().is_some_and(|m| {
                    m.coverage(p.0 / w, p.1 / h, source.width, source.height) > 1.0 / 255.0
                });
            // Parsing sometimes assigns the antialias fringe to neither face
            // nor neck. A newly covering face may occlude that narrow old
            // boundary, while truly separate people farther away stay frozen.
            // This does not override explicit hair/ear/accessory/user masks.
            let covered_old_fringe = other_person
                && !contraction_only
                && inside(p, &target)
                && polygon
                    .iter()
                    .enumerate()
                    .any(|(j, &a)| closest(p, a, polygon[(j + 1) % polygon.len()]).0 <= 3.0);
            frozen.push(
                protected >= 1.0 - 1e-6 || accessory || (other_person && !covered_old_fringe),
            );
            // Existing hair remains fixed in the output, but its observed
            // color is the visible underlying layer at a skin/hair boundary.
            // Refusing it as a donor leaves a skin-colored band after a cheek
            // contraction. User-frozen pixels, ears, adornments and another
            // person's pixels remain forbidden donors.
            donor_frozen.push(
                protected >= 1.0 - 1e-6
                    || (accessory && !hair)
                    || (other_person && !covered_old_fringe),
            );
            protection.push(protected);
            neck_lighting_known.push(
                person_coverage >= 0.8
                    && face_membership.last().copied().unwrap_or(1.0) <= 1.0 / 255.0
                    && neck_mask.as_ref().is_some_and(|m| {
                        m.coverage(p.0 / w, p.1 / h, source.width, source.height) > 0.8
                    })
                    && protected <= 1e-6
                    && !frozen.last().copied().unwrap_or(true),
            );
        }
    }
    let crop = RgbaImage::new(cw as u32, ch as u32, crop_pixels)?;
    mark!("crop masks");
    // A head alpha includes hair and ears. Using it to sample a facial
    // deformation moves hair texels through the new skin edge and defeats the
    // layer separation precisely where slimming is meant to alter silhouette.
    // The actual moving foreground is the parsed facial class alpha. The head
    // alpha remains the context/protection support above, while visible hair
    // supplies a stationary underlayer for the newly exposed skin strip.
    let moving_alpha = if contour_mask.is_some() {
        &face_membership
    } else {
        &subject
    };
    let mut color_alpha = if face_matte.is_some() {
        crate::face_matting::refine_portrait_alpha(&crop, moving_alpha)
    } else {
        moving_alpha.clone()
    };
    // An independently parsed head is stationary context support around the
    // facial class, including hair/ears. Its confidently empty exterior remains
    // fixed even when an ambiguous local color-line projection suggests skin.
    // A single face tracing reused as head has no such independent evidence.
    let independent_head = contour_matte.is_some() && face_matte != contour_matte;
    if independent_head {
        for (i, alpha) in color_alpha.iter_mut().enumerate() {
            if subject[i] <= 1e-6 {
                *alpha = 0.0;
            }
        }
    }
    // The refined matte represents physical foreground at the RGB edge.
    // Use it consistently for old coverage, holes and donor eligibility. A
    // binary class zero can still contain observed mixed skin: preserving it
    // verbatim or using it as clean backdrop leaves the old silhouette.
    for (i, layer) in known_layer.iter_mut().enumerate() {
        if color_alpha[i] > 1e-6 {
            *layer = 0;
        }
    }
    let color_layers = crate::face_matting::estimate(&crop, &color_alpha);
    mark!("F/B estimate");
    let foreground = color_layers.as_ref().map(|(f, _)| f).unwrap_or(&crop);
    let mut samples = vec![([0.0; 4], 0.0); count];
    let mut changed = vec![false; count];
    let mut holes = vec![false; count];
    for y in 0..ch {
        for x in 0..cw {
            let i = y * cw + x;
            let p = ((left + x) as f32 + 0.5, (top + y) as f32 + 0.5);
            if frozen[i] || !region.contains(p.0 / w, p.1 / h) {
                continue;
            }
            let selection = output_mask
                .as_ref()
                .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                .unwrap_or(1.0);
            if selection <= 1e-6 {
                continue;
            }
            // Deform the facial layer, not the whole head texture. Existing
            // hair/ears/background outside the source face remain fixed unless
            // the target contour explicitly covers that previously exterior
            // region. Raster holes do not become an invented filled polygon.
            let old_coverage = if contour_matte.is_some() {
                color_alpha[i]
            } else {
                face_membership[i]
            };
            if old_coverage < if contour_matte.is_some() { 1e-6 } else { 0.5 }
                && (contraction_only
                    || !(p.0 >= target_bounds.0
                        && p.1 >= target_bounds.1
                        && p.0 <= target_bounds.2
                        && p.1 <= target_bounds.3
                        && !inside(p, &polygon)
                        && inside(p, &target)))
            {
                continue;
            }
            let movement = scale(grid.inverse(&field, p), selection * (1.0 - protection[i]));
            if movement.0.abs() + movement.1.abs() < 1e-5 {
                continue;
            }
            let sample = sample(
                foreground,
                &color_alpha,
                x as f32 + movement.0,
                y as f32 + movement.1,
            );
            // Background with zero old and new foreground is always bit-exact.
            if color_alpha[i] <= 1e-6 && sample.1 <= 1e-6 {
                continue;
            }
            samples[i] = sample;
            changed[i] = true;
            holes[i] = color_alpha[i] > 1e-6 && sample.1 < 1.0 - 1e-6;
        }
    }
    mark!("sample grid");
    let mut background = if holes.iter().any(|&h| h) {
        reconstruct(
            &crop,
            &holes,
            &color_alpha,
            &donor_frozen,
            &donor_excluded,
            &known_layer,
            color_layers.as_ref().map(|(_, b)| b),
        )?
    } else {
        crop.clone()
    };
    let lighting_changed = restore_neck_lighting(
        &mut background,
        &crop,
        &holes,
        &neck_lighting_known,
        &frozen,
        &field,
        (left as f32, top as f32),
    );
    for (i, &light_changed) in lighting_changed.iter().enumerate() {
        if light_changed {
            changed[i] = true;
        }
    }
    mark!("reconstruct");
    let mut rendered = input.clone();
    for i in 0..count {
        if frozen[i] {
            let j = ((top + i / cw) * source.width as usize + left + i % cw) * 4;
            rendered.pixels[j..j + 4].copy_from_slice(&source.pixels[j..j + 4]);
        }
        if !changed[i] {
            continue;
        }
        let (pixel, membership) = samples[i];
        let alpha_bg = background.pixels[i * 4 + 3] as f32 / 255.0;
        let bg_weight = alpha_bg * (1.0 - membership);
        let final_alpha = pixel[3] + bg_weight;
        let j = ((top + i / cw) * source.width as usize + left + i % cw) * 4;
        for (c, &channel) in pixel.iter().enumerate().take(3) {
            rendered.pixels[j + c] = if final_alpha > 1e-6 {
                byte((channel + background.pixels[i * 4 + c] as f32 * bg_weight) / final_alpha)
            } else {
                0
            };
        }
        rendered.pixels[j + 3] = byte(final_alpha * 255.0);
    }
    // ROI/protection/opacity/blend are applied once over the expanded support,
    // never again clipped by the original foreground alpha in engine.rs.
    mark!("compose layers");
    let output = crate::mask::composite(source, &rendered, &support_params);
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("FACE_CORE_PROFILE").is_some() {
        eprintln!(
            "face core final composite {:.1} ms",
            profile_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn portrait(w: u32, h: u32) -> (RgbaImage, Value) {
        let mut pixels = Vec::new();
        let outline: Vec<Point> = (0..48)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 48.0;
                (
                    w as f32 * (0.5 + 0.24 * a.cos()),
                    h as f32 * (0.5 + 0.37 * a.sin()),
                )
            })
            .collect();
        for y in 0..h {
            for x in 0..w {
                let face = alpha((x as f32 + 0.5, y as f32 + 0.5), &outline) > 0.5;
                let color = if face {
                    [220, 155, 100, 255]
                } else {
                    [20 + (x % 16) as u8, 50 + (y % 16) as u8, 130, 255]
                };
                pixels.extend_from_slice(&color);
            }
        }
        let mask = json!({"shape":"polygon","points":outline.iter().map(|p|json!({"x":p.0/w as f32,"y":p.1/h as f32})).collect::<Vec<_>>()});
        (
            RgbaImage::new(w, h, pixels).unwrap(),
            json!({"region":{"x":0.1,"y":0.02,"width":0.8,"height":0.96},"semanticMasks":{"face":mask},"adjustments":{"faceWidth":100}}),
        )
    }
    fn span(image: &RgbaImage, y: usize) -> (usize, usize) {
        let x: Vec<usize> = (0..image.width as usize)
            .filter(|&x| image.pixels[(y * image.width as usize + x) * 4] > 150)
            .collect();
        (*x.first().unwrap(), *x.last().unwrap())
    }
    #[test]
    fn saturated_binary_parsing_removes_old_physical_fringe_with_correct_background() {
        let (w, h) = (160_u32, 240_u32);
        let outline: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (80.0 + 38.4 * a.cos(), 120.0 + 88.8 * a.sin())
            })
            .collect();
        let foreground = [220.0, 155.0, 100.0];
        let background = [20.0, 40.0, 60.0];
        let mut physical = Vec::new();
        let mut confidence = Vec::new();
        let mut pixels = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let distance = outline
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| closest(p, a, outline[(i + 1) % outline.len()]).0)
                    .fold(f32::MAX, f32::min);
                let signed = if inside(p, &outline) {
                    distance
                } else {
                    -distance
                };
                let alpha = (0.5 + signed / 3.0).clamp(0.0, 1.0);
                physical.push(alpha);
                confidence.push(if alpha >= 0.5 { 255_u8 } else { 0 });
                for c in 0..3 {
                    pixels.push(byte(foreground[c] * alpha + background[c] * (1.0 - alpha)));
                }
                pixels.push(255);
            }
        }
        let source = RgbaImage::new(w, h, pixels).unwrap();
        let contour = json!({"shape":"polygon","points":outline.iter().map(|p|
            json!({"x":p.0/w as f32,"y":p.1/h as f32})).collect::<Vec<_>>()});
        let matte = json!({"shape":"raster","width":w,"height":h,"data":confidence});
        let params = json!({"semanticMasks":{"face":matte,"faceOutline":contour,
            "faceContourMask":matte,"faceContour":contour},"landmarks":contour["points"],
            "adjustments":{"faceWidth":100}});
        let output = adjust(&source, &params).unwrap();
        let old_span = span(&source, 120);
        let new_span = span(&output, 120);
        assert!(
            new_span.0 >= old_span.0 + 4 && new_span.1 <= old_span.1 - 4,
            "inner physical contour must move: {old_span:?} -> {new_span:?}"
        );
        // Known F/B means the newly exposed strip has a numerical target,
        // including old alpha=0 class texels whose real opacity was nonzero.
        // Leave the three-pixel physical transition beside the new contour.
        for x in 40..new_span.0.saturating_sub(3) {
            let i = (120 * w as usize + x) * 4;
            for (c, expected) in background.iter().enumerate() {
                assert!(
                    (output.pixels[i + c] as f32 - expected).abs() <= 3.0,
                    "old edge/background contamination at {x}, channel {c}: {} != {expected}",
                    output.pixels[i + c]
                );
            }
        }
        assert!(physical[120 * w as usize + 41] > 0.0 && physical[120 * w as usize + 41] < 0.5);
        for (i, &alpha) in physical.iter().enumerate() {
            if alpha == 0.0 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "pure observed background must stay exact at {i}"
                );
            }
        }
    }
    #[test]
    fn actual_widest_contour_moves_inward_but_exterior_grid_does_not() {
        let (source, p) = portrait(160, 240);
        let output = adjust(&source, &p).unwrap();
        let before = span(&source, 120);
        let after = span(&output, 120);
        assert!(
            after.0 >= before.0 + 4,
            "left contour {before:?} -> {after:?}"
        );
        assert!(
            after.1 <= before.1 - 4,
            "right contour {before:?} -> {after:?}"
        );
        for y in 0..240 {
            for x in 0..160 {
                let i = (y * 160 + x) * 4;
                if source.pixels[i] < 100 {
                    assert_eq!(
                        &output.pixels[i..i + 4],
                        &source.pixels[i..i + 4],
                        "background ({x},{y})"
                    );
                }
            }
        }
        // The original cheek texels are removed, rather than blended back as
        // an immovable, translucent old silhouette.
        for x in before.0..after.0 - 1 {
            assert!(output.pixels[(120 * 160 + x) * 4] < 100);
        }
        let nose = (120 * 160 + 80) * 4;
        assert_eq!(
            &output.pixels[nose..nose + 4],
            &source.pixels[nose..nose + 4]
        );
    }
    #[test]
    fn negative_slimming_expands_foreground_over_stationary_background() {
        let (source, mut p) = portrait(160, 240);
        p["adjustments"]["faceWidth"] = json!(-100);
        let output = adjust(&source, &p).unwrap();
        let before = span(&source, 120);
        let after = span(&output, 120);
        assert!(after.0 <= before.0 - 4 && after.1 >= before.1 + 4);
        for y in 0..240 {
            for x in [0, 10, 149, 159] {
                let i = (y * 160 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
    }
    #[test]
    fn semantic_feature_and_user_frozen_pixels_remain_exact() {
        let (mut source, mut p) = portrait(160, 240);
        let eye = json!({"shape":"polygon","points":[{"x":0.28,"y":0.44},{"x":0.40,"y":0.44},{"x":0.40,"y":0.48},{"x":0.28,"y":0.48}]});
        p["semanticMasks"]["leftEye"] = eye;
        p["protectMask"] =
            json!({"shape":"rect","region":{"x":0.66,"y":0.4,"width":0.15,"height":0.2}});
        for y in 106..114 {
            for x in 46..62 {
                let i = (y * 160 + x) * 4;
                source.pixels[i..i + 4].copy_from_slice(&[5, 15, 25, 255]);
            }
        }
        let output = adjust(&source, &p).unwrap();
        for y in 106..114 {
            for x in 46..62 {
                let i = (y * 160 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
        for y in 98..142 {
            for x in 108..128 {
                let i = (y * 160 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
    }
    #[test]
    fn eye_protection_does_not_freeze_the_adjacent_cheek_silhouette() {
        let (mut source, mut params) = portrait(240, 360);
        params["semanticMasks"]["leftEye"] = json!({"shape":"polygon","points":[
            {"x":0.3125,"y":0.48},{"x":0.37,"y":0.48},
            {"x":0.37,"y":0.52},{"x":0.3125,"y":0.52}
        ]});
        // This eye is close to the face edge, but still has room for the
        // full requested cheek motion. A wide enclosing ellipse would stall
        // the skin boundary even though none of the eye is being covered.
        for y in 175..185 {
            for x in 76..88 {
                let i = (y * 240 + x) * 4;
                source.pixels[i..i + 4].copy_from_slice(&[5, 15, 25, 255]);
            }
        }
        let before = span(&source, 180);
        let output = adjust(&source, &params).unwrap();
        let after = span(&output, 180);
        assert!(
            after.0 >= before.0 + 6,
            "eye-adjacent edge stalled: {before:?} -> {after:?}"
        );
        for y in 175..185 {
            for x in 76..88 {
                let i = (y * 240 + x) * 4;
                assert_eq!(
                    &output.pixels[i..i + 4],
                    &source.pixels[i..i + 4],
                    "observed eye at ({x}, {y})"
                );
            }
        }
    }
    #[test]
    fn opacity_and_roi_are_applied_once() {
        let (source, mut p) = portrait(160, 240);
        p["region"] = json!({"x":0.1,"y":0.4,"width":0.8,"height":0.2});
        let full = adjust(&source, &p).unwrap();
        p["opacity"] = json!(50);
        let half = adjust(&source, &p).unwrap();
        for y in 0..240 {
            for x in 0..160 {
                let i = (y * 160 + x) * 4;
                if !(96..144).contains(&y) {
                    assert_eq!(&half.pixels[i..i + 4], &source.pixels[i..i + 4]);
                } else {
                    for c in 0..3 {
                        let expected =
                            byte((source.pixels[i + c] as f32 + full.pixels[i + c] as f32) * 0.5);
                        assert_eq!(half.pixels[i + c], expected);
                    }
                }
            }
        }
        p["opacity"] = json!(0);
        assert_eq!(adjust(&source, &p).unwrap(), source);
    }
    #[test]
    fn explicit_manual_contour_moves_across_its_original_mask_boundary() {
        let (source, mut p) = portrait(160, 240);
        p["mask"] = p["semanticMasks"]["face"].clone();
        p.as_object_mut().unwrap().remove("semanticMasks");
        let output = adjust(&source, &p).unwrap();
        let before = span(&source, 120);
        let after = span(&output, 120);
        assert!(after.0 >= before.0 + 4 && after.1 <= before.1 - 4);
        p["adjustments"]["faceWidth"] = json!(-100);
        let wide = adjust(&source, &p).unwrap();
        assert!(span(&wide, 120).0 <= before.0 - 4);
    }
    #[test]
    fn bare_rectangle_is_rejected_and_unsupported_controls_do_not_fake_geometry() {
        let (source, _) = portrait(160, 240);
        assert!(adjust(&source, &json!({"adjustments":{"faceWidth":50}})).is_err());
        assert!(adjust(&source, &json!({"adjustments":{"doubleChin":50}})).is_err());
        assert_eq!(adjust(&source, &json!({"adjustments":{}})).unwrap(), source);
    }
    #[test]
    fn transparent_backdrop_and_semtransparent_photograph_keep_alpha_layers() {
        let (mut source, p) = portrait(160, 240);
        for pixel in source.pixels.chunks_exact_mut(4) {
            if pixel[0] < 100 {
                pixel[3] = 0;
            }
        }
        let output = adjust(&source, &p).unwrap();
        let before = span(&source, 120);
        let after = span(&output, 120);
        assert!(output.pixels[(120 * 160 + before.0) * 4 + 3] < 20);
        assert_eq!(output.pixels[(120 * 160 + after.0 + 2) * 4 + 3], 255);
        let (mut source, p) = portrait(160, 240);
        for pixel in source.pixels.chunks_exact_mut(4) {
            pixel[3] = 128;
        }
        let output = adjust(&source, &p).unwrap();
        assert!(output.pixels.chunks_exact(4).all(|p| p[3] == 128));
    }

    #[test]
    fn each_supported_width_control_changes_its_actual_contour_in_both_directions() {
        let (source, mut params) = portrait(240, 360);
        let top = 360.0 * 0.13;
        let height = 360.0 * 0.74;
        for (key, position, positive_narrows) in [
            ("faceWidth", 0.50, true),
            ("temples", 0.28, true),
            ("cheekbones", 0.48, true),
            ("jaw", 0.76, true),
            ("foreheadWidth", 0.16, false),
            ("chinWidth", 0.89, false),
            ("chinPoint", 0.93, true),
        ] {
            let y = (top + height * position) as usize;
            let original = span(&source, y);
            for amount in [-100, 100] {
                params["adjustments"] = json!({key:amount});
                let edited = adjust(&source, &params).unwrap();
                let current = span(&edited, y);
                let narrowing = (amount > 0) == positive_narrows;
                if narrowing {
                    assert!(
                        current.0 > original.0 && current.1 < original.1,
                        "{key} {amount}: {original:?} -> {current:?}"
                    );
                } else {
                    assert!(
                        current.0 < original.0 && current.1 > original.1,
                        "{key} {amount}: {original:?} -> {current:?}"
                    );
                }
                // This is a visible geometry contract, not a pixel-change
                // test: both sides must move in the named anatomical region.
                let forehead = span(&edited, 54);
                if key != "foreheadWidth" {
                    assert_eq!(
                        forehead,
                        span(&source, 54),
                        "{key} should not alter forehead apex"
                    );
                }
            }
        }
    }

    fn column_span(image: &RgbaImage, x: usize) -> (usize, usize) {
        let ys: Vec<usize> = (0..image.height as usize)
            .filter(|&y| image.pixels[(y * image.width as usize + x) * 4] > 150)
            .collect();
        (*ys.first().unwrap(), *ys.last().unwrap())
    }
    #[test]
    fn forehead_and_chin_lengths_move_only_their_endpoint_and_keep_feature_pixels() {
        let (mut source, mut params) = portrait(240, 360);
        let feature = |x: f32, y: f32, rw: f32, rh: f32| json!({"shape":"polygon","points":[{"x":x,"y":y},{"x":x+rw,"y":y},{"x":x+rw,"y":y+rh},{"x":x,"y":y+rh}]});
        params["semanticMasks"]["leftEye"] = feature(0.35, 0.35, 0.08, 0.03);
        params["semanticMasks"]["rightEye"] = feature(0.57, 0.35, 0.08, 0.03);
        params["semanticMasks"]["lips"] = feature(0.42, 0.63, 0.16, 0.04);
        let mut feature_pixels = Vec::new();
        for (x0, x1, y0, y1) in [
            (85, 101, 127, 135),
            (139, 153, 127, 135),
            (104, 136, 228, 239),
        ] {
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * 240 + x) * 4;
                    source.pixels[i..i + 4].copy_from_slice(&[5, 15, 25, 255]);
                    feature_pixels.push(i);
                }
            }
        }
        let original = column_span(&source, 120);
        for key in ["forehead", "chin"] {
            for amount in [-100, 100] {
                params["adjustments"] = json!({key:amount});
                let output = adjust(&source, &params).unwrap();
                let current = column_span(&output, 120);
                if key == "forehead" {
                    assert_eq!(current.1, original.1);
                    assert!(
                        (current.0 < original.0) == (amount > 0),
                        "{key} {amount}: {original:?} -> {current:?}"
                    );
                } else {
                    assert_eq!(current.0, original.0);
                    assert!(
                        (current.1 > original.1) == (amount > 0),
                        "{key} {amount}: {original:?} -> {current:?}"
                    );
                }
                for &i in &feature_pixels {
                    assert_eq!(
                        &output.pixels[i..i + 4],
                        &source.pixels[i..i + 4],
                        "{key} moved a feature"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_controls_are_exact_and_a_second_face_in_the_support_background_is_untouched() {
        let (mut source, mut params) = portrait(160, 240);
        for y in 100..124 {
            for x in 10..28 {
                let i = (y * 160 + x) * 4;
                source.pixels[i..i + 4].copy_from_slice(&[180, 125, 65, 255]);
            }
        }
        let output = adjust(&source, &params).unwrap();
        for y in 100..124 {
            for x in 10..28 {
                let i = (y * 160 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
        for &key in CONTOUR_KEYS {
            params["adjustments"] = json!({key:0});
            assert_eq!(adjust(&source, &params).unwrap(), source);
        }
    }

    #[test]
    fn eye_axis_roll_preserves_the_same_contour_travel_in_pixel_coordinates() {
        let (base, mut params) = portrait(280, 380);
        let original = points(&params["semanticMasks"]["face"], 280.0, 380.0);
        let mut measured = Vec::new();
        for angle in [-0.55_f32, 0.0, 0.55] {
            let across = (angle.cos(), angle.sin());
            let down = (-across.1, across.0);
            let transform = |p: Point| {
                add(
                    (140.0, 190.0),
                    add(scale(across, p.0 - 140.0), scale(down, p.1 - 190.0)),
                )
            };
            let outline: Vec<Point> = original.iter().map(|&p| transform(p)).collect();
            let mut source = base.clone();
            for y in 0..380 {
                for x in 0..280 {
                    let i = (y * 280 + x) * 4;
                    let face = alpha((x as f32 + 0.5, y as f32 + 0.5), &outline) > 0.5;
                    source.pixels[i..i + 4].copy_from_slice(if face {
                        &[220, 155, 100, 255]
                    } else {
                        &[20, 50, 130, 255]
                    });
                }
            }
            let poly = |ps: Vec<Point>| json!({"shape":"polygon","points":ps.iter().map(|p|json!({"x":p.0/280.0,"y":p.1/380.0})).collect::<Vec<_>>()});
            params["semanticMasks"]["face"] = poly(outline);
            for (key, cx) in [("leftEye", 112.0), ("rightEye", 168.0)] {
                params["semanticMasks"][key] = poly(vec![
                    transform((cx - 8.0, 143.0)),
                    transform((cx + 8.0, 143.0)),
                    transform((cx + 8.0, 149.0)),
                    transform((cx - 8.0, 149.0)),
                ]);
            }
            params["region"] = json!({"x":0.0,"y":0.0,"width":1.0,"height":1.0});
            let output = adjust(&source, &params).unwrap();
            let read_span = |image: &RgbaImage| {
                let ds: Vec<i32> = (-110..=110)
                    .filter(|&d| {
                        let p = add((140.0, 190.0), scale(across, d as f32));
                        let i = (p.1.round() as usize * 280 + p.0.round() as usize) * 4;
                        image.pixels[i] > 150
                    })
                    .collect();
                (*ds.first().unwrap(), *ds.last().unwrap())
            };
            let before = read_span(&source);
            let after = read_span(&output);
            let travel = (after.0 - before.0, before.1 - after.1);
            assert!(travel.0 >= 7 && travel.1 >= 7, "roll {angle}: {travel:?}");
            measured.push(travel);
        }
        let upright = measured[1];
        for travel in measured {
            assert!((travel.0 - upright.0).abs() <= 2 && (travel.1 - upright.1).abs() <= 2);
        }
    }

    fn raster_params(source: &RgbaImage, mut params: Value) -> Value {
        let polygon = points(
            &params["semanticMasks"]["face"],
            source.width as f32,
            source.height as f32,
        );
        let data: Vec<u8> = (0..source.width as usize * source.height as usize)
            .map(|i| {
                byte(
                    alpha(
                        (
                            (i % source.width as usize) as f32 + 0.5,
                            (i / source.width as usize) as f32 + 0.5,
                        ),
                        &polygon,
                    ) * 255.0,
                )
            })
            .collect();
        params["semanticMasks"]["faceOutline"] = params["semanticMasks"]["face"].clone();
        params["semanticMasks"]["face"] =
            json!({"shape":"raster","width":source.width,"height":source.height,"data":data});
        params
    }
    #[test]
    fn pixel_matte_moves_the_true_edge_and_rejects_an_unrelated_landmark_oval() {
        let (source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        let out = adjust(&source, &p).unwrap();
        let before = span(&source, 120);
        let after = span(&out, 120);
        assert!(after.0 >= before.0 + 4 && after.1 <= before.1 - 4);
        for point in p["semanticMasks"]["faceOutline"]["points"]
            .as_array_mut()
            .unwrap()
        {
            let x = point["x"].as_f64().unwrap();
            let y = point["y"].as_f64().unwrap();
            point["x"] = json!(0.5 + (x - 0.5) * 0.65);
            point["y"] = json!(0.5 + (y - 0.5) * 0.65);
        }
        assert!(adjust(&source, &p)
            .unwrap_err()
            .to_string()
            .contains("does not follow"));
        p["semanticMasks"]
            .as_object_mut()
            .unwrap()
            .remove("faceOutline");
        assert!(adjust(&source, &p)
            .unwrap_err()
            .to_string()
            .contains("matching faceOutline"));
    }
    #[test]
    fn gradual_probability_edge_is_a_valid_half_alpha_isoline() {
        let (source, p) = portrait(160, 240);
        let polygon = points(&p["semanticMasks"]["face"], 160.0, 240.0);
        let data: Vec<u8> = (0..160 * 240)
            .map(|i| {
                let p = ((i % 160) as f32 + 0.5, (i / 160) as f32 + 0.5);
                let distance = polygon
                    .iter()
                    .enumerate()
                    .map(|(j, &a)| closest(p, a, polygon[(j + 1) % polygon.len()]).0)
                    .fold(f32::MAX, f32::min);
                let signed = if alpha(p, &polygon) >= 0.5 {
                    distance
                } else {
                    -distance
                };
                byte(255.0 / (1.0 + (-signed / 8.0).exp()))
            })
            .collect();
        let matte = json!({"shape":"raster","width":160,"height":240,"data":data});
        // This edge cannot reach .25/.75 inside three pixels. Its actual .5
        // level set is still the supplied contour, as in BiSeNet softmax output.
        validate_isoline(&matte, &polygon, &source, "faceOutline").unwrap();
        let unrelated: Vec<Point> = polygon
            .iter()
            .map(|p| (80.0 + (p.0 - 80.0) * 0.5, 120.0 + (p.1 - 120.0) * 0.5))
            .collect();
        assert!(validate_isoline(&matte, &unrelated, &source, "faceOutline").is_err());
    }
    #[test]
    fn inner_face_boundary_moves_against_hair_without_moving_outer_hairstyle() {
        let (mut source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        p["landmarks"] = p["semanticMasks"]["faceOutline"]["points"].clone();
        p["semanticMasks"]["faceContour"] = p["semanticMasks"]["faceOutline"].clone();
        p["semanticMasks"]["faceContourMask"] = p["semanticMasks"]["face"].clone();
        let head: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (80.0 + 66.0 * a.cos(), 120.0 + 108.0 * a.sin())
            })
            .collect();
        let head_data: Vec<u8> = (0..160 * 240)
            .map(|i| {
                let a = alpha(((i % 160) as f32 + 0.5, (i / 160) as f32 + 0.5), &head);
                if a > 0.5 && source.pixels[i * 4] < 100 {
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[12, 9, 6, 255]);
                }
                byte(a * 255.0)
            })
            .collect();
        p["semanticMasks"]["face"] =
            json!({"shape":"raster","width":160,"height":240,"data":head_data});
        p["semanticMasks"]["faceOutline"] = json!({"shape":"polygon","points":head.iter().map(|p|json!({"x":p.0/160.0,"y":p.1/240.0})).collect::<Vec<_>>()});
        let before = span(&source, 120);
        let output = adjust(&source, &p).unwrap();
        let after = span(&output, 120);
        assert!(
            after.0 >= before.0 + 4 && after.1 <= before.1 - 4,
            "skin edge {before:?} -> {after:?}"
        );
        for x in 10..24 {
            let i = (120 * 160 + x) * 4;
            assert_eq!(
                &output.pixels[i..i + 4],
                &source.pixels[i..i + 4],
                "outer hair/background at {x}"
            );
        }
        for i in 0..160 * 240 {
            if p["semanticMasks"]["faceContourMask"]["data"][i]
                .as_u64()
                .unwrap()
                == 0
            {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "contraction changed original exterior pixel {i}"
                );
            }
        }
        for x in before.0..after.0 - 1 {
            let i = (120 * 160 + x) * 4;
            assert!(
                output.pixels[i] < 30 && output.pixels[i + 2] < 30,
                "skin should reveal real neighboring hair: {:?}",
                &output.pixels[i..i + 4]
            );
        }
    }

    #[test]
    fn jaw_reveals_neck_below_the_face_and_keeps_backdrop_donors_separate() {
        let (mut source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        let mut neck = vec![0_u8; 160 * 240];
        let mut person: Vec<u8> = p["semanticMasks"]["face"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        for y in 150..240 {
            for x in 54..106 {
                let i = y * 160 + x;
                person[i] = 255;
                if person[i] > 0 && source.pixels[i * 4] < 100 {
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[95, 55, 30, 255]);
                    neck[i] = 255;
                }
            }
        }
        p["semanticMasks"]["neck"] = json!({"shape":"raster","width":160,"height":240,"data":neck});
        p["subjectMask"] = json!({"shape":"raster","width":160,"height":240,"data":person});
        p["adjustments"] = json!({"jaw":100});
        let output = adjust(&source, &p).unwrap();
        let before = span(&source, 190);
        let after = span(&output, 190);
        assert!(after.0 > before.0 && after.1 < before.1);
        let mut revealed_neck = 0;
        for x in before.0..after.0 {
            let i = (190 * 160 + x) * 4;
            if output.pixels[i] < 150 {
                assert!(
                    output.pixels[i] > 75 && output.pixels[i + 2] < 60,
                    "jaw must reveal neck, not blue scene backdrop: {:?}",
                    &output.pixels[i..i + 4]
                );
                revealed_neck += 1;
            }
        }
        assert!(revealed_neck > 0);
        p["adjustments"] = json!({"faceWidth":100});
        let output = adjust(&source, &p).unwrap();
        let before = span(&source, 120);
        let after = span(&output, 120);
        for x in before.0..after.0 - 1 {
            let i = (120 * 160 + x) * 4;
            assert!(
                output.pixels[i] < 70 && output.pixels[i + 2] > 100,
                "cheek must reveal background, not neck: {:?}",
                &output.pixels[i..i + 4]
            );
        }
        for y in 214..238 {
            let i = (y * 160 + 80) * 4;
            assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
        }
    }
    #[test]
    fn uncertain_face_neck_transition_does_not_freeze_the_old_chin_as_dark_dots() {
        let (mut source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        p["semanticMasks"]["faceContour"] = p["semanticMasks"]["faceOutline"].clone();
        p["semanticMasks"]["faceContourMask"] = p["semanticMasks"]["face"].clone();
        let face: Vec<u8> = p["semanticMasks"]["face"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let mut person = face.clone();
        let mut neck = vec![0_u8; 160 * 240];
        let mut uncertain_edge = Vec::new();
        for y in 180..240 {
            for x in 54..106 {
                let i = y * 160 + x;
                person[i] = 255;
                if face[i] == 0 {
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[95, 55, 30, 255]);
                    neck[i] = 255;
                } else if face[i] < 128 {
                    // Real semantic class transitions are not complementary
                    // physical alphas. Both face and neck confidence can be
                    // below .5 while the person mask remains confident.
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[160, 100, 65, 255]);
                    neck[i] = 90;
                    uncertain_edge.push(i);
                }
            }
        }
        assert!(!uncertain_edge.is_empty());
        p["subjectMask"] = json!({"shape":"raster","width":160,"height":240,"data":person});
        p["semanticMasks"]["neck"] = json!({"shape":"raster","width":160,"height":240,"data":neck});
        p["adjustments"] = json!({"chin":100});
        let output = adjust(&source, &p).unwrap();
        for i in uncertain_edge {
            assert!(
                output.pixels[i * 4] > 195,
                "old shadow at ({}, {}) stayed frozen: {:?}",
                i % 160,
                i / 160,
                &output.pixels[i * 4..i * 4 + 4]
            );
        }
    }
    #[test]
    fn uncovered_neck_uses_observed_shadow_detail_and_never_changes_existing_neck() {
        let (w, h) = (64_usize, 64_usize);
        let polygon: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (32.0 + 18.0 * a.cos(), 24.0 + 20.0 * a.sin())
            })
            .collect();
        let frame = Frame::new(&polygon, None, w as f32, h as f32);
        let target = target_contour(&polygon, frame, &json!({"chin":-100}));
        let field = ContourField {
            source: &polygon,
            target: &target,
            anchors: &[],
            support: 8.0,
            mls: mls::RigidMls::new(vec![]),
        };
        let pixels: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let shadow = 80 + ((i / w).saturating_sub(45) * 5).min(120) as u8;
                [shadow, shadow - 20, shadow - 35, 255]
            })
            .collect();
        let crop = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let known: Vec<bool> = (0..w * h)
            .map(|i| i / w >= 45 && (25..39).contains(&(i % w)))
            .collect();
        let holes: Vec<bool> = (0..w * h)
            .map(|i| (41..44).contains(&(i / w)) && (30..34).contains(&(i % w)))
            .collect();
        let mut background = crop.clone();
        for (i, &hole) in holes.iter().enumerate() {
            if hole {
                background.pixels[i * 4..i * 4 + 4].copy_from_slice(&[200, 180, 165, 255]);
            }
        }
        restore_exposed_neck_legacy(&mut background, &crop, &holes, &known, &field, (0.0, 0.0));
        for (i, &hole) in holes.iter().enumerate() {
            if hole {
                if i / w == 41 {
                    assert!(
                        background.pixels[i * 4] < 120,
                        "new neck lost the actual jaw shadow at {i}"
                    );
                } else {
                    assert!(
                        background.pixels[i * 4] < 200,
                        "new neck never used its corresponding observed donor at {i}"
                    );
                }
            } else {
                assert_eq!(
                    &background.pixels[i * 4..i * 4 + 4],
                    &crop.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }
    #[test]
    fn contracted_jaw_repositions_neck_illumination_without_warping_texture_or_scene() {
        let (w, h) = (64_usize, 64_usize);
        let polygon: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (32.0 + 18.0 * a.cos(), 24.0 + 20.0 * a.sin())
            })
            .collect();
        let frame = Frame::new(&polygon, None, w as f32, h as f32);
        let target = target_contour(&polygon, frame, &json!({"chin":-100}));
        let field = ContourField {
            source: &polygon,
            target: &target,
            anchors: &[],
            support: 8.0,
            mls: mls::RigidMls::new(vec![]),
        };
        let known: Vec<bool> = (0..w * h)
            .map(|i| i / w >= 45 && (20..44).contains(&(i % w)))
            .collect();
        let pixels: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let light = 80 + ((i / w).saturating_sub(45) * 5).min(120) as u8;
                let texture = if i % 2 == 0 { 16 } else { 0 };
                [light + texture, light - 20, light - 35, 255]
            })
            .collect();
        let crop = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let mut result = crop.clone();
        let holes = vec![false; w * h];
        let mut frozen = vec![false; w * h];
        frozen[48 * w + 32] = true;
        let changed = restore_neck_lighting(
            &mut result,
            &crop,
            &holes,
            &known,
            &frozen,
            &field,
            (0.0, 0.0),
        );
        let p = 48 * w + 31;
        assert!(changed[p]);
        assert!(
            result.pixels[p * 4] > crop.pixels[p * 4] + 5,
            "the old fixed shadow was not repositioned"
        );
        for i in 0..w * h {
            if !known[i] || frozen[i] {
                assert_eq!(
                    &result.pixels[i * 4..i * 4 + 4],
                    &crop.pixels[i * 4..i * 4 + 4],
                    "scene/protection changed at {i}"
                );
            }
        }
        let left = 47 * w + 30;
        let right = 47 * w + 31;
        let before = crop.pixels[left * 4] as i16 - crop.pixels[right * 4] as i16;
        let after = result.pixels[left * 4] as i16 - result.pixels[right * 4] as i16;
        assert!(
            (after - before).abs() <= 1,
            "neck texture moved or disappeared: {before} -> {after}"
        );
    }

    #[test]
    fn old_jaw_mixed_pixels_are_not_preserved_as_stationary_neck_texture() {
        let (w, h) = (64_usize, 64_usize);
        let polygon: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (32.0 + 18.0 * a.cos(), 24.0 + 20.0 * a.sin())
            })
            .collect();
        let frame = Frame::new(&polygon, None, w as f32, h as f32);
        let target = target_contour(&polygon, frame, &json!({"chin":-100}));
        let field = ContourField {
            source: &polygon,
            target: &target,
            anchors: &[],
            support: 8.0,
            mls: mls::RigidMls::new(vec![]),
        };
        let known: Vec<bool> = (0..w * h)
            .map(|i| i / w >= 44 && (20..44).contains(&(i % w)))
            .collect();
        let pixels: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let light = 80 + ((i / w).saturating_sub(44) * 5).min(120) as u8;
                [light, light - 20, light - 35, 255]
            })
            .collect();
        let mut crop = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let pixel = 44 * w + 32;
        crop.pixels[pixel * 4..pixel * 4 + 4].copy_from_slice(&[190, 170, 155, 255]);
        let mut result = crop.clone();
        let holes = vec![false; w * h];
        let frozen = vec![false; w * h];
        let changed = restore_neck_lighting(
            &mut result,
            &crop,
            &holes,
            &known,
            &frozen,
            &field,
            (0.0, 0.0),
        );
        assert!(changed[pixel]);
        assert!(
            result.pixels[pixel * 4] < 110,
            "old mixed skin fragment survived: {:?}",
            &result.pixels[pixel * 4..pixel * 4 + 4]
        );
    }

    #[test]
    fn rgb_edge_offset_from_the_class_contour_is_demixed_inside_the_neck_layer() {
        let (w, h) = (64_usize, 64_usize);
        let polygon: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (32.0 + 18.0 * a.cos(), 24.0 + 20.0 * a.sin())
            })
            .collect();
        let frame = Frame::new(&polygon, None, w as f32, h as f32);
        let target = target_contour(&polygon, frame, &json!({"chin":-100}));
        let field = ContourField {
            source: &polygon,
            target: &target,
            anchors: &[],
            support: 8.0,
            mls: mls::RigidMls::new(vec![]),
        };
        let known: Vec<bool> = (0..w * h)
            .map(|i| i / w >= 44 && (20..44).contains(&(i % w)))
            .collect();
        let mut crop = RgbaImage::new(
            w as u32,
            h as u32,
            vec![[80, 60, 45, 255]; w * h]
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();
        let offset_pixel = 47 * w + 32;
        crop.pixels[offset_pixel * 4..offset_pixel * 4 + 4].copy_from_slice(&[190, 170, 155, 255]);
        let mut result = crop.clone();
        let changed = restore_neck_lighting(
            &mut result,
            &crop,
            &vec![false; w * h],
            &known,
            &vec![false; w * h],
            &field,
            (0.0, 0.0),
        );
        assert!(changed[offset_pixel]);
        assert!(
            result.pixels[offset_pixel * 4] < 150,
            "offset mixed edge was kept as static texture: {:?}",
            &result.pixels[offset_pixel * 4..offset_pixel * 4 + 4]
        );
        for (i, &known_pixel) in known.iter().enumerate() {
            if !known_pixel {
                assert_eq!(
                    &result.pixels[i * 4..i * 4 + 4],
                    &crop.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }

    #[test]
    fn old_neck_join_is_continuous_and_keeps_protection_and_distant_texture() {
        let (w, h) = (96_usize, 96_usize);
        let polygon: Vec<Point> = (0..96)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 96.0;
                (48.0 + 27.0 * a.cos(), 36.0 + 30.0 * a.sin())
            })
            .collect();
        let frame = Frame::new(&polygon, None, w as f32, h as f32);
        let target = target_contour(&polygon, frame, &json!({"chin":-100}));
        let field = ContourField {
            source: &polygon,
            target: &target,
            anchors: &[],
            support: 12.0,
            mls: mls::RigidMls::new(vec![]),
        };
        let known: Vec<bool> = (0..w * h)
            .map(|i| i / w >= 66 && (30..66).contains(&(i % w)))
            .collect();
        let pixels: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let light = 80 + ((i / w).saturating_sub(66) * 4).min(120) as u8;
                [light, light - 20, light - 35, 255]
            })
            .collect();
        let crop = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
        let mut result = crop.clone();
        let mut frozen = vec![false; w * h];
        frozen[67 * w + 47] = true;
        let changed = restore_neck_lighting(
            &mut result,
            &crop,
            &vec![false; w * h],
            &known,
            &frozen,
            &field,
            (0.0, 0.0),
        );
        assert!(changed[66 * w + 48]);
        let profile: Vec<_> = (66..74)
            .map(|y| result.pixels[(y * w + 48) * 4] as i16)
            .collect();
        assert!(
            profile
                .windows(2)
                .all(|pair| pair[1] - pair[0] >= 0 && pair[1] - pair[0] <= 7),
            "old neck join introduced an impulse: {profile:?}"
        );
        for i in 0..w * h {
            if !known[i] || frozen[i] || i / w >= 88 {
                assert_eq!(
                    &result.pixels[i * 4..i * 4 + 4],
                    &crop.pixels[i * 4..i * 4 + 4],
                    "join correction escaped neck/protection/falloff at {i}"
                );
            }
        }
    }

    #[test]
    fn expanding_chin_covers_a_missed_two_pixel_fringe_without_unfreezing_other_people() {
        let (mut source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        let polygon = points(&p["semanticMasks"]["faceOutline"], 160.0, 240.0);
        p["semanticMasks"]["faceContour"] = p["semanticMasks"]["faceOutline"].clone();
        p["semanticMasks"]["faceContourMask"] = p["semanticMasks"]["face"].clone();
        let mut person: Vec<u8> = p["semanticMasks"]["face"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let mut neck = vec![0_u8; 160 * 240];
        let mut missed = Vec::new();
        for y in 210..240 {
            for x in 54..106 {
                let i = y * 160 + x;
                person[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[95, 55, 30, 255]);
                let distance = polygon
                    .iter()
                    .enumerate()
                    .map(|(j, &a)| {
                        closest(
                            (x as f32 + 0.5, y as f32 + 0.5),
                            a,
                            polygon[(j + 1) % polygon.len()],
                        )
                        .0
                    })
                    .fold(f32::MAX, f32::min);
                if distance <= 2.5 && (70..90).contains(&x) {
                    missed.push(i);
                } else {
                    neck[i] = 255;
                }
            }
        }
        // A genuinely separate confidently segmented person stays fixed.
        for y in 100..124 {
            for x in 10..28 {
                let i = y * 160 + x;
                person[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[180, 125, 65, 255]);
            }
        }
        assert!(!missed.is_empty());
        p["subjectMask"] = json!({"shape":"raster","width":160,"height":240,"data":person});
        p["semanticMasks"]["neck"] = json!({"shape":"raster","width":160,"height":240,"data":neck});
        p["adjustments"] = json!({"chin":100});
        let output = adjust(&source, &p).unwrap();
        for i in missed {
            assert!(output.pixels[i * 4] > 195, "missed old fringe left at {i}");
        }
        for y in 100..124 {
            for x in 10..28 {
                let i = (y * 160 + x) * 4;
                assert_eq!(&output.pixels[i..i + 4], &source.pixels[i..i + 4]);
            }
        }
    }
    #[test]
    fn full_person_matte_excludes_ears_neck_and_clothing_from_background_donors() {
        let (mut source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        let mut person = p["semanticMasks"]["face"]["data"]
            .as_array()
            .unwrap()
            .clone();
        for y in 98..142 {
            for x in 34..43 {
                let i = y * 160 + x;
                if source.pixels[i * 4] < 100 {
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[5, 220, 25, 255]);
                    person[i] = json!(255);
                }
            }
        }
        p["subjectMask"] = json!({"shape":"raster","width":160,"height":240,"data":person});
        let out = adjust(&source, &p).unwrap();
        for x in 43..46 {
            let i = (120 * 160 + x) * 4;
            assert!(
                out.pixels[i] < 100 && out.pixels[i + 1] < 100,
                "person color leaked into the revealed backdrop at {x}: {:?}",
                &out.pixels[i..i + 4]
            );
        }
        for y in 98..142 {
            for x in 34..43 {
                let i = (y * 160 + x) * 4;
                if source.pixels[i + 1] == 220 {
                    assert_eq!(&out.pixels[i..i + 4], &source.pixels[i..i + 4]);
                }
            }
        }
    }
    #[test]
    fn every_accessory_and_ear_pixel_stays_frozen_during_face_reshaping() {
        let (mut source, p) = portrait(160, 240);
        let mut p = raster_params(&source, p);
        let mut accessories = vec![0_u8; 160 * 240];
        let mut ears = accessories.clone();
        for y in 95..165 {
            for x in 43..47 {
                let i = y * 160 + x;
                accessories[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    245,
                    (y * 3 % 255) as u8,
                    8,
                    255,
                ]);
            }
            for x in 112..118 {
                let i = y * 160 + x;
                ears[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    230,
                    140,
                    (y * 7 % 255) as u8,
                    255,
                ]);
            }
        }
        p["semanticMasks"]["accessories"] =
            json!({"shape":"raster","width":160,"height":240,"data":accessories});
        p["semanticMasks"]["ears"] = json!({"shape":"raster","width":160,"height":240,"data":ears});
        for adjustments in [
            json!({"faceWidth":100}),
            json!({"faceWidth":-100}),
            json!({"jaw":100,"chin":100}),
        ] {
            p["adjustments"] = adjustments;
            let output = adjust(&source, &p).unwrap();
            for i in 0..160 * 240 {
                if accessories[i] > 0 || ears[i] > 0 {
                    assert_eq!(
                        &output.pixels[i * 4..i * 4 + 4],
                        &source.pixels[i * 4..i * 4 + 4],
                        "protected accessory/ear {i}"
                    );
                }
            }
        }
    }
    #[test]
    #[ignore = "manual native kernel benchmark uses local cached photo graphs"]
    fn native_cached_face_kernel_benchmark() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test/results/face-core-mls-matting");
        let name = std::env::var("FACE_BENCH_CASE").unwrap_or_else(|_| "portrait-makeup".into());
        let decoded = image::open(root.join(format!("{name}-source.png")))
            .unwrap()
            .to_rgba8();
        let source = RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw()).unwrap();
        let graph: Value = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("{name}-faceWidth-60.json"))).unwrap(),
        )
        .unwrap();
        let params = &graph["operations"][0]["params"];
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let out = adjust(&source, params).unwrap();
            assert_ne!(source.pixels, out.pixels);
            eprintln!(
                "{name} {}x{} kernel {:.1} ms",
                source.width,
                source.height,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
