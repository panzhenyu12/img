//! Face contour editing separates the detected face foreground from its
//! backdrop. The original oval is a source alpha, never an output clipping
//! mask. Joint face/neck edits reconstruct a stable source underlayer and
//! transport its paired premultiplied foreground; source colors stay consistent.
//! Pixel parsing supplies both the head alpha and the skin/hair face boundary.
//! A manual or legacy mesh contour remains a geometric approximation.

use crate::engine::{byte, percent};
use crate::mask::{coverage, PreparedMask, Region};
use crate::{ImageError, RgbaImage};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::VecDeque;

/// Automatic material classification is distinct from explicit user freezing.
/// Only the native ear experiment changes the observed ear motion policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObservedEarMotion {
    Fixed,
    JointFlexible,
}

impl ObservedEarMotion {
    fn follows_joint_field(self) -> bool {
        match self {
            Self::Fixed => false,
            Self::JointFlexible => true,
        }
    }
}

#[path = "face_flow.rs"]
mod flow;
#[path = "face_mls.rs"]
mod mls;

#[cfg(test)]
#[path = "face_profile_replay.rs"]
mod profile_replay;

#[cfg(test)]
#[path = "face_fringe_regression.rs"]
mod fringe_regression;

#[cfg(test)]
#[path = "face_edge_continuity_regression.rs"]
mod edge_continuity_regression;

#[path = "face_scene_edge.rs"]
mod scene_edge;

#[path = "face_layer_pair.rs"]
pub(crate) mod layer_pair;

#[path = "face_ear_support.rs"]
mod ear_support;

#[path = "face_source_plate.rs"]
mod source_plate;

#[path = "face_source_layers.rs"]
mod source_layers;
pub use source_layers::{ObservationRegion, SourceLayers};

#[path = "face_source_cache.rs"]
mod source_cache;
pub use source_cache::{FaceSourceCache, FaceSourceCacheStats};

#[path = "face_optical_domain.rs"]
mod optical_domain;

#[cfg(test)]
#[path = "face_source_pair_regression.rs"]
mod source_pair_regression;

#[cfg(test)]
#[path = "face_material_regression.rs"]
mod material_regression;

#[cfg(test)]
#[path = "face_joint_support_regression.rs"]
mod joint_support_regression;

#[cfg(test)]
#[path = "face_background_regression.rs"]
mod background_regression;

#[cfg(test)]
#[path = "face_occlusion_regression.rs"]
mod occlusion_regression;

#[cfg(test)]
#[path = "face_curve_candidate.rs"]
mod curve_candidate;

#[cfg(test)]
#[path = "face_displacement_curve.rs"]
mod displacement_curve;

#[cfg(test)]
#[path = "face_field_regression.rs"]
mod field_regression;

#[cfg(test)]
#[path = "face_anchor_transition.rs"]
mod anchor_transition;

#[path = "face_positive_kernel.rs"]
mod positive_kernel;

#[path = "face_joint_domain.rs"]
mod joint_domain;

#[path = "face_hairline.rs"]
mod hairline;

#[path = "face_scalar_field.rs"]
mod scalar_field;

#[cfg(test)]
#[path = "face_hairline_regression.rs"]
mod hairline_regression;

#[path = "face_upper_width.rs"]
mod upper_width;

#[cfg(test)]
#[path = "face_scene_role_regression.rs"]
mod scene_role_regression;

type Point = (f32, f32);
// Fairing and velocity construction must use one fixed-DOF contract.
const MIN_CONTROL_GAIN: f32 = 0.05;
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

/// The adaptive contour profile is safe to select only when the browser has
/// supplied an observed contour and independent eye/lip landmarks. Legacy
/// hand-drawn or synthetic face polygons do not contain enough anatomy to
/// infer local radii, so they retain the conservative axial profile.
fn adaptive_face_evidence(semantics: Option<&Value>) -> bool {
    let Some(masks) = semantics.and_then(Value::as_object) else {
        return false;
    };
    let polygon_with_points = |name: &str, minimum_points: usize| {
        masks
            .get(name)
            .filter(|mask| mask.get("shape").and_then(Value::as_str) == Some("polygon"))
            .and_then(|mask| mask.get("points"))
            .and_then(Value::as_array)
            .is_some_and(|points| points.len() >= minimum_points)
    };
    // A short hand drawn polygon does not contain enough curvature samples to
    // estimate section radii; keep the legacy path for those explicit inputs.
    polygon_with_points("faceContour", 24)
        && polygon_with_points("leftEye", 3)
        && polygon_with_points("rightEye", 3)
        && polygon_with_points("lips", 3)
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

fn add_fixed_boundary_control(
    p: Point,
    pitch: f32,
    boundary_controls: &mut Vec<Point>,
    control_positions: &mut Vec<Point>,
    requested_positions: &mut Vec<Point>,
    feasible_positions: &mut Vec<Point>,
    velocity_values: &mut Vec<Point>,
    movable: &mut Vec<bool>,
) {
    if boundary_controls
        .iter()
        .any(|&q| dot(sub(p, q), sub(p, q)) < pitch * pitch)
    {
        return;
    }
    if let Some(j) = control_positions.iter().position(|&q| q == p) {
        feasible_positions[j] = p;
        velocity_values[j] = (0.0, 0.0);
        movable[j] = false;
        return;
    }
    // A zero-velocity free handle is not a fixed material constraint.
    boundary_controls.push(p);
    control_positions.push(p);
    requested_positions.push(p);
    feasible_positions.push(p);
    velocity_values.push((0.0, 0.0));
    movable.push(false);
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
fn signed_distance(p: Point, polygon: &[Point]) -> f32 {
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
    if inside {
        distance
    } else {
        -distance
    }
}

fn alpha_with_feather(p: Point, polygon: &[Point], feather: f32) -> f32 {
    if !feather.is_finite() || feather <= 0.0 {
        return if inside(p, polygon) { 1.0 } else { 0.0 };
    }
    smoothstep(-feather, feather, signed_distance(p, polygon))
}

fn alpha(p: Point, polygon: &[Point]) -> f32 {
    alpha_with_feather(p, polygon, 0.5)
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

fn vertical_contour_offset(frame: Frame, p: Point, values: &Value) -> f32 {
    let y = (frame.coordinates(p).1 - frame.top) / frame.height().max(1.0);
    frame.height()
        * (percent(values, "chin") * 0.075 * smoothstep(0.74, 1.0, y)
            - percent(values, "forehead") * 0.060 * (1.0 - smoothstep(0.0, 0.30, y)))
}

/// Bound the accumulated anatomical lateral request without introducing the
/// derivative jump of `clamp`. Multiple controls can legitimately overlap
/// (for example faceWidth + jaw + cheekbones); a hard cap at that join creates
/// a visible curvature kink which a later transport field cannot remove.
/// `tanh` keeps the small-request slope unchanged while approaching the same
/// safety bound smoothly for extreme combinations.
fn smooth_lateral_limit(value: f32, limit: f32) -> f32 {
    if !value.is_finite() || !limit.is_finite() || limit <= 0.0 {
        return 0.0;
    }
    limit * libm::tanhf(value / limit)
}

/// Return the scalar motion-domain taper used around fixed attachments.
///
/// The source-paired compositor has an explicit material boundary (hair/ear,
/// accessory or user protection).  Its contour must enter that boundary with
/// a shallow tangent; a short taper spends the requested displacement in one
/// or two raster rows and reads as a platform.  Keep the legacy envelope for
/// the legacy compositor, and widen only the source-paired path.  The full
/// cheek remains at unit gain beyond this band, so the requested face-width
/// response is preserved.
fn motion_domain_taper(frame_width: f32, continuous_source_domain: bool) -> f32 {
    if !frame_width.is_finite() || frame_width <= 0.0 {
        return if continuous_source_domain { 10.0 } else { 6.0 };
    }
    if continuous_source_domain {
        (frame_width * 0.16).clamp(10.0, 36.0)
    } else {
        (frame_width * 0.12).clamp(6.0, 24.0)
    }
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
            let du = -side * frame.width() * smooth_lateral_limit(inward, 0.11) * lateral;
            let dv = vertical_contour_offset(frame, p, values);
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

/// Blend the exact contour transport into the cached grid transport over a
/// finite band. A hard branch at the old contour cutoff makes neighbouring
/// pixels use unrelated inverse positions and is a direct source of one-pixel
/// cracks. Both inputs are displacement vectors in the same destination frame.
fn blend_grid_motion(exact: Point, grid: Point, distance: f32, step: f32) -> Point {
    let cutoff = 3.0 + step * 2.0_f32.sqrt();
    let transition = smoothstep((cutoff - step).max(0.0), cutoff + step, distance);
    add(scale(exact, 1.0 - transition), scale(grid, transition))
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
        }
        let exact = field.inverse_unanchored(p);
        movement = blend_grid_motion(exact.0, movement, exact.1, self.step);
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

/// Build an anatomical lower-neck motion envelope for the isolated candidate
/// profile.  The distance is measured through the observed neck component
/// from the face/neck seam, so a rotated or slanted neck is not cut by an
/// image-space y threshold.  Non-neck pixels retain full support; disconnected
/// neck islands receive no support and therefore cannot become guessed motion.
#[cfg(test)]
fn geodesic_neck_motion(
    neck: &[f32],
    face: &[f32],
    owned: &[bool],
    w: usize,
    h: usize,
) -> Option<Vec<f32>> {
    if neck.len() != w * h || face.len() != w * h || owned.len() != w * h {
        return None;
    }
    let is_neck = |i: usize| neck[i] >= 0.5 && face[i] <= 0.5;
    let mut distance = vec![u16::MAX; w * h];
    let mut queue = VecDeque::new();
    for i in 0..w * h {
        if !owned[i] || face[i] <= 0.5 {
            continue;
        }
        for n in neighbors(i, w, h) {
            if is_neck(n) && distance[n] == u16::MAX {
                distance[n] = 0;
                queue.push_back(n);
            }
        }
    }
    if queue.is_empty() {
        return None;
    }
    while let Some(i) = queue.pop_front() {
        let next = distance[i].saturating_add(1);
        for n in neighbors(i, w, h) {
            if is_neck(n) && next < distance[n] {
                distance[n] = next;
                queue.push_back(n);
            }
        }
    }
    let mut row_widths = Vec::new();
    let mut max_distance = 0_u16;
    for y in 0..h {
        let width = (0..w)
            .filter(|&x| is_neck(y * w + x) && distance[y * w + x] != u16::MAX)
            .count();
        if width > 0 {
            row_widths.push(width as f32);
        }
    }
    for &d in &distance {
        if d != u16::MAX {
            max_distance = max_distance.max(d);
        }
    }
    if row_widths.is_empty() || max_distance < 2 {
        return None;
    }
    row_widths.sort_by(f32::total_cmp);
    let width = row_widths[row_widths.len() / 2].max(2.0);
    let length = max_distance as f32 + 1.0;
    let start = (0.5 * width).max(1.0);
    let end = (2.5 * width).min(0.22 * length).max(start + 1.0);
    let mut motion = vec![1.0; w * h];
    for i in 0..w * h {
        if !is_neck(i) {
            continue;
        }
        if distance[i] == u16::MAX {
            motion[i] = 0.0;
            continue;
        }
        let t = ((distance[i] as f32 - start) / (end - start)).clamp(0.0, 1.0);
        // Quintic smoothstep has zero first derivative at both ends, avoiding
        // a visible velocity kink where the neck becomes stationary.
        let h5 = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        motion[i] = 1.0 - h5;
    }
    Some(motion)
}

/// Propagate only clean exterior donors into the newly exposed band. The
/// surviving original foreground and frozen pixels are never donors. This
/// deterministic local fill cannot reconstruct complex hidden scene content.
#[cfg(test)]
fn reconstruct(
    source: &RgbaImage,
    holes: &[bool],
    subject: &[f32],
    frozen: &[bool],
    donor_excluded: &[bool],
    known_layer: &[u8],
    estimated_background: Option<&RgbaImage>,
) -> Result<RgbaImage, ImageError> {
    reconstruct_with_scene(
        source,
        holes,
        subject,
        frozen,
        donor_excluded,
        known_layer,
        estimated_background,
        None,
    )
}

fn reconstruct_with_scene(
    source: &RgbaImage,
    holes: &[bool],
    subject: &[f32],
    frozen: &[bool],
    donor_excluded: &[bool],
    known_layer: &[u8],
    estimated_background: Option<&RgbaImage>,
    scene_background: Option<&[Option<[f32; 3]>]>,
) -> Result<RgbaImage, ImageError> {
    reconstruct_source_background(
        source,
        holes,
        subject,
        frozen,
        donor_excluded,
        known_layer,
        estimated_background,
        scene_background,
        false,
    )
}

fn reconstruct_source_background(
    source: &RgbaImage,
    holes: &[bool],
    subject: &[f32],
    frozen: &[bool],
    donor_excluded: &[bool],
    known_layer: &[u8],
    estimated_background: Option<&RgbaImage>,
    scene_background: Option<&[Option<[f32; 3]>]>,
    independent_scene_color: bool,
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
    // The lower jaw can uncover neck rather than the stationary backdrop.
    // Hair and scene are both observed parts of that same backdrop: separating
    // their solve domains invents a hard Voronoi seam across an unknown strip.
    // Keep hair's donor eligibility above, but only neck is a separate layer.
    let mut layer: Vec<u8> = known_layer.iter().map(|&k| u8::from(k == 1)).collect();
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
    // The source-mixture bounds below are straight-RGB bounds for an opaque
    // photograph. Do not apply them to premultiplied transparent layers.
    let opaque_layers = source.pixels.chunks_exact(4).all(|p| p[3] == 255)
        && estimated_background.is_some_and(|b| b.pixels.chunks_exact(4).all(|p| p[3] == 255));
    struct BackgroundEvidence {
        prior: [f32; 4],
        precision: f32,
        scene_prior: Option<[f32; 3]>,
        visible_min: [f32; 3],
        visible_max: [f32; 3],
        visible_center: [f32; 3],
        directions: f32,
        feasible_min: [f32; 3],
        feasible_max: [f32; 3],
    }
    let evidence: Vec<_> = hole_indices
        .iter()
        .map(|&i| {
            let scene_prior = scene_background.and_then(|scene| scene[i]);
            if estimated_background.is_none() && scene_prior.is_none() {
                return None;
            }
            let a = subject[i].clamp(0.0, 1.0);
            let rgba = &estimated_background.unwrap_or(source).pixels[i * 4..i * 4 + 4];
            let opacity = rgba[3] as f32 / 255.0;
            let mut e = BackgroundEvidence {
                prior: [
                    rgba[0] as f32 * opacity,
                    rgba[1] as f32 * opacity,
                    rgba[2] as f32 * opacity,
                    rgba[3] as f32,
                ],
                // A fully opaque transported foreground has no source-mixture
                // observation for B. Its F/B prior is therefore disabled;
                // only clean observed backdrop rays may guide the hidden B.
                // A scene model already supplies the separate unit-weight
                // term below. Its copied F/B estimate is not an independent
                // high-precision source-mixture observation of B.
                precision: if a >= 1.0
                    || estimated_background.is_none()
                    || (independent_scene_color && scene_prior.is_some())
                {
                    0.0
                } else {
                    4.0 * ((1.0 - a) / a.max(1.0 / 255.0)).powi(2)
                },
                scene_prior,
                visible_min: [255.0; 3],
                visible_max: [0.0; 3],
                visible_center: [0.0; 3],
                directions: 0.0,
                feasible_min: [0.0; 3],
                feasible_max: [255.0; 3],
            };
            if opaque_layers && a < 1.0 {
                for c in 0..3 {
                    // Necessary source-mixture bounds, with half a byte of source
                    // quantization uncertainty; no uncertain F color is fixed.
                    let observed = source.pixels[4 * i + c] as f32;
                    e.feasible_min[c] =
                        ((observed - 0.5 - 255.0 * a) / (1.0 - a)).clamp(0.0, 255.0);
                    e.feasible_max[c] = ((observed + 0.5) / (1.0 - a)).clamp(0.0, 255.0);
                }
                // Original visible colors only. Never recycle the BFS fill as
                // observation. One nearest donor in each direction limits the
                // influence of correlated pixels in a large constant patch.
                let mut ray_values = [[0.0_f32; 8]; 3];
                let mut ray_count = 0usize;
                for (dx, dy) in [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    for step in 1..=12 {
                        let x = (i % w) as i32 + dx * step;
                        let y = (i / w) as i32 + dy * step;
                        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                            break;
                        }
                        let j = y as usize * w + x as usize;
                        if target[j]
                            || protected[j]
                            || layer[j] != layer[i]
                            || source.pixels[j * 4 + 3] != 255
                        {
                            continue;
                        }
                        for c in 0..3 {
                            let v = source.pixels[j * 4 + c] as f32;
                            e.visible_min[c] = e.visible_min[c].min(v);
                            e.visible_max[c] = e.visible_max[c].max(v);
                            ray_values[c][ray_count] = v;
                        }
                        ray_count += 1;
                        e.directions += 1.0;
                        break;
                    }
                }
                // A channel-wise median is robust to one dark hair/shirt ray.
                // Keep the min/max envelope for feasibility diagnostics, but
                // use the robust center for the continuous local data term.
                for c in 0..3 {
                    let mut values = ray_values[c][..ray_count].to_vec();
                    values.sort_by(|a, b| a.total_cmp(b));
                    if let Some(&median) = values.get(ray_count / 2) {
                        e.visible_center[c] = median;
                    }
                }
            }
            Some(e)
        })
        .collect();
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
            for (slot, &i) in hole_indices.iter().enumerate() {
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
                // A partly visible backdrop is an observation, not a hole
                // with arbitrary color. Keep its F/B estimate as a screened
                // data term. The conditioning of C=aF+(1-a)B gives relative
                // background precision ((1-a)/a)^2: low-alpha edge pixels
                // strongly retain their local backdrop; opaque holes use only
                // the spatial continuation. This acts inside reconstruction,
                // without filtering the final face or changing its motion.
                let mut denominator = donor_count as f32;
                if let Some(e) = &evidence[slot] {
                    for c in 0..4 {
                        sum[c] += e.precision * e.prior[c];
                    }
                    denominator += e.precision;
                }
                let scene_prior = evidence[slot].as_ref().and_then(|e| e.scene_prior);
                if denominator <= 0.0 && scene_prior.is_none() {
                    continue;
                }
                for (c, (channel, total)) in colors[i].iter_mut().zip(sum).enumerate() {
                    let mut total = total;
                    let mut denominator = denominator;
                    if c < 3 {
                        if let Some(b) = scene_prior {
                            // A fitted backdrop is one local observation, not
                            // an exact replacement for this hidden texel. Put
                            // it in the same spatial solve as adjacent holes;
                            // its availability boundary must not draw an edge.
                            total += b[c];
                            denominator += 1.0;
                        }
                    }
                    // A validated scene RGB prior can be the only observation
                    // in a surrounded hole. It provides no alpha observation;
                    // retain that channel's initializer when it has no donors.
                    if denominator <= 0.0 {
                        continue;
                    }
                    let mut value = total / denominator;
                    if c < 3 {
                        if let Some(e) = &evidence[slot] {
                            if e.directions > 0.0 {
                                let observed =
                                    e.visible_center[c].clamp(e.visible_min[c], e.visible_max[c]);
                                // Continuous squared-distance penalty to the
                                // observed range. An intersection/no-intersection
                                // switch creates discontinuities as alpha varies.
                                value = (total + e.directions * observed)
                                    / (denominator + e.directions);
                            }
                            value = value.clamp(e.feasible_min[c], e.feasible_max[c]);
                        }
                    }
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
    // Excluded pixels can be temporary paths for discovering clean donors.
    // Their zero foreground alpha still means the underlayer is already
    // visible. Returning the temporary fill there polluted expansion edges.
    for (i, &a) in subject.iter().enumerate() {
        if a <= 1e-6 {
            result.pixels[i * 4..i * 4 + 4].copy_from_slice(&source.pixels[i * 4..i * 4 + 4]);
        }
    }
    Ok(result)
}

/// Repair the short source face strip exposed by a contraction with an
/// outward background ray. The full F/B plate is intentionally regularized
/// over the complete moving layer, but a destination hole needs the local
/// backdrop immediately beyond the observed source silhouette. A ray keeps
/// the local blur/lighting gradient and avoids the broad dark bands produced
/// when a scene prior wins a large harmonic solve.
fn repair_exposed_face_background(
    background: &mut RgbaImage,
    crop: &RgbaImage,
    holes: &[bool],
    face_membership: &[f32],
    color_alpha: &[f32],
    frozen: &[bool],
    donor_excluded: &[bool],
    known_layer: &[u8],
    polygon: &[Point],
    origin: Point,
) {
    let width = crop.width as usize;
    let height = crop.height as usize;
    let count = width * height;
    if polygon.len() < 3 || holes.len() != count {
        return;
    }
    let clean_background: Vec<f32> = (0..count)
        .map(|i| {
            if color_alpha[i] <= 0.04 && !frozen[i] && !donor_excluded[i] && known_layer[i] == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    if clean_background.iter().all(|&v| v <= 0.0) {
        return;
    }
    for i in 0..count {
        if !holes[i] || face_membership[i] <= 1e-6 {
            continue;
        }
        let p = (
            origin.0 + (i % width) as f32 + 0.5,
            origin.1 + (i / width) as f32 + 0.5,
        );
        if !inside(p, polygon) {
            continue;
        }
        let mut nearest = (f32::MAX, 0, 0.0);
        for j in 0..polygon.len() {
            let (distance, t) = closest(p, polygon[j], polygon[(j + 1) % polygon.len()]);
            if distance < nearest.0 {
                nearest = (distance, j, t);
            }
        }
        let (_, edge, t) = nearest;
        let next = (edge + 1) % polygon.len();
        let boundary = add(scale(polygon[edge], 1.0 - t), scale(polygon[next], t));
        let normal = neck_normal(polygon, edge, t, 6.0);
        if dot(normal, normal) <= 1e-6 {
            continue;
        }
        // Move just outside the old silhouette, then advance slowly with the
        // hole depth so a local background gradient is preserved.
        let search_start = 1.5 + nearest.0 * 0.15;
        let mut donor = None;
        for step in 0..=24 {
            let q = add(boundary, scale(normal, search_start + step as f32));
            let candidate = sample(
                crop,
                &clean_background,
                q.0 - origin.0 - 0.5,
                q.1 - origin.1 - 0.5,
            );
            if candidate.1 >= 0.65 && candidate.0[3] > 1e-6 {
                donor = Some(candidate);
                break;
            }
        }
        let Some(donor) = donor else { continue };
        let weight = (0.72 + 0.20 * face_membership[i]).clamp(0.0, 0.95);
        for c in 0..3 {
            let value = donor.0[c] / donor.0[3].max(1e-6);
            let current = background.pixels[i * 4 + c] as f32;
            background.pixels[i * 4 + c] = byte(current * (1.0 - weight) + value * weight);
        }
    }
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

/// A confidently parsed mixed source edge can have fully opaque lobes outside
/// its .5 contour. Inverse sampling moves those lobes below the requested jaw.
/// Use the requested contour as an opacity upper bound only on a contracting
/// lower jaw with nearby observed neck. The physical source colors, neck
/// illumination and all hair/scene/ear edges retain their existing paths.
fn clip_supported_jaw_opacity(
    sampled: &mut ([f32; 4], f32),
    p: Point,
    crop: &RgbaImage,
    neck_support: &[f32],
    field: &ContourField<'_>,
    frame: Frame,
    origin: Point,
) {
    if sampled.1 <= 1e-6 || frame.coordinates(p).1 <= frame.top + frame.height() * 0.76 {
        return;
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
    if distance >= 8.0 {
        return;
    }
    let next = (j + 1) % field.source.len();
    let old = add(
        scale(field.source[j], 1.0 - t),
        scale(field.source[next], t),
    );
    let new = add(
        scale(field.target[j], 1.0 - t),
        scale(field.target[next], t),
    );
    let normal = neck_normal(field.source, j, t, 6.0);
    if dot(sub(old, new), normal) <= 1.0 {
        return;
    }
    let q = add(old, scale(normal, 6.0));
    let (_, support) = sample(
        crop,
        neck_support,
        q.0 - origin.0 - 0.5,
        q.1 - origin.1 - 0.5,
    );
    if support < 0.999 {
        return;
    }
    let limited = sampled.1.min(alpha(p, field.target));
    let ratio = limited / sampled.1;
    sampled.0 = sampled.0.map(|c| c * ratio);
    sampled.1 = limited;
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

fn validate_motion_interface(
    source: &RgbaImage,
    params: &Value,
    interface: Option<&Value>,
) -> Result<(), ImageError> {
    let (w, h) = (source.width as f32, source.height as f32);
    let semantics = params.get("semanticMasks");
    if let Some(interface) = interface {
        let error = || invalid("invalid source motion interface; no optical-contour fallback");
        let matte = interface.get("mask").ok_or_else(error)?;
        let hair = semantics.and_then(|m| m.get("hair")).ok_or_else(error)?;
        let mw = matte
            .get("width")
            .and_then(Value::as_u64)
            .ok_or_else(error)?;
        let mh = matte
            .get("height")
            .and_then(Value::as_u64)
            .ok_or_else(error)?;
        let data = matte
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(error)?;
        if matte.get("shape").and_then(Value::as_str) != Some("raster")
            || mw == 0
            || mh == 0
            || mw.checked_mul(mh) != Some(data.len() as u64)
            || matte.get("width") != hair.get("width")
            || matte.get("height") != hair.get("height")
            || data.iter().any(|v| {
                !v.as_f64()
                    .is_some_and(|n| n.is_finite() && (0.0..=255.0).contains(&n))
            })
        {
            return Err(error());
        }
        let outline = interface.get("contour").ok_or_else(error)?;
        let vertices = outline
            .get("points")
            .and_then(Value::as_array)
            .ok_or_else(error)?;
        if outline.get("shape").and_then(Value::as_str) != Some("polygon")
            || vertices.len() < 3
            || vertices.iter().any(|v| {
                ["x", "y"].iter().any(|key| {
                    !v.get(*key)
                        .and_then(Value::as_f64)
                        .is_some_and(|n| n.is_finite() && (0.0..=1.0).contains(&n))
                })
            })
        {
            return Err(error());
        }
        validate_isoline(
            matte,
            &points(outline, w, h),
            source,
            "sourceMotionInterface",
        )?;
    }
    Ok(())
}

pub(crate) fn adjust(source: &RgbaImage, params: &Value) -> Result<RgbaImage, ImageError> {
    adjust_impl(source, params, None, None)
}

pub(crate) fn adjust_with_source_layers(
    source: &RgbaImage,
    params: &Value,
    layers: &SourceLayers<'_>,
) -> Result<RgbaImage, ImageError> {
    adjust_with_source_cache_impl(source, params, layers, None)
}

pub(crate) fn adjust_with_source_cache(
    source: &RgbaImage,
    params: &Value,
    layers: &SourceLayers<'_>,
    cache: &mut FaceSourceCache,
) -> Result<RgbaImage, ImageError> {
    adjust_with_source_cache_impl(source, params, layers, Some(cache))
}

fn adjust_with_source_cache_impl(
    source: &RgbaImage,
    params: &Value,
    layers: &SourceLayers<'_>,
    cache: Option<&mut FaceSourceCache>,
) -> Result<RgbaImage, ImageError> {
    layers.validate(source)?;
    validate_motion_interface(source, params, layers.motion_interface)?;
    if percent(params.get("adjustments").unwrap_or(params), "symmetry").abs() > 1e-6 {
        return Err(invalid(
            "source-layer contour editing requires symmetry as a separate source preparation step",
        ));
    }
    adjust_impl(source, params, Some(layers), cache)
}

fn adjust_impl(
    source: &RgbaImage,
    params: &Value,
    source_layers: Option<&SourceLayers<'_>>,
    mut source_cache: Option<&mut FaceSourceCache>,
) -> Result<RgbaImage, ImageError> {
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
    #[cfg(test)]
    if source_layers.is_none() && params.get("__sourceMotionInterface").is_some() {
        let mode = params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str);
        if !mode.is_some_and(|m| {
            [
                "shared-cheek-v1",
                "shared-cheek-curve-v4",
                "shared-cheek-ear-v1",
                "shared-cheek-ear-periodic-v1",
                "source-domain-continuous-v1",
            ]
            .contains(&m)
        }) {
            return Err(invalid(
                "invalid source motion interface; no optical-contour fallback",
            ));
        }
        validate_motion_interface(source, params, params.get("__sourceMotionInterface"))?;
    }
    // Upper-face width edits meet a special ownership case: the observed face
    // contour is the skin/hair interface, while the outer head and explicit
    // facial features must stay fixed. Promote only an evidence-backed,
    // connected hair band into the skin ownership and let the established
    // source-forward joint field transport it. This keeps the public path
    // continuous without inventing a hidden outer silhouette.
    if (source_layers.is_some() || params
        .get("experimentalFaceProfile")
        .and_then(Value::as_str)
        .map(|mode| {
            mode == "upper-shared-v1"
                || cfg!(test)
                    && ["shared-cheek-v1", "shared-cheek-curve-v4", "shared-cheek-ear-v1", "shared-cheek-ear-periodic-v1", "source-domain-continuous-v1"]
                        .contains(&mode)
        })
        .unwrap_or(true))
        // The shared hairline ownership is valid for a combined contour edit
        // too.  Restricting it to an upper-face-only request made
        // `faceWidth + foreheadWidth` and `faceWidth + temples` silently drop
        // the upper control: the lower contour field ran with the hair band
        // frozen as an obstacle.  The recursive guard below prevents a second
        // promotion, so all contour controls continue through one transport
        // field and one source/matte ownership decision.
        && percent(values, "symmetry").abs() <= 1e-6
        && params
            .get("__upperSharedCandidateApplied")
            .and_then(Value::as_bool)
            != Some(true)
        && params.get("__legacyUpperShared").and_then(Value::as_bool) != Some(true)
    {
        if let Some(prepared) = upper_width::promote_with_source(
            params,
            &polygon,
            frame,
            source.width,
            source.height,
            source_layers.is_some(),
            source_layers.and_then(|l| l.motion_interface),
        ) {
            // Keep the candidate entirely inside the established ownership,
            // matting and source-forward transport path. The only change is
            // the observed shared hair/skin band; no output pixels are painted
            // by the candidate itself.
            return adjust_impl(source, &prepared, source_layers, source_cache);
        }
    }
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
    let interior_hairline = true;
    #[cfg(test)]
    let interior_hairline =
        interior_hairline && params.get("__legacyHairline").and_then(Value::as_bool) != Some(true);
    if interior_hairline {
        if let Some(hairline) = hairline::prepare(source, &support_params, &polygon, frame) {
            // Compose the dedicated interior hairline map with the established
            // contour map. A tiny second slider must not switch the forehead
            // back to a different algorithm. Apply opacity/blend only once.
            let mut lower_params = support_params.clone().into_owned();
            if lower_params.get("adjustments").is_some() {
                lower_params["adjustments"]["forehead"] = json!(0);
            } else {
                lower_params["forehead"] = json!(0);
            }
            lower_params["opacity"] = json!(100);
            lower_params["blendMode"] = json!("normal");
            let lower = adjust_impl(
                source,
                &lower_params,
                source_layers,
                source_cache.as_deref_mut(),
            )?;
            let rendered = hairline.render(&lower);
            let mut composite_params = support_params.into_owned();
            if composite_params
                .get("blendMode")
                .and_then(Value::as_str)
                .unwrap_or("normal")
                == "normal"
            {
                composite_params
                    .as_object_mut()
                    .expect("validated params")
                    .remove("mask");
                composite_params
                    .as_object_mut()
                    .expect("validated params")
                    .remove("protectMask");
            }
            let mut output = crate::mask::composite(source, &rendered, &composite_params);
            for (i, original) in source.pixels.chunks_exact(4).enumerate() {
                if rendered.pixels[i * 4..i * 4 + 4] == *original {
                    output.pixels[i * 4..i * 4 + 4].copy_from_slice(original);
                }
            }
            return Ok(output);
        }
    }
    let legacy_target = target_contour(&polygon, frame, values);
    let target = if adaptive_face_evidence(semantics)
        && params.get("__legacyFaceTarget").and_then(Value::as_bool) != Some(true)
    {
        crate::face_profile::target_contour_adaptive(
            &polygon,
            (
                frame.origin,
                frame.across,
                frame.down,
                frame.left,
                frame.right,
                frame.top,
                frame.bottom,
            ),
            semantics,
            w,
            h,
            values,
        )
        .unwrap_or_else(|_| legacy_target.clone())
    } else {
        legacy_target
    };
    // Native research replay only. The parameter is deliberately absent from
    // the public protocol and this branch does not exist in production builds.
    #[cfg(test)]
    let target = if params
        .get("experimentalFaceProfile")
        .and_then(Value::as_str)
        == Some("segmented-lateral-v1")
        || params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str)
            == Some("adaptive-lateral-v2")
    {
        let frame_tuple = (
            frame.origin,
            frame.across,
            frame.down,
            frame.left,
            frame.right,
            frame.top,
            frame.bottom,
        );
        let mode = params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str);
        if mode == Some("adaptive-lateral-v2") {
            crate::face_profile::target_contour_adaptive(
                &polygon,
                frame_tuple,
                semantics,
                w,
                h,
                values,
            )
        } else {
            crate::face_profile::target_contour(&polygon, frame_tuple, semantics, w, h, values)
        }
        .map_err(invalid)?
    } else {
        target
    };
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
    let observed_raster = |value: &Value| {
        let width = value.get("width").and_then(Value::as_u64).unwrap_or(0);
        let height = value.get("height").and_then(Value::as_u64).unwrap_or(0);
        value.get("shape").and_then(Value::as_str) == Some("raster")
            && width > 0
            && height > 0
            && width.checked_mul(height).is_some_and(|count| {
                value
                    .get("data")
                    .and_then(Value::as_array)
                    .is_some_and(|data| data.len() as u64 == count)
            })
    };
    let region = Region::read(support_params.get("region"));
    let context_outline = head_outline.map(|m| points(m, w, h)).unwrap_or_default();
    let pad = field.support + frame.width() * 0.11 + 4.0;
    // A source decomposition cannot change its crop when the edit crosses a
    // pixel boundary. Reserve the legal contour displacement in the anatomical
    // frame once, independent of this target. Field trajectories still require
    // their own support/boundary verification; this is not a flow bound.
    let stable_source_layers = face_contour.is_some()
        && face_matte.is_some()
        && contour_matte.is_some()
        // The exact source-mixture interval below is for opaque photographs.
        // Keep transparent/RGBA callers on their established layer path before
        // changing the source crop, permissions or reconstruction domain.
        && source.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255)
        && semantics
            .and_then(|m| m.get("skin"))
            .is_some_and(observed_raster)
        && semantics
            .and_then(|m| m.get("neck"))
            .is_some_and(observed_raster);
    let max_du = frame.width() * 0.11;
    let max_dv = frame.height() * 0.135;
    let pad_x = if stable_source_layers {
        pad.max(max_du * frame.across.0.abs() + max_dv * frame.down.0.abs() + 4.0)
    } else {
        pad
    };
    let pad_y = if stable_source_layers {
        pad.max(max_du * frame.across.1.abs() + max_dv * frame.down.1.abs() + 4.0)
    } else {
        pad
    };
    let left = (polygon
        .iter()
        .chain(target.iter().filter(|_| !stable_source_layers))
        .chain(&context_outline)
        .map(|p| p.0)
        .fold(f32::MAX, f32::min)
        - pad_x)
        .floor()
        .max(0.0) as usize;
    let top = (polygon
        .iter()
        .chain(target.iter().filter(|_| !stable_source_layers))
        .chain(&context_outline)
        .map(|p| p.1)
        .fold(f32::MAX, f32::min)
        - pad_y)
        .floor()
        .max(0.0) as usize;
    let right = (polygon
        .iter()
        .chain(target.iter().filter(|_| !stable_source_layers))
        .chain(&context_outline)
        .map(|p| p.0)
        .fold(f32::MIN, f32::max)
        + pad_x)
        .ceil()
        .min(w) as usize;
    let bottom = (polygon
        .iter()
        .chain(target.iter().filter(|_| !stable_source_layers))
        .chain(&context_outline)
        .map(|p| p.1)
        .fold(f32::MIN, f32::max)
        + pad_y)
        .ceil()
        .min(h) as usize;
    let (cw, ch) = (right - left, bottom - top);
    let count = cw * ch;
    let source_matte = face_matte.map(PreparedMask::new);
    let contour_mask = contour_matte.map(PreparedMask::new);
    let person_mask = params.get("subjectMask").map(PreparedMask::new);
    let protect_mask = params.get("protectMask").map(PreparedMask::new);
    // Independent source material evidence is identity/ownership, not optical
    // alpha. It is consumed throughout the fringe and matte release paths.
    let garment_mask = params.get("__sourceGarment").map(PreparedMask::new);
    let neck_mask = params
        .get("__sourceNeckSupport")
        .or_else(|| semantics.and_then(|m| m.get("neck")))
        .map(PreparedMask::new);
    let hair_mask = semantics.and_then(|m| m.get("hair")).map(PreparedMask::new);
    // Research-only source depth evidence. Hair color/class alone cannot tell
    // a front strand from a stationary layer behind the cheek. Public calls
    // keep the established conservative foreground-hair contract.
    let rear_hair_mask: Option<PreparedMask<'_>> = None;
    #[cfg(test)]
    let rear_hair_mask = params
        .get("__rearHairMask")
        .map(PreparedMask::new)
        .or(rear_hair_mask);
    // Current parser projects can transport the connected face/upper-neck
    // foreground together. Older projects and manual tracings keep their
    // established face-only path when this observed raster evidence is absent.
    let joint_skin = params
        .get("__sourceSkinSupport")
        .or_else(|| semantics.and_then(|m| m.get("skin")))
        .filter(|m| observed_raster(m))
        .map(PreparedMask::new);
    let mut joint_neck = face_contour.is_some()
        && face_matte.is_some()
        && contour_matte.is_some()
        && joint_skin.is_some()
        && semantics
            .and_then(|m| m.get("neck"))
            .is_some_and(observed_raster);
    let ear_material = semantics.and_then(|m| m.get("ears"));
    let flexible_ears = source_layers.is_some();
    #[cfg(test)]
    let flexible_ears = flexible_ears
        || params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str)
            .is_some_and(|mode| {
                [
                    "shared-cheek-ear-v1",
                    "shared-cheek-ear-periodic-v1",
                    "source-domain-continuous-v1",
                ]
                .contains(&mode)
            });
    let observed_ear_motion = if flexible_ears
        && stable_source_layers
        && joint_neck
        && ear_material.is_some_and(observed_raster)
        && ear_material
            .and_then(|m| m.get("data"))
            .and_then(Value::as_array)
            .is_some_and(|data| data.iter().any(|v| v.as_f64().is_some_and(|v| v >= 127.5)))
    {
        ObservedEarMotion::JointFlexible
    } else {
        ObservedEarMotion::Fixed
    };
    let ear_mask = if observed_ear_motion.follows_joint_field() {
        ear_material.map(PreparedMask::new)
    } else {
        None
    };
    // A source attachment plan is permission, not a material alpha. It must
    // arrive before source connectivity is decided; releasing an accessory
    // only after tracing leaves an obsolete disconnected-ear lock behind it.
    let attachment_support = source_layers.and_then(|l| l.moving_attachment_support);
    #[cfg(test)]
    let attachment_support =
        attachment_support.or_else(|| params.get("__sourceMovingAttachmentSupport"));
    let moving_attachment_mask = if let Some(mask) = attachment_support {
        let valid_data = mask
            .get("data")
            .and_then(Value::as_array)
            .is_some_and(|data| {
                data.len() == source.width as usize * source.height as usize
                    && data.iter().all(|v| matches!(v.as_u64(), Some(0 | 255)))
            });
        if !observed_ear_motion.follows_joint_field()
            || mask.get("shape").and_then(Value::as_str) != Some("raster")
            || mask.get("width").and_then(Value::as_u64) != Some(source.width as u64)
            || mask.get("height").and_then(Value::as_u64) != Some(source.height as u64)
            || !valid_data
        {
            return Err(invalid("source moving attachment support requires a full-source binary raster and joint observed ear mode"));
        }
        Some(PreparedMask::new(mask))
    } else {
        None
    };
    let attachment_feature_masks: Vec<_> = if moving_attachment_mask.is_some() {
        [
            "leftEye",
            "rightEye",
            "leftBrow",
            "rightBrow",
            "nose",
            "lips",
            "teeth",
        ]
        .into_iter()
        .filter_map(|key| semantics.and_then(|m| m.get(key)))
        .map(PreparedMask::new)
        .collect()
    } else {
        Vec::new()
    };
    let rigid_masks: Vec<_> = ["accessories", "ears"]
        .into_iter()
        .filter(|key| *key != "ears" || !observed_ear_motion.follows_joint_field())
        .filter_map(|key| {
            semantics
                .and_then(|m| m.get(key))
                .map(|m| (key, PreparedMask::new(m)))
        })
        .collect();
    let output_mask = support_params.get("mask").map(PreparedMask::new);
    let fixed_masks: Vec<_> = ["accessories", "ears", "hair"]
        .into_iter()
        .filter(|key| *key != "ears" || !observed_ear_motion.follows_joint_field())
        .filter_map(|key| {
            semantics
                .and_then(|m| m.get(key))
                .map(|m| (key, PreparedMask::new(m)))
        })
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
    let mut inferred_person_fixed = Vec::with_capacity(count);
    let mut attachment_fixed = Vec::with_capacity(count);
    let mut moving_attachment = Vec::with_capacity(count);
    let mut protection = Vec::with_capacity(count);
    let mut donor_excluded = Vec::with_capacity(count);
    let mut known_layer = Vec::with_capacity(count);
    let mut neck_membership = Vec::with_capacity(count);
    let mut ear_membership = Vec::with_capacity(if observed_ear_motion.follows_joint_field() {
        count
    } else {
        0
    });
    let mut material_fixed = Vec::with_capacity(count);
    let mut neck_lighting_known = Vec::with_capacity(count);
    let mut legacy_fringe_permissions = Vec::with_capacity(count);
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
            let garment = garment_mask
                .as_ref()
                .is_some_and(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5);
            material_fixed.push(garment);
            donor_excluded.push(person_coverage > 1.0 / 255.0 || garment);
            let protected = protect_mask
                .as_ref()
                .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                .unwrap_or(0.0);
            let hair = hair_mask
                .as_ref()
                .is_some_and(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5);
            let rear_hair = hair
                && rear_hair_mask.as_ref().is_some_and(|m| {
                    m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5
                });
            let planned_attachment = moving_attachment_mask
                .as_ref()
                .is_some_and(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5);
            if planned_attachment
                && (protected > 0.0
                    || hair
                    || garment_mask.as_ref().is_some_and(|m| {
                        m.coverage(p.0 / w, p.1 / h, source.width, source.height) > 0.0
                    })
                    || attachment_feature_masks
                        .iter()
                        .any(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5))
            {
                return Err(invalid("source moving attachment support conflicts with protected material; recompute the source plan"));
            }
            moving_attachment.push(planned_attachment);
            if planned_attachment {
                *donor_excluded.last_mut().expect("current pixel") = true;
            }
            let rigid_attachment = rigid_masks.iter().any(|(name, m)| {
                (*name != "accessories" || !planned_attachment)
                    && m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5
            });
            let accessory = (hair && !rear_hair) || rigid_attachment;
            attachment_fixed.push(accessory);
            neck_membership.push(
                neck_mask
                    .as_ref()
                    .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                    .unwrap_or(0.0),
            );
            let ear_coverage = if observed_ear_motion.follows_joint_field() {
                let coverage = ear_mask
                    .as_ref()
                    .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                    .unwrap_or(0.0);
                ear_membership.push(coverage);
                coverage
            } else {
                0.0
            };
            // A movable ear remains foreground material, never a backdrop
            // donor. Automatic motion permission does not change donor identity.
            if observed_ear_motion.follows_joint_field() && ear_coverage > 1.0 / 255.0 {
                *donor_excluded.last_mut().expect("current pixel") = true;
            }
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
            let other_person = person_coverage >= 0.8 && !rear_hair
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
            legacy_fringe_permissions.push(covered_old_fringe);
            // The joint source decomposition must not depend on whether the
            // current destination happens to cover an uncertain source edge.
            let covered_old_fringe = covered_old_fringe && !(stable_source_layers && joint_neck);
            inferred_person_fixed.push(other_person && !covered_old_fringe);
            frozen.push(
                protected >= 1.0 - 1e-6
                    || accessory
                    || garment
                    || (other_person && !covered_old_fringe),
            );
            // Existing hair remains fixed in the output, but its observed
            // color is the visible underlying layer at a skin/hair boundary.
            // Refusing it as a donor leaves a skin-colored band after a cheek
            // contraction. User-frozen pixels, ears, adornments and another
            // person's pixels remain forbidden donors.
            // A connected moving ear is excluded via donor_excluded above.
            // This array also locks its hidden backdrop reconstruction; using
            // it for a moving ear would strand the old ear color in that plate.
            donor_frozen.push(
                protected >= 1.0 - 1e-6
                    || garment
                    || rigid_attachment
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
    // A finite observed seed from parser face and skin-group(1,10,14).
    // The isolated ear policy additionally admits observed connected ear
    // material into this trimap. Class confidence is not physical opacity;
    // the existing matting stage below must estimate that independently.
    let mut joint_alpha = face_membership.clone();
    let mut joint_owned = vec![false; count];
    let mut joint_ear_prior = if observed_ear_motion.follows_joint_field() {
        vec![0.0_f32; count]
    } else {
        Vec::new()
    };
    if joint_neck {
        let skin = joint_skin.as_ref().expect("raster evidence checked");
        for (i, alpha) in joint_alpha.iter_mut().enumerate() {
            let p = (
                left as f32 + (i % cw) as f32 + 0.5,
                top as f32 + (i / cw) as f32 + 0.5,
            );
            if observed_ear_motion.follows_joint_field() && !frozen[i] {
                joint_ear_prior[i] = ear_membership[i];
            }
            *alpha = if material_fixed[i] {
                0.0
            } else {
                (*alpha)
                    .max(skin.coverage(p.0 / w, p.1 / h, source.width, source.height))
                    .max(joint_ear_prior.get(i).copied().unwrap_or(0.0))
            };
        }
        let mut queue = VecDeque::new();
        for i in 0..count {
            let p = (
                left as f32 + (i % cw) as f32 + 0.5,
                top as f32 + (i / cw) as f32 + 0.5,
            );
            if !attachment_fixed[i]
                && !material_fixed[i]
                && protection[i] < 1.0 - 1e-6
                && face_membership[i] > 0.5
                && inside(p, &polygon)
            {
                joint_owned[i] = true;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for n in neighbors(i, cw, ch) {
                if !joint_owned[n]
                    && !attachment_fixed[n]
                    && !material_fixed[n]
                    && protection[n] < 1.0 - 1e-6
                    && (joint_alpha[n] > 0.5 || moving_attachment[n])
                {
                    joint_owned[n] = true;
                    queue.push_back(n);
                }
            }
        }
        if !joint_owned.iter().enumerate().any(|(i, &owned)| {
            let p = (
                left as f32 + (i % cw) as f32 + 0.5,
                top as f32 + (i / cw) as f32 + 0.5,
            );
            owned
                && face_membership[i] <= 0.5
                && skin.coverage(p.0 / w, p.1 / h, source.width, source.height) > 0.5
                && neck_mask.as_ref().unwrap().coverage(
                    p.0 / w,
                    p.1 / h,
                    source.width,
                    source.height,
                ) > 0.8
        }) {
            joint_neck = false;
        }
        if observed_ear_motion.follows_joint_field() && !joint_neck {
            return Err(invalid(
                "observed ear motion requires a connected face/neck material layer",
            ));
        }
        // Preserve only the physical trimap next to the observed connected
        // component. Disconnected skin islands cannot become moving people.
        if joint_neck {
            let raster = semantics.and_then(|m| m.get("skin")).unwrap();
            let mw = raster.get("width").and_then(Value::as_u64).unwrap() as f32;
            let mh = raster.get("height").and_then(Value::as_u64).unwrap() as f32;
            let fringe = (2.0 + 1.5 * (w / mw).max(h / mh).max(1.0)).ceil() as u16;
            let mut distance = vec![u16::MAX; count];
            let mut queue = VecDeque::new();
            for (i, &owned) in joint_owned.iter().enumerate() {
                if owned {
                    distance[i] = 0;
                    queue.push_back(i);
                }
            }
            while let Some(i) = queue.pop_front() {
                let d = distance[i].saturating_add(1);
                if d > fringe {
                    continue;
                }
                for n in neighbors(i, cw, ch) {
                    if d < distance[n] {
                        distance[n] = d;
                        queue.push_back(n);
                    }
                }
            }
            for i in 0..count {
                if !joint_owned[i] && (distance[i] > fringe || joint_alpha[i] > 0.5) {
                    joint_alpha[i] = 0.0;
                    if let Some(prior) = joint_ear_prior.get_mut(i) {
                        *prior = 0.0;
                    }
                }
            }
            if observed_ear_motion.follows_joint_field() {
                // Ear motion is a component permission, not an opacity
                // threshold. A core-free component fully inside the source
                // crop must not append disconnected-ear protection. A weak
                // isolated classification on cheek skin retains the existing
                // material policy instead of becoming a new fixed obstacle.
                // Crop-edge components stay conservative: a core may lie
                // outside this raster, so absence has not been established.
                // Source component identity is determined BEFORE protection
                // barriers so a protected core cannot release its soft tail.
                // This does not change the source alpha prior, donor policy,
                // or face/neck connectivity and inferred-fringe rules.
                let source_ear_support =
                    ear_support::SourceEarSupport::new(&ear_membership, cw, ch);
                let mut allowed = vec![false; count];
                for i in 0..count {
                    if ear_membership[i] <= 1.0 / 255.0 {
                        continue;
                    }
                    // Reuse established fixed material and explicit user
                    // protection as connection barriers. A low-confidence
                    // hair/accessory overlap is not new hard-freeze evidence;
                    // the existing material path still owns its soft edge.
                    allowed[i] = protection[i] <= 0.0
                        && !material_fixed[i]
                        && !attachment_fixed[i]
                        && !(frozen[i] && !inferred_person_fixed[i]);
                }
                let disconnected_ear = source_ear_support.disconnected(&allowed, &joint_owned);
                for i in 0..count {
                    if disconnected_ear[i] {
                        attachment_fixed[i] = true;
                        frozen[i] = true;
                        donor_frozen[i] = true;
                        donor_excluded[i] = true;
                        joint_owned[i] = false;
                        joint_alpha[i] = 0.0;
                        joint_ear_prior[i] = 0.0;
                    }
                }
            }
        }
    }
    // A missing/disconnected neck falls back to the established face-only
    // renderer, whose destination occlusion permissions are a separate path.
    if stable_source_layers && !joint_neck {
        for (i, &covered) in legacy_fringe_permissions.iter().enumerate() {
            if !covered {
                continue;
            }
            let p = ((left + i % cw) as f32 + 0.5, (top + i / cw) as f32 + 0.5);
            let rigid_attachment = rigid_masks.iter().any(|(name, m)| {
                (*name != "accessories" || !moving_attachment[i])
                    && m.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5
            });
            inferred_person_fixed[i] = false;
            frozen[i] = protection[i] >= 1.0 - 1e-6 || attachment_fixed[i] || material_fixed[i];
            donor_frozen[i] = protection[i] >= 1.0 - 1e-6 || rigid_attachment || material_fixed[i];
        }
    }
    let moving_alpha = if joint_neck {
        &joint_alpha
    } else if contour_mask.is_some() {
        &face_membership
    } else {
        &subject
    };
    // Polygon antialiasing is geometric coverage, just as parser confidence
    // is semantic coverage. Neither is physical opacity at a decoded RGB
    // boundary. Refine both before imposing any source-mixture constraint;
    // otherwise a fractional polygon texel on opaque skin strands old skin
    // in the reconstructed backdrop. Ambiguous colors/RGBA retain the input.
    let mut color_alpha = crate::face_matting::refine_portrait_alpha(&crop, moving_alpha);
    // An independently parsed head is stationary context support around the
    // facial class, including hair/ears. Its confidently empty exterior remains
    // fixed even when an ambiguous local color-line projection suggests skin.
    // A single face tracing reused as head has no such independent evidence.
    let independent_head = contour_matte.is_some() && face_matte != contour_matte;
    if independent_head {
        for (i, alpha) in color_alpha.iter_mut().enumerate() {
            if subject[i] <= 1e-6 && (!joint_neck || joint_alpha[i] <= 1e-6) {
                *alpha = 0.0;
            }
        }
    }
    if joint_neck {
        // A color-line fit can explain ordinary backdrop texture as tiny
        // foreground coverage. Source material needs observed head/skin or
        // selected-person support; the independently verified fringe recovery
        // below may still restore skin omitted by either semantic mask.
        for (i, alpha) in color_alpha.iter_mut().enumerate() {
            if subject[i] <= 1e-6 && joint_alpha[i] <= 1e-6 && !donor_excluded[i] {
                *alpha = 0.0;
            }
        }
    }
    if let Some(rear_hair) = &rear_hair_mask {
        for (i, a) in color_alpha.iter_mut().enumerate() {
            let p = ((left + i % cw) as f32 + 0.5, (top + i / cw) as f32 + 0.5);
            if moving_alpha[i] <= 1e-6
                && rear_hair.coverage(p.0 / w, p.1 / h, source.width, source.height) >= 0.5
            {
                *a = 0.0;
            }
        }
    }
    // Recover only physical alpha omitted by semantic zero at the connected
    // face/neck edge. This does not alter owned components or frozen geometry.
    if joint_neck && independent_head {
        let raster = semantics.and_then(|m| m.get("skin")).unwrap();
        let mw = raster.get("width").and_then(Value::as_u64).unwrap() as f32;
        let mh = raster.get("height").and_then(Value::as_u64).unwrap() as f32;
        let fringe = (2.0 + 1.5 * (w / mw).max(h / mh).max(1.0)).ceil() as usize;
        let mut person = Vec::with_capacity(count);
        let mut forbidden = Vec::with_capacity(count);
        let mut semantic_edge = Vec::with_capacity(count);
        let skin_mask = params
            .get("__sourceSkinSupport")
            .or_else(|| semantics.and_then(|m| m.get("skin")))
            .map(PreparedMask::new);
        for i in 0..count {
            let nx = (left as f32 + (i % cw) as f32 + 0.5) / w;
            let ny = (top as f32 + (i / cw) as f32 + 0.5) / h;
            person.push(
                person_mask
                    .as_ref()
                    .map(|m| m.coverage(nx, ny, source.width, source.height))
                    .unwrap_or(0.0),
            );
            let skin_value = skin_mask
                .as_ref()
                .map(|m| m.coverage(nx, ny, source.width, source.height))
                .unwrap_or(0.0);
            let neck_value = neck_mask
                .as_ref()
                .map(|m| m.coverage(nx, ny, source.width, source.height))
                .unwrap_or(0.0);
            semantic_edge.push(
                skin_value
                    .max(neck_value)
                    .max(joint_ear_prior.get(i).copied().unwrap_or(0.0)),
            );
            forbidden.push(
                frozen[i]
                    || inferred_person_fixed[i]
                    || protection[i] > 1e-6
                    || fixed_masks.iter().any(|(name, m)| {
                        (*name != "accessories" || !moving_attachment[i])
                            && m.coverage(nx, ny, source.width, source.height) > 1e-6
                    }),
            );
        }
        color_alpha = crate::face_matting::recover_connected_portrait_fringe(
            &crop,
            &color_alpha,
            &joint_owned,
            &person,
            &semantic_edge,
            &forbidden,
            fringe,
        );
        // Recheck inferred portrait-edge freezes against connected source
        // evidence. This changes neither the edit target nor the output by
        // painting; verified source material is transported by the same field.
        let strict_forbidden: Vec<_> = (0..count)
            .map(|i| {
                let nx = (left as f32 + (i % cw) as f32 + 0.5) / w;
                let ny = (top as f32 + (i / cw) as f32 + 0.5) / h;
                protection[i] > 1e-6
                    || material_fixed[i]
                    || (frozen[i] && !inferred_person_fixed[i])
                    || fixed_masks.iter().any(|(name, m)| {
                        (*name != "accessories" || !moving_attachment[i])
                            && m.coverage(nx, ny, source.width, source.height) > 1e-6
                    })
            })
            .collect();
        let recover_fringe = true;
        #[cfg(test)]
        let recover_fringe = recover_fringe
            && params
                .get("__legacyInferredFringe")
                .and_then(Value::as_bool)
                != Some(true);
        if recover_fringe {
            let recovered = crate::face_source_fringe::recover_inferred_fringe(
                &crop,
                &color_alpha,
                &joint_owned,
                &person,
                &semantic_edge,
                &strict_forbidden,
                &inferred_person_fixed,
                fringe,
            );
            for (i, &released) in recovered.released.iter().enumerate() {
                if released && !material_fixed[i] {
                    frozen[i] = false;
                }
            }
            color_alpha = recovered.alpha;
        }
    }
    if let Some(layers) = source_layers {
        if !stable_source_layers || !joint_neck {
            return Err(invalid(
                "face source layers require a connected face and neck material domain",
            ));
        }
        for i in 0..count {
            let x = left + i % cw;
            let y = top + i / cw;
            if !layers.contains(x, y) {
                continue;
            }
            color_alpha[i] = if material_fixed[i] {
                0.0
            } else {
                layers.optical_alpha[y * source.width as usize + x]
            };
            if color_alpha[i] > 1e-6
                && inferred_person_fixed[i]
                && !attachment_fixed[i]
                && !material_fixed[i]
                && protection[i] < 1.0 - 1e-6
            {
                frozen[i] = false;
            }
        }
    }
    #[cfg(test)]
    if let Some(observed) = params.get("__physicalAlpha").map(PreparedMask::new) {
        // Isolated source-matting experiment. No generated output pixels are
        // supplied: the same source RGB, contour request and compositor run.
        let observed_region = Region::read(
            params
                .get("__physicalAlpha")
                .and_then(|m| m.get("sourceRegion")),
        );
        let observed_selection = params
            .get("__physicalAlpha")
            .and_then(|m| m.get("sourceSelection"))
            .map(PreparedMask::new);
        let selected_source = |p: Point| {
            observed_region.contains(p.0 / w, p.1 / h)
                && observed_selection.as_ref().is_none_or(|mask| {
                    mask.coverage(p.0 / w, p.1 / h, source.width, source.height) > 0.5
                })
        };
        for i in 0..count {
            let p = ((left + i % cw) as f32 + 0.5, (top + i / cw) as f32 + 0.5);
            if !selected_source(p) {
                continue;
            }
            color_alpha[i] = if material_fixed[i] {
                0.0
            } else {
                observed.coverage(p.0 / w, p.1 / h, source.width, source.height)
            };
            if color_alpha[i] > 1e-6
                && inferred_person_fixed[i]
                && !attachment_fixed[i]
                && !material_fixed[i]
                && protection[i] < 1.0 - 1e-6
            {
                frozen[i] = false;
            }
        }
        if params.get("__refinePhysicalAlpha").and_then(Value::as_bool) == Some(true) {
            let refined = crate::face_matting::refine_portrait_alpha(&crop, &color_alpha);
            for i in 0..count {
                let p = ((left + i % cw) as f32 + 0.5, (top + i / cw) as f32 + 0.5);
                if selected_source(p) {
                    color_alpha[i] = if attachment_fixed[i] || material_fixed[i] {
                        0.0
                    } else {
                        refined[i]
                    };
                }
            }
        }
    }
    // The existing source physical matting trimap is the uncertain own-face
    // edge. PersonAlpha is the selected person's combined class confidence,
    // not an independently detected other-person instance. Keep its real
    // clothing support, but do not turn uncertainty at that validated facial
    // silhouette into an immovable geometric obstacle. Explicit/user masks
    // remain authoritative and donor exclusion does not change.
    let parser_edge_uncertainty = contour_matte
        .map(|m| {
            let mw = m.get("width").and_then(Value::as_u64).unwrap_or(1) as f32;
            let mh = m.get("height").and_then(Value::as_u64).unwrap_or(1) as f32;
            (w / mw).max(h / mh).max(1.0) * 1.5
        })
        .unwrap_or(0.0);
    let own_edge_uncertainty = 2.0 + parser_edge_uncertainty;
    for i in 0..count {
        if !joint_neck || !inferred_person_fixed[i] {
            continue;
        }
        let p = (
            left as f32 + (i % cw) as f32 + 0.5,
            top as f32 + (i / cw) as f32 + 0.5,
        );
        let contour_distance = polygon
            .iter()
            .enumerate()
            .map(|(k, &a)| closest(p, a, polygon[(k + 1) % polygon.len()]).0)
            .fold(f32::MAX, f32::min);
        // Existing source-alpha localization uncertainty from validate_isoline
        // plus the physical RGB trimap. This scales with parser raster pitch.
        let own_face_uncertain_edge = (independent_head
            && contour_distance <= own_edge_uncertainty)
            || (joint_neck && joint_owned[i] && joint_alpha[i] > 0.8);
        // Ear motion changes the observed ear material only. It must not
        // upgrade uncertain face/neck classifications to explicit locks.
        // Disconnected ears have already been restored to attachment_fixed.
        let explicit_fixed =
            protection[i] >= 1.0 - 1e-6 || attachment_fixed[i] || material_fixed[i];
        if own_face_uncertain_edge && !explicit_fixed {
            frozen[i] = false;
            // Inferred other-person texels remain forbidden as color donors.
        }
    }
    // A semantic garment identity is fixed material, never moving skin. Keep
    // it out of the physical alpha before F/B estimation and source pairing.
    for (i, alpha) in color_alpha.iter_mut().enumerate() {
        if material_fixed[i] {
            *alpha = 0.0;
        }
    }
    mark!("source material ownership");
    // Establish a stationary scene estimate from the unedited photograph.
    // The decoded edge can carry a skin/shadow contribution even where every
    // semantic mask has already reached zero. Such a contribution must travel
    // with the same continuous field as the face, rather than being erased by
    // a threshold on a destination sample or on the requested face polygon.
    let scene_edge = if joint_neck {
        let blocked: Vec<_> = (0..count)
            .map(|i| {
                let p = ((left + i % cw) as f32 + 0.5, (top + i / cw) as f32 + 0.5);
                frozen[i] || attachment_fixed[i] || protection[i] > 1e-6
                    // Rear hair may be occluded by new face coverage, but it
                    // remains a stationary layer, never a missing skin fringe.
                    || rear_hair_mask.as_ref().is_some_and(|m| {
                        m.coverage(p.0 / w, p.1 / h, source.width, source.height) > 1e-6
                    })
            })
            .collect();
        let support: Vec<_> = (0..count)
            .map(|i| {
                subject[i]
                    .max(joint_alpha[i])
                    .max(neck_membership[i])
                    .max(if donor_excluded[i] { 1.0 } else { 0.0 })
            })
            .collect();
        #[cfg(test)]
        if std::env::var_os("FACE_SOURCE_LAYERS_TRACE").is_some() {
            if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
                // Source-only replay evidence for scene-model rejection.
                // Keep the inputs before minimum-alpha updates; the later
                // F/B dump alone cannot reproduce this estimator's decision.
                let directory = std::path::Path::new(path);
                std::fs::create_dir_all(directory).map_err(|e| invalid(&e.to_string()))?;
                for (name, values) in [
                    ("source-scene-alpha.f32", &color_alpha),
                    ("source-scene-support.f32", &support),
                ] {
                    let bytes: Vec<_> = values.iter().flat_map(|a| a.to_le_bytes()).collect();
                    std::fs::write(directory.join(name), bytes)
                        .map_err(|e| invalid(&e.to_string()))?;
                }
                for (name, values) in [
                    ("source-scene-blocked.u8", &blocked),
                    ("source-scene-owned.u8", &joint_owned),
                ] {
                    let bytes: Vec<_> = values.iter().map(|&b| u8::from(b)).collect();
                    std::fs::write(directory.join(name), bytes)
                        .map_err(|e| invalid(&e.to_string()))?;
                }
                std::fs::write(
                    directory.join("source-scene-meta.json"),
                    serde_json::to_vec(&json!({
                        "width": cw, "height": ch,
                        "pixelIndexOrigin": [left, top],
                        "fringe": own_edge_uncertainty.ceil() as usize
                    }))
                    .unwrap(),
                )
                .map_err(|e| invalid(&e.to_string()))?;
            }
        }
        Some(scene_edge::prepare(
            &crop,
            &color_alpha,
            &joint_owned,
            &blocked,
            &support,
            own_edge_uncertainty.ceil() as usize,
        ))
    } else {
        None
    };
    if let Some(evidence) = &scene_edge {
        for (i, a) in color_alpha.iter_mut().enumerate() {
            *a = a.max(evidence.minimum_alpha[i]);
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
    // The model center need not be feasible at the conservative alpha bound.
    // Only source-paired opaque layers can project B and derive its matching F.
    // A legacy/manual joint path may lack stable source layers; keep its old
    // projection and prior precision until it supports that same pairing.
    let independent_scene_color = stable_source_layers
        && joint_neck
        && crop.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255);
    let reconstruction_scene = scene_edge.as_ref().map(|e| {
        if independent_scene_color {
            e.color_prior()
        } else {
            e.background.clone()
        }
    });
    mark!("source scene evidence");
    let mut color_layers = if stable_source_layers && joint_neck {
        optical_domain::estimate(&crop, &color_alpha)
    } else {
        crate::face_matting::estimate(&crop, &color_alpha)
    };
    #[cfg(test)]
    if std::env::var_os("FACE_SOURCE_LAYERS_TRACE").is_some() {
        if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
            // Freeze the actual estimator input before scene priors or source
            // pairing. A local neural-alpha file alone is not this full matte.
            let directory = std::path::Path::new(path);
            std::fs::create_dir_all(directory).map_err(|error| invalid(&error.to_string()))?;
            let alpha_bytes: Vec<_> = color_alpha.iter().flat_map(|a| a.to_le_bytes()).collect();
            std::fs::write(directory.join("source-color-alpha.f32"), alpha_bytes)
                .map_err(|error| invalid(&error.to_string()))?;
            for (name, image) in [
                ("source-estimator-input.png", Some(&crop)),
                (
                    "source-estimated-foreground.png",
                    color_layers.as_ref().map(|(f, _)| f),
                ),
                (
                    "source-estimated-background.png",
                    color_layers.as_ref().map(|(_, b)| b),
                ),
            ] {
                if let Some(image) = image {
                    image::save_buffer(
                        directory.join(name),
                        &image.pixels,
                        image.width,
                        image.height,
                        image::ColorType::Rgba8,
                    )
                    .map_err(|error| invalid(&error.to_string()))?;
                }
            }
        }
    }
    if let (Some(evidence), Some((foreground, background))) =
        (&reconstruction_scene, &mut color_layers)
    {
        for (i, prior) in evidence.iter().enumerate() {
            let Some(b) = prior else { continue };
            let a = color_alpha[i];
            for c in 0..3 {
                background.pixels[i * 4 + c] = byte(b[c]);
                if a > 1e-6 {
                    // This is only an initial color prior: its model center
                    // need not be feasible at the conservative alpha bound.
                    // Complete source pairing below jointly projects B and
                    // derives its matching F before opaque joint transport.
                    foreground.pixels[i * 4 + c] =
                        byte((crop.pixels[i * 4 + c] as f32 - (1.0 - a) * b[c]) / a);
                }
            }
        }
    }
    mark!("F/B estimate");
    // Reconstruct the complete SOURCE underlayer before sampling. Choosing the
    // source B from destination holes made F jump whenever a hole closed, even
    // though each pair separately reconstructed C. This plate never receives
    // target coverage or the inverse field as input.
    let mut paired_background = crop.clone();
    let paired_foreground = if stable_source_layers && joint_neck {
        let source_holes: Vec<_> = (0..count)
            .map(|i| color_alpha[i] > 1e-6 && !frozen[i])
            .collect();
        let input = source_cache::BackgroundInput {
            source: &crop,
            holes: &source_holes,
            alpha: &color_alpha,
            donor_frozen: &donor_frozen,
            donor_excluded: &donor_excluded,
            known_layer: &known_layer,
            estimated_background: color_layers.as_ref().map(|(_, b)| b),
            reconstruction_scene: reconstruction_scene.as_deref(),
            independent_scene_color,
        };
        let (background, foreground) = match source_cache.as_deref_mut() {
            Some(cache) => cache.reconstruct_and_pair(input)?,
            None => source_cache::reconstruct_and_pair(input)?,
        };
        paired_background = background;
        foreground
    } else {
        None
    };
    mark!("source background reconstruction and pair");
    #[cfg(test)]
    if let Some(path) = params.get("__sourcePlatePrior").and_then(Value::as_str) {
        if paired_foreground.is_none() {
            return Err(invalid(
                "source plate prior requires paired opaque source layers",
            ));
        }
        // Apply after physical pairing. Re-pairing afterwards would project
        // signed contrast back into the stationary plate and undo the edit.
        source_plate::apply_cached(
            source,
            &crop,
            &mut color_alpha,
            &mut paired_background,
            &frozen,
            (left, top),
            std::path::Path::new(path),
        )
        .map_err(|e| invalid(&e))?;
    }
    if let Some(layers) = source_layers {
        if paired_foreground.is_none() {
            return Err(invalid(
                "face source layers require complete face, skin and neck raster evidence",
            ));
        }
        layers.apply_scene(
            &crop,
            &mut color_alpha,
            &mut paired_background,
            &frozen,
            (left, top),
        )?;
    }
    mark!("source scene prior");
    let foreground = paired_foreground
        .as_ref()
        .or_else(|| color_layers.as_ref().map(|(f, _)| f))
        .unwrap_or(&crop);

    let neck_support: Vec<f32> = neck_lighting_known
        .iter()
        .map(|&v| if v { 1.0 } else { 0.0 })
        .collect();
    // Research candidate: use the same source-role partition for the motion
    // transition that is used by the source compositor. It is deliberately
    // opt-in; the default path keeps the accepted legacy falloffs until this
    // candidate passes real-photo image review.
    let source_domain_profile = source_layers.is_some()
        || params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str)
            .is_some_and(|profile| {
                matches!(profile, "source-domain-v1" | "source-domain-continuous-v1")
            });
    // The continuous candidate inherits the observed skin/adjacent-hair
    // material domain prepared above. Its flow extends beyond the old optical
    // coverage, while unshared hair, fixed attachments and user protection
    // remain continuous motion barriers. A stationary scene layer does not
    // turn the foreground's original silhouette into a zero-velocity wall.
    // The explicit source-layer API enables this path in ordinary builds;
    // legacy graph calls retain their prior path. Photo quality still requires
    // acceptance for the supplied source layers and requested operation.
    let continuous_source_domain = source_layers.is_some();
    #[cfg(test)]
    let continuous_source_domain = continuous_source_domain
        || params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str)
            == Some("source-domain-continuous-v1");
    let source_domain = if source_domain_profile && joint_neck {
        let active: Vec<bool> = (0..count)
            .map(|i| {
                joint_owned[i]
                    || face_membership[i] > 0.5
                    || neck_membership[i] > 0.5
                    || moving_attachment[i]
            })
            .collect();
        let blocked: Vec<bool> = (0..count)
            .map(|i| {
                if continuous_source_domain {
                    // Same-layer fixed material constrains the field. The
                    // neighbouring shared hair band has already been admitted
                    // by promote_interface and is absent from attachment_fixed.
                    // Scene and other-person output locks remain separate.
                    material_fixed[i] || attachment_fixed[i] || protection[i] >= 1.0 - 1e-6
                } else {
                    frozen[i] || material_fixed[i] || attachment_fixed[i]
                }
            })
            .collect();
        let origin = (left as f32 + 0.5, top as f32 + 0.5);
        // A short barrier taper makes the first contour samples outside a
        // fixed hair/ear attachment jump from almost zero gain to full cheek
        // gain over only a few pixels.  The resulting transported silhouette
        // is numerically valid, but its tangent develops a visible platform
        // at the jaw/neck join.  Keep the motion domain itself authoritative
        // and widen only this scalar transition for the source paired path;
        // the legacy path keeps its historical envelope.  The cheek interior
        // is already beyond the taper, so requested face width is unchanged.
        let taper = motion_domain_taper(frame.width(), continuous_source_domain);
        if continuous_source_domain {
            // The optical source coverage is not the domain of exp(v).
            // Reserve both directions of travel beyond the old silhouette;
            // this only permits evaluation of the foreground flow there.
            // Actual foreground/background ownership stays in the compositor.
            let travel = polygon
                .iter()
                .zip(&target)
                .map(|(&p, &q)| {
                    let d = sub(q, p);
                    dot(d, d).sqrt()
                })
                .fold(0.0_f32, f32::max)
                * 2.0;
            joint_domain::MotionDomain::build_with_motion_halo(
                cw, ch, origin, &active, &blocked, travel, taper,
            )
        } else {
            joint_domain::MotionDomain::build(cw, ch, origin, &active, &blocked, taper)
        }
    } else {
        None
    };
    #[cfg(test)]
    if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
        std::fs::create_dir_all(path).map_err(|error| invalid(&error.to_string()))?;
        std::fs::write(
            std::path::Path::new(path).join("source-domain-state.json"),
            serde_json::to_vec(&json!({
                "profile": params.get("experimentalFaceProfile"),
                "sourceDomainProfile": source_domain_profile,
                "jointNeck": joint_neck,
                "stableSourceLayers": stable_source_layers,
                "continuous": continuous_source_domain
            }))
            .unwrap(),
        )
        .map_err(|error| invalid(&error.to_string()))?;
    }
    mark!("motion domain");
    let velocity = if joint_neck {
        // One source-forward velocity authority. No nearest-segment/MLS
        // inverse offset enters the velocity. Positive control weights avoid RBF
        // overshoot; fixed material participates as zero-valued observations.
        // Direct control values are corrected by
        // the measured advected source-contour residual after exp(v).
        // Fixed context is not an anatomical attachment. A separate person or a
        // user-protected patch stays exact, while nearby skin may shrink away from
        // it. Only observed ears, hair and accessories taper the feasible contour.
        let distance_map = |seeds: &[bool]| {
            let mut distance = vec![u16::MAX; count];
            let mut queue = VecDeque::new();
            for (i, &fixed) in seeds.iter().enumerate() {
                if fixed {
                    distance[i] = 0;
                    queue.push_back(i);
                }
            }
            while let Some(i) = queue.pop_front() {
                let next = distance[i].saturating_add(1);
                for n in neighbors(i, cw, ch) {
                    if next < distance[n] {
                        distance[n] = next;
                        queue.push_back(n);
                    }
                }
            }
            distance
        };
        let context_fixed: Vec<_> = frozen
            .iter()
            .zip(&attachment_fixed)
            .map(|(&fixed, &attachment)| fixed && !attachment)
            .collect();
        let frozen_distance = distance_map(&context_fixed);
        let attachment_distance = distance_map(&attachment_fixed);
        let nearest_distance = |distance: &[u16], p: Point| {
            let x = (p.0 - left as f32 - 0.5)
                .clamp(0.0, (cw - 1) as f32)
                .round() as usize;
            let y = (p.1 - top as f32 - 0.5).clamp(0.0, (ch - 1) as f32).round() as usize;
            distance[y * cw + x] as f32
        };
        let velocity_taper = (frame.width() * 0.035).clamp(6.0, 14.0);
        let attachment_taper = (frame.width() * 0.14).max(8.0);
        #[cfg(test)]
        let geodesic_neck = params.get("__geodesicNeck").and_then(Value::as_bool) == Some(true);
        #[cfg(not(test))]
        let geodesic_neck = false;
        #[cfg(test)]
        let geodesic_neck_field = geodesic_neck
            .then(|| geodesic_neck_motion(&neck_membership, &face_membership, &joint_owned, cw, ch))
            .flatten();
        #[cfg(not(test))]
        let geodesic_neck_field: Option<Vec<f32>> = None;
        let external_constraint = |p: Point| {
            if !region.contains(p.0 / w, p.1 / h) {
                return 0.0;
            }
            let selection = output_mask
                .as_ref()
                .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                .unwrap_or(1.0);
            let protected = protect_mask
                .as_ref()
                .map(|m| m.coverage(p.0 / w, p.1 / h, source.width, source.height))
                .unwrap_or(0.0);
            selection * (1.0 - protected)
        };
        // Intended lower-neck motion falloff; the unchanged ROI can cut it short.
        // Compact support must already vanish there; no general ROI continuity guarantee.
        // This is a motion falloff, not an alpha cutoff or a made-up skin border.
        let legacy_neck_motion = |p: Point| {
            if joint_neck {
                1.0 - smoothstep(
                    frame.bottom + frame.height() * 0.04,
                    frame.bottom + frame.height() * 0.30,
                    frame.coordinates(p).1,
                )
            } else {
                1.0
            }
        };
        let neck_motion = |p: Point| {
            if geodesic_neck {
                let x = (p.0 - left as f32 - 0.5)
                    .clamp(0.0, (cw - 1) as f32)
                    .round() as usize;
                let y = (p.1 - top as f32 - 0.5).clamp(0.0, (ch - 1) as f32).round() as usize;
                geodesic_neck_field
                    .as_ref()
                    .map(|motion| motion[y * cw + x])
                    .unwrap_or_else(|| legacy_neck_motion(p))
            } else {
                legacy_neck_motion(p)
            }
        };
        let feature_weight = |p: Point| field.anchor_weight(p);
        let positive_kernel = true;
        #[cfg(test)]
        let positive_kernel = positive_kernel
            && params.get("__legacyFaceField").and_then(Value::as_bool) != Some(true)
            && !params
                .get("experimentalFaceProfile")
                .and_then(Value::as_str)
                .is_some_and(|mode| {
                    [
                        "supported-rbf-v1",
                        "feature-distance-v1",
                        "segmented-lateral-v1",
                    ]
                    .contains(&mode)
                });
        let supported_kernel = false;
        #[cfg(test)]
        let supported_kernel = supported_kernel
            || params
                .get("experimentalFaceProfile")
                .and_then(Value::as_str)
                == Some("supported-rbf-v1");
        #[cfg(test)]
        let feature_ramps = if params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str)
            == Some("feature-distance-v1")
        {
            Some(anchor_transition::FeatureRamps::new(&anchors, &polygon).map_err(invalid)?)
        } else {
            None
        };
        #[cfg(test)]
        let feature_weight = |p: Point| {
            feature_ramps
                .as_ref()
                .map(|ramps| ramps.weight(p))
                .unwrap_or_else(|| feature_weight(p))
        };
        let velocity_weight = |p: Point| {
            // The continuous candidate applies same-layer attachment_fixed
            // through the shared domain: it is both an output lock and a
            // continuous field barrier. Do not multiply the legacy attachment
            // distance ramp again here. Shared hair admitted upstream is no
            // longer attachment_fixed; unrelated scene/context output locks
            // remain separate from the foreground velocity definition.
            let attachment_transition = if continuous_source_domain {
                1.0
            } else {
                smoothstep(
                    2.0,
                    velocity_taper,
                    nearest_distance(&attachment_distance, p),
                )
            };
            let context_transition = if continuous_source_domain {
                1.0
            } else {
                smoothstep(0.0, 4.0, nearest_distance(&frozen_distance, p))
            };
            let legacy = attachment_transition
                * context_transition
                * external_constraint(p)
                * feature_weight(p)
                * neck_motion(p);
            source_domain
                .as_ref()
                .map(|domain| legacy * domain.weight(p))
                .unwrap_or(legacy)
        };
        let mut control_positions = Vec::new();
        let mut requested_positions = Vec::new();
        let mut feasible_positions = Vec::new();
        let mut velocity_values = Vec::new();
        let mut movable = Vec::new();
        let mut vertical_displacements = Vec::new();
        for k in 0..128 {
            let at = perimeter * k as f32 / 128.0;
            let edge = lengths
                .partition_point(|&d| d <= at)
                .saturating_sub(1)
                .min(polygon.len() - 1);
            let t = (at - lengths[edge]) / (lengths[edge + 1] - lengths[edge]).max(1e-6);
            let next = (edge + 1) % polygon.len();
            let p = add(scale(polygon[edge], 1.0 - t), scale(polygon[next], t));
            let requested = add(scale(target[edge], 1.0 - t), scale(target[next], t));
            let distance = nearest_distance(&attachment_distance, p);
            let achievable = if source_domain.is_some() {
                source_domain
                    .as_ref()
                    .map(|domain| domain.weight(p))
                    .unwrap_or(0.0)
                    * external_constraint(p)
            } else {
                smoothstep(2.0, attachment_taper, distance) * external_constraint(p)
            };
            let q = add(p, scale(sub(requested, p), achievable));
            vertical_displacements.push(
                (vertical_contour_offset(frame, polygon[edge], values) * (1.0 - t)
                    + vertical_contour_offset(frame, polygon[next], values) * t)
                    * achievable,
            );
            let weight = velocity_weight(p);
            control_positions.push(p);
            requested_positions.push(requested);
            feasible_positions.push(q);
            movable.push(weight > MIN_CONTROL_GAIN && dot(sub(q, p), sub(q, p)) > 0.01);
            velocity_values.push(if weight > MIN_CONTROL_GAIN {
                scale(
                    sub(q, p),
                    if supported_kernel || positive_kernel {
                        1.0
                    } else {
                        1.0 / weight
                    },
                )
            } else {
                (0.0, 0.0)
            });
        }
        #[cfg(test)]
        if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
            let controls = (0..128)
                .map(|i| {
                    let p = control_positions[i];
                    json!({
                        "p": p,
                        "requested": requested_positions[i],
                        "feasible": feasible_positions[i],
                        "domain": source_domain.as_ref().map(|d| d.weight(p)),
                        "external": external_constraint(p),
                        "velocityWeight": velocity_weight(p)
                    })
                })
                .collect::<Vec<_>>();
            std::fs::create_dir_all(path).map_err(|error| invalid(&error.to_string()))?;
            std::fs::write(
                std::path::Path::new(path).join("source-domain-controls.json"),
                serde_json::to_vec(&controls).unwrap(),
            )
            .map_err(|error| invalid(&error.to_string()))?;
        }
        #[cfg(test)]
        let pre_projection_positions = feasible_positions.clone();
        let center_u = (frame.left + frame.right) * 0.5;
        let mut projection_groups = Vec::new();
        for orientation in [1.0_f32, -1.0] {
            let mut side_indices: Vec<_> = (0..control_positions.len())
                .filter(|&i| {
                    let (u, v) = frame.coordinates(control_positions[i]);
                    let y = (v - frame.top) / frame.height();
                    orientation * (u - center_u) < -frame.width() * 0.08
                        && (0.35..0.985).contains(&y)
                })
                .collect();
            side_indices.sort_by(|&a, &b| {
                frame
                    .coordinates(control_positions[a])
                    .1
                    .total_cmp(&frame.coordinates(control_positions[b]).1)
            });
            projection_groups.push((side_indices, orientation));
        }
        #[cfg(test)]
        let periodic_curve = params
            .get("experimentalFaceProfile")
            .and_then(Value::as_str)
            == Some("shared-cheek-ear-periodic-v1");
        #[cfg(test)]
        let physical_bending_length = projection_groups
            .iter()
            .map(|(indices, _)| {
                indices
                    .windows(2)
                    .map(|pair| {
                        let delta = sub(control_positions[pair[1]], control_positions[pair[0]]);
                        dot(delta, delta).sqrt()
                    })
                    .sum::<f32>()
            })
            .sum::<f32>()
            / 50.0;
        #[cfg(test)]
        if periodic_curve {
            // Solve the observed contour once, including the chin and both
            // joins. Separate side arcs may overlap at nonzero motion, so an
            // arbitrary open-arc cut cannot supply valid boundary conditions.
            // The one physical bending length is the mean of the two former
            // core lengths / 25; it does not grow with full-contour length.
            projection_groups = vec![((0..control_positions.len()).collect(), 1.0)];
        }
        // Lateral projection retains independently requested axial movement.
        // A lateral regularizer does not certify the full 2D tangent or field.
        for (side_indices, orientation) in projection_groups {
            let samples: Vec<_> = side_indices
                .iter()
                .map(|&i| {
                    let (u, v) = frame.coordinates(control_positions[i]);
                    let desired = frame.coordinates(feasible_positions[i]).0;
                    let requested = frame
                        .coordinates(add(
                            control_positions[i],
                            scale(
                                sub(requested_positions[i], control_positions[i]),
                                external_constraint(control_positions[i]),
                            ),
                        ))
                        .0;
                    // Curve fairing and the field must agree on the same
                    // fixed degrees of freedom. Controls below this gain
                    // are zeroed by both velocity-value construction passes.
                    let fixed = velocity_weight(control_positions[i]) <= MIN_CONTROL_GAIN;
                    (
                        v,
                        orientation * u,
                        orientation * desired,
                        orientation * requested,
                        fixed,
                    )
                })
                .collect();
            #[cfg(not(test))]
            let projected =
                flow::project_side_curve(&samples).map_err(|message| invalid(&message))?;
            #[cfg(test)]
            let projected = if periodic_curve {
                displacement_curve::project_closed_curve(&samples, physical_bending_length)
                    .map_err(|message| invalid(&message))?
            } else if params
                .get("experimentalFaceProfile")
                .and_then(Value::as_str)
                .is_some_and(|mode| {
                    ["shared-cheek-curve-v4", "shared-cheek-ear-v1"].contains(&mode)
                })
            {
                displacement_curve::project_side_curve(&samples)
                    .map_err(|message| invalid(&message))?
            } else if params
                .get("experimentalFaceProfile")
                .and_then(Value::as_str)
                == Some("layered-curve-v1")
            {
                curve_candidate::project_side_curve(&samples)
                    .map_err(|message| invalid(&message))?
            } else {
                flow::project_side_curve(&samples).map_err(|message| invalid(&message))?
            };
            for (&i, &oriented_u) in side_indices.iter().zip(&projected) {
                let (_, v) = frame.coordinates(control_positions[i]);
                feasible_positions[i] =
                    frame.position(oriented_u * orientation, v + vertical_displacements[i]);
                #[cfg(test)]
                if periodic_curve
                    && oriented_u == frame.coordinates(control_positions[i]).0
                    && vertical_displacements[i] == 0.0
                {
                    feasible_positions[i] = control_positions[i];
                }
                let weight = velocity_weight(control_positions[i]);
                movable[i] = weight > MIN_CONTROL_GAIN
                    && dot(
                        sub(feasible_positions[i], control_positions[i]),
                        sub(feasible_positions[i], control_positions[i]),
                    ) > 0.01;
                velocity_values[i] = if weight > MIN_CONTROL_GAIN {
                    scale(
                        sub(feasible_positions[i], control_positions[i]),
                        if supported_kernel || positive_kernel {
                            1.0
                        } else {
                            1.0 / weight
                        },
                    )
                } else {
                    (0.0, 0.0)
                };
            }
        }
        for anchor in &anchors {
            let candidates = std::iter::once(anchor.center).chain((0..8).map(|k| {
                if anchor.polygon.is_empty() {
                    let a = std::f32::consts::TAU * k as f32 / 8.0;
                    add(
                        anchor.center,
                        add(
                            scale(anchor.across, anchor.rx * libm::cosf(a)),
                            scale(anchor.down, anchor.ry * libm::sinf(a)),
                        ),
                    )
                } else {
                    anchor.polygon[k * anchor.polygon.len() / 8]
                }
            }));
            for p in candidates {
                if control_positions
                    .iter()
                    .any(|&q| dot(sub(p, q), sub(p, q)) < 1.0)
                {
                    continue;
                }
                control_positions.push(p);
                requested_positions.push(p);
                feasible_positions.push(p);
                velocity_values.push((0.0, 0.0));
                movable.push(false);
            }
        }
        let boundary_constraints = positive_kernel;
        #[cfg(test)]
        let boundary_constraints = boundary_constraints
            && params
                .get("experimentalFaceProfile")
                .and_then(Value::as_str)
                != Some("positive-field-v1");
        if boundary_constraints {
            // Observed fixed material is a source boundary condition, not
            // merely a post-interpolation multiplier. Use the same physical
            // sampling pitch as the moving contour; no image-space repair.
            let pitch = (perimeter / 128.0).max(1.0);
            let mut boundary_controls: Vec<Point> = Vec::new();
            for i in 0..count {
                if !frozen[i] || !neighbors(i, cw, ch).any(|n| !frozen[n] && color_alpha[n] > 1e-6)
                {
                    continue;
                }
                let p = (
                    left as f32 + (i % cw) as f32 + 0.5,
                    top as f32 + (i / cw) as f32 + 0.5,
                );
                add_fixed_boundary_control(
                    p,
                    pitch,
                    &mut boundary_controls,
                    &mut control_positions,
                    &mut requested_positions,
                    &mut feasible_positions,
                    &mut velocity_values,
                    &mut movable,
                );
            }
        }
        let support_weights: Vec<f32> = (0..count)
            .map(|i| {
                let p = (
                    left as f32 + (i % cw) as f32 + 0.5,
                    top as f32 + (i / cw) as f32 + 0.5,
                );
                velocity_weight(p)
            })
            .collect();
        // Research comparisons retain their historical RBF, but a positive
        // kernel does not need its dense factorization or five unused solves.
        #[cfg(test)]
        let rbf = (!positive_kernel).then(|| {
            if supported_kernel {
                flow::CompactRbf::new_supported(
                    control_positions.clone(),
                    (frame.width() * 0.31).max(12.0),
                    control_positions
                        .iter()
                        .map(|&p| velocity_weight(p))
                        .collect(),
                )
            } else {
                flow::CompactRbf::new(control_positions.clone(), (frame.width() * 0.31).max(12.0))
            }
        });
        let positive_interpolator = positive_kernel::PositiveKernel::new(
            control_positions.clone(),
            (frame.width() * 0.31).max(12.0),
        );
        let mut velocity = flow::VelocityGrid::new(
            (left as f32 + 0.5, top as f32 + 0.5),
            cw,
            ch,
            vec![(0.0, 0.0); count],
        );
        mark!("motion controls and support");
        #[cfg(test)]
        let mut control_diagnostics = Vec::new();
        for iteration in 0..5 {
            #[cfg(test)]
            if params.get("__faceFieldTrace").is_some() {
                control_diagnostics.push(json!({"iteration":iteration,"values":velocity_values,
                    "weights":control_positions.iter().map(|&p| velocity_weight(p)).collect::<Vec<_>>() }));
            }
            #[cfg(test)]
            let coefficients = rbf.as_ref().map(|solver| solver.solve(&velocity_values));
            let vectors = (0..count)
                .map(|i| {
                    if support_weights[i] <= 1e-6 {
                        return (0.0, 0.0);
                    }
                    let p = (
                        left as f32 + (i % cw) as f32 + 0.5,
                        top as f32 + (i / cw) as f32 + 0.5,
                    );
                    #[cfg(not(test))]
                    let raw = positive_interpolator.evaluate(&velocity_values, p);
                    #[cfg(test)]
                    let raw = if positive_kernel {
                        positive_interpolator.evaluate(&velocity_values, p)
                    } else {
                        rbf.as_ref()
                            .expect("RBF mode has a solver")
                            .evaluate(coefficients.as_ref().expect("RBF mode has coefficients"), p)
                    };
                    scale(raw, support_weights[i])
                })
                .collect();
            velocity =
                flow::VelocityGrid::new((left as f32 + 0.5, top as f32 + 0.5), cw, ch, vectors);
            for i in 0..control_positions.len() {
                if !movable[i] {
                    continue;
                }
                let p = control_positions[i];
                let q = velocity.transport(p, 1.0);
                let error = sub(feasible_positions[i], q);
                if iteration < 4 {
                    let midpoint = scale(add(p, q), 0.5);
                    let weight =
                        (velocity_weight(p) + velocity_weight(midpoint) + velocity_weight(q)) / 3.0;
                    velocity_values[i] = add(
                        velocity_values[i],
                        scale(
                            error,
                            if supported_kernel || positive_kernel {
                                0.65
                            } else {
                                0.65 / weight.max(0.1)
                            },
                        ),
                    );
                }
            }
        }
        #[cfg(test)]
        if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
            anchor_transition::write_trace(
                path,
                &velocity,
                (left as f32 + 0.5, top as f32 + 0.5),
                cw,
                ch,
                &control_positions,
                &requested_positions,
                &feasible_positions,
                &pre_projection_positions,
                feature_ramps.as_ref(),
            )
            .map_err(|error| invalid(&error))?;
            std::fs::write(
                std::path::Path::new(path).join("control-iterations.json"),
                serde_json::to_vec(&control_diagnostics).unwrap(),
            )
            .map_err(|error| invalid(&error.to_string()))?;
            if observed_ear_motion.follows_joint_field() {
                // Independent photo audits need the material connection and
                // original ear evidence, not just a lower aggregate error.
                // RGB channels: original ear prior, joint ownership, frozen.
                let material_map: Vec<u8> = (0..count)
                    .flat_map(|i| {
                        [
                            byte(ear_membership[i] * 255.0),
                            if joint_owned[i] { 255 } else { 0 },
                            if frozen[i] { 255 } else { 0 },
                        ]
                    })
                    .collect();
                image::save_buffer(
                    std::path::Path::new(path).join("ear-material.png"),
                    &material_map,
                    cw as u32,
                    ch as u32,
                    image::ColorType::Rgb8,
                )
                .map_err(|error| invalid(&error.to_string()))?;
                let ear_indices: Vec<_> =
                    (0..count).filter(|&i| ear_membership[i] >= 0.5).collect();
                let report = json!({
                    "policy":"joint-flexible-observed-ear",
                    "origin":[left,top],"width":cw,"height":ch,
                    "materialMapChannels":["observedEarPrior","jointOwned","frozen"],
                    "observedEarPixels":ear_indices.len(),
                    "connectedEarPixels":ear_indices.iter().filter(|&&i| joint_owned[i]).count(),
                    "frozenEarPixels":ear_indices.iter().filter(|&&i| frozen[i]).count(),
                    "explicitlyProtectedEarPixels":ear_indices.iter().filter(|&&i| protection[i]>=1.0-1e-6).count(),
                    "earPriorIsPhysicalOpacity":false,
                    "rigidEarGuarantee":false,
                    "accessoryMotion":if moving_attachment_mask.is_some() {"source-plan"} else {"fixed"},
                    "plannedMovingAttachmentPixels":moving_attachment.iter().filter(|&&v| v).count()
                });
                std::fs::write(
                    std::path::Path::new(path).join("ear-material.json"),
                    serde_json::to_vec(&report).unwrap(),
                )
                .map_err(|error| invalid(&error.to_string()))?;
            }
            if let Some(probes) = params.get("__faceProbePoints").and_then(Value::as_array) {
                let coefficients = rbf.as_ref().map(|solver| solver.solve(&velocity_values));
                let report: Vec<_> = probes.iter().filter_map(|q| {
                    let p = (q.get(0)?.as_f64()? as f32, q.get(1)?.as_f64()? as f32);
                    let x = (p.0-left as f32-0.5).clamp(0.0,(cw-1) as f32).round() as usize;
                    let y = (p.1-top as f32-0.5).clamp(0.0,(ch-1) as f32).round() as usize;
                    let i = y*cw+x;
                    Some(json!({"p":p,"forward":velocity.transport(p,1.0),
                        "rawVelocity":if positive_kernel {positive_interpolator.evaluate(&velocity_values,p)} else {rbf.as_ref().expect("RBF mode has a solver").evaluate(coefficients.as_ref().expect("RBF mode has coefficients"),p)},"velocity":velocity.velocity(p),
                        "weight":velocity_weight(p),"featureWeight":feature_weight(p),
                        "contextDistance":frozen_distance[i],"attachmentDistance":attachment_distance[i],
                        "frozen":frozen[i],"inferredFixed":inferred_person_fixed[i],"attachmentFixed":attachment_fixed[i],
                        "faceMembership":face_membership[i],"colorAlpha":color_alpha[i],"jointOwned":joint_owned[i],
                        "earMembership":ear_membership.get(i).copied().unwrap_or(0.0),
                        "jointEarPrior":joint_ear_prior.get(i).copied().unwrap_or(0.0)}))
                }).collect();
                std::fs::write(
                    std::path::Path::new(path).join("probes.json"),
                    serde_json::to_vec(&report).unwrap(),
                )
                .map_err(|error| invalid(&error.to_string()))?;
            }
        }
        Some(velocity)
    } else {
        None
    };
    mark!("velocity grid solve");
    let mut grid =
        (!joint_neck).then(|| ContourGrid::new((left as f32 + 0.5, top as f32 + 0.5), cw, ch));
    let mut samples = vec![([0.0; 4], 0.0); count];
    #[cfg(test)]
    let target_alpha_used = vec![1.0_f32; count];
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
            // A contraction exposes the original backdrop. The source-role
            // candidate must never write a transported foreground into a
            // destination with no observed physical coverage; otherwise the
            // shared field can create horizontal donor stripes at the jaw.
            // Expansion keeps the established observed-foreground path.
            if source_domain_profile && contraction_only && color_alpha[i] <= 1e-6 {
                continue;
            }
            // Face and upper neck are one transported source material. A
            // semantic face/neck class boundary cannot clip the write domain:
            // adjacent opaque skin texels must use the same sampling map.
            // Exact protection is handled above, and the velocity already
            // decays to zero toward the lower neck.
            // Coverage describes the observed source material. The joint
            // path can include connected neck and permitted attachments;
            // source coverage is therefore not just the facial polygon.
            let old_coverage = if contour_matte.is_some() {
                color_alpha[i]
            } else {
                face_membership[i]
            };
            // Joint motion transports the physical face/neck alpha itself.
            // The face-only requested polygon cannot clip this layer: it omits
            // the upper neck, observed antialias fringe and feasible attachment
            // transition. The old/new transported coverage check below owns
            // the write support; unrelated scene remains exact when both are 0.
            if !joint_neck
                && old_coverage < if contour_matte.is_some() { 1e-6 } else { 0.5 }
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
            let movement = if let Some(velocity) = &velocity {
                flow::inverse(velocity, p)
            } else {
                scale(
                    grid.as_mut().expect("face-only grid").inverse(&field, p),
                    selection * (1.0 - protection[i]),
                )
            };
            if movement.0.abs() + movement.1.abs() < 1e-5 {
                continue;
            }
            let mut sample = if paired_foreground.is_some() {
                layer_pair::sample_residual(
                    &crop,
                    &color_alpha,
                    &paired_background,
                    x as f32 + movement.0,
                    y as f32 + movement.1,
                )
            } else {
                sample(
                    foreground,
                    &color_alpha,
                    x as f32 + movement.0,
                    y as f32 + movement.1,
                )
            };
            // The inverse sample transports physical source opacity. The
            // requested contour is a geometric constraint, not a second alpha
            // matte; multiplying by it would cut through opaque material.
            // A contraction exposes the original background. Do not let an
            // inverse sample that crosses the old edge paint foreground into
            // that destination; this is the source of ear/cheek color blocks.
            // Expansion still uses the existing path to cover a new target
            // strip with observed foreground.
            if !joint_neck && contraction_only && old_coverage <= 1e-6 {
                continue;
            }
            if !joint_neck && contour_matte.is_some() {
                clip_supported_jaw_opacity(
                    &mut sample,
                    p,
                    &crop,
                    &neck_support,
                    &field,
                    frame,
                    (left as f32, top as f32),
                );
            }
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
    let keep_source_plate = source_layers.is_some();
    #[cfg(test)]
    let keep_source_plate = keep_source_plate || params.get("__sourcePlatePrior").is_some();
    let mut background = if keep_source_plate {
        // A caller-supplied source scene is already paired with the sampled
        // foreground residual. Revealing it must preserve its spatial texture;
        // rebuilding it from destination holes would silently discard that
        // scene and break the declared immutable-layer contract.
        paired_background.clone()
    } else if paired_foreground.is_some() && holes.iter().any(|&h| h) {
        // The paired plate is an immutable F/B sampling plate. It contains a
        // deliberately regularized underlayer across the whole moving face,
        // which is correct for residual foreground transport but too coarse
        // to expose directly at a newly shortened jaw. Reconstruct the final
        // destination holes from their actual support and scene evidence so
        // the revealed strip follows the observed backdrop continuously.
        reconstruct_source_background(
            &crop,
            &holes,
            &color_alpha,
            &donor_frozen,
            &donor_excluded,
            &known_layer,
            color_layers.as_ref().map(|(_, b)| b),
            reconstruction_scene.as_deref(),
            independent_scene_color,
        )?
    } else if holes.iter().any(|&h| h) {
        reconstruct_with_scene(
            &crop,
            &holes,
            &color_alpha,
            &donor_frozen,
            &donor_excluded,
            &known_layer,
            color_layers.as_ref().map(|(_, b)| b),
            reconstruction_scene.as_deref(),
        )?
    } else {
        crop.clone()
    };
    if source_domain_profile && joint_neck && contraction_only && !keep_source_plate {
        repair_exposed_face_background(
            &mut background,
            &crop,
            &holes,
            &face_membership,
            &color_alpha,
            &frozen,
            &donor_excluded,
            &known_layer,
            &polygon,
            (left as f32, top as f32),
        );
    }
    let lighting_changed = if joint_neck {
        vec![false; count]
    } else {
        restore_neck_lighting(
            &mut background,
            &crop,
            &holes,
            &neck_lighting_known,
            &frozen,
            &field,
            (left as f32, top as f32),
        )
    };
    for (i, &light_changed) in lighting_changed.iter().enumerate() {
        if light_changed {
            changed[i] = true;
        }
    }
    mark!("reconstruct");
    #[cfg(test)]
    if std::env::var_os("FACE_SOURCE_LAYERS_TRACE").is_some() {
        if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
            // Preserve the actual immutable plate and write support so local
            // color experiments can replay complete floating-point composition,
            // rather than adding an approximate delta to a rounded PNG.
            let directory = std::path::Path::new(path);
            // This is the final source coverage actually used by the sampler,
            // including an opt-in source-scene lower bound when requested.
            std::fs::write(
                directory.join("source-color-alpha.f32"),
                color_alpha
                    .iter()
                    .flat_map(|a| a.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| invalid(&e.to_string()))?;
            image::save_buffer(
                directory.join("source-paired-background.png"),
                &paired_background.pixels,
                paired_background.width,
                paired_background.height,
                image::ColorType::Rgba8,
            )
            .map_err(|e| invalid(&e.to_string()))?;
            image::save_buffer(
                directory.join("destination-reconstructed-background.png"),
                &background.pixels,
                background.width,
                background.height,
                image::ColorType::Rgba8,
            )
            .map_err(|e| invalid(&e.to_string()))?;
            for (name, values) in [
                ("render-changed.u8", &changed),
                ("render-frozen.u8", &frozen),
            ] {
                std::fs::write(
                    directory.join(name),
                    values.iter().map(|&v| u8::from(v)).collect::<Vec<_>>(),
                )
                .map_err(|e| invalid(&e.to_string()))?;
            }
            for (name, values) in [
                ("source-donor-frozen.u8", &donor_frozen),
                ("source-donor-excluded.u8", &donor_excluded),
                ("source-moving-attachment.u8", &moving_attachment),
            ] {
                std::fs::write(
                    directory.join(name),
                    values.iter().map(|&v| u8::from(v)).collect::<Vec<_>>(),
                )
                .map_err(|e| invalid(&e.to_string()))?;
            }
            std::fs::write(directory.join("source-known-layer.u8"), &known_layer)
                .map_err(|e| invalid(&e.to_string()))?;
            let scene: Vec<u8> = (0..count)
                .flat_map(|i| {
                    reconstruction_scene
                        .as_ref()
                        .and_then(|v| v[i])
                        .unwrap_or([f32::NAN; 3])
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                })
                .collect();
            std::fs::write(directory.join("source-reconstruction-scene.f32"), scene)
                .map_err(|e| invalid(&e.to_string()))?;
        }
    }
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
        // Residual foreground is paired with the immutable source plate.
        // Replacing that whole plate, including its already visible fraction,
        // creates a finite colour jump even for infinitesimal motion. Only
        // the positive coverage difference uses the reconstructed destination
        // scene. This is the maximal-overlap approximation for local contour
        // contraction, not a recovery of subpixel transparency/occlusion.
        // With opaque old material this admits the entire new backdrop; at
        // identity it retains the exact source decomposition.
        let revealed_fraction = if paired_foreground.is_some() && membership < 1.0 {
            ((color_alpha[i] - membership).max(0.0) / (1.0 - membership)).min(1.0)
        } else {
            1.0
        };
        let j = ((top + i / cw) * source.width as usize + left + i % cw) * 4;
        for (c, &channel) in pixel.iter().enumerate().take(3) {
            let reconstructed = background.pixels[i * 4 + c] as f32;
            let paired = paired_background.pixels[i * 4 + c] as f32;
            let visible_background = paired + revealed_fraction * (reconstructed - paired);
            rendered.pixels[j + c] = if final_alpha > 1e-6 {
                byte((channel + visible_background * bg_weight) / final_alpha)
            } else {
                0
            };
        }
        rendered.pixels[j + 3] = byte(final_alpha * 255.0);
    }
    #[cfg(test)]
    if let (Some(path), Some(probes), Some(velocity)) = (
        params.get("__faceFieldTrace").and_then(Value::as_str),
        params.get("__faceProbePoints").and_then(Value::as_array),
        velocity.as_ref(),
    ) {
        let report: Vec<_> = probes.iter().filter_map(|q| {
            let p = (q.get(0)?.as_f64()? as f32, q.get(1)?.as_f64()? as f32);
            let x = (p.0-left as f32-0.5).clamp(0.0,(cw-1) as f32).round() as usize;
            let y = (p.1-top as f32-0.5).clamp(0.0,(ch-1) as f32).round() as usize;
            let i = y*cw+x;
            let j = ((top+y)*source.width as usize+left+x)*4;
            let movement = flow::inverse(velocity, p);
            let sampled = if paired_foreground.is_some() {
                layer_pair::sample_residual(&crop, &color_alpha, &paired_background,
                    x as f32+movement.0,y as f32+movement.1)
            } else {
                sample(foreground, &color_alpha, x as f32+movement.0,y as f32+movement.1)
            };
            Some(json!({"p":p,"inverseDisplacement":movement,"sampledAlpha":sampled.1,
                "sampledPremultiplied":sampled.0,"oldAlpha":color_alpha[i],"faceMembership":face_membership[i],
                "jointAlpha":joint_alpha[i],"jointOwned":joint_owned[i],"headAlpha":subject[i],
                "earMembership":ear_membership.get(i).copied().unwrap_or(0.0),
                "jointEarPrior":joint_ear_prior.get(i).copied().unwrap_or(0.0),
                "sceneBackground":scene_edge.as_ref().and_then(|e|e.background[i]),
                "sceneMinimumAlpha":scene_edge.as_ref().map(|e|e.minimum_alpha[i]),
                "targetAlpha":alpha(p,&target),"targetAlphaUsed":target_alpha_used[i],"frozen":frozen[i],"changed":changed[i],"hole":holes[i],
                "knownLayer":known_layer[i],"donorFrozen":donor_frozen[i],"donorExcluded":donor_excluded[i],
                "source":&source.pixels[j..j+4],"output":&rendered.pixels[j..j+4],
                "renderSample":samples[i],
                "background":&background.pixels[i*4..i*4+4],
                "pairedBackground":&paired_background.pixels[i*4..i*4+4],
                "estimatedBackground":color_layers.as_ref().map(|(_,b)| &b.pixels[i*4..i*4+4])}))
        }).collect();
        std::fs::write(
            std::path::Path::new(path).join("render-probes.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .map_err(|error| invalid(&error.to_string()))?;
    }
    // ROI/protection/opacity/blend are applied once over the expanded support,
    // never again clipped by the original foreground alpha in engine.rs.
    mark!("compose layers");
    let mut output = if joint_neck {
        // Normal blending uses selection/protection in the velocity once.
        // Color blend modes additionally retain the public soft blend weights.
        let mut composite_params = support_params.into_owned();
        if composite_params
            .get("blendMode")
            .and_then(Value::as_str)
            .unwrap_or("normal")
            == "normal"
        {
            composite_params
                .as_object_mut()
                .expect("validated params")
                .remove("mask");
            composite_params
                .as_object_mut()
                .expect("validated params")
                .remove("protectMask");
        }
        crate::mask::composite(source, &rendered, &composite_params)
    } else {
        crate::mask::composite(source, &rendered, &support_params)
    };
    if joint_neck {
        // Non-normal blend modes must not recolor untouched or hard-frozen
        // pixels merely because the transported layer copied the source there.
        for (i, original) in source.pixels.chunks_exact(4).enumerate() {
            if rendered.pixels[i * 4..i * 4 + 4] == *original {
                output.pixels[i * 4..i * 4 + 4].copy_from_slice(original);
            }
        }
    }
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

    #[test]
    fn fixed_boundary_is_independent_of_nearby_free_handle_velocity() {
        let field = |epsilon: f32| {
            let mut positions = vec![(7.5, 7.5), (12.0, 7.5)];
            let mut requested = positions.clone();
            let mut feasible = positions.clone();
            let mut values = vec![(3.0, 0.0), (epsilon, 0.0)];
            let mut movable = vec![true, true];
            let mut boundary = Vec::new();
            add_fixed_boundary_control(
                (12.5, 7.5),
                1.0,
                &mut boundary,
                &mut positions,
                &mut requested,
                &mut feasible,
                &mut values,
                &mut movable,
            );
            assert_eq!(boundary, vec![(12.5, 7.5)]);
            assert_eq!(movable, vec![true, true, false]);
            let kernel = positive_kernel::PositiveKernel::new(positions, 8.0);
            kernel.evaluate(&values, (10.5, 7.5))
        };
        let zero = field(0.0);
        let tiny = field(0.00001);
        assert!((zero.0 - tiny.0).hypot(zero.1 - tiny.1) < 0.00002);
    }

    #[test]
    fn exact_fixed_boundary_collision_overrides_free_motion() {
        let p = (12.5, 7.5);
        let mut positions = vec![p];
        let mut requested = vec![(15.5, 7.5)];
        let mut feasible = requested.clone();
        let mut values = vec![(3.0, 0.0)];
        let mut movable = vec![true];
        add_fixed_boundary_control(
            p,
            1.0,
            &mut Vec::new(),
            &mut positions,
            &mut requested,
            &mut feasible,
            &mut values,
            &mut movable,
        );
        assert_eq!(positions, vec![p]);
        assert_eq!(requested, vec![(15.5, 7.5)]);
        assert_eq!(feasible, vec![p]);
        assert_eq!(values, vec![(0.0, 0.0)]);
        assert_eq!(movable, vec![false]);
    }

    pub(super) fn joint_portrait() -> (RgbaImage, Value) {
        let (short_source, mut params) = portrait(160, 240);
        let mut pixels = short_source.pixels;
        pixels.extend((0..160 * 80).flat_map(|_| [20_u8, 50, 130, 255]));
        let mut source = RgbaImage::new(160, 320, pixels).unwrap();
        for point in params["semanticMasks"]["face"]["points"]
            .as_array_mut()
            .unwrap()
        {
            point["y"] = json!(point["y"].as_f64().unwrap() * 0.75);
        }
        let mut params = raster_params(&source, params);
        params["semanticMasks"]["faceContour"] = params["semanticMasks"]["faceOutline"].clone();
        params["semanticMasks"]["faceContourMask"] = params["semanticMasks"]["face"].clone();
        let face: Vec<u8> = params["semanticMasks"]["face"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let mut skin = face.clone();
        let mut neck = vec![0_u8; 160 * 320];
        let mut person = face.clone();
        for y in 180..320 {
            for x in 70..90 {
                let i = y * 160 + x;
                skin[i] = 255;
                person[i] = 255;
                if face[i] < 128 {
                    neck[i] = 255;
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                        205 + (x % 7) as u8,
                        145 + (y % 5) as u8,
                        90,
                        255,
                    ]);
                }
            }
        }
        let mask = |data: Vec<u8>| json!({"shape":"raster","width":160,"height":320,"data":data});
        params["semanticMasks"]["skin"] = mask(skin);
        params["semanticMasks"]["neck"] = mask(neck);
        params["subjectMask"] = mask(person);
        (source, params)
    }

    pub(super) fn public_adjust(
        source: &RgbaImage,
        params: &Value,
    ) -> Result<RgbaImage, ImageError> {
        crate::apply_graph_json(
            source,
            &json!({"version":"1","operations":[{
                "id":"face-joint-contract","type":"face.adjust","params":params
            }]})
            .to_string(),
        )
    }

    #[test]
    fn source_attachment_plan_prevents_a_stale_disconnected_ear_lock() {
        let (mut source, mut params) = joint_portrait();
        let size = 160 * 320;
        let mut ears = vec![0_u8; size];
        let mut accessory = vec![0_u8; size];
        // A slender ear joins the cheek above a stud. The parser assigns the
        // stud to accessory rather than skin/ear, leaving a class gap.
        for y in 110..155 {
            for x in 35..=if y < 115 { 45 } else { 40 } {
                let i = y * 160 + x;
                ears[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[205, 145, 95, 255]);
                if (130..134).contains(&y) {
                    ears[i] = 0;
                    accessory[i] = 255;
                    source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[210, 195, 40, 255]);
                }
            }
        }
        let mask = |data: Vec<u8>| json!({"shape":"raster","width":160,"height":320,"data":data});
        params["semanticMasks"]["ears"] = mask(ears);
        params["semanticMasks"]["accessories"] = mask(accessory.clone());
        params["experimentalFaceProfile"] = json!("shared-cheek-ear-periodic-v1");
        params["adjustments"] = json!({"faceWidth":30});
        params["__faceProbePoints"] = json!([[37.5, 150.5], [37.5, 131.5]]);
        let root =
            std::env::temp_dir().join(format!("face-moving-attachment-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let render = |name: &str, p: &Value| {
            let mut p = p.clone();
            let path = root.join(name);
            p["__faceFieldTrace"] = json!(path);
            let result = adjust(&source, &p).unwrap();
            let probes: Value =
                serde_json::from_slice(&std::fs::read(path.join("render-probes.json")).unwrap())
                    .unwrap();
            (result, probes)
        };
        let (baseline, old) = render("fixed", &params);
        assert_eq!(
            old[0]["frozen"], true,
            "counterexample must reproduce the stale lower-ear lock"
        );
        assert_eq!(old[0]["jointOwned"], false);
        let mut zero = params.clone();
        zero["__sourceMovingAttachmentSupport"] = mask(vec![0; size]);
        let (no_plan_motion, _) = render("empty-plan", &zero);
        assert_eq!(
            baseline, no_plan_motion,
            "an empty plan must preserve existing behavior"
        );
        let mut planned = params.clone();
        planned["__sourceMovingAttachmentSupport"] = mask(accessory.clone());
        let (_, new) = render("planned", &planned);
        assert_eq!(
            new[0]["frozen"], false,
            "planned stud must not cut off the attached ear below it"
        );
        assert_eq!(new[0]["jointOwned"], true);
        assert_eq!(
            new[1]["jointOwned"], true,
            "permission must bridge the parser class gap"
        );
        assert_eq!(old[1]["jointAlpha"], 0.0);
        assert_eq!(
            new[1]["jointAlpha"], 0.0,
            "motion support must not be promoted into opacity"
        );
        assert_eq!(
            new[1]["donorExcluded"], true,
            "moving adornment is never a backdrop donor"
        );
        assert_eq!(
            new[1]["donorFrozen"], false,
            "old stud must not be pinned into the backdrop plate"
        );
        for conflict in ["user", "hair", "feature", "garment"] {
            let mut blocked = planned.clone();
            match conflict {
                "user" => blocked["protectMask"] = mask(accessory.clone()),
                "hair" => blocked["semanticMasks"]["hair"] = mask(accessory.clone()),
                "feature" => blocked["semanticMasks"]["leftEye"] = mask(accessory.clone()),
                "garment" => blocked["__sourceGarment"] = mask(accessory.clone()),
                _ => unreachable!(),
            }
            let result = adjust(&source, &blocked);
            assert!(
                result.is_err(),
                "planned attachment must reject {conflict} protection rather than remove it"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn detected_joint_foreground_keeps_the_skin_join_connected_and_low_neck_fixed() {
        let (source, params) = joint_portrait();
        let output = public_adjust(&source, &params).unwrap();
        for y in 180..220 {
            for x in 76..84 {
                let i = (y * 160 + x) * 4;
                assert!(
                    output.pixels[i] > 150 && output.pixels[i + 2] < 120,
                    "scene hole at face/neck join {x},{y}: {:?}",
                    &output.pixels[i..i + 4]
                );
            }
        }
        for y in 280..320 {
            for x in 70..90 {
                let i = (y * 160 + x) * 4;
                assert_eq!(
                    &output.pixels[i..i + 4],
                    &source.pixels[i..i + 4],
                    "low neck moved {x},{y}"
                );
            }
        }
        let before = span(&source, 120);
        let after = span(&output, 120);
        assert!(
            after.0 >= before.0 + 4 && after.1 <= before.1 - 4,
            "joint foreground cancelled slimming {before:?}->{after:?}"
        );
    }

    #[test]
    fn missing_zero_and_disconnected_neck_evidence_keep_the_face_only_contract() {
        let (source, params) = joint_portrait();
        for variant in [
            "missing-skin",
            "missing-neck",
            "zero-neck",
            "disconnected-neck",
            "zero-skin",
            "zero-skin-overlapped-neck",
            "neck-only-inside-face",
        ] {
            let mut params = params.clone();
            match variant {
                "missing-skin" => {
                    params["semanticMasks"]
                        .as_object_mut()
                        .unwrap()
                        .remove("skin");
                }
                "missing-neck" => {
                    params["semanticMasks"]
                        .as_object_mut()
                        .unwrap()
                        .remove("neck");
                }
                "zero-neck" => {
                    params["semanticMasks"]["neck"]["data"] = json!(vec![0_u8; 160 * 320]);
                }
                "disconnected-neck" => {
                    let mut neck = vec![0_u8; 160 * 320];
                    for y in 2..8 {
                        for x in 2..8 {
                            neck[y * 160 + x] = 255;
                        }
                    }
                    params["semanticMasks"]["neck"]["data"] = json!(neck);
                }
                "zero-skin" => {
                    params["semanticMasks"]["skin"]["data"] = json!(vec![0_u8; 160 * 320]);
                }
                "zero-skin-overlapped-neck" => {
                    params["semanticMasks"]["skin"]["data"] = json!(vec![0_u8; 160 * 320]);
                    params["semanticMasks"]["neck"]["data"] =
                        params["semanticMasks"]["face"]["data"].clone();
                }
                "neck-only-inside-face" => {
                    params["semanticMasks"]["neck"]["data"] =
                        params["semanticMasks"]["face"]["data"].clone();
                }
                _ => unreachable!(),
            }
            let output = public_adjust(&source, &params).unwrap();
            let mut legacy = params.clone();
            legacy["semanticMasks"]
                .as_object_mut()
                .unwrap()
                .remove("skin");
            assert_eq!(
                output,
                public_adjust(&source, &legacy).unwrap(),
                "fallback changed {variant}"
            );
            assert_ne!(
                output.pixels, source.pixels,
                "fallback stopped valid editing {variant}"
            );
        }
    }

    #[test]
    fn joint_zero_controls_selection_and_full_protection_are_exact() {
        let (source, params) = joint_portrait();
        for field in ["adjustments", "mask", "protectMask"] {
            let mut params = params.clone();
            params[field] = match field {
                "adjustments" => json!({}),
                "mask" => {
                    json!({"shape":"raster","width":160,"height":320,"data":vec![0_u8;160*320]})
                }
                _ => json!({"shape":"raster","width":160,"height":320,"data":vec![255_u8;160*320]}),
            };
            assert_eq!(
                public_adjust(&source, &params).unwrap(),
                source,
                "identity contract failed for {field}"
            );
        }
    }

    #[test]
    fn joint_non_normal_blend_preserves_hard_and_soft_mask_permissions() {
        let (source, mut params) = joint_portrait();
        params["blendMode"] = json!("multiply");
        params["mask"] = json!({"shape":"raster","width":160,"height":320,
            "data":(0..160*320).map(|i|if i%160<80 {255_u8}else{0}).collect::<Vec<_>>()});
        params["protectMask"] =
            json!({"shape":"rect","x":0.18,"y":0.25,"width":0.23,"height":0.12});
        let output = public_adjust(&source, &params).unwrap();
        assert_ne!(
            output.pixels, source.pixels,
            "non-normal selection lost valid editing"
        );
        for y in 0..320 {
            for x in 0..160 {
                let nx = (x as f32 + 0.5) / 160.0;
                let ny = (y as f32 + 0.5) / 320.0;
                let protected = coverage(&params["protectMask"], nx, ny, 160, 320) >= 1.0 - 1e-6;
                if x >= 80 || protected || x < 16 || !(7..314).contains(&y) {
                    let i = (y * 160 + x) * 4;
                    assert_eq!(
                        &output.pixels[i..i + 4],
                        &source.pixels[i..i + 4],
                        "permission leaked at {x},{y}"
                    );
                }
            }
        }
        for (field, byte) in [("mask", 1_u8), ("protectMask", 254_u8)] {
            let mut soft = params.clone();
            soft.as_object_mut().unwrap().remove("mask");
            soft.as_object_mut().unwrap().remove("protectMask");
            soft[field] =
                json!({"shape":"raster","width":160,"height":320,"data":vec![byte;160*320]});
            let output = public_adjust(&source, &soft).unwrap();
            let maximum = output
                .pixels
                .iter()
                .zip(&source.pixels)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            assert!(
                maximum <= 3,
                "near-zero {field} allowed full multiply effect: {maximum}"
            );
        }
    }

    #[test]
    fn joint_width_controls_move_both_physical_sides_and_vertical_controls_move_material_endpoints()
    {
        let (source, mut params) = joint_portrait();
        let top = 240.0 * 0.13;
        let height = 240.0 * 0.74;
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
            let before = span(&source, y);
            for amount in [-100, 100] {
                params["adjustments"] = json!({key:amount});
                let output = public_adjust(&source, &params).unwrap();
                let after = span(&output, y);
                if (amount > 0) == positive_narrows {
                    assert!(
                        after.0 > before.0 && after.1 < before.1,
                        "{key} {amount} did not narrow {before:?}->{after:?}"
                    );
                } else {
                    assert!(
                        after.0 < before.0 && after.1 > before.1,
                        "{key} {amount} did not widen {before:?}->{after:?}"
                    );
                }
            }
        }
        let endpoints = |image: &RgbaImage| {
            let ys: Vec<_> = (0..320)
                .filter(|&y| image.pixels[(y * 160 + 80) * 4] >= 218)
                .collect();
            (*ys.first().unwrap(), *ys.last().unwrap())
        };
        let before = endpoints(&source);
        for key in ["forehead", "chin"] {
            for amount in [-100, 100] {
                params["adjustments"] = json!({key:amount});
                let output = public_adjust(&source, &params).unwrap();
                let after = endpoints(&output);
                if key == "chin" {
                    assert!(
                        (after.1 as i32 - before.1 as i32) * amount > 0,
                        "chin endpoint did not move {amount}: {before:?}->{after:?}"
                    );
                } else {
                    assert!(
                        (before.0 as i32 - after.0 as i32) * amount > 0,
                        "forehead endpoint did not move {amount}: {before:?}->{after:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn joint_mixed_vertical_sliders_are_continuous_across_zero() {
        let (source, mut params) = joint_portrait();
        params["adjustments"] = json!({"faceWidth":70});
        let base = public_adjust(&source, &params).unwrap();
        for key in ["chin", "forehead"] {
            for amount in [-0.0001_f64, 0.0001] {
                params["adjustments"] = json!({"faceWidth":70,key:amount});
                let nearby = public_adjust(&source, &params).unwrap();
                let maximum = nearby
                    .pixels
                    .iter()
                    .zip(&base.pixels)
                    .map(|(&a, &b)| a.abs_diff(b))
                    .max()
                    .unwrap();
                assert!(
                    maximum <= 1,
                    "{key} crossed zero with a discontinuous reroute: {maximum}"
                );
            }
        }
        params["adjustments"] = json!({"faceWidth":70,"chin":60,"forehead":-40,"jaw":20});
        let combined = public_adjust(&source, &params).unwrap();
        assert_ne!(
            combined.pixels, base.pixels,
            "mixed vertical/lateral controls were discarded"
        );
        for i in (280 * 160)..(320 * 160) {
            assert_eq!(
                &combined.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "mixed controls moved low neck"
            );
        }
    }

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
    fn lateral_limit_is_smooth_when_multiple_face_controls_reach_the_bound() {
        let limit = 0.11;
        let samples = [-0.111_f32, -0.11, -0.109, 0.109, 0.11, 0.111];
        let values: Vec<_> = samples
            .into_iter()
            .map(|value| smooth_lateral_limit(value, limit))
            .collect();
        // The bounded target remains strictly monotone in both directions;
        // a hard clamp would make the slope exactly zero after the cap.
        assert!(values.windows(2).all(|pair| pair[1] > pair[0]));
        let left_slope = (values[2] - values[1]) / 0.001;
        let right_slope = (values[4] - values[3]) / 0.001;
        assert!(left_slope > 0.25 && right_slope > 0.25);
        assert!(values.iter().all(|value| value.abs() < limit));
    }

    #[test]
    fn source_paired_motion_domain_uses_a_wider_attachment_taper() {
        let legacy = motion_domain_taper(220.0, false);
        let paired = motion_domain_taper(220.0, true);
        assert_eq!(legacy, 24.0);
        assert_eq!(paired, 35.2);
        assert!(paired > legacy);
        // Small and very large faces stay within bounded raster-safe bands.
        assert_eq!(motion_domain_taper(1.0, true), 10.0);
        assert_eq!(motion_domain_taper(1000.0, true), 36.0);
    }

    #[test]
    fn contour_grid_transport_has_no_cutoff_jump() {
        let exact = (2.0, -1.0);
        let grid = (2.8, -0.4);
        let step = 4.0;
        let cutoff = 3.0 + step * 2.0_f32.sqrt();
        let left = blend_grid_motion(exact, grid, cutoff - 1e-3, step);
        let right = blend_grid_motion(exact, grid, cutoff + 1e-3, step);
        let jump = sub(right, left).0.hypot(sub(right, left).1);
        assert!(jump < 0.001, "grid/exact cutoff jump was {jump}");
        let far = blend_grid_motion(exact, grid, cutoff + step * 2.0, step);
        assert!((far.0 - grid.0).abs() < 1e-5);
        assert!((far.1 - grid.1).abs() < 1e-5);
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
    fn supported_contracted_jaw_clips_opaque_source_lobes_without_changing_color() {
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
        let crop = RgbaImage::new(
            w as u32,
            h as u32,
            vec![[80, 60, 45, 255]; w * h]
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();
        let clean: Vec<f32> = (0..w * h)
            .map(|i| {
                if i / w >= 45 && (20..44).contains(&(i % w)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        // True target bottom is y=41.0. A source mixed-alpha lobe falsely
        // sampled as .9 must not survive at y=41.3 or y=42.0.
        for y in [41.3, 42.0] {
            let p = (32.0, y);
            let mut rgba = ([198.0, 144.0, 90.0, 0.9], 0.9);
            clip_supported_jaw_opacity(&mut rgba, p, &crop, &clean, &field, frame, (0.0, 0.0));
            let expected = alpha(p, &target);
            assert!(
                (rgba.1 - expected).abs() < 1e-6,
                "opacity lobe survived at {p:?}: {} > {expected}",
                rgba.1
            );
            if rgba.1 > 1e-6 {
                for (c, expected) in [220.0, 160.0, 100.0].iter().enumerate() {
                    assert!(
                        (rgba.0[c] / rgba.0[3] - expected).abs() < 1e-4,
                        "face color changed instead of opacity"
                    );
                }
            } else {
                assert_eq!(rgba.0, [0.0; 4]);
            }
        }
        // A correct fully internal sample remains exact, including alpha.
        let mut interior = ([198.0, 144.0, 90.0, 0.9], 0.9);
        clip_supported_jaw_opacity(
            &mut interior,
            (32.0, 40.0),
            &crop,
            &clean,
            &field,
            frame,
            (0.0, 0.0),
        );
        assert_eq!(interior, ([198.0, 144.0, 90.0, 0.9], 0.9));
    }

    #[test]
    fn jaw_opacity_clipping_requires_observed_neck_and_never_limits_expansion_or_cheek() {
        let (w, h) = (64_usize, 64_usize);
        let polygon: Vec<Point> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.0;
                (32.0 + 18.0 * a.cos(), 24.0 + 20.0 * a.sin())
            })
            .collect();
        let frame = Frame::new(&polygon, None, w as f32, h as f32);
        let crop = RgbaImage::new(
            w as u32,
            h as u32,
            vec![[80, 60, 45, 255]; w * h]
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();
        let value = ([198.0, 144.0, 90.0, 0.9], 0.9);
        for (control, p, clean) in [
            (-100, (32.0, 42.0), vec![0.0; w * h]),
            (-100, (32.0, 42.0), vec![0.998; w * h]),
            (-100, (50.0, 24.0), vec![1.0; w * h]),
            (100, (32.0, 47.0), vec![1.0; w * h]),
        ] {
            let target = target_contour(&polygon, frame, &json!({"chin":control}));
            let field = ContourField {
                source: &polygon,
                target: &target,
                anchors: &[],
                support: 8.0,
                mls: mls::RigidMls::new(vec![]),
            };
            let mut actual = value;
            clip_supported_jaw_opacity(&mut actual, p, &crop, &clean, &field, frame, (0.0, 0.0));
            assert_eq!(
                actual, value,
                "unsupported/non-contracting edge changed at {p:?}"
            );
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
