//! Isolated candidate for upper-face width editing.
//!
//! `face.adjust` currently treats the observed hair/ear/accessory masks as a
//! stationary obstacle. That is correct for the outside head silhouette, but
//! it also prevents an upper-face (forehead/temple) contour from moving when
//! the contour touches hair. This candidate treats only an observed,
//! connected skin/hair interface as a shared material domain. The outside
//! head boundary, eyes/brows/nose/mouth, accessories and explicit protection
//! remain zero-velocity constraints.
//!
//! The production entry point uses only the evidence-bounded `promote_interface`
//! adapter below; the direct field renderer remains a test-only reference for
//! checking direction, topology and local Jacobians before transport changes.

use super::*;

#[cfg(test)]
fn normal_at(polygon: &[Point], index: usize, point: Point, frame: Frame) -> Option<Point> {
    let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
    let next = polygon[(index + 1) % polygon.len()];
    let tangent = sub(next, previous);
    let length = dot(tangent, tangent).sqrt();
    if !length.is_finite() || length <= 1e-5 {
        return None;
    }
    let candidate = (tangent.1 / length, -tangent.0 / length);
    let opposite = scale(candidate, -1.0);
    let (u, _) = frame.coordinates(point);
    let side = (u - (frame.left + frame.right) * 0.5).signum();
    if side == 0.0 {
        return None;
    }
    let desired = scale(frame.across, side);
    Some(if dot(candidate, desired) >= dot(opposite, desired) {
        candidate
    } else {
        opposite
    })
}

#[cfg(test)]
fn profile(y: f32, start: f32, peak: f32, end: f32) -> f32 {
    if y < peak {
        smoothstep(start, peak, y)
    } else {
        1.0 - smoothstep(peak, end, y)
    }
}

#[cfg(test)]
pub(super) struct UpperWidth {
    velocity: flow::VelocityGrid,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    editable: Vec<bool>,
}

/// Build a bounded parameter variant that lets the established joint face
/// field see a narrow, observed hair band as moving facial material. This is
/// intentionally a semantic-input transformation rather than an output paint:
/// source RGB is untouched, the outer head/hair remains fixed, and the normal
/// `face.adjust` ownership/transport code still performs the render.
///
/// The returned value carries `__upperSharedPrepared=true` so a recursive
/// adjustment cannot promote the same band twice. `None` means that the input lacks
/// the three raster masks or has no eligible connected hair band.
pub(super) fn promote_interface(
    params: &Value,
    polygon: &[Point],
    frame: Frame,
    source_width: u32,
    source_height: u32,
) -> Option<Value> {
    promote_with_source(params, polygon, frame, source_width, source_height, false, None)
}

pub(super) fn promote_with_source(
    params: &Value, polygon: &[Point], frame: Frame,
    source_width: u32, source_height: u32,
    continuous_source: bool, source_interface: Option<&Value>,
) -> Option<Value> {
    let values = params.get("adjustments").unwrap_or(params);
    let shared_cheek = continuous_source && percent(values, "faceWidth").abs() > 1e-6;
    #[cfg(test)]
    let shared_cheek = shared_cheek || params
        .get("experimentalFaceProfile")
        .and_then(Value::as_str)
        .is_some_and(|mode| {
            [
                "shared-cheek-v1",
                "shared-cheek-curve-v4",
                "shared-cheek-ear-v1",
                "shared-cheek-ear-periodic-v1",
                "source-domain-continuous-v1",
            ]
            .contains(&mode)
        })
        && percent(values, "faceWidth").abs() > 1e-6;
    if !shared_cheek
        && percent(values, "foreheadWidth").abs() < 1e-6
        && percent(values, "temples").abs() < 1e-6
    {
        return None;
    }
    if params.get("__upperSharedPrepared").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let semantics = params.get("semanticMasks")?.as_object()?;
    let head_value = semantics.get("face")?;
    // Source material and its geometric interface may be detected at a finer
    // scale than the retained optical silhouette. Use that source interface
    // only for motion admission; do not replace optical/target graph fields.
    let motion_interface = if shared_cheek { source_interface } else { None };
    #[cfg(test)]
    let motion_interface = motion_interface.or_else(|| shared_cheek.then(|| params.get("__sourceMotionInterface")).flatten());
    let skin_value = match motion_interface {
        Some(interface) => interface.get("mask")?,
        None => semantics.get("faceContourMask")?,
    };
    let motion_polygon = if let Some(interface) = motion_interface {
        let contour = interface.get("contour")?;
        if contour.get("shape").and_then(Value::as_str) != Some("polygon") {
            return None;
        }
        let vertices = contour.get("points")?.as_array()?;
        if vertices.len() < 3
            || vertices.iter().any(|p| {
                ["x", "y"].iter().any(|key| {
                    !p.get(*key)
                        .and_then(Value::as_f64)
                        .is_some_and(|v| v.is_finite() && (0.0..=1.0).contains(&v))
                })
            })
        {
            return None;
        }
        Some(points(contour, source_width as f32, source_height as f32))
    } else {
        None
    };
    let polygon = motion_polygon.as_deref().unwrap_or(polygon);
    let hair_value = semantics.get("hair")?;
    let connected_skin_value = semantics.get("skin")?;
    for value in [head_value, skin_value, hair_value, connected_skin_value] {
        if value.get("shape").and_then(Value::as_str) != Some("raster") {
            return None;
        }
    }
    let mask_size = |value: &Value| {
        let width = value.get("width")?.as_u64()? as usize;
        let height = value.get("height")?.as_u64()? as usize;
        let data = value.get("data")?.as_array()?;
        (width > 0 && height > 0 && width.checked_mul(height) == Some(data.len()))
            .then_some((width, height))
    };
    let (mask_width, mask_height) = mask_size(hair_value)?;
    if mask_size(head_value)? != (mask_width, mask_height)
        || mask_size(skin_value)? != (mask_width, mask_height)
        || mask_size(connected_skin_value)? != (mask_width, mask_height)
    {
        return None;
    }
    let head = PreparedMask::new(head_value);
    let skin = PreparedMask::new(skin_value);
    let hair = PreparedMask::new(hair_value);
    let width = source_width as f32;
    let height = source_height as f32;
    // A promoted raster node contributes to an entire bilinear cell around
    // it, not just its centre. Keep that support, the two-pixel outer hair
    // guard, and one source sampling pixel away from the observed head exterior.
    // This source-only test is identical for narrowing and widening. It is
    // a domain preflight, not a proof about every integrated trajectory.
    let clearance_x = 1 + (3.0 * mask_width as f32 / width).ceil() as usize;
    let clearance_y = 1 + (3.0 * mask_height as f32 / height).ceil() as usize;
    // Research-only occupancy evidence for a reviewed single-person source.
    // Head class uncertainty can connect through the neck/clothing classes to
    // the image border without crossing the actual person exterior. Aggregate
    // person confidence may close that false exterior channel. It is neither
    // physical opacity nor instance segmentation; it does not admit nodes or
    // change the original head predicate, moving band or explicit protection.
    #[cfg(test)]
    let outer_subject = params
        .get("__experimentalOuterSubjectOccupancy")
        .and_then(Value::as_bool)
        .filter(|&enabled| enabled && shared_cheek)
        .and_then(|_| params.get("subjectMask"))
        .filter(|value| value.get("shape").and_then(Value::as_str) == Some("raster"))
        .filter(|value| mask_size(value) == Some((mask_width, mask_height)))
        .and_then(|value| value.get("data"))
        .and_then(Value::as_array);
    #[cfg(not(test))]
    let outer_subject: Option<&Vec<Value>> = None;
    // Separate source-reviewed experiment: non-background class occupancy may
    // extend the moving interface through an uncertain head-class join. This
    // is a material-role hypothesis, not a measurement of physical opacity.
    // Extension nodes retain all geometric/hair/clearance gates below and
    // must connect to the original high-head eligible band. No public caller
    // enables it. Extension-node centers on observed fixed material are
    // excluded; finer protection footprints still require the downstream
    // geometric constraints and final unchanged-pixel verification.
    #[cfg(test)]
    let shared_admission = outer_subject.is_some()
        && params
            .get("__experimentalSharedSubjectAdmission")
            .and_then(Value::as_bool)
            == Some(true);
    #[cfg(not(test))]
    let shared_admission = false;
    let admission_protect = shared_admission
        .then(|| params.get("protectMask").map(PreparedMask::new))
        .flatten();
    let admission_material: Vec<_> = if shared_admission {
        [
            semantics.get("ears"),
            semantics.get("accessories"),
            params.get("__sourceGarment"),
        ]
        .into_iter()
        .flatten()
        .map(PreparedMask::new)
        .collect()
    } else {
        Vec::new()
    };
    let opaque: Vec<bool> = head_value
        .get("data")?
        .as_array()?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_f64()
                .is_some_and(|v| v.is_finite() && v >= 0.98 * 255.0)
                || outer_subject.is_some_and(|data| {
                    data[i]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && (0.98 * 255.0..=255.0).contains(&v))
                })
        })
        .collect();
    // Class uncertainty at an internal skin/hair join is not automatically an
    // outer head edge. Use only low-head samples connected to the image edge;
    // do not fill their RGB or change the original head confidence.
    let mut exterior = vec![false; mask_width * mask_height];
    let mut exterior_queue = VecDeque::new();
    for i in 0..exterior.len() {
        let (x, y) = (i % mask_width, i / mask_width);
        if !opaque[i] && (x == 0 || y == 0 || x + 1 == mask_width || y + 1 == mask_height) {
            exterior[i] = true;
            exterior_queue.push_back(i);
        }
    }
    while let Some(i) = exterior_queue.pop_front() {
        for n in neighbors(i, mask_width, mask_height) {
            if !opaque[n] && !exterior[n] {
                exterior[n] = true;
                exterior_queue.push_back(n);
            }
        }
    }
    let clear_head_support = |x: usize, y: usize| {
        if x < clearance_x
            || y < clearance_y
            || x + clearance_x >= mask_width
            || y + clearance_y >= mask_height
        {
            return false;
        }
        (y - clearance_y..=y + clearance_y)
            .all(|yy| (x - clearance_x..=x + clearance_x).all(|xx| !exterior[yy * mask_width + xx]))
    };
    // New low-head material needs positive person evidence over its entire
    // bilinear footprint plus the original clearance, not merely at its node.
    // Unlike exterior flooding this also rejects enclosed background holes;
    // a high head-class false positive cannot fill a low-person sample here.
    let clear_subject_support = |x: usize, y: usize| {
        if x < clearance_x
            || y < clearance_y
            || x + clearance_x >= mask_width
            || y + clearance_y >= mask_height
        {
            return false;
        }
        outer_subject.is_some_and(|data| {
            (y - clearance_y..=y + clearance_y).all(|yy| {
                (x - clearance_x..=x + clearance_x).all(|xx| {
                    data[yy * mask_width + xx]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && (0.98 * 255.0..=255.0).contains(&v))
                })
            })
        })
    };
    let band = if shared_cheek {
        (frame.width() * 0.18).max(3.0)
    } else {
        (frame.width() * 0.055).clamp(3.0, 14.0)
    };
    let mut promote = vec![false; mask_width * mask_height];
    let mut extension = shared_admission.then(|| vec![false; mask_width * mask_height]);
    for y in 0..mask_height {
        for x in 0..mask_width {
            let index = y * mask_width + x;
            let p = (
                (x as f32 + 0.5) * width / mask_width as f32,
                (y as f32 + 0.5) * height / mask_height as f32,
            );
            let (u, v) = frame.coordinates(p);
            let normalized_y = (v - frame.top) / frame.height().max(1.0);
            let observed_head =
                head.coverage(p.0 / width, p.1 / height, source_width, source_height) >= 0.98;
            let possible_extension = shared_admission
                && clear_subject_support(x, y)
                && outer_subject.is_some_and(|data| {
                    data[index]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && (0.98 * 255.0..=255.0).contains(&v))
                })
                && !admission_protect.as_ref().is_some_and(|mask| {
                    mask.coverage(p.0 / width, p.1 / height, source_width, source_height) > 0.0
                })
                && !admission_material.iter().any(|mask| {
                    mask.coverage(p.0 / width, p.1 / height, source_width, source_height) >= 0.5
                });
            if !(0.02..=if shared_cheek { 0.985 } else { 0.57 }).contains(&normalized_y)
                || (u - (frame.left + frame.right) * 0.5).abs() < frame.width() * 0.06
                || (!observed_head && !possible_extension)
                || hair.coverage(p.0 / width, p.1 / height, source_width, source_height)
                    < if shared_cheek { 0.05 } else { 0.7 }
                || skin.coverage(p.0 / width, p.1 / height, source_width, source_height)
                    > if shared_cheek { 0.85 } else { 0.15 }
                || inside(p, polygon)
            {
                continue;
            }
            let edge_distance = polygon
                .iter()
                .enumerate()
                .map(|(i, &a)| closest(p, a, polygon[(i + 1) % polygon.len()]).0)
                .fold(f32::MAX, f32::min);
            if edge_distance <= band && (!shared_cheek || clear_head_support(x, y)) {
                promote[index] = observed_head;
                if let Some(extension) = extension.as_mut() {
                    extension[index] = !observed_head && possible_extension;
                }
            }
        }
    }
    if let Some(extension) = extension {
        // Existing seeds may be classified as eligible by the historical
        // head rule while downstream ownership keeps them fixed (for example
        // an observed accessory or protected pixel). They must not serve as
        // bridges that unlock a new low-head island. Keep the old `promote`
        // result intact; restrict only the queue used for extension BFS.
        let movable_seed = |index: usize| {
            let x = index % mask_width;
            let y = index / mask_width;
            let p = (
                (x as f32 + 0.5) * width / mask_width as f32,
                (y as f32 + 0.5) * height / mask_height as f32,
            );
            !admission_protect.as_ref().is_some_and(|mask| {
                mask.coverage(p.0 / width, p.1 / height, source_width, source_height) > 0.0
            }) && !admission_material.iter().any(|mask| {
                mask.coverage(p.0 / width, p.1 / height, source_width, source_height) >= 0.5
            })
        };
        let mut queue: VecDeque<_> = promote
            .iter()
            .enumerate()
            .filter_map(|(i, &seed)| (seed && movable_seed(i)).then_some(i))
            .collect();
        while let Some(i) = queue.pop_front() {
            for n in neighbors(i, mask_width, mask_height) {
                if extension[n] && !promote[n] {
                    promote[n] = true;
                    queue.push_back(n);
                }
            }
        }
    }
    if !promote.iter().any(|&value| value) {
        return None;
    }
    let mut prepared = params.clone();
    let masks = prepared.get_mut("semanticMasks")?.as_object_mut()?;
    let update_raster = |value: &mut Value, set: &[bool], fill: u8| {
        let data = value.get_mut("data")?.as_array_mut()?;
        for (index, item) in data.iter_mut().enumerate() {
            if set[index] {
                *item = Value::from(fill);
            }
        }
        Some(())
    };
    // Keep faceContourMask paired with faceContour. It is validated as an
    // observed silhouette before the experiment runs; the shared band is
    // carried by the connected skin evidence below instead of redefining the
    // contour polygon.
    if let Some(skin) = masks.get_mut("skin") {
        update_raster(skin, &promote, 255)?;
    }
    update_raster(masks.get_mut("hair")?, &promote, 0)?;
    prepared["__upperSharedPrepared"] = Value::Bool(true);
    prepared["__upperSharedCandidateApplied"] = Value::Bool(true);
    Some(prepared)
}

/// Prepare an upper-face width field from an observed skin/hair interface.
///
/// `foreheadWidth > 0` expands and `< 0` narrows the upper face, matching the
/// existing face slider contract. `temples > 0` narrows and `< 0` expands.
/// Only controls with opaque head material on both sides and hair outside /
/// skin inside are accepted; no guessed oval boundary is introduced.
#[cfg(test)]
pub(super) fn prepare(
    source: &RgbaImage,
    params: &Value,
    polygon: &[Point],
    frame: Frame,
) -> Option<UpperWidth> {
    let values = params.get("adjustments").unwrap_or(params);
    let forehead_width = percent(values, "foreheadWidth");
    let temples = percent(values, "temples");
    if forehead_width.abs() < 1e-6 && temples.abs() < 1e-6 {
        return None;
    }
    if polygon.len() < 3 {
        return None;
    }
    let semantics = params.get("semanticMasks")?;
    let raster = |key: &str| {
        semantics
            .get(key)
            .filter(|m| m.get("shape").and_then(Value::as_str) == Some("raster"))
            .map(PreparedMask::new)
    };
    let head = raster("face")?;
    let skin = raster("faceContourMask")?;
    let hair = raster("hair")?;
    let (w, h) = (source.width as f32, source.height as f32);
    let read = |mask: &PreparedMask, p: Point| {
        mask.coverage(p.0 / w, p.1 / h, source.width, source.height)
    };
    let radius = (frame.width() * 0.43).max(12.0);
    let pitch = (frame.width() / 64.0).max(1.0);
    let mut controls = Vec::new();
    let mut requests = Vec::new();
    let center_u = (frame.left + frame.right) * 0.5;

    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let edge_length = sub(b, a).0.hypot(sub(b, a).1);
        let steps = (edge_length / pitch).ceil().max(1.0) as usize;
        for step in 0..steps {
            let p = add(a, scale(sub(b, a), step as f32 / steps as f32));
            let (u, v) = frame.coordinates(p);
            let y = (v - frame.top) / frame.height().max(1.0);
            // Forehead and temples are upper-face controls. Avoid the centre
            // crown where a lateral width edit has no observable side.
            if !(0.02..=0.57).contains(&y) || (u - center_u).abs() < frame.width() * 0.08 {
                continue;
            }
            let Some(normal) = normal_at(polygon, index, p, frame) else {
                continue;
            };
            let mut hair_evidence = false;
            let mut skin_evidence = false;
            let max_probe = (frame.width() * 0.055).ceil().max(3.0) as usize;
            for distance in 1..=max_probe {
                let outward = add(p, scale(normal, distance as f32));
                let inward = add(p, scale(normal, -(distance as f32)));
                if read(&head, outward) < 0.98 || read(&head, inward) < 0.98 {
                    break;
                }
                hair_evidence |= read(&hair, outward) >= 0.7;
                skin_evidence |= read(&skin, inward) >= 0.7;
                if hair_evidence && skin_evidence {
                    break;
                }
            }
            if !hair_evidence || !skin_evidence {
                continue;
            }
            let forehead = profile(y, 0.0, 0.16, 0.36);
            let temple = profile(y, 0.10, 0.28, 0.46);
            // Preserve the established slider signs: foreheadWidth positive
            // widens, while temples positive narrows.
            let amount =
                frame.width() * (forehead_width * 0.060 * forehead - temples * 0.050 * temple);
            if amount.abs() <= 0.01 {
                continue;
            }
            controls.push(p);
            requests.push(scale(normal, amount));
        }
    }
    if controls.len() < 4 {
        return None;
    }

    let mut left = w;
    let mut top = h;
    let mut right = 0.0_f32;
    let mut bottom = 0.0_f32;
    for &p in &controls {
        left = left.min(p.0 - radius - 2.0);
        top = top.min(p.1 - radius - 2.0);
        right = right.max(p.0 + radius + 2.0);
        bottom = bottom.max(p.1 + radius + 2.0);
    }
    let (left, top) = (
        left.floor().max(0.0) as usize,
        top.floor().max(0.0) as usize,
    );
    let (right, bottom) = (right.ceil().min(w) as usize, bottom.ceil().min(h) as usize);
    if right <= left + 2 || bottom <= top + 2 {
        return None;
    }
    let (width, height) = (right - left, bottom - top);
    let count = width * height;
    let region = Region::read(params.get("region"));
    let selection = params.get("mask").map(PreparedMask::new);
    let protect = params.get("protectMask").map(PreparedMask::new);
    let fixed: Vec<_> = [
        "ears",
        "accessories",
        "leftEye",
        "rightEye",
        "leftBrow",
        "rightBrow",
        "nose",
        "noseBridge",
        "noseWings",
        "lips",
        "teeth",
    ]
    .iter()
    .filter_map(|key| semantics.get(key))
    .map(PreparedMask::new)
    .collect();
    let position = |i: usize| {
        (
            left as f32 + (i % width) as f32 + 0.5,
            top as f32 + (i / width) as f32 + 0.5,
        )
    };
    let mut forbidden = vec![false; count];
    for i in 0..count {
        let p = position(i);
        let (_, v) = frame.coordinates(p);
        let y = (v - frame.top) / frame.height().max(1.0);
        let inside_head = read(&head, p) >= 0.98;
        let selected = selection.as_ref().map(|m| read(m, p)).unwrap_or(1.0);
        let protected = protect.as_ref().map(|m| read(m, p)).unwrap_or(0.0);
        forbidden[i] = !inside_head
            || !(0.0..=0.62).contains(&y)
            || !region.contains(p.0 / w, p.1 / h)
            || selected <= 1e-6
            || protected >= 1.0 - 1e-6
            || fixed.iter().any(|m| read(m, p) > 1e-6)
            || i % width == 0
            || i % width == width - 1
            || i / width == 0
            || i / width == height - 1;
    }
    let mut owned = vec![false; count];
    let mut queue = VecDeque::new();
    for &p in &controls {
        let x = (p.0 - left as f32 - 0.5)
            .round()
            .clamp(0.0, (width - 1) as f32) as usize;
        let y = (p.1 - top as f32 - 0.5)
            .round()
            .clamp(0.0, (height - 1) as f32) as usize;
        let i = y * width + x;
        if !forbidden[i] && !owned[i] {
            owned[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for n in neighbors(i, width, height) {
            if !forbidden[n] && !owned[n] {
                owned[n] = true;
                queue.push_back(n);
            }
        }
    }
    // Distance to every forbidden pixel gives a conservative clearance for
    // both source sampling and the scalar-field transition. Unowned islands
    // are treated as fixed context even when they are close to the controls.
    let mut distance = vec![u16::MAX; count];
    let mut distance_queue = VecDeque::new();
    for i in 0..count {
        if forbidden[i] || !owned[i] {
            distance[i] = 0;
            distance_queue.push_back(i);
        }
    }
    while let Some(i) = distance_queue.pop_front() {
        let next = distance[i].saturating_add(1);
        for n in neighbors(i, width, height) {
            if next < distance[n] {
                distance[n] = next;
                distance_queue.push_back(n);
            }
        }
    }
    let kernel = positive_kernel::PositiveKernel::new(controls.clone(), radius);
    let requested_caps: Vec<f32> = (0..count)
        .map(|i| {
            if distance[i] == 0 || distance[i] as f32 / std::f32::consts::SQRT_2 <= 4.0 {
                return 0.0;
            }
            let p = position(i);
            let value = kernel.evaluate(&requests, p);
            let support = kernel.evaluate(&vec![(1.0, 0.0); controls.len()], p).0;
            if support <= 1e-6 {
                0.0
            } else {
                // The positive kernel is normalized against an explicit zero
                // background, so this remains inside the requested convex
                // hull and tapers continuously at support boundaries.
                let normalized = scale(value, 1.0 / support.max(1e-12));
                normalized.0.hypot(normalized.1)
            }
        })
        .collect();
    let scalar = scalar_field::bounded_scalar_field(width, height, &requested_caps, 0.45);
    let vectors: Vec<Point> = (0..count)
        .map(|i| {
            if scalar[i] <= 1e-6 {
                return (0.0, 0.0);
            }
            let p = position(i);
            let raw = kernel.evaluate(&requests, p);
            let magnitude = raw.0.hypot(raw.1).max(1e-6);
            scale(raw, scalar[i] / magnitude)
        })
        .collect();
    let velocity = flow::VelocityGrid::new(position(0), width, height, vectors);
    Some(UpperWidth {
        velocity,
        left,
        top,
        width,
        height,
        editable: owned
            .into_iter()
            .zip(scalar)
            .map(|(owned, amount)| owned && amount > 1e-6)
            .collect(),
    })
}

#[cfg(test)]
impl UpperWidth {
    /// Render only the connected shared domain. The source image is sampled
    /// directly, so fixed context and all pixels outside the candidate ROI are
    /// preserved byte-for-byte.
    pub(super) fn render(&self, source: &RgbaImage) -> RgbaImage {
        let mut rendered = source.clone();
        let full_alpha = vec![1.0; source.width as usize * source.height as usize];
        for i in 0..self.width * self.height {
            if !self.editable[i] {
                continue;
            }
            let p = (
                self.left as f32 + (i % self.width) as f32 + 0.5,
                self.top as f32 + (i / self.width) as f32 + 0.5,
            );
            let movement = flow::inverse(&self.velocity, p);
            if movement.0.abs() + movement.1.abs() < 1e-5 {
                continue;
            }
            let (pixel, _) = sample(
                source,
                &full_alpha,
                p.0 + movement.0 - 0.5,
                p.1 + movement.1 - 0.5,
            );
            let j =
                ((self.top + i / self.width) * source.width as usize + self.left + i % self.width)
                    * 4;
            for c in 0..3 {
                rendered.pixels[j + c] = byte(pixel[c] / pixel[3].max(1e-6));
            }
            rendered.pixels[j + 3] = byte(pixel[3] * 255.0);
        }
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 160;
    const H: usize = 240;

    fn raster(data: Vec<u8>) -> Value {
        json!({"shape":"raster","width":W,"height":H,"data":data})
    }

    fn polygon(points: &[Point]) -> Value {
        json!({"shape":"polygon","points":points.iter().map(|&(x,y)|
            json!({"x":x/W as f32,"y":y/H as f32})).collect::<Vec<_>>()})
    }

    fn alpha(p: Point, polygon: &[Point]) -> f32 {
        let mut inside = false;
        for i in 0..polygon.len() {
            let a = polygon[i];
            let b = polygon[(i + 1) % polygon.len()];
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
        }
        if inside {
            1.0
        } else {
            0.0
        }
    }

    fn fixture() -> (RgbaImage, Vec<Point>, Frame, Value) {
        let head: Vec<Point> = (0..64)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::TAU / 64.0;
                (80.0 + 62.0 * angle.cos(), 101.0 + 89.0 * angle.sin())
            })
            .collect();
        let face = vec![
            (44.0, 60.0),
            (54.0, 52.0),
            (106.0, 52.0),
            (116.0, 60.0),
            (125.0, 92.0),
            (123.0, 137.0),
            (107.0, 168.0),
            (80.0, 181.0),
            (53.0, 168.0),
            (37.0, 137.0),
            (35.0, 92.0),
        ];
        let mut head_data = Vec::with_capacity(W * H);
        let mut skin_data = Vec::with_capacity(W * H);
        let mut hair_data = Vec::with_capacity(W * H);
        let mut pixels = Vec::with_capacity(W * H * 4);
        for y in 0..H {
            for x in 0..W {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let h = alpha(p, &head);
                let s = alpha(p, &face);
                let hair = (h - s).max(0.0);
                head_data.push(byte(255.0 * h));
                skin_data.push(byte(255.0 * s));
                hair_data.push(byte(255.0 * hair));
                let mut rgb = if h > 0.0 {
                    [30.0, 22.0, y as f32]
                } else {
                    [15.0, 35.0, 170.0]
                };
                if s > 0.0 {
                    rgb = [210.0, 150.0, y as f32];
                }
                pixels.extend([byte(rgb[0]), byte(rgb[1]), byte(rgb[2]), 255]);
            }
        }
        let mut params = json!({
            "region":{"x":0,"y":0,"width":1,"height":1},
            "semanticMasks":{
                "face":raster(head_data),
                "faceContourMask":raster(skin_data.clone()),
                "skin":raster(skin_data),
                "hair":raster(hair_data),
                "faceContour":polygon(&face),
                "leftEye":{"shape":"polygon","points":[]},
                "rightEye":{"shape":"polygon","points":[]}
            },
            "adjustments":{"foreheadWidth":0,"temples":0}
        });
        // An explicit empty feature list is legal for this isolated field and
        // keeps the fixture focused on the material interface.
        params["semanticMasks"]["faceOutline"] = polygon(&head);
        let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
        let frame = Frame::new(&face, None, W as f32, H as f32);
        (source, face, frame, params)
    }

    fn skin_span(image: &RgbaImage, y: usize) -> (usize, usize) {
        let xs: Vec<_> = (0..W)
            .filter(|&x| image.pixels[(y * W + x) * 4] > 130)
            .collect();
        (*xs.first().unwrap(), *xs.last().unwrap())
    }

    #[test]
    fn upper_width_moves_both_directions_and_keeps_outer_context_exact() {
        let (source, face, frame, mut params) = fixture();
        for (key, positive_widens) in [("foreheadWidth", true), ("temples", false)] {
            let row = if key == "foreheadWidth" { 68 } else { 94 };
            let before = skin_span(&source, row);
            for amount in [-100.0, 100.0] {
                params["adjustments"] = json!({key: amount});
                let field = prepare(&source, &params, &face, frame)
                    .expect("synthetic skin/hair interface must produce controls");
                let output = field.render(&source);
                let after = skin_span(&output, row);
                let widens = after.0 < before.0 && after.1 > before.1;
                assert_eq!(
                    widens,
                    (amount > 0.0) == positive_widens,
                    "{key} {amount}: {before:?} -> {after:?}"
                );
                let head = (0..64)
                    .map(|i| {
                        let angle = i as f32 * std::f32::consts::TAU / 64.0;
                        (80.0 + 62.0 * angle.cos(), 101.0 + 89.0 * angle.sin())
                    })
                    .collect::<Vec<_>>();
                for y in 0..H {
                    for x in 0..W {
                        let p = (x as f32 + 0.5, y as f32 + 0.5);
                        if alpha(p, &head) <= 0.0 {
                            assert_eq!(
                                &output.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4],
                                &source.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4],
                                "outer context moved at {x},{y}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn upper_width_field_has_no_negative_local_jacobian() {
        let (source, face, frame, mut params) = fixture();
        params["adjustments"] = json!({"foreheadWidth":100});
        let field = prepare(&source, &params, &face, frame).unwrap();
        let eps = 0.5;
        for y in (field.top + 8..field.top + field.height - 8).step_by(4) {
            for x in (field.left + 8..field.left + field.width - 8).step_by(4) {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let map = |q: Point| add(q, flow::inverse(&field.velocity, q));
                let px = map((p.0 + eps, p.1));
                let mx = map((p.0 - eps, p.1));
                let py = map((p.0, p.1 + eps));
                let my = map((p.0, p.1 - eps));
                let dx = scale(sub(px, mx), 1.0 / (2.0 * eps));
                let dy = scale(sub(py, my), 1.0 / (2.0 * eps));
                let determinant = dx.0 * dy.1 - dx.1 * dy.0;
                assert!(
                    determinant > 0.10,
                    "folded field at {p:?}: det={determinant}"
                );
            }
        }
        assert_eq!(field.render(&source).pixels.len(), source.pixels.len());
    }

    #[test]
    fn interface_promotion_is_bounded_and_idempotent() {
        let (source, face, frame, mut params) = fixture();
        params["adjustments"] = json!({"foreheadWidth":100});
        let before_skin: usize = params["semanticMasks"]["skin"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|value| value.as_u64().unwrap_or(0) > 127)
            .count();
        let before_hair: usize = params["semanticMasks"]["hair"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|value| value.as_u64().unwrap_or(0) > 127)
            .count();
        let prepared = promote_interface(&params, &face, frame, W as u32, H as u32)
            .expect("synthetic interface must have an eligible band");
        let after_skin: usize = prepared["semanticMasks"]["skin"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|value| value.as_u64().unwrap_or(0) > 127)
            .count();
        let after_hair: usize = prepared["semanticMasks"]["hair"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|value| value.as_u64().unwrap_or(0) > 127)
            .count();
        assert!(after_skin > before_skin, "no skin interface was promoted");
        assert_eq!(before_hair - after_hair, after_skin - before_skin);
        assert_eq!(prepared["__upperSharedPrepared"], json!(true));
        assert!(promote_interface(&prepared, &face, frame, W as u32, H as u32).is_none());
        // Promotion is metadata-only; the source image is never rewritten.
        assert_eq!(source.pixels.len(), W * H * 4);
    }

    #[test]
    fn shared_cheek_keeps_outer_head_sampling_border_unmodified() {
        let (_, face, frame, mut params) = fixture();
        params["adjustments"] = json!({"faceWidth":30});
        params["experimentalFaceProfile"] = json!("shared-cheek-v1");
        let prepared = promote_interface(&params, &face, frame, W as u32, H as u32)
            .expect("interior hair must still be eligible");
        fn mask<'a>(p: &'a Value, key: &str) -> PreparedMask<'a> {
            PreparedMask::new(&p["semanticMasks"][key])
        }
        let head = mask(&params, "face");
        let old_hair = mask(&params, "hair");
        let new_hair = mask(&prepared, "hair");
        let old_skin = mask(&params, "skin");
        let new_skin = mask(&prepared, "skin");
        let read = |m: &PreparedMask, x: f32, y: f32| {
            m.coverage(x / W as f32, y / H as f32, W as u32, H as u32)
        };
        let mut checked = 0;
        let mut changed_interior = 0;
        for y in 1..H - 1 {
            for x in 1..W - 1 {
                // Off-lattice probes catch a promoted mask texel leaking
                // through bilinear interpolation even when its centre is safe.
                for dx in [0.25, 0.5, 0.75] {
                    let p = (x as f32 + dx, y as f32 + 0.5);
                    if read(&old_hair, p.0, p.1) <= 0.0 {
                        continue;
                    }
                    let border = [-1.0, 0.0, 1.0].into_iter().any(|ox| {
                        [-1.0, 0.0, 1.0]
                            .into_iter()
                            .any(|oy| read(&head, p.0 + ox, p.1 + oy) < 0.98)
                    });
                    if border {
                        checked += 1;
                        assert_eq!(
                            read(&old_hair, p.0, p.1),
                            read(&new_hair, p.0, p.1),
                            "shared motion leaked into outer hair support at {p:?}"
                        );
                        assert_eq!(read(&old_skin, p.0, p.1), read(&new_skin, p.0, p.1));
                    } else if read(&old_hair, p.0, p.1) != read(&new_hair, p.0, p.1) {
                        changed_interior += 1;
                    }
                }
            }
        }
        assert!(checked > 100, "outer hair coverage must not be vacuous");
        assert!(
            changed_interior > 100,
            "freezing the whole edit is not a fix"
        );
    }

    #[test]
    fn continuous_profile_inherits_shared_interface_and_keeps_source_locks() {
        let (mut source, face, frame, mut params) = fixture();
        // Supply an actually connected neck, not just a present raster. The
        // normal renderer requires observed skin/neck outside the facial
        // contour before it enables the joint field.
        let mut neck = vec![0_u8; W * H];
        for y in 170..H {
            for x in 70..90 {
                let i = y * W + x;
                neck[i] = 255;
                params["semanticMasks"]["skin"]["data"][i] = json!(255);
                params["semanticMasks"]["hair"]["data"][i] = json!(0);
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[210, 150, y as u8, 255]);
            }
        }
        params["semanticMasks"]["neck"] = raster(neck.clone());
        let head: Vec<_> = params["semanticMasks"]["face"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() > 0)
            .collect();
        let occupied: Vec<_> = head
            .iter()
            .zip(&neck)
            .map(|(&head, &neck)| head || neck > 0)
            .collect();
        params["subjectMask"] = raster(occupied.iter().map(|&p| if p { 255 } else { 0 }).collect());
        let hair: Vec<_> = params["semanticMasks"]["hair"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() > 0)
            .collect();
        // Observable source hair texture makes both shared motion and its
        // explicit protection distinguishable from an accidental identity.
        for i in 0..W * H {
            if hair[i] {
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                    30 + ((i % W) % 4) as u8 * 8,
                    22 + ((i / W) % 5) as u8,
                    50 + ((i / W) / 2) as u8,
                    255,
                ]);
            }
        }
        let mut protection = vec![0_u8; W * H];
        for y in 104..110 {
            for x in 129..133 {
                assert!(hair[y * W + x]);
                protection[y * W + x] = 255;
            }
        }
        params["protectMask"] = raster(protection.clone());
        params["experimentalFaceProfile"] = json!("source-domain-continuous-v1");
        // Exercise the entry validator too: omitting this profile from its
        // source-interface capability list must fail rather than be hidden by
        // the simpler helper-only promotion check below.
        params["__sourceMotionInterface"] = json!({
            "mask": params["semanticMasks"]["faceContourMask"],
            "contour": params["semanticMasks"]["faceContour"]
        });
        struct TraceDirectory(std::path::PathBuf);
        impl Drop for TraceDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let trace = TraceDirectory(std::env::temp_dir().join(format!(
            "continuous-profile-shared-interface-{}-{nonce}",
            std::process::id()
        )));
        for amount in [-30, 30] {
            params["adjustments"] = json!({"faceWidth":amount});
            let prepared = promote_interface(&params, &face, frame, W as u32, H as u32)
                .expect("continuous profile must admit the existing shared interface");
            let mut reference_params = params.clone();
            reference_params["experimentalFaceProfile"] = json!("shared-cheek-v1");
            let reference = promote_interface(&reference_params, &face, frame, W as u32, H as u32)
                .expect("existing shared profile supplies the comparison domain");
            assert_eq!(
                prepared["semanticMasks"], reference["semanticMasks"],
                "profile wiring must inherit the same source domain, not a new band rule"
            );
            assert_eq!(prepared["protectMask"], params["protectMask"]);
            for key in ["face", "faceOutline", "faceContourMask", "faceContour"] {
                assert_eq!(prepared["semanticMasks"][key], params["semanticMasks"][key]);
            }
            assert!(promote_interface(&prepared, &face, frame, W as u32, H as u32).is_none());

            let explicit = super::adjust(&source, &prepared).unwrap();
            let mut traced_params = params.clone();
            let trace_path = trace.0.join(amount.to_string());
            traced_params["__faceFieldTrace"] = json!(trace_path);
            let automatic = super::adjust(&source, &traced_params).unwrap();
            let state: Value = serde_json::from_slice(
                &std::fs::read(trace_path.join("source-domain-state.json")).unwrap(),
            )
            .unwrap();
            for key in [
                "sourceDomainProfile",
                "jointNeck",
                "stableSourceLayers",
                "continuous",
            ] {
                assert_eq!(
                    state[key],
                    json!(true),
                    "fixture bypassed the continuous joint renderer: {state}"
                );
            }
            assert_eq!(automatic, explicit,
                "continuous face.adjust bypassed or repeated shared-interface preparation at {amount}");
            let mut unprotected_params = params.clone();
            unprotected_params
                .as_object_mut()
                .unwrap()
                .remove("protectMask");
            let unprotected = super::adjust(&source, &unprotected_params).unwrap();
            let would_change = (0..W * H)
                .filter(|&i| {
                    protection[i] > 0
                        && unprotected.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4]
                })
                .count();
            assert!(
                would_change > 0,
                "protection fixture was not exercised at {amount}"
            );
            let mut changed_shared_hair = 0;
            let mut checked_outer_hair = 0;
            for y in 0..H {
                for x in 0..W {
                    let i = y * W + x;
                    let changed =
                        automatic.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4];
                    // Source-label oracle: outer hair stays exact. The
                    // connected neck may legitimately cover adjacent old
                    // background when widening; only the upper-head exterior
                    // and distant lower gutters are fixed background here.
                    // These regions come from fixture construction, never
                    // the candidate's changed/frozen classifications.
                    let exterior_neighbor = (-2_isize..=2).any(|dy| {
                        (-2_isize..=2).any(|dx| {
                            let xx = x as isize + dx;
                            let yy = y as isize + dy;
                            xx < 0
                                || yy < 0
                                || xx >= W as isize
                                || yy >= H as isize
                                || !occupied[yy as usize * W + xx as usize]
                        })
                    });
                    let fixed_background = !occupied[i] && (y < 170 || x < 60 || x >= 100);
                    if protection[i] > 0 || fixed_background || (hair[i] && exterior_neighbor) {
                        assert!(
                            !changed,
                            "source protection / outer head changed at {x},{y}, amount={amount}"
                        );
                        checked_outer_hair += usize::from(hair[i] && exterior_neighbor);
                    }
                    changed_shared_hair += usize::from(hair[i] && changed);
                }
            }
            assert!(checked_outer_hair > 100, "outer contour check was empty");
            assert!(
                changed_shared_hair > 0,
                "shared hair was silently left frozen at {amount}"
            );
            eprintln!("continuous shared interface {amount}: changed hair={changed_shared_hair}, outer hair checked={checked_outer_hair}, protection counterfactual={would_change}");
        }
    }

    #[test]
    fn interface_promotion_rejects_misaligned_or_missing_connected_skin() {
        let (_, face, frame, mut params) = fixture();
        params["adjustments"] = json!({"foreheadWidth":30});
        let original = params["semanticMasks"]["skin"].clone();
        // Same length with a different layout is still a different geometry.
        params["semanticMasks"]["skin"]["width"] = json!(W * 2);
        params["semanticMasks"]["skin"]["height"] = json!(H / 2);
        assert!(promote_interface(&params, &face, frame, W as u32, H as u32).is_none());
        params["semanticMasks"]["skin"] = original;
        params["semanticMasks"]["skin"]["data"]
            .as_array_mut()
            .unwrap()
            .push(json!(255));
        assert!(promote_interface(&params, &face, frame, W as u32, H as u32).is_none());
        params["semanticMasks"]
            .as_object_mut()
            .unwrap()
            .remove("skin");
        assert!(promote_interface(&params, &face, frame, W as u32, H as u32).is_none());
    }

    #[test]
    fn real_holdouts_accept_the_isolated_upper_shared_hook() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap();
        let fixtures = [
            "portrait-woman",
            "portrait-hair",
            "portrait-makeup",
            "portrait-roll",
        ];
        let fresh_replay = std::env::var_os("UPPER_SHARED_REPLAY").is_some();
        for name in fixtures {
            let source_path = if fresh_replay {
                root.join("test/images").join(format!("{name}.jpg"))
            } else {
                root.join("test/results/face-core-review")
                    .join(format!("{name}-source.png"))
            };
            let graph_path = if fresh_replay {
                std::path::PathBuf::from("/tmp").join(format!("{name}-upper-shared.json"))
            } else {
                root.join("test/results/face-core-review")
                    .join(format!("{name}-faceWidth-60.json"))
            };
            if !source_path.exists() || !graph_path.exists() {
                continue;
            }
            let decoded = image::load_from_memory(&std::fs::read(&source_path).unwrap())
                .unwrap()
                .to_rgba8();
            let source =
                RgbaImage::new(decoded.width(), decoded.height(), decoded.into_raw()).unwrap();
            let graph: Value =
                serde_json::from_slice(&std::fs::read(&graph_path).unwrap()).unwrap();
            let mut params = graph["operations"][0]["params"].clone();
            if !fresh_replay {
                params["adjustments"] = json!({"foreheadWidth":-60});
                params["experimentalFaceProfile"] = json!("upper-shared-v1");
            }
            if fresh_replay {
                let output = super::adjust(&source, &params)
                    .unwrap_or_else(|error| panic!("{name} fresh upper hook failed: {error}"));
                let changed = source
                    .pixels
                    .chunks_exact(4)
                    .zip(output.pixels.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count();
                let output_path = std::path::PathBuf::from("/tmp")
                    .join(format!("{name}-upper-shared-output.png"));
                let image = image::RgbaImage::from_raw(output.width, output.height, output.pixels)
                    .expect("valid output buffer");
                image.save(&output_path).expect("write fresh replay output");
                eprintln!("upper-shared fresh {name}: changed={changed}, output={output_path:?}");
                assert!(
                    changed > 0,
                    "{name} fresh upper hook produced identity output"
                );
                continue;
            }
            let width = source.width as f32;
            let height = source.height as f32;
            let polygon = super::points(&params["semanticMasks"]["faceContour"], width, height);
            let frame = Frame::new(&polygon, Some(&params["semanticMasks"]), width, height);
            let prepared = promote_interface(&params, &polygon, frame, source.width, source.height);
            let promoted = prepared
                .as_ref()
                .map(|value| {
                    value["semanticMasks"]["hair"]["data"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(params["semanticMasks"]["hair"]["data"].as_array().unwrap())
                        .filter(|(after, before)| {
                            before.as_u64().unwrap_or(0) > 127 && after.as_u64().unwrap_or(0) == 0
                        })
                        .count()
                })
                .unwrap_or(0);
            eprintln!("upper-shared {name}: promoted={promoted}");
            assert!(
                promoted > 0,
                "{name} has no eligible observed upper hair band"
            );
        }
    }
}

#[cfg(test)]
mod outer_occupancy_tests {
    use super::*;
    fn fixture() -> (Value, Vec<Point>, Frame) {
        let (w, h) = (40usize, 40usize);
        let raster = |data: Vec<u8>| json!({"shape":"raster","width":w,"height":h,"data":data});
        let mut head = vec![0; w * h];
        let mut subject = head.clone();
        let mut hair = head.clone();
        let mut skin = head.clone();
        let mut contour = head.clone();
        for y in 2..38 {
            for x in 2..38 {
                subject[y * w + x] = 255;
            }
        }
        for y in 4..34 {
            for x in 5..35 {
                head[y * w + x] = 255;
                hair[y * w + x] = 255;
            }
        }
        for y in 6..32 {
            for x in 10..25 {
                skin[y * w + x] = 255;
                contour[y * w + x] = 255;
                hair[y * w + x] = 0;
            }
        }
        // An internal head-class channel reaches the bottom image border.
        // The separate aggregate person occupancy continues across it.
        for y in 12..40 {
            head[y * w + 28] = 0;
        }
        let poly = vec![(10., 6.), (25., 6.), (25., 32.), (10., 32.)];
        let frame = Frame::new(&poly, None, 40., 40.);
        let params = json!({"adjustments":{"faceWidth":60},"experimentalFaceProfile":"shared-cheek-ear-periodic-v1","__experimentalOuterSubjectOccupancy":true,"subjectMask":raster(subject),"semanticMasks":{"face":raster(head),"faceContourMask":raster(contour),"hair":raster(hair),"skin":raster(skin),"garment":{"shape":"rect"}},"protectMask":{"shape":"rect","region":{"x":0.7,"y":0.2,"width":0.05,"height":0.6}}});
        (params, poly, frame)
    }
    fn promoted(p: &Value, poly: &[Point], frame: Frame, x: usize, y: usize) -> bool {
        promote_interface(p, poly, frame, 40, 40).is_some_and(|q| {
            q["semanticMasks"]["hair"]["data"][y * 40 + x] == 0
                && p["semanticMasks"]["hair"]["data"][y * 40 + x] != 0
        })
    }
    #[test]
    fn internal_head_class_channel_does_not_define_person_exterior() {
        let (mut p, poly, f) = fixture();
        assert!(promoted(&p, &poly, f, 26, 20));
        p["__experimentalOuterSubjectOccupancy"] = json!(false);
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn real_background_gap_remains_blocked() {
        let (mut p, poly, f) = fixture();
        for y in 12..40 {
            p["subjectMask"]["data"][y * 40 + 28] = json!(0);
        }
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn outer_subject_boundary_keeps_full_original_clearance() {
        let (mut p, poly, f) = fixture();
        for y in 0..40 {
            for x in 28..40 {
                p["subjectMask"]["data"][y * 40 + x] = json!(0);
                p["semanticMasks"]["face"]["data"][y * 40 + x] = json!(0);
            }
        }
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn subject_occupancy_cannot_admit_non_head_material() {
        let (mut p, poly, f) = fixture();
        p["semanticMasks"]["face"]["data"][20 * 40 + 26] = json!(200);
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn missing_or_misaligned_subject_falls_back() {
        let (mut p, poly, f) = fixture();
        p.as_object_mut().unwrap().remove("subjectMask");
        assert!(!promoted(&p, &poly, f, 26, 20));
        let (mut p, poly, f) = fixture();
        p["subjectMask"]["width"] = json!(1);
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn preparation_does_not_redefine_protection_garment_or_geometry() {
        let (p, poly, f) = fixture();
        let out = promote_interface(&p, &poly, f, 40, 40).unwrap();
        for key in ["protectMask", "subjectMask", "adjustments"] {
            assert_eq!(p[key], out[key]);
        }
        for key in ["garment", "face", "faceContourMask"] {
            assert_eq!(p["semanticMasks"][key], out["semanticMasks"][key]);
        }
        // All changed nodes still satisfy the original head predicate.
        for i in 0..1600 {
            if p["semanticMasks"]["hair"]["data"][i] != out["semanticMasks"]["hair"]["data"][i] {
                assert_eq!(p["semanticMasks"]["face"]["data"][i], 255);
            }
        }
    }
    #[test]
    fn non_shared_cheek_ignores_candidate_opt_in() {
        let (mut p, poly, f) = fixture();
        p["experimentalFaceProfile"] = json!("upper-shared-v1");
        p["adjustments"] = json!({"foreheadWidth":60});
        let on = promote_interface(&p, &poly, f, 40, 40);
        p["__experimentalOuterSubjectOccupancy"] = json!(false);
        let off = promote_interface(&p, &poly, f, 40, 40);
        // The flag itself is retained, so compare only altered semantic masks.
        assert_eq!(
            on.map(|v| v["semanticMasks"].clone()),
            off.map(|v| v["semanticMasks"].clone())
        );
    }
    fn shared_admission_fixture() -> (Value, Vec<Point>, Frame) {
        let (mut p, poly, frame) = fixture();
        p["__experimentalSharedSubjectAdmission"] = json!(true);
        p["semanticMasks"]["face"]["data"][20 * 40 + 26] = json!(200);
        p["semanticMasks"]["hair"]["data"][20 * 40 + 26] = json!(160);
        (p, poly, frame)
    }
    #[test]
    fn connected_internal_class_uncertainty_needs_both_explicit_opt_ins() {
        let (mut p, poly, f) = shared_admission_fixture();
        assert!(promoted(&p, &poly, f, 26, 20));
        p["__experimentalSharedSubjectAdmission"] = json!(false);
        assert!(!promoted(&p, &poly, f, 26, 20));
        p["__experimentalSharedSubjectAdmission"] = json!(true);
        p["__experimentalOuterSubjectOccupancy"] = json!(false);
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn isolated_low_head_hair_island_is_not_new_moving_material() {
        let (mut p, poly, f) = shared_admission_fixture();
        for (x, y) in [(25, 20), (27, 20), (26, 19), (26, 21)] {
            p["semanticMasks"]["hair"]["data"][y * 40 + x] = json!(0);
        }
        assert!(!promoted(&p, &poly, f, 26, 20));
        assert!(promoted(&p, &poly, f, 26, 18));
    }
    #[test]
    fn connected_extension_cannot_cross_missing_hair_or_weak_person_evidence() {
        let (mut p, poly, f) = shared_admission_fixture();
        p["subjectMask"]["data"][20 * 40 + 26] = json!(200);
        assert!(!promoted(&p, &poly, f, 26, 20));
        p["subjectMask"]["data"][20 * 40 + 26] = json!(255);
        p["semanticMasks"]["hair"]["data"][20 * 40 + 26] = json!(0);
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn connected_extension_does_not_erase_real_person_exterior_clearance() {
        let (mut p, poly, f) = shared_admission_fixture();
        for y in 12..40 {
            p["subjectMask"]["data"][y * 40 + 28] = json!(0);
        }
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn connected_extension_does_not_expand_geometric_edit_support() {
        let (mut p, poly, f) = shared_admission_fixture();
        for (x, y) in [(24, 20), (32, 20), (26, 33)] {
            p["semanticMasks"]["face"]["data"][y * 40 + x] = json!(200);
            p["semanticMasks"]["hair"]["data"][y * 40 + x] = json!(160);
            assert!(!promoted(&p, &poly, f, x, y));
        }
    }
    #[test]
    fn protected_old_seeds_cannot_bridge_to_new_extension_nodes() {
        let (mut p, poly, f) = shared_admission_fixture();
        let mut protected = vec![255u8; 40 * 40];
        protected[20 * 40 + 26] = 0;
        p["protectMask"] = json!({"shape":"raster","width":40,"height":40,"data":protected});
        assert!(!promoted(&p, &poly, f, 26, 20));
    }
    #[test]
    fn connected_extension_respects_observed_protection_and_material_identity() {
        let (p, poly, f) = shared_admission_fixture();
        for key in ["protectMask", "__sourceGarment", "accessories", "ears"] {
            let mut q = p.clone();
            let protected =
                json!({"shape":"rect","region":{"x":0.65,"y":0.5,"width":0.025,"height":0.025}});
            if ["accessories", "ears"].contains(&key) {
                q["semanticMasks"][key] = protected;
            } else {
                q[key] = protected;
            }
            assert!(!promoted(&q, &poly, f, 26, 20), "{key}");
            assert!(
                promoted(&q, &poly, f, 26, 18),
                "original eligible material stays unchanged for {key}"
            );
        }
    }

    #[test]
    fn connected_extension_rejects_enclosed_low_subject_holes_in_its_support() {
        for false_positive_head in [false, true] {
            let (mut p, poly, f) = shared_admission_fixture();
            // This one-texel background hole is enclosed by occupied person
            // samples, so exterior flood-fill cannot reach it. Head confidence
            // must not override the hole, even when it is a false positive.
            p["subjectMask"]["data"][21 * 40 + 27] = json!(0);
            p["semanticMasks"]["face"]["data"][21 * 40 + 27] =
                json!(if false_positive_head { 255 } else { 0 });
            assert!(
                !promoted(&p, &poly, f, 26, 20),
                "false_positive_head={false_positive_head}"
            );
        }
    }

    fn motion_interface_fixture() -> (Value, Vec<Point>, Frame) {
        let (mut p, poly, frame) = fixture();
        p["__experimentalSharedSubjectAdmission"] = json!(true);
        // The fine source parser finds hair just inside the earlier optical
        // contour. Its own skin/hair interface is two source pixels inward.
        // The anatomical frame deliberately remains that of the old contour.
        p["semanticMasks"]["face"]["data"][20 * 40 + 24] = json!(200);
        p["semanticMasks"]["hair"]["data"][20 * 40 + 24] = json!(160);
        let mut mask = p["semanticMasks"]["faceContourMask"].clone();
        for y in 0..40 {
            for x in 23..40 {
                mask["data"][y * 40 + x] = json!(0);
            }
        }
        let contour = json!({"shape":"polygon","points":[
            {"x":10.0/40.0,"y":6.0/40.0},
            {"x":23.0/40.0,"y":6.0/40.0},
            {"x":23.0/40.0,"y":32.0/40.0},
            {"x":10.0/40.0,"y":32.0/40.0}
        ]});
        p["semanticMasks"]["faceContour"] = json!({"shape":"polygon","points":[
            {"x":10.0/40.0,"y":6.0/40.0},
            {"x":25.0/40.0,"y":6.0/40.0},
            {"x":25.0/40.0,"y":32.0/40.0},
            {"x":10.0/40.0,"y":32.0/40.0}
        ]});
        p["__sourceMotionInterface"] = json!({"mask":mask,"contour":contour});
        (p, poly, frame)
    }

    #[test]
    fn same_revision_motion_interface_admits_observed_internal_neighbor_hair() {
        let (p, poly, frame) = motion_interface_fixture();
        let mut without_interface = p.clone();
        without_interface
            .as_object_mut()
            .unwrap()
            .remove("__sourceMotionInterface");
        assert!(
            !promoted(&without_interface, &poly, frame, 24, 20),
            "old optical contour deliberately rejects its interior"
        );
        assert!(
            promoted(&p, &poly, frame, 24, 20),
            "fine source interface must admit its eligible connected neighbor"
        );
    }

    #[test]
    fn motion_interface_does_not_redefine_source_optical_evidence() {
        let (p, poly, frame) = motion_interface_fixture();
        let out = promote_interface(&p, &poly, frame, 40, 40).unwrap();
        assert_eq!(out["semanticMasks"]["hair"]["data"][20 * 40 + 24], 0);
        for key in [
            "subjectMask",
            "protectMask",
            "adjustments",
            "__sourceMotionInterface",
        ] {
            assert_eq!(p[key], out[key], "source input {key}");
        }
        for key in ["face", "faceContourMask", "faceContour", "garment"] {
            assert_eq!(
                p["semanticMasks"][key], out["semanticMasks"][key],
                "optical input {key}"
            );
        }
    }

    #[test]
    fn identical_motion_interface_preserves_legacy_promotion() {
        let (mut p, poly, frame) = motion_interface_fixture();
        p.as_object_mut().unwrap().remove("__sourceMotionInterface");
        let legacy = promote_interface(&p, &poly, frame, 40, 40).unwrap();
        p["__sourceMotionInterface"] = json!({
            "mask":p["semanticMasks"]["faceContourMask"].clone(),
            "contour":p["semanticMasks"]["faceContour"].clone()
        });
        let same = promote_interface(&p, &poly, frame, 40, 40).unwrap();
        assert_eq!(legacy["semanticMasks"], same["semanticMasks"]);
        assert_eq!(legacy["subjectMask"], same["subjectMask"]);
    }

    #[test]
    fn motion_interface_does_not_override_connected_extension_protection() {
        let (p, poly, frame) = motion_interface_fixture();
        assert!(promoted(&p, &poly, frame, 24, 20));
        for key in ["protectMask", "__sourceGarment", "ears", "accessories"] {
            let mut q = p.clone();
            let mut data = vec![0_u8; 40 * 40];
            data[20 * 40 + 24] = 255;
            let barrier = json!({"shape":"raster","width":40,"height":40,"data":data});
            if ["ears", "accessories"].contains(&key) {
                q["semanticMasks"][key] = barrier;
            } else {
                q[key] = barrier;
            }
            assert!(!promoted(&q, &poly, frame, 24, 20), "{key}");
            assert!(
                promoted(&q, &poly, frame, 25, 18),
                "unrelated eligible neighbor remains available for {key}"
            );
        }
    }

    #[test]
    fn motion_interface_is_source_defined_for_both_slider_directions() {
        let (mut p, poly, frame) = motion_interface_fixture();
        p["adjustments"]["faceWidth"] = json!(30);
        let positive = promote_interface(&p, &poly, frame, 40, 40).unwrap();
        p["adjustments"]["faceWidth"] = json!(-30);
        let negative = promote_interface(&p, &poly, frame, 40, 40).unwrap();
        assert_eq!(positive["semanticMasks"], negative["semanticMasks"]);
        assert_eq!(positive["subjectMask"], negative["subjectMask"]);
    }

    #[test]
    fn motion_interface_keeps_outer_hair_far_hair_and_true_background_boundaries() {
        let (p, poly, frame) = motion_interface_fixture();
        assert!(promoted(&p, &poly, frame, 24, 20));
        for (x, y) in [(6, 20), (32, 20), (26, 33)] {
            assert!(
                !promoted(&p, &poly, frame, x, y),
                "outside support at {x},{y}"
            );
        }
        let mut hole = p.clone();
        hole["subjectMask"]["data"][21 * 40 + 24] = json!(0);
        assert!(
            !promoted(&hole, &poly, frame, 24, 20),
            "an enclosed real background hole still blocks full extension support"
        );
    }
}
