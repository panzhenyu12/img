//! Internal hairline transport. Hair and skin on the two sides of an observed
//! interface share one velocity; the outer head, facial features and explicit
//! protection are stationary. This does not infer hidden hair or a new head.

use super::*;

pub(super) fn prepare(
    source: &RgbaImage,
    params: &Value,
    polygon: &[Point],
    frame: Frame,
) -> Option<Hairline> {
    let values = params.get("adjustments").unwrap_or(params);
    if percent(values, "forehead").abs() < 1e-6 {
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
    let radius = (frame.width() * 0.45).max(12.0);
    let pitch = (frame.width() / 64.0).max(1.0);
    let mut controls = Vec::new();
    let mut requests = Vec::new();
    // Resample the observed outline, not an invented horizontal forehead.
    // A valid control has visible skin below and visible hair above it.
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let steps = (sub(b, a).0.hypot(sub(b, a).1) / pitch).ceil().max(1.0) as usize;
        for step in 0..steps {
            let p = add(a, scale(sub(b, a), step as f32 / steps as f32));
            let y = (frame.coordinates(p).1 - frame.top) / frame.height();
            if y >= 0.30 || y < -0.40 {
                continue;
            }
            // Class probabilities need not complement each other. Seek
            // confident material across wider uncertainty bands, requiring
            // opaque head material all along the observation path.
            let mut hair_evidence = false;
            let mut skin_evidence = false;
            for distance in 1..=(frame.width() * 0.045).ceil().max(3.0) as usize {
                let above = add(p, scale(frame.down, -(distance as f32)));
                let below = add(p, scale(frame.down, distance as f32));
                if read(&head, above) < 0.98 || read(&head, below) < 0.98 {
                    break;
                }
                hair_evidence |= read(&hair, above) >= 0.7;
                skin_evidence |= read(&skin, below) >= 0.7;
                if hair_evidence && skin_evidence {
                    break;
                }
            }
            if !hair_evidence || !skin_evidence {
                continue;
            }
            controls.push(p);
            requests.push(scale(frame.down, vertical_contour_offset(frame, p, values)));
        }
    }
    if controls.len() < 4 {
        return None;
    }
    let (mut left, mut top, mut right, mut bottom) = (w, h, 0.0_f32, 0.0_f32);
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
    let (cw, ch) = (right - left, bottom - top);
    let count = cw * ch;
    let interface_distance: Vec<f32> = (0..count)
        .map(|i| {
            let p = (
                left as f32 + (i % cw) as f32 + 0.5,
                top as f32 + (i / cw) as f32 + 0.5,
            );
            controls
                .iter()
                .map(|&q| sub(p, q).0.hypot(sub(p, q).1))
                .fold(f32::MAX, f32::min)
        })
        .collect();
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
            left as f32 + (i % cw) as f32 + 0.5,
            top as f32 + (i / cw) as f32 + 0.5,
        )
    };
    let mut distance = vec![u16::MAX; count];
    let mut queue = VecDeque::new();
    for i in 0..count {
        let p = position(i);
        let (_, v) = frame.coordinates(p);
        let y = (v - frame.top) / frame.height();
        // The independently observed opaque head is the composite material
        // authority. Class uncertainty inside it is not transparency or a
        // fixed obstacle. Explicit features/accessories remain authoritative.
        let inside_head = read(&head, p) >= 0.98;
        // The anatomical oval sets scale and direction, not the lateral edge
        // of observed head material. Preserve the vertical band; the selected
        // opaque-head component and explicit masks determine lateral support.
        let within_selected_height = (-0.40..0.42).contains(&y);
        let forbidden = !inside_head
            || interface_distance[i] >= radius - 2.0
            || !within_selected_height
            || !region.contains(p.0 / w, p.1 / h)
            || selection.as_ref().is_some_and(|m| read(m, p) <= 1e-6)
            || protect.as_ref().is_some_and(|m| read(m, p) >= 1.0 - 1e-6)
            || fixed.iter().any(|m| read(m, p) > 1e-6)
            || i % cw == 0
            || i % cw == cw - 1
            || i / cw == 0
            || i / cw == ch - 1;
        if forbidden {
            distance[i] = 0;
            queue.push_back(i);
        }
    }
    // A semantic head raster can contain disconnected material from another
    // person. Only components touched by an observed selected hairline may
    // receive motion; spatial proximity to a kernel is not ownership evidence.
    let mut owned = vec![false; count];
    let mut owned_queue = VecDeque::new();
    for &p in &controls {
        let x = (p.0 - left as f32 - 0.5)
            .round()
            .clamp(0.0, (cw - 1) as f32) as usize;
        let y = (p.1 - top as f32 - 0.5).round().clamp(0.0, (ch - 1) as f32) as usize;
        let i = y * cw + x;
        if distance[i] != 0 && !owned[i] {
            owned[i] = true;
            owned_queue.push_back(i);
        }
    }
    while let Some(i) = owned_queue.pop_front() {
        for n in neighbors(i, cw, ch) {
            if distance[n] != 0 && !owned[n] {
                owned[n] = true;
                owned_queue.push_back(n);
            }
        }
    }
    for i in 0..count {
        if !owned[i] && distance[i] != 0 {
            distance[i] = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let next = distance[i].saturating_add(1);
        // Chebyshev distance matches the square bilinear sampling footprint.
        let (x, y) = (i % cw, i / cw);
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let Some(xx) = x.checked_add_signed(dx).filter(|&x| x < cw) else {
                    continue;
                };
                let Some(yy) = y.checked_add_signed(dy).filter(|&y| y < ch) else {
                    continue;
                };
                let n = yy * cw + xx;
                if next < distance[n] {
                    distance[n] = next;
                    queue.push_back(n);
                }
            }
        }
    }
    let interpolator = positive_kernel::PositiveKernel::new(controls.clone(), radius);
    let unity = vec![(1.0, 0.0); controls.len()];
    // Two Chebyshev rings before the 3x3 average retain one stationary
    // ring afterwards. Every bilinear cell touching a forbidden texel then
    // has zero velocity at all four corners. Keep the actual forbidden
    // material unchanged; the former four-pixel Euclidean lower bound froze
    // additional reachable interior controls.
    let requested_caps: Vec<f32> = (0..count)
        .map(|i| {
            if distance[i] <= 2 {
                return 0.0;
            }
            let p = position(i);
            let normalization = interpolator.evaluate(&unity, p).0;
            // Remove only the kernel's zero background, not a material support
            // weight. Numerator/denominator have the same positive weights:
            // their ratio stays inside the convex hull of requested controls.
            let request = scale(
                interpolator.evaluate(&requests, p),
                1.0 / normalization.max(1e-12),
            );
            let external = selection.as_ref().map(|m| read(m, p)).unwrap_or(1.0)
                * (1.0 - protect.as_ref().map(|m| read(m, p)).unwrap_or(0.0));
            dot(request, frame.down).abs() * external
        })
        .collect();
    // Greatest field below the requested caps with bounded adjacent scalar
    // differences. All vectors share the face-down direction, so a cardinal
    // bound of .45 gives a bilinear gradient norm <=sqrt(2)*.45. This bounds
    // the ideal continuous flow's strain, not floating-point raster topology.
    let bounded = scalar_field::bounded_scalar_field(cw, ch, &requested_caps, 0.45);
    let sign = -percent(values, "forehead").signum();
    let vectors = (0..count)
        .map(|i| {
            // Positive translation-invariant averaging preserves the difference
            // bound. The two-ring guard leaves all forbidden texels and their
            // complete bilinear sampling footprint stationary after this pass.
            let (x, y) = (i % cw, i / cw);
            let mut value = 0.0;
            for (dy, wy) in [(-1_isize, 1.0), (0, 2.0), (1, 1.0)] {
                for (dx, wx) in [(-1_isize, 1.0), (0, 2.0), (1, 1.0)] {
                    let (xx, yy) = (x as isize + dx, y as isize + dy);
                    if xx >= 0 && yy >= 0 && xx < cw as isize && yy < ch as isize {
                        value += wx * wy * bounded[yy as usize * cw + xx as usize] / 16.0;
                    }
                }
            }
            scale(frame.down, sign * value)
        })
        .collect();
    let velocity = flow::VelocityGrid::new(position(0), cw, ch, vectors);
    #[cfg(test)]
    if let Some(path) = params.get("__faceFieldTrace").and_then(Value::as_str) {
        let targets: Vec<_> = controls
            .iter()
            .zip(&requests)
            .map(|(&p, &v)| add(p, v))
            .collect();
        anchor_transition::write_trace(
            path,
            &velocity,
            position(0),
            cw,
            ch,
            &controls,
            &targets,
            &targets,
            &targets,
            None,
        )
        .expect("write hairline trace");
        let mut crossed = 0;
        let mut max_round_trip = 0.0_f32;
        for i in 0..count {
            let p = position(i);
            let q = velocity.transport(p, -1.0);
            if q == p {
                continue;
            }
            let x = (q.0 - position(0).0).floor() as isize;
            let y = (q.1 - position(0).1).floor() as isize;
            for (x, y) in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
                if x < 0
                    || y < 0
                    || x >= cw as isize
                    || y >= ch as isize
                    || distance[y as usize * cw + x as usize] == 0
                {
                    crossed += 1;
                }
            }
            let restored = velocity.transport(q, 1.0);
            max_round_trip = max_round_trip.max(sub(restored, p).0.hypot(sub(restored, p).1));
        }
        std::fs::write(std::path::Path::new(path).join("sampling.json"), serde_json::to_vec(&json!({"forbiddenSourceContributors":crossed,"maxRoundTripPx":max_round_trip,"controls":controls.len()})).unwrap()).expect("write sampling diagnostics");
    }
    Some(Hairline {
        velocity,
        left,
        top,
        width: cw,
        height: ch,
    })
}

pub(super) struct Hairline {
    velocity: flow::VelocityGrid,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}
impl Hairline {
    #[cfg(test)]
    pub(super) fn transport_for_test(&self, p: Point, direction: f32) -> Point {
        self.velocity.transport(p, direction)
    }

    pub(super) fn render(&self, source: &RgbaImage) -> RgbaImage {
        let (left, top, cw, ch) = (self.left, self.top, self.width, self.height);
        let mut rendered = source.clone();
        let full_alpha = vec![1.0; source.width as usize * source.height as usize];
        for i in 0..cw * ch {
            let p = (
                left as f32 + (i % cw) as f32 + 0.5,
                top as f32 + (i / cw) as f32 + 0.5,
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
            let j = ((top + i / cw) * source.width as usize + left + i % cw) * 4;
            for c in 0..3 {
                rendered.pixels[j + c] = if pixel[3] > 1e-6 {
                    byte(pixel[c] / pixel[3])
                } else {
                    0
                };
            }
            rendered.pixels[j + 3] = byte(pixel[3] * 255.0);
        }
        rendered
    }
}
