//! Source-material admission candidate for vertical eye edits.
//!
//! A segmentation mask is not an ocular-material mask: in portrait photos it
//! commonly includes the upper lid crease, lashes, and specular halo.  This
//! isolated candidate keeps only connected, interior support of the detected
//! aperture.  It is deliberately not wired into the public renderer until
//! real-photo holdouts validate the chosen thresholds.

type Point = (f32, f32);

fn inside(point: Point, polygon: &[Point]) -> bool {
    let mut hit = false;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        if (a.1 > point.1) != (b.1 > point.1)
            && point.0 < (b.0 - a.0) * (point.1 - a.1) / (b.1 - a.1) + a.0
        {
            hit = !hit;
        }
    }
    hit
}

fn point_segment_distance(point: Point, a: Point, b: Point) -> f32 {
    let edge = (b.0 - a.0, b.1 - a.1);
    let denom = edge.0 * edge.0 + edge.1 * edge.1;
    let t = if denom > 1e-6 {
        ((point.0 - a.0) * edge.0 + (point.1 - a.1) * edge.1) / denom
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    let q = (a.0 + edge.0 * t, a.1 + edge.1 * t);
    (point.0 - q.0).hypot(point.1 - q.1)
}

fn distance_to_boundary(point: Point, polygon: &[Point]) -> f32 {
    polygon
        .iter()
        .enumerate()
        .map(|(i, &a)| point_segment_distance(point, a, polygon[(i + 1) % polygon.len()]))
        .fold(f32::INFINITY, f32::min)
}

fn neighbors(index: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let x = index % width;
    let y = index / width;
    [
        (x > 0).then_some(index - 1),
        (x + 1 < width).then_some(index + 1),
        (y > 0).then_some(index - width),
        (y + 1 < height).then_some(index + width),
    ]
    .into_iter()
    .flatten()
}

/// Admit a conservative ocular source region from a soft segmentation.
///
/// Pixels must be inside the supplied eye aperture, at least `margin` pixels
/// away from its boundary, and belong to a connected candidate component with
/// enough interior support.  The return value is binary (`0`/`1`) so callers
/// cannot mistake it for physical alpha.  Invalid input returns an empty
/// mask, which is the safe behavior for an edit that needs source material.
pub(super) fn admit(
    width: usize,
    height: usize,
    candidate: &[f32],
    polygon: &[Point],
    margin: f32,
    min_component: usize,
) -> Vec<f32> {
    if width == 0
        || height == 0
        || candidate.len() != width.saturating_mul(height)
        || polygon.len() < 3
        || !margin.is_finite()
        || margin < 0.0
        || min_component == 0
        || polygon.iter().any(|p| !p.0.is_finite() || !p.1.is_finite())
        || candidate.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Vec::new();
    }
    let mut admitted = vec![false; candidate.len()];
    for y in 0..height {
        for x in 0..width {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            let index = y * width + x;
            admitted[index] = candidate[index] >= 0.5
                && inside(point, polygon)
                && distance_to_boundary(point, polygon) >= margin;
        }
    }

    // Drop tiny or one-pixel-wide islands. This prevents a lash/eyeliner
    // segment that touches the aperture from becoming a movable source layer.
    let mut seen = vec![false; admitted.len()];
    for start in 0..admitted.len() {
        if !admitted[start] || seen[start] {
            continue;
        }
        let mut queue = vec![start];
        seen[start] = true;
        let mut component = Vec::new();
        while let Some(index) = queue.pop() {
            component.push(index);
            for next in neighbors(index, width, height) {
                if admitted[next] && !seen[next] {
                    seen[next] = true;
                    queue.push(next);
                }
            }
        }
        if component.len() < min_component {
            for index in component {
                admitted[index] = false;
            }
        }
    }
    admitted
        .into_iter()
        .map(|accepted| if accepted { 1.0 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(left: f32, top: f32, right: f32, bottom: f32) -> Vec<Point> {
        vec![(left, top), (right, top), (right, bottom), (left, bottom)]
    }

    #[test]
    fn upper_lid_boundary_and_isolated_lash_are_rejected() {
        let width = 24;
        let height = 14;
        let polygon = rect(4.0, 3.0, 20.0, 11.0);
        let mut candidate = vec![0.0; width * height];
        // A full upper-lid/eyelash line sits on the aperture edge.
        for x in 4..20 {
            candidate[3 * width + x] = 1.0;
        }
        // Ocular interior (sclera/iris support).
        for y in 6..9 {
            for x in 7..17 {
                candidate[y * width + x] = 1.0;
            }
        }
        // An isolated dark lash outside the connected interior.
        candidate[4 * width + 19] = 1.0;
        let accepted = admit(width, height, &candidate, &polygon, 1.25, 4);
        assert!(accepted.iter().skip(3 * width).take(width).all(|&v| v == 0.0));
        assert!(accepted[7 * width + 12] == 1.0);
        assert_eq!(accepted[4 * width + 19], 0.0);
    }

    #[test]
    fn disconnected_or_malformed_source_is_rejected_without_guessing() {
        let candidate = vec![1.0; 8 * 8];
        assert!(admit(8, 8, &candidate, &rect(1.0, 1.0, 7.0, 7.0), 1.0, 100).iter().all(|&v| v == 0.0));
        assert!(admit(8, 8, &candidate, &[], 1.0, 1).is_empty());
        assert!(admit(8, 8, &candidate, &rect(1.0, 1.0, 7.0, 7.0), -1.0, 1).is_empty());
    }

    #[test]
    fn interior_material_is_stable_under_boundary_margin() {
        let width = 12;
        let height = 10;
        let polygon = rect(2.0, 2.0, 10.0, 8.0);
        let candidate = vec![1.0; width * height];
        let accepted = admit(width, height, &candidate, &polygon, 1.0, 4);
        assert_eq!(accepted.iter().filter(|&&v| v > 0.0).count(), 24);
        assert!(accepted[5 * width + 6] > 0.0);
        assert_eq!(accepted[2 * width + 5], 0.0);
    }
}
