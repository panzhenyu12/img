//! Isolated causal experiment: use the observed feature-to-contour gap for
//! the velocity transition instead of a fixed two-pixel band. Not production.
use super::*;
use std::path::Path;

pub(super) struct FeatureRamps<'a> {
    anchors: &'a [Anchor],
    gaps: Vec<Option<f32>>,
}

fn edge_distance(p: Point, polygon: &[Point]) -> f32 {
    polygon
        .iter()
        .enumerate()
        .map(|(i, &a)| closest(p, a, polygon[(i + 1) % polygon.len()]).0)
        .fold(f32::MAX, f32::min)
}

fn cross(a: Point, b: Point) -> f32 {
    a.0 * b.1 - a.1 * b.0
}

fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let ab = sub(b, a);
    let cd = sub(d, c);
    let denominator = cross(ab, cd);
    if denominator.abs() < 1e-8 {
        return false;
    }
    let t = cross(sub(c, a), cd) / denominator;
    let u = cross(sub(c, a), ab) / denominator;
    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

impl<'a> FeatureRamps<'a> {
    pub(super) fn new(anchors: &'a [Anchor], contour: &[Point]) -> Result<Self, &'static str> {
        let mut gaps = Vec::new();
        for anchor in anchors {
            if anchor.polygon.is_empty() {
                gaps.push(None);
                continue;
            }
            if anchor.polygon.iter().any(|&p| !inside(p, contour)) {
                return Err("feature-distance experiment requires an interior feature polygon");
            }
            let mut gap = f32::MAX;
            for (i, &a) in anchor.polygon.iter().enumerate() {
                let b = anchor.polygon[(i + 1) % anchor.polygon.len()];
                for (j, &c) in contour.iter().enumerate() {
                    let d = contour[(j + 1) % contour.len()];
                    if segments_intersect(a, b, c, d) {
                        gap = 0.0;
                    }
                    gap = gap
                        .min(closest(a, c, d).0)
                        .min(closest(b, c, d).0)
                        .min(closest(c, a, b).0)
                        .min(closest(d, a, b).0);
                }
            }
            if !gap.is_finite() || gap <= 1e-6 {
                return Err("feature-distance experiment has no positive feature/contour gap");
            }
            gaps.push(Some(gap));
        }
        Ok(Self { anchors, gaps })
    }

    pub(super) fn weight(&self, p: Point) -> f32 {
        self.anchors
            .iter()
            .zip(&self.gaps)
            .map(|(anchor, gap)| match gap {
                None => 1.0 - anchor.freeze(p),
                Some(_) if inside(p, &anchor.polygon) => 0.0,
                Some(gap) => smoothstep(0.0, *gap, edge_distance(p, &anchor.polygon)),
            })
            .product()
    }
}

/// Only the actual final integrated flow is exported. Small local trace
/// files are diagnostic evidence, never inputs to the renderer.
pub(super) fn write_trace(
    path: &str,
    velocity: &flow::VelocityGrid,
    origin: Point,
    width: usize,
    height: usize,
    controls: &[Point],
    requested: &[Point],
    feasible: &[Point],
    pre_projection: &[Point],
    ramps: Option<&FeatureRamps<'_>>,
) -> Result<(), String> {
    let path = Path::new(path);
    std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    let mut forward = Vec::with_capacity(width * height * 8);
    let mut inverse_sample = Vec::with_capacity(width * height * 8);
    for y in 0..height {
        for x in 0..width {
            let p = (origin.0 + x as f32, origin.1 + y as f32);
            let q = velocity.transport(p, 1.0);
            forward.extend_from_slice(&q.0.to_le_bytes());
            forward.extend_from_slice(&q.1.to_le_bytes());
            // Match the renderer's f32 local source-sampling arithmetic.
            // Do not recover this offset by subtracting rounded absolute
            // forward coordinates, or quantize it through the forward map.
            let movement = flow::inverse(velocity, p);
            let source_sample = (x as f32 + movement.0, y as f32 + movement.1);
            inverse_sample.extend_from_slice(&source_sample.0.to_le_bytes());
            inverse_sample.extend_from_slice(&source_sample.1.to_le_bytes());
        }
    }
    std::fs::write(path.join("forward.f32"), forward).map_err(|e| e.to_string())?;
    std::fs::write(path.join("inverse-sample.f32"), inverse_sample).map_err(|e| e.to_string())?;
    let realized: Vec<_> = controls
        .iter()
        .map(|&p| velocity.transport(p, 1.0))
        .collect();
    let metadata = json!({"origin":origin,"width":width,"height":height,
        "forwardFile":"forward.f32","coordinateConvention":"source pixel centers to absolute output coordinates",
        "inverseSampleFile":"inverse-sample.f32",
        "inverseSampleCoordinateConvention":"row-major interleaved little-endian f32 (x, y); destination crop pixel indices to source crop pixel indices; integer (0, 0) is the first source pixel center; add origin for absolute source pixel-center coordinates",
        "inverseSampleScope":"coordinates for the complete final-flow grid; renderer selection, frozen-pixel and zero-motion skips are not encoded",
        "controls":controls,"requested":requested,"feasible":feasible,"realized":realized,
        "preProjectionContourTarget":pre_projection,
        "featureGaps":ramps.map(|r| &r.gaps),"integrationSteps":velocity.steps});
    std::fs::write(
        path.join("metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<Anchor>, Vec<Point>) {
        (
            vec![Anchor {
                center: (40.0, 40.0),
                across: (1.0, 0.0),
                down: (0.0, 1.0),
                rx: 8.0,
                ry: 3.0,
                polygon: vec![(30.0, 38.0), (50.0, 38.0), (50.0, 42.0), (30.0, 42.0)],
            }],
            vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        )
    }
    #[test]
    fn measured_gap_fixes_complete_feature_and_retains_every_contour_target() {
        let (anchors, contour) = fixture();
        let ramps = FeatureRamps::new(&anchors, &contour).unwrap();
        assert_eq!(ramps.gaps, vec![Some(30.0)]);
        for y in 38..42 {
            for x in 30..50 {
                assert_eq!(ramps.weight((x as f32 + 0.5, y as f32 + 0.5)), 0.0);
            }
        }
        for k in 0..100 {
            assert_eq!(ramps.weight((k as f32, 0.0)), 1.0);
            assert_eq!(ramps.weight((0.0, k as f32)), 1.0);
        }
    }
    #[test]
    fn transition_remains_continuous_beyond_legacy_ellipse_cutoff() {
        let (anchors, contour) = fixture();
        let ramps = FeatureRamps::new(&anchors, &contour).unwrap();
        let mut last = 0.0;
        for k in 0..=400 {
            let w = ramps.weight((40.0, 42.0 + k as f32 * 0.1));
            assert!(w >= last - 1e-6);
            assert!(w - last <= 1.5 / 30.0 * 0.1 + 1e-6);
            last = w;
        }
        assert_eq!(last, 1.0);
    }
    #[test]
    fn overlapping_feature_and_contour_is_rejected_without_an_arbitrary_ramp() {
        let (mut anchors, contour) = fixture();
        anchors[0].polygon[0] = (0.0, 38.0);
        assert!(FeatureRamps::new(&anchors, &contour).is_err());
    }
}
