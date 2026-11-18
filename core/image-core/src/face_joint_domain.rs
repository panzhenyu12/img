//! Source-material driven motion domain for the face contour candidate.
//!
//! The production path historically built two unrelated falloffs: one from
//! rigid attachments and another from frozen context. That lets a contour
//! sample cross a material-role boundary with a different gain from its
//! neighbour. This module builds one scalar domain from the same source roles
//! used by the compositor. It is intentionally opt-in until real-photo
//! replays prove that the shared domain improves the rendered seam.

use super::{neighbors, Point};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub(super) struct MotionDomain {
    width: usize,
    height: usize,
    weights: Vec<f32>,
    origin: Point,
    pixel_pitch: f32,
}

impl MotionDomain {
    /// Extend the velocity definition around observed moving material.
    ///
    /// `travel` bounds the integrated path length, not just a target endpoint
    /// displacement. Both flow directions need this support: an expanding
    /// source edge leaves its original skin raster, and inverse sampling starts
    /// on the newly covered side of that raster. The full-gain halo followed
    /// by a smooth falloff prevents the old material boundary from becoming an
    /// absorbing zero-velocity wall.
    ///
    /// This changes no material ownership, optical alpha, donor eligibility or
    /// output permission. A halo texel is a place where a velocity can be
    /// evaluated, not newly movable hair or background. `blocked` still marks
    /// exact motion barriers and stops propagation across them. Near a barrier
    /// or crop boundary the available travel can be smaller than requested.
    pub(super) fn build_with_motion_halo(
        width: usize,
        height: usize,
        origin: Point,
        active: &[bool],
        blocked: &[bool],
        travel: f32,
        taper: f32,
    ) -> Option<Self> {
        let count = width.checked_mul(height)?;
        if width < 2 || height < 2 || active.len() != count || blocked.len() != count
            || !travel.is_finite() || travel < 0.0 || !taper.is_finite() || taper <= 0.0
            || !origin.0.is_finite() || !origin.1.is_finite()
        {
            return None;
        }
        // Reuse the established hard-barrier envelope independently of source
        // occupancy. Only the bounded halo below decides far-field support.
        let mut domain = Self::build(width, height, origin, &vec![true; count], blocked, taper)?;
        let mut distance = vec![usize::MAX; count];
        let mut queue = VecDeque::new();
        for i in 0..count {
            if active[i] && !blocked[i] {
                distance[i] = 0;
                queue.push_back(i);
            }
        }
        // The four-neighbour metric is a conservative path metric: it cannot
        // cross a blocked corner. Multiplying a Euclidean travel budget by
        // sqrt(2) covers all orientations in unobstructed raster space; the
        // two-pixel guard covers bilinear footprints and subpixel contours.
        // This bounded overestimate avoids expensive all-pairs distances. It
        // is not a claim that the discrete field is globally C1 or isotropic.
        let full_support = travel * std::f32::consts::SQRT_2 + 2.0;
        let outer_support = full_support + taper;
        if !outer_support.is_finite() {
            return None;
        }
        while let Some(i) = queue.pop_front() {
            let next = distance[i] + 1;
            if next as f32 > outer_support + 1.0 {
                continue;
            }
            for n in neighbors(i, width, height) {
                if !blocked[n] && next < distance[n] {
                    distance[n] = next;
                    queue.push_back(n);
                }
            }
        }
        for i in 0..count {
            let support = if distance[i] == usize::MAX {
                0.0
            } else {
                1.0 - quintic((distance[i] as f32 - full_support) / taper)
            };
            let (x, y) = (i % width, i / width);
            let border_distance = x.min(y).min(width - 1 - x).min(height - 1 - y);
            // VelocityGrid uses zero outside the crop. Matching that value at
            // the crop perimeter removes a second artificial support jump.
            let border = quintic(border_distance as f32 / taper);
            domain.weights[i] *= support * border;
        }
        Some(domain)
    }

    /// Build a continuous source-role transition.
    ///
    /// `active` is the admitted connected material (skin plus explicitly
    /// reviewed moving attachments). `blocked` is every exact material
    /// boundary that must stay stationary (hard attachment, garment and
    /// protection). Inactive pixels get zero weight, while active pixels with
    /// no nearby hard boundary retain full weight. This distinction matters:
    /// treating the whole unknown exterior as a distance seed suppresses the
    /// facial contour itself and turns a safe transition into a no-op. The
    /// distance ramp is quintic, so value and first derivative vanish at a
    /// hard boundary.
    pub(super) fn build(
        width: usize,
        height: usize,
        origin: Point,
        active: &[bool],
        blocked: &[bool],
        taper: f32,
    ) -> Option<Self> {
        if width < 2
            || height < 2
            || active.len() != width * height
            || blocked.len() != width * height
            || !taper.is_finite()
            || taper <= 0.0
        {
            return None;
        }
        let mut distance = vec![u16::MAX; width * height];
        let mut queue = VecDeque::new();
        for i in 0..distance.len() {
            if blocked[i] {
                distance[i] = 0;
                queue.push_back(i);
            }
        }
        // An admitted source domain may legitimately have no hard material
        // barrier (for example a face-only crop with no user protection,
        // garment or second-person component). In that case every active
        // texel has full motion gain; treating the empty seed set as "no
        // domain" silently fell back to the legacy taper and made the
        // continuous cheek candidate a no-op. Keep `None` for malformed input
        // above, but represent an unbounded valid domain with MAX distances.
        while let Some(i) = queue.pop_front() {
            let next = distance[i].saturating_add(1);
            for n in neighbors(i, width, height) {
                if next < distance[n] {
                    distance[n] = next;
                    queue.push_back(n);
                }
            }
        }
        let mut weights = vec![0.0; distance.len()];
        for i in 0..distance.len() {
            if !active[i] || blocked[i] {
                continue;
            }
            let t = if distance[i] == u16::MAX {
                1.0
            } else {
                (distance[i] as f32 / taper).clamp(0.0, 1.0)
            };
            // Quintic smoothstep keeps the transition C2 at both ends. The
            // source role itself remains the authority; this only controls
            // motion gain and never changes alpha or donor ownership.
            weights[i] = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        }
        Some(Self {
            width,
            height,
            weights,
            origin,
            pixel_pitch: 1.0,
        })
    }

    pub(super) fn weight(&self, p: Point) -> f32 {
        // Contour controls rarely land exactly on pixel centres. Rounding the
        // query to one source texel made a valid control hit an inactive
        // exterior texel and gave it a hard zero weight, so whole cheek arcs
        // silently became no-op while the lower chin (which happened to land
        // on skin texels) still moved. Interpolate the source-role domain at
        // the actual sub-pixel location instead. This preserves zero at a
        // hard boundary while giving a continuous transition across the
        // antialiased face/skin raster.
        let x = (p.0 - self.origin.0) / self.pixel_pitch;
        let y = (p.1 - self.origin.1) / self.pixel_pitch;
        if x < 0.0 || y < 0.0 || x > (self.width - 1) as f32 || y > (self.height - 1) as f32 {
            return 0.0;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let at = |xx: usize, yy: usize| self.weights[yy * self.width + xx];
        let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    #[cfg(test)]
    pub(super) fn weights(&self) -> &[f32] {
        &self.weights
    }
}

fn quintic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::flow::VelocityGrid;

    fn constant_field(domain: &MotionDomain, velocity: Point) -> VelocityGrid {
        VelocityGrid::new(
            domain.origin,
            domain.width,
            domain.height,
            domain.weights.iter().map(|&w| (velocity.0 * w, velocity.1 * w)).collect(),
        )
    }

    #[test]
    fn motion_halo_retains_signed_travel_across_the_original_skin_boundary() {
        let (width, height) = (100, 80);
        let active: Vec<_> = (0..width * height)
            .map(|i| (40..60).contains(&(i % width)) && (30..50).contains(&(i / width)))
            .collect();
        let domain = MotionDomain::build_with_motion_halo(
            width, height, (0.5, 0.5), &active, &vec![false; width * height], 10.0, 6.0,
        ).unwrap();
        for speed in [-10.0, 10.0] {
            let flow = constant_field(&domain, (speed, 0.0));
            for p in [(40.5, 40.5), (59.5, 40.5)] {
                for direction in [-1.0, 1.0] {
                    let q = flow.transport(p, direction);
                    assert!((q.0 - (p.0 + speed * direction)).abs() < 0.001,
                        "old skin boundary stopped signed motion: {p:?} -> {q:?}, speed={speed}, direction={direction}");
                    assert!((q.1 - p.1).abs() < 0.001);
                    let recovered = flow.transport(q, -direction);
                    assert!((recovered.0 - p.0).abs() < 0.001);
                }
            }
        }
    }

    #[test]
    fn motion_halo_covers_oblique_travel_and_keeps_the_distant_scene_fixed() {
        let (width, height) = (100, 80);
        let active: Vec<_> = (0..width * height)
            .map(|i| (40..60).contains(&(i % width)) && (30..50).contains(&(i / width)))
            .collect();
        let origin = (4000.5, 7000.5);
        let domain = MotionDomain::build_with_motion_halo(
            width, height, origin, &active, &vec![false; width * height], 10.0, 6.0,
        ).unwrap();
        // Every direction has the same requested path length. The old binary
        // support stalls at the original rectangle corner in outward sectors.
        for k in 0..32 {
            let angle = k as f32 * std::f32::consts::TAU / 32.0;
            let velocity = (10.0 * angle.cos(), 10.0 * angle.sin());
            let flow = constant_field(&domain, velocity);
            for p in [(origin.0 + 40.0, origin.1 + 30.0), (origin.0 + 59.0, origin.1 + 49.0)] {
                let q = flow.transport(p, 1.0);
                assert!((q.0 - p.0 - velocity.0).hypot(q.1 - p.1 - velocity.1) < 0.002,
                    "oblique support lost travel at angle {angle}: {p:?} -> {q:?}");
            }
            for p in [origin, (origin.0 + 95.0, origin.1 + 40.0), (origin.0 + 50.0, origin.1 + 79.0)] {
                for direction in [-1.0, 1.0] {
                    assert_eq!(flow.transport(p, direction), p, "distant/crop field moved");
                }
            }
        }
    }

    #[test]
    fn motion_halo_respects_hard_barriers_and_cannot_recruit_material_across_them() {
        let (width, height) = (100, 80);
        let active: Vec<_> = (0..width * height)
            .map(|i| (30..40).contains(&(i % width)) && (30..50).contains(&(i / width)))
            .collect();
        let blocked: Vec<_> = (0..width * height)
            .map(|i| (55..58).contains(&(i % width)))
            .collect();
        let domain = MotionDomain::build_with_motion_halo(
            width, height, (0.5, 0.5), &active, &blocked, 25.0, 6.0,
        ).unwrap();
        for speed in [-20.0, 20.0] {
            let flow = constant_field(&domain, (speed, 0.0));
            for i in 0..width * height {
                if blocked[i] || i % width >= 58 {
                    let p = (0.5 + (i % width) as f32, 0.5 + (i / width) as f32);
                    assert_eq!(domain.weight(p), 0.0);
                    for direction in [-1.0, 1.0] {
                        assert_eq!(flow.transport(p, direction), p,
                            "motion crossed/recruited the fixed material at {p:?}");
                    }
                }
            }
            // An incoming skin trajectory must stop before the barrier; a
            // halo permits leaving old skin, never crossing explicit locks.
            let start = (45.5, 40.5);
            let q = flow.transport(start, speed.signum());
            assert!(q.0 > start.0 + 1.0 && q.0 < 55.5, "barrier trajectory {q:?}");
        }
    }

    #[test]
    fn motion_halo_without_source_material_is_identity_and_invalid_budgets_abstain() {
        let domain = MotionDomain::build_with_motion_halo(
            8, 8, (0.5, 0.5), &[false; 64], &[false; 64], 10.0, 3.0,
        ).unwrap();
        assert!(domain.weights.iter().all(|&w| w == 0.0));
        for budget in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(MotionDomain::build_with_motion_halo(
                8, 8, (0.5, 0.5), &[true; 64], &[false; 64], budget, 3.0,
            ).is_none());
        }
    }

    #[test]
    fn hard_material_and_unknown_pixels_have_zero_motion() {
        let movable = vec![true; 7 * 5];
        let mut blocked = vec![false; 7 * 5];
        blocked[2 * 7 + 3] = true;
        let mut movable = movable;
        movable[0] = false;
        let domain = MotionDomain::build(7, 5, (0.5, 0.5), &movable, &blocked, 3.0).unwrap();
        assert_eq!(domain.weight((3.5, 2.5)), 0.0);
        assert_eq!(domain.weight((0.5, 0.5)), 0.0);
        assert!(domain.weight((6.5, 4.5)) > 0.99);
    }

    #[test]
    fn transition_is_monotone_and_has_no_gain_jump() {
        let movable = vec![true; 12 * 3];
        let mut blocked = vec![false; 12 * 3];
        blocked[0] = true;
        let domain = MotionDomain::build(12, 3, (0.5, 0.5), &movable, &blocked, 5.0).unwrap();
        let row = &domain.weights()[0..12];
        assert_eq!(row[0], 0.0);
        for pair in row.windows(2) {
            assert!(pair[1] + 1e-6 >= pair[0]);
            assert!((pair[1] - pair[0]).abs() <= 0.4);
        }
    }

    #[test]
    fn subpixel_queries_interpolate_the_source_role_transition() {
        let movable = vec![true; 4 * 2];
        let mut blocked = vec![false; 4 * 2];
        // The left column is a fixed material boundary. A query halfway
        // between the boundary and the first free texel must receive the
        // continuous interpolated gain instead of rounding to zero.
        blocked[0] = true;
        blocked[4] = true;
        let domain = MotionDomain::build(4, 2, (0.5, 0.5), &movable, &blocked, 2.0).unwrap();
        assert_eq!(domain.weight((0.5, 0.5)), 0.0);
        let half = domain.weight((1.0, 0.5));
        assert!(half > 0.0 && half < domain.weight((1.5, 0.5)));
    }

    #[test]
    fn barrier_free_source_material_retains_full_motion_support() {
        let mut active = vec![true; 4 * 4];
        active[0] = false;
        let domain = MotionDomain::build(4, 4, (0.5, 0.5), &active, &[false; 16], 3.0)
            .expect("valid material does not require a stationary barrier");
        assert_eq!(domain.weight((0.5, 0.5)), 0.0);
        assert_eq!(domain.weight((1.5, 1.5)), 1.0);
        assert_eq!(domain.weight((2.25, 2.75)), 1.0);
    }

    #[test]
    fn missing_material_domain_abstains() {
        assert!(MotionDomain::build(1, 4, (0.5, 0.5), &[true; 4], &[false; 4], 2.0).is_none());
        assert!(MotionDomain::build(4, 4, (0.5, 0.5), &[true; 3], &[false; 16], 2.0).is_none());
    }
}
