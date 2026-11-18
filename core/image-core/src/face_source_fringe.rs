//! Recheck inferred portrait-edge freezes using connected source evidence.
//! An inferred person label is not an explicit user or attachment constraint.

use crate::RgbaImage;

pub(crate) struct Recovery {
    pub alpha: Vec<f32>,
    pub released: Vec<bool>,
}

pub(crate) fn recover_inferred_fringe(
    source: &RgbaImage,
    alpha: &[f32],
    owned: &[bool],
    person: &[f32],
    semantic_edge: &[f32],
    explicit_forbidden: &[bool],
    inferred_fixed: &[bool],
    fringe: usize,
) -> Recovery {
    let mut result = Recovery {
        alpha: alpha.to_vec(),
        released: vec![false; alpha.len()],
    };
    let candidates: Vec<_> = (0..alpha.len())
        .filter(|&i| inferred_fixed[i] && !owned[i] && !explicit_forbidden[i] && person[i] >= 0.8)
        .collect();
    if candidates.is_empty() {
        return result;
    }
    // Recheck weak, nonzero semantic alpha as unknown physical opacity. The
    // original person confidence remains high, so these are never background
    // donors. Ownership, explicit constraints, RGB and distance gates are
    // unchanged; the existing verifier also handles continuous illumination.
    let mut verification_alpha = alpha.to_vec();
    for &i in &candidates {
        verification_alpha[i] = 0.0;
    }
    let verified = crate::face_matting::recover_connected_portrait_fringe(
        source,
        &verification_alpha,
        owned,
        person,
        semantic_edge,
        explicit_forbidden,
        fringe,
    );
    for i in candidates {
        if verified[i] > 1e-6 {
            result.alpha[i] = verified[i];
            result.released[i] = true;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 40;
    const H: usize = 28;

    struct Fixture {
        source: RgbaImage,
        alpha: Vec<f32>,
        owned: Vec<bool>,
        person: Vec<f32>,
        semantic: Vec<f32>,
        explicit: Vec<bool>,
        inferred: Vec<bool>,
    }

    impl Fixture {
        fn neck_edge() -> Self {
            let count = W * H;
            let mut pixels = [20_u8, 30, 40, 255].repeat(count);
            let mut alpha = vec![0.0; count];
            let mut owned = vec![false; count];
            let mut person = vec![0.0; count];
            let mut semantic = vec![0.0; count];
            let explicit = vec![false; count];
            let mut inferred = vec![false; count];
            for y in 0..H {
                for x in 12..W {
                    let i = y * W + x;
                    // Independent physical truth: this entire half-plane is
                    // opaque skin; the parser omits three boundary columns.
                    person[i] = 1.0;
                    semantic[i] = 1.0;
                    let shift = if x < 15 && (8..=14).contains(&y) {
                        if y == 8 || y == 14 {
                            12
                        } else {
                            16
                        }
                    } else {
                        0
                    };
                    pixels[i * 4..i * 4 + 4].copy_from_slice(&[190, 130 + shift, 100 + shift, 255]);
                    if x >= 15 {
                        alpha[i] = 1.0;
                        owned[i] = true;
                    } else {
                        alpha[i] = 0.04;
                        inferred[i] = true;
                    }
                }
            }
            Self {
                source: RgbaImage::new(W as u32, H as u32, pixels).unwrap(),
                alpha,
                owned,
                person,
                semantic,
                explicit,
                inferred,
            }
        }

        fn recover(&self) -> Recovery {
            recover_inferred_fringe(
                &self.source,
                &self.alpha,
                &self.owned,
                &self.person,
                &self.semantic,
                &self.explicit,
                &self.inferred,
                5,
            )
        }
    }

    #[test]
    fn weak_nonzero_neck_fringe_recovers_physical_alpha_and_continuous_lighting() {
        let fixture = Fixture::neck_edge();
        let result = fixture.recover();
        for y in 2..H - 2 {
            for x in 12..15 {
                let i = y * W + x;
                // Rows 9..13 have an independently specified chromaticity
                // offset beyond the one-pixel color-line residual tolerance.
                // They require continuity from the compatible lit endpoints.
                assert!(result.released[i], "neck edge stayed frozen at {x},{y}");
                assert!(
                    (result.alpha[i] - 1.0).abs() < 0.01,
                    "opaque edge has alpha {} at {x},{y}",
                    result.alpha[i]
                );
            }
        }
        for i in 0..W * H {
            if !fixture.inferred[i] {
                assert_eq!(result.alpha[i], fixture.alpha[i]);
                assert!(!result.released[i]);
            }
        }
    }

    #[test]
    fn explicit_constraint_keeps_inferred_edge_frozen() {
        let mut fixture = Fixture::neck_edge();
        let i = 10 * W + 12;
        fixture.explicit[i] = true;
        let result = fixture.recover();
        assert_eq!(result.alpha[i], fixture.alpha[i]);
        assert!(!result.released[i]);
        assert!(result.released[6 * W + 12]);
    }

    #[test]
    fn same_colored_person_outside_connected_fringe_remains_fixed() {
        let mut fixture = Fixture::neck_edge();
        // Identical observed skin and person evidence cannot overcome the
        // absence of adjacency to the selected, connected portrait component.
        for y in 4..H - 4 {
            let i = y * W + 4;
            fixture.source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[190, 130, 100, 255]);
            fixture.alpha[i] = 0.04;
            fixture.person[i] = 1.0;
            fixture.semantic[i] = 1.0;
            fixture.inferred[i] = true;
        }
        let result = fixture.recover();
        for y in 4..H - 4 {
            let i = y * W + 4;
            assert_eq!(result.alpha[i], fixture.alpha[i]);
            assert!(!result.released[i]);
        }
        assert!(result.released[6 * W + 12]);
    }

    #[test]
    fn no_inferred_candidates_is_exact_identity() {
        let mut fixture = Fixture::neck_edge();
        fixture.inferred.fill(false);
        let result = fixture.recover();
        assert_eq!(result.alpha, fixture.alpha);
        assert!(result.released.iter().all(|&released| !released));
    }

    #[test]
    fn already_owned_material_is_not_reclassified_as_fringe() {
        let mut fixture = Fixture::neck_edge();
        let i = 10 * W + 16;
        fixture.inferred[i] = true;
        fixture.alpha[i] = 0.87;
        assert!(fixture.owned[i]);
        let result = fixture.recover();
        assert_eq!(result.alpha[i], 0.87);
        assert!(!result.released[i]);
    }

    #[test]
    fn nearby_inferred_pixel_with_incompatible_rgb_remains_fixed() {
        let mut fixture = Fixture::neck_edge();
        let i = 10 * W + 12;
        fixture.source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[20, 240, 30, 255]);
        let result = fixture.recover();
        assert_eq!(result.alpha[i], fixture.alpha[i]);
        assert!(!result.released[i]);
        assert!(result.released[6 * W + 12]);
    }
}
