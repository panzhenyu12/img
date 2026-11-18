//! Disjoint material ownership for portrait deformation.
//!
//! A detector's masks are evidence, not a compositing layer.  In particular,
//! a weak face/neck transition must not silently become a scene donor: doing
//! that is the source of the dark or coloured seams that appear when a jaw is
//! contracted.  This module defines the small, deterministic ownership policy
//! used by the next semantic compositor candidate.  It is intentionally kept
//! separate from the current renderer until it passes the real-photo holdout.

/// Material classes used by a portrait compositor.  `Unknown` is deliberate:
/// an ambiguous transition is safer to abstain from than to label as scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Owner {
    Scene,
    MovingFace,
    NeckUnderlay,
    Hair,
    Accessory,
    Unknown,
}

/// Per-pixel semantic evidence in the normalized 0..=1 range.  Values are
/// probabilities/coverage from independent masks; they are not mixed RGB.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Evidence {
    pub face: f32,
    pub neck: f32,
    pub hair: f32,
    pub accessory: f32,
    pub subject: f32,
    pub protected: f32,
}

fn finite(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Resolve mutually exclusive material ownership.
///
/// Precedence follows visible occlusion: explicit user protection/accessories,
/// hair, the moving face, and then the observed neck underlay.  A transition
/// with evidence from multiple incompatible classes remains `Unknown` unless
/// one class wins by a clear margin.  This prevents the old failure mode where
/// `known_layer == 0` was interpreted as clean background.
pub(crate) fn classify(e: Evidence) -> Owner {
    if [e.face, e.neck, e.hair, e.accessory, e.subject, e.protected]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return Owner::Unknown;
    }
    let e = Evidence {
        face: finite(e.face),
        neck: finite(e.neck),
        hair: finite(e.hair),
        accessory: finite(e.accessory),
        subject: finite(e.subject),
        protected: finite(e.protected),
    };
    const HARD: f32 = 0.80;
    const SOFT: f32 = 0.20;
    const MARGIN: f32 = 0.15;

    if e.protected >= HARD || e.accessory >= HARD {
        return Owner::Accessory;
    }
    // Hair is an observed occluder.  It owns the boundary whenever it has
    // strong evidence, even when a soft neck raster overlaps at the join.
    if e.hair >= HARD {
        return Owner::Hair;
    }
    if e.face >= HARD {
        // Face is the front moving layer.  A strong face class wins over a
        // strong neck class only when the two masks are not tied at a seam.
        if e.neck < HARD || e.face >= e.neck + MARGIN {
            return Owner::MovingFace;
        }
    }
    if e.neck >= HARD && e.face < HARD {
        return Owner::NeckUnderlay;
    }

    // A single weak class can still be a useful observed underlay, but mixed
    // weak evidence must abstain.  `subject` alone is not enough to infer a
    // clean scene pixel: it may denote clothing or another person.
    let active = [e.face, e.neck, e.hair, e.accessory, e.subject]
        .into_iter()
        .filter(|value| *value >= SOFT)
        .count();
    if active == 0 {
        return Owner::Scene;
    }
    if active == 1 {
        if e.neck >= SOFT {
            return Owner::NeckUnderlay;
        }
        if e.face >= SOFT {
            return Owner::MovingFace;
        }
        if e.hair >= SOFT {
            return Owner::Hair;
        }
    }
    Owner::Unknown
}

/// Whether an owner is safe as a donor for a newly exposed scene hole.
pub(crate) fn clean_scene_donor(owner: Owner) -> bool {
    matches!(owner, Owner::Scene)
}

/// Whether an owner is a valid observed lower-layer donor for a contracted
/// face.  Unknown and moving foreground pixels are intentionally excluded.
pub(crate) fn clean_underlay_donor(owner: Owner) -> bool {
    matches!(owner, Owner::NeckUnderlay | Owner::Scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(face: f32, neck: f32, hair: f32, accessory: f32, subject: f32) -> Evidence {
        Evidence {
            face,
            neck,
            hair,
            accessory,
            subject,
            protected: 0.0,
        }
    }

    #[test]
    fn clean_scene_requires_absence_of_semantic_evidence() {
        assert_eq!(classify(e(0.0, 0.0, 0.0, 0.0, 0.0)), Owner::Scene);
        assert!(!clean_scene_donor(classify(e(0.0, 0.0, 0.0, 0.0, 0.25))));
        assert_eq!(classify(e(0.0, 0.0, 0.0, 0.0, 0.25)), Owner::Unknown);
    }

    #[test]
    fn hair_owns_a_neck_overlap_at_the_visible_join() {
        assert_eq!(classify(e(0.0, 0.95, 0.96, 0.0, 1.0)), Owner::Hair);
        assert!(!clean_underlay_donor(classify(e(
            0.0, 0.95, 0.96, 0.0, 1.0
        ))));
    }

    #[test]
    fn face_wins_over_neck_only_with_a_clear_front_layer_margin() {
        assert_eq!(classify(e(0.96, 0.80, 0.0, 0.0, 1.0)), Owner::MovingFace);
        assert_eq!(classify(e(0.90, 0.90, 0.0, 0.0, 1.0)), Owner::Unknown);
        assert!(!clean_underlay_donor(classify(e(
            0.90, 0.90, 0.0, 0.0, 1.0
        ))));
    }

    #[test]
    fn protection_and_accessory_have_highest_precedence() {
        let mut evidence = e(1.0, 1.0, 1.0, 0.9, 1.0);
        assert_eq!(classify(evidence), Owner::Accessory);
        evidence.accessory = 0.0;
        evidence.protected = 0.81;
        assert_eq!(classify(evidence), Owner::Accessory);
    }

    #[test]
    fn non_finite_values_cannot_become_donors() {
        assert_eq!(
            classify(e(f32::NAN, f32::INFINITY, 0.0, 0.0, 0.0)),
            Owner::Unknown
        );
    }
}
