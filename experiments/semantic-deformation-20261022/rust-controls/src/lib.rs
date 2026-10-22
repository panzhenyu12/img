//! Source-coordinate target generation only. This crate does not warp pixels.
use serde::{Deserialize, Serialize};

pub type Point = [f64; 2];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaceFrame {
    pub origin: Point,
    pub across: Point,
    pub down: Point,
    pub bounds: [f64; 4],
    pub midline_u: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub max_radius_fraction: f64,
    pub chin_fade_start: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContourPoint {
    pub id: String,
    pub source: Point,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Motion {
    FixedPosition,
    ShapePreservingTranslation { anchor: Point },
    Editable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub id: String,
    pub label: String,
    pub source_points: Vec<Point>,
    pub motion: Motion,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationStatus {
    Hypothesis,
    UserConfirmed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationEvidence {
    pub provenance: String,
    pub status: RelationStatus,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contact {
    pub on_a: Point,
    pub on_b: Point,
}

/// Relations are evidence, not constraints silently inferred from labels/masks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Relation {
    Connected {
        a: String,
        b: String,
        contacts: Vec<Contact>,
        evidence: RelationEvidence,
    },
    Occludes {
        front: String,
        back: String,
        boundary: Vec<Point>,
        evidence: RelationEvidence,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub frame: FaceFrame,
    pub calibration: Calibration,
    pub amount: f64,
    pub contour: Vec<ContourPoint>,
    #[serde(default)]
    pub parts: Vec<Part>,
    #[serde(default)]
    pub relations: Vec<Relation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target {
    pub id: String,
    pub part_id: Option<String>,
    pub source: Point,
    pub target: Point,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedMotion {
    FixedPosition,
    ShapePreservingTranslation { anchor: Point, translation: Point },
    Editable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartRule {
    pub id: String,
    pub label: String,
    pub motion: ResolvedMotion,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    pub schema_version: u32,
    pub controller: String,
    pub coordinate_space: String,
    pub targets: Vec<Target>,
    pub part_rules: Vec<PartRule>,
    pub relations: Vec<Relation>,
    pub limitations: Vec<String>,
}

pub fn generate(request: &Request) -> Result<Output, String> {
    validate(request)?;
    let mut result = Output {
        schema_version: 1,
        controller: "whole_face_width_v1".into(),
        coordinate_space: "original_source_pixels".into(),
        targets: request.contour.iter().map(|p| Target {
            id: p.id.clone(), part_id: None, source: p.source,
            target: add(p.source, displacement(request, p.source)),
        }).collect(),
        part_rules: vec![],
        relations: request.relations.clone(),
        limitations: vec![
            "Targets only: no dense mapping, rasterization, detection, fill or aesthetic acceptance.".into(),
            "Relations are passed through as evidence; no connection or occlusion is enforced here.".into(),
            "Point targets and part rules need a feasibility-aware downstream solver.".into(),
            "Opposite signs are independent edits of the source, not sequential inverse edits.".into(),
        ],
    };
    for part in &request.parts {
        let motion = match part.motion {
            Motion::FixedPosition => ResolvedMotion::FixedPosition,
            Motion::ShapePreservingTranslation { anchor } => {
                ResolvedMotion::ShapePreservingTranslation {
                    anchor,
                    translation: displacement(request, anchor),
                }
            }
            Motion::Editable => ResolvedMotion::Editable,
        };
        for (index, source) in part.source_points.iter().copied().enumerate() {
            let translation = match motion {
                ResolvedMotion::FixedPosition => [0., 0.],
                ResolvedMotion::ShapePreservingTranslation { translation, .. } => translation,
                ResolvedMotion::Editable => displacement(request, source),
            };
            result.targets.push(Target {
                id: format!("{}:{index}", part.id),
                part_id: Some(part.id.clone()),
                source,
                target: add(source, translation),
            });
        }
        result.part_rules.push(PartRule {
            id: part.id.clone(),
            label: part.label.clone(),
            motion,
        });
    }
    if result.targets.iter().any(|t| !finite(t.target)) {
        return Err("target computation overflowed".into());
    }
    Ok(result)
}

fn add(a: Point, b: Point) -> Point {
    if b == [0., 0.] {
        a
    } else {
        [a[0] + b[0], a[1] + b[1]]
    }
}
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn finite(p: Point) -> bool {
    p.iter().all(|v| v.is_finite())
}

fn validate(request: &Request) -> Result<(), String> {
    use std::collections::HashSet;
    let frame = &request.frame;
    let [left, top, right, bottom] = frame.bounds;
    if !finite(frame.origin)
        || !finite(frame.across)
        || !finite(frame.down)
        || frame.bounds.iter().any(|v| !v.is_finite())
        || !frame.midline_u.is_finite()
        || !(right - left).is_finite()
        || !(bottom - top).is_finite()
        || right <= left
        || bottom <= top
        || frame.midline_u <= left
        || frame.midline_u >= right
    {
        return Err("invalid finite frame bounds, origin or midline".into());
    }
    let determinant = frame.across[0] * frame.down[1] - frame.across[1] * frame.down[0];
    // Native source frames are stored as f32: accept their quantization without
    // silently normalizing directions and changing the original target request.
    const FRAME_TOLERANCE: f64 = 1e-6;
    if (dot(frame.across, frame.across) - 1.).abs() > FRAME_TOLERANCE
        || (dot(frame.down, frame.down) - 1.).abs() > FRAME_TOLERANCE
        || dot(frame.across, frame.down).abs() > FRAME_TOLERANCE
        || (determinant - 1.).abs() > FRAME_TOLERANCE
    {
        return Err("frame across/down must be unit orthogonal vectors with determinant +1".into());
    }
    if !request.amount.is_finite() || !(-100. ..=100.).contains(&request.amount) {
        return Err("amount must be finite and within [-100,100]".into());
    }
    let calibration = &request.calibration;
    if !calibration.max_radius_fraction.is_finite()
        || !(0. ..=0.45).contains(&calibration.max_radius_fraction)
        || !calibration.chin_fade_start.is_finite()
        || !(0. ..1.).contains(&calibration.chin_fade_start)
    {
        return Err(
            "calibration requires max_radius_fraction in [0,0.45] and chin_fade_start in [0,1)"
                .into(),
        );
    }
    if request.contour.len() < 3 {
        return Err("at least three source contour points required".into());
    }
    let mut target_ids = HashSet::new();
    for p in &request.contour {
        if p.id.trim().is_empty() || !target_ids.insert(p.id.clone()) || !finite(p.source) {
            return Err("contour points require unique nonempty ids and finite coordinates".into());
        }
    }
    let mut parts = HashSet::new();
    for part in &request.parts {
        if part.id.trim().is_empty()
            || !parts.insert(part.id.as_str())
            || part.label.trim().is_empty()
            || part.source_points.is_empty()
            || part.source_points.iter().any(|p| !finite(*p))
        {
            return Err(
                "parts require unique nonempty ids, a label and finite nonempty source_points"
                    .into(),
            );
        }
        if let Motion::ShapePreservingTranslation { anchor } = part.motion {
            if !finite(anchor) {
                return Err("translation anchor must be finite".into());
            }
        }
        for index in 0..part.source_points.len() {
            if !target_ids.insert(format!("{}:{index}", part.id)) {
                return Err("contour/part target ids collide".into());
            }
        }
    }
    for relation in &request.relations {
        let (a, b, evidence) = match relation {
            Relation::Connected {
                a,
                b,
                contacts,
                evidence,
            } => {
                if contacts.is_empty()
                    || contacts.iter().any(|c| !finite(c.on_a) || !finite(c.on_b))
                {
                    return Err("connection requires finite explicit contact pairs".into());
                }
                (a, b, evidence)
            }
            Relation::Occludes {
                front,
                back,
                boundary,
                evidence,
            } => {
                if boundary.len() < 2 || boundary.iter().any(|p| !finite(*p)) {
                    return Err(
                        "occlusion requires at least two finite observed boundary points".into(),
                    );
                }
                (front, back, evidence)
            }
        };
        if a == b || !parts.contains(a.as_str()) || !parts.contains(b.as_str()) {
            return Err("relations must reference two distinct declared parts".into());
        }
        if evidence.provenance.trim().is_empty()
            || !evidence.confidence.is_finite()
            || !(0. ..=1.).contains(&evidence.confidence)
        {
            return Err(
                "relation requires provenance, explicit status and confidence in [0,1]".into(),
            );
        }
    }
    Ok(())
}

fn displacement(request: &Request, p: Point) -> Point {
    if request.amount == 0. {
        return [0., 0.];
    }
    let frame = &request.frame;
    let relative = [p[0] - frame.origin[0], p[1] - frame.origin[1]];
    let radius = dot(relative, frame.across) - frame.midline_u;
    let height =
        (dot(relative, frame.down) - frame.bounds[1]) / (frame.bounds[3] - frame.bounds[1]);
    let fade_start = request.calibration.chin_fade_start;
    let t = ((height - fade_start) / (1. - fade_start)).clamp(0., 1.);
    let envelope = 1. - t * t * (3. - 2. * t);
    let d = -request.amount / 100. * request.calibration.max_radius_fraction * radius * envelope;
    [d * frame.across[0], d * frame.across[1]]
}
