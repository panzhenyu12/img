//! Source-coordinate target generation only. This crate does not warp pixels.
use serde::{Deserialize, Serialize};

pub mod arc;

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
pub struct ProfileKnot {
    pub height: f64,
    pub gain: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionalProfile {
    pub knots: Vec<ProfileKnot>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regional_profile: Option<RegionalProfile>,
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
    let profile = request.regional_profile.as_ref().map(ProfileSpline::new).transpose()?;
    let mut result = Output {
        schema_version: 1,
        controller: if profile.is_some() { "regional_contour_v1" } else { "whole_face_width_v1" }.into(),
        coordinate_space: "original_source_pixels".into(),
        targets: request.contour.iter().map(|p| Target {
            id: p.id.clone(), part_id: None, source: p.source,
            target: add(p.source, displacement(request, profile.as_ref(), p.source)),
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
                    translation: displacement(request, profile.as_ref(), anchor),
                }
            }
            Motion::Editable => ResolvedMotion::Editable,
        };
        for (index, source) in part.source_points.iter().copied().enumerate() {
            let translation = match motion {
                ResolvedMotion::FixedPosition => [0., 0.],
                ResolvedMotion::ShapePreservingTranslation { translation, .. } => translation,
                ResolvedMotion::Editable => displacement(request, profile.as_ref(), source),
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
    if profile.is_some() {
        result.limitations.push("Regional gain is an explicit new target intent, not an equal-strength repair of whole_face_width_v1 or an aesthetic certificate.".into());
        result.limitations.push("Profile interpolation is C1 between knots only; constant exterior extension may have an endpoint slope jump. No dense warp continuity is asserted.".into());
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
    if let Some(profile) = &request.regional_profile {
        if profile.knots.len() < 2
            || profile.knots.first().unwrap().height != 0.
            || profile.knots.last().unwrap().height != 1.
            || profile.knots.iter().any(|k| !k.height.is_finite() || !k.gain.is_finite()
                || k.gain < 0. || !(calibration.max_radius_fraction * k.gain).is_finite()
                || calibration.max_radius_fraction * k.gain > 0.45)
            || profile.knots.windows(2).any(|pair| pair[0].height >= pair[1].height)
        {
            return Err("regional profile needs increasing finite heights from 0 to 1, nonnegative finite gains, and calibration*gain <= 0.45".into());
        }
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

fn displacement(request: &Request, profile: Option<&ProfileSpline>, p: Point) -> Point {
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
    let d = match profile { Some(spline) => d * spline.at(height), None => d };
    [d * frame.across[0], d * frame.across[1]]
}

/// Shape-preserving cubic Hermite interpolation of explicitly supplied gains.
/// The weighted harmonic interior slopes are the PCHIP rule; endpoint slopes
/// use its one-sided monotonicity limiter. No input gain or edit is normalized.
struct ProfileSpline<'a> {
    knots: &'a [ProfileKnot],
    slopes: Vec<f64>,
}

impl<'a> ProfileSpline<'a> {
    fn new(profile: &'a RegionalProfile) -> Result<Self, String> {
        let knots = &profile.knots;
        let widths: Vec<f64> = knots.windows(2).map(|k| k[1].height-k[0].height).collect();
        let secants: Vec<f64> = knots.windows(2).zip(&widths)
            .map(|(k, h)| (k[1].gain-k[0].gain)/h).collect();
        if secants.iter().any(|v| !v.is_finite()) {
            return Err("regional knot spacing produces nonfinite slopes".into());
        }
        let mut slopes = vec![0.; knots.len()];
        if knots.len() == 2 {
            slopes.fill(secants[0]);
        } else {
            for i in 1..knots.len()-1 {
                let (a, b) = (secants[i-1], secants[i]);
                if a != 0. && b != 0. && a.signum() == b.signum() {
                    let w1 = 2.*widths[i]+widths[i-1];
                    let w2 = widths[i]+2.*widths[i-1];
                    // Scale the harmonic mean to avoid overflow for tiny
                    // secants or cancellation between opposite slopes.
                    let scale = a.abs().min(b.abs());
                    slopes[i] = scale * ((w1+w2)/(w1*(scale/a)+w2*(scale/b)));
                }
            }
            let end = |h0: f64, h1: f64, d0: f64, d1: f64| {
                let slope = ((2.*h0+h1)*d0-h0*d1)/(h0+h1);
                if slope.signum() != d0.signum() { 0. }
                else if d0.signum() != d1.signum() && slope.abs() > 3.*d0.abs() { 3.*d0 }
                else { slope }
            };
            slopes[0] = end(widths[0], widths[1], secants[0], secants[1]);
            let n = widths.len();
            slopes[n] = end(widths[n-1], widths[n-2], secants[n-1], secants[n-2]);
        }
        if slopes.iter().any(|v| !v.is_finite()) {
            return Err("regional interpolation slope overflowed".into());
        }
        Ok(Self { knots, slopes })
    }

    fn at(&self, height: f64) -> f64 {
        if height <= 0. { return self.knots[0].gain; }
        if height >= 1. { return self.knots.last().unwrap().gain; }
        let right = self.knots.partition_point(|k| k.height < height);
        if self.knots[right].height == height { return self.knots[right].gain; }
        let left = right-1;
        let (a, b) = (&self.knots[left], &self.knots[right]);
        if a.gain == b.gain { return a.gain; }
        let h = b.height-a.height;
        let t = (height-a.height)/h; let u = 1.-t;
        // Bernstein evaluation is stable near endpoints and uses nonnegative
        // basis weights. PCHIP slopes preserve monotonicity of each interval.
        let c1 = a.gain+h*self.slopes[left]/3.;
        let c2 = b.gain-h*self.slopes[right]/3.;
        u*u*u*a.gain + 3.*u*u*t*c1 + 3.*u*t*t*c2 + t*t*t*b.gain
    }
}
