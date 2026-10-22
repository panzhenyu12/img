//! Explicit source-path targets for an independently selected contour arc.
use crate::{FaceFrame, Point};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub edge: usize,
    pub t: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arc {
    pub id: String,
    pub start: Boundary,
    pub end: Boundary,
    pub direction: i8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub frame: FaceFrame,
    pub amount: f64,
    pub max_radius_fraction: f64,
    pub contour: Vec<Point>,
    pub arcs: Vec<Arc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target {
    pub index: usize,
    pub source: Point,
    pub target: Point,
    pub point_gain: f64,
    pub arc_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolvedArc {
    pub id: String,
    pub start_arclength_px: f64,
    pub end_arclength_px: f64,
    pub length_px: f64,
    pub direction: i8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    pub schema_version: u32,
    pub controller: String,
    pub coordinate_space: String,
    pub targets: Vec<Target>,
    pub arcs: Vec<ResolvedArc>,
    pub limitations: Vec<String>,
}

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
}

fn on_segment(a: Point, b: Point, p: Point) -> bool {
    (a[0].min(b[0])..=a[0].max(b[0])).contains(&p[0])
        && (a[1].min(b[1])..=a[1].max(b[1])).contains(&p[1])
}

fn validate_simple_contour(p: &[Point]) -> Result<(), String> {
    let n = p.len();
    let mut twice_area = 0.;
    for i in 0..n {
        let (a, b, c) = (p[i], p[(i+1)%n], p[(i+2)%n]);
        let turn = cross(a, b, c);
        if !turn.is_finite() {
            return Err("simple contour geometry overflowed".into());
        }
        if turn == 0. && ((c != b && on_segment(a, b, c)) || (a != b && on_segment(b, c, a))) {
            return Err("source contour must be simple without adjacent backtracking".into());
        }
        twice_area += cross(p[0], a, b);
        for j in i+2..n {
            if i == 0 && j == n-1 { continue; }
            let (c, d) = (p[j], p[(j+1)%n]);
            let turns = [cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)];
            if turns.iter().any(|t| !t.is_finite()) {
                return Err("simple contour geometry overflowed".into());
            }
            let proper = turns[0].signum() != turns[1].signum()
                && turns[2].signum() != turns[3].signum()
                && turns.iter().all(|&t| t != 0.);
            let touching = (turns[0] == 0. && on_segment(a,b,c))
                || (turns[1] == 0. && on_segment(a,b,d))
                || (turns[2] == 0. && on_segment(c,d,a))
                || (turns[3] == 0. && on_segment(c,d,b));
            if proper || touching {
                return Err("source contour must be simple without crossings or nonadjacent touches".into());
            }
        }
    }
    if !twice_area.is_finite() || twice_area == 0. {
        return Err("source contour must be simple with finite nonzero area".into());
    }
    Ok(())
}

pub fn generate(request: &Request) -> Result<Output, String> {
    // Reuse the established frame/amount/calibration checks without changing
    // any existing controller behavior or silently normalizing source axes.
    crate::validate(&crate::Request {
        frame: request.frame.clone(),
        amount: request.amount,
        calibration: crate::Calibration {
            max_radius_fraction: request.max_radius_fraction,
            chin_fade_start: 0.88,
        },
        regional_profile: None,
        contour: request.contour.iter().enumerate().map(|(i, &source)|
            crate::ContourPoint { id: i.to_string(), source }).collect(),
        parts: vec![], relations: vec![],
    })?;
    if request.arcs.is_empty() {
        return Err("at least one explicitly selected source arc is required".into());
    }
    validate_simple_contour(&request.contour)?;
    let n = request.contour.len();
    let mut lengths = Vec::with_capacity(n);
    let mut cumulative = vec![0.];
    for i in 0..n {
        let a = request.contour[i]; let b = request.contour[(i+1)%n];
        let length = (b[0]-a[0]).hypot(b[1]-a[1]);
        if !length.is_finite() || length <= 0. {
            return Err("source contour edges must have finite positive length".into());
        }
        lengths.push(length);
        let next = cumulative[i]+length;
        if !next.is_finite() || next <= cumulative[i] {
            return Err("source arclength cannot represent all edges".into());
        }
        cumulative.push(next);
    }
    let perimeter = cumulative[n];
    let boundary = |p: &Boundary| -> Result<f64, String> {
        if p.edge >= n || !p.t.is_finite() || !(0. ..=1.).contains(&p.t) {
            return Err("arc boundary needs an existing edge and finite t in [0,1]".into());
        }
        // A closed path's last endpoint is the same point as arclength zero.
        // This does not clip or reorder the supplied boundary.
        Ok((cumulative[p.edge]+p.t*lengths[p.edge]).rem_euclid(perimeter))
    };
    let mut ids = std::collections::HashSet::new();
    let mut arcs = Vec::with_capacity(request.arcs.len());
    for arc in &request.arcs {
        if arc.id.trim().is_empty() || !ids.insert(arc.id.as_str()) || ![-1, 1].contains(&arc.direction) {
            return Err("arcs need distinct nonempty ids and direction +1 or -1".into());
        }
        let start = boundary(&arc.start)?; let end = boundary(&arc.end)?;
        let length = (f64::from(arc.direction)*(end-start)).rem_euclid(perimeter);
        if length == 0. {
            return Err("arc endpoints must be distinct on the source path".into());
        }
        arcs.push(ResolvedArc { id: arc.id.clone(), start_arclength_px: start,
            end_arclength_px: end, length_px: length, direction: arc.direction });
    }
    // Compare continuous circular intervals, not only existing contour rows:
    // two short arcs may overlap entirely between sampled source points.
    let intervals = |a: &ResolvedArc| -> Vec<(f64, f64)> {
        let (start, end) = if a.direction == 1 {
            (a.start_arclength_px, a.end_arclength_px)
        } else {
            (a.end_arclength_px, a.start_arclength_px)
        };
        // Split using the original canonical endpoints. Reconstructing end
        // as start+length-perimeter can round a shared endpoint beyond itself,
        // inventing an overlap. Exact interval comparisons also retain real
        // overlaps as small as one representable step; no epsilon is applied.
        if start < end { vec![(start, end)] }
        else { vec![(start, perimeter), (0., end)] }
    };
    for i in 0..arcs.len() {
        for j in i+1..arcs.len() {
            for a in intervals(&arcs[i]) {
                for b in intervals(&arcs[j]) {
                    if a.1.min(b.1) > a.0.max(b.0) {
                        return Err("selected source arcs overlap over a positive length".into());
                    }
                }
            }
        }
    }
    let mut targets = Vec::with_capacity(n);
    for (index, &source) in request.contour.iter().enumerate() {
        let mut gain = 0.; let mut arc_id = None;
        for a in &arcs {
            let distance = (f64::from(a.direction)*(cumulative[index]-a.start_arclength_px)).rem_euclid(perimeter);
            if distance > 0. && distance < a.length_px {
                let t = distance/a.length_px;
                gain = 64.*(t*(1.-t)).powi(3);
                arc_id = Some(a.id.clone());
            }
        }
        let target = if gain == 0. || request.amount == 0. || request.max_radius_fraction == 0. {
            source
        } else {
            let frame = &request.frame;
            let relative = [source[0]-frame.origin[0], source[1]-frame.origin[1]];
            let radius = crate::dot(relative, frame.across)-frame.midline_u;
            let d = -request.amount/100.*request.max_radius_fraction*radius*gain;
            crate::add(source, [d*frame.across[0], d*frame.across[1]])
        };
        if !crate::finite(target) {
            return Err("arc target computation overflowed".into());
        }
        targets.push(Target { index, source, target, point_gain: gain, arc_id });
    }
    Ok(Output {
        schema_version: 1, controller: "jaw_arc_v1".into(),
        coordinate_space: "original_source_pixels".into(), targets, arcs,
        limitations: vec![
            "Explicit selected source arcs only; no detection, visibility, anatomy or automatic editing authorization is inferred.".into(),
            "The compact gain is C2 at its source-arclength endpoints; no final dense-field continuity, injectivity or aesthetic guarantee is asserted.".into(),
            "This is a local jaw editing intent, not an equal-area or equal-strength substitute for whole-face width.".into(),
            "Attachments, dense mapping, material preservation and rendering require separate validation.".into(),
            "Opposite signs are independent edits of the original source, not sequential inverse edits.".into(),
        ],
    })
}
