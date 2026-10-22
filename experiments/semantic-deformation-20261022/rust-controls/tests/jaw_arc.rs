use semantic_deformation_controls::{arc::{generate, Arc, Boundary, Request}, FaceFrame};

fn request() -> Request {
    Request {
        frame: FaceFrame { origin: [0., 0.], across: [1., 0.], down: [0., 1.],
            bounds: [-20., 0., 20., 40.], midline_u: 0. },
        amount: 50., max_radius_fraction: 0.2,
        contour: vec![[-20., 0.], [20., 0.], [20., 20.], [20., 40.], [0., 40.], [-20., 40.], [-20., 20.]],
        arcs: vec![Arc { id: "right".into(), start: Boundary { edge: 1, t: 0. },
            end: Boundary { edge: 3, t: 0. }, direction: 1 },
            Arc { id: "left".into(), start: Boundary { edge: 0, t: 0. },
                end: Boundary { edge: 5, t: 0. }, direction: -1 }],
    }
}

#[test]
fn rectangle_arc_midpoints_move_while_endpoints_and_other_rows_are_exact() {
    let r = request();
    let output = generate(&r).unwrap();
    assert_eq!(output.controller, "jaw_arc_v1");
    assert_eq!(output.targets[2].target, [18., 20.]);
    assert_eq!(output.targets[6].target, [-18., 20.]);
    for i in [0, 1, 3, 4, 5] {
        assert_eq!(output.targets[i].source, r.contour[i]);
        assert_eq!(output.targets[i].target, r.contour[i]);
        assert_eq!(output.targets[i].point_gain, 0.);
    }
    assert_eq!(output.targets[2].point_gain, 1.);
    assert_eq!(output.targets[6].point_gain, 1.);
    assert!(output.arcs.iter().all(|a| a.length_px == 40.));
}

#[test]
fn zero_and_signed_edits_preserve_source_asymmetry_without_changing_height() {
    let mut r = request();
    for p in &mut r.contour { if p[0] < 0. { p[0] *= 2.; } }
    r.frame.bounds[0] = -40.;
    let before = r.contour.clone();
    let positive = generate(&r).unwrap();
    assert_eq!(positive.targets[2].target, [18., 20.]);
    assert_eq!(positive.targets[6].target, [-36., 20.]);
    r.amount = -50.;
    let negative = generate(&r).unwrap();
    for ((a, b), p) in positive.targets.iter().zip(&negative.targets).zip(&before) {
        assert_eq!(a.target[0]+b.target[0], 2.*p[0]);
        assert_eq!(a.target[1], p[1]);
        assert_eq!(b.target[1], p[1]);
        assert_eq!(a.point_gain, b.point_gain);
    }
    r.amount = 0.;
    assert!(generate(&r).unwrap().targets.iter().all(|p| p.source == p.target));
    assert_eq!(r.contour, before);
}

#[test]
fn quarter_gain_and_endpoint_derivatives_have_independent_expected_values() {
    let mut r = request();
    let e = 1e-3;
    r.contour = vec![[-20., 0.], [20., 0.], [20., e], [20., 2.*e], [20., 10.],
                    [20., 40.-2.*e], [20., 40.-e], [20., 40.], [-20., 40.]];
    r.arcs.truncate(1); r.arcs[0].end.edge = 7;
    let p = generate(&r).unwrap().targets;
    assert!((p[4].point_gain-27./64.).abs() < 1e-14);
    assert!((p[4].target[0]-19.15625).abs() < 1e-12);
    assert_eq!(p[4].target[1], 10.);
    for (a, b, c) in [(1, 2, 3), (7, 6, 5)] {
        assert_eq!(p[a].point_gain, 0.);
        let first = (p[b].point_gain-p[a].point_gain)/e;
        let second = (p[c].point_gain-2.*p[b].point_gain+p[a].point_gain)/(e*e);
        assert!(first.abs() < 1e-8, "endpoint first derivative {first}");
        assert!(second.abs() < 1e-5, "endpoint second derivative {second}");
    }
}

#[test]
fn rotation_translation_and_scale_preserve_the_original_arc_intent() {
    let mut r = request();
    let original = generate(&r).unwrap();
    let transform = |p: [f64; 2]| [13.-2.5*p[1], 17.+2.5*p[0]];
    r.frame.origin = transform(r.frame.origin);
    r.frame.across = [0., 1.]; r.frame.down = [-1., 0.];
    r.frame.bounds = r.frame.bounds.map(|v| v*2.5);
    for p in &mut r.contour { *p = transform(*p); }
    for (old, new) in original.targets.iter().zip(generate(&r).unwrap().targets) {
        assert_eq!(new.target, transform(old.target));
        assert_eq!(new.point_gain, old.point_gain);
    }
}

#[test]
fn continuous_overlap_between_rows_is_rejected_but_shared_endpoints_are_allowed() {
    let mut r = request();
    r.arcs[0] = Arc { id: "a".into(), start: Boundary { edge: 1, t: 0.1 },
        end: Boundary { edge: 1, t: 0.4 }, direction: 1 };
    r.arcs[1] = Arc { id: "b".into(), start: Boundary { edge: 1, t: 0.3 },
        end: Boundary { edge: 1, t: 0.6 }, direction: 1 };
    assert!(generate(&r).unwrap_err().contains("overlap"));
    r.arcs[1].start.t = 0.4;
    assert!(generate(&r).is_ok());
    // The same interval in reverse must also be caught.
    r.arcs[1].start.t = 0.6; r.arcs[1].end.t = 0.3; r.arcs[1].direction = -1;
    assert!(generate(&r).unwrap_err().contains("overlap"));
}

fn fractional_wrap_request() -> Request {
    Request {
        frame: FaceFrame { origin: [0.5, 0.5], across: [1., 0.], down: [0., 1.],
            bounds: [-0.5, -0.5, 0.5, 0.5], midline_u: 0. },
        amount: 50., max_radius_fraction: 0.2,
        contour: vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        arcs: vec![Arc { id: "wrapped".into(), start: Boundary { edge: 0, t: 0.8 },
            end: Boundary { edge: 0, t: 0.2 }, direction: 1 },
            Arc { id: "following".into(), start: Boundary { edge: 0, t: 0.2 },
                end: Boundary { edge: 0, t: 0.3 }, direction: 1 }],
    }
}

#[test]
fn wrapped_fractional_arcs_accept_a_shared_endpoint_without_reconstructing_it() {
    let r = fractional_wrap_request();
    let result = generate(&r).unwrap();
    assert_eq!(result.arcs[0].end_arclength_px, 0.2);
    assert_eq!(result.arcs[1].start_arclength_px, 0.2);
    let mut reversed = r;
    for a in &mut reversed.arcs {
        std::mem::swap(&mut a.start, &mut a.end);
        a.direction = -1;
    }
    assert!(generate(&reversed).is_ok());
}

#[test]
fn even_one_ulp_of_continuous_overlap_remains_rejected() {
    let mut r = fractional_wrap_request();
    let before = f64::from_bits(0.2f64.to_bits()-1);
    assert!(before < 0.2);
    r.arcs[1].start.t = before;
    assert!(generate(&r).unwrap_err().contains("overlap"));
    for a in &mut r.arcs {
        std::mem::swap(&mut a.start, &mut a.end);
        a.direction = -1;
    }
    assert!(generate(&r).unwrap_err().contains("overlap"));
}

#[test]
fn arcs_can_cross_row_zero_and_equivalent_endpoint_representations_are_identical() {
    let mut r = request();
    r.arcs = vec![Arc { id: "wrap".into(), start: Boundary { edge: 6, t: 0. },
        end: Boundary { edge: 1, t: 0. }, direction: 1 }];
    let first = generate(&r).unwrap();
    assert_eq!(first.arcs[0].length_px, 60.);
    assert!((first.targets[0].point_gain-512./729.).abs()<1e-14);
    assert_eq!(first.targets[1].target, r.contour[1]);
    assert_eq!(first.targets[6].target, r.contour[6]);
    r.arcs[0].end = Boundary { edge: 0, t: 1. };
    assert_eq!(serde_json::to_value(generate(&r).unwrap()).unwrap(), serde_json::to_value(first).unwrap());
    r.arcs[0].start = Boundary { edge: 1, t: 0. };
    assert!(generate(&r).unwrap_err().contains("distinct"));
}

#[test]
fn malformed_requests_are_rejected_without_reordering_or_clipping() {
    let mut cases = vec![];
    let mut r = request(); r.arcs[0].start.t = -0.001; cases.push(r);
    let mut r = request(); r.arcs[0].end.edge = r.contour.len(); cases.push(r);
    let mut r = request(); r.arcs[0].direction = 0; cases.push(r);
    let mut r = request(); r.arcs[1].id = r.arcs[0].id.clone(); cases.push(r);
    let mut r = request(); r.contour[1] = r.contour[0]; cases.push(r);
    let mut r = request(); r.amount = f64::INFINITY; cases.push(r);
    let mut r = request(); r.max_radius_fraction = 0.4501; cases.push(r);
    let mut r = request(); r.frame.across = [2., 0.]; cases.push(r);
    let mut r = request(); r.arcs.clear(); cases.push(r);
    for r in cases { assert!(generate(&r).is_err()); }
}

#[test]
fn crossing_contours_do_not_get_assigned_an_arclength_edit() {
    let mut r = request();
    r.contour.swap(1, 5);
    assert!(generate(&r).unwrap_err().contains("simple"));
}

#[test]
fn real_json_cli_emits_targets_and_rejects_unknown_or_invalid_fields() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let run = |input: &str| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_jaw-arc-targets"))
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
        child.wait_with_output().unwrap()
    };
    let request = serde_json::to_value(request()).unwrap();
    let valid = run(&request.to_string());
    assert!(valid.status.success(), "{}", String::from_utf8_lossy(&valid.stderr));
    let result: serde_json::Value = serde_json::from_slice(&valid.stdout).unwrap();
    assert_eq!(result["controller"], "jaw_arc_v1");
    assert_eq!(result["targets"][2]["target"], serde_json::json!([18.,20.]));
    let mut unknown = request.clone(); unknown["arcs"][0]["automatic_visible"] = true.into();
    let mut outside = request.clone(); outside["amount"] = 101.into();
    for input in [unknown.to_string(), outside.to_string(), "{}".into()] {
        let output = run(&input);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(serde_json::from_slice::<serde_json::Value>(&output.stderr).unwrap()["error"].is_string());
    }
}
