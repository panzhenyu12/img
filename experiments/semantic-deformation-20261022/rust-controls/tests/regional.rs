use semantic_deformation_controls::*;

fn request(knots: &[(f64, f64)]) -> Request {
    let mut r: Request = serde_json::from_str(include_str!("../examples/whole-width.json")).unwrap();
    r.regional_profile = Some(RegionalProfile {
        knots: knots.iter().map(|&(height, gain)| ProfileKnot { height, gain }).collect(),
    });
    r.parts.clear();
    r.relations.clear();
    r.amount = 100.;
    r.calibration.max_radius_fraction = 0.2;
    r.calibration.chin_fade_start = 0.99;
    r
}

fn targets(r: &Request, heights: &[f64]) -> Vec<Point> {
    let mut r = r.clone();
    r.contour = heights.iter().enumerate().map(|(i, &h)| ContourPoint {
        id: i.to_string(), source: [50., 200.*h],
    }).collect();
    generate(&r).unwrap().targets.into_iter().map(|v| v.target).collect()
}

#[test]
fn unity_profile_preserves_all_legacy_targets_including_chin_and_parts() {
    let mut r: Request = serde_json::from_str(include_str!("../examples/whole-width.json")).unwrap();
    let old = generate(&r).unwrap();
    r.regional_profile = Some(RegionalProfile { knots: vec![
        ProfileKnot { height: 0., gain: 1. }, ProfileKnot { height: 0.43, gain: 1. },
        ProfileKnot { height: 1., gain: 1. },
    ]});
    let new = generate(&r).unwrap();
    for (a, b) in old.targets.iter().zip(new.targets) {
        assert_eq!(a.target, b.target);
    }
    assert_eq!(new.controller, "regional_contour_v1");
}

#[test]
fn regional_dip_keeps_other_anchors_and_flat_regions_unchanged() {
    let r = request(&[(0., 1.), (0.2, 1.), (0.4, 1.), (0.5, 0.5),
                      (0.6, 1.), (0.8, 1.), (1., 1.)]);
    let result = targets(&r, &[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]);
    for index in [0, 1, 2, 3, 7, 8, 9] {
        assert_eq!(result[index][0], 40.);
    }
    assert_eq!(result[5][0], 45.);
    assert!(result[4][0] > 40. && result[4][0] < 45.);
    assert!(result[6][0] > 40. && result[6][0] < 45.);
}

#[test]
fn monotone_intervals_do_not_overshoot_and_internal_derivatives_join() {
    let knots = [(0., 0.15), (0.2, 0.6), (0.4, 1.1), (0.6, 0.4), (0.8, 0.9), (1., 1.)];
    let r = request(&knots);
    for pair in knots.windows(2).take(4) {
        let heights: Vec<_> = (0..=100).map(|i| pair[0].0+(pair[1].0-pair[0].0)*i as f64/100.).collect();
        let result = targets(&r, &heights);
        let gains: Vec<_> = result.iter().map(|p| (50.-p[0])/10.).collect();
        for gain in &gains {
            assert!(*gain >= pair[0].1.min(pair[1].1)-1e-12);
            assert!(*gain <= pair[0].1.max(pair[1].1)+1e-12);
        }
        let sign = (pair[1].1-pair[0].1).signum();
        assert!(gains.windows(2).all(|p| sign*(p[1]-p[0]) >= -1e-12));
    }
    let epsilon = 1e-6;
    for &(h, _) in &knots[1..5] {
        let p = targets(&r, &[h-epsilon, h, h+epsilon]);
        let left = (p[1][0]-p[0][0])/epsilon;
        let right = (p[2][0]-p[1][0])/epsilon;
        assert!((left-right).abs() < 0.003, "derivative jump at {h}: {left}/{right}");
    }
}

#[test]
fn regional_sign_zero_and_similarity_transform_use_the_original_frame() {
    let mut r = request(&[(0., 1.), (0.4, 0.6), (0.7, 1.3), (1., 1.)]);
    r.amount = 60.;
    let positive = generate(&r).unwrap();
    r.amount = -60.;
    let negative = generate(&r).unwrap();
    for (a, b) in positive.targets.iter().zip(negative.targets) {
        assert!((a.target[0]+b.target[0]-2.*a.source[0]).abs() < 1e-12);
        assert_eq!(a.target[1]+b.target[1], 2.*a.source[1]);
    }
    r.amount = 0.;
    assert!(generate(&r).unwrap().targets.iter().all(|p| p.target == p.source));
    r.amount = 60.;
    let transform = |p: Point| [13.-2.5*p[1], 17.+2.5*p[0]];
    r.frame.origin = transform(r.frame.origin);
    r.frame.across = [0., 1.]; r.frame.down = [-1., 0.];
    r.frame.bounds = r.frame.bounds.map(|v| 2.5*v);
    for point in &mut r.contour { point.source = transform(point.source); }
    for (a, b) in positive.targets.iter().zip(generate(&r).unwrap().targets) {
        let expected = transform(a.target);
        assert!((expected[0]-b.target[0]).abs() < 1e-12);
        assert!((expected[1]-b.target[1]).abs() < 1e-12);
    }
}

#[test]
fn part_translation_uses_same_profile_anchor_without_activating_relations() {
    let mut r = request(&[(0., 1.), (0.5, 0.5), (1., 1.)]);
    r.parts = vec![
        Part { id: "fixed".into(), label: "hair".into(), source_points: vec![[40., 100.]], motion: Motion::FixedPosition },
        Part { id: "eye".into(), label: "eye".into(), source_points: vec![[30., 80.], [50., 120.]],
               motion: Motion::ShapePreservingTranslation { anchor: [40., 100.] } },
    ];
    r.relations.push(Relation::Connected { a: "fixed".into(), b: "eye".into(),
        contacts: vec![Contact { on_a: [40., 100.], on_b: [40., 100.] }],
        evidence: RelationEvidence { provenance: "test hypothesis only".into(), status: RelationStatus::Hypothesis, confidence: 0.5 } });
    let output = generate(&r).unwrap();
    let points: Vec<_> = output.targets.iter().filter(|t| t.part_id.is_some()).collect();
    assert_eq!(points[0].target, [40., 100.]);
    assert_eq!(points[1].target, [26., 80.]);
    assert_eq!(points[2].target, [46., 120.]);
    assert_eq!(output.relations.len(), 1);
}

#[test]
fn invalid_profiles_fail_without_sorting_clamping_or_normalizing() {
    for knots in [vec![], vec![(0., 1.)], vec![(0.1, 1.), (1., 1.)],
                  vec![(0., 1.), (0.9, 1.)], vec![(0., 1.), (0.7, 1.), (0.6, 1.), (1., 1.)],
                  vec![(0., 1.), (0.5, 1.), (0.5, 1.), (1., 1.)],
                  vec![(0., 1.), (0.5, -0.01), (1., 1.)],
                  vec![(0., 1.), (0.5, f64::NAN), (1., 1.)],
                  vec![(0., 1.), (0.5, 2.26), (1., 1.)]] {
        assert!(generate(&request(&knots)).is_err(), "accepted {knots:?}");
    }
}
