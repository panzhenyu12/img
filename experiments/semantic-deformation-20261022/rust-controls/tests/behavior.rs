use semantic_deformation_controls::*;

#[test]
fn cached_f32_frames_remain_usable_without_normalizing_the_request() {
    for across in [
        [0.9975677132606506, -0.06970371305942535],
        [0.9976974129676819, -0.06782297790050507],
    ] {
        let mut req = request();
        req.frame.across = across;
        req.frame.down = [-across[1], across[0]];
        assert!(generate(&req).is_ok());
    }
    let mut invalid = request();
    invalid.frame.across = [1.00001, 0.];
    assert!(generate(&invalid).is_err());
}

fn request() -> Request {
    Request {
        frame: FaceFrame {
            origin: [0., 0.],
            across: [1., 0.],
            down: [0., 1.],
            bounds: [-100., 0., 100., 200.],
            midline_u: 0.,
        },
        calibration: Calibration {
            max_radius_fraction: 0.2,
            chin_fade_start: 0.88,
        },
        amount: 100.,
        contour: vec![
            ContourPoint {
                id: "upper".into(),
                source: [-80., 20.],
            },
            ContourPoint {
                id: "cheek".into(),
                source: [100., 96.],
            },
            ContourPoint {
                id: "chin".into(),
                source: [10., 200.],
            },
        ],
        parts: vec![],
        relations: vec![],
    }
}

#[test]
fn targets_are_equivariant_under_image_rotation_translation_and_scale() {
    let mut source = request();
    source.frame.midline_u = 10.;
    source.parts.push(Part {
        id: "eye".into(),
        label: "eye".into(),
        source_points: vec![[30., 50.], [50., 50.], [40., 60.]],
        motion: Motion::ShapePreservingTranslation { anchor: [40., 55.] },
    });
    let baseline = generate(&source).unwrap();
    let scale = 2.5;
    let rotate = |p: Point| [-p[1], p[0]];
    let transform = |p: Point| [17. - scale * p[1], -31. + scale * p[0]];
    let mut transformed = source.clone();
    transformed.frame.origin = transform(source.frame.origin);
    transformed.frame.across = rotate(source.frame.across);
    transformed.frame.down = rotate(source.frame.down);
    transformed.frame.bounds = source.frame.bounds.map(|v| scale * v);
    transformed.frame.midline_u *= scale;
    for p in &mut transformed.contour {
        p.source = transform(p.source);
    }
    for part in &mut transformed.parts {
        for p in &mut part.source_points {
            *p = transform(*p);
        }
        if let Motion::ShapePreservingTranslation { anchor } = &mut part.motion {
            *anchor = transform(*anchor);
        }
    }
    let result = generate(&transformed).unwrap();
    for (a, b) in baseline.targets.iter().zip(&result.targets) {
        let expected = transform(a.target);
        assert!((expected[0] - b.target[0]).abs() < 1e-12);
        assert!((expected[1] - b.target[1]).abs() < 1e-12);
    }
}

#[test]
fn relation_hypotheses_remain_explicit_and_do_not_override_motion() {
    let mut req = request();
    req.parts = vec![
        Part {
            id: "hair".into(),
            label: "hair".into(),
            source_points: vec![[80., 60.]],
            motion: Motion::FixedPosition,
        },
        Part {
            id: "face".into(),
            label: "skin".into(),
            source_points: vec![[80., 70.]],
            motion: Motion::Editable,
        },
    ];
    req.relations = vec![Relation::Occludes {
        front: "hair".into(),
        back: "face".into(),
        boundary: vec![[80., 60.], [80., 70.]],
        evidence: RelationEvidence {
            provenance: "segmentation boundary, occlusion not independently confirmed".into(),
            status: RelationStatus::Hypothesis,
            confidence: 0.6,
        },
    }];
    let output = generate(&req).unwrap();
    assert_eq!(output.targets[3].target, [80., 60.]);
    assert_eq!(output.targets[4].target, [64., 70.]);
    assert!(matches!(
        output.relations[0],
        Relation::Occludes {
            evidence: RelationEvidence {
                status: RelationStatus::Hypothesis,
                ..
            },
            ..
        }
    ));
    if let Relation::Occludes { back, .. } = &mut req.relations[0] {
        *back = "unknown".into();
    }
    assert!(generate(&req).is_err());
}

#[test]
fn calibration_has_a_smooth_chin_transition_and_does_not_need_eye_count() {
    let mut req = request();
    req.contour = vec![
        ContourPoint {
            id: "before".into(),
            source: [50., 175.999],
        },
        ContourPoint {
            id: "start".into(),
            source: [50., 176.],
        },
        ContourPoint {
            id: "after".into(),
            source: [50., 176.001],
        },
        ContourPoint {
            id: "middle".into(),
            source: [50., 188.],
        },
        ContourPoint {
            id: "tip".into(),
            source: [50., 200.],
        },
    ];
    let output = generate(&req).unwrap();
    assert_eq!(output.targets[0].target[0], 40.);
    assert_eq!(output.targets[1].target[0], 40.);
    assert!((output.targets[2].target[0] - 40.).abs() < 1e-7);
    assert!((output.targets[3].target[0] - 45.).abs() < 1e-12);
    assert_eq!(output.targets[4].target[0], 50.);
}

#[test]
fn invalid_inputs_are_rejected_without_guessing_or_clamping() {
    let invalid = [
        {
            let mut r = request();
            r.amount = 101.;
            r
        },
        {
            let mut r = request();
            r.amount = f64::NAN;
            r
        },
        {
            let mut r = request();
            r.frame.across = [2., 0.];
            r
        },
        {
            let mut r = request();
            r.frame.down = [1., 0.];
            r
        },
        {
            let mut r = request();
            r.frame.bounds[3] = r.frame.bounds[1];
            r
        },
        {
            let mut r = request();
            r.calibration.chin_fade_start = 1.;
            r
        },
        {
            let mut r = request();
            r.calibration.max_radius_fraction = 0.9;
            r
        },
        {
            let mut r = request();
            r.contour[0].source[0] = f64::INFINITY;
            r
        },
        {
            let mut r = request();
            r.contour[1].id = r.contour[0].id.clone();
            r
        },
        {
            let mut r = request();
            r.contour.clear();
            r
        },
    ];
    for r in invalid {
        assert!(generate(&r).is_err(), "accepted invalid request: {r:?}");
    }
}

#[test]
fn part_motion_is_operation_specific_and_shape_preserving_does_not_lock_position() {
    let mut req = request();
    let points = vec![[30., 60.], [50., 60.], [40., 70.]];
    // The same semantic label is deliberately assigned three different roles.
    req.parts = vec![
        Part {
            id: "fixed".into(),
            label: "eye".into(),
            source_points: vec![[20., 50.]],
            motion: Motion::FixedPosition,
        },
        Part {
            id: "moving".into(),
            label: "eye".into(),
            source_points: points.clone(),
            motion: Motion::ShapePreservingTranslation { anchor: [40., 60.] },
        },
        Part {
            id: "editable".into(),
            label: "eye".into(),
            source_points: vec![[60., 60.]],
            motion: Motion::Editable,
        },
    ];
    let result = generate(&req).unwrap();
    let targets = |id: &str| {
        result
            .targets
            .iter()
            .filter(|t| t.part_id.as_deref() == Some(id))
            .collect::<Vec<_>>()
    };
    assert_eq!(targets("fixed")[0].target, [20., 50.]);
    let moved = targets("moving");
    assert_eq!(moved.len(), 3);
    assert_eq!(moved[0].target, [22., 60.]);
    assert_eq!(moved[1].target, [42., 60.]);
    assert_eq!(moved[2].target, [32., 70.]);
    assert_eq!(targets("editable")[0].target, [48., 60.]);
    assert_eq!(result.part_rules.len(), 3);
}

#[test]
fn signed_whole_width_keeps_upper_active_and_chin_connected() {
    let mut req = request();
    req.amount = 0.;
    let zero = generate(&req).unwrap();
    assert!(zero.targets.iter().all(|p| p.source == p.target));
    req.amount = 100.;
    let narrow = generate(&req).unwrap();
    assert_eq!(narrow.targets[0].target, [-64., 20.]);
    assert_eq!(narrow.targets[1].target, [80., 96.]);
    assert_eq!(narrow.targets[2].target, [10., 200.]);
    req.amount = -100.;
    let wide = generate(&req).unwrap();
    assert_eq!(wide.targets[0].target, [-96., 20.]);
    assert_eq!(wide.targets[1].target, [120., 96.]);
    for (a, b) in narrow.targets.iter().zip(&wide.targets) {
        assert_eq!(
            [a.target[0] + b.target[0], a.target[1] + b.target[1]],
            [2. * a.source[0], 2. * a.source[1]]
        );
    }
}
