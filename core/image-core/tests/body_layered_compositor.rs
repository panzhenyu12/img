use image_core::{
    composite_body_source_layers, composite_body_source_layers_with_reconstruction,
    BodyLayeredSource, RgbaImage, SourceReconstruction,
};

#[test]
fn quantized_transparent_source_does_not_transport_rounding_error_with_the_field() {
    let source = RgbaImage::new(2, 1, vec![10, 10, 10, 255, 11, 11, 11, 255]).unwrap();
    let scene = [10.49, 10.49, 10.49, 10.51, 10.51, 10.51];
    let layers = BodyLayeredSource {
        width: 2,
        height: 1,
        front_alpha: &[0.0; 2],
        front_rgb: &[0.0; 6],
        rear_alpha: &[0.0; 2],
        rear_source_rgb: &scene,
        scene_rgb: &scene,
    };
    assert!(composite_body_source_layers(&source, &layers, &[[0.0, 0.0]; 2]).is_err(),
        "the existing exact-source contract must remain strict");
    for dx in [0.0, -1.0, -0.5, -0.00001, 0.00001, 0.5, 1.0] {
        let result = composite_body_source_layers_with_reconstruction(
            &source, &layers, &[[dx, 0.0]; 2], SourceReconstruction::NearestByte,
        ).unwrap();
        // There is no moving material. P=0 and the destination B stays put.
        // Using observed C instead of latent S would move +/-0.49 rounding
        // residuals and change x0 from10 to11 under a one-pixel shift.
        assert_eq!(result, source, "quantization noise moved at dx={dx}");
    }
}

#[test]
fn quantized_material_transport_uses_latent_premultiplication_without_rounding_residuals() {
    let source = RgbaImage::new(2, 1, vec![50, 50, 50, 255, 100, 100, 100, 255]).unwrap();
    let scene = [0.0, 0.0, 0.0, (100.4 - 10.6) / 0.75,
        (100.4 - 10.6) / 0.75, (100.4 - 10.6) / 0.75];
    let lower = [50.4, 50.4, 50.4, 100.4, 100.4, 100.4];
    let layers = BodyLayeredSource {
        width: 2, height: 1, front_alpha: &[0.0; 2], front_rgb: &[0.0; 6],
        rear_alpha: &[1.0, 0.25], rear_source_rgb: &lower, scene_rgb: &scene,
    };
    let out = composite_body_source_layers_with_reconstruction(
        &source, &layers, &[[1.0, 0.0], [0.0, 0.0]], SourceReconstruction::NearestByte,
    ).unwrap();
    // The supplied source material is P=10.6 over a black destination, so
    // the result must be11. Using observed C to recover P gives10.2; adding
    // destination C-S=-.4 after transport also gives10.2. Both wrongly yield10.
    assert_eq!(&out.pixels[..4], &[11, 11, 11, 255]);
    assert_eq!(&out.pixels[4..], &source.pixels[4..]);

    let source = RgbaImage::new(2, 1, vec![0, 0, 0, 255, 100, 100, 100, 255]).unwrap();
    let lower = [0.0, 0.0, 0.0, 100.4, 100.4, 100.4];
    let scene = [0.0, 0.0, 0.0, 100.4 / 0.75, 100.4 / 0.75, 100.4 / 0.75];
    let layers = BodyLayeredSource {
        width: 2, height: 1, front_alpha: &[0.0; 2], front_rgb: &[0.0; 6],
        rear_alpha: &[0.0, 0.25], rear_source_rgb: &lower, scene_rgb: &scene,
    };
    // P=0 is valid on a bright source and remains bounded on a black scene;
    // substituting rounded C would incorrectly infer P=-.4.
    let out = composite_body_source_layers_with_reconstruction(
        &source, &layers, &[[1.0, 0.0], [0.0, 0.0]], SourceReconstruction::NearestByte,
    ).unwrap();
    assert_eq!(out, source);
}

#[test]
fn quantized_reconstruction_cells_are_half_open_and_never_extend_the_color_domain() {
    for (byte, latent, accepted) in [
        (100, 99.5, true), (100, 100.499999, true),
        (100, 100.5, false), (100, 99.499999, false),
        (0, 0.0, true), (0, 0.499999, true), (0, 0.5, false),
        (0, -0.1, false), (255, 254.5, true), (255, 255.0, true),
        (255, 255.1, false), (255, 255.5, false),
        (100, f64::NAN, false),
    ] {
        let source = RgbaImage::new(1, 1, vec![byte, byte, byte, 255]).unwrap();
        let lower = [latent; 3];
        let layers = BodyLayeredSource {
            width: 1, height: 1, front_alpha: &[0.0], front_rgb: &[0.0; 3],
            rear_alpha: &[1.0], rear_source_rgb: &lower, scene_rgb: &[0.0; 3],
        };
        let output = composite_body_source_layers_with_reconstruction(
            &source, &layers, &[[0.0, 0.0]], SourceReconstruction::NearestByte,
        );
        assert_eq!(output.is_ok(), accepted, "C={byte}, S={latent}");
        if accepted { assert_eq!(output.unwrap(), source); }
    }
}

#[test]
fn quantized_soft_front_composes_before_validation_and_tiny_motion_jumps_at_most_one_byte() {
    for lower_at_zero in [100.4, 100.999999] {
        let source = RgbaImage::new(2, 1, vec![70, 70, 70, 255, 90, 90, 90, 255]).unwrap();
        let lower = [lower_at_zero, lower_at_zero, lower_at_zero, 108.4, 108.4, 108.4];
        let layers = BodyLayeredSource {
            width: 2, height: 1, front_alpha: &[0.5, 1.0],
            front_rgb: &[40.0, 40.0, 40.0, 90.4, 90.4, 90.4],
            rear_alpha: &[1.0; 2], rear_source_rgb: &lower, scene_rgb: &[0.0; 6],
        };
        let mut previous = source.clone();
        for dx in [-0.00001, 0.0, 0.00001] {
            let out = composite_body_source_layers_with_reconstruction(
                &source, &layers, &[[dx, 0.0]; 2], SourceReconstruction::NearestByte,
            ).unwrap();
            // The first source is Q(.5*40+.5*100.4)=70, not Q(rear)=100.
            // The second legitimately crosses a rounding threshold under a
            // tiny motion: continuity of latent RGB allows a one-byte step.
            let expected = if lower_at_zero > 100.9 && dx > 0.0 { 71 } else { 70 };
            assert_eq!(&out.pixels[..4], &[expected, expected, expected, 255]);
            assert_eq!(&out.pixels[4..], &source.pixels[4..], "opaque front moved");
            assert!(out.pixels.iter().zip(&previous.pixels)
                .all(|(&a, &b)| a.abs_diff(b) <= 1));
            previous = out;
        }
    }
}

#[test]
fn quantized_zero_motion_still_rejects_a_physically_impossible_scene() {
    let source = RgbaImage::new(1, 1, vec![100, 100, 100, 255]).unwrap();
    let layers = BodyLayeredSource {
        width: 1, height: 1, front_alpha: &[0.0], front_rgb: &[0.0; 3],
        rear_alpha: &[0.0], rear_source_rgb: &[100.4; 3], scene_rgb: &[100.5; 3],
    };
    let error = composite_body_source_layers_with_reconstruction(
        &source, &layers, &[[0.0, 0.0]], SourceReconstruction::NearestByte,
    ).unwrap_err();
    // Both colors independently round to plausible bytes, but P=-.1 is not
    // material. Quantization tolerance must never relax the physical bound.
    assert!(error.to_string().contains("bounded material"));
}

struct Fixture {
    source: RgbaImage,
    front_alpha: Vec<f64>,
    front: Vec<f64>,
    rear_alpha: Vec<f64>,
    rear: Vec<f64>,
    scene: Vec<f64>,
}

impl Fixture {
    fn new() -> Self {
        let (width, height) = (16usize, 5usize);
        let mut fixture = Self {
            source: RgbaImage::new(width as u32, height as u32, vec![255; width * height * 4])
                .unwrap(),
            front_alpha: vec![0.0; width * height],
            front: [210.0, 30.0, 40.0].repeat(width * height),
            rear_alpha: vec![0.0; width * height],
            rear: [30.0, 60.0, 90.0].repeat(width * height),
            scene: [30.0, 60.0, 90.0].repeat(width * height),
        };
        for i in 0..width * height {
            let x = i % width;
            if (3..13).contains(&x) {
                fixture.rear_alpha[i] = 1.0;
                // Multiples of four keep fractional sleeve edges exactly
                // representable in the observed RGB8 photograph.
                fixture.rear[i * 3..i * 3 + 3].copy_from_slice(&[
                    100.0 + x as f64 * 4.0,
                    102.0,
                    80.0,
                ]);
            }
            fixture.front_alpha[i] = match x {
                6..=8 => 1.0,
                5 => 0.25,
                9 => 0.75,
                _ => 0.0,
            };
            let a = fixture.front_alpha[i];
            for c in 0..3 {
                let value = a * fixture.front[i * 3 + c] + (1.0 - a) * fixture.rear[i * 3 + c];
                // Choose exact observed source by solving the mixed front
                // red value instead of rounding its reconstruction contract.
                if value.fract() != 0.0 {
                    fixture.front[i * 3 + c] += (value.round() - value) / a;
                }
                fixture.source.pixels[i * 4 + c] = value.round() as u8;
            }
        }
        fixture
    }

    fn layers(&self) -> BodyLayeredSource<'_> {
        BodyLayeredSource {
            width: self.source.width,
            height: self.source.height,
            front_alpha: &self.front_alpha,
            front_rgb: &self.front,
            rear_alpha: &self.rear_alpha,
            rear_source_rgb: &self.rear,
            scene_rgb: &self.scene,
        }
    }
}

#[test]
fn mixed_edge_zero_and_tiny_signed_motion_preserve_the_observed_photo() {
    let f = Fixture::new();
    for amount in [0.0, -0.00001, 0.00001] {
        let field = vec![[amount, 0.0]; f.front_alpha.len()];
        assert_eq!(
            composite_body_source_layers(&f.source, &f.layers(), &field).unwrap(),
            f.source
        );
    }
}

#[test]
fn rear_moves_both_directions_without_duplicating_or_moving_the_front_sleeve() {
    let f = Fixture::new();
    let source_before = f.source.clone();
    for amount in [-2.0, 2.0] {
        let field = vec![[amount, 0.0]; f.front_alpha.len()];
        let out = composite_body_source_layers(&f.source, &f.layers(), &field).unwrap();
        let destination = if amount < 0.0 { 10usize } else { 4usize };
        let sampled = (destination as f32 + amount) as usize;
        assert_eq!(f.front_alpha[sampled], 1.0);
        for c in 0..3 {
            assert_eq!(
                out.pixels[destination * 4 + c],
                f.rear[sampled * 3 + c] as u8,
                "sampling under an old opaque sleeve must reveal the lower material"
            );
        }
        assert_ne!(out, f.source);
        for i in 0..f.front_alpha.len() {
            if f.front_alpha[i] == 1.0 {
                assert_eq!(
                    &out.pixels[i * 4..i * 4 + 4],
                    &f.source.pixels[i * 4..i * 4 + 4]
                );
            }
        }
        assert!(field.iter().all(|&d| d == [amount, 0.0]));
    }
    assert_eq!(f.source, source_before);
}

#[test]
fn zero_motion_does_not_bypass_source_pairing_and_rear_feasibility() {
    let mut f = Fixture::new();
    let field = vec![[0.0, 0.0]; f.front_alpha.len()];
    f.rear[0] += 1.0;
    assert!(composite_body_source_layers(&f.source, &f.layers(), &field)
        .unwrap_err()
        .to_string()
        .contains("reconstruct"));
    f.rear[0] -= 1.0;
    f.scene[0] += 1.0;
    assert!(composite_body_source_layers(&f.source, &f.layers(), &field)
        .unwrap_err()
        .to_string()
        .contains("bounded material"));
}

#[test]
fn dimensions_invalid_values_and_transparent_source_are_rejected() {
    let mut f = Fixture::new();
    let field = vec![[0.0, 0.0]; f.front_alpha.len()];
    assert!(
        composite_body_source_layers(&f.source, &f.layers(), &field[..field.len() - 1]).is_err()
    );
    for bad in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
        f.front_alpha[0] = bad;
        assert!(composite_body_source_layers(&f.source, &f.layers(), &field).is_err());
    }
    f.front_alpha[0] = 0.0;
    let mut bad_field = field.clone();
    bad_field[2][1] = f32::NAN;
    assert!(composite_body_source_layers(&f.source, &f.layers(), &bad_field).is_err());
    f.front[0] = 256.0;
    assert!(composite_body_source_layers(&f.source, &f.layers(), &field).is_err());
    f.front[0] = 210.0;
    f.source.pixels[3] = 254;
    assert!(composite_body_source_layers(&f.source, &f.layers(), &field).is_err());
}

#[test]
fn fractional_edges_are_continuous_and_zero_field_pixels_are_exact() {
    let f = Fixture::new();
    let mut previous = None;
    for amount in [-0.1, -0.05, 0.0, 0.05, 0.1] {
        let mut field = vec![[amount, 0.0]; f.front_alpha.len()];
        field[3] = [0.0, 0.0];
        let out = composite_body_source_layers(&f.source, &f.layers(), &field).unwrap();
        assert_eq!(&out.pixels[12..16], &f.source.pixels[12..16]);
        if let Some(before) = previous {
            let difference = out
                .pixels
                .iter()
                .zip(before)
                .map(|(&a, b): (&u8, u8)| (a as i16 - b as i16).abs())
                .max()
                .unwrap();
            assert!(
                difference <= 8,
                "smooth small movement must not introduce a finite mixed-edge jump"
            );
        }
        previous = Some(out.pixels);
    }
}

#[test]
fn soft_rear_edge_transports_its_material_over_the_destination_scene() {
    let source = RgbaImage::new(
        4,
        1,
        [10u8, 60, 100, 40]
            .into_iter()
            .flat_map(|v| [v, v, v, 255])
            .collect(),
    )
    .unwrap();
    let rear: Vec<_> = [10.0, 60.0, 100.0, 40.0]
        .into_iter()
        .flat_map(|v| [v; 3])
        .collect();
    let scene: Vec<_> = [10.0, 20.0, 30.0, 40.0]
        .into_iter()
        .flat_map(|v| [v; 3])
        .collect();
    let layers = BodyLayeredSource {
        width: 4,
        height: 1,
        front_alpha: &[0.0; 4],
        front_rgb: &[0.0; 12],
        rear_alpha: &[0.0, 0.5, 1.0, 0.0],
        rear_source_rgb: &rear,
        scene_rgb: &scene,
    };
    let out = composite_body_source_layers(&source, &layers, &[[0.5, 0.0]; 4]).unwrap();
    // At x0, transported premultiplied material is 25 and coverage .25.
    // The destination scene is 10, so 25 + .75*10 = 32.5 -> 33.
    assert_eq!(&out.pixels[0..4], &[33, 33, 33, 255]);
    // At x2, 50 + .5*30 = 65. Warping the observed RGB would give 70.
    assert_eq!(&out.pixels[8..12], &[65, 65, 65, 255]);
}

#[test]
fn outside_canvas_samples_clamp_without_transparent_border_bands() {
    let source = RgbaImage::new(2, 1, vec![100, 100, 100, 255, 180, 180, 180, 255]).unwrap();
    let layers = BodyLayeredSource {
        width: 2,
        height: 1,
        front_alpha: &[0.0; 2],
        front_rgb: &[0.0; 6],
        rear_alpha: &[1.0; 2],
        rear_source_rgb: &[100.0, 100.0, 100.0, 180.0, 180.0, 180.0],
        scene_rgb: &[0.0; 6],
    };
    let out = composite_body_source_layers(
        &source,
        &layers,
        &[[f32::MAX, f32::MAX], [f32::MIN, f32::MIN]],
    )
    .unwrap();
    assert_eq!(out.pixels, vec![180, 180, 180, 255, 100, 100, 100, 255]);
}

#[test]
fn admitted_source_roundoff_remains_admissible_when_transported_to_a_black_scene() {
    let source = RgbaImage::new(2, 1, vec![0, 0, 0, 255, 100, 100, 100, 255]).unwrap();
    let lower = [
        0.0,
        0.0,
        0.0,
        255.0 - 1.3e-12,
        255.0 - 1.3e-12,
        255.0 - 1.3e-12,
    ];
    let layers = BodyLayeredSource {
        width: 2,
        height: 1,
        front_alpha: &[0.0, 1.0],
        front_rgb: &[0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
        rear_alpha: &[0.0, 0.0],
        rear_source_rgb: &lower,
        scene_rgb: &[0.0, 0.0, 0.0, 255.0, 255.0, 255.0],
    };
    // The tolerated source calculation loses 1.3e-12 at a bright background.
    // Transport must carry that arithmetic error bound to the black scene,
    // rather than reject a source pair already admitted by this same API.
    let out = composite_body_source_layers(&source, &layers, &[[1.0, 0.0], [0.0, 0.0]]).unwrap();
    assert_eq!(out, source);
}

#[test]
fn two_dimensional_interpolation_composes_both_soft_layers_analytically() {
    let triples = |values: [f64; 4]| values.into_iter().flat_map(|v| [v; 3]).collect::<Vec<_>>();
    let source = RgbaImage::new(
        2,
        2,
        [140u8, 200, 80, 180]
            .into_iter()
            .flat_map(|v| [v, v, v, 255])
            .collect(),
    )
    .unwrap();
    let rear = triples([60.0, 200.0, 60.0, 97.5]);
    let scene = triples([20.0, 40.0, 60.0, 80.0]);
    let front = triples([220.0, 0.0, 140.0, 180.0]);
    let layers = BodyLayeredSource {
        width: 2,
        height: 2,
        front_alpha: &[0.5, 0.0, 0.25, 1.0],
        front_rgb: &front,
        rear_alpha: &[0.5, 1.0, 0.0, 0.25],
        rear_source_rgb: &rear,
        scene_rgb: &scene,
    };
    let out = composite_body_source_layers(
        &source,
        &layers,
        &[[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    .unwrap();
    // P_rear average=71.875; a=.4375; destination B=20 => rear=83.125.
    // The stationary half-covered front contributes110: result151.5625.
    assert_eq!(&out.pixels[..4], &[152, 152, 152, 255]);
    assert_eq!(&out.pixels[4..], &source.pixels[4..]);
}
