use image_core::{composite_body_source_layers, BodyLayeredSource, RgbaImage};

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
