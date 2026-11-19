use image_core::{
    adjust_body_with_layered_source, adjust_body_with_layered_source_with_reconstruction,
    prepare_body_inverse_displacement, BodyLayeredSource, RgbaImage, SourceReconstruction,
};
use serde_json::{json, Value};

struct BodyFixture {
    source: RgbaImage,
    front_alpha: Vec<f64>,
    front: Vec<f64>,
    rear_alpha: Vec<f64>,
    rear: Vec<f64>,
    scene: Vec<f64>,
}

impl BodyFixture {
    fn new() -> Self {
        let (w, h) = (120usize, 160usize);
        let mut f = Self {
            source: RgbaImage::new(w as u32, h as u32, vec![255; w * h * 4]).unwrap(),
            front_alpha: vec![0.0; w * h],
            front: [210.0, 50.0, 60.0].repeat(w * h),
            rear_alpha: vec![0.0; w * h],
            rear: vec![0.0; w * h * 3],
            scene: vec![0.0; w * h * 3],
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let bg = [50.0 + (x % 9) as f64, 80.0, 40.0 + (y % 13) as f64];
                f.scene[i * 3..i * 3 + 3].copy_from_slice(&bg);
                let rear = if (25..95).contains(&x) && (10..150).contains(&y) {
                    f.rear_alpha[i] = 1.0;
                    [80.0 + x as f64, 100.0, 220.0]
                } else {
                    bg
                };
                f.rear[i * 3..i * 3 + 3].copy_from_slice(&rear);
                if (20..39).contains(&x) && (40..69).contains(&y) {
                    f.front_alpha[i] = 1.0;
                }
                for c in 0..3 {
                    f.source.pixels[i * 4 + c] = if f.front_alpha[i] == 1.0 {
                        f.front[i * 3 + c] as u8
                    } else {
                        rear[c] as u8
                    };
                }
            }
        }
        f
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

    fn params(&self, amount: i32) -> Value {
        json!({"bodyPart":"waist", "region":{"x":0.01,"y":0.01,"width":0.98,"height":0.98},
            "landmarks":[{"x":0.5,"y":0.06},{"x":0.5,"y":0.5},{"x":0.5,"y":0.94}],
            "subjectMask":{"shape":"raster","width":120,"height":160,
                "data":self.rear_alpha.iter().map(|a|(a*255.0) as u8).collect::<Vec<_>>()},
            "adjustments":{"waist":amount}})
    }
}

fn observed_waist_width(image: &RgbaImage) -> usize {
    // Row chosen from source geometry. Blue identifies the visible rear cloth,
    // while neither the foreground sleeve nor textured scene is blue.
    (0..120)
        .filter(|x| image.pixels[(80 * 120 + x) * 4 + 2] > 150)
        .count()
}

#[test]
fn normal_parameters_change_visible_waist_width_both_ways_and_keep_front_fixed() {
    let f = BodyFixture::new();
    let original_width = observed_waist_width(&f.source);
    assert_eq!(original_width, 70);
    for amount in [-30, 30] {
        let output =
            adjust_body_with_layered_source(&f.source, &f.params(amount), &f.layers()).unwrap();
        let changed_width = observed_waist_width(&output);
        if amount > 0 {
            assert!(
                changed_width < original_width,
                "positive waist must visibly narrow the cloth: {changed_width}"
            );
        } else {
            assert!(
                changed_width > original_width,
                "negative waist must visibly widen the cloth: {changed_width}"
            );
        }
        for i in 0..f.front_alpha.len() {
            if f.front_alpha[i] == 1.0 || i % 120 < 5 || i % 120 > 114 {
                assert_eq!(
                    &output.pixels[i * 4..i * 4 + 4],
                    &f.source.pixels[i * 4..i * 4 + 4]
                );
            }
        }
    }
}

#[test]
fn zero_is_identity_but_does_not_accept_wrong_source_material() {
    let mut f = BodyFixture::new();
    assert_eq!(
        adjust_body_with_layered_source(&f.source, &f.params(0), &f.layers()).unwrap(),
        f.source
    );
    f.rear[0] += 1.0;
    for amount in [0, 30] {
        assert!(
            adjust_body_with_layered_source(&f.source, &f.params(amount), &f.layers()).is_err()
        );
    }
    f.rear[0] -= 1.0;
    f.source.pixels.pop();
    assert!(adjust_body_with_layered_source(&f.source, &f.params(30), &f.layers()).is_err());
}

#[test]
fn explicit_protection_and_zero_opacity_are_honored() {
    let f = BodyFixture::new();
    let mut params = f.params(30);
    params["protectMask"] =
        json!({"shape":"rect","region":{"x":0.15,"y":0.4,"width":0.2,"height":0.2}});
    let output = adjust_body_with_layered_source(&f.source, &params, &f.layers()).unwrap();
    for y in 65..95 {
        for x in 19..41 {
            let i = (y * 120 + x) * 4;
            assert_eq!(&output.pixels[i..i + 4], &f.source.pixels[i..i + 4]);
        }
    }
    assert_ne!(
        observed_waist_width(&output),
        observed_waist_width(&f.source)
    );
    params["opacity"] = json!(0);
    assert_eq!(
        adjust_body_with_layered_source(&f.source, &params, &f.layers()).unwrap(),
        f.source
    );
}

#[test]
fn explicit_quantized_adjustment_keeps_zero_front_protection_and_outside_byte_exact() {
    let mut f = BodyFixture::new();
    // All actual latent colors have a .4 fractional component while the
    // observed byte photograph is unchanged. The geometry still sees C.
    for value in f.front.iter_mut().chain(&mut f.rear).chain(&mut f.scene) {
        *value += 0.4;
    }
    let source_before = f.source.clone();
    assert!(adjust_body_with_layered_source(&f.source, &f.params(0), &f.layers()).is_err());
    let original_width = observed_waist_width(&f.source);
    for amount in [0, -30, 30] {
        let mut params = f.params(amount);
        params["protectMask"] =
            json!({"shape":"rect","region":{"x":0.15,"y":0.4,"width":0.2,"height":0.2}});
        let output = adjust_body_with_layered_source_with_reconstruction(
            &f.source, &params, &f.layers(), SourceReconstruction::NearestByte,
        ).unwrap();
        if amount == 0 {
            assert_eq!(output, f.source);
        } else if amount > 0 {
            assert!(observed_waist_width(&output) < original_width);
        } else {
            assert!(observed_waist_width(&output) > original_width);
        }
        for y in 0..160 {
            for x in 0..120 {
                let i = y * 120 + x;
                let protected = (65..95).contains(&y) && (19..41).contains(&x);
                if protected || f.front_alpha[i] == 1.0 || x < 5 || x > 114 {
                    assert_eq!(&output.pixels[i * 4..i * 4 + 4],
                        &f.source.pixels[i * 4..i * 4 + 4]);
                }
            }
        }
    }
    assert_eq!(f.source, source_before);
    // Even an identity request must reject a nonphysical prepared scene.
    f.scene[0] += 1.0;
    let error = adjust_body_with_layered_source_with_reconstruction(
        &f.source, &f.params(0), &f.layers(), SourceReconstruction::NearestByte,
    ).unwrap_err();
    assert!(error.to_string().contains("bounded material"));
}

#[test]
fn coarse_inner_geometry_does_not_cut_a_continuous_shirt_into_two_edges() {
    let f = BodyFixture::new();
    for amount in [-100, -60, -30, 30, 60, 100] {
        let mut params = f.params(amount);
        // A detector underestimates the real visible cloth by five pixels
        // on each side. This must not create a jump to identity inside it.
        params["partMask"] = json!({"shape":"rect", "region":{
            "x":0.25,"y":0.0625,"width":0.5,"height":0.875}});
        let field = prepare_body_inverse_displacement(&f.source, &params).unwrap();
        for x in 1..119 {
            let previous = (x - 1) as f32 + field[80 * 120 + x - 1][0];
            let current = x as f32 + field[80 * 120 + x][0];
            assert!(
                current > previous,
                "inverse coordinates folded at x={x}, amount={amount}: {previous}->{current}"
            );
        }
        let output = adjust_body_with_layered_source(&f.source, &params, &f.layers()).unwrap();
        let occupied: Vec<_> = (0..120)
            .filter(|x| output.pixels[(80 * 120 + x) * 4 + 2] > 150)
            .collect();
        assert_eq!(
            occupied.len(),
            occupied.last().unwrap() - occupied[0] + 1,
            "one continuous shirt must retain a single contour on each side"
        );
    }
}

#[test]
fn asymmetric_body_field_stays_ordered_with_selection_and_protection() {
    let mut f = BodyFixture::new();
    for y in 0..160 {
        let left = (25.0 + 6.0 * (y as f64 / 15.0).sin()).round() as usize;
        let right = (95.0 + 5.0 * (y as f64 / 19.0).cos()).round() as usize;
        for x in 0..120 {
            let i = y * 120 + x;
            let cloth = (10..150).contains(&y) && (left..right).contains(&x);
            f.rear_alpha[i] = if cloth { 1.0 } else { 0.0 };
            for c in 0..3 {
                f.rear[i * 3 + c] = if cloth {
                    [80.0 + x as f64, 100.0, 220.0][c]
                } else {
                    f.scene[i * 3 + c]
                };
                f.source.pixels[i * 4 + c] = if f.front_alpha[i] == 1.0 {
                    f.front[i * 3 + c]
                } else {
                    f.rear[i * 3 + c]
                } as u8;
            }
        }
    }
    for amount in [-100, -60, -30, 30, 60, 100] {
        for mode in ["unrestricted", "selected", "protected"] {
            let mut params = f.params(amount);
            if mode == "selected" {
                params["mask"] =
                    json!({"shape":"rect","region":{"x":0.12,"y":0.1,"width":0.76,"height":0.8}});
            } else if mode == "protected" {
                params["protectMask"] =
                    json!({"shape":"rect","region":{"x":0.63,"y":0.35,"width":0.08,"height":0.2}});
            }
            let field = prepare_body_inverse_displacement(&f.source, &params).unwrap();
            for y in 0..159 {
                for x in 0..119 {
                    let i = y * 120 + x;
                    let dx = [field[i + 1][0] - field[i][0], field[i + 1][1] - field[i][1]];
                    let dy = [
                        field[i + 120][0] - field[i][0],
                        field[i + 120][1] - field[i][1],
                    ];
                    let jacobian = (1.0 + dx[0]) * (1.0 + dy[1]) - dx[1] * dy[0];
                    assert!(
                        jacobian > 0.0,
                        "folded {mode} field at {x},{y}, amount={amount}, J={jacobian}"
                    );
                }
            }
        }
    }
}

#[test]
fn same_thigh_controls_combine_targets_before_inversion_and_opposites_cancel() {
    let (w, h) = (240usize, 160usize);
    let mut source = RgbaImage::new(w as u32, h as u32, [40, 60, 80, 255].repeat(w * h)).unwrap();
    let mut alpha = vec![0.0; w * h];
    for y in 10..150 {
        for x in 85..155 {
            let i = y * w + x;
            source.pixels[i * 4..i * 4 + 3].copy_from_slice(&[110, 150, 220]);
            alpha[i] = 1.0;
        }
    }
    let zero_alpha = vec![0.0; w * h];
    let rear: Vec<f64> = source
        .pixels
        .chunks_exact(4)
        .flat_map(|p| p[..3].iter().map(|v| *v as f64))
        .collect();
    let scene = [40.0, 60.0, 80.0].repeat(w * h);
    let layers = BodyLayeredSource {
        width: w as u32,
        height: h as u32,
        front_alpha: &zero_alpha,
        front_rgb: &scene,
        rear_alpha: &alpha,
        rear_source_rgb: &rear,
        scene_rgb: &scene,
    };
    let mut params = json!({"bodyPart":"left-thigh",
        "landmarks":[{"x":0.5,"y":0.125},{"x":0.5,"y":0.875}],
        "subjectMask":{"shape":"rect","region":{"x":85.0/240.0,"y":10.0/160.0,"width":70.0/240.0,"height":140.0/160.0}},
        "partMask":{"shape":"rect","region":{"x":85.0/240.0,"y":10.0/160.0,"width":70.0/240.0,"height":140.0/160.0}},
        "adjustments":{"legs":100,"thighs":100}});
    let output = adjust_body_with_layered_source(&source, &params, &layers).unwrap();
    let width = (0..w)
        .filter(|x| output.pixels[(80 * w + x) * 4 + 2] > 150)
        .count();
    // Two 26% contour requests yield 52% width reduction. Pixel sampling
    // permits one pixel of edge quantization, not silently halved strength.
    assert!(
        (33..=35).contains(&width),
        "combined requested contour width: {width}"
    );
    params["adjustments"]["thighs"] = json!(-100);
    assert_eq!(
        adjust_body_with_layered_source(&source, &params, &layers).unwrap(),
        source
    );
    assert!(prepare_body_inverse_displacement(&source, &params)
        .unwrap()
        .iter()
        .all(|d| *d == [0.0, 0.0]));
}
