use super::*;

#[test]
fn source_scene_prior_is_not_duplicated_as_a_high_precision_mixture_estimate() {
    // Independently specified truth: a=.1, F=[250,100,80], B=[200,200,200].
    // Conservative projection of source C into the model's +/-8 envelope is
    // [205,192,192]. That nearest-to-C point is an alpha-bound witness, not B.
    let alpha = [0., 0., 0.1, 0.1, 0.1, 0., 0.];
    let clean = [200_u8, 200, 200, 255];
    let observed = [205_u8, 190, 188, 255];
    let source = RgbaImage::new(
        7,
        1,
        alpha
            .iter()
            .flat_map(|&a| if a > 0. { observed } else { clean })
            .collect(),
    )
    .unwrap();
    let holes = alpha.map(|a| a > 0.);
    let projected_prior = RgbaImage::new(
        7,
        1,
        holes
            .iter()
            .flat_map(|&hole| if hole { [205, 192, 192, 255] } else { clean })
            .collect(),
    )
    .unwrap();
    let independent_scene: Vec<_> = holes
        .iter()
        .map(|&hole| hole.then_some([200.; 3]))
        .collect();
    let unpaired = reconstruct_with_scene(
        &source,
        &holes,
        &alpha,
        &[false; 7],
        &[false; 7],
        &[0; 7],
        Some(&projected_prior),
        Some(&independent_scene),
    )
    .unwrap();
    assert_eq!(
        &unpaired.pixels[3 * 4..3 * 4 + 4],
        &[205, 192, 192, 255],
        "unpaired legacy reconstruction must retain its existing prior behavior"
    );
    let mut background = reconstruct_source_background(
        &source,
        &holes,
        &alpha,
        &[false; 7],
        &[false; 7],
        &[0; 7],
        Some(&projected_prior),
        Some(&independent_scene),
        true,
    )
    .unwrap();
    layer_pair::pair(&source, &alpha, &mut background).unwrap();
    for i in 0..7 {
        assert_eq!(
            &background.pixels[i * 4..i * 4 + 4],
            &clean,
            "source projection must not dominate the clean independent scene at {i}"
        );
        let (pixel, membership) =
            layer_pair::sample_residual(&source, &alpha, &background, i as f32, 0.);
        for c in 0..3 {
            let restored = byte(pixel[c] + (1. - membership) * background.pixels[i * 4 + c] as f32);
            assert_eq!(
                restored,
                source.pixels[i * 4 + c],
                "source color pairing changed at {i},{c}"
            );
        }
    }
}

#[test]
fn admitted_scene_centers_reconstruct_a_known_mixed_edge_from_noisy_donors() {
    const W: usize = 64;
    const H: usize = 40;
    let mut pixels = Vec::with_capacity(W * H * 4);
    let mut alpha = Vec::with_capacity(W * H);
    let mut owned = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let a = if x < 28 {
                1.
            } else if x <= 30 {
                0.1
            } else {
                0.
            };
            let b = if x > 30 {
                200. + match (x + y) % 4 {
                    0 => 3.,
                    1 => -3.,
                    _ => 0.,
                }
            } else {
                200.
            };
            for f in [250., 100., 80.] {
                pixels.push(byte(a * f + (1. - a) * b));
            }
            pixels.push(255);
            alpha.push(a);
            owned.push(x < 28);
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let blocked = vec![false; W * H];
    let evidence = scene_edge::prepare(&source, &alpha, &owned, &blocked, &alpha, 4);
    let centers = evidence.color_prior();
    let holes: Vec<_> = alpha.iter().map(|&a| a > 0.).collect();
    let mut errors = Vec::new();
    for scene in [&evidence.background, &centers] {
        let mut prior = source.clone();
        for (i, color) in scene.iter().enumerate() {
            if let Some(b) = color {
                for c in 0..3 {
                    prior.pixels[i * 4 + c] = byte(b[c]);
                }
            }
        }
        let mut b = reconstruct_source_background(
            &source,
            &holes,
            &alpha,
            &blocked,
            &blocked,
            &vec![0; W * H],
            Some(&prior),
            Some(scene),
            true,
        )
        .unwrap();
        layer_pair::pair(&source, &alpha, &mut b).unwrap();
        let mut total = 0.;
        let mut count = 0;
        for y in 10..30 {
            for x in 28..=30 {
                let i = y * W + x;
                assert!(evidence.background[i].is_some());
                assert!(evidence.minimum_alpha[i] <= alpha[i]);
                for c in 0..3 {
                    total += (b.pixels[i * 4 + c] as f32 - 200.).abs();
                    count += 1;
                }
                let (p, a) = layer_pair::sample_residual(&source, &alpha, &b, x as f32, y as f32);
                for c in 0..3 {
                    assert_eq!(
                        byte(p[c] + (1. - a) * b.pixels[i * 4 + c] as f32),
                        source.pixels[i * 4 + c]
                    );
                }
            }
        }
        for i in 0..W * H {
            if alpha[i] == 0. {
                assert_eq!(
                    &b.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4]
                );
            }
        }
        errors.push(total / count as f32);
    }
    eprintln!(
        "actual prepare -> color prior -> reconstruct truth MAE: projected={}, center={}",
        errors[0], errors[1]
    );
    assert!(
        errors[1] < 1.5,
        "known hidden scene remains tinted: {errors:?}"
    );
    assert!(
        errors[0] > errors[1] + 0.75,
        "test must distinguish a caller using the alpha-bound projection: {errors:?}"
    );
}
