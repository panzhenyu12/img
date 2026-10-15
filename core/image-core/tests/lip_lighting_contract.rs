use image_core::{apply_graph_json, RgbaImage};
use serde_json::json;

fn render(rgb: [u8; 3], amount: f32) -> [u8; 4] {
    let input = RgbaImage::new(
        32,
        24,
        (0..32 * 24)
            .flat_map(|_| [rgb[0], rgb[1], rgb[2], 173])
            .collect(),
    )
    .unwrap();
    let graph = json!({"operations":[{"type":"feature.adjust","params":{
        "adjustments":{"lipBrightness":amount},"semanticMasks":{
            "lips":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}
        }
    }}]});
    let output = apply_graph_json(&input, &graph.to_string()).unwrap();
    output.pixels[(12 * 32 + 16) * 4..(12 * 32 + 16) * 4 + 4]
        .try_into()
        .unwrap()
}
fn lin(x: u8) -> f64 {
    let v = x as f64 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
#[test]
fn public_lip_brightness_preserves_black_creases_instead_of_painting_them_pale() {
    assert_eq!(render([0, 0, 0], 100.0), [0, 0, 0, 173]);
    let dark = render([28, 6, 10], 100.0);
    assert!(
        dark[0] < 48 && dark[1] < 16,
        "dark lip crease must remain dark, got {dark:?}"
    );
}
#[test]
fn public_lip_brightness_preserves_linear_material_chromaticity() {
    for rgb in [[150, 40, 55], [93, 27, 32], [210, 75, 110]] {
        let original: Vec<_> = rgb.into_iter().map(lin).collect();
        for amount in [-100.0, 100.0] {
            let out = render(rgb, amount);
            let actual: Vec<_> = out[..3].iter().map(|&x| lin(x)).collect();
            let a: f64 = original.iter().sum();
            let b: f64 = actual.iter().sum();
            for c in 0..3 {
                assert!((original[c]/a-actual[c]/b).abs()<0.012,"brightness must retain lip pigment chromaticity: {rgb:?} -> {out:?} at {amount}");
            }
            assert_eq!(out[3], 173);
            assert!(
                if amount > 0.0 {
                    u16::from(out[0]) > u16::from(rgb[0]) + 12
                } else {
                    u16::from(out[0]) + 10 < u16::from(rgb[0])
                },
                "control must visibly affect diffuse lip material: {rgb:?} -> {out:?}"
            );
        }
    }
}

#[test]
fn public_lip_brightness_is_monotone_and_continuous_through_zero() {
    for rgb in [
        [28, 6, 10],
        [150, 40, 55],
        [93, 27, 32],
        [210, 75, 110],
        [220, 210, 210],
    ] {
        let mut previous = render(rgb, -100.0);
        for amount in [-30.0, -1.0, 0.0, 1.0, 30.0, 100.0] {
            let current = render(rgb, amount);
            for c in 0..3 {
                assert!(
                    current[c] >= previous[c],
                    "{rgb:?} at {amount}: {previous:?} -> {current:?}"
                );
            }
            if amount.abs() <= 1.0 {
                for c in 0..3 {
                    assert!(
                        (current[c] as i16 - rgb[c] as i16).abs() <= 1,
                        "a small slider change must not jump"
                    );
                }
            }
            previous = current;
        }
        assert_eq!(render(rgb, 0.0), [rgb[0], rgb[1], rgb[2], 173]);
    }
}

#[test]
fn public_lip_brightness_keeps_teeth_skin_protection_and_transparent_texels() {
    let mut pixels: Vec<u8> = (0..32 * 24).flat_map(|_| [150, 40, 55, 173]).collect();
    for y in 0..24 {
        for x in 0..32 {
            let i = (y * 32 + x) * 4;
            if x == 11 && y == 8 {
                pixels[i + 3] = 0;
            }
            if x == 21 && y == 10 {
                pixels[i + 3] = 73;
            }
            if (14..18).contains(&x) && (10..14).contains(&y) {
                pixels[i..i + 3].copy_from_slice(&[230, 224, 211]);
            }
        }
    }
    let input = RgbaImage::new(32, 24, pixels).unwrap();
    for amount in [-100.0, -30.0, 0.0, 30.0, 100.0] {
        let graph = json!({"operations":[{"type":"feature.adjust","params":{
            "region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6},
            "mask":{"shape":"rect","region":{"x":0.22,"y":0.22,"width":0.56,"height":0.56}},
            "protectMask":{"shape":"rect","region":{"x":0.25,"y":0.25,"width":0.07,"height":0.5}},
            "adjustments":{"lipBrightness":amount},"semanticMasks":{
                "lips":{"shape":"rect","region":{"x":0.25,"y":0.25,"width":0.5,"height":0.5}},
                "teeth":{"shape":"rect","region":{"x":0.42,"y":0.38,"width":0.16,"height":0.24}}
            }
        }}]});
        let output = apply_graph_json(&input, &graph.to_string()).unwrap();
        let mut changed = 0;
        for y in 0..24 {
            for x in 0..32 {
                let i = (y * 32 + x) * 4;
                let nx = (x as f32 + 0.5) / 32.0;
                let ny = (y as f32 + 0.5) / 24.0;
                let lip = (0.25..=0.75).contains(&nx) && (0.25..=0.75).contains(&ny);
                let tooth = (0.42..=0.58).contains(&nx) && (0.38..=0.62).contains(&ny);
                let protected = (0.25..=0.32).contains(&nx) && (0.25..=0.75).contains(&ny);
                assert_eq!(
                    input.pixels[i + 3],
                    output.pixels[i + 3],
                    "alpha changed at {x},{y}"
                );
                if !lip || tooth || protected || input.pixels[i + 3] == 0 || amount == 0.0 {
                    assert_eq!(
                        &input.pixels[i..i + 4],
                        &output.pixels[i..i + 4],
                        "protected domain changed at {x},{y}, amount {amount}"
                    );
                } else if input.pixels[i..i + 4] != output.pixels[i..i + 4] {
                    changed += 1;
                }
            }
        }
        if amount != 0.0 {
            assert!(
                changed > 50,
                "a protected edit must still modify eligible lip texels"
            );
        }
    }
}

#[test]
fn public_negative_lip_brightness_does_not_lock_saturated_pigment_as_white_gloss() {
    for rgb in [[255, 50, 70], [255, 150, 180]] {
        let dark = render(rgb, -100.0);
        assert!(
            u16::from(dark[0]) + 30 < u16::from(rgb[0]),
            "saturated lip pigment still needs to darken: {rgb:?} -> {dark:?}"
        );
        let original: Vec<_> = rgb.into_iter().map(lin).collect();
        let actual: Vec<_> = dark[..3].iter().map(|&v| lin(v)).collect();
        let a: f64 = original.iter().sum();
        let b: f64 = actual.iter().sum();
        for c in 0..3 {
            assert!((original[c] / a - actual[c] / b).abs() < 0.012);
        }
    }
    assert_eq!(
        render([255, 255, 255], -100.0),
        [255, 255, 255, 173],
        "neutral mirror highlights are not diffuse lipstick pigment"
    );
}
