use image_core::{apply_graph_json, RgbaImage};
use serde_json::json;

#[test]
fn public_lip_brightness_has_no_frozen_halo_outside_the_inner_mouth() {
    let source = RgbaImage::new(
        128,
        96,
        (0..128 * 96).flat_map(|_| [170, 60, 80, 255]).collect(),
    )
    .unwrap();
    let graph = json!({"operations":[{"type":"feature.adjust","params":{
        "adjustments":{"lipBrightness":-100},"semanticMasks":{
            "lips":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
            "teeth":{"shape":"rect","feather":20,"region":{"x":0.5,"y":0,"width":0.25,"height":1}}
        }
    }}]});
    let output = apply_graph_json(&source, &graph.to_string()).unwrap();
    let at = |x: usize| -> &[u8] {
        let i = (48 * 128 + x) * 4;
        &output.pixels[i..i + 4]
    };
    assert!(at(62)[0]<167,"real lip just outside mouth polygon must not be completely frozen by the teeth feather: {:?}",at(62));
    assert_eq!(
        at(65),
        &[170, 60, 80, 255],
        "the actual inner mouth remains protected"
    );
    assert!(
        at(40)[0] < 135,
        "boundary repair must retain the full C adjustment in the lip core"
    );
    let worst_jump = (54..68)
        .map(|x| (i16::from(at(x + 1)[0]) - i16::from(at(x)[0])).abs())
        .max()
        .unwrap();
    assert!(
        worst_jump <= 24,
        "a soft mouth boundary must not become an abrupt light stripe, jump {worst_jump}"
    );
}

#[test]
fn public_oral_fringe_respects_explicit_protection_transparency_and_zero() {
    let mut pixels: Vec<u8> = (0..128 * 96)
        .flat_map(|i| [170, 60, 80, (i % 256) as u8])
        .collect();
    pixels[(48 * 128 + 62) * 4 + 3] = 0;
    let source = RgbaImage::new(128, 96, pixels).unwrap();
    for amount in [-100_i32, -30, -1, 0, 1, 30, 100] {
        let graph = json!({"operations":[{"type":"feature.adjust","params":{
            "adjustments":{"lipBrightness":amount},
            "region":{"x":0.1,"y":0.2,"width":0.8,"height":0.6},
            "mask":{"shape":"rect","region":{"x":0.2,"y":0.1,"width":0.7,"height":0.8}},
            "protectMask":{"shape":"rect","region":{"x":0.47,"y":0.35,"width":0.03,"height":0.1}},
            "semanticMasks":{
                "lips":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},
                "teeth":{"shape":"rect","feather":20,"region":{"x":0.5,"y":0,"width":0.25,"height":1}}
            }
        }}]});
        let output = apply_graph_json(&source, &graph.to_string()).unwrap();
        for (x, y) in [(62, 48), (62, 40), (65, 48)] {
            let i = (y * 128 + x) * 4;
            assert_eq!(
                &output.pixels[i..i + 4],
                &source.pixels[i..i + 4],
                "fringe protected domain changed at {x},{y}, amount={amount}"
            );
        }
        for (before, after) in source
            .pixels
            .chunks_exact(4)
            .zip(output.pixels.chunks_exact(4))
        {
            assert_eq!(before[3], after[3]);
        }
        for y in 0..96 {
            for x in 0..128 {
                let i = (y * 128 + x) * 4;
                let nx = (x as f32 + 0.5) / 128.0;
                let ny = (y as f32 + 0.5) / 96.0;
                let selected = (0.2..=0.9).contains(&nx) && (0.2..=0.8).contains(&ny);
                if !selected || source.pixels[i + 3] == 0 {
                    assert_eq!(
                        &output.pixels[i..i + 4],
                        &source.pixels[i..i + 4],
                        "selection or hidden RGB changed at {x},{y}"
                    );
                }
                if amount.abs() <= 1 {
                    for c in 0..3 {
                        assert!(
                            (i16::from(output.pixels[i + c]) - i16::from(source.pixels[i + c]))
                                .abs()
                                <= 1
                        );
                    }
                }
            }
        }
        if amount == 0 {
            assert_eq!(output.pixels, source.pixels);
        }
    }
}

#[test]
fn public_explicit_soft_teeth_masks_keep_their_conservative_protection_semantics() {
    let source = RgbaImage::new(
        128,
        96,
        (0..128 * 96).flat_map(|_| [170, 60, 80, 173]).collect(),
    )
    .unwrap();
    let cases = [
        (
            json!({"shape":"raster","width":1,"height":1,"data":[64]}),
            62,
        ),
        (
            json!({"shape":"rect","feather":20,"invert":true,"region":{"x":0.5,"y":0,"width":0.25,"height":1}}),
            65,
        ),
        (
            json!({"shape":"ellipse","feather":20,"region":{"x":0.5,"y":0.25,"width":0.25,"height":0.5}}),
            62,
        ),
        (
            json!({"shape":"linear","from":{"x":0,"y":0},"to":{"x":1,"y":0}}),
            62,
        ),
        (
            json!({"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1},"strokes":[{"x":0.5,"y":0.5,"radius":0.5,"strength":75,"erase":true}]}),
            64,
        ),
    ];
    for (teeth, x) in cases {
        let graph = json!({"operations":[{"type":"feature.adjust","params":{
            "adjustments":{"lipBrightness":-100},"semanticMasks":{
                "lips":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}},"teeth":teeth
            }
        }}]});
        let out = apply_graph_json(&source, &graph.to_string()).unwrap();
        let i = (48 * 128 + x) * 4;
        assert_eq!(
            &out.pixels[i..i + 4],
            &source.pixels[i..i + 4],
            "soft tooth coverage must not be silently reinterpreted as a geometric exterior: {}",
            graph
        );
    }
}

#[test]
fn public_teeth_only_semantics_never_become_a_lip_edit_selection() {
    let source = RgbaImage::new(
        32,
        24,
        (0..32 * 24)
            .flat_map(|i| [230, 220, 192, (i % 256) as u8])
            .collect(),
    )
    .unwrap();
    for amount in [-100, -1, 0, 1, 100] {
        let graph = json!({"operations":[{"type":"feature.adjust","params":{
            "adjustments":{"lipBrightness":amount,"lipColor":60,"lipSaturation":30},"semanticMasks":{
                "teeth":{"shape":"raster","width":1,"height":1,"data":[64]}
            }
        }}]});
        let out = apply_graph_json(&source, &graph.to_string()).unwrap();
        assert_eq!(out.pixels, source.pixels);
    }
}

#[test]
fn public_negative_lip_brightness_uses_a_bounded_outer_feather_slope() {
    let source = RgbaImage::new(
        64,
        32,
        (0..64 * 32).flat_map(|_| [200, 80, 100, 255]).collect(),
    )
    .unwrap();
    let graph = json!({"operations":[{"type":"feature.adjust","params":{
        "adjustments":{"lipBrightness":-100},
        "semanticMasks":{"lips":{"shape":"rect","feather":100,
            "region":{"x":0.25,"y":0.25,"width":0.5,"height":0.5}}}
    }}]});
    let output = apply_graph_json(&source, &graph.to_string()).unwrap();
    let luma = |x: usize| {
        let i = (16 * 64 + x) * 4;
        0.2126 * f32::from(output.pixels[i])
            + 0.7152 * f32::from(output.pixels[i + 1])
            + 0.0722 * f32::from(output.pixels[i + 2])
    };
    let steps: Vec<f32> = (8..27).map(|x| luma(x + 1) - luma(x)).collect();
    assert!(
        steps.iter().all(|step| *step <= 0.01),
        "negative brightness must remain monotone across the outer lip edge: {steps:?}"
    );
    let worst = steps.iter().map(|step| step.abs()).fold(0.0, f32::max);
    assert!(
        worst <= 3.0,
        "inverse-smoothstep edge response must bound a one-pixel luma step, got {worst}"
    );
    assert_eq!(
        &output.pixels[(16 * 64 + 8) * 4..(16 * 64 + 8) * 4 + 4],
        &[200, 80, 100, 255],
        "outside the feather must remain untouched"
    );
    assert!(luma(22) < 80.0, "the full lip interior must still darken");
}

#[test]
fn public_lip_alias_disables_geometric_edge_inversion() {
    let source = RgbaImage::new(
        32,
        24,
        (0..32 * 24).flat_map(|_| [180, 70, 90, 255]).collect(),
    )
    .unwrap();
    let common = json!({"adjustments":{"lipBrightness":-100}});
    let alias_graph = json!({"operations":[{"type":"feature.adjust","params":{
        "adjustments":common["adjustments"].clone(),
        "semanticMasks":{
            "lips":{"shape":"polygon","feather":100,"points":[
                {"x":0.101,"y":0.101},{"x":0.102,"y":0.101},
                {"x":0.102,"y":0.102},{"x":0.101,"y":0.102}]},
            "lip":{"shape":"raster","width":1,"height":1,"data":[200]}
        }
    }}]});
    let raster_graph = json!({"operations":[{"type":"feature.adjust","params":{
        "adjustments":common["adjustments"].clone(),
        "semanticMasks":{"lips":{"shape":"raster","width":1,"height":1,"data":[200]}}
    }}]});
    let alias_output = apply_graph_json(&source, &alias_graph.to_string()).unwrap();
    let raster_output = apply_graph_json(&source, &raster_graph.to_string()).unwrap();
    assert_eq!(
        alias_output.pixels, raster_output.pixels,
        "a legacy lip alias must keep conservative coverage semantics instead of
         inverting a different geometric mask"
    );
}
