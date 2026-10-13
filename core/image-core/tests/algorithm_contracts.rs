//! Black-box effect contracts against the public graph, not private formulas.
//! Measured effects do not substitute for the real-photo visual benchmark.
use image_core::{apply_graph_json, RgbaImage};
use serde_json::{json, Value};

fn image(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> RgbaImage {
    let mut pixels = Vec::new();
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&f(x, y));
        }
    }
    RgbaImage::new(w, h, pixels).unwrap()
}
fn run(src: &RgbaImage, kind: &str, params: Value) -> RgbaImage {
    apply_graph_json(
        src,
        &json!({"operations":[{"type":kind,"params":params}]}).to_string(),
    )
    .unwrap()
}
fn pixel(img: &RgbaImage, x: u32, y: u32) -> &[u8] {
    let i = ((y * img.width + x) * 4) as usize;
    &img.pixels[i..i + 4]
}
fn rect(x: f32, y: f32, w: f32, h: f32) -> Value {
    json!({"shape":"polygon","points":[{"x":x,"y":y},{"x":x+w,"y":y},{"x":x+w,"y":y+h},{"x":x,"y":y+h}],"feather":0})
}
fn grid() -> RgbaImage {
    image(128, 192, |x, y| [(x * 2) as u8, y as u8, 80, 255])
}

#[test]
fn global_geometry_has_correct_coordinates_and_identity() {
    let src = grid();
    for kind in [
        "identity",
        "color.basic",
        "color.hsl",
        "skin.retouch",
        "face.adjust",
        "feature.adjust",
        "body.adjust",
        "clothing.adjust",
        "warp.grid",
        "warp.liquify",
        "repair.clone",
        "repair.heal",
    ] {
        assert_eq!(
            run(
                &src,
                kind,
                if [
                    "face.adjust",
                    "feature.adjust",
                    "body.adjust",
                    "clothing.adjust"
                ]
                .contains(&kind)
                {
                    json!({"adjustments":{}})
                } else {
                    json!({})
                }
            ),
            src,
            "zero {kind}"
        );
    }
    assert_eq!(run(&src, "rotate", json!({"degrees":0})), src);
    assert_eq!(run(&src, "resize", json!({"width":128,"height":192})), src);
    let cropped = run(&src, "crop", json!({"x":13,"y":21,"width":31,"height":45}));
    assert_eq!((cropped.width, cropped.height), (31, 45));
    assert_eq!(pixel(&cropped, 0, 0), pixel(&src, 13, 21));
    for (axis, x, y) in [("horizontal", 127, 0), ("vertical", 0, 191)] {
        let flipped = run(&src, "flip", json!({"axis":axis}));
        assert_eq!(pixel(&flipped, 0, 0), pixel(&src, x, y));
        assert_eq!(run(&flipped, "flip", json!({"axis":axis})), src);
    }
    let rotated = run(&src, "rotate", json!({"degrees":90}));
    assert_eq!((rotated.width, rotated.height), (192, 128));
    assert_eq!(pixel(&rotated, 0, 0), pixel(&src, 0, 191));
    assert_eq!(run(&rotated, "rotate", json!({"degrees":-90})), src);
    let perspective = run(
        &src,
        "perspective",
        json!({"corners":[{"x":0,"y":0},{"x":1,"y":0},{"x":1,"y":1},{"x":0,"y":1}]}),
    );
    assert_eq!(perspective, src);
}

#[test]
fn grid_moves_feature_to_destination_and_preserves_fixed_anchors() {
    let src = grid();
    let out = run(
        &src,
        "warp.grid",
        json!({"points":[
            {"inX":0,"inY":0,"outX":0,"outY":0},{"inX":1,"inY":0,"outX":1,"outY":0},
            {"inX":0,"inY":1,"outX":0,"outY":1},{"inX":1,"inY":1,"outX":1,"outY":1},
            {"inX":0.4,"inY":0.5,"outX":0.6,"outY":0.5}
        ]}),
    );
    for (x, y) in [(0, 0), (127, 0), (0, 191), (127, 191)] {
        assert_eq!(pixel(&out, x, y), pixel(&src, x, y));
    }
    assert!((pixel(&out, 76, 96)[0] as i32 - 102).abs() < 3);
}

#[test]
fn liquify_modes_have_correct_direction_and_compact_support() {
    let src = grid();
    let apply = |mode: &str, direction: i32, strength: i32| {
        run(
            &src,
            "warp.liquify",
            json!({"strokes":[{"x":0.5,"y":0.5,"toX":0.55,"toY":0.5,"radius":0.25,"strength":strength,"mode":mode,"direction":direction}]}),
        )
    };
    for mode in ["push", "pinch", "bloat", "twirl", "reconstruct"] {
        assert_eq!(apply(mode, 1, 0), src, "zero {mode}");
        let out = apply(mode, 1, 80);
        for y in 0..192 {
            for x in 0..128 {
                if ((x as f32 - 64.).powi(2) + (y as f32 - 96.).powi(2)).sqrt() >= 32. {
                    assert_eq!(pixel(&out, x, y), pixel(&src, x, y), "support {mode}");
                }
            }
        }
    }
    assert!(pixel(&apply("push", 1, 100), 64, 96)[0] < pixel(&src, 64, 96)[0]);
    assert!(pixel(&apply("pinch", 1, 100), 72, 96)[0] > pixel(&src, 72, 96)[0]);
    assert!(pixel(&apply("bloat", 1, 100), 72, 96)[0] < pixel(&src, 72, 96)[0]);
    assert!(pixel(&apply("twirl", 1, 100), 72, 96)[1] < 96);
    assert!(pixel(&apply("twirl", -1, 100), 72, 96)[1] > 96);
}

#[test]
fn liquify_multi_stroke_preserves_frozen_pixels_before_later_sampling() {
    let src = grid();
    let stroke_a =
        json!({"x":0.47,"y":0.5,"toX":0.57,"toY":0.5,"radius":0.25,"strength":100,"mode":"push"});
    let stroke_b =
        json!({"x":0.58,"y":0.5,"toX":0.48,"toY":0.5,"radius":0.25,"strength":100,"mode":"push"});
    let protection = rect(0.48, 0.3, 0.08, 0.4);
    let params = |strokes: Value| json!({"strokes":strokes,"protectMask":protection,"region":{"x":0.2,"y":0.2,"width":0.6,"height":0.6}});
    let together = run(&src, "warp.liquify", params(json!([stroke_a, stroke_b])));
    let first = run(&src, "warp.liquify", params(json!([stroke_a])));
    let separate = run(&first, "warp.liquify", params(json!([stroke_b])));
    assert_eq!(
        together, separate,
        "frozen source may not be deformed between strokes and sampled by a later stroke"
    );
    for y in 58..134 {
        for x in 62..71 {
            assert_eq!(pixel(&together, x, y), pixel(&src, x, y));
        }
    }
}

fn feature_params(key: &str, amount: i32, mask_name: &str) -> Value {
    let mut masks = json!({});
    for name in [
        "leftEye",
        "rightEye",
        "leftBrow",
        "rightBrow",
        "lips",
        "nose",
        "noseBridge",
        "noseWings",
        "noseTip",
        "noseRoot",
        "noseBase",
        "leftNostril",
        "rightNostril",
    ] {
        masks[name] = if name == mask_name {
            rect(0.38, 0.4, 0.24, 0.18)
        } else {
            rect(0.05, 0.04, 0.06, 0.05)
        };
    }
    // Keep a horizontal pixel-space facial basis, with remote right feature.
    if mask_name != "leftEye" {
        masks["leftEye"] = rect(0.05, 0.04, 0.06, 0.05);
    }
    if mask_name != "rightEye" {
        masks["rightEye"] = rect(0.88, 0.04, 0.06, 0.05);
    }
    // Symmetric eyes must have the same y to avoid tilting the isolated test.
    if mask_name == "leftEye" {
        masks["rightEye"] = rect(0.85, 0.4, 0.10, 0.18);
    }
    if mask_name == "leftBrow" {
        masks["leftEye"] = rect(0.05, 0.04, 0.06, 0.05);
    }
    json!({"region":{"x":0,"y":0,"width":1,"height":1},"semanticMasks":masks,"adjustments":{key:amount}})
}

#[test]
fn all_published_feature_geometries_stay_local_and_reversible_at_zero() {
    let src = grid();
    let cases = [
        ("eyeSize", "leftEye"),
        ("eyeWidth", "leftEye"),
        ("eyeHeight", "leftEye"),
        ("eyeSpacing", "leftEye"),
        ("eyeTilt", "leftEye"),
        ("eyeCorner", "leftEye"),
        ("eyelid", "leftEye"),
        ("eyeInnerCorner", "leftEye"),
        ("eyeOuterCorner", "leftEye"),
        ("browLength", "leftBrow"),
        ("browWidth", "leftBrow"),
        ("browThickness", "leftBrow"),
        ("browPeak", "leftBrow"),
        ("browTail", "leftBrow"),
        ("browSpacing", "leftBrow"),
        ("noseBridge", "noseBridge"),
        ("noseWings", "noseWings"),
        ("noseLength", "nose"),
        ("noseTip", "noseTip"),
        ("nostril", "leftNostril"),
        ("noseRoot", "noseRoot"),
        ("noseBase", "noseBase"),
        ("mouthWidth", "lips"),
        ("mouthHeight", "lips"),
        ("upperLip", "lips"),
        ("lowerLip", "lips"),
        ("philtrum", "lips"),
        ("lipPeak", "lips"),
        ("lipCorner", "lips"),
    ];
    for (key, name) in cases {
        assert_eq!(
            run(&src, "feature.adjust", feature_params(key, 0, name)),
            src,
            "zero {key}"
        );
        let pos = run(&src, "feature.adjust", feature_params(key, 65, name));
        let neg = run(&src, "feature.adjust", feature_params(key, -65, name));
        // The real feature area must receive displacement of its own texture,
        // and every remote pixel beyond compact feature support stays fixed.
        let mut forward = 0;
        let mut backward = 0;
        let mut direction_correlation = 0i64;
        for y in 50..155 {
            for x in 30..96 {
                forward += usize::from(pixel(&pos, x, y) != pixel(&src, x, y));
                backward += usize::from(pixel(&neg, x, y) != pixel(&src, x, y));
                for c in 0..2 {
                    let a = pixel(&pos, x, y)[c] as i64 - pixel(&src, x, y)[c] as i64;
                    let b = pixel(&neg, x, y)[c] as i64 - pixel(&src, x, y)[c] as i64;
                    direction_correlation += a * b;
                }
            }
        }
        assert!(
            forward > 5 && backward > 5,
            "{key} must operate on its detected feature: {forward}/{backward}"
        );
        assert!(
            direction_correlation < 0,
            "positive/negative {key} must displace local texture in opposite directions"
        );
        for (x, y) in [(0, 191), (127, 191), (0, 96), (127, 96)] {
            assert_eq!(
                pixel(&pos, x, y),
                pixel(&src, x, y),
                "remote background {key}"
            );
        }
        let mut p = feature_params(key, 65, name);
        p["protectMask"] = rect(0.2, 0.2, 0.6, 0.6);
        let protected = run(&src, "feature.adjust", p);
        for (x, y) in [(64, 96), (56, 90), (72, 105)] {
            assert_eq!(pixel(&protected, x, y), pixel(&src, x, y), "freeze {key}");
        }
    }
}

#[test]
fn eye_and_mouth_size_controls_expand_real_silhouettes() {
    let src = image(200, 200, |x, y| {
        if (88..112).contains(&x) && (90..110).contains(&y) {
            [0, 0, 0, 255]
        } else {
            [180, 150, 130, 255]
        }
    });
    let area = |im: &RgbaImage| im.pixels.chunks_exact(4).filter(|p| p[0] < 50).count();
    for key in [
        "eyeSize",
        "eyeWidth",
        "eyeHeight",
        "mouthWidth",
        "mouthHeight",
        "browLength",
        "browThickness",
        "noseWings",
    ] {
        let name = if key.starts_with("eye") {
            "leftEye"
        } else if key.starts_with("mouth") {
            "lips"
        } else if key.starts_with("brow") {
            "leftBrow"
        } else {
            "noseWings"
        };
        let mut p = feature_params(key, 80, name);
        p["semanticMasks"][name] = rect(0.43, 0.43, 0.14, 0.14);
        if key.starts_with("eye") {
            p["semanticMasks"]["rightEye"] = rect(0.85, 0.43, 0.1, 0.14);
        }
        let larger = run(&src, "feature.adjust", p.clone());
        p["adjustments"][key] = json!(-80);
        let smaller = run(&src, "feature.adjust", p);
        assert!(area(&larger) > area(&src), "positive expands {key}");
        assert!(area(&smaller) < area(&src), "negative contracts {key}");
    }
}

#[test]
fn nose_alignment_uses_actual_asymmetry_and_preserves_bridge() {
    let src = image(200, 200, |x, y| {
        let center = 100.0 + (y as f32 - 90.0) * 0.38;
        if (90..122).contains(&y) && (x as f32 - center).abs() < 2.5 {
            [5, 5, 5, 255]
        } else {
            [180, 140, 110, 255]
        }
    });
    let masks = json!({"leftEye":rect(0.2,0.3,0.18,0.08),"rightEye":rect(0.62,0.3,0.18,0.08),"nose":rect(0.42,0.4,0.2,0.25),"noseRoot":rect(0.48,0.43,0.04,0.04),"noseTip":rect(0.53,0.57,0.04,0.04)});
    let out = run(
        &src,
        "feature.adjust",
        json!({"semanticMasks":masks,"adjustments":{"noseStraight":100}}),
    );
    let center = |im: &RgbaImage, y: u32| {
        let xs: Vec<_> = (80..125).filter(|&x| pixel(im, x, y)[0] < 70).collect();
        assert!(!xs.is_empty());
        xs.iter().sum::<u32>() as f32 / xs.len() as f32
    };
    assert!((center(&out, 118) - 100.0).abs() < (center(&src, 118) - 100.0).abs() * 0.4);
    assert_eq!(pixel(&out, 100, 90), pixel(&src, 100, 90));
    assert_eq!(pixel(&out, 55, 68), pixel(&src, 55, 68));
    let straight = image(200, 200, |x, y| {
        if (90..122).contains(&y) && (98..103).contains(&x) {
            [5, 5, 5, 255]
        } else {
            [180, 140, 110, 255]
        }
    });
    let mut masks = masks;
    masks["noseTip"] = rect(0.48, 0.57, 0.04, 0.04);
    assert_eq!(
        run(
            &straight,
            "feature.adjust",
            json!({"semanticMasks":masks,"adjustments":{"noseStraight":100}})
        ),
        straight
    );
}

#[test]
fn segmented_body_controls_move_silhouette_and_preserve_background() {
    let w = 120;
    let h = 160;
    let src = image(w, h, |x, y| {
        if (40..80).contains(&x) && (24..136).contains(&y) {
            [195, 110, 80, 255]
        } else {
            [20, (y % 60) as u8, 100, 255]
        }
    });
    let alpha: Vec<u8> = (0..w * h)
        .map(|i| {
            if (40..80).contains(&(i % w)) && (24..136).contains(&(i / w)) {
                255
            } else {
                0
            }
        })
        .collect();
    let count = |im: &RgbaImage| im.pixels.chunks_exact(4).filter(|p| p[0] > 150).count();
    for (part, key) in [
        ("waist", "waist"),
        ("waist", "belly"),
        ("hips", "hips"),
        ("torso", "shoulder"),
        ("left-upper-arm", "arms"),
        ("left-forearm", "forearms"),
        ("left-thigh", "legs"),
        ("left-thigh", "thighs"),
        ("left-calf", "calves"),
    ] {
        let points = if part == "waist" {
            json!([{"x":0.5,"y":0.15},{"x":0.5,"y":0.5},{"x":0.5,"y":0.85}])
        } else if part == "hips" {
            json!([{"x":0.35,"y":0.5},{"x":0.65,"y":0.5}])
        } else if part == "torso" {
            json!([{"x":0.35,"y":0.15},{"x":0.65,"y":0.15},{"x":0.35,"y":0.85},{"x":0.65,"y":0.85}])
        } else {
            json!([{"x":0.5,"y":0.15},{"x":0.5,"y":0.85}])
        };
        let mut p = json!({"bodyPart":part,"region":{"x":0.1,"y":0.05,"width":0.8,"height":0.9},"landmarks":points,"subjectMask":{"shape":"raster","width":w,"height":h,"data":alpha},"adjustments":{key:80}});
        let slim = run(&src, "body.adjust", p.clone());
        p["adjustments"][key] = json!(-80);
        let wide = run(&src, "body.adjust", p.clone());
        assert!(count(&slim) < count(&src), "slim contour {part}/{key}");
        assert!(count(&wide) > count(&src), "wide contour {part}/{key}");
        for y in 0..h {
            for x in 0..w {
                if alpha[(y * w + x) as usize] == 0 {
                    assert_eq!(
                        pixel(&slim, x, y),
                        pixel(&src, x, y),
                        "fixed backdrop {part}/{key}"
                    );
                }
            }
        }
        p["adjustments"][key] = json!(0);
        assert_eq!(run(&src, "body.adjust", p), src, "zero {key}");
    }
}

#[test]
fn repair_clones_requested_source_and_heal_removes_uniform_spot() {
    let src = image(100, 100, |x, y| {
        if (45..55).contains(&x) && (45..55).contains(&y) {
            [0, 0, 0, 77]
        } else {
            [170, 130, 100, 77]
        }
    });
    for kind in ["repair.clone", "repair.heal"] {
        let out = run(
            &src,
            kind,
            json!({"strokes":[{"x":0.5,"y":0.5,"sourceX":0.2,"sourceY":0.2,"radius":0.13,"strength":100}]}),
        );
        assert_eq!(pixel(&out, 50, 50), &[170, 130, 100, 77]);
        assert_eq!(pixel(&out, 0, 0), pixel(&src, 0, 0));
    }
    let out = run(
        &src,
        "repair.inpaint",
        json!({"mask":rect(0.44,0.44,0.12,0.12),"radius":0.04,"iterations":16}),
    );
    assert_eq!(pixel(&out, 50, 50), &[170, 130, 100, 77]);
    assert_eq!(pixel(&out, 0, 0), pixel(&src, 0, 0));
    let paint = run(
        &src,
        "paint.brush",
        json!({"color":[220,40,70],"strokes":[{"x":0.5,"y":0.5,"radius":0.1,"strength":100}]}),
    );
    assert_eq!(pixel(&paint, 50, 50), &[220, 40, 70, 255]);
    assert_eq!(pixel(&paint, 0, 0), pixel(&src, 0, 0));
}

#[test]
fn unsupported_or_unlocated_named_edits_fail_instead_of_modifying_background() {
    let src = grid();
    for (kind, key) in [
        ("face.adjust", "faceWidth"),
        ("feature.adjust", "eyeSize"),
        ("body.adjust", "waist"),
        ("clothing.adjust", "fit"),
        ("face.adjust", "doubleChin"),
        ("feature.adjust", "eyeBag"),
        ("feature.adjust", "tearTrough"),
        ("feature.adjust", "noseLipAngle"),
        ("body.adjust", "legLength"),
    ] {
        let graph = json!({"operations":[{"type":kind,"params":{"adjustments":{key:50}}}]});
        assert!(
            apply_graph_json(&src, &graph.to_string()).is_err(),
            "{kind}/{key}"
        );
    }
}
