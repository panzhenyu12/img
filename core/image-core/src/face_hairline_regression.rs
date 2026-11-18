//! Public-entry contract for an observed skin/hair interface inside a head.
//! Head silhouette and skin silhouette are deliberately different. The hair
//! mask is present and mutually exclusive with skin: deleting protection is
//! not a valid way to satisfy the intended forehead-height edit.

use super::tests::public_adjust;
use super::*;

const W: usize = 160;
const H: usize = 240;

fn raster(data: Vec<u8>) -> Value {
    json!({"shape":"raster","width":W,"height":H,"data":data})
}

fn polygon(points: &[Point]) -> Value {
    json!({"shape":"polygon","points":points.iter().map(|&(x,y)|
        json!({"x":x / W as f32,"y":y / H as f32})).collect::<Vec<_>>()})
}

fn rectangle(x0: usize, y0: usize, x1: usize, y1: usize) -> Value {
    polygon(&[
        (x0 as f32, y0 as f32),
        (x1 as f32, y0 as f32),
        (x1 as f32, y1 as f32),
        (x0 as f32, y1 as f32),
    ])
}

fn pixel(image: &RgbaImage, x: usize, y: usize) -> &[u8] {
    &image.pixels[(y * W + x) * 4..(y * W + x) * 4 + 4]
}

/// Known geometry: head ellipse is centred at (80,101), with radii (62,89).
/// The central observed skin/hair boundary is horizontal at y=52, while the
/// eyebrows start at y=86 and eyes at y=98. There is enough observed hair and
/// skin on both sides for a roughly 4.6 px bidirectional forehead request.
/// In opaque head material, B=20+y independently encodes source vertical
/// texture coordinates, including across the skin/hair interface itself.
fn thick_hair_portrait() -> (RgbaImage, Value) {
    let head: Vec<Point> = (0..64)
        .map(|i| {
            let angle = i as f32 * std::f32::consts::TAU / 64.0;
            (80.0 + 62.0 * angle.cos(), 101.0 + 89.0 * angle.sin())
        })
        .collect();
    let face = vec![
        (44.0, 60.0),
        (54.0, 52.0),
        (106.0, 52.0),
        (116.0, 60.0),
        (125.0, 92.0),
        (123.0, 137.0),
        (107.0, 168.0),
        (80.0, 181.0),
        (53.0, 168.0),
        (37.0, 137.0),
        (35.0, 92.0),
    ];
    let mut head_data = Vec::with_capacity(W * H);
    let mut face_data = Vec::with_capacity(W * H);
    let mut hair_data = Vec::with_capacity(W * H);
    let mut skin_data = Vec::with_capacity(W * H);
    let mut neck_data = Vec::with_capacity(W * H);
    let mut person_data = Vec::with_capacity(W * H);
    let mut pixels = Vec::with_capacity(W * H * 4);
    for y in 0..H {
        for x in 0..W {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let head_alpha = alpha(p, &head);
            let face_alpha = alpha(p, &face);
            let neck_alpha = if (70..90).contains(&x) && y >= 168 {
                1.0
            } else {
                0.0
            };
            let skin_alpha = face_alpha.max(neck_alpha);
            let hair_alpha = (head_alpha - face_alpha).max(0.0);
            let person_alpha = head_alpha.max(neck_alpha);
            head_data.push(byte(255.0 * head_alpha));
            face_data.push(byte(255.0 * face_alpha));
            hair_data.push(byte(255.0 * hair_alpha));
            skin_data.push(byte(255.0 * skin_alpha));
            neck_data.push(byte(255.0 * neck_alpha));
            person_data.push(byte(255.0 * person_alpha));
            let mut rgb = [15.0, 40.0, 80.0];
            if neck_alpha > 0.0 {
                rgb = [210.0, 150.0, 90.0];
            }
            let texture = 20.0 + y as f32;
            for (c, value) in [30.0, 22.0, texture].into_iter().enumerate() {
                rgb[c] = rgb[c] * (1.0 - head_alpha) + value * head_alpha;
            }
            for (c, value) in [210.0, 150.0, texture].into_iter().enumerate() {
                rgb[c] = rgb[c] * (1.0 - face_alpha) + value * face_alpha;
            }
            pixels.extend([byte(rgb[0]), byte(rgb[1]), byte(rgb[2]), 255]);
        }
    }
    let mut params = json!({
        "region":{"x":0,"y":0,"width":1,"height":1},
        "semanticMasks":{
            "face":raster(head_data), "faceOutline":polygon(&head),
            "faceContourMask":raster(face_data), "faceContour":polygon(&face),
            "hair":raster(hair_data), "skin":raster(skin_data), "neck":raster(neck_data),
            "leftEye":rectangle(51,98,65,103), "rightEye":rectangle(95,98,109,103),
            "leftBrow":rectangle(49,86,67,90), "rightBrow":rectangle(93,86,111,90),
            "nose":rectangle(75,113,85,139), "lips":rectangle(68,151,92,158),
            "ears":rectangle(21,99,26,112), "accessories":rectangle(126,115,131,128)
        },
        "subjectMask":raster(person_data),
        "protectMask":rectangle(44,38,50,49),
        "adjustments":{"forehead":0}
    });
    // Distinct textured markings make exact preservation observable.
    for (name, x0, y0, x1, y1) in [
        ("leftEye", 51, 98, 65, 103),
        ("rightEye", 95, 98, 109, 103),
        ("leftBrow", 49, 86, 67, 90),
        ("rightBrow", 93, 86, 111, 90),
        ("ears", 21, 99, 26, 112),
        ("accessories", 126, 115, 131, 128),
    ] {
        // The numeric extents also define the polygon, avoiding rounded
        // normalized coordinates being treated as a different fixture.
        params["semanticMasks"][name] = rectangle(x0, y0, x1, y1);
        for y in y0..y1 {
            for x in x0..x1 {
                let i = (y * W + x) * 4;
                pixels[i..i + 4].copy_from_slice(&[7 + (x % 4) as u8, 12 + (y % 5) as u8, 24, 255]);
            }
        }
    }
    (RgbaImage::new(W as u32, H as u32, pixels).unwrap(), params)
}

fn first_skin(image: &RgbaImage, x: usize) -> usize {
    (34..80)
        .find(|&y| pixel(image, x, y)[0] > 130)
        .expect("observed central forehead must remain visible")
}

fn skin_span(image: &RgbaImage, y: usize) -> (usize, usize) {
    let xs: Vec<_> = (0..W).filter(|&x| pixel(image, x, y)[0] > 130).collect();
    (
        *xs.first().expect("observed skin row must be nonempty"),
        *xs.last().unwrap(),
    )
}

#[test]
fn forehead_moves_the_observed_hairline_both_ways_without_splitting_material_or_moving_context() {
    let (source, params) = thick_hair_portrait();
    assert_eq!(public_adjust(&source, &params).unwrap(), source);
    for amount in [-100, -60, 60, 100] {
        let mut params = params.clone();
        params["adjustments"] = json!({"forehead":amount});
        let output = public_adjust(&source, &params).unwrap();
        for x in [65, 72, 80, 88, 95] {
            let before = first_skin(&source, x);
            let after = first_skin(&output, x);
            assert!(
                if amount > 0 {
                    after + 2 <= before
                } else {
                    after >= before + 2
                },
                "forehead changed pixels without moving its observed hairline: x={x}, amount={amount}, {before}->{after}"
            );
        }
        for x in 65..=95 {
            let edge = first_skin(&output, x);
            if x > 65 {
                assert!(
                    edge.abs_diff(first_skin(&output, x - 1)) <= 1,
                    "adjacent hairline columns disconnected at x={x}, amount={amount}"
                );
            }
            let mut previous_texture = pixel(&output, x, 34)[2];
            for y in 35..80 {
                let actual = pixel(&output, x, y);
                assert_eq!(actual[3], 255, "transparent crack at {x},{y}");
                assert!(
                    if y < edge {
                        actual[0] <= 130
                    } else {
                        actual[0] > 130
                    },
                    "skin/hair fragments or a second material edge at {x},{y}, amount={amount}"
                );
                // B=20+y has the same source slope in both materials. A
                // nonnegative, at-most-two-code step allows quantization and
                // mild compression, but catches reversal or torn texture.
                let step = actual[2] as i16 - previous_texture as i16;
                assert!(
                    (0..=2).contains(&step),
                    "skin/hair texture reversed or jumped at {x},{y}: step={step}, amount={amount}"
                );
                previous_texture = actual[2];
            }
        }
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let head = params["semanticMasks"]["face"]["data"][i].as_u64().unwrap();
                let protected = (44..50).contains(&x) && (38..49).contains(&y);
                let eye_brow = (((49..67).contains(&x) || (93..111).contains(&x))
                    && (86..90).contains(&y))
                    || (((51..65).contains(&x) || (95..109).contains(&x))
                        && (98..103).contains(&y));
                let accessory = (126..131).contains(&x) && (115..128).contains(&y);
                let ear = (21..26).contains(&x) && (99..112).contains(&y);
                if head < 255 || protected || eye_brow || accessory || ear {
                    assert_eq!(
                        pixel(&output, x, y), pixel(&source, x, y),
                        "head exterior, feature or protected material moved at {x},{y}, amount={amount}"
                    );
                }
            }
        }
    }
}

#[test]
fn uncertain_hair_class_inside_an_observed_opaque_head_does_not_freeze_the_hairline() {
    let (source, mut uncertain) = thick_hair_portrait();
    let mut reference = uncertain.clone();
    // Same source material and physical head alpha. Only hair class confidence
    // becomes uncertain over four rows, as independent parsers often disagree
    // at a dark hairline. This must not create a new frozen physical boundary.
    for y in 48..52 {
        for x in 54..106 {
            let i = y * W + x;
            let old = uncertain["semanticMasks"]["hair"]["data"][i]
                .as_u64()
                .unwrap();
            uncertain["semanticMasks"]["hair"]["data"][i] = json!(old * (52 - y) as u64 / 4);
        }
    }
    for amount in [-100, -60, 60, 100] {
        reference["adjustments"] = json!({"forehead":amount});
        uncertain["adjustments"] = reference["adjustments"].clone();
        let expected = public_adjust(&source, &reference).unwrap();
        let actual = public_adjust(&source, &uncertain).unwrap();
        for x in [65, 72, 80, 88, 95] {
            let before = first_skin(&source, x);
            let after = first_skin(&actual, x);
            assert!(
                if amount > 0 {
                    after + 2 <= before
                } else {
                    after >= before + 2
                },
                "class uncertainty cancelled forehead motion at {x}, amount={amount}"
            );
            assert!(
                after.abs_diff(first_skin(&expected, x)) <= 1,
                "class uncertainty changed physical interface target at {x}, amount={amount}"
            );
        }
    }
}

#[test]
fn nearby_disconnected_skin_and_hair_do_not_join_the_selected_hairline_motion() {
    let (mut source, mut params) = thick_hair_portrait();
    // A second material island is inside the head's bounding box and less
    // than 25 px from the selected hairline, but has no material connection
    // to it. The surrounding gap is explicit background in both RGB and
    // semantic evidence. Its own skin/hair edge is not the selected contour.
    // A spatially compact kernel alone can cross this gap; ownership cannot.
    for y in 27..43 {
        for x in 66..88 {
            let i = y * W + x;
            let island = (70..84).contains(&x) && (31..39).contains(&y);
            let island_skin = island && y >= 35;
            let island_hair = island && !island_skin;
            for (name, present) in [
                ("face", island),
                ("faceContourMask", island_skin),
                ("skin", island_skin),
                ("hair", island_hair),
            ] {
                params["semanticMasks"][name]["data"][i] = json!(if present { 255 } else { 0 });
            }
            params["subjectMask"]["data"][i] = json!(if island { 255 } else { 0 });
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&if island {
                [
                    if island_skin { 210 } else { 30 },
                    30 + 9 * ((x - 70) % 4) as u8,
                    45 + 16 * ((y - 31) % 3) as u8,
                    255,
                ]
            } else {
                [15, 40, 80, 255]
            });
        }
    }
    for amount in [-100, 100] {
        params["adjustments"] = json!({"forehead":amount});
        let output = public_adjust(&source, &params).unwrap();
        for y in 27..43 {
            for x in 66..88 {
                assert_eq!(
                    pixel(&output, x, y),
                    pixel(&source, x, y),
                    "disconnected material or its background gap moved at {x},{y}, amount={amount}"
                );
            }
        }
        // The scene must still carry out the selected edit. Returning the
        // complete source is not a valid solution to the ownership contract.
        let before = first_skin(&source, 96);
        let after = first_skin(&output, 96);
        assert!(
            if amount > 0 { after + 2 <= before } else { after >= before + 2 },
            "isolated material cancelled the unrelated selected hairline: amount={amount}, {before}->{after}"
        );
    }
}

#[test]
fn moving_hairline_never_samples_the_colored_material_in_a_forbidden_patch() {
    // The source's entire green-channel range is <=150 except for the
    // forbidden patch, whose G=255 is a unique donor marker. Therefore any
    // value >150 outside that patch proves contamination from fixed material.
    // Place explicit protection above the interface, and an accessory below
    // it, so opposite slider directions exercise both sides of the boundary.
    for (kind, marker_y) in [("protectMask", 43..48), ("accessories", 57..62)] {
        let (mut source, mut params) = thick_hair_portrait();
        let mut mask = vec![0_u8; W * H];
        let original = if kind == "protectMask" {
            (44..50, 38..49)
        } else {
            (126..131, 115..128)
        };
        for y in original.1 {
            for x in original.0.clone() {
                mask[y * W + x] = 255;
            }
        }
        for y in marker_y.clone() {
            for x in 76..84 {
                let i = y * W + x;
                mask[i] = 255;
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[0, 255, 0, 255]);
            }
        }
        if kind == "protectMask" {
            params["protectMask"] = raster(mask);
        } else {
            params["semanticMasks"]["accessories"] = raster(mask);
        }
        for amount in [-100, 100] {
            params["adjustments"] = json!({"forehead":amount});
            let output = public_adjust(&source, &params).unwrap();
            for y in 0..H {
                for x in 0..W {
                    if (76..84).contains(&x) && marker_y.contains(&y) {
                        assert_eq!(
                            pixel(&output, x, y),
                            pixel(&source, x, y),
                            "forbidden marker moved at {x},{y}, kind={kind}, amount={amount}"
                        );
                    } else {
                        assert!(
                            pixel(&output, x, y)[1] <= 150,
                            "fixed material was sampled into moving texture at {x},{y}, kind={kind}, amount={amount}: {:?}",
                            pixel(&output, x, y)
                        );
                    }
                }
            }
            let before = first_skin(&source, 65);
            let after = first_skin(&output, 65);
            assert!(
                if amount > 0 { after + 2 <= before } else { after >= before + 2 },
                "local protection cancelled the unrelated selected hairline: kind={kind}, amount={amount}, {before}->{after}"
            );
        }
    }
}

#[test]
fn non_normal_blending_keeps_stationary_and_protected_material_exact() {
    let (source, mut params) = thick_hair_portrait();
    params["blendMode"] = json!("multiply");
    assert_eq!(public_adjust(&source, &params).unwrap(), source);
    for amount in [-60, 60] {
        params["adjustments"] = json!({"forehead":amount});
        let output = public_adjust(&source, &params).unwrap();
        let mut changed_interface_pixels = 0;
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let head = params["semanticMasks"]["face"]["data"][i].as_u64().unwrap();
                let protected = (44..50).contains(&x) && (38..49).contains(&y);
                let eye =
                    ((51..65).contains(&x) || (95..109).contains(&x)) && (98..103).contains(&y);
                let brow =
                    ((49..67).contains(&x) || (93..111).contains(&x)) && (86..90).contains(&y);
                let ear = (21..26).contains(&x) && (99..112).contains(&y);
                let accessory = (126..131).contains(&x) && (115..128).contains(&y);
                // Lower face/neck is outside forehead-height motion; it must
                // not be multiplied just because it was copied into a layer.
                if head < 255 || y >= 110 || protected || eye || brow || ear || accessory {
                    assert_eq!(
                        pixel(&output, x, y),
                        pixel(&source, x, y),
                        "multiply recolored stationary material at {x},{y}, amount={amount}"
                    );
                }
                if (65..96).contains(&x)
                    && (44..72).contains(&y)
                    && pixel(&output, x, y) != pixel(&source, x, y)
                {
                    changed_interface_pixels += 1;
                }
            }
        }
        // A non-normal blend changes color, so source-coordinate texture and
        // thresholded material-edge oracles from normal mode do not apply.
        assert!(
            changed_interface_pixels > 30,
            "multiply blend silently cancelled the forehead edit: amount={amount}"
        );
    }
}

fn mean_signed_hairline_motion(source: &RgbaImage, output: &RgbaImage, amount: i32) -> f32 {
    [65, 72, 80, 88, 95]
        .into_iter()
        .map(|x| {
            (first_skin(source, x) as f32 - first_skin(output, x) as f32) * amount.signum() as f32
        })
        .sum::<f32>()
        / 5.0
}

#[test]
fn duplicate_source_outline_mask_matches_the_same_unmasked_support() {
    let (source, mut unmasked) = thick_hair_portrait();
    // Legacy duplicate masks are source outlines, not output selections.
    // Compare on the same legacy-expanded support: this fixture has face
    // bounds (35,52)..(125,181), with 18% of the 90 px face width as padding.
    // An unrelated whole-image versus tight-region change is not this test.
    unmasked["region"] = json!({
        "x":18.8 / W as f64, "y":35.8 / H as f64,
        "width":122.4 / W as f64, "height":161.4 / H as f64
    });
    for amount in [-60, 60] {
        unmasked["adjustments"] = json!({"forehead":amount});
        let mut duplicate = unmasked.clone();
        duplicate["mask"] = duplicate["semanticMasks"]["faceContour"].clone();
        let reference = public_adjust(&source, &unmasked).unwrap();
        let actual = public_adjust(&source, &duplicate).unwrap();
        assert!(
            mean_signed_hairline_motion(&source, &reference, amount) >= 2.0,
            "reference support must permit a real forehead edit"
        );
        assert_eq!(
            actual, reference,
            "redundant source contour became an output clipping mask: amount={amount}"
        );
    }
}

#[test]
fn soft_selection_and_soft_protection_preserve_intermediate_hairline_motion() {
    let (source, params) = thick_hair_portrait();
    for amount in [-60, 60] {
        let mut full = params.clone();
        full["adjustments"] = json!({"forehead":amount});
        let reference = public_adjust(&source, &full).unwrap();
        let full_motion = mean_signed_hairline_motion(&source, &reference, amount);
        assert!(
            full_motion >= 2.0,
            "full-strength reference must be observable"
        );
        for key in ["mask", "protectMask"] {
            let mut partial = full.clone();
            partial[key] = raster(vec![128; W * H]);
            let actual = public_adjust(&source, &partial).unwrap();
            let motion = mean_signed_hairline_motion(&source, &actual, amount);
            assert!(
                motion >= 0.8 && motion <= full_motion - 0.8,
                "soft {key} must yield an intermediate material-edge change, not zero/full motion or a second color attenuation: amount={amount}, partial={motion}, full={full_motion}"
            );
        }
    }
}

#[test]
fn upper_width_and_temples_use_the_public_source_forward_path() {
    let (source, params) = thick_hair_portrait();
    for (key, row, positive_widens) in [
        ("foreheadWidth", 68_usize, true),
        ("temples", 94_usize, false),
    ] {
        let before = skin_span(&source, row);
        for amount in [-100, 100] {
            let mut edited = params.clone();
            edited["adjustments"] = json!({key: amount});
            let output = public_adjust(&source, &edited).unwrap();
            let after = skin_span(&output, row);
            let widens = after.0 < before.0 && after.1 > before.1;
            assert_eq!(
                widens,
                (amount > 0) == positive_widens,
                "public {key}={amount} did not move the observed upper contour: {before:?} -> {after:?}"
            );
            for (xs, ys) in [
                (51..65, 98..103),
                (95..109, 98..103),
                (49..67, 86..90),
                (93..111, 86..90),
                (75..85, 113..139),
                (68..92, 151..158),
                (126..131, 115..128),
            ] {
                for y in ys {
                    for x in xs.clone() {
                        assert_eq!(
                            pixel(&output, x, y),
                            pixel(&source, x, y),
                            "public {key}={amount} moved a protected feature at {x},{y}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn upper_width_controls_remain_active_when_combined_with_face_width() {
    let (source, params) = thick_hair_portrait();
    for (key, row, positive_widens) in [
        ("foreheadWidth", 68_usize, true),
        ("temples", 94_usize, false),
    ] {
        let mut lower_only = params.clone();
        lower_only["adjustments"] = json!({"faceWidth": 60});
        let lower = public_adjust(&source, &lower_only).unwrap();
        let lower_span = skin_span(&lower, row);
        let mut combined = params.clone();
        combined["adjustments"] = json!({"faceWidth": 60, key: 60});
        let output = public_adjust(&source, &combined).unwrap();
        let span = skin_span(&output, row);
        assert_ne!(
            span, lower_span,
            "combined faceWidth + {key} dropped the upper control at row {row}"
        );
        let widens = span.0 < lower_span.0 && span.1 > lower_span.1;
        assert_eq!(
            widens, positive_widens,
            "combined faceWidth + {key} moved the upper contour in the wrong direction: {lower_span:?} -> {span:?}"
        );
        for (xs, ys) in [
            (51..65, 98..103),
            (95..109, 98..103),
            (49..67, 86..90),
            (93..111, 86..90),
            (75..85, 113..139),
            (68..92, 151..158),
            (126..131, 115..128),
        ] {
            for y in ys {
                for x in xs.clone() {
                    assert_eq!(
                        pixel(&output, x, y),
                        pixel(&source, x, y),
                        "combined faceWidth + {key} moved protected material at {x},{y}"
                    );
                }
            }
        }
    }
}

#[test]
fn another_face_slider_crossing_zero_does_not_cancel_the_forehead_or_move_features() {
    let (source, params) = thick_hair_portrait();
    for forehead in [-60, 60] {
        let mut base = params.clone();
        base["adjustments"] = json!({"forehead":forehead,"faceWidth":0});
        let reference = public_adjust(&source, &base).unwrap();
        assert!(mean_signed_hairline_motion(&source, &reference, forehead) >= 2.0);
        for width in [-1, 0, 1] {
            let mut combined = base.clone();
            combined["adjustments"]["faceWidth"] = json!(width);
            let actual = public_adjust(&source, &combined).unwrap();
            assert!(
                mean_signed_hairline_motion(&source, &actual, forehead) >= 2.0,
                "crossing another slider's zero cancelled forehead motion: forehead={forehead}, width={width}"
            );
            for x in [65, 72, 80, 88, 95] {
                assert!(
                    first_skin(&actual, x).abs_diff(first_skin(&reference, x)) <= 1,
                    "unrelated near-zero slider switched the hairline algorithm: x={x}, forehead={forehead}, width={width}"
                );
            }
            for (xs, ys) in [
                (51..65, 98..103),
                (95..109, 98..103),
                (49..67, 86..90),
                (93..111, 86..90),
                (75..85, 113..139),
                (68..92, 151..158),
                (44..50, 38..49),
            ] {
                for y in ys {
                    for x in xs.clone() {
                        assert_eq!(
                            pixel(&actual, x, y), pixel(&source, x, y),
                            "combined edit moved a protected feature at {x},{y}, forehead={forehead}, width={width}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn combined_face_and_hairline_apply_operation_opacity_exactly_once() {
    let (mut source, mut params) = thick_hair_portrait();
    // Expose a textured connected neck in this combined-edit fixture. The
    // hairline-only fixture has a closed curtain of hair behind the chin;
    // that would make the second edit unobservable and miss double opacity.
    // RGB and semantic material are changed together, retaining the hair
    // mask everywhere actual hair remains in the source.
    for y in 168..H {
        for x in 70..90 {
            let i = y * W + x;
            params["semanticMasks"]["hair"]["data"][i] = json!(0);
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                210,
                150,
                188 + ((y - 168) / 2) as u8,
                255,
            ]);
        }
    }
    params["adjustments"] = json!({"forehead":60,"faceWidth":60,"chin":25});
    params["blendMode"] = json!("normal");
    params["opacity"] = json!(100);
    let opaque = public_adjust(&source, &params).unwrap();
    let changed_upper =
        (40..80).any(|y| (65..96).any(|x| pixel(&opaque, x, y) != pixel(&source, x, y)));
    let changed_lower =
        (160..215).any(|y| (70..90).any(|x| pixel(&opaque, x, y) != pixel(&source, x, y)));
    assert!(
        changed_upper && changed_lower,
        "the opacity fixture needs both upper and lower edits"
    );
    params["opacity"] = json!(0);
    assert_eq!(public_adjust(&source, &params).unwrap(), source);
    params["opacity"] = json!(50);
    let half = public_adjust(&source, &params).unwrap();
    // Every source and rendered pixel is opaque here, so one normal 50%
    // composite is independently the integer average with halves rounded up.
    // This checks the complete composed result, including overlapping support.
    for i in 0..source.pixels.len() {
        let expected = ((source.pixels[i] as u16 + opaque.pixels[i] as u16 + 1) / 2) as u8;
        assert_eq!(
            half.pixels[i], expected,
            "operation opacity was applied more than once at byte {i}: source={}, opaque={}, half={}",
            source.pixels[i], opaque.pixels[i], half.pixels[i]
        );
    }
}

fn narrow_anatomical_frame_params(mut params: Value) -> Value {
    params["landmarks"] = json!((0..32)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 32.;
            json!({"x":(80.+20.*a.cos())/W as f32,"y":(115.+65.*a.sin())/H as f32})
        })
        .collect::<Vec<_>>());
    params
}

#[test]
fn observed_head_domain_recovers_side_interface_outside_landmark_width() {
    let (source, params) = thick_hair_portrait();
    let mut params = narrow_anatomical_frame_params(params);
    for amount in [-60, 60] {
        params["adjustments"] = json!({"forehead":amount});
        let after = public_adjust(&source, &params).unwrap();
        let original = first_skin(&source, 104);
        let new = first_skin(&after, 104);
        assert!(
            if amount > 0 {
                new + 2 <= original
            } else {
                new >= original + 2
            },
            "observed side interface failed: {amount}, source {original}, actual {new}"
        );
    }
}

#[test]
fn observed_head_domain_keeps_second_island_outside_old_frame_exact() {
    let (mut source, params) = thick_hair_portrait();
    let mut params = narrow_anatomical_frame_params(params);
    for y in 28..50 {
        for x in 106..130 {
            let i = y * W + x;
            let island = (111..123).contains(&x) && (34..44).contains(&y);
            let skin = island && y >= 39;
            for (name, present) in [
                ("face", island),
                ("faceContourMask", skin),
                ("skin", skin),
                ("hair", island && !skin),
            ] {
                params["semanticMasks"][name]["data"][i] = json!(if present { 255 } else { 0 });
            }
            params["subjectMask"]["data"][i] = json!(if island { 255 } else { 0 });
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&if island {
                [
                    if skin { 210 } else { 30 },
                    (x % 5) as u8 * 30,
                    (y % 4) as u8 * 40,
                    255,
                ]
            } else {
                [15, 40, 80, 255]
            });
        }
    }
    for amount in [-60, 60] {
        params["adjustments"] = json!({"forehead":amount});
        let out = public_adjust(&source, &params).unwrap();
        for y in 28..50 {
            for x in 106..130 {
                assert_eq!(
                    pixel(&out, x, y),
                    pixel(&source, x, y),
                    "disconnected island/background changed {x},{y}, {amount}"
                );
            }
        }
        assert!(
            mean_signed_hairline_motion(&source, &out, amount) >= 2.,
            "second island must not cancel chosen forehead"
        );
    }
}

#[test]
fn observed_head_domain_respects_explicit_region_and_selection() {
    let (source, params) = thick_hair_portrait();
    let params = narrow_anatomical_frame_params(params);
    for kind in ["region", "selection"] {
        for amount in [-60, 60] {
            let mut p = params.clone();
            p["adjustments"] = json!({"forehead":amount});
            if kind == "region" {
                p["region"] = json!({"x":0.,"y":0.,"width":100./W as f32,"height":1.});
            } else {
                p["mask"] = raster(
                    (0..W * H)
                        .map(|i| if i % W < 100 { 255 } else { 0 })
                        .collect(),
                );
            }
            let out = public_adjust(&source, &p).unwrap();
            for y in 0..H {
                for x in 100..W {
                    assert_eq!(
                        pixel(&out, x, y),
                        pixel(&source, x, y),
                        "outside {kind} changed {x},{y}, {amount}"
                    );
                }
            }
            let old = first_skin(&source, 80);
            let new = first_skin(&out, 80);
            assert!(
                if amount > 0 {
                    new + 2 <= old
                } else {
                    new >= old + 2
                },
                "valid center cancelled by {kind}"
            );
        }
    }
}

#[test]
fn sampling_guard_preserves_reachable_small_motion_next_to_fixed_material() {
    // Independent analytic fixture: the observed head begins at y=40, the
    // skin/hair interface is y=44.25, and its center is far from all side
    // controls. A +/-1 request is exactly +/-0.08145 px vertically. All
    // sampling footprints along that short trajectory remain in known head
    // material; an additional four-pixel collar is not a material constraint.
    let outline = vec![(40., 44.25), (120., 44.25), (120., 180.), (40., 180.)];
    let mut head = vec![0_u8; W * H];
    let mut skin = vec![0_u8; W * H];
    let mut hair = vec![0_u8; W * H];
    let mut protect = vec![0_u8; W * H];
    let mut pixels = vec![0_u8; W * H * 4];
    let inside_head = |x: usize, y: usize| (20..140).contains(&x) && (40..190).contains(&y);
    let protected = |x: usize, y: usize| (102..108).contains(&x) && (40..47).contains(&y);
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let observed = inside_head(x, y);
            let face = observed && (40..120).contains(&x) && y >= 44;
            head[i] = if observed { 255 } else { 0 };
            skin[i] = if face { 255 } else { 0 };
            hair[i] = if observed && !face { 255 } else { 0 };
            protect[i] = if protected(x, y) { 255 } else { 0 };
            let color = if !observed || protected(x, y) {
                [0, 255, 0, 255]
            } else if face {
                [210, 120, 20 + y as u8, 255]
            } else {
                [30, 22, 20 + y as u8, 255]
            };
            pixels[i * 4..i * 4 + 4].copy_from_slice(&color);
        }
    }
    let source = RgbaImage::new(W as u32, H as u32, pixels).unwrap();
    let mut params = json!({"region":{"x":0,"y":0,"width":1,"height":1},
        "semanticMasks":{"face":raster(head),"faceContourMask":raster(skin),"hair":raster(hair)},
        "protectMask":raster(protect),"adjustments":{"forehead":0}});
    let frame = Frame::new(&outline, params.get("semanticMasks"), W as f32, H as f32);
    for amount in [-1., 1.] {
        params["adjustments"] = json!({"forehead":amount});
        let prepared = hairline::prepare(&source, &params, &outline, frame)
            .expect("observed interface must be detected");
        let p = (80., 44.25);
        let q = prepared.transport_for_test(p, 1.);
        let expected = -amount * 0.08145_f32;
        assert!((q.1-p.1-expected).abs()<=0.02*expected.abs(),
            "reachable near-boundary request attenuated: amount={amount}, requested={expected}, realized={}",q.1-p.1);
        assert!(
            (q.0 - p.0).abs() < 1e-5,
            "vertical control introduced lateral motion"
        );
        // Known physical boundary and its complete one-cell sampling support
        // must remain stationary in both integration directions.
        for direction in [-1., 1.] {
            for fixed in [(80., 39.5), (80., 40.5), (104.5, 43.5)] {
                assert_eq!(
                    prepared.transport_for_test(fixed, direction),
                    fixed,
                    "fixed source/support moved: {fixed:?}"
                );
            }
        }
        // Audit actual inverse transport, using independent fixture geometry
        // rather than the solver's distance map or its own sampling counter.
        for y in 0..H {
            for x in 0..W {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let q = prepared.transport_for_test(p, -1.);
                if q == p {
                    continue;
                }
                let sx = (q.0 - 0.5).floor() as isize;
                let sy = (q.1 - 0.5).floor() as isize;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (xx, yy) = (sx + dx, sy + dy);
                    assert!(
                        xx >= 0 && yy >= 0 && (xx as usize) < W && (yy as usize) < H,
                        "sample outside source"
                    );
                    assert!(
                        inside_head(xx as usize, yy as usize)
                            && !protected(xx as usize, yy as usize),
                        "moving pixel sampled forbidden source {xx},{yy} from {p:?} -> {q:?}"
                    );
                }
            }
        }
        let output = prepared.render(&source);
        for y in 0..H {
            for x in 0..W {
                if !inside_head(x, y) || protected(x, y) {
                    assert_eq!(
                        pixel(&output, x, y),
                        pixel(&source, x, y),
                        "fixed RGB changed {x},{y}"
                    );
                } else {
                    assert!(
                        pixel(&output, x, y)[1] <= 120,
                        "forbidden green marker contaminated moving material {x},{y}"
                    );
                }
            }
        }
    }
}
