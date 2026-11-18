//! Continuity contract at a face/neck/backdrop junction.  This fixture is an
//! independently specified opaque photograph with observed semantic layers.
//! Optical expectations come from known fixture materials and coverage; flow
//! traces locate the source sample without defining its expected colour.

use super::tests::{joint_portrait, public_adjust};
use super::*;

const W: usize = 160;
const H: usize = 320;

struct TraceDirectory(std::path::PathBuf);

impl Drop for TraceDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn trace_directory(label: &str) -> TraceDirectory {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    TraceDirectory(std::env::temp_dir().join(format!("{label}-{}-{nonce}", std::process::id())))
}

fn hair_and_neck_portrait() -> (RgbaImage, Value) {
    let (mut source, mut params) = joint_portrait();
    let mut hair = vec![0_u8; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            if source.pixels[i * 4] < 100 {
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[85, 170, 178, 255]);
            }
            // Two observed hair strips belong to the same head context but
            // stay separated from the edit's lower-neck test window.
            if (40..200).contains(&y) && ((20..36).contains(&x) || (124..140).contains(&x)) {
                hair[i] = 255;
                params["semanticMasks"]["face"]["data"][i] = json!(255);
                params["subjectMask"]["data"][i] = json!(255);
                source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[25, 20, 18, 255]);
            }
        }
    }
    params["semanticMasks"]["hair"] = json!({"shape":"raster","width":W,"height":H,"data":hair});
    (source, params)
}

fn largest_lower_edge_jump(a: &RgbaImage, b: &RgbaImage) -> (u8, usize, usize) {
    let mut worst = (0, 0, 0);
    for y in 170..250 {
        for x in 55..105 {
            let i = (y * W + x) * 4;
            let difference = (0..3)
                .map(|c| a.pixels[i + c].abs_diff(b.pixels[i + c]))
                .max()
                .unwrap();
            if difference > worst.0 {
                worst = (difference, x, y);
            }
        }
    }
    worst
}

#[test]
fn infinitesimal_slider_change_cannot_switch_neck_material_to_backdrop() {
    let (source, mut params) = hair_and_neck_portrait();
    let mut render = |amount: f32| {
        params["adjustments"] = json!({"faceWidth":amount});
        public_adjust(&source, &params).unwrap()
    };
    // Find the worst interval, rather than pinning a renderer-specific slider
    // threshold. A continuous resampling/compositing pipeline converges to the
    // same decoded RGB as this public input interval tends to zero.
    let (mut low, mut high) = (20.0_f32, 100.0_f32);
    let (mut low_image, mut high_image) = (render(low), render(high));
    for _ in 0..12 {
        let middle = (low + high) * 0.5;
        let middle_image = render(middle);
        let left = largest_lower_edge_jump(&low_image, &middle_image);
        let right = largest_lower_edge_jump(&middle_image, &high_image);
        if left.0 >= right.0 {
            high = middle;
            high_image = middle_image;
        } else {
            low = middle;
            low_image = middle_image;
        }
    }
    let (jump, x, y) = largest_lower_edge_jump(&low_image, &high_image);
    assert!(
        jump <= 4,
        "faceWidth {low}→{high} switches lower-edge material by {jump} RGB levels at {x},{y}; tiny input changes must not toggle opaque scene/skin ownership"
    );
}

#[test]
fn connected_face_neck_material_follows_one_sampling_map_across_the_class_boundary() {
    let (mut source, mut params) = joint_portrait();
    // A broad observed upper neck joins the jaw on both sides. The original
    // narrow, central neck reaches only near the almost-stationary chin and
    // cannot expose a lateral transport/write mismatch at the class border.
    for y in 170..H {
        for x in 55..105 {
            let i = y * W + x;
            params["semanticMasks"]["skin"]["data"][i] = json!(255);
            params["subjectMask"]["data"][i] = json!(255);
            if params["semanticMasks"]["faceContourMask"]["data"][i]
                .as_u64()
                .unwrap()
                < 128
            {
                params["semanticMasks"]["neck"]["data"][i] = json!(255);
            }
        }
    }
    let skin: Vec<bool> = params["semanticMasks"]["skin"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64() == Some(255))
        .collect();
    let face: Vec<bool> = params["semanticMasks"]["faceContourMask"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() > 0)
        .collect();
    // One known opaque material continues from the face into the neck. The
    // class boundary has no color edge, illumination change or occluder.
    // These affine color channels make bilinear transport independently
    // observable; expected RGB is evaluated analytically, not by calling the
    // renderer's sampler, alpha reconstruction or changed-pixel classifier.
    let rgb = |x: f64| {
        [
            128.0 + 3.0 * (x - 80.0),
            150.0 + 2.0 * (x - 80.0),
            90.0 + (x - 80.0),
        ]
    };
    for y in 0..H {
        for x in 0..W {
            if skin[y * W + x] {
                for (c, value) in rgb(x as f64).into_iter().enumerate() {
                    source.pixels[(y * W + x) * 4 + c] = value.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    let trace = trace_directory("face-neck-class-continuity");
    params["experimentalFaceProfile"] = json!("source-domain-v1");
    params["adjustments"] = json!({"faceWidth":60});
    params["__faceFieldTrace"] = json!(trace.0);
    let output = adjust(&source, &params).unwrap();
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(trace.0.join("metadata.json")).unwrap()).unwrap();
    let cw = metadata["width"].as_u64().unwrap() as usize;
    let ch = metadata["height"].as_u64().unwrap() as usize;
    // Trace origin denotes a pixel center; inverse samples are local indices.
    let left = metadata["origin"][0].as_f64().unwrap() - 0.5;
    let top = metadata["origin"][1].as_f64().unwrap() - 0.5;
    let inverse: Vec<f32> = std::fs::read(trace.0.join("inverse-sample.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(inverse.len(), cw * ch * 2);
    let opaque_neighborhood = |x: f64, y: f64| {
        let (x0, y0) = (x.floor() as isize, y.floor() as isize);
        (-1..=2).all(|dy| {
            (-1..=2).all(|dx| {
                let (xx, yy) = (x0 + dx, y0 + dy);
                xx >= 40
                    && xx < 120
                    && yy >= 0
                    && yy < H as isize
                    && skin[yy as usize * W + xx as usize]
            })
        })
    };
    let mut errors = vec![None; W * H];
    let mut max_error = (0.0_f64, 0_usize, 0_usize, 0_usize, 0.0_f64, 0_u8);
    let mut moving_neck_samples = 0;
    for y in 174..230 {
        for x in 58..102 {
            let (lx, ly) = (x as isize - left as isize, y as isize - top as isize);
            if lx < 0 || ly < 0 || lx >= cw as isize || ly >= ch as isize {
                continue;
            }
            let i = (ly as usize * cw + lx as usize) * 2;
            let (sx, sy) = (left + inverse[i] as f64, top + inverse[i + 1] as f64);
            if !opaque_neighborhood(x as f64, y as f64) || !opaque_neighborhood(sx, sy) {
                continue;
            }
            let expected = rgb(sx);
            let residual: [f64; 3] =
                std::array::from_fn(|c| output.pixels[(y * W + x) * 4 + c] as f64 - expected[c]);
            errors[y * W + x] = Some(residual);
            if !face[y * W + x] && (sx - x as f64).abs() >= 0.25 {
                moving_neck_samples += 1;
            }
            for c in 0..3 {
                if residual[c].abs() > max_error.0 {
                    max_error = (
                        residual[c].abs(),
                        x,
                        y,
                        c,
                        expected[c],
                        output.pixels[(y * W + x) * 4 + c],
                    );
                }
            }
        }
    }
    let mut cross_class_pairs = 0;
    let mut max_step_error = (0.0_f64, 0_usize, 0_usize);
    let mut max_boundary_transport_error = 0.0_f64;
    for y in 174..229 {
        for x in 58..102 {
            let i = y * W + x;
            if face[i] == face[i + W] {
                continue;
            }
            if let (Some(a), Some(b)) = (errors[i], errors[i + W]) {
                cross_class_pairs += 1;
                for c in 0..3 {
                    max_boundary_transport_error =
                        max_boundary_transport_error.max(a[c].abs()).max(b[c].abs());
                    let error = (b[c] - a[c]).abs();
                    if error > max_step_error.0 {
                        max_step_error = (error, x, y);
                    }
                }
            }
        }
    }
    eprintln!("face/neck affine oracle: moving neck={moving_neck_samples}, boundary pairs={cross_class_pairs}, window RGB error={max_error:?}, excess boundary step={max_step_error:?}, boundary transport error={max_boundary_transport_error}");
    assert!(
        moving_neck_samples > 0,
        "fixture must contain actual neck motion, not an identity pass"
    );
    assert!(
        cross_class_pairs > 0,
        "fixture must exercise the internal face/neck class boundary"
    );
    // Rounding each adjacent byte permits at most one code value of error
    // in their difference. Semantic labels must not introduce another step.
    assert!(max_step_error.0 <= 1.01,
        "continuous skin gains a class-boundary color step of {:.3} at {},{}; both sides must follow the same inverse map", max_step_error.0, max_step_error.1, max_step_error.2);
    assert!(max_boundary_transport_error <= 0.51,
        "both sides of the internal material boundary must match affine transport within byte rounding: max RGB error={max_boundary_transport_error}");
}

#[test]
fn joint_partial_coverage_preserves_zero_and_reveals_the_known_background() {
    let (mut source, mut params) = joint_portrait();
    let foreground = [220.0_f64, 155.0, 100.0];
    let background = [20.0_f64, 40.0, 60.0];
    let mut physical = vec![0.0_f64; W * H];
    // The photograph has known, independently supersampled coverage. This
    // oracle does not use the renderer's matte refinement, polygon alpha,
    // foreground/background estimates or incremental-reveal formula.
    for y in 0..H {
        for x in 0..W {
            let mut covered = 0;
            for sy in 0..8 {
                for sx in 0..8 {
                    let px = x as f64 + (sx as f64 + 0.5) / 8.0;
                    let py = y as f64 + (sy as f64 + 0.5) / 8.0;
                    let face = ((px - 80.0) / 38.4).powi(2) + ((py - 120.0) / 88.8).powi(2) <= 1.0;
                    let neck = (70.0..90.0).contains(&px) && py >= 180.0;
                    covered += usize::from(face || neck);
                }
            }
            let i = y * W + x;
            physical[i] = covered as f64 / 64.0;
            for c in 0..3 {
                source.pixels[i * 4 + c] = (foreground[c] * physical[i]
                    + background[c] * (1.0 - physical[i]))
                    .round() as u8;
            }
        }
    }
    let mut protected = vec![0_u8; W * H];
    for y in 116..122 {
        for x in 47..53 {
            let i = y * W + x;
            assert_eq!(physical[i], 1.0);
            protected[i] = 255;
            // Known texture makes a protection test non-vacuous even though
            // the surrounding foreground is constant in this colour oracle.
            source.pixels[i * 4..i * 4 + 4].copy_from_slice(&[
                190 + (x % 3) as u8 * 15,
                135 + (y % 3) as u8 * 10,
                90,
                255,
            ]);
        }
    }
    params["experimentalFaceProfile"] = json!("shared-cheek-curve-v4");
    let observed_alpha: Vec<u8> = physical.iter().map(|a| (a * 255.0).round() as u8).collect();
    // Keep source coverage controlled so a failure distinguishes transport /
    // composition from automatic matting. This test does not certify the
    // automatic matte, unknown backgrounds, hair depth or natural face shape.
    params["__physicalAlpha"] =
        json!({"shape":"raster","width":W,"height":H,"data":observed_alpha});
    let protection = json!({"shape":"raster","width":W,"height":H,"data":protected});
    params["protectMask"] = protection.clone();
    for amount in [-0.01_f64, 0.01] {
        params["adjustments"] = json!({"faceWidth":amount});
        let actual = adjust(&source, &params).unwrap();
        let largest_change = actual
            .pixels
            .iter()
            .zip(&source.pixels)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(largest_change <= 1,
            "known partially covered source gains a finite colour jump at {amount}: {largest_change} RGB codes");
        for (i, &p) in protected.iter().enumerate() {
            if p > 0 {
                assert_eq!(
                    &actual.pixels[i * 4..i * 4 + 4],
                    &source.pixels[i * 4..i * 4 + 4],
                    "explicit protection changed at tiny amount {amount}, pixel {},{}",
                    i % W,
                    i / W
                );
            }
        }
    }
    params.as_object_mut().unwrap().remove("protectMask");
    params["adjustments"] = json!({"faceWidth":60});
    let trace = trace_directory("face-known-partial-background");
    params["__faceFieldTrace"] = json!(trace.0);
    params["__faceProbePoints"] = json!((92..160)
        .flat_map(|y| (0..W).filter_map({
            let physical = &physical;
            move |x| {
                (0.0 < physical[y * W + x] && physical[y * W + x] < 1.0)
                    .then_some([x as f64 + 0.5, y as f64 + 0.5])
            }
        }))
        .collect::<Vec<_>>());
    let output = adjust(&source, &params).unwrap();
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(trace.0.join("metadata.json")).unwrap()).unwrap();
    let cw = metadata["width"].as_u64().unwrap() as usize;
    let ch = metadata["height"].as_u64().unwrap() as usize;
    let left = metadata["origin"][0].as_f64().unwrap() - 0.5;
    let top = metadata["origin"][1].as_f64().unwrap() - 0.5;
    let inverse: Vec<f32> = std::fs::read(trace.0.join("inverse-sample.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(inverse.len(), cw * ch * 2);
    let mut moved_rows = [0_usize; 2];
    let mut exposed_partial = 0;
    let mut worst_background_error = (0.0_f64, 0_usize, 0_usize, 0_usize);
    for y in 92..160 {
        let before: Vec<_> = (0..W).filter(|&x| physical[y * W + x] >= 0.5).collect();
        let after: Vec<_> = (0..W)
            .filter(|&x| output.pixels[(y * W + x) * 4] >= 150)
            .collect();
        let before = (*before.first().unwrap(), *before.last().unwrap());
        let after = (*after.first().unwrap(), *after.last().unwrap());
        moved_rows[0] += usize::from(after.0 >= before.0 + 2);
        moved_rows[1] += usize::from(after.1 + 2 <= before.1);
        for x in 0..W {
            let i = y * W + x;
            if !(0.0 < physical[i] && physical[i] < 1.0) {
                continue;
            }
            let lx = x as isize - left as isize;
            let ly = y as isize - top as isize;
            if lx < 0 || ly < 0 || lx >= cw as isize || ly >= ch as isize {
                continue;
            }
            let at = (ly as usize * cw + lx as usize) * 2;
            let sx = left + inverse[at] as f64;
            let sy = top + inverse[at + 1] as f64;
            let sx0 = sx.floor() as isize;
            let sy0 = sy.floor() as isize;
            // An inverse coordinate is observed geometry, not an optical
            // oracle. All four source texels are independently known pure
            // backdrop here, so any convex subpixel sampler must return B.
            let pure_background_sample = (0..=1).all(|dy| {
                (0..=1).all(|dx| {
                    let xx = sx0 + dx;
                    let yy = sy0 + dy;
                    xx >= 0
                        && xx < W as isize
                        && yy >= 0
                        && yy < H as isize
                        && physical[yy as usize * W + xx as usize] == 0.0
                })
            });
            if !pure_background_sample {
                continue;
            }
            exposed_partial += 1;
            for (c, &expected) in background.iter().enumerate() {
                let error = (output.pixels[i * 4 + c] as f64 - expected).abs();
                if error > worst_background_error.0 {
                    worst_background_error = (error, x, y, c);
                }
            }
        }
    }
    eprintln!("known joint background: moved rows={moved_rows:?}/68, exposed partial pixels={exposed_partial}, worst background error={worst_background_error:?}");
    let probes: Value =
        serde_json::from_slice(&std::fs::read(trace.0.join("render-probes.json")).unwrap())
            .unwrap();
    for probe in probes.as_array().unwrap() {
        let x = probe["p"][0].as_f64().unwrap() as usize;
        let y = probe["p"][1].as_f64().unwrap() as usize;
        let supplied_alpha = observed_alpha[y * W + x] as f64 / 255.0;
        assert!((probe["oldAlpha"].as_f64().unwrap() - supplied_alpha).abs() <= 1e-6,
            "controlled source coverage was altered at {x},{y}: supplied={supplied_alpha}, renderer={}", probe["oldAlpha"]);
        if (x, y) == (44, 153) || (x, y) == (worst_background_error.1, worst_background_error.2) {
            eprintln!(
                "known physical alpha={} at {x},{y}; renderer={probe}",
                physical[y * W + x]
            );
        }
    }
    assert!(
        moved_rows.iter().all(|&n| n >= 32),
        "significant contraction must actually move both sides: {moved_rows:?}/68"
    );
    assert!(
        exposed_partial >= 20,
        "fixture must exercise old partial coverage becoming exposed: {exposed_partial}"
    );
    assert!(worst_background_error.0 <= 3.0,
        "newly exposed old partial coverage must return to the known backdrop, not retain a skin halo: {worst_background_error:?}");

    params["protectMask"] = protection;
    params.as_object_mut().unwrap().remove("__faceFieldTrace");
    params.as_object_mut().unwrap().remove("__faceProbePoints");
    let protected_output = adjust(&source, &params).unwrap();
    let would_change = protected
        .iter()
        .enumerate()
        .filter(|&(i, &p)| {
            p > 0 && output.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4]
        })
        .count();
    assert!(would_change > 0, "protection must prevent a real edit");
    for (i, &p) in protected.iter().enumerate() {
        if p > 0 || i % W < 20 || i % W >= 140 {
            assert_eq!(
                &protected_output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "protected or distant observed background changed at {},{}",
                i % W,
                i / W
            );
        }
    }
}

#[test]
fn explicit_source_plate_reveals_its_known_texture_without_destination_reconstruction() {
    let (mut source, mut params) = joint_portrait();
    let mut backdrop = source.clone();
    let mut physical = vec![0.0_f64; W * H];
    let mut protected = vec![0_u8; W * H];
    // Independent hidden scene texture. Its phase and high frequencies are
    // known before rendering; no harmonic fill or nearby donor average is an
    // acceptable substitute for this explicitly supplied source-space plate.
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let b = [
                20 + ((x * 17 + y * 29) % 67) as u8,
                50 + ((x * 31 + y * 13) % 59) as u8,
                110 + ((x * 11 + y * 23) % 71) as u8,
            ];
            backdrop.pixels[i * 4..i * 4 + 4].copy_from_slice(&[b[0], b[1], b[2], 255]);
            let mut covered = 0;
            for sy in 0..8 {
                for sx in 0..8 {
                    let px = x as f64 + (sx as f64 + 0.5) / 8.0;
                    let py = y as f64 + (sy as f64 + 0.5) / 8.0;
                    let face = ((px - 80.0) / 38.4).powi(2) + ((py - 120.0) / 88.8).powi(2) <= 1.0;
                    let neck = (70.0..90.0).contains(&px) && py >= 180.0;
                    covered += usize::from(face || neck);
                }
            }
            physical[i] = covered as f64 / 64.0;
            let f = if (47..53).contains(&x) && (116..122).contains(&y) {
                assert_eq!(physical[i], 1.0);
                protected[i] = 255;
                [190 + (x % 3) as u8 * 15, 135 + (y % 3) as u8 * 10, 90]
            } else {
                [220, 155, 100]
            };
            for c in 0..3 {
                source.pixels[i * 4 + c] =
                    (f[c] as f64 * physical[i] + b[c] as f64 * (1.0 - physical[i])).round() as u8;
            }
        }
    }
    let files = trace_directory("face-explicit-texture-plate");
    std::fs::create_dir_all(&files.0).unwrap();
    for (name, input) in [("source.png", &source), ("backdrop.png", &backdrop)] {
        image::save_buffer_with_format(
            files.0.join(name),
            &input.pixels,
            input.width,
            input.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();
    }
    std::fs::write(
        files.0.join("weights.f32"),
        (0..W * H)
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let descriptor = files.0.join("prior.json");
    std::fs::write(
        &descriptor,
        serde_json::to_vec(&json!({
            "source":"source.png", "image":"backdrop.png", "weights":"weights.f32"
        }))
        .unwrap(),
    )
    .unwrap();
    params["experimentalFaceProfile"] = json!("source-domain-continuous-v1");
    params["__sourcePlatePrior"] = json!(descriptor);
    params["__physicalAlpha"] = json!({
        "shape":"raster", "width":W, "height":H,
        "data":physical.iter().map(|a| (a * 255.0).round() as u8).collect::<Vec<_>>()
    });
    let protection = json!({"shape":"raster", "width":W, "height":H, "data":protected});
    params["protectMask"] = protection.clone();
    for amount in [0.0_f64, -0.01, 0.01] {
        params["adjustments"] = json!({"faceWidth":amount});
        let output = adjust(&source, &params).unwrap();
        let maximum = output
            .pixels
            .iter()
            .zip(&source.pixels)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(maximum <= u8::from(amount != 0.0),
            "explicit texture plate breaks source continuity at {amount}: maximum RGB change {maximum}");
        for (i, &p) in protected.iter().enumerate().filter(|(_, p)| **p > 0) {
            assert_eq!(
                &output.pixels[i * 4..i * 4 + 4],
                &source.pixels[i * 4..i * 4 + 4],
                "protected source changed at {amount}, index {i}, mask {p}"
            );
        }
    }
    params.as_object_mut().unwrap().remove("protectMask");
    params["adjustments"] = json!({"faceWidth":60});
    params["__faceFieldTrace"] = json!(files.0.join("trace"));
    let output = adjust(&source, &params).unwrap();
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(files.0.join("trace/metadata.json")).unwrap())
            .unwrap();
    let cw = metadata["width"].as_u64().unwrap() as usize;
    let ch = metadata["height"].as_u64().unwrap() as usize;
    let left = metadata["origin"][0].as_f64().unwrap() - 0.5;
    let top = metadata["origin"][1].as_f64().unwrap() - 0.5;
    let inverse: Vec<f32> = std::fs::read(files.0.join("trace/inverse-sample.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(inverse.len(), cw * ch * 2);
    let mut revealed_opaque = 0;
    let mut incorrect_opaque = 0;
    let mut worst = (0_u8, 0_usize, 0_usize, 0_usize);
    for y in 92..160 {
        for x in 30..130 {
            let i = y * W + x;
            if physical[i] != 1.0 {
                continue;
            }
            let lx = x as isize - left as isize;
            let ly = y as isize - top as isize;
            assert!((0..cw as isize).contains(&lx) && (0..ch as isize).contains(&ly));
            let at = (ly as usize * cw + lx as usize) * 2;
            let sx = (left + inverse[at] as f64).floor() as isize;
            let sy = (top + inverse[at + 1] as f64).floor() as isize;
            // Flow only establishes that the old opaque texel was truly
            // uncovered. The colour oracle is B(x,y), not a warped B or any
            // renderer trace of its reconstructed/paired background.
            let exposed = (0..=1).all(|dy| {
                (0..=1).all(|dx| {
                    let (xx, yy) = (sx + dx, sy + dy);
                    (0..W as isize).contains(&xx)
                        && (0..H as isize).contains(&yy)
                        && physical[yy as usize * W + xx as usize] == 0.0
                })
            });
            if !exposed {
                continue;
            }
            revealed_opaque += 1;
            incorrect_opaque +=
                usize::from(output.pixels[i * 4..i * 4 + 4] != backdrop.pixels[i * 4..i * 4 + 4]);
            for c in 0..3 {
                let error = output.pixels[i * 4 + c].abs_diff(backdrop.pixels[i * 4 + c]);
                if error > worst.0 {
                    worst = (error, x, y, c);
                }
            }
        }
    }
    eprintln!("explicit source texture: exposed opaque={revealed_opaque}, incorrect={incorrect_opaque}, worst={worst:?}");
    assert!(
        revealed_opaque >= 20,
        "the test must reveal a substantial formerly opaque strip, got {revealed_opaque}"
    );
    // Run the protection counterfactual before the texture assertion so a
    // failing colour regression still reports that this contract was checked.
    params["protectMask"] = protection;
    params.as_object_mut().unwrap().remove("__faceFieldTrace");
    let protected_output = adjust(&source, &params).unwrap();
    let would_change = protected
        .iter()
        .enumerate()
        .filter(|&(i, &p)| {
            p > 0 && output.pixels[i * 4..i * 4 + 4] != source.pixels[i * 4..i * 4 + 4]
        })
        .count();
    assert!(
        would_change > 0,
        "explicit protection must prevent a real edit"
    );
    for (i, &p) in protected.iter().enumerate().filter(|(_, p)| **p > 0) {
        assert_eq!(
            &protected_output.pixels[i * 4..i * 4 + 4],
            &source.pixels[i * 4..i * 4 + 4],
            "protected texture changed at {i}, mask {p}"
        );
    }
    eprintln!("explicit source texture: protected counterfactual={would_change}/36; zero and both epsilon signs passed");
    assert_eq!(
        incorrect_opaque, 0,
        "destination reconstruction replaced the immutable explicit scene texture: worst={worst:?}"
    );
}
