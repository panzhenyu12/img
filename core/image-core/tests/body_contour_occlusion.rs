use image_core::{prepare_body_inverse_displacement, RgbaImage};
use serde_json::json;

/// An object in front of the centre of the torso must not pin a visible,
/// distant side seam at the same height. This is the laptop/jacket failure:
/// the input has one straight side, but the old field added two waist lobes.
#[test]
fn a_central_occluder_does_not_split_the_opposite_waist_curve() {
    let (w, h) = (240usize, 240usize);
    let mut pixels = vec![255; w * h * 4];
    let mut subject = vec![0u8; w * h];
    let mut part = vec![0u8; w * h];
    let mut protect = vec![0u8; w * h];
    for y in 20..220 {
        for x in 40..205 {
            let i = y * w + x;
            subject[i] = 255;
            part[i] = 255;
            pixels[i * 4..i * 4 + 3].copy_from_slice(&[50, 60, 70]);
        }
    }
    for y in 105..140 {
        for x in 30..132 {
            let i = y * w + x;
            part[i] = 0;
            protect[i] = 255;
            pixels[i * 4..i * 4 + 3].copy_from_slice(&[180, 190, 200]);
        }
    }
    let source = RgbaImage::new(w as u32, h as u32, pixels).unwrap();
    let mask = |data: &[u8]| json!({"shape":"raster","width":w,"height":h,"data":data});
    for amount in [-30, 30] {
        let params = json!({"bodyPart":"waist",
            "region":{"x":0.01,"y":0.01,"width":0.98,"height":0.98},
            "landmarks":[{"x":0.5,"y":0.1},{"x":0.5,"y":0.5},{"x":0.5,"y":0.9}],
            "subjectMask":mask(&subject),"partMask":mask(&part),
            "protectMask":mask(&protect),"adjustments":{"waist":amount}});
        let field = prepare_body_inverse_displacement(&source, &params).unwrap();
        let edge: Vec<f32> = (78..164).map(|y| field[y * w + 203][0].abs()).collect();
        assert!(field[120 * w + 203][0].abs() > 5.0,
            "the opposite visible side was pinned by an occluded bone axis at {amount}");
        let peak = edge.iter().copied().fold(0.0, f32::max);
        let variation: f32 = edge.windows(2).map(|v| (v[1] - v[0]).abs()).sum();
        // A one-lobed waist envelope has no extra valleys/peaks. Allow 2%
        // numerical slack; do not merely check for a positive Jacobian.
        let single_lobe = 2.0 * peak - edge[0] - edge[edge.len() - 1];
        assert!(variation <= single_lobe * 1.02 + 0.01,
            "occluder added side-curve lobes: variation={variation}, single={single_lobe}");
        for (i, &fixed) in protect.iter().enumerate() {
            if fixed != 0 { assert_eq!(field[i], [0.0, 0.0]); }
        }
    }
}

#[test]
fn a_mislocated_axis_in_open_background_cannot_adopt_a_detached_side_piece() {
    let (w, h) = (120usize, 160usize);
    let mut part = vec![0u8; w * h];
    for y in 25..140 { for x in 80..105 { part[y * w + x] = 255; }}
    let source = RgbaImage::new(w as u32, h as u32, [80, 90, 100, 255].repeat(w*h)).unwrap();
    let mask = json!({"shape":"raster","width":w,"height":h,"data":part});
    let params = json!({"bodyPart":"waist",
        "region":{"x":0.01,"y":0.01,"width":0.98,"height":0.98},
        "landmarks":[{"x":0.5,"y":0.1},{"x":0.5,"y":0.5},{"x":0.5,"y":0.9}],
        "subjectMask":mask,"partMask":mask,"adjustments":{"waist":30}});
    let field = prepare_body_inverse_displacement(&source, &params).unwrap();
    assert!(field.iter().all(|&d| d == [0.0, 0.0]),
        "a background gap was treated as evidence of an occluded torso");
}
