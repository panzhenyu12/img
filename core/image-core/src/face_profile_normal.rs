//! Physical-arclength normal estimates for a closed source contour.
//!
//! This estimates a direction; it does not certify that an input polygon or
//! offset target is simple, contained, or anatomically meaningful. The caller
//! must establish simple source topology before interpreting winding as inside.

/// Estimate one outward unit normal per original vertex. `support` is the
/// physical distance walked in each direction on the unchanged closed curve,
/// and must be positive and smaller than half its perimeter. Adjacent duplicate
/// vertices are allowed and receive identical estimates. No input vertices are
/// smoothed, resampled, dropped, or reordered in the returned correspondence.
pub(super) fn outward_normals(
    polygon: &[(f32, f32)],
    support: f32,
) -> Result<Vec<(f32, f32)>, &'static str> {
    if polygon.len() < 3 {
        return Err("at least three contour vertices required");
    }
    if !support.is_finite() || support <= 0.0 {
        return Err("normal support must be finite and positive");
    }
    if polygon
        .iter()
        .any(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return Err("contour vertices must be finite");
    }
    let points: Vec<(f64, f64)> = polygon.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let origin = points[0];
    let mut twice_area = 0.0_f64;
    let mut arclength = Vec::with_capacity(points.len() + 1);
    arclength.push(0.0_f64);
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        arclength.push(arclength[i] + (b.0 - a.0).hypot(b.1 - a.1));
        // Translating the area calculation avoids cancellation caused by a
        // large image-coordinate origin. f64 preserves the f32 input geometry.
        twice_area += (a.0 - origin.0) * (b.1 - origin.1) - (a.1 - origin.1) * (b.0 - origin.0);
    }
    let perimeter = arclength[points.len()];
    if perimeter == 0.0 || twice_area.abs() <= 64.0 * f64::EPSILON * perimeter * perimeter {
        return Err("contour has degenerate signed area");
    }
    let support = support as f64;
    if support >= perimeter * 0.5 {
        return Err("normal support must be smaller than half the perimeter");
    }
    let point_at = |distance: f64| {
        let distance = distance.rem_euclid(perimeter);
        // Upper-bound selection skips consecutive zero-length edges. The
        // selected interval has positive length even at duplicate vertices.
        let i = arclength
            .partition_point(|&s| s <= distance)
            .saturating_sub(1)
            .min(points.len() - 1);
        let t = (distance - arclength[i]) / (arclength[i + 1] - arclength[i]);
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
    };
    let winding = twice_area.signum();
    (0..points.len())
        .map(|i| {
            let a = point_at(arclength[i] - support);
            let b = point_at(arclength[i] + support);
            let dx = b.0 - a.0;
            let dy = b.1 - a.1;
            let length = dx.hypot(dy);
            if length <= 64.0 * f64::EPSILON * perimeter {
                return Err("normal direction is undefined at the requested support");
            }
            Ok((
                (winding * dy / length) as f32,
                (-winding * dx / length) as f32,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::outward_normals;

    #[test]
    fn concave_wall_points_into_the_notch_not_away_from_the_centroid() {
        let polygon = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (7.0, 10.0),
            (7.0, 6.0),
            (7.0, 3.0),
            (3.0, 3.0),
            (3.0, 10.0),
            (0.0, 10.0),
        ];
        let normals = outward_normals(&polygon, 1.0).unwrap();
        // The material is right of x=7, and the notch is left of it.
        assert_eq!(normals[4], (-1.0, 0.0));
        // Shrinking along -normal must remain in the material.
        assert!(polygon[4].0 - 0.1 * normals[4].0 > polygon[4].0);
    }

    #[test]
    fn winding_and_nonuniform_collinear_subdivision_preserve_material_normals() {
        let coarse = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let refined = [
            (0.0, 0.0),
            (9.75, 0.0),
            (10.0, 0.0),
            (10.0, 0.3),
            (10.0, 10.0),
            (8.0, 10.0),
            (0.0, 10.0),
            (0.0, 9.1),
        ];
        let expected = outward_normals(&coarse, 2.0).unwrap();
        let corner = -std::f32::consts::FRAC_1_SQRT_2;
        assert!((expected[0].0 - corner).abs() < 1e-6);
        assert!((expected[0].1 - corner).abs() < 1e-6);
        for points in [refined.to_vec(), refined.iter().copied().rev().collect()] {
            let actual = outward_normals(&points, 2.0).unwrap();
            for (i, &point) in coarse.iter().enumerate() {
                let j = points.iter().position(|p| *p == point).unwrap();
                assert!(
                    (actual[j].0 - expected[i].0).abs() < 1e-6,
                    "normal x changed at {point:?}: {:?} vs {:?}",
                    actual[j],
                    expected[i]
                );
                assert!(
                    (actual[j].1 - expected[i].1).abs() < 1e-6,
                    "normal y changed at {point:?}: {:?} vs {:?}",
                    actual[j],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn repeated_adjacent_and_closing_vertices_keep_original_correspondence() {
        let polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let repeated = [
            polygon[0], polygon[0], polygon[1], polygon[1], polygon[1], polygon[2], polygon[3],
            polygon[0], polygon[0],
        ];
        let expected = outward_normals(&polygon, 2.0).unwrap();
        let actual = outward_normals(&repeated, 2.0).unwrap();
        assert_eq!(actual.len(), repeated.len());
        for (i, point) in repeated.iter().enumerate() {
            let j = polygon.iter().position(|p| p == point).unwrap();
            assert_eq!(actual[i], expected[j]);
        }
    }

    #[test]
    fn rejects_invalid_support_nonfinite_and_degenerate_geometry() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        for support in [0.0, -1.0, f32::NAN, f32::INFINITY, 20.0, 21.0] {
            assert!(outward_normals(&square, support).is_err());
        }
        for polygon in [
            vec![],
            vec![(1.0, 1.0); 4],
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            vec![(0.0, 0.0), (f32::NAN, 1.0), (2.0, 2.0)],
            vec![(0.0, 0.0), (1.0, f32::INFINITY), (2.0, 2.0)],
        ] {
            assert!(outward_normals(&polygon, 1.0).is_err());
        }
    }

    #[test]
    fn rigid_motion_and_uniform_scale_transform_the_estimated_direction() {
        let polygon = [(0.0, 0.0), (9.0, 0.0), (10.0, 8.0), (1.0, 10.0)];
        let transformed: Vec<_> = polygon
            .iter()
            .map(|&(x, y)| (32.0 - 2.0 * y, 64.0 + 2.0 * x))
            .collect();
        let before = outward_normals(&polygon, 1.5).unwrap();
        let after = outward_normals(&transformed, 3.0).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((b.0 + a.1).abs() < 1e-6);
            assert!((b.1 - a.0).abs() < 1e-6);
            assert!((b.0.hypot(b.1) - 1.0).abs() < 1e-6);
        }
    }
}
