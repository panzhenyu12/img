//! Positive compact interpolation for connected portrait velocity fields.
//!
//! Inverse-square Shepard weights are multiplied by the compact Wendland C2
//! kernel. A zero-velocity background has weight one. This keeps the field in
//! the convex hull of zero and the supplied control velocities, including at
//! the support boundary; normalizing only the compact weights would instead
//! jump to zero where the final active control disappears.
//!
//! This bounds interpolation overshoot, not the supplied control values or the
//! derivative of the integrated flow. In particular, close conflicting targets
//! and residual updates can still create large strain. Nonzero constants are
//! attenuated away from controls, so realized contour motion must be measured.

use std::collections::HashMap;

type Point = (f32, f32);

/// Radius-sized cells store only potentially overlapping controls. Each control
/// is appended to the surrounding nine cells in its original input order, so
/// evaluating a cell preserves the dense implementation's accumulation order.
struct CandidateIndex {
    radius: f64,
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl CandidateIndex {
    fn cell(p: Point, radius: f64) -> Option<(i32, i32)> {
        // A sparse map never allocates according to coordinate span. Still use
        // a conservative coordinate/radius limit: at this scale, f64 division
        // of f32 inputs resolves integer cell boundaries with ample precision.
        // Extreme coordinates and nonfinite queries retain the dense path.
        const MAX_CELL_COORDINATE: f64 = (1_u32 << 20) as f64;
        let x = p.0 as f64 / radius;
        let y = p.1 as f64 / radius;
        if !x.is_finite()
            || !y.is_finite()
            || x.abs() > MAX_CELL_COORDINATE
            || y.abs() > MAX_CELL_COORDINATE
        {
            return None;
        }
        Some((x.floor() as i32, y.floor() as i32))
    }

    fn new(controls: &[Point], radius: f64) -> Option<Self> {
        let mut cells = HashMap::<_, Vec<usize>>::new();
        for (index, &control) in controls.iter().enumerate() {
            let (x, y) = Self::cell(control, radius)?;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    cells.entry((x + dx, y + dy)).or_default().push(index);
                }
            }
        }
        Some(Self { radius, cells })
    }

    fn candidates(&self, p: Point) -> Option<&[usize]> {
        let cell = Self::cell(p, self.radius)?;
        Some(self.cells.get(&cell).map_or(&[], Vec::as_slice))
    }
}

pub(super) struct PositiveKernel {
    controls: Vec<Point>,
    radius_squared: f64,
    candidate_index: Option<CandidateIndex>,
}

impl PositiveKernel {
    pub(super) fn new(controls: Vec<Point>, radius: f32) -> Self {
        assert!(radius.is_finite() && radius > 0.0, "positive finite radius");
        assert!(
            controls.iter().all(|p| p.0.is_finite() && p.1.is_finite()),
            "finite control positions"
        );
        let candidate_index = CandidateIndex::new(&controls, radius as f64);
        Self {
            controls,
            radius_squared: (radius as f64).powi(2),
            candidate_index,
        }
    }

    pub(super) fn evaluate(&self, values: &[Point], p: Point) -> Point {
        assert_eq!(values.len(), self.controls.len(), "one value per control");
        if let Some(indices) = self
            .candidate_index
            .as_ref()
            .and_then(|index| index.candidates(p))
        {
            self.evaluate_indices(values, p, indices.iter().copied())
        } else {
            self.evaluate_indices(values, p, 0..self.controls.len())
        }
    }

    fn evaluate_indices(
        &self,
        values: &[Point],
        p: Point,
        indices: impl Iterator<Item = usize>,
    ) -> Point {
        let mut numerator = (0.0_f64, 0.0_f64);
        // Explicit zero-velocity background: its numerator contribution is 0.
        let mut denominator = 1.0_f64;
        let mut coincident_sum = (0.0_f64, 0.0_f64);
        let mut coincident_count = 0_usize;
        for index in indices {
            let control = self.controls[index];
            let value = values[index];
            // Convert before subtraction. Near distinct f32 coordinates are
            // not merged by a distance threshold or destabilized by a solve.
            let dx = p.0 as f64 - control.0 as f64;
            let dy = p.1 as f64 - control.1 as f64;
            let distance_squared = dx * dx + dy * dy;
            if distance_squared == 0.0 {
                // Exact interpolation for unique controls. Conflicting values
                // at an identical coordinate cannot all be interpolated; their
                // mean is the continuous inverse-square limit of this model.
                coincident_sum.0 += value.0 as f64;
                coincident_sum.1 += value.1 as f64;
                coincident_count += 1;
                continue;
            }
            if distance_squared >= self.radius_squared {
                continue;
            }
            let r_squared = distance_squared / self.radius_squared;
            let r = r_squared.sqrt();
            let t = 1.0 - r;
            let weight = t.powi(4) * (4.0 * r + 1.0) / r_squared;
            numerator.0 += weight * value.0 as f64;
            numerator.1 += weight * value.1 as f64;
            denominator += weight;
        }
        if coincident_count > 0 {
            return (
                (coincident_sum.0 / coincident_count as f64) as f32,
                (coincident_sum.1 / coincident_count as f64) as f32,
            );
        }
        (
            (numerator.0 / denominator) as f32,
            (numerator.1 / denominator) as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(p: Point) -> f32 {
        p.0.hypot(p.1)
    }

    // Independent dense reference: intentionally does not call the indexed
    // evaluator or share its candidate selection/accumulation helper.
    fn dense_reference(controls: &[Point], values: &[Point], radius: f32, p: Point) -> Point {
        let radius_squared = (radius as f64).powi(2);
        let mut numerator = (0.0_f64, 0.0_f64);
        let mut denominator = 1.0_f64;
        let mut coincident_sum = (0.0_f64, 0.0_f64);
        let mut coincident_count = 0_usize;
        for (&control, &value) in controls.iter().zip(values) {
            let dx = p.0 as f64 - control.0 as f64;
            let dy = p.1 as f64 - control.1 as f64;
            let distance_squared = dx * dx + dy * dy;
            if distance_squared == 0.0 {
                coincident_sum.0 += value.0 as f64;
                coincident_sum.1 += value.1 as f64;
                coincident_count += 1;
                continue;
            }
            if distance_squared >= radius_squared {
                continue;
            }
            let r_squared = distance_squared / radius_squared;
            let r = r_squared.sqrt();
            let t = 1.0 - r;
            let weight = t.powi(4) * (4.0 * r + 1.0) / r_squared;
            numerator.0 += weight * value.0 as f64;
            numerator.1 += weight * value.1 as f64;
            denominator += weight;
        }
        if coincident_count > 0 {
            return (
                (coincident_sum.0 / coincident_count as f64) as f32,
                (coincident_sum.1 / coincident_count as f64) as f32,
            );
        }
        (
            (numerator.0 / denominator) as f32,
            (numerator.1 / denominator) as f32,
        )
    }

    fn assert_dense_bits(kernel: &PositiveKernel, values: &[Point], radius: f32, p: Point) {
        let actual = kernel.evaluate(values, p);
        let expected = dense_reference(&kernel.controls, values, radius, p);
        assert_eq!(
            (actual.0.to_bits(), actual.1.to_bits()),
            (expected.0.to_bits(), expected.1.to_bits()),
            "bitwise dense equivalence at {p:?}, radius {radius}"
        );
    }

    #[test]
    fn indexed_field_is_bitwise_dense_for_many_controls_negative_cells_and_translations() {
        for radius in [0.25, 8.0, 64.0] {
            for translation in [(0.0, 0.0), (-512.0, 400.0), (8192.0, -16384.0)] {
                let controls: Vec<_> = (0..193)
                    .map(|i| {
                        (
                            translation.0 + ((i * 37) % 197) as f32 * radius / 8.0 - radius * 12.0,
                            translation.1 + ((i * 71) % 193) as f32 * radius / 8.0 - radius * 12.0,
                        )
                    })
                    .collect();
                let values: Vec<_> = (0..controls.len())
                    .map(|i| {
                        let scale = [1.0e-12, -1.0e12, 1.0, -1.0][i % 4];
                        ((i % 13) as f32 * scale, (i % 17) as f32 * -scale)
                    })
                    .collect();
                let kernel = PositiveKernel::new(controls, radius);
                let index = kernel
                    .candidate_index
                    .as_ref()
                    .expect("ordinary coordinates indexed");
                assert!(index.candidates(translation).unwrap().len() < kernel.controls.len());
                for y in -20..=20 {
                    for x in -20..=20 {
                        let p = (
                            translation.0 + x as f32 * radius * 0.625,
                            translation.1 + y as f32 * radius * 0.625,
                        );
                        assert_dense_bits(&kernel, &values, radius, p);
                    }
                }
                for &p in &kernel.controls {
                    assert_dense_bits(&kernel, &values, radius, p);
                }
            }
        }
    }

    #[test]
    fn indexed_field_preserves_support_cell_boundaries_and_duplicate_accumulation_order() {
        for radius in [0.1_f32, 1.0, 10.0] {
            let positions = vec![
                (-radius, 0.0),
                (radius * 5.0, radius),
                (0.0, 0.0),
                (radius, 0.0),
                (0.0, 0.0),
                (-radius * 5.0, -radius),
                (0.0, 0.0),
                (0.0, radius),
            ];
            let values = [
                (3.0, -7.0),
                (11.0, 9.0),
                (1.0e30, -1.0e30),
                (-8.0, 3.0),
                (-1.0e30, 1.0e30),
                (21.0, 9.0),
                (7.0, -11.0),
                (-4.0, 6.0),
            ];
            let kernel = PositiveKernel::new(positions.clone(), radius);
            let candidates = kernel
                .candidate_index
                .as_ref()
                .unwrap()
                .candidates((0.0, 0.0))
                .unwrap();
            assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(candidates.len() < positions.len());
            for &p in &positions {
                assert_dense_bits(&kernel, &values, radius, p);
            }
            let inside = f32::from_bits(radius.to_bits() - 1);
            let outside = f32::from_bits(radius.to_bits() + 1);
            for coordinate in [
                -outside,
                -radius,
                -inside,
                -f32::from_bits(1),
                -0.0,
                0.0,
                f32::from_bits(1),
                inside,
                radius,
                outside,
            ] {
                for p in [
                    (coordinate, 0.0),
                    (0.0, coordinate),
                    (coordinate, coordinate),
                ] {
                    assert_dense_bits(&kernel, &values, radius, p);
                }
            }
        }
    }

    #[test]
    fn extreme_coordinates_and_nonfinite_queries_fall_back_to_bitwise_dense_behavior() {
        let radius = 1.0;
        let values = [(3.0, -7.0), (-2.0, 5.0)];
        let kernel = PositiveKernel::new(vec![(0.0, 0.0), (1.0, -1.0)], radius);
        let limit = (1_u32 << 20) as f32;
        for p in [
            (limit, -limit),
            (limit * 2.0, 0.0),
            (-limit * 2.0, 0.0),
            (f32::MAX, -f32::MAX),
            (f32::INFINITY, 0.0),
            (0.0, f32::NEG_INFINITY),
            (f32::NAN, 0.0),
            (0.0, f32::NAN),
        ] {
            assert_dense_bits(&kernel, &values, radius, p);
        }
        assert!(kernel
            .candidate_index
            .as_ref()
            .unwrap()
            .candidates((f32::MAX, 0.0))
            .is_none());
        assert!(kernel
            .candidate_index
            .as_ref()
            .unwrap()
            .candidates((f32::NAN, 0.0))
            .is_none());
        let extreme = PositiveKernel::new(vec![(0.0, 0.0), (f32::MAX, -f32::MAX)], radius);
        assert!(extreme.candidate_index.is_none());
        for &p in &extreme.controls {
            assert_dense_bits(&extreme, &values, radius, p);
        }
        let tiny = PositiveKernel::new(
            vec![(0.0, 0.0), (f32::from_bits(1), 0.0)],
            f32::from_bits(1),
        );
        for p in [(0.0, 0.0), (f32::from_bits(1), 0.0), (f32::MAX, 0.0)] {
            assert_dense_bits(&tiny, &values, f32::from_bits(1), p);
        }
        let empty = PositiveKernel::new(Vec::new(), radius);
        assert_dense_bits(&empty, &[], radius, (f32::NAN, f32::INFINITY));
    }

    #[test]
    fn signed_control_velocities_are_interpolated_exactly() {
        let positions = vec![(10.0, 20.0), (30.0, 5.0), (48.0, 40.0)];
        let values = [(8.0, -4.0), (-7.0, 9.0), (0.0, 0.0)];
        let kernel = PositiveKernel::new(positions.clone(), 30.0);
        for (&p, &value) in positions.iter().zip(&values) {
            assert_eq!(kernel.evaluate(&values, p), value);
            for offset in [(0.0001, 0.0), (0.0, -0.0001)] {
                let q = kernel.evaluate(&values, (p.0 + offset.0, p.1 + offset.1));
                assert!(norm((q.0 - value.0, q.1 - value.1)) < 0.00001);
            }
        }
    }

    #[test]
    fn zero_field_and_empty_support_are_exact_identity_velocities() {
        let kernel = PositiveKernel::new(vec![(10.0, 10.0), (30.0, 40.0)], 20.0);
        for y in -20..80 {
            for x in -20..80 {
                assert_eq!(
                    kernel.evaluate(&[(0.0, 0.0); 2], (x as f32, y as f32)),
                    (0.0, 0.0)
                );
            }
        }
        assert_eq!(
            kernel.evaluate(&[(5.0, 9.0), (-7.0, 11.0)], (-50.0, 0.0)),
            (0.0, 0.0)
        );
        assert_eq!(
            PositiveKernel::new(Vec::new(), 10.0).evaluate(&[], (0.0, 0.0)),
            (0.0, 0.0)
        );
    }

    #[test]
    fn field_cannot_overshoot_control_convex_hull_in_any_checked_direction() {
        let kernel = PositiveKernel::new(vec![(0.0, 0.0), (8.0, 0.0), (4.0, 9.0)], 18.0);
        let values = [(8.0, 0.0), (0.0, 6.0), (-2.0, -3.0)];
        for y in -20..30 {
            for x in -20..30 {
                let actual = kernel.evaluate(&values, (x as f32 * 0.7, y as f32 * 0.7));
                assert!(norm(actual) <= 8.0 + 0.00001);
                for k in 0..32 {
                    let angle = k as f32 * std::f32::consts::TAU / 32.0;
                    let direction = (angle.cos(), angle.sin());
                    let project = |v: Point| direction.0 * v.0 + direction.1 * v.1;
                    let upper = values.iter().map(|&v| project(v)).fold(0.0_f32, f32::max);
                    let lower = values.iter().map(|&v| project(v)).fold(0.0_f32, f32::min);
                    assert!(project(actual) <= upper + 0.00001);
                    assert!(project(actual) >= lower - 0.00001);
                }
            }
        }
    }

    #[test]
    fn last_control_disappears_continuously_at_the_compact_support_boundary() {
        let kernel = PositiveKernel::new(vec![(0.0, 0.0)], 10.0);
        let values = [(8.0, -4.0)];
        let a = norm(kernel.evaluate(&values, (9.9, 0.0)));
        let b = norm(kernel.evaluate(&values, (9.99, 0.0)));
        let c = norm(kernel.evaluate(&values, (9.999, 0.0)));
        assert!(a < 0.000001);
        assert!(b < a * 0.001);
        assert!(c < b * 0.001);
        assert_eq!(kernel.evaluate(&values, (10.0, 0.0)), (0.0, 0.0));
        assert_eq!(kernel.evaluate(&values, (10.01, 0.0)), (0.0, 0.0));
    }

    #[test]
    fn coincident_and_near_coincident_controls_are_finite_without_merging_distinct_positions() {
        let positions = vec![(1.0, 1.0), (1.0, 1.0), (1.00002, 1.0)];
        let values = [(8.0, -4.0), (4.0, 2.0), (-3.0, 7.0)];
        let kernel = PositiveKernel::new(positions.clone(), 10.0);
        assert_eq!(kernel.evaluate(&values, positions[0]), (6.0, -1.0));
        assert_eq!(kernel.evaluate(&values, positions[2]), values[2]);
        for x in [0.9999999, 1.0000001, 1.00001, 1.000019, 1.000021] {
            let actual = kernel.evaluate(&values, (x, 1.0));
            assert!(actual.0.is_finite() && actual.1.is_finite());
            assert!(norm(actual) <= norm(values[0]) + 0.00001);
        }
    }

    #[test]
    fn translation_rotation_and_uniform_scale_preserve_the_field_contract() {
        let positions = vec![(0.0, 0.0), (5.0, 0.0), (2.0, 4.0)];
        let values = [(3.0, -1.0), (-2.0, 4.0), (1.0, 2.0)];
        let kernel = PositiveKernel::new(positions.clone(), 12.0);
        let transform_point = |p: Point| (100.0 - 2.0 * p.1, -50.0 + 2.0 * p.0);
        let transform_vector = |v: Point| (-2.0 * v.1, 2.0 * v.0);
        let transformed =
            PositiveKernel::new(positions.into_iter().map(transform_point).collect(), 24.0);
        let transformed_values: Vec<_> = values.into_iter().map(transform_vector).collect();
        for p in [(0.0, 0.0), (2.5, 1.0), (-3.0, 4.0), (40.0, 40.0)] {
            let expected = transform_vector(kernel.evaluate(&values, p));
            let actual = transformed.evaluate(&transformed_values, transform_point(p));
            assert!(norm((actual.0 - expected.0, actual.1 - expected.1)) < 0.00001);
        }
    }
}
