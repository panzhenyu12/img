//! A bounded, nonnegative scalar velocity on an eight-neighbour pixel graph.
//!
//! Each requested value is an upper bound, not an equality target. This
//! computes the greatest field below every bound whose adjacent differences
//! are at most `slope` on cardinal edges and `slope * sqrt(2)` diagonally:
//! `value(i) = min_j(request(j) + shortest_path_distance(i, j))`.
//! Explicit zero bounds remain zero and constrain nearby velocity before
//! integration. No rendered pixels or final displacements are clipped here.
//!
//! The guarantee is discrete, up to f32 arithmetic. For bilinear interpolation
//! on each cell, the cardinal bounds imply a scalar gradient norm at most
//! `sqrt(2) * slope`. They do not prove the final numerical warp or its sampled
//! image is fold-free. The caller must also make the grid's exterior coherent.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub(crate) fn bounded_scalar_field(
    width: usize,
    height: usize,
    requested: &[f32],
    slope: f32,
) -> Vec<f32> {
    assert!(width > 0 && height > 0, "nonempty scalar grid");
    let count = width
        .checked_mul(height)
        .expect("scalar grid size overflow");
    assert_eq!(requested.len(), count, "one bound per scalar grid cell");
    assert!(
        requested
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "finite nonnegative scalar bounds"
    );
    assert!(
        slope.is_finite() && slope >= 0.0,
        "finite nonnegative slope"
    );
    let diagonal = slope * std::f32::consts::SQRT_2;
    assert!(diagonal.is_finite(), "finite diagonal slope");

    // The IEEE bits of nonnegative finite f32 values have numeric ordering.
    // Canonicalise -0.0 too, because its sign bit would break that ordering.
    let mut field: Vec<f32> = requested
        .iter()
        .map(|&value| if value == 0.0 { 0.0 } else { value })
        .collect();
    let mut frontier: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::from(
        field
            .iter()
            .enumerate()
            .map(|(i, &value)| Reverse((value.to_bits(), i)))
            .collect::<Vec<_>>(),
    );
    while let Some(Reverse((bits, i))) = frontier.pop() {
        if bits != field[i].to_bits() {
            continue;
        }
        let value = f32::from_bits(bits);
        let (x, y) = (i % width, i / width);
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let Some(nx) = x.checked_add_signed(dx).filter(|&x| x < width) else {
                    continue;
                };
                let Some(ny) = y.checked_add_signed(dy).filter(|&y| y < height) else {
                    continue;
                };
                let j = ny * width + nx;
                let candidate = value + if dx == 0 || dy == 0 { slope } else { diagonal };
                // An overflowed sum is +infinity and cannot improve a finite
                // bound. It never enters the heap's finite numeric ordering.
                if candidate < field[j] {
                    field[j] = candidate;
                    frontier.push(Reverse((candidate.to_bits(), j)));
                }
            }
        }
    }
    field
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn one_dimensional_bounds_form_the_known_lower_envelope_without_wrapping_rows() {
        let requested = [9.0, 9.0, 0.25, 9.0, 9.0, 9.0, 0.0];
        let expected = [1.25, 0.75, 0.25, 0.75, 1.0, 0.5, 0.0];
        assert_eq!(bounded_scalar_field(7, 1, &requested, 0.5), expected);
        assert_eq!(bounded_scalar_field(1, 7, &requested, 0.5), expected);
        assert_eq!(bounded_scalar_field(1, 1, &[3.0], 0.5), [3.0]);
    }

    #[test]
    fn interior_zero_obstacle_uses_diagonal_and_cardinal_geometric_costs() {
        let mut requested = vec![9.0; 25];
        requested[2 * 5 + 2] = 0.0;
        let actual = bounded_scalar_field(5, 5, &requested, 1.0);
        close(actual[2 * 5 + 2], 0.0);
        close(actual[2 * 5 + 1], 1.0);
        close(actual[1 * 5 + 1], std::f32::consts::SQRT_2);
        close(actual[0], 2.0 * std::f32::consts::SQRT_2);
        close(actual[1], 1.0 + std::f32::consts::SQRT_2);
        close(actual[2], 2.0);
        assert_eq!(actual[0], actual[24]);
    }

    #[test]
    fn zeros_stay_fixed_and_an_attainable_target_keeps_its_full_magnitude() {
        let mut requested = vec![1.2; 49];
        for y in 0..7 {
            for x in 0..7 {
                if x == 0 || y == 0 || x == 6 || y == 6 {
                    requested[y * 7 + x] = 0.0;
                }
            }
        }
        let actual = bounded_scalar_field(7, 7, &requested, 0.45);
        assert_eq!(
            actual[3 * 7 + 3],
            1.2,
            "feasible target must not be rescaled"
        );
        for (i, &cap) in requested.iter().enumerate() {
            if cap == 0.0 {
                assert_eq!(actual[i], 0.0, "fixed boundary {i}");
            }
        }
        close(actual[1 * 7 + 3], 0.45);
        close(actual[2 * 7 + 3], 0.9);
    }

    #[test]
    fn all_requested_caps_and_all_eight_edge_bounds_hold() {
        let (width, height) = (19, 13);
        // Deterministic irregular caps, including interior and boundary zeros;
        // no expected solution is produced by copying the minimisation.
        let requested: Vec<_> = (0..width * height)
            .map(|i| {
                if i % 29 == 0 {
                    0.0
                } else {
                    ((i * 71 + 13) % 113) as f32 / 11.0
                }
            })
            .collect();
        let slope = 0.45;
        let actual = bounded_scalar_field(width, height, &requested, slope);
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                assert!(actual[i].is_finite() && actual[i] >= 0.0);
                assert!(actual[i] <= requested[i], "increased bound at {x},{y}");
                for (dx, dy, bound) in [
                    (1, 0, slope),
                    (0, 1, slope),
                    (1, 1, slope * std::f32::consts::SQRT_2),
                    (-1, 1, slope * std::f32::consts::SQRT_2),
                ] {
                    let Some(nx) = x.checked_add_signed(dx).filter(|&x| x < width) else {
                        continue;
                    };
                    let Some(ny) = y.checked_add_signed(dy).filter(|&y| y < height) else {
                        continue;
                    };
                    assert!(
                        (actual[i] - actual[ny * width + nx]).abs() <= bound + 1e-6,
                        "edge slope exceeded at {x},{y} -> {nx},{ny}"
                    );
                }
            }
        }
    }

    #[test]
    fn already_feasible_constants_and_ramps_are_not_attenuated() {
        assert_eq!(bounded_scalar_field(3, 3, &[2.0; 9], 0.45), [2.0; 9]);
        let ramp: Vec<_> = (0..30).map(|i| 0.125 * (i % 6 + i / 6) as f32).collect();
        assert_eq!(bounded_scalar_field(6, 5, &ramp, 0.25), ramp);
        assert_eq!(bounded_scalar_field(3, 1, &[2.0, 1.0, 3.0], 0.0), [1.0; 3]);
        let zero = bounded_scalar_field(2, 2, &[0.0, -0.0, 0.0, 0.0], 0.45);
        assert!(zero.iter().all(|value| value.to_bits() == 0));
    }

    #[test]
    fn malformed_grids_and_nonfinite_or_negative_bounds_are_rejected() {
        for (width, height, requested, slope) in [
            (0, 1, vec![], 0.45),
            (1, 0, vec![], 0.45),
            (2, 2, vec![1.0], 0.45),
            (1, 1, vec![f32::NAN], 0.45),
            (1, 1, vec![f32::INFINITY], 0.45),
            (1, 1, vec![-0.1], 0.45),
            (1, 1, vec![1.0], -0.45),
            (1, 1, vec![1.0], f32::NAN),
            (1, 1, vec![1.0], f32::INFINITY),
            (1, 1, vec![1.0], f32::MAX),
        ] {
            assert!(std::panic::catch_unwind(|| {
                bounded_scalar_field(width, height, &requested, slope)
            })
            .is_err());
        }
        assert!(
            std::panic::catch_unwind(|| { bounded_scalar_field(usize::MAX, 2, &[], 0.45) })
                .is_err()
        );
    }
}
