//! Experimental source-relative contour displacement projection.
//! This module is not enabled by the production face path.
//!
//! Source anatomy is not required to be monotone or convex. Only the added
//! lateral displacement is faired and slope-limited. These one-dimensional
//! conditions do not certify the integrated image field, topology or appearance.

/// Samples: (source anatomical vertical coordinate, oriented source lateral,
/// desired lateral, requested lateral limit, exact fixed flag).
pub(super) fn project_side_curve(
    samples: &[(f32, f32, f32, f32, bool)],
) -> Result<Vec<f32>, String> {
    project(
        samples,
        BoundaryPolicy::LegacyEndpointLocks,
        None,
        CurveTopology::Open,
    )
}

/// Research candidate: fair the *added lateral displacement* while keeping
/// the requested signed area and its vertical first moment.  The old
/// displacement candidate minimized curvature but could pay for smoothness by
/// shrinking the edit (for the user -100 replay, area response fell from
/// 9.35% to 5.93%).  For a side contour the area contribution is
/// `integral displacement * d(vertical)`, so the two linear moments below are
/// the geometric quantities that must survive fairing.  This is intentionally
/// opt-in; it still only certifies the one-dimensional contour, not the full
/// image transport or material seam.
pub(super) fn project_preserving_area_moments(
    samples: &[(f32, f32, f32, f32, bool)],
    physical_bending_length: f32,
) -> Result<Vec<f32>, String> {
    use clarabel::{algebra::CscMatrix, solver::*};

    let n = samples.len();
    if n < 3 {
        return Ok(samples.iter().map(|s| s.2).collect());
    }
    if !physical_bending_length.is_finite() || physical_bending_length <= 0.0 {
        return Err("area-moment curve requires a positive finite bending length".into());
    }
    if samples
        .iter()
        .any(|s| !s.0.is_finite() || !s.1.is_finite() || !s.2.is_finite() || !s.3.is_finite())
    {
        return Err("area-moment curve requires finite samples".into());
    }

    // Work in normalized displacement z=d/L and use vertical quadrature for
    // the data term.  This matches the first variation of polygon area rather
    // than allowing dense or oblique sampling to gain authority.
    let mut vertical_gaps: Vec<f64> = samples
        .windows(2)
        .map(|pair| (pair[1].0 as f64 - pair[0].0 as f64).abs())
        .collect();
    let chord_gaps: Vec<f64> = samples
        .windows(2)
        .map(|pair| {
            (pair[1].0 as f64 - pair[0].0 as f64).hypot(pair[1].1 as f64 - pair[0].1 as f64)
        })
        .collect();
    // A tangent or detector duplicate can have zero vertical span; retain a
    // small chord quadrature there rather than dropping the control.
    let vertical_total: f64 = vertical_gaps.iter().sum();
    if vertical_total <= 1e-9 {
        vertical_gaps = chord_gaps.clone();
    } else {
        for (v, &c) in vertical_gaps.iter_mut().zip(&chord_gaps) {
            if *v <= 1e-9 {
                *v = c.max(1e-6);
            }
        }
    }
    let length: f64 = vertical_gaps.iter().sum();
    if !length.is_finite() || length <= 1e-9 {
        return Err("area-moment curve has no measurable vertical span".into());
    }
    let mut weights = vec![0.0; n];
    weights[0] = vertical_gaps[0] * 0.5;
    weights[n - 1] = vertical_gaps[n - 2] * 0.5;
    for i in 1..n - 1 {
        weights[i] = (vertical_gaps[i - 1] + vertical_gaps[i]) * 0.5;
    }
    for weight in &mut weights {
        *weight /= length;
    }
    let mut ordinate = vec![0.0; n];
    for i in 1..n {
        ordinate[i] = ordinate[i - 1] + vertical_gaps[i - 1] / length;
    }
    let desired: Vec<f64> = samples
        .iter()
        .map(|s| (s.2 as f64 - s.1 as f64) / length)
        .collect();
    let target_area: f64 = weights.iter().zip(&desired).map(|(w, d)| w * d).sum();
    let target_moment: f64 = weights
        .iter()
        .zip(&ordinate)
        .zip(&desired)
        .map(|((w, t), d)| w * t * d)
        .sum();

    let mut p = vec![vec![0.0_f64; n]; n];
    let mut q = vec![0.0_f64; n];
    for i in 0..n {
        p[i][i] = 2.0 * weights[i];
        q[i] = -2.0 * weights[i] * desired[i];
    }
    // Physical displacement bending: source curvature is deliberately absent
    // from this energy.  Otherwise a detector's old contour kink is replayed
    // in the new face shape.
    let ell = physical_bending_length as f64;
    let normalized_ell = ell / length;
    for i in 1..n - 1 {
        let h0 = (vertical_gaps[i - 1] / length).max(1e-6);
        let h1 = (vertical_gaps[i] / length).max(1e-6);
        let dual = ((h0 + h1) * 0.5).max(1e-6);
        let row = [
            1.0 / (h0 * dual),
            -(1.0 / h0 + 1.0 / h1) / dual,
            1.0 / (h1 * dual),
        ];
        let weight = normalized_ell.powi(4) * dual;
        for (a, &ca) in [i - 1, i, i + 1].iter().zip(row.iter()) {
            for (b, &cb) in [i - 1, i, i + 1].iter().zip(row.iter()) {
                p[*a][*b] += 2.0 * weight * ca * cb;
            }
        }
    }

    let mut rows = Vec::<Vec<f64>>::new();
    let mut bounds = Vec::<f64>::new();
    // True fixed material is an equality. The two area moments are high-weight
    // quadratic terms rather than hard equalities: a detector attachment can
    // make the exact requested moment infeasible under a C1 slope bound. A
    // measured residual is safer than making the whole edit fail or silently
    // replacing the contour with the unconstrained target.
    for (i, sample) in samples.iter().enumerate() {
        if sample.4 {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            rows.push(row);
            bounds.push(0.0);
        }
    }
    let mut row = vec![0.0; n];
    let mut moment_row = vec![0.0; n];
    for i in 0..n {
        row[i] = weights[i];
        moment_row[i] = weights[i] * ordinate[i];
    }
    let moment_weight = 1.0e6_f64;
    for (a, &ra) in row.iter().enumerate() {
        for (b, &rb) in row.iter().enumerate() {
            p[a][b] += 2.0 * moment_weight * ra * rb;
        }
        q[a] -= 2.0 * moment_weight * target_area * ra;
    }
    for (a, &ra) in moment_row.iter().enumerate() {
        for (b, &rb) in moment_row.iter().enumerate() {
            p[a][b] += 2.0 * moment_weight * ra * rb;
        }
        q[a] -= 2.0 * moment_weight * target_moment * ra;
    }
    let equalities = rows.len();
    for (i, sample) in samples.iter().enumerate() {
        let requested = (sample.2 as f64 - sample.1 as f64) / length;
        let capacity = (sample.3 as f64 - sample.1 as f64) / length;
        // Fairing is allowed to spend less local motion when the transition
        // slope or an attachment requires it. The high-weight global moments
        // preserve the edit strength; constraining the interval between
        // requested and capacity would make that redistribution impossible.
        let lo = requested.min(capacity).min(0.0);
        let hi = requested.max(capacity).max(0.0);
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        rows.push(row);
        bounds.push(hi);
        let mut row = vec![0.0; n];
        row[i] = -1.0;
        rows.push(row);
        bounds.push(-lo);
    }
    for i in 0..n - 1 {
        let h = (chord_gaps[i] / length).max(1e-6);
        for sign in [-1.0, 1.0] {
            let mut row = vec![0.0; n];
            row[i] = -sign;
            row[i + 1] = sign;
            rows.push(row);
            // C1 transition guard. The high-weight moments retain the
            // requested strength while this bound removes concentrated jumps
            // without making a detector attachment infeasible.
            bounds.push(0.75 * h);
        }
    }
    let cones = vec![
        ZeroConeT(equalities),
        NonnegativeConeT(rows.len() - equalities),
    ];
    let settings = DefaultSettings {
        verbose: false,
        max_iter: 150,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(
        &CscMatrix::from(&p),
        &q,
        &CscMatrix::from(&rows),
        &bounds,
        &cones,
        settings,
    );
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(format!(
            "area-moment curve QP status {:?}",
            solver.solution.status
        ));
    }
    let z = &solver.solution.x;
    if z.iter().any(|v| !v.is_finite()) {
        return Err("area-moment curve QP produced a non-finite displacement".into());
    }
    let max_residual = rows
        .iter()
        .zip(&bounds)
        .enumerate()
        .map(|(i, (r, &b))| {
            let value: f64 = r.iter().zip(z).map(|(a, d)| a * d).sum();
            let residual = value - b;
            if i < equalities {
                residual.abs()
            } else {
                residual.max(0.0)
            }
        })
        .fold(0.0_f64, f64::max);
    if max_residual > 2e-6 {
        return Err(format!(
            "area-moment curve QP violated constraints ({max_residual})"
        ));
    }
    Ok(samples
        .iter()
        .zip(z)
        .map(|(sample, &value)| (sample.1 as f64 + value * length) as f32)
        .collect())
}

/// Project an observed contour arc including its unchanged neighbouring
/// context. Only actual fixed samples and zero local capacity are immovable;
/// a nonzero motion at an arbitrary slice boundary is not a hard constraint.
///
/// The caller passes the ORIGINAL core arc's bending length in image pixels
/// (currently L_core/25). Extending the context must not enlarge that length.
/// This routine preserves source order; it neither sorts samples nor creates
/// missing context. The caller must include the join with the untouched arc.
pub(super) fn project_context_curve(
    samples: &[(f32, f32, f32, f32, bool)],
    physical_bending_length: f32,
) -> Result<Vec<f32>, String> {
    if !physical_bending_length.is_finite() || physical_bending_length <= 0.0 {
        return Err("context curve requires a positive finite physical bending length".into());
    }
    project(
        samples,
        BoundaryPolicy::ObservedConstraintsOnly,
        Some(physical_bending_length as f64),
        CurveTopology::Open,
    )
}

/// Joint projection of one complete source contour in cyclic source order.
/// Do not repeat its first sample at the end. The final-to-first edge has the
/// same data quadrature, bending and slope constraints as every other edge;
/// no cut, nonzero endpoint lock or sequential overwrite exists.
///
/// The current caller supplies (L_left_core + L_right_core)/50: the mean of
/// the preceding experiment's two physical bending lengths, not L_closed/25.
/// This is a different, explicit experiment; source/request/capacity remain
/// unchanged. Periodic displacement smoothness is not a topology certificate.
pub(super) fn project_closed_curve(
    samples: &[(f32, f32, f32, f32, bool)],
    physical_bending_length: f32,
) -> Result<Vec<f32>, String> {
    if !physical_bending_length.is_finite() || physical_bending_length <= 0.0 {
        return Err("closed curve requires a positive finite physical bending length".into());
    }
    project(
        samples,
        BoundaryPolicy::ObservedConstraintsOnly,
        Some(physical_bending_length as f64),
        CurveTopology::Closed,
    )
}

/// Add ORIGINAL cyclic contour neighbours until each end contains two existing
/// zero-capacity samples. Capacity is the caller's signed, clamped desired
/// displacement with true fixed samples already set to zero. No new capacity,
/// interpolation, vertical sorting or approximate-zero threshold is introduced.
/// If no such open arc contains the core, a periodic solve is required instead.
pub(super) fn extend_zero_capacity_context(
    core_indices: &[usize],
    capacity: &[f32],
) -> Result<Vec<usize>, String> {
    let count = capacity.len();
    if count < 2 || core_indices.len() < 2 || core_indices.len() > count {
        return Err("context arc requires at least two valid source indices".into());
    }
    if capacity.iter().any(|v| !v.is_finite()) || core_indices.iter().any(|&i| i >= count) {
        return Err("context arc requires finite capacities and in-range indices".into());
    }
    let next = |i: usize| if i + 1 == count { 0 } else { i + 1 };
    let previous = |i: usize| if i == 0 { count - 1 } else { i - 1 };
    let forward = if next(core_indices[0]) == core_indices[1] {
        true
    } else if previous(core_indices[0]) == core_indices[1] {
        false
    } else {
        return Err("context core is not contiguous in source contour order".into());
    };
    if core_indices.windows(2).any(|edge| {
        (if forward {
            next(edge[0])
        } else {
            previous(edge[0])
        }) != edge[1]
    }) {
        return Err("context core is not contiguous in source contour order".into());
    }
    let mut indices = core_indices.to_vec();
    for before in [true, false] {
        loop {
            let end = if before {
                &indices[..2]
            } else {
                &indices[indices.len() - 2..]
            };
            if end.iter().all(|&i| capacity[i] == 0.0) {
                break;
            }
            if indices.len() == count {
                return Err(
                    "context arc has no zero-capacity end; periodic projection required".into(),
                );
            }
            let anchor = if before {
                indices[0]
            } else {
                *indices.last().unwrap()
            };
            let index = if forward == before {
                previous(anchor)
            } else {
                next(anchor)
            };
            if before {
                indices.insert(0, index);
            } else {
                indices.push(index);
            }
        }
    }
    Ok(indices)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BoundaryPolicy {
    LegacyEndpointLocks,
    ObservedConstraintsOnly,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CurveTopology {
    Open,
    Closed,
}

fn project(
    samples: &[(f32, f32, f32, f32, bool)],
    boundary_policy: BoundaryPolicy,
    physical_bending_length: Option<f64>,
    topology: CurveTopology,
) -> Result<Vec<f32>, String> {
    use clarabel::{algebra::CscMatrix, solver::*};

    let n = samples.len();
    let closed = topology == CurveTopology::Closed;
    if closed && n < 3 {
        return Err("closed displacement curve requires at least three source points".into());
    }
    if samples
        .iter()
        .any(|s| !s.0.is_finite() || !s.1.is_finite() || !s.2.is_finite() || !s.3.is_finite())
    {
        return Err("displacement curve requires finite coordinates".into());
    }
    if n == 0 || samples.iter().all(|s| s.4 || s.2 == s.1 || s.3 == s.1) {
        return Ok(samples.iter().map(|s| s.1).collect());
    }
    if n == 1 {
        let s = samples[0];
        return Ok(vec![s.2.clamp(s.1.min(s.3), s.1.max(s.3))]);
    }

    // Source chord lengths approximate source arc length. Do not measure gaps
    // only along y: a side curve may turn or contain equal vertical coordinates.
    let mut gaps: Vec<f64> = samples
        .windows(2)
        .map(|s| (s[1].0 as f64 - s[0].0 as f64).hypot(s[1].1 as f64 - s[0].1 as f64))
        .collect();
    if closed {
        gaps.push(
            (samples[0].0 as f64 - samples[n - 1].0 as f64)
                .hypot(samples[0].1 as f64 - samples[n - 1].1 as f64),
        );
    }
    if gaps.iter().any(|&h| h == 0.0) {
        return Err("displacement curve requires distinct adjacent source points".into());
    }
    let length: f64 = gaps.iter().sum();
    for h in &mut gaps {
        *h /= length;
    }
    // Solve in z=d/L and normalized source arc coordinate t=s/L. This avoids
    // image-unit-dependent solver tolerances and makes uniform scaling commute
    // with the projection. In physical units the objective is
    //   (1/L) integral [(d-desired)^2 + ell^4 (d''(s))^2] ds.
    // The legacy slice uses ell=L/25. The contextual path receives the core's
    // original ell explicitly, independently of its expanded sample domain.
    let bending_weight = physical_bending_length
        .map(|ell| (ell / length).powi(4))
        .unwrap_or_else(|| (1.0_f64 / 25.0).powi(4));
    let mut p = vec![vec![0.0; n]; n];
    let mut q = vec![0.0; n];
    // The caller already reduces desired motion where attachment reachability
    // is low. Fairing cannot spend the larger raw request in that region.
    // Clamp desired to the original requested direction/range, then use it as
    // both the objective target and the local displacement capacity.
    let desired_lateral: Vec<_> = samples
        .iter()
        .map(|s| s.2.clamp(s.1.min(s.3), s.1.max(s.3)))
        .collect();
    let desired: Vec<_> = samples
        .iter()
        .zip(&desired_lateral)
        .map(|(s, &target)| (target as f64 - s.1 as f64) / length)
        .collect();
    // Preserve the previous experiment's endpoint locks only in its own API.
    // Context arcs use observed zero-capacity neighbours for their boundary;
    // closed curves have no endpoints. All non-legacy constraints admit zero,
    // so zero displacement is always feasible
    // even when the requested motion is too steep near a fixed attachment.
    let locked: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.4 {
                Some(0.0)
            } else if boundary_policy == BoundaryPolicy::LegacyEndpointLocks
                && (i < 2 || i + 2 >= n)
            {
                Some(desired[i])
            } else {
                None
            }
        })
        .collect();

    // Propagate feasible intervals through the one-dimensional slope bounds.
    // A locked endpoint may conflict with another lock or a low-capacity
    // interior. Report that incompatibility; never substitute the raw target.
    let mut reachable = (0.0, 0.0);
    for i in 0..n {
        let mut interval = locked[i]
            .map(|d| (d, d))
            .unwrap_or((desired[i].min(0.0), desired[i].max(0.0)));
        if i > 0 {
            let reach = 0.25 * gaps[i - 1];
            interval.0 = interval.0.max(reachable.0 - reach);
            interval.1 = interval.1.min(reachable.1 + reach);
        }
        if interval.0 > interval.1 + 1e-10 {
            return Err(format!(
                "displacement curve motion bounds conflict with slope limit at sample {i}"
            ));
        }
        reachable = interval;
    }
    for i in 0..n {
        // Trapezoidal data quadrature gives a denser part of the same source
        // curve no extra authority merely because it contains more samples.
        let width = if closed {
            (gaps[(i + n - 1) % n] + gaps[i]) * 0.5
        } else {
            (if i > 0 { gaps[i - 1] } else { 0.0 } + if i + 1 < n { gaps[i] } else { 0.0 }) * 0.5
        };
        p[i][i] = 2.0 * width;
        q[i] = -2.0 * width * desired[i];
    }
    for i in if closed { 0..n } else { 1..n - 1 } {
        let previous = (i + n - 1) % n;
        let next = (i + 1) % n;
        let (h0, h1) = (gaps[previous], gaps[i]);
        let dual = (h0 + h1) * 0.5;
        let indices = [previous, i, next];
        let curvature = [
            1.0 / (h0 * dual),
            -(1.0 / h0 + 1.0 / h1) / dual,
            1.0 / (h1 * dual),
        ];
        for (&a, &ca) in indices.iter().zip(&curvature) {
            for (&b, &cb) in indices.iter().zip(&curvature) {
                p[a][b] += 2.0 * bending_weight * dual * ca * cb;
            }
        }
    }

    let mut rows = Vec::<Vec<f64>>::new();
    let mut bounds = Vec::<f64>::new();
    for (i, &lock) in locked.iter().enumerate() {
        if let Some(displacement) = lock {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            rows.push(row);
            bounds.push(displacement);
        }
    }
    let equalities = rows.len();
    for (i, &limit) in desired.iter().enumerate() {
        for (sign, bound) in [(1.0, limit.max(0.0)), (-1.0, -limit.min(0.0))] {
            let mut row = vec![0.0; n];
            row[i] = sign;
            rows.push(row);
            bounds.push(bound);
        }
    }
    for (i, &gap) in gaps.iter().enumerate() {
        // Bound added displacement slope in either direction, independently
        // of the sign of the source contour slope or its existing curvature.
        for sign in [-1.0, 1.0] {
            let mut row = vec![0.0; n];
            row[i] = -sign;
            row[(i + 1) % n] = sign;
            rows.push(row);
            bounds.push(0.25 * gap);
        }
    }
    let cones = vec![
        ZeroConeT(equalities),
        NonnegativeConeT(rows.len() - equalities),
    ];
    let settings = DefaultSettings {
        verbose: false,
        max_iter: 100,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(
        &CscMatrix::from(&p),
        &q,
        &CscMatrix::from(&rows),
        &bounds,
        &cones,
        settings,
    );
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(format!(
            "displacement curve QP status {:?}",
            solver.solution.status
        ));
    }
    let displacement = &solver.solution.x;
    if displacement.iter().any(|v| !v.is_finite())
        || rows.iter().zip(&bounds).enumerate().any(|(i, (row, &b))| {
            let residual: f64 = row
                .iter()
                .zip(displacement)
                .map(|(a, d)| a * d)
                .sum::<f64>()
                - b;
            if i < equalities {
                residual.abs() > 1e-7
            } else {
                residual > 1e-7
            }
        })
    {
        return Err("displacement curve QP violated a motion constraint".into());
    }
    Ok(samples
        .iter()
        .zip(displacement)
        .zip(&desired_lateral)
        .enumerate()
        .map(|(i, ((s, &d), &target))| {
            if s.4 || s.1 == target {
                s.1
            } else if locked[i].is_some() {
                target
            } else {
                ((s.1 as f64 + length * d) as f32).clamp(s.1.min(target), s.1.max(target))
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        extend_zero_capacity_context, project_closed_curve, project_context_curve,
        project_side_curve,
    };

    fn closed_mixed_capacity_samples() -> Vec<(f32, f32, f32, f32, bool)> {
        (0..32)
            .map(|i| {
                // Half-step sampling gives both signs but no zero-capacity cut.
                let angle = (i as f32 + 0.5) * std::f32::consts::TAU / 32.0;
                let v = 50.0 + 40.0 * angle.sin();
                let u = 100.0 + 30.0 * angle.cos();
                let raw = -6.0 * angle.cos();
                // The low-capacity first point makes the last-to-first constraint
                // observable. Ignoring that edge would permit a sharp seam.
                let desired = if i == 0 { raw * 0.025 } else { raw };
                (v, u, u + desired, u + raw, false)
            })
            .collect()
    }

    fn assert_closed_bounds(samples: &[(f32, f32, f32, f32, bool)], output: &[f32]) {
        assert_context_bounds(samples, output);
        let last = samples.len() - 1;
        let chord = (samples[0].0 - samples[last].0).hypot(samples[0].1 - samples[last].1);
        let jump = (output[0] - samples[0].1) - (output[last] - samples[last].1);
        assert!(
            jump.abs() <= 0.25 * chord + 1e-4,
            "closing edge slope violation: {jump}/{chord}"
        );
    }

    #[test]
    fn closed_curve_handles_both_capacity_signs_without_a_zero_cut() {
        let samples = closed_mixed_capacity_samples();
        let capacity: Vec<_> = samples.iter().map(|s| s.2 - s.1).collect();
        assert!(capacity.iter().all(|&d| d != 0.0));
        assert!(extend_zero_capacity_context(&[4, 5, 6], &capacity).is_err());
        let output = project_closed_curve(&samples, 5.0).unwrap();
        assert_closed_bounds(&samples, &output);
        let motion: Vec<_> = output.iter().zip(&samples).map(|(q, s)| q - s.1).collect();
        assert!(
            motion.iter().any(|&d| d > 2.0) && motion.iter().any(|&d| d < -2.0),
            "joint projection erased a sign of the requested edit"
        );
        assert!(
            motion[31].abs() < capacity[31].abs() - 1.0,
            "closing low-capacity point did not constrain its actual neighbour"
        );
    }

    #[test]
    fn closed_curve_is_independent_of_cyclic_start_index_and_traversal_direction() {
        let samples = closed_mixed_capacity_samples();
        let output = project_closed_curve(&samples, 5.0).unwrap();
        let rotated: Vec<_> = (0..samples.len())
            .map(|i| samples[(i + 7) % samples.len()])
            .collect();
        let rotated_output = project_closed_curve(&rotated, 5.0).unwrap();
        for i in 0..samples.len() {
            assert!((rotated_output[i] - output[(i + 7) % samples.len()]).abs() < 0.001);
        }
        let reversed: Vec<_> = samples.iter().copied().rev().collect();
        let reversed_output = project_closed_curve(&reversed, 5.0).unwrap();
        for i in 0..samples.len() {
            assert!((reversed_output[i] - output[samples.len() - 1 - i]).abs() < 0.001);
        }
    }

    #[test]
    fn closed_curve_preserves_protection_zero_sign_and_physical_scale() {
        let mut samples = closed_mixed_capacity_samples();
        samples[0].4 = true;
        samples[16].4 = true;
        let output = project_closed_curve(&samples, 5.0).unwrap();
        assert_closed_bounds(&samples, &output);
        for sign in [-1.0, 1.0] {
            let scale = 2.0;
            let transformed: Vec<_> = samples
                .iter()
                .map(|s| {
                    let u = scale * s.1 - 30.0;
                    (
                        scale * s.0 + 70.0,
                        u,
                        u + sign * scale * (s.2 - s.1),
                        u + sign * scale * (s.3 - s.1),
                        s.4,
                    )
                })
                .collect();
            let result = project_closed_curve(&transformed, scale * 5.0).unwrap();
            assert_closed_bounds(&transformed, &result);
            for i in 0..samples.len() {
                let expected = transformed[i].1 + sign * scale * (output[i] - samples[i].1);
                assert!((result[i] - expected).abs() < 0.001);
            }
        }
        let zero: Vec<_> = samples.iter().map(|s| (s.0, s.1, s.1, s.3, s.4)).collect();
        assert_eq!(
            project_closed_curve(&zero, 5.0).unwrap(),
            samples.iter().map(|s| s.1).collect::<Vec<_>>()
        );
    }

    // Local source geometry 16..35 from the cached user -30 ear replay. The
    // tuples retain desired and raw displacements separately; no photo/mask
    // data is embedded. Original core 27..55 has physical bending length 7.8651.
    fn recorded_user_context() -> Vec<(f32, f32, f32, f32, bool)> {
        [
            (-70.0459728, -120.9205192, 0.0, 0.0, false),
            (-63.0215207, -120.6332847, 0.0, 0.0, false),
            (-56.0774797, -119.5619100, -0.1275057, -0.4256091, false),
            (-49.1409709, -118.4171012, -0.6114661, -0.8910449, false),
            (-42.2044776, -117.2722621, -1.3115880, -1.3565415, false),
            (-35.2679689, -116.1274533, -1.8219770, -1.8219770, false),
            (-28.3314601, -114.9826445, -2.2874128, -2.2874128, false),
            (-21.3949666, -113.8378359, -2.7528790, -2.7528790, false),
            (-14.4438961, -112.8081902, -3.1950900, -3.1950900, false),
            (-7.4407424, -112.1905341, -3.5541386, -3.5541386, false),
            (-0.4375890, -111.5728474, -3.9131873, -3.9131873, false),
            (6.5655647, -110.9551912, -4.2722360, -4.2722360, false),
            (13.5338231, -110.0854603, -4.4756407, -4.5000860, false),
            (20.4415283, -108.7781348, -3.4873949, -4.5000555, false),
            (27.3492638, -107.4707785, -1.9934598, -4.5000249, false),
            (34.1331132, -105.6272208, -0.7750751, -4.5000860, false),
            (40.8337909, -103.5490868, -0.0562148, -4.5000858, false),
            (47.2380501, -100.6488452, 0.0, -4.5000860, true),
            (53.7758884, -98.1219505, 0.0, -4.5001165, true),
            (60.5661522, -96.3003808, 0.0, -4.5000860, true),
        ]
        .into_iter()
        .map(|(v, u, d, raw, fixed)| (v, u, u + d, u + raw, fixed))
        .collect()
    }

    fn assert_context_bounds(samples: &[(f32, f32, f32, f32, bool)], output: &[f32]) {
        for (s, &value) in samples.iter().zip(output) {
            let target = s.2.clamp(s.1.min(s.3), s.1.max(s.3));
            assert!(value >= s.1.min(target) - 1e-4 && value <= s.1.max(target) + 1e-4);
            if s.4 || target == s.1 {
                assert_eq!(value, s.1);
            }
        }
        for i in 1..samples.len() {
            let chord = (samples[i].0 - samples[i - 1].0).hypot(samples[i].1 - samples[i - 1].1);
            let slope_numerator = (output[i] - samples[i].1) - (output[i - 1] - samples[i - 1].1);
            assert!(slope_numerator.abs() <= 0.25 * chord + 1e-4);
        }
    }

    #[test]
    fn observed_context_reduces_the_recorded_artificial_boundary_bend_within_capacity() {
        let samples = recorded_user_context();
        let output = project_context_curve(&samples, 7.865106).unwrap();
        assert_context_bounds(&samples, &output);
        let turn = |ordinates: &[f32], i: usize| {
            let a = (
                ordinates[i] - ordinates[i - 1],
                samples[i].0 - samples[i - 1].0,
            );
            let b = (
                ordinates[i + 1] - ordinates[i],
                samples[i + 1].0 - samples[i].0,
            );
            (a.0 * b.1 - a.1 * b.0)
                .atan2(a.0 * b.0 + a.1 * b.1)
                .to_degrees()
        };
        let source: Vec<_> = samples.iter().map(|s| s.1).collect();
        let desired: Vec<_> = samples.iter().map(|s| s.2).collect();
        let before = (turn(&desired, 12) - turn(&source, 12)).abs();
        let after = (turn(&output, 12) - turn(&source, 12)).abs();
        assert!(
            before > 8.0 && after + 3.0 < before,
            "boundary bend {before} -> {after}"
        );
        assert!(samples[12].1 - output[12] > 3.0, "context erased the edit");
        assert!(
            output[12] - samples[12].2 > 0.3,
            "bad input peak stayed artificially locked"
        );
    }

    #[test]
    fn recorded_strong_request_is_feasible_without_nonzero_endpoint_locks() {
        // Woman source controls 95..88, using twice the recorded -30 motion.
        // The unchanged-mask raw +/-60 trace agrees to 0.000092px. This is a
        // geometry feasibility regression, not a photo or naturalness approval.
        let rows = [
            (12.6680710, -93.6798342, -6.4514837, -6.4514837, false),
            (17.8847281, -93.6415906, -6.5914081, -6.5914081, false),
            (23.1014461, -93.6033428, -6.5429561, -6.7314018, false),
            (28.3097242, -93.3186545, -5.0728230, -6.7476134, false),
            (33.4908545, -92.8398296, -2.4238637, -6.7475525, false),
            (38.5724309, -91.7059197, -0.8710771, -6.7475525, false),
            (43.7523122, -91.0857553, -0.0397678, -6.7476176, false),
            (48.6908146, -89.4650593, 0.0, -6.7475567, true),
        ];
        for sign in [-1.0, 1.0] {
            let samples: Vec<_> = rows
                .iter()
                .map(|&(v, u, d, raw, fixed)| (v, u, u + sign * d, u + sign * raw, fixed))
                .collect();
            assert!(project_side_curve(&samples)
                .unwrap_err()
                .contains("motion bounds conflict"));
            let output = project_context_curve(&samples, 6.235901).unwrap();
            assert_context_bounds(&samples, &output);
            let first_motion = (output[1] - samples[1].1).abs();
            assert!(
                first_motion > 1.0 && first_motion < 6.34,
                "low downstream capacity requires upstream reduction: {first_motion}"
            );
        }
    }

    #[test]
    fn context_projection_preserves_sign_scale_and_exact_zero_requests() {
        let samples = recorded_user_context();
        let original = project_context_curve(&samples, 7.865106).unwrap();
        for sign in [-1.0, 1.0] {
            for scale in [0.5, 3.0] {
                let transformed: Vec<_> = samples
                    .iter()
                    .map(|s| {
                        let u = scale * s.1 + 200.0;
                        (
                            scale * s.0 - 40.0,
                            u,
                            u + sign * scale * (s.2 - s.1),
                            u + sign * scale * (s.3 - s.1),
                            s.4,
                        )
                    })
                    .collect();
                let output = project_context_curve(&transformed, scale * 7.865106).unwrap();
                assert_context_bounds(&transformed, &output);
                for i in 0..output.len() {
                    let expected = transformed[i].1 + sign * scale * (original[i] - samples[i].1);
                    assert!((output[i] - expected).abs() < 0.001);
                }
            }
        }
        let zero: Vec<_> = samples.iter().map(|s| (s.0, s.1, s.1, s.3, s.4)).collect();
        assert_eq!(
            project_context_curve(&zero, 7.865106).unwrap(),
            samples.iter().map(|s| s.1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn context_expansion_keeps_cyclic_source_order_in_both_directions() {
        let capacity = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let forward = extend_zero_capacity_context(&[7, 0, 1], &capacity).unwrap();
        assert_eq!(forward, vec![5, 6, 7, 0, 1, 2, 3]);
        let reverse = extend_zero_capacity_context(&[1, 0, 7], &capacity).unwrap();
        assert_eq!(reverse, forward.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn context_expansion_rejects_noncontiguous_or_unbounded_open_arcs() {
        assert!(extend_zero_capacity_context(&[1, 3], &[0.0; 8])
            .unwrap_err()
            .contains("not contiguous"));
        assert!(extend_zero_capacity_context(&[2, 3], &[1.0; 8])
            .unwrap_err()
            .contains("periodic projection required"));
        let mut only_one_zero = [1.0; 8];
        only_one_zero[6] = 0.0;
        assert!(extend_zero_capacity_context(&[2, 3], &only_one_zero)
            .unwrap_err()
            .contains("periodic projection required"));
    }

    fn descending_peak(sign: f32) -> Vec<(f32, f32, f32, f32, bool)> {
        (0..=20)
            .map(|i| {
                let y = i as f32 * 5.0;
                let source = 80.0 - 0.3 * y;
                let displacement = if i == 0 || i == 20 {
                    0.0
                } else {
                    sign * 4.0 * (std::f32::consts::PI * y / 100.0).sin().powi(2)
                };
                (
                    y,
                    source,
                    source + displacement,
                    source + displacement,
                    i == 0,
                )
            })
            .collect()
    }

    #[test]
    fn descending_source_with_fixed_first_point_keeps_both_signed_cheek_peaks() {
        for sign in [-1.0, 1.0] {
            let samples = descending_peak(sign);
            let output = project_side_curve(&samples).unwrap();
            assert_eq!(output[0], samples[0].1);
            let peak = sign * (output[10] - samples[10].1);
            assert!(
                peak > 3.0,
                "fixed endpoint erased a feasible cheek peak: {peak}"
            );
            for i in 0..samples.len() - 1 {
                let a = output[i] - samples[i].1;
                let b = output[i + 1] - samples[i + 1].1;
                let chord =
                    (samples[i + 1].0 - samples[i].0).hypot(samples[i + 1].1 - samples[i].1);
                assert!((b - a).abs() <= 0.25 * chord + 1e-4);
                assert!(sign * a >= -1e-4);
                assert!(a.abs() <= (samples[i].3 - samples[i].1).abs() + 1e-4);
            }
        }
    }

    #[test]
    fn fixed_guard_does_not_erase_the_recorded_midface_expansion_request() {
        // Minimized source coordinates 70 -> 64 from the shared-cheek guard
        // regression. The upper guard is fixed; control 64 has an independently
        // requested -1.4976px displacement. These are geometry, not photo data.
        let source = [
            (100.572998, 42.602562),
            (103.804512, 42.324604),
            (107.035995, 42.046642),
            (110.273605, 41.875679),
            (113.515724, 41.783787),
            (116.757858, 41.691891),
            (119.999977, 41.599998),
        ];
        let mut samples: Vec<_> = source
            .iter()
            .enumerate()
            .map(|(i, &(y, x))| (y, x, x - 1.4976006, x - 1.4976006, i == 0))
            .collect();
        // Under endpoint-tangent preservation, the original minimized
        // fixture is infeasible: it locks a 1.4976px jump over only 3.24px.
        // Retain that evidence instead of silently relaxing the endpoint.
        assert!(project_side_curve(&samples)
            .unwrap_err()
            .contains("motion bounds conflict"));
        // A caller-provided reachable upper transition makes the same midface
        // request feasible. This .5px taper is a synthetic contract example,
        // not a claim that the recorded graph supplied that value.
        samples[1].2 = samples[1].1 - 0.5;
        let output = project_side_curve(&samples).unwrap();
        assert_eq!(output[0], source[0].1);
        assert!(
            source[6].1 - output[6] > 1.4,
            "guard erased the clear midface request: {output:?}"
        );
    }

    #[test]
    fn reversing_requested_displacements_reverses_only_the_edit() {
        let positive = descending_peak(1.0);
        let negative = descending_peak(-1.0);
        let a = project_side_curve(&positive).unwrap();
        let b = project_side_curve(&negative).unwrap();
        for i in 0..a.len() {
            assert!((a[i] + b[i] - 2.0 * positive[i].1).abs() < 1e-4);
        }
    }

    #[test]
    fn fairing_cannot_spend_motion_inside_a_low_reachability_interval() {
        for sign in [-1.0, 1.0] {
            let mut samples = descending_peak(sign);
            for (i, sample) in samples.iter_mut().enumerate() {
                if (8..=12).contains(&i) {
                    sample.2 = sample.1 + sign * 0.15;
                }
            }
            let output = project_side_curve(&samples).unwrap();
            for (i, (&actual, sample)) in output.iter().zip(&samples).enumerate() {
                let motion = sign * (actual - sample.1);
                let capacity = sign * (sample.2 - sample.1);
                assert!(
                    motion >= -1e-5 && motion <= capacity + 1e-5,
                    "local reachability exceeded at {i}: {motion} > {capacity}"
                );
            }
            assert!(
                sign * (output[5] - samples[5].1) > 0.5,
                "capacity bound erased a separate reachable cheek region"
            );
        }
    }

    #[test]
    fn subcurve_endpoint_values_and_tangents_keep_the_callers_desired_motion() {
        for sign in [-1.0, 1.0] {
            let mut samples = descending_peak(sign);
            for (i, d) in [(0, 0.6), (1, 1.0), (19, 0.5), (20, 0.25)] {
                samples[i].2 = samples[i].1 + sign * d;
                samples[i].3 = samples[i].1 + sign * 4.0;
            }
            let output = project_side_curve(&samples).unwrap();
            // Explicit protection wins over the nonzero desired endpoint.
            assert_eq!(output[0], samples[0].1);
            for i in [1, 19, 20] {
                assert_eq!(output[i], samples[i].2, "subcurve attachment moved at {i}");
            }
        }
    }

    #[test]
    fn zero_and_hard_fixed_requests_are_exact_and_steep_edits_respect_the_slope_bound() {
        let zero = [
            (0.0, 30.0, 30.0, 30.0, false),
            (2.0, 8.0, 8.0, 8.0, false),
            (2.0, 12.0, 12.0, 12.0, false),
            (8.0, 1.0, 1.0, 1.0, false),
        ];
        assert_eq!(
            project_side_curve(&zero).unwrap(),
            vec![30.0, 8.0, 12.0, 1.0]
        );
        assert_eq!(
            project_side_curve(&[(4.0, 12.0, 16.0, 16.0, true)]).unwrap(),
            vec![12.0]
        );

        let mut samples = descending_peak(1.0);
        samples[10].4 = true;
        let output = project_side_curve(&samples).unwrap();
        assert_eq!(output[0], samples[0].1);
        assert_eq!(output[10], samples[10].1);
        assert!(
            output[5] - samples[5].1 > 0.5,
            "fixed middle erased a separate feasible edit"
        );

        for sign in [-1.0, 1.0] {
            let abrupt = [
                (0.0, 0.0, 0.0, 0.0, true),
                (1.0, 0.0, sign * 10.0, sign * 10.0, false),
            ];
            assert!(project_side_curve(&abrupt)
                .unwrap_err()
                .contains("motion bounds conflict"));

            // The jump is adjustable only in the interior; the first/last
            // two source-relative displacements must retain their zero joins.
            let interior: Vec<_> = (0..7)
                .map(|i| {
                    let d = if (2..=4).contains(&i) {
                        sign * 10.0
                    } else {
                        0.0
                    };
                    (i as f32, 0.0, d, d, false)
                })
                .collect();
            let output = project_side_curve(&interior).unwrap();
            for i in [0, 1, 5, 6] {
                assert_eq!(output[i], 0.0);
            }
            assert!(
                (sign * output[3] - 0.5).abs() < 1e-4,
                "slope constraint inactive: {output:?}"
            );
            for edge in output.windows(2) {
                assert!((edge[1] - edge[0]).abs() <= 0.25001);
            }
        }
    }

    #[test]
    fn uniform_scale_and_translation_commute_with_projection() {
        let source = descending_peak(-1.0);
        let baseline = project_side_curve(&source).unwrap();
        for scale in [0.25, 3.5] {
            let transformed: Vec<_> = source
                .iter()
                .map(|s| {
                    (
                        scale * s.0 - 35.0,
                        scale * s.1 + 140.0,
                        scale * s.2 + 140.0,
                        scale * s.3 + 140.0,
                        s.4,
                    )
                })
                .collect();
            let output = project_side_curve(&transformed).unwrap();
            for (actual, expected) in output.iter().zip(&baseline) {
                assert!(((*actual - 140.0) / scale - expected).abs() < 0.001);
            }
        }
    }

    #[test]
    fn denser_sampling_on_only_one_part_does_not_move_the_cheek_peak() {
        let coarse = descending_peak(1.0);
        let reference = project_side_curve(&coarse).unwrap();
        let mut coordinates: Vec<_> = coarse.iter().map(|s| s.0).collect();
        coordinates.extend((1..80).map(|i| 10.0 + i as f32 * 0.5));
        coordinates.sort_by(f32::total_cmp);
        coordinates.dedup();
        let refined: Vec<_> = coordinates
            .iter()
            .map(|&y| {
                let source = 80.0 - 0.3 * y;
                let d = if y == 0.0 || y == 100.0 {
                    0.0
                } else {
                    4.0 * (std::f32::consts::PI * y / 100.0).sin().powi(2)
                };
                (y, source, source + d, source + d, y == 0.0)
            })
            .collect();
        let output = project_side_curve(&refined).unwrap();
        for (s, expected) in coarse.iter().zip(reference) {
            let index = coordinates.binary_search_by(|y| y.total_cmp(&s.0)).unwrap();
            assert!(
                (output[index] - expected).abs() < 0.1,
                "sampling density changed the edit at y={}: {} -> {}",
                s.0,
                expected,
                output[index]
            );
        }
    }

    #[test]
    fn area_moment_fairing_preserves_strength_and_fixed_attachments() {
        let samples: Vec<_> = (0..=20)
            .map(|i| {
                let y = i as f32 * 5.0;
                let source = 80.0 - 0.3 * y + 0.7 * (y * 0.11).sin();
                let desired = 7.0 * (std::f32::consts::PI * y / 100.0).sin().powi(2);
                let fixed = i == 0 || i == 20;
                (y, source, source + desired, source + desired * 1.25, fixed)
            })
            .collect();
        let output = super::project_preserving_area_moments(&samples, 8.0).unwrap();
        assert_eq!(output[0], samples[0].1);
        assert_eq!(output[20], samples[20].1);
        for (sample, &value) in samples.iter().zip(&output) {
            let displacement = value - sample.1;
            assert!(displacement >= -1e-4);
            assert!(displacement <= (sample.3 - sample.1) + 1e-3);
        }
        let weights = |i: usize| {
            if i == 0 || i == samples.len() - 1 {
                2.5
            } else {
                5.0
            }
        };
        let requested_area: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, s)| weights(i) * (s.2 - s.1) as f64)
            .sum();
        let actual_area: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, s)| weights(i) * (output[i] - s.1) as f64)
            .sum();
        let requested_moment: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, s)| weights(i) * (i as f64 / 20.0) * (s.2 - s.1) as f64)
            .sum();
        let actual_moment: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, s)| weights(i) * (i as f64 / 20.0) * (output[i] - s.1) as f64)
            .sum();
        assert!(
            (actual_area - requested_area).abs() < 0.02,
            "area {requested_area} -> {actual_area}"
        );
        assert!(
            (actual_moment - requested_moment).abs() < 0.02,
            "moment {requested_moment} -> {actual_moment}"
        );
    }

    #[test]
    fn area_moment_fairing_is_signed_and_near_zero_continuous() {
        let build = |amount: f32| {
            (0..=20)
                .map(|i| {
                    let y = i as f32 * 5.0;
                    let source = 80.0 - 0.3 * y;
                    let desired = amount * (std::f32::consts::PI * y / 100.0).sin().powi(2);
                    (
                        y,
                        source,
                        source + desired,
                        source + desired * 1.25,
                        i == 0 || i == 20,
                    )
                })
                .collect::<Vec<_>>()
        };
        let zero = build(0.0);
        let positive = super::project_preserving_area_moments(&build(0.1), 8.0).unwrap();
        let negative = super::project_preserving_area_moments(&build(-0.1), 8.0).unwrap();
        let identity = super::project_preserving_area_moments(&zero, 8.0).unwrap();
        for (sample, &value) in zero.iter().zip(&identity) {
            assert_eq!(value, sample.1);
        }
        for i in 0..positive.len() {
            let source = zero[i].1;
            assert!((positive[i] - source) >= -1e-4);
            assert!((negative[i] - source) <= 1e-4);
            assert!(((positive[i] - source) + (negative[i] - source)).abs() < 0.002);
        }
    }
}
