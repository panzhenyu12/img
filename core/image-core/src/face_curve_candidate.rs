//! Isolated curve fairing candidate, only evaluated with explicit rear-layer evidence.
//! The unconditioned candidate failed real-photo validation; it is never the default.

pub(super) fn project_side_curve(
    samples: &[(f32, f32, f32, f32, bool)],
) -> Result<Vec<f32>, String> {
    use clarabel::{algebra::CscMatrix, solver::*};
    let n = samples.len();
    if n < 3 {
        return Ok(samples.iter().map(|s| s.2).collect());
    }
    // Work in source-relative displacement. Besides preserving the original
    // anatomical curvature, this avoids subtracting large absolute image
    // coordinates inside the quadratic objective.
    let gaps: Vec<f64> = samples
        .windows(2)
        .map(|s| (s[1].0 - s[0].0).max(0.1) as f64)
        .collect();
    let length: f64 = gaps.iter().sum();
    let mut ordinate = vec![0.0; n];
    for i in 1..n {
        ordinate[i] = ordinate[i - 1] + gaps[i - 1];
    }
    let widths: Vec<_> = (0..n)
        .map(|i| {
            (if i > 0 { gaps[i - 1] } else { 0.0 } + if i + 1 < n { gaps[i] } else { 0.0 }) * 0.5
        })
        .collect();
    let weights: Vec<_> = widths.iter().map(|w| w / length).collect();
    let source: Vec<_> = samples.iter().map(|s| s.1 as f64).collect();
    let desired: Vec<_> = samples.iter().map(|s| (s.2 - s.1) as f64).collect();
    let mut slope_rows = Vec::with_capacity(n - 2);
    let mut curvature_rows = Vec::with_capacity(n - 2);
    let mut dual_widths = Vec::with_capacity(n - 2);
    for i in 0..n - 2 {
        let dual = (gaps[i] + gaps[i + 1]) * 0.5;
        let mut slope = vec![0.0; n];
        slope[i] = 1.0 / gaps[i];
        slope[i + 1] = -1.0 / gaps[i] - 1.0 / gaps[i + 1];
        slope[i + 2] = 1.0 / gaps[i + 1];
        curvature_rows.push(slope.iter().map(|v| v / dual).collect::<Vec<_>>());
        slope_rows.push(slope);
        dual_widths.push(dual);
    }
    let mut rows = Vec::<Vec<f64>>::new();
    let mut bounds = Vec::<f64>::new();
    let mut equalities = 0;
    for i in 0..n {
        if samples[i].4 {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            rows.push(row);
            bounds.push(0.0);
            equalities += 1;
        }
    }
    for i in 0..n {
        let limit = (samples[i].3 - samples[i].1) as f64;
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        rows.push(row);
        bounds.push(limit.max(0.0));
        let mut row = vec![0.0; n];
        row[i] = -1.0;
        rows.push(row);
        bounds.push(-limit.min(0.0));
    }
    for i in 0..n - 1 {
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        row[i + 1] = -1.0;
        rows.push(row);
        bounds.push((source[i + 1] - source[i]).max(0.0));
    }
    for slope in &slope_rows {
        let old: f64 = slope.iter().zip(&source).map(|(a, b)| a * b).sum();
        rows.push(slope.iter().map(|v| -v).collect());
        bounds.push(old.max(0.0));
    }
    let solve = |p: &[Vec<f64>],
                 q: &[f64],
                 a: &[Vec<f64>],
                 b: &[f64],
                 eq: usize|
     -> Result<Vec<f64>, String> {
        let cones = vec![ZeroConeT(eq), NonnegativeConeT(a.len() - eq)];
        let settings = DefaultSettings {
            verbose: false,
            max_iter: 100,
            ..DefaultSettings::default()
        };
        let mut solver = DefaultSolver::new(
            &CscMatrix::from(p),
            q,
            &CscMatrix::from(a),
            b,
            &cones,
            settings,
        );
        solver.solve();
        if matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) {
            Ok(solver.solution.x.clone())
        } else {
            Err(format!("side curve QP status {:?}", solver.solution.status))
        }
    };
    let add_energy = |p: &mut [Vec<f64>], row: &[f64], weight: f64| {
        for (i, &a) in row.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            for (j, &b) in row.iter().enumerate() {
                p[i][j] += 2.0 * weight * a * b;
            }
        }
    };
    let mut p = vec![vec![0.0; n]; n];
    let mut q = vec![0.0; n];
    for i in 0..n {
        p[i][i] = 2.0 * weights[i];
        q[i] = -2.0 * weights[i] * desired[i];
    }
    // A physical bending length, rather than an index-based [1,-2,1]
    // penalty, keeps the projection stable when only part of a curve is
    // sampled more densely. The first pass remains a close target fit.
    for (row, &dual) in curvature_rows.iter().zip(&dual_widths) {
        add_energy(&mut p, row, (length / 25.0).powi(4) * dual / length);
    }
    let projected = solve(&p, &q, &rows, &bounds, equalities)?;
    if projected.iter().all(|d| d.abs() < 1e-7) {
        return Ok(samples.iter().map(|s| s.1).collect());
    }

    // The second pass limits concentrated displacement curvature. Preserve
    // both moved area and its vertical first moment: smoothing must not obtain
    // an easy win by reducing the requested edit or moving it along the jaw.
    // Source irregularities are never part of this penalty.
    let mut fair_rows: Vec<Vec<f64>> = rows[..equalities]
        .iter()
        .map(|row| row.iter().copied().chain([0.0]).collect())
        .collect();
    let mut fair_bounds = bounds[..equalities].to_vec();
    for moment in 0..2 {
        let row: Vec<_> = weights
            .iter()
            .zip(&ordinate)
            .map(|(&w, &y)| if moment == 0 { w } else { w * y / length })
            .collect();
        fair_bounds.push(row.iter().zip(&projected).map(|(a, b)| a * b).sum());
        fair_rows.push(row.into_iter().chain([0.0]).collect());
    }
    let fair_equalities = equalities + 2;
    fair_rows.extend(
        rows[equalities..]
            .iter()
            .map(|row| row.iter().copied().chain([0.0]).collect()),
    );
    fair_bounds.extend_from_slice(&bounds[equalities..]);
    // The desired contour already contains attachment reachability. Global
    // fairing must not spend its conserved area by increasing motion where
    // both that local target and the first feasible projection are small.
    // The first projection remains feasible under this conservative envelope.
    for i in 0..n {
        let capacity = desired[i].abs().max(projected[i].abs());
        for sign in [1.0, -1.0] {
            let mut row = vec![0.0; n + 1];
            row[i] = sign;
            fair_rows.push(row);
            fair_bounds.push(capacity);
        }
    }
    // Keep the order of the original displacement rises and falls. Without
    // this, a global fairing solve can move a cheek peak into a low-mobility
    // attachment interval even while conserving its global first moment.
    for i in 0..n - 1 {
        let delta = projected[i + 1] - projected[i];
        if delta.abs() <= 1e-6 {
            continue;
        }
        let mut row = vec![0.0; n + 1];
        row[i] = delta.signum();
        row[i + 1] = -delta.signum();
        fair_rows.push(row);
        fair_bounds.push(0.0);
    }
    let mut fair_p = vec![vec![0.0; n + 1]; n + 1];
    let mut fair_q = vec![0.0; n + 1];
    for i in 0..n {
        fair_p[i][i] = 2.0 * weights[i];
        fair_q[i] = -2.0 * weights[i] * projected[i];
    }
    fair_p[n][n] = 2.0;
    // The epigraph is normalized to displacement units over a quarter of
    // the observed interval. Its squared cost limits the worst local bend;
    // the integrated term distributes bends within that bound continuously.
    for (curvature, &dual) in curvature_rows.iter().zip(&dual_widths) {
        let scaled: Vec<_> = curvature
            .iter()
            .map(|v| v * (length / 4.0).powi(2))
            .collect();
        add_energy(&mut fair_p, &scaled, dual / length);
        for sign in [1.0, -1.0] {
            fair_rows.push(scaled.iter().map(|v| v * sign).chain([-1.0]).collect());
            fair_bounds.push(0.0);
        }
    }
    // The first projection itself is a feasible point for the second pass.
    // Retain it on numerical failure rather than failing an otherwise valid
    // edit because of nearly coincident contour ordinates.
    let displacement = solve(&fair_p, &fair_q, &fair_rows, &fair_bounds, fair_equalities)
        .ok()
        .filter(|d| d.iter().all(|v| v.is_finite()))
        .unwrap_or(projected);
    Ok(samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.4 {
                s.1
            } else {
                s.1 + displacement[i] as f32
            }
        })
        .collect())
}
