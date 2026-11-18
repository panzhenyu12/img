//! Stationary-velocity transport for connected portrait layers.
//! The historical compact RBF remains available only to regression replays.
//! RK4 integrates the same bilinear interior grid in opposite directions;
//! continuity across the grid boundary additionally requires zero velocity
//! there. No final displacement blending, nearest-contour branch or alpha cap.

#[cfg(test)]
use super::{add, scale};
use super::{dot, sub, Point};
#[cfg(test)]
use nalgebra::{DMatrix, DVector};

/// Wendland (1995) compact positive definite C2 basis in two dimensions.
/// Bilinear samples are Lipschitz inside the grid. A zero exterior is continuous
/// only where compact support also vanishes at the boundary. Raster constraints
/// therefore do not imply a globally C2 flow.
#[cfg(test)]
pub(super) struct CompactRbf {
    controls: Vec<Point>,
    radius: f32,
    solver: nalgebra::linalg::Cholesky<f32, nalgebra::Dyn>,
    observation_weights: Vec<f32>,
}
#[cfg(test)]
impl CompactRbf {
    pub(super) fn new(controls: Vec<Point>, radius: f32) -> Self {
        let weights = vec![1.0; controls.len()];
        Self::with_observation_weights(controls, radius, weights)
    }
    fn with_observation_weights(controls: Vec<Point>, radius: f32, weights: Vec<f32>) -> Self {
        let n = controls.len();
        let mut matrix = DMatrix::from_fn(n, n, |i, j| {
            weights[i] * Self::basis(sub(controls[i], controls[j]), radius) * weights[j]
        });
        for i in 0..n {
            matrix[(i, i)] += 0.002;
        }
        let solver = matrix
            .cholesky()
            .expect("regularized positive-definite Wendland matrix");
        Self {
            controls,
            radius,
            solver,
            observation_weights: weights,
        }
    }
    /// Solve weighted observations in the kernel system itself. This is an
    /// isolated experiment until actual targets and source textures pass.
    #[cfg(test)]
    pub(super) fn new_supported(controls: Vec<Point>, radius: f32, weights: Vec<f32>) -> Self {
        assert_eq!(controls.len(), weights.len());
        assert!(weights
            .iter()
            .all(|w| w.is_finite() && (0.0..=1.0).contains(w)));
        Self::with_observation_weights(controls, radius, weights)
    }
    fn basis(delta: Point, radius: f32) -> f32 {
        let r = dot(delta, delta).sqrt() / radius;
        if r >= 1.0 {
            return 0.0;
        }
        let t = 1.0 - r;
        t * t * t * t * (4.0 * r + 1.0)
    }
    pub(super) fn solve(&self, values: &[Point]) -> Vec<Point> {
        let x = self.solver.solve(&DVector::from_iterator(
            values.len(),
            values.iter().map(|p| p.0),
        ));
        let y = self.solver.solve(&DVector::from_iterator(
            values.len(),
            values.iter().map(|p| p.1),
        ));
        x.iter().copied().zip(y.iter().copied()).collect()
    }
    pub(super) fn evaluate(&self, coefficients: &[Point], p: Point) -> Point {
        self.controls
            .iter()
            .zip(coefficients)
            .zip(&self.observation_weights)
            .fold((0.0, 0.0), |sum, ((&control, &coefficient), &weight)| {
                add(
                    sum,
                    scale(
                        coefficient,
                        weight * Self::basis(sub(p, control), self.radius),
                    ),
                )
            })
    }
}

pub(super) struct VelocityGrid {
    origin: Point,
    width: usize,
    height: usize,
    vectors: Vec<Point>,
    pub(super) steps: usize,
}

impl VelocityGrid {
    pub(super) fn new(origin: Point, width: usize, height: usize, vectors: Vec<Point>) -> Self {
        let mut max_gradient = 0.0_f32;
        for y in 0..height.saturating_sub(1) {
            for x in 0..width.saturating_sub(1) {
                let i = y * width + x;
                let dx = sub(vectors[i + 1], vectors[i]);
                let dy = sub(vectors[i + width], vectors[i]);
                max_gradient = max_gradient.max((dot(dx, dx) + dot(dy, dy)).sqrt());
            }
        }
        let steps = ((max_gradient * 4.0).ceil() as usize).clamp(16, 128);
        Self {
            origin,
            width,
            height,
            vectors,
            steps,
        }
    }

    /// Interpolate the same f32 grid in local f64 coordinates. Rounding each
    /// RK4 stage back to an absolute f32 image coordinate quantizes the path;
    /// on a large photograph, even a whole substep can disappear.
    fn velocity_local(&self, x: f64, y: f64) -> (f64, f64) {
        if x < 0.0 || y < 0.0 || x > (self.width - 1) as f64 || y > (self.height - 1) as f64 {
            return (0.0, 0.0);
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        let a = self.vectors[y0 * self.width + x0];
        let b = self.vectors[y0 * self.width + x1];
        let c = self.vectors[y1 * self.width + x0];
        let d = self.vectors[y1 * self.width + x1];
        let bilinear = |a: f32, b: f32, c: f32, d: f32| {
            (a as f64 * (1.0 - tx) + b as f64 * tx) * (1.0 - ty)
                + (c as f64 * (1.0 - tx) + d as f64 * tx) * ty
        };
        (bilinear(a.0, b.0, c.0, d.0), bilinear(a.1, b.1, c.1, d.1))
    }

    #[cfg(test)]
    pub(super) fn velocity(&self, p: Point) -> Point {
        let v = self.velocity_local(
            p.0 as f64 - self.origin.0 as f64,
            p.1 as f64 - self.origin.1 as f64,
        );
        (v.0 as f32, v.1 as f32)
    }

    fn displacement(&self, p: Point, direction: f32) -> (f64, f64) {
        let start = (
            p.0 as f64 - self.origin.0 as f64,
            p.1 as f64 - self.origin.1 as f64,
        );
        if self.velocity_local(start.0, start.1) == (0.0, 0.0) {
            return (0.0, 0.0);
        }
        let dt = direction as f64 / self.steps as f64;
        let mut delta = (0.0, 0.0);
        for _ in 0..self.steps {
            let x = start.0 + delta.0;
            let y = start.1 + delta.1;
            let k1 = self.velocity_local(x, y);
            let k2 = self.velocity_local(x + k1.0 * dt * 0.5, y + k1.1 * dt * 0.5);
            let k3 = self.velocity_local(x + k2.0 * dt * 0.5, y + k2.1 * dt * 0.5);
            let k4 = self.velocity_local(x + k3.0 * dt, y + k3.1 * dt);
            delta.0 += (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) * (dt / 6.0);
            delta.1 += (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) * (dt / 6.0);
        }
        delta
    }

    pub(super) fn transport(&self, p: Point, direction: f32) -> Point {
        let delta = self.displacement(p, direction);
        ((p.0 as f64 + delta.0) as f32, (p.1 as f64 + delta.1) as f32)
    }
}

/// Integrate the inverse trajectory through the same stationary grid. RK4
/// remains a numerical approximation, not a topology or material certificate.
pub(super) fn inverse(grid: &VelocityGrid, p: Point) -> Point {
    // Keep the small sampling offset: subtracting two rounded absolute f32
    // coordinates would reintroduce cancellation after accurate integration.
    let delta = grid.displacement(p, -1.0);
    (delta.0 as f32, delta.1 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wendland_rbf_preserves_control_direction_and_compact_support() {
        let controls = vec![
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (50.0, 50.0),
        ];
        let rbf = CompactRbf::new(controls.clone(), 60.0);
        let values = vec![(8.0, 0.0), (-8.0, 0.0), (-6.0, 0.0), (6.0, 0.0), (0.0, 0.0)];
        let coefficients = rbf.solve(&values);
        for (p, expected) in controls.into_iter().zip(values) {
            let got = rbf.evaluate(&coefficients, p);
            assert!(
                (got.0 - expected.0).abs() < 0.03,
                "control residual {got:?} {expected:?}"
            );
            assert_eq!(got.1, 0.0, "a width-only field cannot add vertical strain");
        }
        assert_eq!(rbf.evaluate(&coefficients, (-100.0, 50.0)), (0.0, 0.0));
    }
    #[test]
    fn affine_svf_matches_analytic_exponential_and_its_inverse() {
        let w = 121;
        let h = 121;
        let origin = (100.5, 100.5);
        let c = (160.5, 160.5);
        let k = 0.1;
        let vectors = (0..w * h)
            .map(|i| {
                (
                    (origin.0 + (i % w) as f32 - c.0) * k,
                    -(origin.1 + (i / w) as f32 - c.1) * k,
                )
            })
            .collect();
        let grid = VelocityGrid::new(origin, w, h, vectors);
        for p in [(130.5, 140.5), (180.5, 170.5), (160.5, 160.5)] {
            let q = grid.transport(p, 1.0);
            let expected = (
                c.0 + (p.0 - c.0) * libm::expf(k),
                c.1 + (p.1 - c.1) * libm::expf(-k),
            );
            let d = sub(q, expected);
            assert!(
                dot(d, d).sqrt() < 0.002,
                "analytic exponential mismatch {p:?} {q:?} {expected:?}"
            );
            let reverse = sub(grid.transport(q, -1.0), p);
            assert!(dot(reverse, reverse).sqrt() < 0.002);
        }
    }
    #[test]
    fn a_zero_velocity_protected_pixel_is_fixed_for_both_flow_directions() {
        let w = 80;
        let h = 80;
        let origin = (0.5, 0.5);
        let vectors = (0..w * h)
            .map(|i| {
                if (i % w < 20) || (i / w < 20) {
                    (0.0, 0.0)
                } else {
                    (5.0, 2.0)
                }
            })
            .collect();
        let grid = VelocityGrid::new(origin, w, h, vectors);
        for p in [(5.5, 25.5), (19.5, 35.5), (30.5, 19.5)] {
            for sign in [-1.0, 1.0] {
                assert_eq!(grid.transport(p, sign), p);
            }
        }
    }

    #[test]
    fn small_motion_on_a_large_photo_survives_rk4_substeps() {
        // The analytic flow of a constant velocity is p(t)=p(0)+t*v.
        // Each old f32 substep is below one coordinate ULP at this origin,
        // even though the total requested motion is representable.
        let origin = (4000.5, 4000.5);
        let p = (4020.5, 4020.5);
        let grid = VelocityGrid::new(origin, 64, 64, vec![(0.001, 0.0); 64 * 64]);
        for direction in [-1.0, 1.0] {
            let expected = (
                (p.0 as f64 + direction as f64 * 0.001_f32 as f64) as f32,
                p.1,
            );
            assert_eq!(grid.transport(p, direction), expected);
        }
    }

    #[test]
    fn a_remote_gradient_cannot_erase_or_amplify_constant_local_motion() {
        // A distant corner raises the global integration step count, but
        // neither trajectory enters its bilinear support. Both exact flows
        // must therefore be the same constant translation.
        for (origin, speed) in [(2000.5, 0.005), (8000.5, 0.1)] {
            let p = (origin + 20.0, origin + 20.0);
            let uniform = vec![(speed, 0.0); 64 * 64];
            let calm = VelocityGrid::new((origin, origin), 64, 64, uniform.clone());
            let mut remote = uniform;
            remote[0] = (40.0, 0.0);
            let busy = VelocityGrid::new((origin, origin), 64, 64, remote);
            assert!(busy.steps > calm.steps);
            for direction in [-1.0, 1.0] {
                let expected = ((p.0 as f64 + direction as f64 * speed as f64) as f32, p.1);
                assert_eq!(busy.transport(p, direction), expected, "origin={origin}");
                assert_eq!(calm.transport(p, direction), expected, "origin={origin}");
            }
        }
    }

    #[test]
    fn inverse_displacement_preserves_subpixel_motion_independent_of_image_origin() {
        let vectors: Vec<_> = (0..64 * 64)
            .map(|i| {
                let x = (i % 64) as f32 - 32.0;
                let y = (i / 64) as f32 - 32.0;
                (-y * 0.003, x * 0.003)
            })
            .collect();
        let near = VelocityGrid::new((0.5, 0.5), 64, 64, vectors.clone());
        let far = VelocityGrid::new((8000.5, 8000.5), 64, 64, vectors);
        // Translating a grid and its query leaves the underlying field and
        // its displacement invariant, including nonconstant stage samples.
        assert_eq!(
            inverse(&near, (22.5, 23.5)),
            inverse(&far, (8022.5, 8023.5))
        );
        let constant = VelocityGrid::new((8000.5, 8000.5), 64, 64, vec![(0.00001, 0.0); 64 * 64]);
        let delta = inverse(&constant, (8022.5, 8023.5));
        assert!(
            (delta.0 + 0.00001).abs() < 1e-10,
            "lost inverse displacement {delta:?}"
        );
        assert_eq!(delta.1, 0.0);
    }
}

/// Project a source side curve onto targets that add no lateral reversals or
/// concave lobes. Existing source silhouette noise remains permitted; the
/// unedited source is always feasible. Clarabel solves the convex QP.
/// Samples: (anatomical vertical coordinate, oriented source lateral,
/// desired lateral, requested lateral limit, exact attachment flag).
pub(super) fn project_side_curve(
    samples: &[(f32, f32, f32, f32, bool)],
) -> Result<Vec<f32>, String> {
    use clarabel::{algebra::CscMatrix, solver::*};
    let n = samples.len();
    if n < 3 {
        return Ok(samples.iter().map(|s| s.2).collect());
    }
    let mut p = vec![vec![0.0_f64; n]; n];
    let mut q = vec![0.0_f64; n];
    for i in 0..n {
        p[i][i] = 2.0;
        q[i] = -2.0 * samples[i].2 as f64;
    }
    // Bending regularization acts on displacement, not on original anatomy.
    for i in 0..n - 2 {
        let indices = [i, i + 1, i + 2];
        let coefficients = [1.0, -2.0, 1.0];
        let old = (samples[i].1 - 2.0 * samples[i + 1].1 + samples[i + 2].1) as f64;
        for (&a, &ca) in indices.iter().zip(&coefficients) {
            q[a] -= 2.0 * 3.0 * old * ca;
            for (&b, &cb) in indices.iter().zip(&coefficients) {
                p[a][b] += 2.0 * 3.0 * ca * cb;
            }
        }
    }
    let mut rows = Vec::<Vec<f64>>::new();
    let mut bounds = Vec::<f64>::new();
    let mut equalities = 0;
    for i in 0..n {
        if samples[i].4 {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            rows.push(row);
            bounds.push(samples[i].1 as f64);
            equalities += 1;
        }
    }
    for i in 0..n {
        let lo = samples[i].1.min(samples[i].3) as f64;
        let hi = samples[i].1.max(samples[i].3) as f64;
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
        let old = samples[i + 1].1 - samples[i].1;
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        row[i + 1] = -1.0;
        rows.push(row);
        bounds.push(-old.min(0.0) as f64);
    }
    for i in 0..n - 2 {
        let h0 = (samples[i + 1].0 - samples[i].0).max(0.1) as f64;
        let h1 = (samples[i + 2].0 - samples[i + 1].0).max(0.1) as f64;
        let old = (samples[i + 2].1 - samples[i + 1].1) as f64 / h1
            - (samples[i + 1].1 - samples[i].1) as f64 / h0;
        let mut row = vec![0.0; n];
        row[i] = -1.0 / h0;
        row[i + 1] = 1.0 / h0 + 1.0 / h1;
        row[i + 2] = -1.0 / h1;
        rows.push(row);
        bounds.push(-old.min(0.0));
    }
    let inequalities = rows.len() - equalities;
    let cones = vec![ZeroConeT(equalities), NonnegativeConeT(inequalities)];
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
        return Err(format!("side curve QP status {:?}", solver.solution.status));
    }
    Ok(solver.solution.x.iter().map(|&v| v as f32).collect())
}

#[cfg(test)]
mod curve_tests {
    use super::*;

    #[test]
    fn fixed_attachments_cannot_create_a_new_expansion_lobe() {
        let samples = [
            (0.0, 20.0, 20.0, 20.0, true),
            (10.0, 20.5, 20.5, 20.5, true),
            (20.0, 22.0, 14.0, 14.0, false),
            (30.0, 25.0, 17.0, 17.0, false),
            (40.0, 30.0, 22.0, 22.0, false),
            (50.0, 38.0, 30.0, 30.0, false),
        ];
        let q = project_side_curve(&samples).unwrap();
        assert!((q[0] - 20.0).abs() < 1e-4 && (q[1] - 20.5).abs() < 1e-4);
        for i in 0..q.len() - 1 {
            assert!(q[i + 1] >= q[i] - 1e-4, "new side reversal {q:?}");
        }
        for i in 0..q.len() - 2 {
            let first = (q[i + 1] - q[i]) / (samples[i + 1].0 - samples[i].0);
            let next = (q[i + 2] - q[i + 1]) / (samples[i + 2].0 - samples[i + 1].0);
            assert!(next - first >= -1e-5, "new concave lobe {q:?}");
        }
        assert!(
            samples[5].1 - q[5] > 7.9,
            "clear lower jaw must retain meaningful requested expansion"
        );
    }
    #[test]
    fn source_pixel_irregularity_is_feasible_without_unsolicited_reshaping() {
        let samples = [
            (0.0, 10.0, 10.0, 10.0, false),
            (7.0, 10.4, 10.4, 10.4, false),
            (14.0, 10.2, 10.2, 10.2, false),
            (20.0, 12.5, 12.5, 12.5, false),
            (35.0, 13.0, 13.0, 13.0, false),
        ];
        let q = project_side_curve(&samples).unwrap();
        for (got, s) in q.iter().zip(samples) {
            assert!((got - s.1).abs() < 1e-3, "source modified {q:?}");
        }
    }
}
