//! Rigid moving least squares, Schaefer, McPhail and Warren (SIGGRAPH 2006).
//! This is the weighted-centroid / normalized rigid-map equation, expressed
//! as its equivalent 2D covariance rotation; it is not inverse-distance IDW.
//! Reference comparison: Jarvis73/Moving-Least-Squares `mls_rigid_deformation`.
//! The caller chooses portrait handles and applies visibility/protection.

type Point = (f32, f32);

pub(super) struct RigidMls {
    controls: Vec<(Point, Point)>,
}

impl RigidMls {
    pub(super) fn new(controls: Vec<(Point, Point)>) -> Self {
        Self { controls }
    }

    pub(super) fn map(&self, v: Point) -> Point {
        if self.controls.is_empty() {
            return v;
        }
        let mut weight_sum = 0.0;
        let (mut pstar, mut qstar) = ((0.0, 0.0), (0.0, 0.0));
        for &(p, q) in &self.controls {
            let distance = (p.0 - v.0).powi(2) + (p.1 - v.1).powi(2);
            if distance < 1e-8 {
                return q;
            }
            // alpha=1 in the paper: w_i = |p_i-v|^-2.
            let weight = distance.recip();
            weight_sum += weight;
            pstar.0 += p.0 * weight;
            pstar.1 += p.1 * weight;
            qstar.0 += q.0 * weight;
            qstar.1 += q.1 * weight;
        }
        pstar.0 /= weight_sum;
        pstar.1 /= weight_sum;
        qstar.0 /= weight_sum;
        qstar.1 /= weight_sum;
        let (mut a, mut b) = (0.0, 0.0);
        for &(p, q) in &self.controls {
            let weight = ((p.0 - v.0).powi(2) + (p.1 - v.1).powi(2)).recip();
            let (px, py) = (p.0 - pstar.0, p.1 - pstar.1);
            let (qx, qy) = (q.0 - qstar.0, q.1 - qstar.1);
            a += weight * (px * qx + py * qy);
            b += weight * (px * qy - py * qx);
        }
        let norm = a.hypot(b);
        let (vx, vy) = (v.0 - pstar.0, v.1 - pstar.1);
        if norm < 1e-12 {
            // A single/degenerate handle defines translation, not NaN.
            return (qstar.0 + vx, qstar.1 + vy);
        }
        (
            qstar.0 + (a * vx - b * vy) / norm,
            qstar.1 + (b * vx + a * vy) / norm,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(actual: Point, expected: Point) {
        assert!(
            (actual.0 - expected.0).abs() < 2e-4,
            "{actual:?} != {expected:?}"
        );
        assert!(
            (actual.1 - expected.1).abs() < 2e-4,
            "{actual:?} != {expected:?}"
        );
    }
    #[test]
    fn reproduces_global_rigid_transform_at_unseen_points() {
        let transform = |p: Point| (18.0 - p.1, 7.0 + p.0);
        let handles = [(0.0, 0.0), (20.0, 0.0), (20.0, 16.0), (0.0, 16.0)];
        let map = RigidMls::new(handles.iter().map(|&p| (p, transform(p))).collect());
        for p in [(7.3, 4.2), (-2.0, 25.0), (12.0, 8.0)] {
            near(map.map(p), transform(p));
        }
    }
    #[test]
    fn exact_handles_and_degenerate_translation_are_finite() {
        let controls = vec![((0.0, 0.0), (3.0, 4.0)), ((12.0, 8.0), (15.0, 12.0))];
        let map = RigidMls::new(controls.clone());
        for &(p, q) in &controls {
            near(map.map(p), q);
        }
        near(
            RigidMls::new(vec![controls[0]]).map((20.0, 30.0)),
            (23.0, 34.0),
        );
        near(map.map((4.5, 7.8)), (7.5, 11.8));
    }
    #[test]
    fn agrees_with_independent_paper_matrix_equation() {
        // Independently evaluate the 2×2 A_i equation used in the NumPy
        // reference, without its integer output quantization/image clipping.
        let controls = [
            ((0.0, 0.0), (2.0, -1.0)),
            ((15.0, 1.0), (13.0, 3.0)),
            ((12.0, 18.0), (14.0, 20.0)),
            ((-2.0, 14.0), (-1.0, 12.0)),
        ];
        let v: Point = (7.3, 4.2);
        let weights: Vec<f64> = controls
            .iter()
            .map(|(p, _)| {
                1.0 / ((p.0 as f64 - v.0 as f64).powi(2) + (p.1 as f64 - v.1 as f64).powi(2))
            })
            .collect();
        let sum: f64 = weights.iter().sum();
        let centroid = |to: bool| {
            let mut c = [0.0, 0.0];
            for ((p, q), w) in controls.iter().zip(&weights) {
                let h = if to { q } else { p };
                c[0] += h.0 as f64 * w / sum;
                c[1] += h.1 as f64 * w / sum;
            }
            c
        };
        let ps = centroid(false);
        let qs = centroid(true);
        let dv = [v.0 as f64 - ps[0], v.1 as f64 - ps[1]];
        let mut t = [0.0, 0.0];
        for ((p, q), w) in controls.iter().zip(&weights) {
            let ph = [p.0 as f64 - ps[0], p.1 as f64 - ps[1]];
            let qh = [q.0 as f64 - qs[0], q.1 as f64 - qs[1]];
            let aa = [
                [ph[0] * dv[0] + ph[1] * dv[1], ph[0] * dv[1] - ph[1] * dv[0]],
                [ph[1] * dv[0] - ph[0] * dv[1], ph[1] * dv[1] + ph[0] * dv[0]],
            ];
            for (j, out) in t.iter_mut().enumerate() {
                *out += (qh[0] * aa[0][j] + qh[1] * aa[1][j]) * w / sum;
            }
        }
        let normalization = dv[0].hypot(dv[1]) / t[0].hypot(t[1]);
        let expected = (
            (qs[0] + t[0] * normalization) as f32,
            (qs[1] + t[1] * normalization) as f32,
        );
        near(RigidMls::new(controls.to_vec()).map(v), expected);
    }
}
