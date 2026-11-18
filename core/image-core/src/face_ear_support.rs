//! Source ear evidence and its permitted connection are separate decisions.
//! The semantic prior remains confidence, never physical opacity. Returned
//! disconnected pixels only add protection; this module cannot clear a lock.

use super::neighbors;
use std::collections::VecDeque;

pub(super) struct SourceEarSupport<'a> {
    raw: &'a [f32],
    may_have_core: Vec<bool>,
    width: usize,
    height: usize,
}

impl<'a> SourceEarSupport<'a> {
    pub(super) fn new(raw: &'a [f32], width: usize, height: usize) -> Self {
        assert_eq!(raw.len(), width * height);
        // Determine whether a COMPLETE source component has an ear core
        // before considering protection or connection to moving tissue.
        // Otherwise a protected core can make its soft tail look like a
        // separate low-confidence island and accidentally bypass the lock.
        // A support component that reaches this raster's border may have its
        // core outside the crop. Preserve its old conservative protection;
        // only a fully observed interior component can be called core-free.
        let mut may_have_core = vec![false; raw.len()];
        let mut queue = VecDeque::new();
        for (i, &membership) in raw.iter().enumerate() {
            let touches_crop = i % width == 0
                || i % width + 1 == width
                || i / width == 0
                || i / width + 1 == height;
            if membership >= 0.5 || (membership > 1.0 / 255.0 && touches_crop) {
                may_have_core[i] = true;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for n in neighbors(i, width, height) {
                if raw[n] > 1.0 / 255.0 && !may_have_core[n] {
                    may_have_core[n] = true;
                    queue.push_back(n);
                }
            }
        }
        Self {
            raw,
            may_have_core,
            width,
            height,
        }
    }

    /// Keep source classification independent of these connection barriers.
    /// In particular, a protected core cannot turn its tail into a new class.
    pub(super) fn disconnected(&self, allowed: &[bool], owned: &[bool]) -> Vec<bool> {
        assert_eq!(allowed.len(), self.raw.len());
        assert_eq!(owned.len(), self.raw.len());
        let mut connected = vec![false; self.raw.len()];
        let mut queue = VecDeque::new();
        for i in 0..self.raw.len() {
            if allowed[i] && owned[i] && self.raw[i] >= 0.5 {
                connected[i] = true;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for n in neighbors(i, self.width, self.height) {
                if self.raw[n] > 1.0 / 255.0 && allowed[n] && !connected[n] {
                    connected[n] = true;
                    queue.push_back(n);
                }
            }
        }
        self.may_have_core
            .iter()
            .enumerate()
            .map(|(i, &may_have_core)| may_have_core && !connected[i])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_isolated_ear_evidence_does_not_freeze_owned_cheek() {
        // The first component has a real ear core. The separated 0.123 island
        // reproduces the weak ear evidence seen on the glasses photo's cheek.
        let mut raw = vec![0.0; 8 * 3];
        raw[9] = 0.9;
        raw[10] = 0.08;
        raw[12] = 0.123;
        raw[13] = 0.02;
        let original = raw.clone();
        let evidence = SourceEarSupport::new(&raw, 8, 3);
        let allowed = vec![true; 24];
        let owned = vec![true; 24];
        assert_eq!(
            evidence.disconnected(&allowed, &owned),
            vec![false; 24],
            "a soft-only island must not append a cheek freeze"
        );
        assert_eq!(raw, original, "the raw semantic evidence is immutable");
    }

    #[test]
    fn disconnected_ear_core_and_its_soft_boundary_all_stay_fixed() {
        let raw = [0.0, 0.02, 0.9, 0.08, 0.0];
        let evidence = SourceEarSupport::new(&raw, 5, 1);
        assert_eq!(
            evidence.disconnected(&[true; 5], &[false; 5]),
            [false, true, true, true, false]
        );
        assert_eq!(raw, [0.0, 0.02, 0.9, 0.08, 0.0]);
    }

    #[test]
    fn protected_ear_core_cannot_release_its_soft_tail_as_noise() {
        let raw = [0.9, 0.07, 0.02];
        let evidence = SourceEarSupport::new(&raw, 3, 1);
        // The core is explicitly protected. Its tail may touch moving skin,
        // but source classification must still remember the protected core.
        assert_eq!(
            evidence.disconnected(&[false, true, true], &[true; 3]),
            [true, true, true]
        );
    }

    #[test]
    fn protection_barrier_still_blocks_a_real_ears_soft_edge() {
        let raw = [0.9, 0.2, 0.1];
        let evidence = SourceEarSupport::new(&raw, 3, 1);
        assert_eq!(
            evidence.disconnected(&[true, false, true], &[true, false, true]),
            [false, true, true]
        );
    }

    #[test]
    fn soft_only_island_keeps_all_existing_material_and_user_protection() {
        let mut raw = vec![0.0; 5 * 3];
        raw[6..9].copy_from_slice(&[0.123, 0.04, 0.08]);
        let mut existing_frozen = vec![false; 15];
        existing_frozen[6] = true;
        existing_frozen[8] = true;
        let allowed: Vec<_> = existing_frozen.iter().map(|v| !v).collect();
        let evidence = SourceEarSupport::new(&raw, 5, 3);
        let additions = evidence.disconnected(&allowed, &[true; 15]);
        assert_eq!(additions, [false; 15]);
        let final_frozen: Vec<_> = existing_frozen
            .iter()
            .zip(additions)
            .map(|(&existing, extra)| existing || extra)
            .collect();
        assert_eq!(
            final_frozen, existing_frozen,
            "an ear-only decision must never clear a hair/accessory/user lock"
        );
        assert!(!allowed[6] && allowed[7] && !allowed[8]);
    }

    #[test]
    fn connected_ear_retains_its_soft_edge_without_new_freeze() {
        let raw = [0.9, 0.06, 0.01];
        let evidence = SourceEarSupport::new(&raw, 3, 1);
        assert_eq!(
            evidence.disconnected(&[true; 3], &[true, false, false]),
            [false; 3]
        );
        assert_eq!(
            raw,
            [0.9, 0.06, 0.01],
            "existing ear confidence is untouched"
        );
    }

    #[test]
    fn source_components_do_not_wrap_rows_or_join_diagonally() {
        let mut raw = vec![0.0; 4 * 4];
        raw[5] = 0.9;
        raw[10] = 0.12;
        let evidence = SourceEarSupport::new(&raw, 4, 4);
        let mut expected = vec![false; 16];
        expected[5] = true;
        assert_eq!(evidence.disconnected(&[true; 16], &[false; 16]), expected);
        let wrapped = [0.0, 0.0, 0.0, 0.9, 0.12, 0.0, 0.0, 0.0];
        let evidence = SourceEarSupport::new(&wrapped, 4, 2);
        assert_eq!(
            evidence.disconnected(&[true; 8], &[true; 8]),
            [false, false, false, false, true, false, false, false]
        );
    }

    #[test]
    fn cropped_soft_component_keeps_disconnected_protection_without_visible_core() {
        let mut raw = vec![0.0; 7 * 4];
        // This soft component reaches the top crop edge; its core may be
        // outside the available raster. The second weak island is bounded.
        raw[2] = 0.06;
        raw[9] = 0.08;
        raw[16] = 0.02;
        raw[11] = 0.12;
        raw[12] = 0.02;
        let evidence = SourceEarSupport::new(&raw, 7, 4);
        let mut expected = vec![false; 28];
        expected[2] = true;
        expected[9] = true;
        expected[16] = true;
        assert_eq!(
            evidence.disconnected(&[true; 28], &[true; 28]),
            expected,
            "a crop-edge component cannot be certified as core-free"
        );
    }
}
