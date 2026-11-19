//! Caller-owned memoization of one pure source-background reconstruction.
//! No target contour, motion domain, velocity field or rendered result is held.
use crate::{ImageError, RgbaImage};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct FaceSourceCacheStats {
    pub hits: usize,
    pub misses: usize,
}

/// Retains one source reconstruction and its exact inputs. The cache owns its
/// arrays and never aliases a caller image or the mutable post-pairing plate.
/// Changed inputs replace the entry. No disk or global cache is used.
#[derive(Default)]
pub struct FaceSourceCache {
    entry: Option<Entry>,
    stats: FaceSourceCacheStats,
}

#[derive(Clone, Copy)]
pub(super) struct BackgroundInput<'a> {
    pub source: &'a RgbaImage,
    pub holes: &'a [bool],
    pub alpha: &'a [f32],
    pub donor_frozen: &'a [bool],
    pub donor_excluded: &'a [bool],
    pub known_layer: &'a [u8],
    pub estimated_background: Option<&'a RgbaImage>,
    pub reconstruction_scene: Option<&'a [Option<[f32; 3]>]>,
    pub independent_scene_color: bool,
}

struct Key {
    source: RgbaImage,
    holes: Vec<bool>,
    alpha: Vec<f32>,
    donor_frozen: Vec<bool>,
    donor_excluded: Vec<bool>,
    known_layer: Vec<u8>,
    estimated_background: Option<RgbaImage>,
    reconstruction_scene: Option<Vec<Option<[f32; 3]>>>,
    independent_scene_color: bool,
}

struct Entry {
    key: Key,
    background: RgbaImage,
    foreground: Option<RgbaImage>,
}

fn floats_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn scene_equal(a: Option<&[Option<[f32; 3]>]>, b: Option<&[Option<[f32; 3]>]>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| match (x, y) {
                    (None, None) => true,
                    (Some(x), Some(y)) => floats_equal(x, y),
                    _ => false,
                })
        }
        _ => false,
    }
}

impl Key {
    fn new(input: BackgroundInput<'_>) -> Self {
        Self {
            source: input.source.clone(),
            holes: input.holes.to_vec(),
            alpha: input.alpha.to_vec(),
            donor_frozen: input.donor_frozen.to_vec(),
            donor_excluded: input.donor_excluded.to_vec(),
            known_layer: input.known_layer.to_vec(),
            estimated_background: input.estimated_background.cloned(),
            reconstruction_scene: input.reconstruction_scene.map(<[_]>::to_vec),
            independent_scene_color: input.independent_scene_color,
        }
    }
    fn matches(&self, input: BackgroundInput<'_>) -> bool {
        self.source == *input.source
            && self.holes == input.holes
            && floats_equal(&self.alpha, input.alpha)
            && self.donor_frozen == input.donor_frozen
            && self.donor_excluded == input.donor_excluded
            && self.known_layer == input.known_layer
            && self.estimated_background.as_ref() == input.estimated_background
            && scene_equal(
                self.reconstruction_scene.as_deref(),
                input.reconstruction_scene,
            )
            && self.independent_scene_color == input.independent_scene_color
    }
}

pub(super) fn reconstruct_and_pair(
    input: BackgroundInput<'_>,
) -> Result<(RgbaImage, Option<RgbaImage>), ImageError> {
    let mut background = super::reconstruct_source_background(
        input.source,
        input.holes,
        input.alpha,
        input.donor_frozen,
        input.donor_excluded,
        input.known_layer,
        input.estimated_background,
        input.reconstruction_scene,
        input.independent_scene_color,
    )?;
    let foreground = super::layer_pair::pair(input.source, input.alpha, &mut background);
    Ok((background, foreground))
}

impl FaceSourceCache {
    /// Release retained source data and reset hit/miss counts.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn stats(&self) -> FaceSourceCacheStats {
        self.stats
    }

    pub(super) fn reconstruct_and_pair(
        &mut self,
        input: BackgroundInput<'_>,
    ) -> Result<(RgbaImage, Option<RgbaImage>), ImageError> {
        if let Some(entry) = &self.entry {
            if entry.key.matches(input) {
                self.stats.hits += 1;
                return Ok((entry.background.clone(), entry.foreground.clone()));
            }
        }
        self.stats.misses += 1;
        // Do not retain a failed or partially prepared result.
        let (background, foreground) = reconstruct_and_pair(input)?;
        self.entry = Some(Entry {
            key: Key::new(input),
            background: background.clone(),
            foreground: foreground.clone(),
        });
        Ok((background, foreground))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reconstruction_dependency_is_part_of_the_exact_cache_key() {
        let source = RgbaImage::new(2, 2, [130, 100, 90, 255].repeat(4)).unwrap();
        let background = RgbaImage::new(2, 2, [70, 80, 90, 255].repeat(4)).unwrap();
        let alpha = [0., 0.5, 1., 0.];
        let holes = [false, true, true, false];
        let locked = [false; 4];
        let known = [0; 4];
        let scene = [None, Some([70., 80., 90.]), None, None];
        let input = BackgroundInput {
            source: &source,
            holes: &holes,
            alpha: &alpha,
            donor_frozen: &locked,
            donor_excluded: &locked,
            known_layer: &known,
            estimated_background: Some(&background),
            reconstruction_scene: Some(&scene),
            independent_scene_color: true,
        };
        let key = Key::new(input);
        assert!(key.matches(input));
        let mut changed_source = source.clone();
        changed_source.pixels[0] += 1;
        assert!(!key.matches(BackgroundInput {
            source: &changed_source,
            ..input
        }));
        let mut changed_alpha = alpha;
        changed_alpha[0] = -0.;
        assert!(
            !key.matches(BackgroundInput {
                alpha: &changed_alpha,
                ..input
            }),
            "compare exact float bits"
        );
        let mut changed_holes = holes;
        changed_holes[0] = true;
        assert!(!key.matches(BackgroundInput {
            holes: &changed_holes,
            ..input
        }));
        let mut changed_locked = locked;
        changed_locked[0] = true;
        assert!(!key.matches(BackgroundInput {
            donor_frozen: &changed_locked,
            ..input
        }));
        assert!(!key.matches(BackgroundInput {
            donor_excluded: &changed_locked,
            ..input
        }));
        let mut changed_known = known;
        changed_known[0] = 1;
        assert!(!key.matches(BackgroundInput {
            known_layer: &changed_known,
            ..input
        }));
        assert!(!key.matches(BackgroundInput {
            estimated_background: None,
            ..input
        }));
        let mut changed_background = background.clone();
        changed_background.pixels[0] += 1;
        assert!(!key.matches(BackgroundInput {
            estimated_background: Some(&changed_background),
            ..input
        }));
        assert!(!key.matches(BackgroundInput {
            reconstruction_scene: None,
            ..input
        }));
        let mut changed_scene = scene;
        changed_scene[1] = Some([70., 80., 91.]);
        assert!(!key.matches(BackgroundInput {
            reconstruction_scene: Some(&changed_scene),
            ..input
        }));
        assert!(!key.matches(BackgroundInput {
            independent_scene_color: false,
            ..input
        }));
    }

    #[test]
    fn returned_plate_mutation_cannot_poison_later_hits_and_clear_releases_entry() {
        let source = RgbaImage::new(3, 3, [130, 100, 90, 255].repeat(9)).unwrap();
        let alpha = [0., 0., 0., 0., 0.5, 0., 0., 0., 0.];
        let holes = [false, false, false, false, true, false, false, false, false];
        let locked = [false; 9];
        let known = [0; 9];
        let input = BackgroundInput {
            source: &source,
            holes: &holes,
            alpha: &alpha,
            donor_frozen: &locked,
            donor_excluded: &locked,
            known_layer: &known,
            estimated_background: Some(&source),
            reconstruction_scene: None,
            independent_scene_color: false,
        };
        let expected = reconstruct_and_pair(input).unwrap();
        let mut cache = FaceSourceCache::default();
        let mut first = cache.reconstruct_and_pair(input).unwrap();
        assert_eq!(first, expected);
        first.0.pixels.fill(0);
        first.1.as_mut().unwrap().pixels.fill(0);
        assert_eq!(cache.reconstruct_and_pair(input).unwrap(), expected);
        assert_eq!(cache.stats(), FaceSourceCacheStats { hits: 1, misses: 1 });
        cache.clear();
        assert!(cache.entry.is_none());
        assert_eq!(cache.stats(), FaceSourceCacheStats::default());
        assert_eq!(cache.reconstruct_and_pair(input).unwrap(), expected);
        assert_eq!(cache.stats(), FaceSourceCacheStats { hits: 0, misses: 1 });
    }
}
