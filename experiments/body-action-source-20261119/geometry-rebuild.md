# Rebuilding geometry after automatic material ownership changes

This is a read-only implementation map for the next automatic candidate. It does not change the existing failed graph, body formula, slider strength, or optical sources.

## Current dependency chain

`landmarks.ts::bodyParts(points,w,h)` (public) derives canonical waist landmarks from the same Pose shoulder and hip centers. `detectPeople` takes `layers.partMasks[id]` from `bodyParsing.ts::deriveBodyLayers`, then calls the public `measuredTorsoSupport` or `measuredLimbSupport`. The measured polygon becomes `DetectedPart.points`; the private `regionForPolygon` computes its bounding region with the existing normalized `.008` margin. The graph writer uses this polygon as `params.mask.points`, that region as `params.region`, the measured part landmarks as `params.landmarks`, and the masks as `partMask/subjectMask/protectMask`.

For a source-reviewed automatic material candidate:

| Input | Required handling |
|---|---|
| Pose points / canonical landmarks | Reuse the frozen source Pose points and current `bodyParts`; do not move the anatomical axis to a candidate-mask centroid. |
| `partMask` | Use the candidate **geometric material ownership** at documented source coordinates. A SAM score or matting alpha is not automatically a geometric part. |
| `subjectMask` | Keep the original source-bound person coverage for the initial comparison. In action-suit, disputed visible jacket interior already has subject=1. Any separate correction requires explicit source evidence. |
| `protectMask` | Preserve existing user/source protection. Do not weaken the laptop/hands protection. New optical front layers must not silently become a geometric hard freeze of hidden moving material. |
| support polygon / `mask.points` | Recompute using the same `measuredTorsoSupport(part,newPartMask,w,h)` and exact current measurement rules. Do not retain the old polygon or enlarge it by hand. |
| `region` | Recompute with the existing `regionForPolygon(newSupport)`; retaining the old region leaves a hard zero-motion boundary even if ownership improves. |
| `bodyPart`, adjustment and selection policy | Keep the same id, `0/±30`, feather, enabled/opacity/blend, and same ordinary API. |
| unavailable status | If the same support measurement cannot establish sufficient visible samples, return unavailable or diagnose that measurement. Do not fabricate hidden waist coverage. |

The source identity candidate may include a complete garment that spans both torso and sleeves. It must therefore remain an instance proposal until a separate anatomy/attachment rule assigns movement regions; feeding the entire jacket mask directly as waist would recreate sleeve dragging.

## Smallest shared entry point

Extract the existing `detectPeople` body-part loop into a public pure helper such as:

```ts
buildBodyPartsFromLayers(
  points: DetectionPoint[],
  layers: Pick<BodyLayers, 'subjectMask' | 'partMasks' | 'protectMask'>,
  imageWidth: number,
  imageHeight: number,
): { parts: DetectedPart[]; unavailableParts: PartUnavailable[] }
```

The helper calls existing `bodyParts`, `measuredTorsoSupport`, `measuredLimbSupport` and `regionForPolygon`; it does not detect, prompt, inpaint or infer alpha. `detectPeople` and an isolated algorithm replay import that same helper, avoiding a second support implementation in an experiment. First verify baseline parity for the frozen current layers; only then feed the source-reviewed new ownership. Reusing the canonical landmarks while rebuilding support/region is the intended controlled change.

No app or service build is needed. Existing Vite/Playwright algorithm harness imports the current TS source and writes one ordinary graph. Current native `image-render` gives final RGB; the public `prepare_body_inverse_displacement` gives the same geometry field for diagnosis without prepared optical layers.

## Why the Rust result otherwise stays ineffective

`body.rs::geometry_target` intersects `partMask` with the refined `subjectMask`, then refines the selected geometry. `motion_controls` rasterizes `mask` and `protectMask`, constructs `ContourProfile` from that target, and computes protected-boundary falloff. `build_motion_field` fixes every point outside `region`, outside `mask`, or inside existing hard protection. Correcting only `partMask` cannot move the action-suit right side beyond the stale selection/region.

`measuredTorsoSupport` currently casts rays from five source-axis samples and accepts the first `< .5` crossing. A foreground occluder, internal gap or wrong material partition can masquerade as an exterior contour. The new candidate must be checked for this failure too. If corrected visible ownership still causes first-hit/internal-boundary ambiguity, document it as a separate measurement failure; do not widen the support until it looks right.

## Implemented isolated adapter

For the immediate automatic-candidate experiment, `rebuild_graph.mjs` calls the current public `bodyParts` and `measuredTorsoSupport` directly. It contains an explicitly documented equivalent of the private `regionForPolygon` bounding calculation, rather than extracting application code during the experiment. Using the original partMask reproduced the frozen graph **byte for byte** (SHA `37b1730580e6afa91f2043dd69721742441198b186f749e141e715727bdbccc0`). This verifies the isolated support path before any candidate ownership is substituted. The proposed shared pure helper remains a later integration choice, not a prerequisite for the experiment.
