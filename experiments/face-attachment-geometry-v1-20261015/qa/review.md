# Perioral geometry refinement — isolated V36 component

2026-10-15. **The final candidate is Geometry V2 inside the V1 experiment directory.** It improves the mouth-corner deformation, but the complete V36 pipeline remains rejected for its retained skin outline, flat reconstructed band and frozen ear-under wedge. Nothing from this experiment was deployed or applied to production.

## Diagnosis

The source-width velocity in V36 was multiplied by the legacy feature anchor, which returns from zero to full motion within two image pixels outside the detected lip polygon. This creates severe local strain at an otherwise fixed lip corner. A continuous and invertible flow can still stretch a crease into a visible dark line; absence of folds is insufficient.

The makeup −100 image reproduces the horizontal dark extension at the image-left mouth corner. At +100 the same corner becomes unnaturally pointed. Woman +100 similarly pulls the left oral commissure into a sharp point. The diagnostic baseline binary reproduced all four original V36 makeup/user ±100 outputs exactly in RGBA (`baseline/parity.json`).

The ear-under wedge has a different cause. In `makeup-right-ear-diagnosis-1x.png`, the transported physical alpha has a smooth outline, while the output's old skin-colored step follows the **frozen mask** boundary. There are no confident ear-motion anchors in that location. This experiment did not change matting, ownership, frozen masks or background reconstruction to conceal that failure; the evidence was handed to the source-ownership workstream.

## Candidate change

Only the source-width contribution uses the new anatomical attachment weight. The other geometric contributions retain their previous weighting, and there is no branch on a tiny chin/jaw/width value.

For lips and teeth, let `w` be the detected feature's width along the anatomical horizontal direction and `c` its minimum clearance from the semantic face contour. The stable skin core is `min(0.08 w, 0.20 c)`. The smooth outer transition ends at `max(core + 2 px, min(0.25 w, 0.80 c))`. This keeps the commissure shadow attached and distributes deformation through surrounding cheek skin. It uses detected geometry, not coordinates or colors specific to a photo. Eye/nose anchors and their existing short support are unchanged.

The first candidate, Geometry V1, only widened the ramp. It improved the large makeup photograph but failed the new narrow-face/wide-mouth fixture. The final V2 adds a stable core and scales the transition by feature width and available clearance. V1 outputs, source and the failed regression are preserved under `outputs-v1/`, `history/` and `qa/perioral-regression-v1-failed.log`.

## Evidence at original resolution

Each `*-perioral-all-strengths-1x.png` contains **unscaled** Source / V36 / V2 crops at +30, −30, +100 and −100. Reviewed all four photos: makeup, woman, hair and the user-provided face. Full-face comparisons are in `outputs/<photo>-<amount>/face-comparison-1x.png`.

- Makeup: the dark horizontal line at −100 and sharp corner at +100 are removed; the mouth follows the original shape at both moderate settings.
- Woman: the sharp left corner at +100 is substantially restored, and the smile remains attached at all four settings. The old bright face border and flat brown strip remain.
- Hair: the smile corners are more stable. Existing hard outer contour and hair remnants remain.
- User photo: the mouth stays visually consistent. This is a small local improvement, not proof that the face outline is satisfactory.

An independent vectorized RK4 audit reads the saved velocity grid and measures the singular values of the local forward-map Jacobian within circles around detected mouth corners. This is a geometry diagnostic, not a perceptual score:

| Photo / amount | V36 max stretch | V1 max stretch | V2 max stretch |
|---|---:|---:|---:|
| Makeup +100 | 2.839 | 1.204 | 1.104 |
| Makeup −100 | 18.855 | 3.653 | 2.248 |
| User +100 | 1.176 | 1.032 | 1.022 |
| User −100 | 2.694 | 1.646 | 1.309 |

Full values, including the 95th percentile, minimum stretch and determinant, are in `perioral-strain.json`. The finite-difference map measurement uses 0.5-pixel offsets and the recorded grid integration step count. It should not be interpreted as a global strain bound or a proof about unsampled pixels.

## Amplitude and protection audit

Twenty cases were rendered: four original photos at 0/±30/±100. Every candidate graph is deeply equal to the corresponding V36 graph. The 16 nonzero cases have byte-identical source physical alpha and projected source-width target curves. No detector outputs, strengths or semantic masks were changed.

- All four zero outputs equal the original RGBA exactly.
- Hair/ears/accessories at confidence ≥0.5 have zero changed pixels in all cases. This does not imply that every soft-confidence boundary pixel is frozen.
- Output alpha has zero changed pixels in all cases.
- The sampled full-image geometry diagnostics report no folds. They are sampled checks, not a mathematical guarantee.
- All actual contour sample changes are below 0.002 px except woman +100, whose maximum local difference is 0.126587 px. Its left peak changes from 27.905045 to 27.807343 px, while its right peak changes from 17.690613 to 17.691010 px.

The exceptional case has a separate **matched-left-motion control**. V36 at slider 99.5403 has a left peak of 27.807129 px, versus V2 at slider 100 with 27.807343 px: difference 0.000214 px. The right V2 peak is 0.037537 px larger than this control. The V36 corner remains pointed at the matched left amplitude, while V2 restores it. See `same-motion-woman100/mouth-comparison-1x.png` and its `report.json`. This is not an exact whole-field equality claim.

## Regression and ablation

Full release regression: **208 unit tests passed, 0 failed, 3 ignored; 10 public graph contracts passed**. This includes the existing mixed vertical-control zero-crossing contract and combined-control check. No existing assertion was weakened.

The added public-graph regression draws a short dark commissure crease extending four pixels outside the detected lip polygon. It requires the crease to remain attached during ±100 width edits while both cheek sides still move at least three pixels. V2 passes. Replacing only its anatomical source-width anchor with the old two-pixel anchor compiles and fails: the crease spans x=52..107 instead of the allowed x=55..104. See `hard-anchor-ablation.log`. This rejects the apparently safe workaround of fixing the mouth by cancelling all cheek motion.

The initial fixture checked only the red channel and therefore included the blue backdrop in its +100 measurement. That harness mistake was corrected to require all three RGB channels below the dark threshold. The failed log is retained. The final ablation fails with the corrected RGB measurement, and the final candidate passes it.

## Frozen artifacts and reproduction

- Final V2 binary SHA-256: `75e4ad8a3bd94d1bf3a85d788c8088204547566319bf3e0221330522057e5dde`.
- Final source hashes and binary path: `../outputs/provenance.json`; verified again after rendering.
- Full per-case data: `report.json`; aggregate checks: `summary.json`.
- Render/recheck script: `render_validate.py`; strain measurement: `measure_perioral_strain.py`.
- Full tests: `full-regression-v2.log`; targeted failure/ablation logs remain in this directory.
- Minimal V36-only code/test patch, excluding diagnostic exports: `../v36-perioral.patch`. **Do not apply this patch directly to production v19.**

## Production assessment

The deployed v19 + physical-fringe-v2 native binary was frozen at SHA-256 `d0fe0b5251f2f4e862ee20dc7381497fde0f198369f7f37b01eb012c8b68a434` in `experiments/face-perioral-production-v1-20261015/`. Replaying the same makeup and user ±100 graphs did not reproduce the prominent V36 mouth-corner failure at original resolution. Its current motion distribution is different.

Therefore the experiment does not establish an independent production bug fix. No production transplant, browser/API rebuild or deployment was performed. The perioral component is retained for subsequent geometry integration, after the remaining source-ownership and reconstruction failures are resolved. It does not justify an 8/10 rating for the product or the whole V36 pipeline.
