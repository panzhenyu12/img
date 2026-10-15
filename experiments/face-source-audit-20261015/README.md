# Face source / transport audit — 2026-10-15

Read-only audit of the saved +100 user source outputs for isolated v22, v23 and v24. No core source, renderer binary, WASM or running service was changed. The canonical image is the existing 1024×1024 user fixture; this is one independent source, not a multi-image quality claim.

## Findings

1. **v22 selects an ear-level join, without neck evidence.** Left join `(349.108, 418.835)` and right `(650.511, 425.980)` have sampled neck confidence **0**. At the inside sample the actual `faceContourMask` is 0 / .076; `(1-face).max(neck) * skin` therefore reports a confident join. That is class absence, not anatomical neck evidence. v23/v24 fix this particular interpretation: left join `(407.137, 552.503)`, right `(613.694, 544.148)`, inside neck .521 / .990.

2. **The v22 visual neck constriction is largely interior motion, not an outer-neck silhouette moving inward.** From row 510 through 600, the saved physical outer contour moves approximately 0 px (sampled five-row maximum .16 px). Meanwhile internal jaw controls at y548,555,561,567 move +5.60,+9.04,+9.94,+9.06 px on the left. The field executes those incompatible semantic targets with max control residual .236 px. Calling this a broken interpolator misses the requested geometry.

3. **v23/v24 impose an early global stop that defeats nonzero neck and lower-jaw targets.** v22 neck velocity weight tapers around image y617→709. v23/v24 change it to y498→582, even though observed outer-neck targets still request several pixels beyond y582. The `weight > .05` control activation threshold shuts down before the continuous weight reaches zero. In v24, left internal jaw control #50 y567 moves **6.95 px**, while #51 y573 has feasible **4.40 px** but actual **.24 px**, `movable=false`; the next y578 control moves -.01 px. External neck at `(405.306,582.100)` requests +3.632 px and moves **0 px**. Max control residual rises to 4.60 px in v23 and 4.16 px in v24. These inactive-target discrepancies must be included in acceptance, not hidden by evaluating only movable controls.

4. **The QP is not the final contour.** External physical guide overrides occur after `afterQP` is recorded; v24 also applies another face lateral taper after QP. Example v24 #74 at y520 has QP x609.810 but final feasible x608.199 (1.612 px override). The fitted field must reconcile this against neighboring controls. The smoothness/shape guarantee of QP cannot automatically be attributed to that final set. v23's extra zero-anchor loop mostly skips because a physical guide already exists at the exact point; the dominant early stop is `neck_motion` plus the .05 threshold, not thousands of newly added anchors.

5. **The black notch is a malformed exposed silhouette; no saved evidence demonstrates a disconnected map.** In the y500–620 jaw/neck window, v23 transports 917 originally opaque pixels to near-zero foreground alpha; all 917 are dark in the final output. v24 has the same count (911 dark). Those pixels lie outside the transported skin layer. The saved sampled Jacobian minimum is .3645/.3938, sampled folds 0, inverse consistency error .000488 px. This is limited sampled evidence, not a mathematical guarantee, but it contradicts explaining these pixels simply as unmapped holes. The measured exterior movement goes ~17.8 px at y510 →5.5 at y540 →5.0 at y565 →0 at y575. Changing inpainting cannot correct that unsuitable silhouette target.

6. **The old donor bridge is an additional, independently isolated color defect.** v23 and v23b saved `flow-geometry.json` and both physical alpha images are byte-identical. The bridge alone changes 2,198 final pixels in `[407,564,616,614]`, max channel difference136. The upper side crease exists before the bridge; the bridge adds lower-neck color edits. v24 then combines a slightly different flow with this old bridge, which is derived from the original `ContourField` target rather than the new transported contour. Avoid explaining every brown boundary as the bridge alone.

## Direct evidence

- `diagnosis.json`: exact input SHA-256, join semantics, full worst-residual / post-QP records, sampled row measurements.
- `v22-controls.json`, `v23-controls.json`, `v24-controls.json`: every source/request/QP/final-feasible/actual control, including inactive controls.
- `v*-alpha-rows.csv`: every image row400–650, center-connected physical alpha .5 source and destination boundary.
- `v*-source-overlay-2x.png`: original with source physical contour green, transported red, source class controls yellow, actual transported controls cyan, inferred joins magenta.
- `v*-neck-alpha-panel.png`: original / final / source physical alpha / transported alpha at 1:1 pixels.
- `bridge-ablation.png`: v23, v23b and v24 final outputs; `bridge-ablation.json` isolates v23→v23b color delta.

The source alpha snapshots are identical across v22/v23/v24. Alpha boundary positions are derived from 8-bit saved diagnostic images and are subject to raster quantization. The source matte itself has irregular edge samples; this audit does not promote its boundary into anatomical ground truth. Some output filenames retain `v22.png` for historical reasons; audit hashes identify actual files. v24 is read from `v24.png`, **not** its inherited stale `v22.png`.

## Next implementation implication

Use one coherent face–neck deformation domain and include actual outer skin, internal jaw, protected features and lower-neck constraints in the same solve. Do not reassign feasible controls after the shape solve; targets must already account for the domain/weight constraints. Shape acceptance should inspect both internal jaw lighting edge and physical outer silhouette, as these are different curves. Preserve the same actual cheek displacement while evaluating the new curve. This is a diagnosis, not a new algorithm or an 8/10 acceptance.

Reproduce the main saved-output audit:

```sh
services/background-worker/.venv/bin/python experiments/face-source-audit-20261015/audit.py
```
