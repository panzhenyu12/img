# V31 / V32 independent source-curve review

Read-only review of the frozen `face-source-curve-v31-20261015` and `face-source-curve-v32-20261015` outputs. No implementation or production changes. Each version has one original, two frozen support choices, 0/±30/±100. This audit does not treat the copied legacy `requested`, `feasible`, or control residuals as the SourceWidth target.

## Confirmed causes

### 1. V31 loses the broad attachment transition when replacing the field

For this image the width is308.001 px. `attachment_taper=.14*width=43.120 px` is used by `SourceWidth::new` only to select the normalization reference. The row motion does not include that weight. The replacement velocity is multiplied by `smoothstep(2,10.780,d_attachment)` instead. Therefore a raw20.020 px motion is turned on within8.780 distance pixels; the smoothstep's maximum derivative permits about **3.42 pixels of velocity change per distance pixel**.

The saved actual alpha confirms abrupt shape changes. In parser+100, the right source boundary near y426.6 has an approximately **67.7°** six-pixel tangent change. The raw section motion there is20.02 px; attachment distance is7 px, and the final inward boundary displacement is19.41 px. The final motion can exceed the initial weighted velocity because contraction moves the trajectory away from the ear and into higher velocity. Expansion moves into the zero-velocity ear boundary and behaves differently. Merely multiplying the initial control by a distance weight is not a final displacement guarantee.

V31 also uses a raster city-block distance and nearest distance lookup to generate grid velocities; the grid is subsequently bilinear. The saved output shows more than raster stair quantization, so changing only the distance metric will not fix the main defect.

### 2. The expansion spike partly transports extra source skin/ear-edge support

The combined support contains a side branch that the first center-outward radius profile does not represent. At anatomical row whose center y is429.6:

- source alpha main interval ends at x647.95;
- a separate skin/ear-edge interval exists at x651.20–665.20;
- -100 transported alpha merges both into an interval ending at x678.95;
- the next row splits again into intervals ending at655.20 and655.95–679.45.

This explains an apparent31 px silhouette extension despite20 px raw motion. It is not proof that the same main-face source point moved31 px. At source `(658,430)`, RGB is `[160,122,109]`, combined skin confidence.914, physical alpha233, ear confidence.0039; parser skin and physical alpha are0 there. At `(648,430)`, combined skin is.794 but physical alpha is0, exposing an internal source-ownership discontinuity. The precise causes of that zero include existing ownership/head clipping and must be audited before promoting raw ZIM-union skin to a canonical material layer.

A safe remedy cannot erase all small components: these may be genuine earlobe skin. The source component must be assigned to its actual anatomical owner and the geometric control representation must cover every transported component. A single first-hit radius cannot control the complete source alpha when the row has multiple branches.

### 3. V32 improves the shelf but flattens the lower cheek by construction

V32 applies the wide attachment weight to the row profile, smooths longitudinally, and renormalizes to20 px. Its physical source alpha is exactly the same as V31 for each support/amount. The brief ear transition is reduced but remains excessive (parser+100 worst right-ear six-pixel tangent difference around49°).

For combined+100, fitting the actual physical boundary over larger bands yields:

| Image-y band | Side | Source tangent | Actual tangent | Actual/source slope |
|---|---|---:|---:|---:|
|435–470|Left|6.93°|16.22°|2.39|
|435–470|Right|-7.65°|-26.14°|3.66|
|470–510|Left|16.41°|4.87°|.289|
|470–510|Right|-18.28°|-3.33°|.176|
|495–535|Left|31.96°|16.09°|.462|
|495–535|Right|-29.88°|-12.18°|.376|

The upper transition steepens while the lower cheek becomes nearly vertical. This is not a texture-blending problem. In V31 the formula is approximately `r_target=(1-k)*r_source+k*r_neck`, where normalization gives k≈.615 on the left and.796 on the right. It inherently compresses source radial slope toward zero. Smoothing and maximum normalization do not enforce an attractive or anatomically coherent target curve.

## Implementable next step

Solve the target outer curve directly. For this near-frontal fixture, per-side `r_target(v)` with a clamped cubic spline or sampled QP is sufficient; a general pose needs an arc representation rather than assuming horizontal sections are anatomical truth.

- Use source tangents and curvature as observations, with confidence around occlusion and raster uncertainty. Fit requested reduction as a preference.
- Give ear-root and face–neck joins explicit position **and tangent** conditions. Include the small permitted neck motion in these conditions from the beginning.
- Bound target tangent changes and curvature over anatomical spans. On a source span with a reliable single sign, prevent a new inward/outward extremum. Do not force one global curvature sign across the whole face.
- Solve the maximum feasible strength while keeping these shape conditions. If20 px cannot be met, retain the requested value, report the achievable value and limiting constraint, and compare with an old algorithm at the same actual displacement. Do not normalize after solving, since that invalidates the shape bounds.
- Derive the transported field from this accepted curve and material-domain conditions. Evaluate its actual transported physical contour against the solved target. Do not append a short spatial taper that silently changes the target again.

### Ear locking is a product/model decision

The user asked to preserve the background; that does not logically require every ear pixel to remain at the same coordinates. A more faithful model can keep each ear rigid while allowing limited translation/rotation with its ear root. Earrings can follow the earlobe as an attached rigid part. Explicit user protection remains hard. This is a distinct mode/contract from the current semantic ear pixel lock and must not silently weaken existing regression assertions. If ears must remain frozen, the shape solver should honestly expose the resulting feasible displacement limit.

## Files and limitations

- `diagnosis.json`: V31 measured source/transported boundaries, attachment distances, model velocities, tangents and hashes for all eight nonzero outputs.
- `v32-diagnosis.json`: same-source V32 measurements and broad-band tangent fits.
- `source-fragment-semantics.json`: source RGB and class/physical alpha at six inspected edge samples for both supports.
- `*-ear-overlay.png`: original and output with source edge green, transported edge red.

The alpha contours use the frozen8-bit diagnostic alpha at.5 and are subject to raster uncertainty. A first-crossing curve can jump between branches, so large tangent changes near the combined ear branch are silhouette evidence, not a material rotation measurement. No rating or acceptance is inferred from these numbers.
