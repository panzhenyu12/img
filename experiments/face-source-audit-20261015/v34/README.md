# V34 independent matched-motion review

## Decision

V34 is worth continuing into independent photographs and mixed-slider continuity work. It is not ready for production and is not an 8/10 result. There is a demonstrated neck-geometry benefit at matched cheek motion, especially ±30; the strong-setting improvement also contains substantial constraint-induced amplitude reduction, reported separately below.

All eight nonzero cases were reviewed at native image pixels in the saved four-column panels. They represent **one original and two frozen support variants**, not eight independent photographs.

## Method

The same original and exact same graph support are used for V34 and immutable guarded V21 binary `777c8948d6349e91806cec3ae02ff4fc04ff93f61843bede2744c22d2dab722e`. Every comparison asserts the source physical alpha PNG is byte-identical. No edited-image redetection, new source masking, candidate edits, native rebuild or production changes occurred.

The physical outer boundary is the first center-connected alpha .5 crossing on each side at anatomical rows whose image-center y is440–535. V21 is calibrated separately to match (a) the maximum absolute boundary change across both sides and (b) the mean absolute change. Sixteen final matches are within **.03 px** of their respective target metric. The initial proportional search is retained separately; bracket refinement resolved its misses. Neck measurements use y550–610.

This is an image-silhouette measurement, not a material-point trajectory. Parser fragments can connect/disconnect and change the first crossing. For example V34 parser-30 has a section motion peak near6 px but a physical-boundary change of7.509 px near y501.6. That is not interpreted as a control-target residual. Copied legacy requested/feasible control fields are never used as V34 targets.

## Comparison

`Neck` below is mean absolute outer-edge change averaged across both sides. The JSON retains left/right metrics, per-row boundaries, target peaks, hashes and every matching attempt.

|Support|V34 input|V34 cheek max / mean px|V21 max-matched input|V21 mean-matched input|Neck V34 / V21-max / V21-mean px|
|---|---:|---:|---:|---:|---:|
|combined|+30|6.018 / 4.677|+29.507|+26.041|0.689 / 2.827 / 2.499|
|combined|+100|11.295 / 7.316|+55.489|+40.846|2.264 / 5.697 / 4.041|
|combined|-30|6.002 / 4.403|-29.930|-23.477|0.752 / 2.693 / 2.172|
|combined|-100|9.372 / 5.803|-46.294|-31.664|2.454 / 4.075 / 2.891|
|parser|+30|6.016 / 4.897|+29.028|+27.424|0.663 / 2.801 / 2.669|
|parser|+100|12.587 / 8.371|+60.406|+47.192|2.176 / 6.281 / 4.760|
|parser|-30|7.509 / 4.785|-30.372|-24.751|0.702 / 2.727 / 2.273|
|parser|-100|12.525 / 7.100|-58.033|-38.453|2.288 / 5.009 / 3.403|

## Supported improvements

- Combined+30 has actual cheek max6.018 px; V21+29.507 matches that max, and V21+26.041 matches its mean. V34 neck mean on the left/right is.818/.560 px, versus3.230/2.424 for max-matched V21 and2.863/2.136 for mean-matched V21. This benefit cannot be explained solely by reducing requested strength.
- Combined+100 has actual max11.295 px and mean7.316 px. Compared with max-matched V21+55.489 and mean-matched V21+40.846, the upper neck is visibly less pinched. At y470–510 its left/right tangent angles are17.11°/-18.55°, near the original16.41°/-18.28°. It avoids V32's almost vertical lower-cheek result on this photo.
- The ±30 images retain a continuous-looking face–neck transition without the v23/v24 dark neckline wedge or donor-bridge block. This is a visual observation from the native-size panels; source edge defects still remain.

## Limits and failures

- The strong request was20.020 px but the combined+100 solved row peaks are about11.25/9.00 px; -100 about9.38/7.24 px. This is a constrained smaller result. It cannot be described as a same-amplitude fix of the old20 px edit.
- A global max or mean match does not match each side or every contour row. Combined+100 left/right actual mean8.341/6.291 px versus mean-matched V21 left/right7.463/7.146 px. Independent side constraints can turn segmentation asymmetry into extra facial asymmetry. Pose-aware interpretation or a soft inter-side preference should be evaluated without forcing all real faces symmetric.
- The original parser support remains visibly unusable at strong contraction: both ears/earrings have retained vertical skin strips, and left-side small skin islands remain. V34's improved field does not fix these source-ownership errors. The combined support reduces those strips but still has uncertain ear-edge branches and is not a generally validated detector.
- There is still only one independent original. This audit does not cover side profiles, glasses, long hair, occlusion, multiple people, image-size scaling, feature protection, or browser/native parity. It does not establish a human-rated quality score.

## Next gate

Proceed with frozen-binary independent source photographs and mixed width/other-slider tests. Record whether the SourceWidth path actually ran; a successful legacy fallback is not new-algorithm coverage. Evaluate the actual transported source-width samples and physical silhouette alongside the intended QP target. Check tiny positive/negative additions of other sliders for mapping/output discontinuities. These are gates for further evaluation, not authorization to label the current parser path fixed.

## Artifacts

- `report.json`: final sixteen matched comparisons, all source/output/binary hashes, per-side and per-row measurements.
- `report-initial-proportional-search.json`: uncorrected matching attempts retained transparently.
- `combined-100-face-comparison-1x.png`, `combined-30-face-comparison-1x.png`: original / V34 / V21 matched maximum / V21 matched mean. Equivalent panels exist for both supports and all four strengths.
- `match.py`, `refine_matches.py`: reproduce baseline-only rendering and matching.
- `v21-*/`: exact graph, immutable-binary rendered output, debug alpha/flow and log for each tested baseline amplitude.
