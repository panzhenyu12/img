# V36 independent review — rejected for visual quality

Date: 2026-10-15. Scope: immutable V36 native binary and frozen V35 input graphs. Candidate and production source were not edited during this audit.

V36 restores actual cheek movement on the woman, hair and makeup photos, but does not produce acceptable photographs. The woman +100 result retains the old bright skin outline and fills the exposed area with a visibly flat brown strip. The makeup photo exposes a hard ear/jaw boundary at +100 and a lateral dark extension at the left mouth corner at −100. These defects reject the candidate regardless of passing unit tests or larger displacement. This audit does not establish an 8/10 quality score and does not recommend deployment.

## Provenance and causal result

- V36 binary: `outputs/image-render-ac0f43bcdb8d4a96e78b80daab37dca1c2f8f701884bb8745b1e60a6d1c94a0b` (path relative to the experiment).
- V35 binary SHA-256: `04775f72148bec39c5994b88670d0ec6b5521c67275209cd94981a1f6b25c2b3`.
- Every Rust source hash is recorded in `audit-report.json` and was checked again at the end of the audit. Only `face.rs` and the diagnostic environment-variable name in `face_source_curve.rs` differ from V35.
- The operative change removes hanging hair from motion anchors, while retaining ear anchors and scalp-root hair anchors above normalized face height 0.30. Output protection and donor logic are unchanged.
- All 19 operation graphs are deeply equal to their corresponding V35 graph; detection, masks and slider values were not changed.
- All 16 nonzero cases have byte-identical source physical alpha PNGs to V35.
- Running the immutable V35 binary with the exact V36 woman +100 source and graph reproduces the previous V35 image byte-for-byte: SHA-256 `2e6087a756f6f5b9d0406a09231586982f492aa6d3e704a20b8104672baeeeb3`. See `ablation-woman100/`.

Together these checks support the narrow causal conclusion that removing the inappropriate hair motion anchors restores movement. They do not validate the resulting contour or reconstruction.

## Regression and ownership boundaries

The full release Rust regression passed: **207 unit tests passed, 0 failed, 3 ignored; all 10 public graph contracts passed**. No assertions were changed. Log: `full-rust-regression.log`.

The image audit includes woman, hair and user photos at 0/±30/±100, plus makeup at ±30/±100, for 19 cases:

- Hair, ears and accessories with mask confidence **≥0.5**: zero changed output pixels in all cases.
- Output RGBA alpha: zero changed pixels in all cases.
- Zero slider: all three outputs exactly equal the source pixels.
- The user photo: all five V36 images are exactly equal in RGBA to V35.

This protection result is threshold-specific. Pixels in the soft confidence interval `(0.000001, 0.5)` can change. The maximum changed counts across cases are 3,353 hair pixels, 150 ear pixels and 414 accessory pixels. The source comment saying “every hair output pixel” should not be interpreted as a promise for every positive hair-confidence pixel. Full per-case counts are in `audit-report.json`.

## Actual movement

These are maximum absolute movements of valid sampled contour points, in original-image pixels. They measure response, not aesthetic quality or movement at every contour point.

| Photo / faceWidth | V35 actual left / right | V36 actual left / right |
|---|---:|---:|
| Woman +30 | 0.000 / 0.785 | 8.435 / 8.421 |
| Woman +100 | 0.001 / 2.272 | 27.905 / 17.691 |
| Woman −100 | 0.001 / 2.368 | 23.222 / 15.292 |
| Hair +100 | 0.002 / 2.952 | 10.695 / 7.884 |
| Makeup +30 | 0.014 / 5.520 | 7.599 / 7.600 |
| Makeup +100 | 0.035 / 6.463 | 24.066 / 25.330 |
| Makeup −30 | 0.014 / 5.521 | 7.599 / 7.600 |
| Makeup −100 | 0.034 / 6.463 | 25.175 / 18.026 |

The makeup case no longer saturates near 6.5 px at both 30 and 100. This improvement also exposes the unresolved composition defects at useful strengths. The diagnostic makeup renders took approximately 5.8–7.4 seconds each; these runs include diagnostic exports and are not an interactive latency benchmark.

## Original-resolution visual failures

The retained white/bright edge on woman +100 is visible in `woman-left-edge-1x.png` (unscaled source/output crop). Three independently inspected original-image texels illustrate the ownership error:

| Position x,y | Source and output RGB, unchanged | Hair confidence | Skin confidence |
|---|---|---:|---:|
| 395,960 | 188,153,134 | 0.960260 | 0.031047 |
| 418,1000 | 217,175,151 | 0.937429 | 0.046248 |
| 453,1040 | 211,169,147 | 0.878012 | 0.064633 |

These visually skin-colored/mixed contour texels are frozen by confident hair ownership, leaving the old contour when the actual face moves. They are diagnostic probes only, not source-specific coordinates proposed for a fix. Probe values: `woman-left-edge-probes.json`.

The flat brown reconstructed strip is a separate failure: correcting ownership does not supply the missing hair/background texture. No test result here establishes that a mask correction alone repairs that strip.

The makeup comparisons are unscaled source / V35 / V36 panels:

- `makeup-100/face-comparison-1x.png`: old skin-colored wedge below the image-right ear with a visibly stepped boundary against the yellow background.
- `makeup--100/face-comparison-1x.png`: dark horizontal extension at the image-left mouth corner; hard transition between the widened cheek and hair.
- `makeup-30/face-comparison-1x.png` and `makeup--30/face-comparison-1x.png`: weaker defects but still visible contour transitions. Reduced strength is not an acceptance remedy.

The woman, hair and makeup evidence prevents treating topology checks, fixed-pixel preservation, or movement amplitude as a substitute for photograph quality. The frozen semantic ownership itself can preserve the wrong visible edge.

## Reproduction

Run the existing audit from the repository root:

```sh
services/background-worker/.venv/bin/python experiments/face-source-curve-v36-20261015/qa/audit.py
```

It analyzes the 15 parent-generated images, reruns the immutable V35 rollback, renders the four makeup cases, and checks source/binary hashes. The parent render outputs, graphs and source files must remain present. `audit-report.json` is the full machine-readable evidence; `summary.json` records the acceptance decision and aggregate checks.
