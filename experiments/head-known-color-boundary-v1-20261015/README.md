# Exact known-color boundaries in foreground/background estimation

Isolated causal experiment derived from frozen whole-face V2 scoped ownership. Production and the V2 source were not modified. The only algorithm change is inside `face_matting::estimate`: physically known endpoint colors are exact observations, `alpha == 1 → F = C`, `alpha == 0 → B = C`. Values such as 0.989 are not relabeled as certain. The experiment introduces no source-residual or transported-residual compositing and changes no alpha, ownership, target or flow.

## Two modes

Both modes share the frozen binary recorded in `source-manifest.json`:

- `KNOWN_COLOR_FINAL_ONLY=1`: run the prior multilevel solver, then restore known colors at final full resolution.
- Environment variable absent: pin known colors after every level resize and each pixel update, then pin again at final output. Neighbors therefore observe exact known material colors during the iterative solve.

This is a project constraint on the existing multilevel estimator, not a claim to implement a new published SOTA method.

## Red/green contract

`known-color-red.log` preserves the old solver failure. On a textured synthetic composite with exact alpha 0, pixel 14's observed background `[226,46,98]` was estimated as `[216,52,98]`.

The new test requires every alpha-1 foreground and alpha-0 background RGB to equal the original observation exactly. All nine existing/new matting module tests pass in iterative mode (`matting-green.log`); the exact-color test also passes in final-only mode (`final-only-known-color-green.log`). No full-core, browser or deployment test is claimed.

## Real-photo experiment

Only the user's original 1024px photo and frozen V2 graph were used, at natural 0/+30/−30. `outputs/report.json` contains source/graph/binary/output hashes and diagnostics. `replay.py` asserts byte-identical physical alpha, source sampling coordinates, target alpha, flow controls, changed/frozen/holes maps versus V2. Output alpha is unchanged; both zero requests are exact identity.

For both nonzero signs there are 207,691 exact foreground pixels, 237,689 exact background pixels and 7,941 mixed-alpha pixels in the crop.

| Diagnostic | Prior V2 | Final-only | Iterative |
|---|---:|---:|---:|
| Known F maximum observed-color error (byte) | 57 | 0 | 0 |
| Known B maximum observed-color error (byte) | 46 | 0 | 0 |
| Mean mixed-pixel maximum-channel reconstruction residual (byte) | 4.067 | 4.067 | 3.750 |
| Maximum mixed reconstruction residual (byte) | 97.022 | 97.022 | 101.523 |

The final-only mode affects 1,071 output pixels at +30 and 950 at −30 relative to V2. Iterative mode affects 3,603 / 3,166. The maximum output delta is 54 bytes in both modes. Thus the known-color correction is measurable; the large residual in ambiguous mixed pixels is unresolved, and the iterative mode's worst residual is larger.

Original-resolution `left-1x.png` and `face-1x.png` panels show source / V2 / candidate. `final-vs-iterative-*-left-1x.png` compares both modes directly. **The outer-ear light/orange doubled edge remains in both candidates. Neither qualifies for image-quality acceptance or an 8/10 claim.** Correct known colors alone do not fix invalid alpha or unknown foreground/background separation. Jewelry deformation and touching-person ownership limitations remain inherited from V2.

## Trace and reproducibility

The existing V2 debug format is preserved under each nonzero output's `debug/`: old physical alpha, premultiplied sampled RGBA, source coordinates, estimated F/B, reconstructed background, ownership/frozen/changed/holes maps and contour geometry. Source coordinates and samples are only authoritative for output reconstruction where `changed.png` is true.

Build with `CARGO_TARGET_DIR=.run/head-known-color-boundary-v1-target cargo build --release --manifest-path experiments/head-known-color-boundary-v1-20261015/Cargo.toml --bin image-render`. Run `services/background-worker/.venv/bin/python experiments/head-known-color-boundary-v1-20261015/replay.py` into a fresh outputs directory; it refuses to overwrite recorded results. The mode is an experimental native environment switch, not a product-facing option.
