# Source-only material candidate experiment

This bounded experiment asks whether the already-local SAM2Matting Tiny checkpoint's **ordinary SAM2 segmentation head** contains independent visible sleeve and torso/coat candidates. It does not run the alpha head, deformation, background completion, or edited-image evaluation.

The proposal rule is fixed: from the same-inference FASHN 18-class posterior, find non-background winning pixels with confidence at least 128/255. Divide their bounding box into 4 columns × 8 rows. In each occupied cell query the eligible pixel nearest the cell center (row-major ties); skip empty cells. Each query has one positive point and returns all three multimasks. No masks are selected or clipped by semantics. Confidence is recorded only as a diagnostic and hidden from review sheets.

The competing explanations are that useful parts exist among candidates but need selection, that the segmentation model merges material parts, or that the fixed spatial sampling misses them. The latter two both fail **this candidate route**; neither disproves other prompts or models. Independent review must find separately useful visible sleeves and coat/torso support without materially wrong layer merging. Failure stops this experiment, with no extra queries, manual masks, output-guided prompts, or matting. Passing only establishes candidate availability on one source, not automatic part ownership or usable deformation.

Local runtime dependencies (not downloaded or committed):

- `services/background-worker/.venv/bin/python`, with NumPy, Pillow, Torch, Torchvision.
- The frozen local vendor and dependency trees in `experiments/sam2matting-tiny-probe-20261015/{vendor,deps}`.
- `.models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt`.
- A source directory with `source.png`, `fashn-classes.u8`, same-inference `fashn-logits.f32`, and `class-evidence.json`.

Commands from the repository root:

```sh
services/background-worker/.venv/bin/python -m unittest discover -s experiments/body-material-candidates-20261119 -p 'test_*.py'
services/background-worker/.venv/bin/python experiments/body-material-candidates-20261119/candidates.py prepare test/results/body-material-candidates-20261119/action-suit --source test/results/body-action-source-20261119/action-suit
services/background-worker/.venv/bin/python experiments/body-material-candidates-20261119/candidates.py run test/results/body-material-candidates-20261119/action-suit
```

Preparation freezes the exact source, posterior, runtime hashes and query coordinates before inference. Run verifies those hashes, instruments one image encoding and zero alpha-head calls, and refuses an existing run. The bank retains full-resolution binary masks, unmodified low-resolution float32 logits, and all predicted scores; compressed arrays and review sheets stay in ignored `test/results/`. No connected-component filtering, hole filling, semantic intersection, or candidate ranking is performed. The output is a set of proposals, never alpha, hidden anatomy, or ground truth.

## First run, 2026-11-19

The untouched `action-suit` source produced 27 occupied cells and **81 candidates**, with one image encoding and zero alpha-head calls. On the existing CPU runtime (Torch 2.2.2, float32, 3 threads), checkpoint loading took 0.820 s, encoding 2.759 s, and all 27 decodes together 2.924 s; these are one-run measurements excluding output I/O, not interactive performance promises. All masks and logits occupy about 25 MiB locally. Five deterministic spatial-sampling tests passed.

Viewing all 81 candidates found proposals for sleeves, shirt, and open jacket panels, alongside merged parts and visibly incomplete or perforated proposals. This does **not** establish a usable material partition. In particular, absence of one pure torso mask does not establish that a joint assignment using several candidates is impossible. Independent source-region evaluation and automatic joint assignment are separate work; neither is implemented by this candidate generator. No masks were hand-corrected and no queries were added after viewing results.

`candidate-bank.npz` contains `masksPacked: uint8[81,162080]`, `lowResolutionLogits: float32[81,256,256]`, `predictedIOU: float32[81]`, and `imageShape: int64[2] = [1013,1280]`. Candidate IDs are in the same order as `run.json.candidates`: spatial query row order, then mask indices 0/1/2. Decode masks using NumPy `unpackbits` with its default big-endian bit order, truncate to `height*width`, then reshape. The bank SHA-256 is `ed95f1a612acdf76d5fa0a57b44d69884642fff6829f5eae82bcfb4cc8e18bc8`; protocol SHA-256 is `8dbfcd21d1537f484ea284cc57ee0fc904878dd2e21281236dc66775de7d5b5c`.

## Same-rule second source

`fullbody-side-fashion.jpg` was decoded again through the existing `body-action-source-20261119/detect_source.mjs` entry point, capturing the posterior from exactly one new FASHN inference. The new source is saved in ignored `test/results/body-material-candidates-20261119/side-fashion-source/`, separately from all older source/manual-layer directories. Its `canonical-comparison.json` records **zero different RGBA pixels** against the old canonical image; both PNG hashes are `f53c0889dc4e7e5ebe39192ad99a48375b6bbd26c2626750a1fb5e1f99a92ca2`. The old image was read only after fresh detection for this identity check, never as an inference/prompt/material input. This photograph had been inspected earlier, so this is a cross-image check under unchanged rules, not a new photographic holdout.

The exact same `candidates.py` and 4×8 sampling rule produced **29 queries / 87 candidates** in `test/results/body-material-candidates-20261119/side-fashion/`, with one encoding (2.485 s), zero alpha-head calls, and 2.615 s total decoding. No query was moved or added. The bank has `masksPacked: uint8[87,136480]`, `lowResolutionLogits: float32[87,256,256]`, and `imageShape=[853,1280]`. Its bank SHA-256 is `db5127854baba03bab7889ff90be7e1cf60fd9480b98dfb05eebeb7d002b04fe`; protocol SHA-256 is `6d6374905e65468338066839d05dfecee94c491415c294c37d544936294e2062`. All candidates are available for independent source-only review. No new matting, background completion, or deformation was run in this second candidate pass.

## Automatic source assignment (separate from the unchanged bank)

`assignment.py BANK NEW_OUTPUT` consumes the frozen candidate bank and the original Pose/FASHN observation. It samples each arm bone at .25/.5/.75, accepting only visible, in-bounds points with a compatible winning class of at least 128/255. A candidate must contain an adjacent observed interval on every arm segment and exclude the opposite arm and torso witnesses; remaining ties use coverage then area, never predicted SAM scores. Missing intermediate observations retain their slots and cannot make endpoints artificially adjacent.

The whole-person proposal must contain all body witnesses; semantic foreground soft-IoU ranks admissible candidates. A torso proposal is compatible clothing within that person minus the two sleeve proposals, restricted to components connected to observed torso witnesses. Other components remain unresolved/fixed material, not optical background or guessed sleeves. Missing roles or no observed torso component reject the assignment. This is a source identity proposal, not physical opacity, a separate coat instance, hidden anatomy, or proven depth.

On `action-suit`, this automatically selects `r3c3-m2`, `r4c0-m1`, and whole-person `r6c2-m0`; candidate IDs are results, never inputs. On `side-fashion` the same single-positive-point bank cannot establish separate torso/sleeve ownership and assignment rejects. Both have been seen during research, so neither is a held-out evaluation. The change from whole-set coverage to per-bone interval coverage and the component rule were motivated by observed failures; they are exploratory rules, not preregistered benchmarks.

```sh
services/background-worker/.venv/bin/python experiments/body-material-candidates-20261119/assignment.py test/results/body-material-candidates-20261119/action-suit /tmp/action-source-assignment
```

Five assignment behavior tests and five fixed-grid sampling tests pass without models. The resulting `torso-mask.json` is bound to source PNG SHA. Use `body-action-source-20261119/rebuild_graph.mjs` to adapt its byte raster to the existing 512-axis schema and remeasure support before native replay; do not replace geometry with a matting alpha. Real geometry failure/fix and limits are recorded in `doc/func/FUNC-084-body-occluded-axis-contours.md`.
