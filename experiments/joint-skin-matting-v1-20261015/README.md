# Direct source skin/neck matting — 2026-10-15

This is an isolated experiment, not the running editor. No candidate here has
passed complete photographic acceptance, and none establishes an 8/10 score.

## Question and controls

Does independently prompting the original RGB for connected face/neck material
recover skin omitted by the old Lab-gated ownership, and improve the final
render? The user image and two previously inspected portraits are exploratory
examples, not independent held-out validation.

- Source-only prompts, original detections, original resolution; no detection
  on an edited output.
- ZIM vendor commit and source/graph/raw alpha hashes in each `report.json`.
- Full/crop × point/box; all four raw candidates retained. Official quality
  score selects index 2 in these runs. A high model score is not matte truth.
- Raw direct skin includes the ears. It must not automatically be interpreted
  as the editable layer or replace user protection.
- Semantic probability, physical alpha, source ownership and target visibility
  are distinct. Zero-motion identity and small source residual cannot prove
  the latter three are correct.

## Completed interventions

1. `outputs/`: initial direct user-photo matte. Independent audit corroborated
   588 source skin pixels below the ears, 287 outside the old support. Sparse
   manual visible-skin labels do not establish opacity=1 ground truth.
2. `composite-v1b/`: direct absolute source/target compositing removes much of
   the doubled skin strip, but has a source reconstruction discrepancy and
   ear/jewelry seams. `composite-v1/` preserves the initial indexing failure.
3. `composite-delta-v2/`: stationary source residual plus moving-skin delta.
   Identity is exact. Independent known-layer tests in `test_delta.py` show
   that alpha and clipped-color errors still leave ghosts. Not accepted.
4. `composite-full-v3/`: extends the source/destination domain beyond the old
   small accessory patch and uses the separately audited observed scene map.
   Removes some leftover dots, but complete attachment quality still fails.
5. `prompts-v2/`: source prompt regression fix. The complement of semantic skin
   previously selected teeth as negative background. `test_prompts.py`
   reproduced this error, then passed after face/person/feature vetoes were
   added. Those vetoes are conservative prompt safeguards, not matting truth.
6. `native-v2-user/`, `native-v2-woman/`, `native-v2-hair/`: all three original
   images rendered at 0, ±30, ±100 by the exact frozen V36 Rust binary. The
   direct mask is temporarily injected as a <=512-pixel semantic skin seed;
   this is a diagnostic input intervention, not the intended public schema.
   Zero is byte-exact. Movement is measured, not assumed unchanged.

## Final-image review of native-v2

All six extreme outputs were inspected at original crop scale. The user's
photo loses most detached old skin, but ear-adjacent wedges/hard joins remain.
The smiling woman has a wide lower-cheek color band at +100. The hair portrait
still has old edge lines at +100 and a hard attachment at -100. Therefore the
new source support is useful evidence, but **the complete native candidate is
rejected for editor integration**.

Native-v2 actual peak movement in original image pixels (left/right):

| Image | +30 | -30 | +100 | -100 |
|---|---|---|---|---|
| User 1024² | 6.006 / 6.006 | 6.005 / 5.902 | 11.107 / 8.991 | 9.406 / 7.221 |
| Woman 1280×1920 | 8.435 / 8.345 | 8.434 / 8.006 | 28.113 / 11.816 | 23.451 / 10.589 |
| Hair 588×768 | 3.395 / 3.395 | 3.395 / 3.395 | 10.746 / 8.236 | 11.175 / 9.831 |

These maxima are not a naturalness score and do not hide left/right asymmetry.
Timings include diagnostics and concurrent CPU activity; not browser latency.

## Reproduce

Use the project Python environment; Pillow/numpy are not installed in system
Python. Output directories intentionally refuse to overwrite prior evidence.

```sh
services/background-worker/.venv/bin/python -m unittest discover \
  -s experiments/joint-skin-matting-v1-20261015 -p test_prompts.py -v
services/background-worker/.venv/bin/python \
  experiments/joint-skin-matting-v1-20261015/run.py \
  --names user-face-contour portrait-woman portrait-hair --out <new-directory>
services/background-worker/.venv/bin/python \
  experiments/joint-skin-matting-v1-20261015/native_replay.py \
  --name portrait-hair --matte <raw-four-candidate-npy> --out <new-directory>
```

Follow-up tests isolate unchanged geometry with direct physical alpha
(`../face-direct-physical-v37-20261015/`) and ear-root pivot/jewelry attachment
(`../ear-accessory-attachment-v1-20261015/`). Both are still experiments.
