# Raw whole-person alpha at a frozen V2 displacement field

This isolated native experiment replaces **only the source coverage used by F/B estimation and image sampling**. It starts from V2 scoped-owner source, without the later known-color-boundary modification. The original V2 geometry, selected-head ownership, frozen pixels, operation mask and region are retained. Production, V2 and the model weights are not modified; no new model was downloaded.

## Source and intervention

Input is the original user photo, `test/results/user-face-contour-20261014/1024/canonical-source.png`. The model candidate is exactly `experiments/whole-person-matte-v1-20261015/outputs/points/alphas.npy[2]`, the official decoder score argmax supplied by the parent probe. The parent retains all eight raw candidates across points and combined-point runs. This experiment does not select a new mask by viewing the edited result.

`inputs/manifest.json` records the upstream NPY hash, index, raw float hash and shape. The 1024×1024 float values are copied without thresholding, RGB refinement or resizing; range is approximately 7.98e-11..1. The crop's actual `moving-physical-f32le.bin` is verified exactly equal to the corresponding raw model crop. Calling this a physical-alpha candidate does not establish physical ground truth.

The override is placed after the original source ownership/frozen decisions and before F/B estimation. It is enabled only by `WHOLE_PERSON_PHYSICAL_ALPHA_F32LE` in this isolated native executable. It does not affect source contour validation or velocity constraints. Source colors, estimated F/B, new samples, holes and reconstructed background are recomputed normally from the replacement coverage. The model's clothing/body coverage does not expand the edit domain or enable body reshaping.

## Reproduction and control

`source-manifest.json` freezes the Rust source, runner and binary. `outputs/report.json` records graph/source/binary/output hashes and measurements. Four requests were run: no-override +30, raw-alpha 0, raw-alpha +30, raw-alpha −30. No-override +30 is pixel-identical to V2; zero is pixel-identical to the original.

For both nonzero raw-alpha renders, all of these are byte-identical to V2: `flow.json`, `requested.json`, `source-xy-f32le.bin`, `frozen.png`, `head-component.png`, `owner-support.png`. Thus the contour displacement and actual inverse field did not change. Output alpha, explicit frozen pixels, unrendered pixels and pixels outside the crop remain exact. Two independent artifact tests pass (`test_artifacts.py`, `artifact-tests.log`). No full-core, WASM or browser tests are claimed for this diagnostic native input override.

All existing V2 trace layers are available under `outputs/raw-{amount}/debug/`. Negative amount is `raw--30`. Source/sample values should be interpreted using `changed.png`; new coverage naturally changes the set of pixels that need color reconstruction.

## Results and rejection

At original scale, −30 visibly reduces the orange duplicate contour near the left ear and the old jagged neck edge. +30 reduces the old gray line above the ear. **A light/gray rim around the outer ear remains**, so the result is not accepted and is not rated ≥8/10. Jewelry still follows V2's nonrigid field, and touching-person ownership remains an inherited limitation.

The raw-alpha candidate differs from V2 on 15,207 pixels at +30 and 10,000 at −30. Source reconstruction residual (maximum RGB channel per pixel) drops from a maximum of 97.02 to 44.23 bytes; its crop-wide mean drops from 0.0854 to 0.0786, a mean dominated by the large already-consistent interior. These measurements do not establish aesthetic quality.

Low-coverage compositing remains unresolved. Where max(old alpha, sampled new alpha) <0.01, raw +30 still changes 1,579 pixels by more than 8 RGB bytes (maximum 36); raw −30 has 569 such pixels (maximum 34). Those pixels should not acquire large color changes from transporting nearly absent foreground. The original control also has this defect, although each candidate's low-coverage domain differs. This separates the improved source coverage from the still-inconsistent color decomposition/reconstruction problem.

The parent is independently investigating a bounded F/B projection satisfying `aF + (1-a)B = C` while keeping the same alpha/field; this experiment contains none of that later change. No residual is added in source coordinates or transported with foreground here.

Runtime was about 35.6/39.0 seconds for raw ±30 in this concurrent host session, with diagnostic file output; no controlled latency or browser-frame-rate claim follows from those wall-clock values.

## Run

Build using `CARGO_TARGET_DIR=.run/whole-person-alpha-override-v1-target cargo build --release --manifest-path experiments/whole-person-alpha-override-v1-20261015/Cargo.toml --bin image-render`. Use `services/background-worker/.venv/bin/python experiments/whole-person-alpha-override-v1-20261015/replay.py` with a fresh output directory; the script refuses to overwrite recorded results. Validate existing artifacts with `services/background-worker/.venv/bin/python experiments/whole-person-alpha-override-v1-20261015/test_artifacts.py`.
