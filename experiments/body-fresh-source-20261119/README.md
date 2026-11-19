# Fresh original-photo body diagnosis

`detect_source.mjs` starts an empty temporary Vite page solely to import the existing body detector. It reads the original photo, invokes current MediaPipe Pose/FASHN once, and writes a source-sized PNG, source-bound observation and one ordinary body graph. It never reads historical prepared inputs or edited output. Existing local models are required; outbound browser requests are blocked. The temporary browser, server and generated module cache close after inference.

`replay.py` verifies the frozen source/graph hashes and runs the existing ordinary native `image-render` for waist `0/-30/+30`. It writes source role overlays and final-RGBA evidence, preserving the original strengths and source geometry. Use the repository Python environment with NumPy/Pillow. `--output-name current-native` permits a separately documented current-binary replay without repeating detection. No build occurs inside either script.

```sh
node experiments/body-fresh-source-20261119/detect_source.mjs test/images/fullbody-side-fashion.jpg /tmp/body-fresh-source-new waist
services/background-worker/.venv/bin/python experiments/body-fresh-source-20261119/replay.py /tmp/body-fresh-source-new
```

The first ordinary body result on the existing side-fashion photo fails material isolation; see `doc/qa/body-fresh-source-failure-20261119.md`. Nonzero pixel changes and unchanged protected pixels do not establish naturalness or correct material ownership. No photographs, generated graphs, model weights or output images belong in Git.

## Local prerequisites

Numerical preparation requires Python with NumPy, SciPy and Pillow. Model stages additionally require PyTorch, Detectron2, the existing official MEMatte vendor/config under `.run/mematte-20261020/vendor`, `.models/mematte-20261020/MEMatte-AIM500-ViTS.pth`, and `.models/background/big-lama.pt`. `model_runtime.py` preserves the established local `.run/` dependency directories, including a macOS x86_64 / CPython 3.11 Detectron2 build path. These are explicit local prerequisites, not an environment supplied by a clean clone; another machine must provide compatible runtimes and adjust those paths. Source-only numerical tests do not load model weights.

Fresh detection also needs the existing `apps/web/node_modules` Vite and Playwright packages, Playwright Chromium, and `apps/web/public/models/{bodyparser.onnx,pose_landmarker_lite.task}`. Ordinary replay requires the existing native `image-render`; layered replay requires the `body-layered-replay` example. These scripts neither install dependencies nor build the renderers.

## Explicit manual source control

`annotate_source.py` declares source material/instance/occlusion roles and review overlays. This fixture-specific annotation was made **after observing the automatic failure**. It is not an automatic detector or recovered posterior. The original body graph remains frozen. Both the identity proposal and source trimaps must be reviewed before model preparation; hidden boundaries and optical alpha are estimates.

`prepare_manual_layers.py` has independent stages `front-prepare`, `front-matte`, `underlay`, `rear-prepare`, `rear-matte`, `scene`, `export`, and `replay`. Models are run only during source preparation. Final `replay` uses the ordinary public Rust layered-body API with the original graph and strengths `0/-30/+30`; it never substitutes optical alpha for the anatomical/geometric masks. Source preparation outputs are hash-bound and refused if already present. The runtime uses existing local model code/weights and does not download or build anything.

The specific frozen v3 experiment includes a separately reviewed `rear-trimap-source-review.json`: the coarse rear polygon was 3–5 px outside the visible original denim transition, so its optical uncertainty band was widened from 4 to 8 px **before the first rear inference**. The original annotation is retained. This source-only revision is not an output-driven alpha adjustment. Reproducing the exact v3 evidence requires that frozen sidecar; omitting it runs the originally requested 4 px band and is a different source experiment.

For a fresh source annotation directory, proceed one stage at a time so source proposals and trimaps can be reviewed before inference:

```sh
services/background-worker/.venv/bin/python experiments/body-fresh-source-20261119/annotate_source.py /tmp/body-fresh-source-new --output-name manual-role-control
services/background-worker/.venv/bin/python experiments/body-fresh-source-20261119/prepare_manual_layers.py front-prepare /tmp/body-fresh-source-new/manual-role-control
# Review the frozen source identity/trimap, then run front-matte and underlay.
# Run rear-prepare, review its frozen source trimap, then rear-matte, scene, export.
services/background-worker/.venv/bin/python experiments/body-fresh-source-20261119/prepare_manual_layers.py replay /tmp/body-fresh-source-new/manual-role-control
```

`audit_visible_edge.py` measures the original-image preselected left denim RGB transition. This only corroborates visible contour response, not alpha truth or hidden right-waist width. Both independent reviewers accepted the frozen manual v3 `0/±30` outputs for this photograph: the known coat/sleeve motion defects disappear while left denim contour motion remains. The automatic source graph still fails material identity. See `doc/func/FUNC-083-body-source-material-control.md` for the bounded acceptance and remaining work. No source arrays, photographs, model weights or generated results belong in Git.
