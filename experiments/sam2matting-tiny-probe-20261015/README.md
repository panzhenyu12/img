# SAM2Matting Tiny source-alpha probe — 2026-10-15

**Inference works on this machine; whole-domain replacement is rejected.** On all three source-only tests, the raw output lowers apparently opaque forehead skin. This result is retained as a failed candidate, not shipped or assigned an aesthetic score.

## Reproduce

Official code: `FudanCVL/SAM2Matting` at `73dd721d77b56749248aefe5e8824d7f61b9d13c`; unmodified archive in `vendor/`, provenance in `vendor-revision.json`. Official Tiny checkpoint is `.models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt`, 215,569,778 bytes, SHA256 `5b9321e3b51bc20f5b84c208746cc083dd3053dd701590f2e88dc8640afcc39d`. HF revision `4315db9c60d27fde396b09765748a0ca6c97bed5`; download ledger lives beside the checkpoint.

Host: x86_64 macOS; Python 3.11.14, torch 2.2.2, CPU float32, 3 intra-op threads, 1 inter-op thread. No MPS, TensorRT, autocast, or compile. The existing background-worker Python interpreter supplies torch. Extra dependencies are installed **only** in this experiment's `deps/`: torchvision 0.17.2, hydra-core 1.3.2, omegaconf 2.3.0, antlr4-python3-runtime 4.9.3, iopath 0.1.10, portalocker 2.10.1, PyYAML 6.0.2, tqdm 4.67.1, typing-extensions 4.14.1, packaging 25.0. The official checkpoint load reports zero missing and unexpected keys. This demonstrates local compatibility, not official support for torch 2.2.2.

```sh
services/background-worker/.venv/bin/python experiments/sam2matting-tiny-probe-20261015/batch_v2.py
services/background-worker/.venv/bin/python experiments/sam2matting-tiny-probe-20261015/test_artifacts.py
```

Inference refuses to overwrite its existing result directory. To rerun, retain this experiment and change the new run's output directory explicitly. The artifact audit runs without inference. Earlier `probe.py` and `audit.py` are retained for the first user-only, old-full-point-seed trial in `outputs-cpu/`; that seed is not the same as the current three-image protocol.

## Comparable protocol and outputs

Each source RGB is unmodified and matched by SHA256 to `prompts-v2/<case>/report.json`. All three masks are exactly `experiments/joint-skin-matting-v1-20261015/prompts-v2/<case>/crop-point/alphas-full.npy[2] > .5`. The official image predictor receives full RGB and this binary seed; no redetection, manual result masking, semantic alpha clipping, or RGB generation is performed. The output is raw model alpha at original resolution.

`outputs-v2-crop-seed/report.json` contains input, model, graph and output hashes. Each case directory has `alpha-raw.npy`, `alpha.png`, `input-mask.png`, and `face-1x.png`; the user case additionally has left/right cheek crops. The green previews use source RGB multiplied by alpha over green. They are **not** recovered foreground RGB or actual reshaping renders.

| Case | Encode / decode | Reference opaque pixels | Alpha lowered below .9 / .5 |
|---|---:|---:|---:|
| user-face-contour | 3.950 / 5.306 s | 137,808 | 16,844 / 9,853 |
| portrait-woman | 3.603 / 3.957 s | 241,630 | 7,273 / 375 |
| portrait-hair | 3.658 / 4.053 s | 42,474 | 1,214 / 560 |

Reference means both ZIM full and crop alpha ≥ .99, **not ground truth**. The full/crop binary masks themselves differ at 16,598 / 5,219 / 2,383 pixels. Batch peak resident memory is 2,921,529,344 bytes. These are individual CPU measurements, not browser timings or a performance distribution.

Two reviewers independently inspected the original-resolution previews: user has a broad transparent forehead and soft neck-bottom; woman has a thin/green forehead edge and altered ear/accessory coverage; hair has a clear transparent forehead notch and soft neck-bottom. Ear/accessory loss in a skin-only mask is not independently a failure without an ownership contract; the unambiguously internal skin loss is sufficient to reject a whole-domain physical-skin replacement. No fixed-geometry warp replay is claimed here.

The meaningful audit checks provenance, exact seed protocol, finite/ranged original-resolution alpha, preserved raw-to-preview conversion, and reported disagreement counts. These data checks prevent comparing the wrong inputs or silently postprocessing a rejected output. They do not certify matting quality or the ≥8/10 goal.

Research decision and candidate primary sources: `doc/research/portrait-layer-research-20261015.md`.
