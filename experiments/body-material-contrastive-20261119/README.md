# Manual contrastive part diagnostic

The unchanged single-point candidate bank failed to distinguish side-fashion's pink sleeve from the pink sweater torso. This is a different input mechanism: two source-visible positive points and two negative points are jointly supplied to SAM2's learned prompt encoder. It is an **explicit manual correction capability test**, not an automatic segmentation result and not a repair of an alleged runtime bug.

`probe.py prepare OUTPUT` freezes two reciprocal queries, source/checkpoint/code hashes and source-point images before inference. The four points were selected from visible source RGB interiors, excluding all pre-existing independent evaluation rectangles. No previous mask, alpha, edited image, or candidate score chooses these points. Existing evaluation rectangles are read only to verify this exclusion. `probe.py run OUTPUT` encodes once and retains every multimask from both queries (six candidates), with no threshold change, postfilter, further query, alpha head, or deformation.

The evaluator must judge held-out material cores **and full visible boundaries**, because satisfying four prompt points does not establish a complete sleeve or torso partition. Missing/falsely swallowed material fails this input rule. A useful result could support a manual correction tool; automatic source-point placement would require a separate test. The precise shoulder/sleeve seam is uncertain from this view and must not be invented from either the prompt labels or mask confidence.

Use the existing local Python/Torch/SAM2Matting runtime, without installation or downloads:

```sh
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/probe.py prepare test/results/body-material-contrastive-20261119/side-fashion
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/probe.py run test/results/body-material-contrastive-20261119/side-fashion
```

The experiment depends on local source `test/results/body-material-candidates-20261119/side-fashion-source/source.png`, the official local SAM2Matting Tiny runtime and checkpoint named in `probe.py`, and the independent source evaluation record in `/tmp/body-independent-20261119/side-fashion/`. Raw photos, points, output masks and logits remain in ignored `test/results/`; no old candidate bank or source metadata is changed.

The first fixed run completed with one encoding (2.476 s), two decoding calls (0.177 s and 0.115 s), all six masks retained, and zero alpha-head calls. The independent reviewer checked the four source points before reviewing output and found them reasonable visible-material interiors. Independent full-image review found `sleeve-m1` to be the only useful front-sleeve **proposal**, with small shoulder-edge gaps; `torso-m1` still includes shoulder sleeve material. The sleeve proposal may enter source matting, but its missing pixels are not established transparency and no complete optical layer is accepted by this result. No prompt was changed or candidate hand-repaired after inference.

## Source-layer causal control

`source_layers.py` prepares a separate source-only control. Its sleeve identity comes only from this run's four manual points and `sleeve-m1`. The six other front roles (hands, wrist connector, bag and handles), full rear support, and coat/scene roles reuse explicitly manual same-source v3 identities as constants. **Four points do not prepare the entire image.** Old sleeve polygons, old alpha, old underlays, old scenes and edited outputs are not inputs. All dependent image layers are recomputed.

Before the first new MEMatte inference, the sleeve trimap is frozen with a resolution-derived uncertainty width `ceil(2*max(width,height)/256)` (10 source pixels here, two SAM decoder cells). Enclosed missing pixels are unknown, never forced transparent. Other manual front identities retain their original 4 px uncertainty; the rear identity retains its previously reviewed 8 px source uncertainty. Matting outputs and hidden RGB remain estimates. Two focused tests verify the enclosed-hole and open-notch trimap behavior; they do not verify photographic quality.

Stages are `prepare`, `front-matte`, `underlay`, `rear-prepare`, `rear-matte`, `scene`, and `export`, each with a new output directory and input hash checks. The later source stages reuse the existing `body-fresh-source-20261119` implementation and local MEMatte/LaMa runtimes. `export` emits the existing five float64 source-layer arrays for the ordinary layered Rust API. It does not change geometry or run deformation; source review and separately version-bound `0/±30` playback are still required.

```sh
services/background-worker/.venv/bin/python -m unittest discover -s experiments/body-material-contrastive-20261119 -p 'test_source_layers.py'
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/source_layers.py prepare test/results/body-material-contrastive-20261119/source-layers
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/source_layers.py front-matte test/results/body-material-contrastive-20261119/source-layers
```

## Source-bound manual point entry

`prompted.py` extracts the same SAM2 invocation into a source/request/scope entry while leaving the historical `probe.py` unchanged. It requires an explicit `manualCorrection` claim, original-to-canonical source binding, one or two positive and negative points per role, and exclusion of all pre-existing evaluation cores. It freezes every input before one image encoding and preserves all three masks per query; masks, scores and logits are retained without score-based selection or postfiltering. Five focused tests cover the input budget, label conflict, source bounds and evaluation exclusion.

```sh
services/background-worker/.venv/bin/python -m unittest discover -s experiments/body-material-contrastive-20261119 -p test_prompted.py
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/prompted.py prepare test/results/body-tilted-source-20261119/tilted-shirt/source.png experiments/body-tilted-source-20261119/manual-source-points.json test/results/body-tilted-source-20261119/source-scope.json test/results/body-material-contrastive-20261119/tilted-shirt
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/prompted.py run test/results/body-material-contrastive-20261119/tilted-shirt
```

The tilted-shirt run used four source interior points in reciprocal sleeve/torso queries, one encoding and no alpha-head calls. This remains manual correction with prior exposure, not a blind automatic benchmark. First inspection finds no complete independent sleeve: `front-sleeve-m0` includes torso, `m1` loses much of the cuff and sleeve interior, and `m2` includes a large chest/lapel region. `torso-m1` separates the roles most clearly but has substantial missing cloth. Frozen cuff-core coverage alone illustrates the limitation: sleeve `m1` covers 9.45%, `m2` covers 99.77%, yet the latter's wrong torso inclusion still fails the full-material requirement. Independent source review is required before any MEMatte or source-layer preparation; no additional query, threshold tuning or old mask repair was performed.

This source control completed both fresh mattes and both fresh hidden-color priors; source pairing reconstructs the original with maximum error `2.84e-14`. Independent source review allowed only the frozen waist task: the small translucent shoulder tail remains an uncertain motion split, not a physical free sleeve boundary or a valid shoulder-edit layer.

The first ordinary playback used renderer SHA `50cd45708507fbed4445c2a6ce276aa9135a1a8b8e96c8289d753b5f2af2cc67` and unchanged source graph SHA `9bcd97980c6bfd5d7ca28f40aed08bebb193e38709268e1743cf860f40411799`. Zero was pixel-identical and −30 rendered, but +30 was rejected by the geometry guard: `body geometry folds near (703,395)`. All three attempts are recorded in `source-layers/ordinary/report.json`; the failed run is retained. This is **not a three-strength acceptance**. The source preparation was not changed to hide the geometry failure.


## Geometry identity correction

The old graph still identified the stationary coat as moving waist material. `geometry.py` derives visible motion identity from the frozen source roles and explicitly named replacement proposal, excluding front/scene identities before rebuilding support. It never reads matte alpha or edited images. Coarse apparel classes are diagnostic only: using them as a hard intersection incorrectly removed 3,810 pixels including real pink knit and skirt lettering, so that version was rejected. The corrected mask restores the original explicit source identities.

The role to replace must be named with `--replacement-role`; the geometry helper contains no image-specific role-name branch. Four focused tests cover fixed-role precedence, enclosed proposal uncertainty, arbitrary explicit role names, and invalid/unbound/noncategorical proposals. Final source and native 0/±30 evidence is recorded in `doc/func/FUNC-085-source-motion-identity-consistency.md`. This remains a manual source correction capability, not automatic body-part recognition.
