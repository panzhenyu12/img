# Semantic deformation experiment

Offline algorithm work only. Whole-face target generation is Rust; the reduced
source-field solver and full-resolution audits are Python reference experiments.
No production Web/Go integration or aesthetic acceptance is claimed.

Install `requirements.txt` in a Python environment, then from the repository root:

```sh
make quality-semantic SEMANTIC_PYTHON=/absolute/path/to/python
make quality-semantic-local SEMANTIC_PYTHON=/absolute/path/to/python
```

The first target uses mathematical fixtures and the independent Rust crate.
The second reads local ignored source arrays and photo bytes for hash checks;
its photo-derived tests skip when that evidence is absent. Photos, source traces,
models, solved arrays, generated images and binaries are deliberately untracked.
Passing these tests does not establish natural-looking edits.

`run_reduced_photo.py --input <frozen-source-input> --output <new-directory>`
compares a reduced variable space against an immutable target contract. An optional
`--refinement <extra-nodes.npy>` adds local source freedom; `--constrained` runs
at most two source SOC active-set rounds. All paths refuse to overwrite results.
The source proxy, actual float32 inverse, material RGB and background lines are
separate gates; failures never imply an accepted image.

The local photo renderer used for historical comparisons is the existing binary
at `.run/face-current-20261019-target3/release/examples/face-photo-transport`.
Its source is in `core/image-core/examples/face-photo-transport.rs`, with the
standalone implementation in `core/image-core/src/face_photo_transport.rs`.
A clean checkout needs local photo evidence and a renderer before replaying
photos. The unit-test target needs neither.

See `rust-controls/README.md` for target behavior and
`doc/func/FUNC-028-semantic-deformation-20261022.md` for acceptance limits.
