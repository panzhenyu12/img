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

`--correct-previous <rejected-case>` performs one 12-second-limited solve after
mapping actual failing inverse cells back to their complete source-cell
enclosures. It retains all previous active constraints and adds a one-cell halo,
with unchanged basis, objective, strength and source roles. The previous case's
input, matrices, coefficients and fields are verified before use. A further call
needs newly added source constraints; when failures are absent or already fully
covered, it rejects a repeated identical solve. It does not run an unbounded retry loop.
The 12-second limit covers Clarabel solving, not assembly, input audits or image
evaluation. All actual inverse, RGB and line checks still run afterward.

`rust-controls/prepare_source_forward_input.py` substitutes a face controller at
the **same signed amount** as its frozen parent. A different amount must use a
matching parent input, preserving neck and earring motion. Copying +30 attachment
targets into a nominal +60 request is rejected even if all hashes are updated.

`prepare_hair_inputs.py --output <new-directory>` constructs the third local
photograph's signed inputs from its own recorded +60 and -60 baselines. Its
18 neck targets retain the same signed amount as the face. Replay of this local
evidence adapter requires **Pillow 11.1.0** and the ignored source corpus;
the synthetic signed-neck tests do not need those files or that Pillow version.
Because the historical RGB audit referenced by its provenance is missing, it
explicitly freezes new source polygon masks bound to a later existing same-photo
audit. It never presents these masks as recovered historical ground truth.

The local photo renderer used for historical comparisons is the existing binary
at `.run/face-current-20261019-target3/release/examples/face-photo-transport`.
Its source is in `core/image-core/examples/face-photo-transport.rs`, with the
standalone implementation in `core/image-core/src/face_photo_transport.rs`.
A clean checkout needs local photo evidence and a renderer before replaying
photos. The unit-test target needs neither.

See `rust-controls/README.md` for target behavior and
`doc/func/FUNC-028-semantic-deformation-20261022.md` for acceptance limits.
