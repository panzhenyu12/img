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

`run_resampling.py --case <geometry-passing-photo-case> --output <new-directory>`
isolates reconstruction quality on the **unchanged** float32 inverse. Its Python
bilinear baseline must reproduce the existing Rust photograph byte for byte.
The candidate uses bounded Lanczos3 in opaque interiors; surrounding 2x2 sample
limits suppress overshoot, with bilinear alpha and boundary/transparency fallback.
No fixed-pixel repair is applied. This is a separate sampling contract: the
historical bilinear material oracle is not treated as its acceptance test.
The 6x6 footprint, encoded RGB domain, potential loss of extrema, and unresolved
anisotropic minification remain limitations. This established reconstruction
method is a focused comparison, not a claim of a new or state-of-the-art algorithm.

The Rust controller's optional `regional_profile` creates an explicit new
`regional_contour_v1` edit. `prepare_regional_profiles.py` builds the first
source-height hypothesis and a uniform control matching its requested polygon
area. The uniform control is explicitly weaker than the original whole-width
operation. It does not silently reinterpret an old strength as a new target.

`prepare_regional_input.py --parent <source-input> --audit <source-audit.json>
--profile <profile.json> --output <new-directory>` binds the new controller to
an original woman/hair source input. It regenerates 128 face targets and scales
each declared neck arc from the same profile at its source entry, retaining the
original arc fade. Nonzero accessory position priors and unsupported material
roles are rejected. Original mappings, fixed pixels, material, weights, Hessian,
and background equalities remain unchanged. An independent SciPy PCHIP reference
checks Rust face targets and the neck rule before the unchanged solver runs.
This adapter deliberately rejects the signed-user format and nested variants.
See `doc/research/regional-face-target-experiment-20261022.md` for the exploratory
comparison and its limitations. The cached source polygons are not anatomical
ground truth or prescribed beauty ratios.

The local photo renderer used for historical comparisons is the existing binary
at `.run/face-current-20261019-target3/release/examples/face-photo-transport`.
Its source is in `core/image-core/examples/face-photo-transport.rs`, with the
standalone implementation in `core/image-core/src/face_photo_transport.rs`.
A clean checkout needs local photo evidence and a renderer before replaying
photos. The unit-test target needs neither.

See `rust-controls/README.md` for target behavior and
`doc/func/FUNC-028-semantic-deformation-20261022.md` for acceptance limits.
