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

`source_regions.py --output <new-directory-under-test/results>` records source
region associations without modifying any target or image deformation. It binds
the historical MediaPipe canonical index order, the actual cached 36 coordinates,
eye/lip polygons, and the original face frame. Graph coordinates use normalized
positions times image size; the source audit uses photograph pixel indices. The
declared half-pixel conversion runs exactly once for graph points; source128 and
the frame are already in index coordinates and remain unchanged.

The output locates each side's eye-height and mouth-height intersections and a
chin-152 projection by continuous contour arc length. It rejects ambiguous,
crossing, missing, reordered or inconsistent source evidence. All associations
retain `visibility_status=unknown` and `automatic_edit_authorized=false`: detector
topology and proximity cannot establish visible skin behind hair or a bone
landmark. The script produces ignored source overlays for visual inspection;
it does not enable a new automatic edit, run detection, or call the solver.

`prepare_jaw_input.py --parent <original-source-input> --audit <source-audit.json>
--sidecar <source-regions.json> --review <source-review.json> --binary <jaw-arc-targets>
--output <new-directory>` prepares the independent `jaw_arc_v1` operation. The
first adapter supports the reviewed woman source's mouth-to-chin arcs only.
It rejects neck targets, accessory roles and nonzero position priors. Rust
targets are checked against an independent source-arclength polynomial. The
other 6,205 target rows, source frame, original Hessian, material roles, fixed
pixels, weights and background constraints remain unchanged for this source.

The frozen source review is a model's offline selection, not a visibility
detector: sidecar visibility stays unknown and automatic editing stays disabled.
`compare_jaw_results.py` reports actual arc-midpoint motion and outside-arc drift
beside source/whole-width/local-jaw images. Same slider number is not equal area
or exactly equal local displacement. See the jaw protocol and visual review
linked from FUNC-028; one development source does not establish beauty quality.

`test_jaw_input.py` uses synthetic inputs only. Optional `test_jaw_local.py`
checks ignored source evidence and Rust/Python parity. Set
`JAW_CONTROLLER_BINARY=/absolute/path/to/jaw-arc-targets` to test a binary in a
separate target directory without replacing hash-bound experimental binaries.

The local photo renderer used for historical comparisons is the existing binary
at `.run/face-current-20261019-target3/release/examples/face-photo-transport`.
Its source is in `core/image-core/examples/face-photo-transport.rs`, with the
standalone implementation in `core/image-core/src/face_photo_transport.rs`.
A clean checkout needs local photo evidence and a renderer before replaying
photos. The unit-test target needs neither.

## Portable source-field experiment

`portable_field.py` constructs a new source problem directly from image size,
source/target points, fixed source pixels, a 2px ROI and optional background
lines. It reuses the reduced solver, physical bending/membrane energies and
continuous inverse evaluation. It does not need historical native traces or
silently reuse another photograph's roles. Full fixed sampling neighborhoods
and a zero boundary collar are built into the field; output pixels are not
patched afterward. Fixed-domain, duplicate-target and hard-line conflicts are
explicit errors. This initial interface does not provide moving rigid materials.

`portable_jaw.py` uses the cached detector contour in its original row order
(133 points in the new bearded portrait). Canonical landmarks locate the edit's
mouth plane and chin projection; they do not replace the detector contour.
The beard contour is a geometric proxy, not a verified physical alpha boundary.
The first source-role rule protects the upper face, nose/lips, declared ear and
accessory coverage, and pixels beyond a bounded jaw neighborhood. Neck and
nearby background inside that neighborhood can deform continuously. Source
review remains explicit; this is not an automatic visibility detector.

`run_portable_jaw.py --evidence <cached-detector.json> --proof <source-proof.json>
--review <source-review.json> --binary <jaw-arc-targets> --output <new-directory>`
runs zero and both signed edits with source, final f32, pixel and target-response
checks. The independent numerical gate rejects missing/incorrect motion even
if fixed pixels stayed unchanged. Each active target (>0.5px) must reach
90–110% of its projected request and all target errors must be at most 0.25px;
these execution bounds do not supply a beauty score. Loaded local algorithm
sources are snapshotted with each new result.

`compare_portable_sampling.py --case <passing-case> --output <new-directory>`
reproduces bilinear pixels exactly before comparing bounded Lanczos3 on the
same f32 inverse. Geometry is not recomputed. It checks all fixed bytes and
alpha separately; the wider kernel still has the limitations described above.
See `doc/func/FUNC-029-portable-jaw-field.md` for actual photo evidence and limits.

`resolution_field.py` transfers a frozen inverse map between aligned isotropic
image sizes using pixel-center coordinates; it samples the original native image,
never an enlarged low-resolution edit. Native pixel nodes agree with the reference
map up to float32 rounding; native-cell reconstruction still needs inspection.
`run_resolution_jaw.py --reference-case <case> --native-case <case> --output <new>`
checks the same original image and contour request, final geometry, target execution,
exact fixed native displacement and fixed image bytes. The independent native
solve remains a comparison, not the output geometry.

The first transfer was rejected because independently rasterized protection masks
disagreed. `prepare_shared_protection.py --reference <run> --native <run> --output
<new>` conservatively unions both masks in reference coordinates. Pass the resulting
`protocol.json` as `run_portable_jaw.py --additional-protection` **before** solving a
new reference field. This keeps all original protection, including full interpolation
supports; no output pixel patch or displacement mask hides the conflict.

`jaw_collocation.py` evaluates the original jaw intent continuously along the
unchanged detector polyline. `--collocation-spacing 2` uses source arc-length
quadrature, with total target weight proportional to length instead of detector
point count. The runner checks separate midpoint probes at <=0.5px pitch even
without collocation: the old sparse user +100 now fails this check despite passing
at the original landmarks. This is a new target energy, not old-weight equivalence.
`--field-stride 1` retains all original grid variables; this resolved small-motion
response errors exposed by native-resolution checks in the man sample.

`source_boundary_support.py` constructs an explicit source-reviewed boundary band
before solving. The user-neck experiment preserved the selected pixels but introduced
a visible shoulder at the upper transition, and -100 failed final geometry. It is
retained as a rejected experiment, not an approved neck-preservation algorithm.

See `rust-controls/README.md` for target behavior and
`doc/func/FUNC-028-semantic-deformation-20261022.md` for acceptance limits.
