# Semantic deformation controls: isolated source-target experiment

Date: 2026-10-22. Scope: whole-face width **target generation only**. No automatic
detection, dense deformation solver, resampling, image completion, production
integration or eight-point quality acceptance is supplied by this crate.

The public seam is `generate(&Request) -> Result<Output, String>` and a JSON stdin /
stdout binary. All coordinates are in the **original source image**, never a
previously edited result. Only `serde` / `serde_json` are dependencies.

```sh
cd experiments/semantic-deformation-20261022/rust-controls
cargo test --offline
cargo run --offline --quiet --bin face-width-targets < examples/whole-width.json
```

CLI success writes JSON on stdout. Invalid input writes a JSON error on stderr,
returns exit code 2, and emits no result. Unknown request fields are rejected,
including unknown motion fields, to catch misspelled protection rules.

## Source-relative whole-face controller

`frame.origin`, `across`, `down` define an orthonormal image-coordinate frame.
`bounds` are `[left, top, right, bottom]` **in that local frame**; `midline_u` is an
explicit face midline, not a guessed average of left and right bounds. The frame
vectors must be unit, perpendicular and have determinant +1. `contour` is any
set of at least three named control points; there is no required ordering, sample
count, detected oval or eye count.

Frame validation permits `1e-6` floating-point error in squared norms,
orthogonality and determinant, accommodating the actual cached native f32 frames.
The frame is never silently normalized. JSON parsing uses serde_json's
`float_roundtrip` option so original f64 source coordinates survive a zero edit.

For a source point `p`, let `u = dot(p-origin, across)`,
`h = (dot(p-origin, down)-top)/(bottom-top)`, and `r = u-midline_u`.

```text
t = clamp((h-chin_fade_start)/(1-chin_fade_start), 0, 1)
chin_envelope = 1 - t*t*(3-2*t)
displacement = -amount/100 * max_radius_fraction * r * chin_envelope * across
target = source + displacement
```

Positive `amount` narrows; negative widens. Both are independently evaluated
against the source, not sequential inverses. Zero returns the source coordinates
unchanged. Upper face and cheeks share the same source-radius scale; only the
lower chin envelope tapers. At the chin bottom displacement is zero and the
envelope's first derivative is zero. These target properties do not establish
continuity or injectivity of a downstream image warp.

`calibration.max_radius_fraction` is mandatory and limited to `[0,0.45]` to bound
the requested radial change. `amount` is in `[-100,100]`; `chin_fade_start` is in
`[0,1)`. These ranges are explicit experiment limits, **not validated anatomical
or beauty limits**. The fixture's `0.2` is an arithmetic example, not a recommended
product setting. At +60 it requests 12% radius reduction above the chin fade.

The formula derives from the source-relative upper-face correction in
`experiments/face-user-source-roles-20261021/whole_face_width.py`. The caller must
supply calibration; this crate does not redo its polygon intersection or cheek
normalization. To match the earlier cheek target, provide its positive full-scale
`k` as `max_radius_fraction` and the same midline. This version omits the old
near-axis lateral gate: boundary points in its old plateau agree, while interior
points deliberately use explicit part rules. It is a **controller variant**, not
a bit-identical port or a newly validated research algorithm.

## Operation-specific part motion

Each part requires `id`, descriptive `label`, `source_points` and an explicit
`motion`. Labels never select motion automatically; even two parts labelled
`eye` can have different roles for different editing operations.

| Input motion | Generated target / rule |
|---|---|
| `{"kind":"fixed_position"}` | Every source point remains at its source position. |
| `{"kind":"shape_preserving_translation","anchor":[x,y]}` | Evaluate the controller once at the source anchor, then apply that same translation to every point. Position changes; pairwise source offsets remain unchanged. |
| `{"kind":"editable"}` | Evaluate the controller at each point independently. |

Output contains `targets` (`id`, optional `part_id`, `source`, `target`) and
`part_rules` (label plus resolved motion, including shared translation). There is
no automatic strength, hard/soft priority or inferred fixed background. A dense
solver must respect the **whole part rule** over the actual material; protecting
a few points alone does not guarantee texture protection between them.

## Connections and occlusion are explicit evidence

Optional `relations` refer to two declared parts. Supported shapes are:

```json
{
  "kind": "connected",
  "a": "face",
  "b": "neck",
  "contacts": [{"on_a": [50, 180], "on_b": [50, 180]}],
  "evidence": {
    "provenance": "manual contact annotation",
    "status": "user_confirmed",
    "confidence": 1.0
  }
}
```

```json
{
  "kind": "occludes",
  "front": "hair",
  "back": "face",
  "boundary": [[20, 30], [25, 100]],
  "evidence": {
    "provenance": "segmentation boundary; depth relation not independently confirmed",
    "status": "hypothesis",
    "confidence": 0.6
  }
}
```

`user_confirmed` is an explicit caller assertion, not a model confidence threshold.
Relations are validated and passed through **without enforcing** attachment,
occlusion or changing motion. A class boundary never automatically becomes a
physical seam, depth order, fixed position or ground truth. The current module
does not resolve inconsistent target/connection sets, infer disocclusion masks,
or detect duplicate ownership of material. Those are downstream work.

## Validation record

Three vertical red→green slices were observed: upper contour stayed unchanged
before the field was implemented; part targets were absent before operation
roles were implemented; out-of-range input was accepted before validation.
Additional regression checks cover signed source symmetry, chin transition,
rotation/translation/scale equivariance, explicit relation hypotheses, and the
real JSON CLI. These tests establish target-generation behavior only. Photo
quality, preservation of full visible material, cross-photo generalization,
and end-to-end response time remain unvalidated by this crate.

The follow-up [three-source target audit](source-audit.md) checks user / woman /
hair source conventions and measures the intentional differences from frozen
whole-face-width targets. It contains no rendered edits or quality score.

## Preparing a source-forward input

`prepare_source_forward_input.py` creates an independent solver input from the
frozen user `+30/input` ledger. It verifies the source photo, user frame,
explicit `midline_u=0`, and the first 128 source rows before invoking the Rust
target CLI. Only those 128 destinations/deltas change; material supports,
weights, locks, and background constraints are copied. The target RHS and
Hessian are rebuilt, and `signed-input-audit.json` records fresh `-30/0/+30`
source requests. The script stops before solving or rendering:

```sh
services/background-worker/.venv/bin/python \
  experiments/semantic-deformation-20261022/rust-controls/prepare_source_forward_input.py \
  --output test/results/semantic-deformation-20261022/user-plus30-rust-controller-input-v1
```

The output directory is local experiment evidence and should remain ignored;
it is not an image-quality result.
