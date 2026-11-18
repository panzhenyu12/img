# FUNC-055 Shared skin/accessory motion (isolated 2026-11-10)

## 2026-11-17 source attachment permission before ear connectivity

The new Lupita holdout exposed an ordering defect: native source preparation
froze the observed stud, then classified the earlobe behind it as disconnected.
Releasing the stud only in the later field solve left 6 conflicting fixed grid
nodes. Enlarging its mask was not a valid repair.

Research replay now accepts `sourceMovingAttachmentSupport` (a path to a
full-source 0/255 raster), injected as private
`__sourceMovingAttachmentSupport` after public graph validation. The same
source-only plan is consumed by the Python field/cache path. Native uses it
before automatic accessory freezing and permits connectivity through the
accessory class gap without inserting the permission into `joint_alpha` or
physical alpha. Moving attachment pixels remain excluded from background
donors, but are not pinned as stationary backdrop reconstruction pixels.
User protection, garment, hair and facial-feature conflicts reject the plan.
There is no public schema change or new default behavior.

A focused native behavior regression reproduces the old disconnected-ear
freeze, verifies the connected result with the source plan, keeps zero plan
output exact, checks zero alpha in the semantic class gap, verifies donor roles,
and rejects the four protected-material conflict types. It passed together
with the 8 existing source-ear support tests using one targeted rustc build.
Independent review found no blocking permission flaw for this research route.
Rust trusts the plan's provenance; the upstream automatic component planner
and its source hash checks remain required. This is not an arbitrary public
user permission mask. Soft hair below .5 remains a conservative later barrier.

On the unchanged Lupita +30 native replay, connected ear core pixels increased
from 1814 to all 1837; frozen ear cores fell from 4 to 0. The 744 planned stud
pixels have zero scene/output/donor freezes and all 744 remain excluded as
background donors. The extra 25 released pixels outside the stud plan all carry
original ear evidence. Raw source ear evidence is byte-identical. These are
permission and connectivity results; final field and image acceptance are
separate. Local provenance and reports are under
`test/results/face-usable-20261112/motion-interface-20261116/native-attachment-source-plan/`
and `holdout-lupita-romano/native-source-attachment-plan/` in that experiment
root. No photos, masks, or binaries are included in this document.


## Purpose

Prevent a face-width edit from stretching the background while leaving an
attached earring behind. The moving face skin and each reviewed accessory use
one continuous source-forward bounded field. The accessory stays rigid and
shares one optimized translation with a small source skin-contact set; it is not pasted
as a second image layer. The observed ear remains flexible.

This document records an algorithm experiment and its evidence. It does not
claim that the full portrait editor is production-ready or that the project as
a whole scores 8/10.

## Source evidence and ownership

The selected 768px accessory alpha is a neural candidate from the reviewed
source graph (`official-accessory-alpha-768.npy`). It is material evidence, not
certified physical opacity. Geometry support and optical alpha stay separate.
The source joint alpha is `1-(1-skin)*(1-accessory)`; the source background is
reconstructed once for that union, then paired and sampled once. The same
source background is used for source residual identity checks and destination
composition. A signed source-only local prior may replace the paired B in an
explicit ownership interval, without a second blend or a second sample.

Stationary donor roles (`donorFrozen`, `donorExcluded`, `knownLayer`) are
carried from the native source replay. Releasing an accessory from output
freeze does not make it an eligible scene donor. Explicit protection conflicts
reject the solve. A synthetic metal-pixel protection regression is retained in
the experiment output.

## Geometry

`geometry.py` reuses the bounded bilinear solver. Four nearest source skin
pixels above each accessory hook define a reviewed attachment hypothesis. The
material and contact points form one rigid group with a soft target weight;
every enclosing grid cell is swept by a one-pixel optical fringe. This avoids
the old independent ear-pivot and delta-composite approach.

`export_field.py` computes the inverse by fixed point from the bounded
source-forward map. The ±60 fields have inverse closure below `2e-11px` before
float32 export, determinant minima about `.758`, and max component derivative
`.24`. Raw material and contacts translate to numerical precision.

An optional `material_gradient_bounds` field in the isolated solver applies a
componentwise derivative bound to source material cells while retaining a
flexible field. With source ear support and limit `.10`, ±100 response falls to
`.587/.601` and local singular lower bounds are about `.858`; the ear is less
pointed than the unconstrained `.24` result. This is a controlled strong-edit
tradeoff, not a justification for unrestricted ±100 editing. The bound alone
does not remove the +100 ear-side yellow/green low-frequency patch: source
background ownership and scene prior remain independent problems.

## Source-layer replay

`build_renderer.py` compiles a tiny `rustc` harness from the current
`face_matting`, `face_optical_domain`, `face_scene_edge`, `face_layer_pair`, and
the exact `neighbors`/`reconstruct_source_background` functions in `face.rs`.
It does not build Cargo or the application. `render.py` writes a joint source
cache, and `sample_cached.rs` reuses that immutable cache for independently
solved fields. Baseline alpha, paired B and RGB match native +60 byte for byte.
Combined ear + earring source-prior output also matches its independent oracle
byte for byte. Cached sampling and full source reconstruction match byte for
byte.

The α-min source experiment raises alpha only where a fixed neural B is
physically byte-feasible and the source material already has nonzero support;
it is not opacity ground truth. Across the tested sweep it removes float gamut
violations while preserving source identity, frozen pixels and zero-edit
identity. A negative test demonstrates that a wrong B or missing soft band can
still fail; α-min is not a universal ownership repair.

## Current tested range

The exact current Rust `target_contour` function was extracted and independently
evaluated for `faceWidth` values `-100,-30,-.001,0,.001,30,100`; each request is
solved separately and rendered at native size. Zero and ±.001 are byte-identical
to the source. ±30 has response about `.9995`, smooth bounded geometry, and
visually connected earrings. Unrestricted ±100 has weaker response (`.752/.833`)
and visible ear over-deformation. The `.10` source-ear bound improves ear shape
but reduces response (`.587/.601`). The practical candidate range is therefore
the ordinary ±30/±60 interaction range until a separate strong-edit model is
reviewed. The `.10` local bound keeps ±30 response about `.9991` in the isolated
replay but remains only a geometry candidate; its source-layer output must pass
the same visual review.

The actual bounded solve takes about `1.25–3.22s` including process I/O in the
isolated macOS runner for this 472×544 crop; one cached Rust sample takes about
`16–23ms` after a `22–40ms` source-identity check. These figures are algorithm
evidence, not browser or WebAssembly latency claims.

## Reproduction

`experiments/face-attachment-motion-20261110/run.py` is the single native entry
point. It takes the original photo, graph, trace, prepared source cache and
existing solver/sampler/target binaries; evaluates actual Rust width targets;
and writes PNGs plus a source SHA256/parameter/timing manifest. It supports
absent accessories as empty groups, explicit attachment requirements, and the
optional source ear derivative bound. The experiment README has a complete
local command. Single-entry -30/0/+30/+60 outputs are byte-identical to the
previous independent replay. It does not download models or copy source photos.

The initial timing above is a quiet-run observation. A later concurrent CLI
run measured 4.5–8s solve, 65–75ms sampling, and 19–28s end-to-end per value
including Python source-mask preparation, inverse export and file I/O. This
runner is currently an offline quality/reproduction tool, not a real-time
browser implementation.

All generated images, masks, traces and binaries remain outside Git under
`/tmp/face-attachment-motion-20261110`. The tracked source is the experiment
directory itself. Typical commands:

```text
python experiments/face-attachment-motion-20261110/build_renderer.py --out /tmp/face-attachment-motion-20261110/renderer
python experiments/face-attachment-motion-20261110/geometry.py ...
python experiments/face-attachment-motion-20261110/export_field.py ...
python experiments/face-attachment-motion-20261110/render.py ...
```

No experiment output is a substitute for a visual review on multiple source
photos. The remaining risks are source ownership of dark/reflective accessory
edges, strong ear deformation, and generalization beyond this reviewed portrait.
