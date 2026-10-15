# Whole-face width V2 — causal hypothesis supported, image quality rejected

This experiment is isolated from production. Production remains v19 + physical-fringe-v2. No service deployment, UI integration, commit or push occurred here. V2 is a width-only curve prototype, not a learned beauty model, competitor preset bundle or completed “small face” feature.

## Hypothesis and intervention

V1 requested an entire face width change but moved mainly the lower face: same-head skin/ear/hair material boundaries were treated as fixed outer boundaries. V2 transports a single observed head + connected neck physical layer under one continuous field. The original skin contour still drives target geometry; background remains a separate reconstructed layer. No source-residual compositing was introduced. Full-head coverage already includes the user's earrings; no special rigid jewelry deformation is implemented.

`faceShape` must be active and observed head/skin/neck raster evidence present for this experimental path. Existing controls and zero shape retain their existing path. Geometric magnitude and the four curve profiles remain unvalidated design hypotheses inherited from V1.

## Ownership correction

The first causal candidate disabled semantic attachment constraints globally within the ROI. `first-global-owner-candidate-face.rs`, its frozen binary and `outputs/` preserve this rejected intermediate implementation. Root review correctly found insufficient instance ownership.

The current scoped candidate selects the connected head component touching the selected face's skin contour. Ear/hair/accessory components become associated only when connected to that selected head component. A parser-resolution owner fringe is distinct from physical alpha; disconnected confident head components are excluded even when nearby. Explicit user protection and other observed people remain fixed. External materials are context, not anatomical attachment constraints on this person's contour.

The corrected candidate is frozen in `scoped-owner-source-manifest.json` and `outputs-scoped-owner/user-face-contour/image-render-2355e919c57a3dbd30041108eeaa7f31f4e582e9254eb9452cad5f045baa0880`. Both ±30 output pixel hashes are **exactly identical** to the first causal candidate on this single-head photo, confirming that ownership correction is independent from the observed ear artifact. This does not prove separation when two semantic components erroneously touch; robust instance segmentation remains required in that case. Independent audit subsequently demonstrated this limitation with a hair bridge joining a second head inside the ROI: about 644/1056 second-head pixels changed under +100 (max RGB delta 152), while separated and explicit-protection controls remained exact. See `independent-owner-audit/report.json` and its frozen replay script. Thus scoped ownership is only a partial safety contract, not complete multi-person support.

## Actual results

The final owner audit also rebuilds `headOutline` from the head mask's 0.5 contour and makes skin/hair mutually exclusive inside the bridge. This consistent-input case changes 645/650 of the second head's 1056 pixels at +100/−100 (maximum RGB delta 152/156); see `independent-owner-audit/report-consistent.json`. The 644/649 counts in the earlier `report.json` refer to the earlier input, not this corrected-input replay. Both establish the unresolved instance-ownership failure.

Only the user's original image was run at `natural` 0/+30/−30. Detection inputs and original RGB were frozen; no redetection or cropping/resizing for rendering. Unannotated comparison panels are at original pixel scale, labeled source/V1/V2. The binary, graph and source hashes and timings are in `outputs-scoped-owner/user-face-contour/report.json`.

For +30, median inward displacement in pixels:

| Face band | Requested L/R | Actual V1 L/R | Actual V2 L/R |
|---|---:|---:|---:|
| Upper | 5.212 / 5.025 | 0.001 / 0 | 5.188 / 4.989 |
| Middle | 4.853 / 4.826 | 0 / 0 | 4.827 / 4.800 |
| Lower | 3.706 / 3.744 | 2.720 / 2.823 | 3.695 / 3.736 |

−30 correctly moves outward with comparable magnitudes. Zero is pixel-exact identity. The two original faceWidth ±30 legacy renders are pixel-identical to production references. V2 changes ~88,800–90,000 RGB pixels, including ~37,000 former hair/ear/accessory pixels; output image alpha remains exact. Those changes are evidence of action, not beauty quality.

**Rejected visually:** +30 produces a light rim and old dark line near the left outer ear; −30 produces an orange double edge there. Root and this experiment's author independently observed those defects at original scale. Whole-head movement has improved functional coverage, but external edge reconstruction is not acceptable and is not rated ≥8/10. Jewelry is transported with the head; its shape/material quality has not passed a dedicated independent review. The follow-up `audit_jewelry.py`/`jewelry-transport-audit.json` fit source-to-destination affine motion inside transported accessory coverage >0.8: +30 stretches the two earrings' principal horizontal dimensions by about 7.9%/8.4%, while −30 compresses them by about 8.1%/8.4%. Non-affine residual P95 is ≤0.30px; this near-affine deformation still fails to establish rigid jewelry preservation. Native CLI timings with debug artifact writes were 2.6–3.1 seconds per 1024px photo, not a browser interactive latency benchmark.

## Tests and preserved failures

- Two whole-head public contracts pass: same-head ears no longer pin the contour; explicit user protection and distant context are exact; a separate second head with hair inside the active ROI remains exact even if its disconnected component contaminates the broad face/subject masks.
- Three inherited public protocol/geometry contracts pass; three geometry unit tests pass. Eight targeted Rust tests total. No full library suite or WASM/browser integration is claimed.
- Two recorded-artifact tests pass: hashes, source/binary provenance, zero identity, ROI and alpha preservation, discarded/background pixels, finite alpha/coordinates, and reconstruction of the recorded layer composition within one byte.
- `whole-head-red.log`: old V1 pins the own-head ear contour, span 42..117 remains 42..117.
- `whole-head-first-green-failed.log`: first attempt gated mobility directly by head alpha >1/255 and still pinned the edge; ownership support and alpha coverage must differ.
- `ownership-red.log`: initial global owner candidate edits a second head at (123,88); scoped ownership fixes the counterexample without disabling the requested main-head edit.
- `artifact-first-test-failed.log`: initial diagnostic assertion wrongly expected identity source coordinates at discarded zero-coverage pixels. Such sampled coordinates are unused; the corrected public invariant checks that output pixels remain byte-exact instead.

## Trace format

For each nonzero amount, `outputs-scoped-owner/user-face-contour/natural-{amount}/debug/` contains:

- `layers.json`: crop `[left, top, width, height]`, branch flags and ownership/fixed counts.
- `head-semantic.png`, `skin-contour.png`, `head-component.png`, `owner-support.png`, `frozen.png`.
- `moving-physical-f32le.bin`: actual old physical alpha, row-major float32 little endian, not merely a display PNG.
- `source-xy-f32le.bin`: global pixel-center source sampling positions `[x,y]` for each destination crop pixel. Some positions were evaluated then discarded when both coverages were zero; gate use by `changed.png`.
- `target-alpha-f32le.bin`: sampled new alpha; zero is a placeholder when `changed=false`.
- `sample-premult-rgba-f32le.bin`: sampled premultiplied RGBA, four floats per crop pixel; RGB are 0..255 premultiplied and alpha is 0..1.
- `crop-source.png`, `foreground.png`, `estimated-background.png`, `reconstructed-background.png`, `changed.png`, `holes.png`.
- `requested.json`, `flow.json`: source/target and requested/feasible/actual contour controls.

Negative amount directory is `natural--30`. Existing face.rs comments describing face-only material behavior concern the legacy path; this experimental branch intentionally changes that material policy.

## Reproduce

Use `services/background-worker/.venv/bin/python` for NumPy/Pillow dependencies. `replay.py --name user-face-contour` refuses to overwrite its existing output directory. Build with an independent `CARGO_TARGET_DIR=.run/whole-face-shape-v2-target`. Review frozen outputs before expanding the image/style matrix or proposing integration.
