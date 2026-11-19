# Source attachment permission

This small source-preparation tool distinguishes fixed eyewear from candidate moving earrings using an existing, source-bound 19-class posterior. It does not render a photograph, run a detector, download a model, or import the historical deformation solver.

The standalone implementation consists of `source_masks.py`, `attachment_source_plan.py`, `attachment_subtypes.py`, `prepare_attachment_source_plan.py`, `consistent_detail_control.py`, and `apply_consistent_source_roles.py`. The two `test_attachment_*.py` files verify the contract. Dependencies are Python, NumPy, SciPy and Pillow.

```sh
python -m unittest test_attachment_source_plan test_attachment_subtypes
python prepare_attachment_source_plan.py \
  --source /path/to/canonical-source.png \
  --graph /path/to/source-role-graph.json \
  --subtype-cache /path/to/exact-source-posterior-cache \
  --out /path/to/new-output-directory
```

The posterior cache must contain `report.json`, `raw-source-role-sidecar.json`, and the selected `.npz` referenced by its provenance. It must bind the same source image, source detection graph, frozen parser model, selected crop, normalized 19-class probabilities, and unchanged current accessory mask. Missing or mismatched evidence is rejected. It is not enough to supply an unbound pair of guessed masks.

The output contains the unchanged graph plus an explicit binary movement permission, `support.json`, and a frozen `partition.json` with the source/graph/model/posterior/implementation hashes and complete component dispositions. The source-rebuild admission function is `attachment_subtypes.attachment_release_from_source`; it verifies the same partition and keeps fixed instances out of downstream release. It never edits the semantic accessory mask or infers optical alpha.

Eyewear is fixed for the current face-width contract. Only a whole, observed earring component with no independent protection conflict receives movement permission. Mixed or unobserved components remain fixed and unresolved. Class evidence and connected components are estimates, not guaranteed accessory-instance truth. A weak disconnected frame fragment cannot be inferred from another component.

Historical `face_pipeline.py`, native source-estimation harnesses, head matting, scene completion, and field-solver adapters in this research directory have their own dependencies. They are not imported by this standalone delivery and are not prerequisites for its tests. A prepared attachment permission alone is not a working portrait editor; final image quality must be validated through the ordinary image-core API.

Implementation details and actual acceptance limits are recorded in `doc/func/FUNC-081-source-attachment-subtypes.md` at the repository root. Local photographs, posterior caches, model weights, binaries and rendered results are excluded from the source delivery.
