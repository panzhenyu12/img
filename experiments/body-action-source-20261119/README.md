# Fresh automatic action-suit body test

This independent original-photo test evaluates visible suited-torso contours, a held laptop, hands and detached sleeves. It does not claim to observe the hidden anatomical waist. It never reads old prepared layers, manually changes source ownership, reduces strength, or downloads a model.

`source_scope.py` freezes review-only source arcs/crops/probe boxes before detection. `detect_source.mjs` uses current `detectPeople` once. A read-only listener captures the same FASHN worker result and saves the raw `1×18×144×96` NCHW float32 logits plus the actual decoded, rounded u8 class masks. `decodeBodyLogits` is evaluated again only to save that same message's class evidence; there is no second inference. The worker and model implementation remain unchanged, and the temporary browser/server/module cache are closed afterward.

```sh
services/background-worker/.venv/bin/python experiments/body-action-source-20261119/source_scope.py /tmp/body-action-fresh
node experiments/body-action-source-20261119/detect_source.mjs test/images/action-suit.jpg /tmp/body-action-fresh/action-suit waist
services/background-worker/.venv/bin/python experiments/body-action-source-20261119/replay.py /tmp/body-action-fresh/action-suit
services/background-worker/.venv/bin/python experiments/body-action-source-20261119/diagnose.py /tmp/body-action-fresh/action-suit
```

Existing local Web detection weights, Playwright, Vite, Python NumPy/Pillow and current native `image-render` / `body-layered-replay` binaries are required. Scripts do not build them. `replay.py` runs the ordinary `0/±30` graph. `diagnose.py` uses the public field-only API and the same-inference class evidence to separate source ownership/geometry failure from compositing artifacts. Its narrow bands around the original frozen arcs are post-replay diagnostic scopes, not physical material truth or preregistered alpha labels.

The initial automatic result preserved the held laptop, hands and sleeve probes but did **not move either preselected visible jacket side**; internal fabric moved instead. This was an effect failure, not a pass merely because the image had no obvious crack. A later source-only automatic candidate and independent-side core repair now produce a smooth bidirectional visible garment response on this image at 0/±30; this bounded result is not general automatic-body usability. See `doc/qa/body-action-source-failure-20261119.md` and `geometry-rebuild.md` for the failure chain, final evidence and shared geometry rebuild entry point. Photographs, logits, arrays and output images stay out of Git.

## Source-only automatic candidate replay

`rebuild_graph.mjs` reads a source-bound torso geometric mask and directly calls current public `bodyParts` and `measuredTorsoSupport` through an isolated Vite page. It recomputes the support polygon and its bounding region using the exact existing private `regionForPolygon` rule; it refuses missing support. It preserves source Pose landmarks, person coverage, original protection, strength and graph policy. The mask input must be `{ "sourceSHA256": "…", "mask": RasterMask }`; unbound bare masks are rejected. Rebuilding from the original torso mask produced the exact same graph file SHA as the frozen ordinary baseline.

```sh
node experiments/body-action-source-20261119/rebuild_graph.mjs /tmp/body-action-fresh/action-suit /tmp/automatic-torso-mask.json /tmp/body-action-fresh/action-suit/automatic-support
services/background-worker/.venv/bin/python experiments/body-action-source-20261119/replay.py /tmp/body-action-fresh/action-suit --graph /tmp/body-action-fresh/action-suit/automatic-support/source-graph.json --output-name automatic-replay
```

The replay validates the candidate's source/graph binding and rejects changes beyond `partMask`, support polygon points and region. This is an algorithm experiment only; no UI or detector integration is performed. The source candidate must be assessed before its rendered result is used to judge effect.
