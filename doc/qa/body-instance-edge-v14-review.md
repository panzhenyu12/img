# Body instance edge recovery v14

## Scope

The browser body parser combines a FASHN semantic mask with a MediaPipe Pose
instance mask. The instance mask is lower resolution and can miss one real
silhouette pixel at the garment boundary. Before v14 that pixel became a frozen
background hole after contraction. v14 recovers only a one-pixel uncertainty
band from the instance raster before the semantic foreground is calculated:

```text
instance(x,y) = max(instance(x+dx/instanceWidth,y+dy/instanceHeight))
for dx,dy in {-1,0,1}² when center instance < 0.6
```

FASHN class probabilities and the protected-class mask still decide ownership;
a gap wider than the one-pixel band remains background. This keeps the earlier
single-person instance protection that removes remote garment-class islands.

## Verification

- `apps/web/tests/body-parts.spec.ts`: 3/3 Playwright tests passed. The new
  synthetic contract verifies an edge one pixel outside the instance is
  recovered while a two-pixel gap remains zero; hands and bags remain protected.
- `npm run build` in `apps/web`: passed.
- `cargo test --release --manifest-path core/image-core/Cargo.toml --lib body`:
  25/25 tests passed. Rust rendering code was unchanged for this candidate.
- Frozen native renderer: SHA-256
  `515ed394104c4b86d17501fbf21dde38c3b9e3bc354bbabefa8c3ef4c321e544`.
- Real photographs: 15 body-part graphs across five photographs replayed at
  `-60, -30, +30, +60` (60 renders). Full provenance and timings are in
  `body-strength-v15/metrics.json`; no render changed a protected or known
  background pixel in the v14 contract replay.

## Visual result

The fashion-standing right-thigh boundary no longer leaves the small dark
island seen in v11. Sleeve, waist, and thigh positive/negative sweeps remain
symmetric at medium strength. The change improves boundary ownership and does
not by itself solve all garment/background seam quality; the body feature should
continue to be visually reviewed before assigning an 8/10 quality score.
