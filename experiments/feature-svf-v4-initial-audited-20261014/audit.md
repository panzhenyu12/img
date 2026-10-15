# Read-only semantic feature SVF audit

Production geometry.rs SHA256: c68070022e8b37e7d9ea250239a7caf305d50f6e536836cd00d1ab480821f9ad.
Source copied to this ignored directory; production source and public WASM were not edited. Only isolated test observers/cases were added. The first combinations test used feature_fixture() dimensions 192x192, not the default 82 matrix's 256x256. Its first varying-alpha identity comparator assumed width256 and is INVALID; alpha-result.json is the corrected width-aware evidence and supersedes that row.

## Confirmed findings

1. eyeTilt +/-100 alone introduces unintended center magnification. Actual final mapper h=0.1px derivatives at both detected eye centers give inverse Jacobian approx0.91455 for both signs, so forward local area is approx1.09343. eyeSize/eyeWidth/eyeHeight remain zero. geometry.rs:833-839 imports finite rotation displacement into stationary velocity; (cos(a)-1)I gives isotropic contraction. The correct infinitesimal rotation generator has no such radial term. Evidence: tilt-result.json. Existing positive-Jacobian regression does not enforce semantic area independence.

2. Frozen transparent RGBA is not exact. Public apply_graph_json on min-source.png/min-graph.json converts semantic-frozen pixel(26,26) from [88,177,233,0] to[0,0,0,0]. geometry.rs:1497 returns identity, remap:72 still resamples it, sample:54 discards hidden RGB at alpha0, and final composite does not enforce semantic-copying. This does not change visible current pixels, but violates literal fixed RGBA contract. Evidence: min-result.json; minimal regression intentionally FAILS. A corrected gradient-alpha192x192 probe reports191 identity mapping hidden RGB changes;62 alpha changes inside moving geometry are normal resampling, not reported as a bug.

## No reproduced old field switch

Semantic feature.adjust always uses integrated flow at1517 and bypasses old1592 donor-return-identity. Stationary/region early returns sit in the domain's zero-speed guard, so no former identity-cut failure was reproduced. Eight additional symmetry/eye/brow and opposite-sign combinations have no observed fold in the existing pixel+h0.25 observer. Five new symmetry/family0->0.001 probes have max map delta<=0.000245px and byte delta<=1. These are additional bounded probes, not a universal invertibility/effect claim.

## Test scope gap

Default82 matrix does not include symmetry; existing tiny-family regression has4 transitions and excludes symmetry. Alpha checks use uniform211 and therefore cannot detect variable-alpha behavior or transparent hidden-RGB exactness. Mapping QA observes the pre-composite mapper; final graph opacity/softmask mixing is an image composite, not a single geometric map.

Minimal failure reproduction:

```sh
MIN_AUDIT_DIR=/Users/modao/work/img/.run/feature-main-audit-20261014 cargo test --manifest-path .run/feature-main-audit-20261014/Cargo.toml --lib independent_minimal_frozen_hidden_rgb_repro -- --nocapture
```

Tilt probe:

```sh
TILT_AUDIT_OUT=/Users/modao/work/img/.run/feature-main-audit-20261014/tilt-result.json cargo test --manifest-path .run/feature-main-audit-20261014/Cargo.toml --lib independent_eye_tilt_area_audit -- --nocapture
```
