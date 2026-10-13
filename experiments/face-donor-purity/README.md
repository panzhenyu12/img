# Face donor purity experiments

These patches are isolated quality experiments. They are not applied to production automatically.

- `pure-donor-v1.patch`: .95 hair/neck donor confidence and a 2px same-material erosion. This changes the classification front as well as its color supply. Rejected after 8 real-image cases: protection passed, but stronger dark jagged patches appeared. See [v1 review](../../doc/qa/face-donor-purity-v1/review.md).
- `pure-donor-v2-fixed-classes.patch`: control experiment preserving v9 classification and reconstruction topology, refining only weak observed colors within their existing layer. Rejected after the same 8 cases: deeper jagged patches remained without a net visual gain. See [v2 review](../../doc/qa/face-donor-purity-v2/review.md).

`provenance.json` identifies the frozen v1 candidate. Source patches apply to the v9 `face.rs` snapshot identified by `source-face-v9.sha256`. A later combined candidate must be rebuilt and assessed again; separate good metrics do not establish combined visual quality.
