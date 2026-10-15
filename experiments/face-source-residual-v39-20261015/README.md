# V39 opaque joint source-residual component — rejected

**Final status: rejected for integration and deployment.** The change passes a public near-zero-edit continuity regression and opacity/protection/RGBA checks, but real photographs preserve old face-edge skin/orange lines. Two reviewers independently reject the visual regression. No ≥8/10 claim is supported.

Only the isolated `src/face.rs` differs from the current production snapshot: 12 added lines and one replaced line, applying source residual composition for joint face/neck with fully opaque crop RGBA. Geometry, alpha refinement, model inputs, and public graph fields remain the same for the component comparison. Production was not changed by this experiment.

## Evidence

- `outputs/`: parent run of three source photographs at 0/±30/±100, current and candidate PNGs. See `manifest.json` and case reports for exact sources/graphs.
- `outputs/user-face-contour/-100/face-1x.png`: candidate retains old contour bright lines within the expanded lower cheek.
- `outputs/portrait-hair/-100/face-1x.png`: candidate adds a clear orange line at the right cheek/neck edge.
- User/hair +100 also retain old source contour fragments; woman ±100 is broadly similar and does not offset these regressions.
- `tests/source_residual_contract.rs`: one public `apply_graph_json` regression. A tiny ±0.001 faceWidth edit must not jump by >1 byte; a strong +100 edit must still visibly narrow both sides by ≥3 px. Old snapshot fails with a 7-byte jump, candidate passes. Logs: `old-core-audit/red.log`, `source-residual-green.log`. The old source snapshot and SHA ledger are retained.
- `public-options-v2/`: 48 public native CLI outputs. Normal/multiply × opacity 0/50/100, plus protect 128/254/255, on opaque and transparent inputs. All assertions pass; 12 transparent input cases are byte-identical to the old implementation. Half protection still moves the opaque contour: source span `[42,117]`, half-protected `[45,114]`, full edit `[47,112]` on the synthetic center row.
- `public-options/`: earlier 40-output probe kept for history; v2 adds nontrivial half protection after the near-full protection cases yielded no changes.

The original source residual can contain real old-edge skin omitted from the estimated foreground. Keeping it stationary improves the near-zero photometric limit but creates unacceptable ghost edges under strong edits. Contract tests are necessary evidence, not a substitute for original-resolution photo inspection.

Independent report: `doc/qa/face-v39-source-residual-contract-review-20261015.md`. Do not merge this candidate on the strength of test counts or the low-alpha bound.
