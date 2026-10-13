# Float harmonic narrow-strip candidate

Status: **not enabled in production**. Production retains the v16 PreparedMask/native F/B cache changes and its prior strip reconstruction.

`harmonic-v17.patch` replaces repeated 8-bit, eight-step diffusion with a premultiplied float RGBA harmonic field, red/black SOR (omega 1.65), a 0.025 channel-update stop and a 384-iteration cap. Fixed observed context and protected donors remain unchanged. The patch also carries two focused regression tests.

A ten-pixel strip in a known affine background had a maximum error of 13 with the original method; the candidate passes a maximum-error <= 1 check. Hidden RGB in transparent donors no longer influences the reconstructed strip. All 28 body-filtered native tests passed for the candidate.

Real A/B replay: five photographs, fifteen parts, four strengths, 60 cases. Contracts passed, but normal-scale review did not show a consistent visual improvement in foliage, ground lines, garment contact boundaries or the existing soft-edge seams. This candidate does not infer missing structured texture. Keep it as an experiment until a more capable reconstruction solves the visible failures.

Evidence: [real review](../../doc/qa/body-harmonic-v17-review.md), [frozen replay](../../doc/qa/body-harmonic-v17/report.json).

To reproduce from a matching source snapshot, apply the patch to an isolated checkout, build a native renderer, freeze its binary, and run:

```sh
services/background-worker/.venv/bin/python scripts/review_body_refinement.py \
  --before /absolute/path/to/frozen-v16-renderer \
  --after /absolute/path/to/frozen-harmonic-renderer \
  --label body-harmonic-recheck
```

The `test/results/body-parts-v14` cached inputs are required. No new detection is performed by this replay, and pixel contracts do not imply an 8/10 visual result.
