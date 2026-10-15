# Alpha correction cap causal audit

2026-10-15. Isolated experiment requested by the parent task. It reuses the three existing corrected-prompt source decompositions, fixed parser cores, donor colors, solve domain, original raw alpha, exact inverse maps and existing A/B layer-order compositor. It performs no new model inference, geometry search, production edit or deployment.

## Question

The current `decompose.py` photometric alpha refinement constrains each mixed pixel to `rawAlpha ± .15`. The hypothesis was that this cap preserves a wrong alpha and creates the woman +100 orange boundary. The controlled comparison changes only this rule:

- `bounded-015`: current implementation, reproduced byte-for-byte.
- `uncapped`: photometric estimate clipped only to physical `[.001,.995]`.
- `donor-interval`: a local diagonal donor-scatter interval; incompatible intervals abstain and retain raw alpha. This is a diagnostic confidence model, not a calibrated confidence guarantee.

All three then use the same RGB solver, the same A/B order hypotheses, the same write gate, same neural or V36 backing and same inverse source coordinates. `compose_ab.py` records the pixel differences against bounded output and proves write domains remain equal for bounded versus uncapped.

## Result

On the existing source images, uncapped fitting lowers source reconstruction error:

| photo | bounded mean byte | uncapped mean byte | pixels touching cap | uncapped A/B output changes |
|---|---:|---:|---:|---:|
| portrait-woman | 0.4380 | 0.1715 | 3,620 | 2,798 / 3,035 (+100, V36) |
| user-face-contour | 0.1292 | 0.1138 | 93 | 0 | 0 / 0 |
| portrait-hair | 0.5029 | 0.3401 | 437 | 565 / 472 (+100, V36) |

The lower source fit does not prove a correct hidden layer. The exact known-background synthetic counterexample `fine-hair-wrong-donor` has real fine hair alpha .65 whose observed RGB lies between the opaque-hair donor and skin donor. Uncapped fitting lowers source error `0.6822→0.0114` byte but changes mean fine-hair alpha `.500→.364`; known new-backing error rises `33.87→43.67` byte. The donor-interval variant also erases this fine hair. A low-separation noisy donor case shows the same tradeoff. A physically impossible donor interval correctly abstains rather than clipping both endpoints into a false physical solution.

Native A/B panels show the old woman boundary remains in the uncapped result. A separate parent experiment already shows part of that line responds to the hidden H/backing color; this audit therefore does not assign all residual line error to alpha. The cap hypothesis is causal for some pixel differences, but removing the cap is rejected because it loses real fine hair and does not solve the original visual symptom.

## Acceptance

`verification.json` confirms 9 frozen source variants, 18 synthetic variants, 21 A/B renders, byte-identical bounded replay, unchanged source cores/domain, preserved raw-alpha and inverse-map hashes, and `algorithmAccepted:false`. No threshold variant is promoted. The current cap remains a safety bound until an independent foreground/background donor or multi-view constraint can disambiguate true fine hair from a mixed RGB color.

Useful evidence:

- `source/report.json`: three photos × three alpha rules and source residuals.
- `renders/report.json`: exact V36/LaMa backing A/B outputs and changed-pixel counts.
- `synthetic/report.json`: known alpha/backing and counterexample metrics.
- `verification.json`: hash and invariant audit.
- `renders/portrait-woman-100/lama-256/left-B-comparison-1x.png`: original, V36, bounded, uncapped and donor-interval side-by-side.
- `synthetic/fine-hair-wrong-donor/comparison-1x.png`: known fine-hair failure.
