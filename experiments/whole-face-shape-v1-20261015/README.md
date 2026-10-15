# Whole-face width curves V1 — rejected for requested behavior

This isolated experiment adds `faceShape: {preset, amount}` to `face.adjust` only. It is not deployed. Profiles are design hypotheses, not learned beauty models and not complete PixelCake-style presets.

## Outcome

The requested contour changes upper, middle and lower face width, but actual production attachment constraints suppress most upper/middle movement. On user natural +30 the upper sides request about 5.21/5.03 px and achieve 0.001/0 px; middle requests 4.85/4.83 px and achieves 0/0 px; lower requests 3.71/3.74 px and achieves 2.72/2.82 px. Woman and hair photos show the same conflict. The overall-width behavior therefore **fails**. Passing synthetic tests does not establish aesthetic quality or 8/10.

The causal hypothesis for V2 is that transporting only skin while fully fixing the same head’s ears/hair imposes incompatible internal material boundaries. V2 will test one head-plus-upper-neck foreground and continuous displacement; background remains separate.

## Evidence

- Seven targeted tests passed: three geometry, one profile and three public-operation contracts. No full library suite was claimed.
- Six legacy faceWidth ±30 photo renders are pixel-identical to saved production references.
- Fifteen experimental photo outputs are preserved: user six, woman six, hair three. The script was reduced during the sequential run; its current matrix is user six/woman three/hair three. Extra woman style outputs preserve the originally expanded protocol.
- `outputs/*/report.json` records hashes, elapsed times, coverage and requested/feasible/actual displacement metrics. Overlays are diagnostic, separate from unannotated source/edited image panels.
- `protocol-red.log` records the original unsupported-field failure; `geometry-first-red.log` records the initial floating-point geometry failure fixed by adding displacement directly in the frame-across direction.

No source-alpha/compositor changes, production integration or deployment occurred.
