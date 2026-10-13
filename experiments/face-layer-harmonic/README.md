# Face layer harmonic experiments

`harmonic-weights-v1.patch` is an isolated candidate based on accepted physical-alpha v10 attempt2. It computes separate scene/neck/hair harmonic fields and mixes exposed holes by one-hot layer weights. The 8-case review found smoother but physically wrong color mud and no recovery of hair or neck texture, so it is rejected. See [review](../../doc/qa/face-layer-harmonic-v1/review.md). Do not apply it to production without a new full replay and visual review.
