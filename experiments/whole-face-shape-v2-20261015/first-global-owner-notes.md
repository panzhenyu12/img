# First causal V2 candidate — not accepted

`outputs/user-face-contour` and frozen image-render-e0dc9e6a48835de02fe558993f9741a78b2c110a3093852e3632cfdc20d5bb51 are from this first candidate. Source is preserved in `first-global-owner-candidate-face.rs`.

This candidate globally disables semantic hair/ear/accessory fixed masks in the ROI when whole-head mode is active. It demonstrates that removing internal fixed material boundaries permits visible whole-face width changes, but ownership is insufficiently constrained. Explicit protection and inferred other-person remain active, yet no disconnected-head contract was present. Parent review requested a proper selected-head-connected support, independently of alpha coverage. External ear fringe still exhibits light/dark boundaries in real photos. Not deployed, not >=8/10.
