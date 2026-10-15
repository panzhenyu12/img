#!/usr/bin/env python3
"""Compare local neural refinement on a frozen portrait-warp candidate.

This is a QA experiment, not a production beauty implementation. It consumes
the warp's recorded physical coverage rather than guessing holes from RGB.
Only newly exposed, nearly empty foreground pixels may change. A model's
success/contract checks do not constitute aesthetic acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=3)
    options = parser.parse_args()
    if options.output.exists():
        raise SystemExit("Use a new output directory; frozen results must not be overwritten")
    options.output.mkdir(parents=True)
    paths = {name: options.debug / name for name in (
        "geometry.json", "old-alpha.png", "new-alpha.png", "frozen.png", "holes.png"
    )}
    source = Image.open(options.candidate).convert("RGBA")
    pixels = np.asarray(source)
    origin = json.loads(paths["geometry.json"].read_text())["origin"]
    left, top = map(int, origin)
    local = {key: np.asarray(Image.open(path).convert("L")) for key, path in paths.items() if key.endswith(".png")}
    shape = local["old-alpha.png"].shape
    if any(value.shape != shape for value in local.values()):
        raise ValueError("Diagnostic alpha/mask dimensions do not match")
    height, width = shape
    if left < 0 or top < 0 or left + width > source.width or top + height > source.height:
        raise ValueError("Diagnostic crop is outside the candidate")
    # A displayed face edge is never overwritten to hide a geometry error.
    # The one-byte cutoff is recorded, so mask selection remains reproducible.
    selected = (local["old-alpha.png"] > 0) & (local["new-alpha.png"] <= 1)
    selected &= (local["holes.png"] > 0) & (local["frozen.png"] == 0)
    target = np.zeros(pixels.shape[:2], dtype=np.uint8)
    target[top:top + height, left:left + width] = selected.astype(np.uint8) * 255
    frozen = np.zeros_like(target)
    frozen[top:top + height, left:left + width] = local["frozen.png"]
    Image.fromarray(target).save(options.output / "target.png")
    Image.fromarray(frozen).save(options.output / "frozen.png")
    source.save(options.output / "candidate.png")

    worker_path = ROOT / "services/background-worker/worker.py"
    spec = importlib.util.spec_from_file_location("face_refinement_worker", worker_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    worker = module.LamaWorker(threads=options.threads)
    report = {
        "status": "experimental; visual review required",
        "candidate": str(options.candidate.resolve()),
        "candidate_sha256": sha(options.candidate),
        "debug_sha256": {key: sha(path) for key, path in paths.items()},
        "target_sha256": sha(options.output / "target.png"),
        "selection": "old coverage > 0; new coverage <= 1/255; holes; not frozen",
        "selected_pixels": int(np.count_nonzero(target)),
        "model_context": "Frozen pixels remain known conditioning context; only target is writable",
        "variants": [],
    }
    # Here frozen output is already excluded from target. It remains useful
    # conditioning (ear/face/neck context). Background removal's donor-exclusion
    # semantics must not be silently reused for portrait-layer reconstruction.
    for context, padding, seam in [("crop", 64, "none"), ("crop", 128, "none"), ("crop", 128, "poisson"), ("full", 128, "none")]:
        name = f"{context}-{padding}-{seam}"
        result, metadata = worker.inpaint(source, target, max_side=512,
            context_mode=context, context_padding=padding, mask_dilation=2, seam_blend=seam)
        edited = np.asarray(result)
        writable = target > 0
        contracts = {
            "outside_changed": int(np.count_nonzero(np.any(edited[~writable] != pixels[~writable], axis=-1))),
            "frozen_changed": int(np.count_nonzero(np.any(edited[frozen > 0] != pixels[frozen > 0], axis=-1))),
            "alpha_changed": int(np.count_nonzero(edited[:, :, 3] != pixels[:, :, 3])),
        }
        if any(contracts.values()):
            raise AssertionError(contracts)
        result.save(options.output / f"{name}.png")
        panel = Image.new("RGBA", (source.width * 2, source.height))
        panel.paste(source, (0, 0)); panel.paste(result, (source.width, 0))
        panel.save(options.output / f"{name}-comparison.png")
        report["variants"].append({"name": name, **metadata, **contracts,
            "output_sha256": sha(options.output / f"{name}.png")})
        (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"name": name, **contracts, "inference_ms": metadata["inference_ms"], "total_ms": metadata["total_ms"]}), flush=True)


if __name__ == "__main__":
    run()
