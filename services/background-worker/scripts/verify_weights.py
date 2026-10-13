"""Verify that the exported generator exactly matches the official checkpoint.

Requires requirements-verify.txt and an explicitly downloaded pinned checkpoint.
Lightning's training callback metadata is discarded; generator tensors are not.
This compares inference weights/buffers, not a retraining reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import types
import zipfile

from download_models import ROOT, ARTIFACTS, valid


class CallbackMetadata:
    def __setstate__(self, state):
        self.state = state


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == "ModelCheckpoint":
            return CallbackMetadata
        return super().find_class(module, name)


def verify(directory: Path) -> dict:
    archive = directory / "big-lama.zip"
    exported = directory / "big-lama.pt"
    if not valid(archive, ARTIFACTS["lama-official-checkpoint"]) or not valid(exported, ARTIFACTS["lama-torchscript"]):
        raise RuntimeError("download the pinned official checkpoint and TorchScript export first")
    destination = directory / "official"
    destination.mkdir(exist_ok=True)
    # Extract only the official generator checkpoint, not arbitrary archive paths.
    checkpoint = destination / "big-lama/models/best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as compressed:
        checkpoint.write_bytes(compressed.read("big-lama/models/best.ckpt"))
    import torch
    package = types.SimpleNamespace(Unpickler=CheckpointUnpickler, load=pickle.load, __name__="checkpoint_metadata_only")
    original = torch.load(checkpoint, map_location="cpu", pickle_module=package)["state_dict"]
    official = {key.removeprefix("generator."): value for key, value in original.items() if key.startswith("generator.")}
    script = torch.jit.load(str(exported), map_location="cpu")
    tensors = {key.removeprefix("generator."): value for key, value in script.state_dict().items()}
    report = {"official_generator_tensors": len(official), "export_generator_tensors": len(tensors), "missing": sorted(set(official) - set(tensors)), "extra": sorted(set(tensors) - set(official)), "matching": 0, "different": [], "generator_sha256": None, "checkpoint_sha256": ARTIFACTS["lama-official-checkpoint"]["sha256"], "export_sha256": ARTIFACTS["lama-torchscript"]["sha256"], "comparison": "exact torch.equal for every generator parameter and buffer after removing generator. prefixes"}
    digest = hashlib.sha256()
    for key, tensor in sorted(official.items()):
        digest.update(key.encode()); digest.update(tensor.numpy().tobytes())
        if key in tensors:
            if torch.equal(tensor, tensors[key]): report["matching"] += 1
            else: report["different"].append(key)
    report["generator_sha256"] = digest.hexdigest()
    if report["missing"] or report["extra"] or report["different"] or not official:
        raise RuntimeError(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / ".models/background")
    parser.add_argument("--report", type=Path, default=ROOT / "doc/qa/background-lama/weight-verification.json")
    options = parser.parse_args()
    report = verify(options.directory)
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
