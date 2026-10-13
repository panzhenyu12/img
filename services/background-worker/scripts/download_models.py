"""Fetch explicitly requested, public inpainting artifacts into an offline cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.request
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[3]
ARTIFACTS = {
    "lama-torchscript": {
        "filename": "big-lama.pt",
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "size": 205669692,
        "md5": "e3aa4aaa15225a33ec84f9f4bc47e500",
        "sha256": "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
        "provenance": "TorchScript export referenced by IOPaint; verify against official checkpoint separately",
    },
    "lama-official-checkpoint": {
        "filename": "big-lama.zip",
        "url": "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip",
        "size": 381428720,
        "sha256": "f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6",
        "provenance": "Checkpoint mirror explicitly linked from advimman/lama official README",
    },
}


def digest(path: pathlib.Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def valid(path: pathlib.Path, artifact: dict) -> bool:
    if not path.is_file() or path.stat().st_size != artifact["size"]:
        return False
    algorithm = "sha256" if "sha256" in artifact else "md5"
    return digest(path, algorithm) == artifact[algorithm]


def download(name: str, directory: pathlib.Path) -> pathlib.Path:
    artifact = ARTIFACTS[name]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / artifact["filename"]
    if not valid(target, artifact):
        part = target.with_suffix(target.suffix + ".part")
        started = time.monotonic()
        for attempt in range(3):
            if valid(part, artifact):
                break
            if part.is_file() and part.stat().st_size >= artifact["size"]:
                part.unlink()  # A complete bad checksum cannot be resumed.
            offset = part.stat().st_size if part.is_file() else 0
            headers = {"User-Agent": "pixel-retouch-model-setup/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(artifact["url"], headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    resumed = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                    total = offset if resumed else 0
                    with part.open("ab" if resumed else "wb") as handle:
                        next_report = total + 32 * 1024 * 1024
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            handle.write(block); total += len(block)
                            if total > artifact["size"]:
                                raise RuntimeError("artifact exceeds the pinned size")
                            if total >= next_report:
                                print(json.dumps({"artifact": name, "downloaded_bytes": total, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
                                next_report += 32 * 1024 * 1024
                if valid(part, artifact):
                    break
            except (OSError, urllib.error.URLError) as error:
                if attempt == 2:
                    raise RuntimeError(f"download failed after 3 attempts for {name}; partial data retained") from error
                print(json.dumps({"artifact": name, "retry": attempt + 1}), flush=True)
                time.sleep(1 + attempt)
        if not valid(part, artifact):
            raise RuntimeError(f"size/checksum mismatch for {name}; rejected {part}")
        part.replace(target)
    metadata = {**artifact, "name": name, "sha256_observed": digest(target, "sha256")}
    target.with_suffix(target.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"artifact": name, "path": str(target), "sha256": metadata["sha256_observed"]}), flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=ARTIFACTS)
    parser.add_argument("--directory", type=pathlib.Path, default=ROOT / ".models/background")
    options = parser.parse_args()
    for name in options.models:
        download(name, options.directory)
