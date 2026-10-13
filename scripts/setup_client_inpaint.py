"""Install a pinned browser-compatible LaMa model; no photograph leaves the client."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "apps/web/public/models/lama-fp32-512.onnx"
SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists() or checksum(TARGET) != SHA256:
        partial = TARGET.with_suffix(".download")
        urllib.request.urlretrieve(URL, partial)
        if checksum(partial) != SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError("browser LaMa checksum mismatch")
        partial.replace(TARGET)
    print(f"{TARGET.name}: SHA256 verified ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
