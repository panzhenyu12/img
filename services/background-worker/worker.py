"""Local LaMa inference and exact edit-mask compositing; no LLM dependency."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / ".models/background/big-lama.pt"
EXPECTED_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
EXPECTED_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


class InputError(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def rgba_image(data: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")
    except (OSError, ValueError) as error:
        raise InputError("image must be a valid PNG/JPEG/WebP") from error


def grayscale_mask(data: bytes | None, size: tuple[int, int]) -> np.ndarray:
    if data is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except (OSError, ValueError) as error:
        raise InputError("mask must be a valid grayscale image") from error
    if image.size != size:
        raise InputError("mask dimensions must exactly match the oriented input image")
    return np.asarray(image, dtype=np.uint8)


def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def poisson_match(source: np.ndarray, guide: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Gradient-domain seam matching, confined strictly to the target pixels.

    The guide includes a predicted context ring. Dirichlet boundaries use the
    immutable source. This borrows Pérez et al. SIGGRAPH 2003; it is not LaMa's model.
    """
    ys, xs = np.nonzero(target)
    count = len(xs)
    if not count:
        return guide, {"iterations_converged": True, "target_pixels": 0}
    if count > 300_000:
        raise InputError("poisson seam blend is limited to 300000 selected pixels; use seam_blend=none")
    height, width = target.shape
    labels = np.full(target.shape, -1, dtype=np.int32)
    labels[ys, xs] = np.arange(count)
    rows = [np.arange(count)]
    cols = [np.arange(count)]
    diagonal = np.zeros(count, dtype=np.float64)
    entries = []
    rhs = np.zeros((count, 3), dtype=np.float64)
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = xs + dx, ys + dy
        valid = (nx >= 0) & (ny >= 0) & (nx < width) & (ny < height)
        ids = np.nonzero(valid)[0]
        diagonal[ids] += 1.0
        rhs[ids] += guide[ys[ids], xs[ids], :3] - guide[ny[ids], nx[ids], :3]
        neighbor = labels[ny[ids], nx[ids]]
        inside = neighbor >= 0
        rows.append(ids[inside])
        cols.append(neighbor[inside])
        entries.append(-np.ones(np.count_nonzero(inside)))
        outside_ids = ids[~inside]
        rhs[outside_ids] += source[ny[outside_ids], nx[outside_ids], :3]
    matrix = sparse.csr_matrix((np.concatenate([diagonal, *entries]), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count))
    preconditioner = sparse.diags(1.0 / np.maximum(diagonal, 1.0))
    output = guide.copy()
    statuses = []
    for channel in range(3):
        solution, status = cg(matrix, rhs[:, channel], x0=guide[ys, xs, channel], M=preconditioner, rtol=1e-5, atol=0.01, maxiter=500)
        statuses.append(int(status))
        if status != 0:
            raise RuntimeError("Poisson seam solver did not converge; keep the raw LaMa result")
        output[ys, xs, channel] = np.clip(solution, 0, 255)
    return output, {"iterations_converged": all(value == 0 for value in statuses), "target_pixels": count}


class LamaWorker:
    def __init__(self, model_path: Path = DEFAULT_MODEL, threads: int = 6):
        self.model_path = Path(model_path)
        self.threads = max(1, min(threads, os.cpu_count() or 1))
        self._model = None
        self._lock = threading.Lock()
        self.model_sha256 = None
        self.load_ms = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise ModelUnavailable("LaMa model is missing; run scripts/download_models.py lama-torchscript")
        started = time.monotonic()
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                md5.update(block); sha256.update(block)
        if md5.hexdigest() != EXPECTED_MD5 or sha256.hexdigest() != EXPECTED_SHA256:
            raise ModelUnavailable("LaMa artifact checksum does not match the pinned IOPaint export")
        import torch
        torch.set_num_threads(self.threads)
        self._model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        self.model_sha256 = sha256.hexdigest()
        self.load_ms = round((time.monotonic() - started) * 1000, 2)
        return self._model

    def inpaint(self, image: Image.Image, mask: np.ndarray, protection: np.ndarray | None = None, *, max_side: int = 1024, context_mode: str = "full", context_padding: int = 256, mask_dilation: int = 4, seam_blend: str = "none") -> tuple[Image.Image, dict]:
        if not 256 <= max_side <= 2048:
            raise InputError("max_side must be between 256 and 2048")
        if not 0 <= mask_dilation <= 32 or not 64 <= context_padding <= 1024:
            raise InputError("invalid mask dilation or context padding")
        if seam_blend not in {"none", "poisson"}:
            raise InputError("seam_blend must be none or poisson")
        if context_mode not in {"full", "crop"}:
            raise InputError("context_mode must be full or crop")
        image = image.convert("RGBA")
        source = np.asarray(image, dtype=np.uint8)
        height, width = source.shape[:2]
        if mask.shape != (height, width):
            raise InputError("mask dimensions must match the image")
        protection = np.zeros_like(mask) if protection is None else protection
        if protection.shape != mask.shape:
            raise InputError("protection dimensions must match the image")
        frozen = protection > 0
        transparent = source[:, :, 3] == 0
        target = (mask > 0) & ~frozen & ~transparent
        metadata = {"model": "lama-big-torchscript", "paper": "https://arxiv.org/abs/2109.07161", "selected_pixels": int(target.sum()), "device": "cpu", "threads": self.threads, "seam_blend": seam_blend, "context_mode": context_mode}
        if not target.any():
            return image.copy(), {**metadata, "inference_ms": 0.0, "total_ms": 0.0, "peak_rss_mb": peak_memory_mb()}
        ys, xs = np.nonzero(target)
        # Full-scene context is the tested default. Cropping is explicit: a
        # faster street crop produced a dark texture hallucination during QA.
        if context_mode == "full":
            box = (0, 0, width, height)
        else:
            box = (max(0, int(xs.min()) - context_padding), max(0, int(ys.min()) - context_padding), min(width, int(xs.max()) + context_padding + 1), min(height, int(ys.max()) + context_padding + 1))
        left, top, right, bottom = box
        selected = target[top:bottom, left:right]
        blocked = frozen[top:bottom, left:right] | transparent[top:bottom, left:right]
        inference_mask = ndimage.binary_dilation(selected, iterations=mask_dilation) if mask_dilation else selected.copy()
        inference_mask |= blocked
        if not (~inference_mask).any():
            raise InputError("inpainting requires known nontransparent unprotected context around the target")
        crop_rgb = source[top:bottom, left:right, :3].copy()
        # Masked object, protected-person, and hidden-alpha RGB cannot affect
        # model input, including resizing and its Fourier receptive field.
        crop_rgb[inference_mask] = 0
        crop_height, crop_width = inference_mask.shape
        scale = min(1.0, max_side / max(crop_width, crop_height))
        scaled_size = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
        if scale < 1:
            # Match the positive area-filter footprint used by the mask. With
            # Lanczos, its wider signed kernel mixes zeroed unknown pixels into
            # nominally known RGB outside a BOX-downsampled mask, creating a
            # dark context halo that LaMa can propagate throughout the hole.
            crop_rgb = np.asarray(Image.fromarray(crop_rgb).resize(scaled_size, Image.Resampling.BOX)).copy()
            # Float coverage keeps even one blocked source pixel unknown. An
            # 8-bit BOX mask can round coverage below 0.5/255 down to zero on
            # strong reductions and silently admit that pixel as context.
            mask_image = Image.fromarray(inference_mask.astype(np.float32))
            inference_mask = np.asarray(mask_image.resize(scaled_size, Image.Resampling.BOX)) > 0
            crop_rgb[inference_mask] = 0
            if not (~inference_mask).any():
                raise InputError("downsampling leaves no known context; increase max_side or use crop context")
        pad_h = (-crop_rgb.shape[0]) % 8
        pad_w = (-crop_rgb.shape[1]) % 8
        padded_rgb = np.pad(crop_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
        padded_mask = np.pad(inference_mask, ((0, pad_h), (0, pad_w)), mode="symmetric")
        started = time.monotonic()
        with self._lock:
            model = self._load()
            import torch
            rgb_tensor = torch.from_numpy(padded_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
            mask_tensor = torch.from_numpy(padded_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            inference_start = time.monotonic()
            with torch.inference_mode():
                prediction = model(rgb_tensor, mask_tensor)[0].permute(1, 2, 0).cpu().numpy()
            inference_ms = (time.monotonic() - inference_start) * 1000
        prediction = np.clip(prediction[:scaled_size[1], :scaled_size[0]] * 255.0, 0, 255)
        predicted_u8 = np.floor(prediction + 0.5).astype(np.uint8)
        if scale < 1:
            prediction = np.asarray(Image.fromarray(predicted_u8).resize((crop_width, crop_height), Image.Resampling.LANCZOS)).astype(np.float64)
        else:
            prediction = prediction.astype(np.float64)
        seam_metadata = {}
        if seam_blend == "poisson":
            prediction, seam_metadata = poisson_match(source[top:bottom, left:right].astype(np.float64), prediction, selected)
        output = source.copy()
        output_crop = output[top:bottom, left:right]
        output_crop[selected, :3] = np.floor(np.clip(prediction[selected, :3], 0, 255) + 0.5).astype(np.uint8)
        # All alpha and every pixel outside the canonical target are retained.
        assert np.array_equal(output[~target], source[~target])
        assert np.array_equal(output[:, :, 3], source[:, :, 3])
        metadata.update({"inference_ms": round(inference_ms, 2), "total_ms": round((time.monotonic() - started) * 1000, 2), "model_load_ms": self.load_ms, "peak_rss_mb": round(peak_memory_mb(), 2), "model_sha256": self.model_sha256, "crop_box": list(box), "model_size": [padded_rgb.shape[1], padded_rgb.shape[0]], "scaled": scale < 1, "mask_dilation": mask_dilation, "seam": seam_metadata})
        return Image.fromarray(output, "RGBA"), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--protect-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--context-mode", choices=["full", "crop"], default="full")
    parser.add_argument("--context-padding", type=int, default=256)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seam-blend", choices=["none", "poisson"], default="none")
    options = parser.parse_args()
    image = rgba_image(options.input.read_bytes())
    mask = grayscale_mask(options.mask.read_bytes(), image.size)
    protection = grayscale_mask(options.protect_mask.read_bytes() if options.protect_mask else None, image.size)
    result, metadata = LamaWorker(options.model, options.threads).inpaint(image, mask, protection, max_side=options.max_side, context_mode=options.context_mode, context_padding=options.context_padding, mask_dilation=options.mask_dilation, seam_blend=options.seam_blend)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(options.output)
    if options.report:
        options.report.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
