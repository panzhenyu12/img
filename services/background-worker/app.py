"""Loopback-only local model worker; the Go API owns sessions and workflows."""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from worker import DEFAULT_MODEL, InputError, LamaWorker, ModelUnavailable, grayscale_mask, rgba_image

app = FastAPI(title="Local background inpainting", version="0.1.0")
worker = LamaWorker(Path(os.getenv("BACKGROUND_LAMA_MODEL", str(DEFAULT_MODEL))), int(os.getenv("BACKGROUND_CPU_THREADS", "6")))


@app.get("/healthz")
def health():
    return {"status": "ok", "model": "lama-big-torchscript", "device": "cpu", "modelPresent": worker.model_path.is_file(), "loaded": worker._model is not None, "llmRequired": False}


@app.post("/v1/inpaint")
def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), protect_mask: UploadFile | None = File(None), max_side: int = Form(1024), context_mode: str = Form("full"), context_padding: int = Form(256), mask_dilation: int = Form(4), seam_blend: str = Form("none")):
    try:
        source = rgba_image(image.file.read())
        selection = grayscale_mask(mask.file.read(), source.size)
        protection = grayscale_mask(protect_mask.file.read() if protect_mask else None, source.size)
        result, metadata = worker.inpaint(source, selection, protection, max_side=max_side, context_mode=context_mode, context_padding=context_padding, mask_dilation=mask_dilation, seam_blend=seam_blend)
        encoded = io.BytesIO(); result.save(encoded, format="PNG")
        return Response(encoded.getvalue(), media_type="image/png", headers={
            "X-Inpaint-Model": metadata["model"], "X-Inpaint-Device": "cpu",
            "X-Inpaint-Inference-Ms": str(metadata["inference_ms"]), "X-Inpaint-Total-Ms": str(metadata["total_ms"]),
            "X-Inpaint-Peak-Rss-Mb": str(metadata["peak_rss_mb"]), "X-Inpaint-Selected-Pixels": str(metadata["selected_pixels"]),
            "X-Inpaint-Seam-Blend": seam_blend,
            "X-Inpaint-Context-Mode": context_mode,
        })
    except InputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8091)
