"""Wrapper contracts; image quality is assessed separately on real photos."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from worker import InputError, LamaWorker, ModelUnavailable, grayscale_mask, poisson_match


class ContextModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, rgb, mask):
        self.inputs.append((rgb.clone(), mask.clone()))
        # A nonlocal donor deliberately makes any leaked RGB affect the fill.
        return torch.full_like(rgb, 0.15) + rgb.mean() * 0.75


def fake_worker():
    worker = LamaWorker(Path("missing"))
    model = ContextModel()
    worker._load = lambda: model
    return worker, model


def source(width=31, height=23):
    ys, xs = np.mgrid[:height, :width]
    pixels = np.stack((30 + xs % 150, 80 + ys % 150, (xs + ys) % 220, np.full_like(xs, 255)), axis=-1).astype(np.uint8)
    pixels[0, :, 3] = 0
    pixels[1, :, 3] = 79
    return pixels


def test_blank_and_fully_protected_masks_never_load_model():
    pixels = source()
    worker = LamaWorker(Path("missing"))
    for mask, protection in [(np.zeros(pixels.shape[:2]), None), (np.ones(pixels.shape[:2]), np.ones(pixels.shape[:2]))]:
        result, report = worker.inpaint(Image.fromarray(pixels), mask, protection)
        np.testing.assert_array_equal(result, pixels)
        assert report["selected_pixels"] == 0
        assert worker._model is None


def test_edit_boundary_protection_and_all_alpha_are_exact():
    pixels = source()
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[:14, 5:20] = 255
    protection = np.zeros_like(mask); protection[3:10, 8:12] = 128
    target = (mask > 0) & (protection == 0) & (pixels[:, :, 3] > 0)
    worker, _ = fake_worker()
    result, report = worker.inpaint(Image.fromarray(pixels), mask, protection)
    result = np.asarray(result)
    np.testing.assert_array_equal(result[~target], pixels[~target])
    np.testing.assert_array_equal(result[:, :, 3], pixels[:, :, 3])
    assert np.any(result[target, :3] != pixels[target, :3])
    assert report["selected_pixels"] == target.sum()


@pytest.mark.parametrize("context_mode", ["full", "crop"])
def test_target_protected_and_hidden_rgb_never_contaminate_resized_model_input(context_mode):
    pixels = source(768, 171)
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[25:140, 430:540] = 255
    protection = np.zeros_like(mask); protection[40:120, 250:330] = 255
    blocked = (mask > 0) | (protection > 0) | (pixels[:, :, 3] == 0)
    changed = pixels.copy(); changed[blocked, :3] = [250, 2, 220]
    worker, model = fake_worker()
    result, _ = worker.inpaint(Image.fromarray(pixels), mask, protection, max_side=256, context_mode=context_mode, mask_dilation=0)
    other, _ = worker.inpaint(Image.fromarray(changed), mask, protection, max_side=256, context_mode=context_mode, mask_dilation=0)
    assert torch.equal(model.inputs[0][0], model.inputs[1][0])
    assert torch.equal(model.inputs[0][1], model.inputs[1][1])
    target = (mask > 0) & (protection == 0) & (pixels[:, :, 3] > 0)
    np.testing.assert_array_equal(np.asarray(result)[target], np.asarray(other)[target])


def test_bad_dimensions_options_and_no_context_are_rejected():
    pixels = source(); mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[4:10, 4:10] = 255
    worker, _ = fake_worker()
    for kwargs in [{"max_side": 255}, {"context_mode": "automatic"}, {"mask_dilation": 33}, {"seam_blend": "blur"}]:
        with pytest.raises(InputError): worker.inpaint(Image.fromarray(pixels), mask, **kwargs)
    with pytest.raises(InputError): worker.inpaint(Image.fromarray(pixels), mask[:-1])
    with pytest.raises(InputError): worker.inpaint(Image.fromarray(pixels), mask, mask[:-1])
    with pytest.raises(InputError, match="known nontransparent"):
        worker.inpaint(Image.fromarray(pixels), np.ones_like(mask))
    with pytest.raises(InputError): grayscale_mask(b"bad", (31, 23))


def test_resizing_never_creates_a_black_halo_in_known_context():
    pixels = np.tile(np.array([80, 120, 160, 255], dtype=np.uint8), (129, 769, 1))
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[40:90, 240:350] = 255
    protection = np.zeros_like(mask); protection[90:110, 500:600] = 255
    worker, model = fake_worker()
    worker.inpaint(Image.fromarray(pixels), mask, protection, max_side=256)
    rgb, inference_mask = model.inputs[0]
    known = (inference_mask[0, 0] == 0).numpy()
    for channel, value in enumerate([80, 120, 160]):
        np.testing.assert_allclose(rgb[0, channel].numpy()[known], value / 255, atol=1e-7)


def test_strong_downsample_preserves_single_pixel_target_protection_and_transparency():
    # Each unknown covers only 1/1024 of its model pixel. uint8 BOX rounding
    # erases this coverage entirely; model context must conservatively exclude it.
    pixels = np.tile(np.array([96, 160, 224, 255], dtype=np.uint8), (1024, 8192, 1))
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[64, 64] = 255
    protection = np.zeros_like(mask); protection[512, 512] = 255
    pixels[768, 768, 3] = 0
    worker, model = fake_worker()
    result, _ = worker.inpaint(Image.fromarray(pixels), mask, protection, max_side=256, mask_dilation=0)
    rgb, inference_mask = model.inputs[0]
    assert tuple(inference_mask.shape) == (1, 1, 32, 256)
    for y, x in [(2, 2), (16, 16), (24, 24)]:
        assert inference_mask[0, 0, y, x].item() == 1
        assert torch.equal(rgb[0, :, y, x], torch.zeros(3))
    target = mask > 0
    np.testing.assert_array_equal(np.asarray(result)[~target], pixels[~target])


def test_downsample_that_blocks_every_model_pixel_is_rejected_before_inference():
    pixels = np.tile(np.array([96, 160, 224, 255], dtype=np.uint8), (512, 512, 1))
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[::2, ::2] = 255
    worker, model = fake_worker()
    with pytest.raises(InputError, match="downsampling leaves no known context"):
        worker.inpaint(Image.fromarray(pixels), mask, max_side=256, mask_dilation=0)
    assert model.inputs == []


def test_missing_or_wrong_artifact_fails_explicitly(tmp_path):
    with pytest.raises(ModelUnavailable, match="missing"): LamaWorker(tmp_path / "none.pt")._load()
    bad = tmp_path / "bad.pt"; bad.write_bytes(b"not the pinned model")
    with pytest.raises(ModelUnavailable, match="checksum"): LamaWorker(bad)._load()


def test_poisson_restores_boundary_on_constant_offset_and_linear_ramp():
    ys, xs = np.mgrid[:23, :31]
    original = np.stack((40 + xs, 60 + ys, 30 + xs + ys, np.full_like(xs, 255)), -1).astype(float)
    target = np.zeros(original.shape[:2], dtype=bool); target[5:18, 7:24] = True
    # A global +80 bias has identical gradients and must match original boundary.
    guide = original[:, :, :3] + 80
    result, report = poisson_match(original, guide, target)
    np.testing.assert_allclose(result[target], original[target, :3], atol=0.02)
    np.testing.assert_array_equal(result[~target], guide[~target])
    assert report["iterations_converged"]


def test_http_uses_real_wrapper_and_exposes_no_llm_dependency(monkeypatch):
    import io
    from fastapi.testclient import TestClient
    import app
    worker, _ = fake_worker(); monkeypatch.setattr(app, "worker", worker)
    client = TestClient(app.app)
    assert client.get("/healthz").json()["llmRequired"] is False
    pixels = source(); mask = np.zeros(pixels.shape[:2], dtype=np.uint8); mask[6:16, 8:18] = 255
    def png_bytes(array):
        output = io.BytesIO(); Image.fromarray(array).save(output, "PNG"); return output.getvalue()
    files = {"image": ("image.png", png_bytes(pixels), "image/png"), "mask": ("mask.png", png_bytes(mask), "image/png")}
    response = client.post("/v1/inpaint", files=files)
    assert response.status_code == 200
    result = np.asarray(Image.open(io.BytesIO(response.content)))
    np.testing.assert_array_equal(result[mask == 0], pixels[mask == 0])
    assert response.headers["x-inpaint-context-mode"] == "full"
    assert client.post("/v1/inpaint", files=files, data={"context_mode": "invalid"}).status_code == 422
    files["mask"] = ("bad.png", png_bytes(mask[:-1]), "image/png")
    assert client.post("/v1/inpaint", files=files).status_code == 422
