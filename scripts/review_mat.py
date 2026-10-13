"""Isolated MAT Places-512 comparison. Does not change the production model.

Uses a pinned safetensors conversion and pinned pure-PyTorch architecture.
Mask-positive, protected and transparent input RGB cannot donate to inference;
the full-size output keeps alpha, protection and unselected pixels exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
ARCH_REV = "5b7681746fe19e563113d27366ba85236bc1ba5b"
MODEL_REV = "bef6e8b7d535abe2d42fdaeafec903abdfc638a3"
MODEL_SHA = "309dd6ce6e04034dc4b6b15c7bd2a4844d158e03eb2a139e0eca6b366e40c0de"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_resize(values, size):
    return np.asarray(Image.fromarray(values.astype(np.float32), "F").resize(size, Image.Resampling.BOX)).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=Path, default=ROOT / f".run/comfyui-inpaint-nodes-{ARCH_REV}")
    parser.add_argument("--model", type=Path, default=ROOT / ".models/background/MAT_Places_512_fp16.safetensors")
    parser.add_argument("--cases", nargs="+", default=["street", "crowd", "architecture"])
    args = parser.parse_args()
    assert digest(args.model) == MODEL_SHA, "Pinned model checksum mismatch"
    sys.path.insert(0, str(args.architecture))
    from mat import load

    torch.set_num_threads(4)
    np.random.seed(0)
    torch.manual_seed(0)
    begin = time.perf_counter()
    model = load({k: v.float() for k, v in load_file(args.model).items()}).float().eval()
    model.z = model.z.float()
    model.label = model.label.float()
    load_ms = (time.perf_counter() - begin) * 1000
    out = ROOT / "test/results/mat"
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "doc/qa/background-lama"
    rows = []
    for case in args.cases:
        source_path = data / f"{case}-source.png"
        source = Image.open(source_path).convert("RGBA")
        original = np.asarray(source)
        target = np.asarray(Image.open(data / f"{case}-mask.png").convert("L")) > 0
        protect_path = data / f"{case}-protect.png"
        protect = np.asarray(Image.open(protect_path).convert("L")) > 0 if protect_path.exists() else np.zeros(target.shape, bool)
        canonical = target & ~protect & (original[:, :, 3] > 0)
        unknown = target | protect | (original[:, :, 3] < 255)
        scale = min(1., 512 / max(source.size))
        rw, rh = tuple(max(1, round(v * scale)) for v in source.size)
        known = box_resize((~unknown).astype(float), (rw, rh))
        resized = np.stack([box_resize(original[:, :, c] * (~unknown), (rw, rh)) / np.maximum(known, 1e-6) for c in range(3)], 2)
        # Any unknown input footprint is a hole; neither removed nor protected
        # RGB is hidden in an apparently-known resampled edge cell.
        mask = known < 1 - 1e-6
        resized[mask] = 0
        x0, y0 = (512 - rw) // 2, (512 - rh) // 2
        rgb = np.zeros((512, 512, 3), np.float32)
        hole = np.ones((512, 512), np.float32)
        rgb[y0:y0+rh, x0:x0+rw] = resized / 255
        hole[y0:y0+rh, x0:x0+rw] = mask
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[None])
        mask_tensor = torch.from_numpy(hole[None, None])
        torch.manual_seed(0)
        begin = time.perf_counter()
        with torch.inference_mode():
            predicted = model(image_tensor, mask_tensor)[0].permute(1, 2, 0).numpy()
        inference_ms = (time.perf_counter() - begin) * 1000
        predicted = np.clip(np.rint(predicted[y0:y0+rh, x0:x0+rw] * 255), 0, 255).astype(np.uint8)
        restored = np.asarray(Image.fromarray(predicted).resize(source.size, Image.Resampling.BICUBIC))
        result = original.copy()
        result[canonical, :3] = restored[canonical]
        output = out / f"{case}-mat.png"
        Image.fromarray(result).save(output)
        change = np.any(result != original, axis=2)
        row = {"case": case, "sourceSha256": digest(source_path), "modelSha256": MODEL_SHA,
               "inferenceMs": round(inference_ms, 2), "input": [rw, rh],
               "outsideChanges": int(change[~canonical].sum()), "protectChanges": int(change[protect].sum()),
               "alphaChanges": int(np.count_nonzero(result[:, :, 3] != original[:, :, 3]))}
        baseline_file = data / ("street-http.png" if case == "street" else f"{case}-http-fixed.png")
        baseline = np.asarray(Image.open(baseline_file).convert("RGBA")).copy()
        baseline[~canonical] = original[~canonical]
        baseline[:, :, 3] = original[:, :, 3]
        if case == "architecture":
            clean = np.asarray(Image.open(data / "architecture-clean.png").convert("RGBA"))
            row["targetMaeLama"] = float(np.abs(baseline[:, :, :3].astype(float) - clean[:, :, :3])[canonical].mean())
            row["targetMaeMat"] = float(np.abs(result[:, :, :3].astype(float) - clean[:, :, :3])[canonical].mean())
        yy, xx = np.nonzero(canonical)
        bounds = (max(0, int(xx.min())-35), max(0, int(yy.min())-35), min(source.width, int(xx.max())+36), min(source.height, int(yy.max())+36))
        views = [source.crop(bounds), Image.fromarray(baseline).crop(bounds), Image.fromarray(result).crop(bounds)]
        grid = Image.new("RGB", (views[0].width * 3, views[0].height + 25), "white")
        draw = ImageDraw.Draw(grid)
        for i, (view, label) in enumerate(zip(views, ["Source", "LaMa", "MAT Places-512"])):
            grid.paste(view, (i*view.width, 25))
            draw.text((i*view.width+4, 6), label, fill="black")
        grid.save(out / f"{case}-comparison.png")
        assert row["outsideChanges"] == row["protectChanges"] == row["alphaChanges"] == 0
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"modelRevision": MODEL_REV, "architectureRevision": ARCH_REV, "loadMs": round(load_ms, 2),
              "torch": torch.__version__, "threads": 4, "seed": 0, "dtype": "float32", "rows": rows,
              "note": "Independent candidate; visual review required. Architecture is a third-party CPU adaptation, not the authors' CUDA implementation."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
