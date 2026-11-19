"""Bounded source-only SAM2 candidate bank, without matting or rendered edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / "experiments/sam2matting-tiny-probe-20261015"
CHECKPOINT = ROOT / ".models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt"
CONFIG = "configs/sam2matting-sam2.1tiny.yaml"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spatial_queries(classes, image_size):
    """One positive point per occupied cell of a 4-column x 8-row support grid.

    Support is the posterior winner being non-background with >=128/255
    confidence. The box and sample positions depend only on this source
    posterior. Empty cells are skipped, never redistributed. Nearest-center
    ties use source row-major order. No anatomy-specific prompt placement.
    """
    if classes.ndim != 3 or classes.shape[0] != 18:
        raise ValueError("Expected official 18-channel CHW posterior")
    support = (classes.argmax(axis=0) != 0) & (classes.max(axis=0) >= 128)
    ys, xs = np.where(support)
    if not len(xs):
        return [], None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    h, w = support.shape
    iw, ih = image_size
    queries = []
    for row in range(8):
        for col in range(4):
            # Integer partition assigns every support pixel to exactly one cell.
            left = x0 + (x1 - x0) * col // 4
            right = x0 + (x1 - x0) * (col + 1) // 4
            top = y0 + (y1 - y0) * row // 8
            bottom = y0 + (y1 - y0) * (row + 1) // 8
            cy, cx = np.where(support[top:bottom, left:right])
            if not len(cx):
                continue
            cx, cy = cx + left, cy + top
            distances = (2 * cx + 1 - left - right) ** 2 + (2 * cy + 1 - top - bottom) ** 2
            selected = int(np.argmin(distances))
            px, py = int(cx[selected]), int(cy[selected])
            queries.append({"id": f"r{row}c{col}", "cell": [row, col],
                            "posteriorXY": [px, py],
                            "xy": [(px + 0.5) * iw / w, (py + 0.5) * ih / h],
                            "winner": int(classes[:, py, px].argmax())})
    return queries, [x0, y0, x1, y1]


def prepare(source_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    evidence = json.loads((source_dir / "class-evidence.json").read_text())
    assert evidence["inferenceMessages"] == 1
    assert sha(source_dir / "fashn-classes.u8") == evidence["classesSHA256"]
    assert sha(source_dir / "fashn-logits.f32") == evidence["logitsSHA256"]
    classes = np.fromfile(source_dir / "fashn-classes.u8", dtype=np.uint8).reshape(evidence["decodedDimensions"])
    source = Image.open(source_dir / "source.png").convert("RGB")
    queries, bbox = spatial_queries(classes, source.size)
    files = {"source": source_dir / "source.png", "posterior": source_dir / "fashn-classes.u8",
             "classEvidence": source_dir / "class-evidence.json", "sameInferenceLogits": source_dir / "fashn-logits.f32",
             "checkpoint": CHECKPOINT, "script": Path(__file__),
             "config": RUNTIME / "vendor/sam2" / CONFIG,
             "predictor": RUNTIME / "vendor/sam2/sam2_image_predictor.py",
             "vendorRevision": RUNTIME / "vendor-revision.json"}
    protocol = {"status": "frozen_before_candidate_inference", "sourceDirectory": str(source_dir.resolve()),
                "files": {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in files.items()},
                "imageSize": list(source.size), "posteriorSupportBox": bbox, "queries": queries,
                "queryLimit": 32, "rule": "4x8 non-background posterior-winner support grid; max posterior >=128/255; one nearest-center point per occupied cell; row-major ties; skip empty cells",
                "prompt": "single positive point; no box, negative point, mask input, manual polygon or output feedback",
                "model": "local SAM2Matting SAM2.1 Tiny checkpoint; only ordinary SAM2 segmentation predictor",
                "decode": {"multimask_output": True, "return_logits": False, "mask_threshold": 0,
                           "max_hole_area": 0, "max_sprinkle_area": 0, "postfilter": None},
                "limits": {"imageEncodings": 1, "alphaHeadCalls": 0, "additionalQueries": 0},
                "question": "Does this fixed candidate bank contain separately useful visible sleeves and coat/torso support?",
                "rivals": ["Part boundaries exist among the candidates but selection is missing", "SAM2 merges garment parts; candidate proposal itself is inadequate", "Fixed points miss a part; this bounded sampling route is inadequate even if another prompting protocol could succeed"],
                "acceptance": "Independent source-image review of every candidate, with no score ranking. Require separate visible sleeves and body garment support without materially wrong layer merging. This is candidate availability only, never alpha/hidden anatomy or deformation validation.",
                "stopRule": "If any required visible material lacks a useful candidate, fail this fixed candidate route; no query retuning, hand correction, new models, matting or deformation in this run."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    preview = source.copy()
    draw = ImageDraw.Draw(preview)
    for q in queries:
        x, y = q["xy"]
        draw.ellipse((x-5, y-5, x+5, y+5), fill="yellow", outline="black", width=2)
        draw.text((x+6, y), q["id"], fill="yellow", stroke_width=1, stroke_fill="black")
    preview.save(output / "source-queries.png")
    print(json.dumps({"protocol": str(output / "protocol.json"), "queries": len(queries), "protocolSha256": sha(output / "protocol.json")}))


def run(output):
    protocol = json.loads((output / "protocol.json").read_text())
    assert not (output / "run.json").exists(), "Never overwrite or rerun a candidate bank"
    for entry in protocol["files"].values():
        assert sha(entry["path"]) == entry["sha256"], entry["path"]
    assert 0 < len(protocol["queries"]) <= 32
    sys.path[:0] = [str(RUNTIME / "deps"), str(RUNTIME / "vendor")]
    import torch
    from sam2.build_sam import build_sam2matting
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    report = {"status": "running", "protocolSha256": sha(output / "protocol.json"),
              "torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 3,
              "imageEncodings": 0, "alphaHeadCalls": 0, "candidates": [], "selectedCandidates": None}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    model = build_sam2matting(CONFIG, str(CHECKPOINT), device="cpu")
    report["loadSeconds"] = time.perf_counter() - start
    forward_image = model.forward_image

    def counted_encoder(*args, **kwargs):
        report["imageEncodings"] += 1
        assert report["imageEncodings"] == 1
        return forward_image(*args, **kwargs)

    def forbidden_alpha(*args, **kwargs):
        report["alphaHeadCalls"] += 1
        raise AssertionError("Matting is outside this segmentation candidate experiment")

    model.forward_image = counted_encoder
    model._forward_alpha_heads = forbidden_alpha
    predictor = SAM2ImagePredictor(model, mask_threshold=0, max_hole_area=0, max_sprinkle_area=0)
    source = Image.open(protocol["files"]["source"]["path"]).convert("RGB")
    all_bits, all_logits, all_scores, cards = [], [], [], []
    width, height = source.size
    preview = np.array(source.resize((320, round(height * 320 / width))))
    (output / "masks").mkdir()
    with torch.inference_mode():
        start = time.perf_counter()
        predictor.set_image(source)
        report["encodeSeconds"] = time.perf_counter() - start
        print("encoded", report["encodeSeconds"], flush=True)
        for query in protocol["queries"]:
            start = time.perf_counter()
            masks, scores, logits = predictor.predict(point_coords=np.asarray([query["xy"]], dtype=np.float32),
                point_labels=np.ones(1, dtype=np.int32), multimask_output=True, return_logits=False)
            elapsed = time.perf_counter() - start
            assert masks.shape == (3, height, width)
            assert np.isfinite(scores).all() and np.isfinite(logits).all()
            for index in range(3):
                name = f'{query["id"]}-m{index}'
                mask = masks[index].astype(bool)
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_image.save(output / "masks" / f"{name}.png")
                all_bits.append(np.packbits(mask.reshape(-1)))
                all_logits.append(logits[index])
                all_scores.append(scores[index])
                report["candidates"].append({"id": name, "query": query["id"], "maskIndex": index,
                    "pixels": int(mask.sum()), "predictedIOUNotTruth": float(scores[index]),
                    "queryDecodeSeconds": elapsed, "sha256": sha(output / "masks" / f"{name}.png")})
                small = np.asarray(mask_image.resize((preview.shape[1], preview.shape[0]), Image.Resampling.NEAREST)) > 0
                shown = np.where(small[..., None], preview, preview * 0.18).astype(np.uint8)
                card = Image.new("RGB", (preview.shape[1], preview.shape[0] + 24), "#202020")
                card.paste(Image.fromarray(shown), (0, 24))
                ImageDraw.Draw(card).text((6, 6), f"{name}  pixels={int(mask.sum())}", fill="white")
                cards.append(card)
            print(query["id"], "candidates=3", "seconds=", round(elapsed, 3), flush=True)
    np.savez_compressed(output / "candidate-bank.npz", masksPacked=np.stack(all_bits),
                        lowResolutionLogits=np.stack(all_logits), predictedIOU=np.asarray(all_scores),
                        imageShape=np.asarray([height, width]))
    for offset in range(0, len(cards), 18):
        subset = cards[offset:offset+18]
        cw, ch = cards[0].size
        sheet = Image.new("RGB", (cw*3, ch*((len(subset)+2)//3)), "#202020")
        for n, card in enumerate(subset):
            sheet.paste(card, ((n%3)*cw, (n//3)*ch))
        sheet.save(output / f"candidates-{offset//18+1:02d}.jpg", quality=92)
    assert report["imageEncodings"] == 1 and report["alphaHeadCalls"] == 0
    report.update(status="candidate_bank_complete_independent_review_pending", queries=len(protocol["queries"]),
                  bankSha256=sha(output / "candidate-bank.npz"))
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source, args.output)
    else:
        run(args.output)
