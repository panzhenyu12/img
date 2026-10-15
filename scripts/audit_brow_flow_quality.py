"""Replay original-photo brow controls through immutable native workers.

Optional photographic QA; default Rust tests keep their own tracked detection
fixture and generated texture. No passing image alone establishes an 8/10 score.
"""
from pathlib import Path
import copy, hashlib, json, subprocess, time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/results/brow-flow-v4-20261014"
DOC = ROOT / "doc/qa/brow-flow-v4-20261014"
CURRENT = OUT / "image-render-final"
BASELINE = ROOT / "test/results/user-face-contour-20261014/baseline-image-render"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def render(binary, source, graph, output):
    before = sha(binary)
    started = time.monotonic()
    subprocess.run([str(binary), "--input", str(source), "--graph", str(graph), "--output", str(output)], check=True, capture_output=True)
    assert sha(binary) == before, "Worker mutated during replay"
    return (time.monotonic() - started) * 1000

def main():
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = json.loads((OUT / "cases-extended.json").read_text())[:28]
    # Four signed controls from the user's photo plus five other originals and
    # a roll derivative. Moderate controls are additional strengths, not sources.
    for key in ["browPeak", "browTail"]:
        case = copy.deepcopy(cases[0]); graph = json.loads(Path(case["graph"]).read_text())
        graph["operations"][0]["params"]["adjustments"] = {key: 40}
        case["label"] = f"user-{key}-40"; gp = OUT / (case["label"] + ".json"); gp.write_text(json.dumps(graph)); case["graph"] = str(gp); cases.append(case)
    for case in cases:
        label = case["label"]; source = Path(case["input"]); gp = Path(case["graph"])
        graph = json.loads(gp.read_text()); params = graph["operations"][0]["params"]
        baseline = OUT / (label + "-baseline.png"); current = OUT / (label + "-native.png")
        baseline_ms = render(BASELINE, source, gp, baseline); current_ms = render(CURRENT, source, gp, current)
        original = Image.open(source).convert("RGBA"); before = Image.open(baseline).convert("RGBA"); after = Image.open(current).convert("RGBA")
        names = ["leftEye", "rightEye", "leftBrow", "rightBrow"]
        points = [point for name in names for point in params["semanticMasks"][name]["points"]]
        bbox = (max(0, int(min(p["x"] for p in points) * original.width) - 24), max(0, int(min(p["y"] for p in points) * original.height) - 24), min(original.width, int(max(p["x"] for p in points) * original.width) + 24), min(original.height, int(max(p["y"] for p in points) * original.height) + 24))
        crops = [im.crop(bbox).convert("RGB") for im in [original, before, after]]
        sheet = Image.new("RGB", (crops[0].width * 3, crops[0].height + 28), "#181d24"); draw = ImageDraw.Draw(sheet)
        for i, (title, crop) in enumerate(zip(["Original", "Baseline", "SVF v4"], crops)):
            sheet.paste(crop, (i * crop.width, 28)); draw.text((i * crop.width + 7, 8), title, fill="white")
        comparison = DOC / (label + "-comparison.png"); sheet.save(comparison)
        a = np.asarray(original); b = np.asarray(after)
        rows.append(dict(label=label, input=str(source.relative_to(ROOT)), inputSha256=sha(source), graph=str(gp.relative_to(ROOT)), graphSha256=sha(gp), output=str(current.relative_to(ROOT)), outputSha256=sha(current), baselineProcessMs=round(baseline_ms, 2), nativeProcessMs=round(current_ms, 2), changedPixels=int(np.any(a[:,:,:3]!=b[:,:,:3],axis=2).sum()), alphaChanges=int((a[:,:,3]!=b[:,:,3]).sum()), comparison=str(comparison.relative_to(ROOT))))
        print(label, round(current_ms, 2), flush=True)
    diagnostic_rows = [{"label":c["label"], **json.loads(Path(c["diagnostics"]).read_text())} for c in json.loads((OUT / "cases-extended.json").read_text())]
    report = dict(scope="Immutable native full EditGraph PNG replay; total process times include I/O. Six independent original photos total, plus one derived roll; moderate strengths are not extra sources. Visual review is separate from field and byte tests.", workers={"baselineSha256":sha(BASELINE),"currentSha256":sha(CURRENT)}, sourceFiles={f.name:sha(f) for f in [ROOT/"core/image-core/src/geometry.rs", ROOT/"core/image-core/src/face.rs", ROOT/"core/image-core/src/color.rs"]}, rows=rows, exactMapperDiagnostics=diagnostic_rows)
    (DOC/"report.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
