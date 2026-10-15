"""Bounded V4b hotfix replay. Requires Pillow; preserves V4 acceptance evidence."""
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/results/feature-svf-v4b-20261014"
DOC = ROOT / "doc/qa/brow-flow-v4-20261014/hotfix-v4b"
BINARY = OUT / "image-render-final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source, graph, target):
    subprocess.run([str(BINARY), "--input", str(source), "--graph", str(graph),
                    "--output", str(target)], check=True, capture_output=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    frozen = sha(BINARY)
    old = json.loads((DOC.parent / "initial-accepted/report.json").read_text())
    ui = json.loads((DOC.parent / "initial-accepted/ui-parity.json").read_text())
    cases = json.loads((ROOT / "test/results/brow-flow-v4-20261014/cases-ui-final.json").read_text())
    by_label = {case["label"]: case for case in cases}
    requests = [(row["label"], ROOT / row["input"], ROOT / row["graph"],
                 ROOT / row["output"], row["outputSha256"]) for row in old["rows"]]
    requests += [(row["label"] + "-previous-ui", Path(by_label[row["label"]]["input"]),
                  Path(by_label[row["label"]]["graph"]), ROOT / row["nativePNG"],
                  sha(ROOT / row["nativePNG"])) for row in ui["rows"] if row["label"] != "user-eyeTilt-100"]
    unchanged = []
    for label, source, graph, previous, expected in requests:
        assert sha(previous) == expected, f"Old evidence changed: {label}"
        target = OUT / (label + "-unchanged-replay.png")
        render(source, graph, target)
        row = dict(label=label, unchanged=sha(target) == expected,
                   previous=str(previous.relative_to(ROOT)), sha256=sha(target),
                   sourceSha256=sha(source), graphSha256=sha(graph))
        unchanged.append(row)
        assert row["unchanged"], row
        print(label, "unchanged", flush=True)

    width = height = 96
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            alpha = 0 if (x + y) % 3 == 0 else 1 if (x + y) % 3 == 1 else (x * 5 + y * 17) % 256
            pixels.extend(((x * 11 + y * 3) % 256, (x * 7 + y * 13) % 256,
                           (x * 19 + y * 17) % 256, alpha))
    source = OUT / "varying-alpha-source.png"
    Image.frombytes("RGBA", (width, height), bytes(pixels)).save(source)
    raw = OUT / "varying-alpha-source.rgba"
    raw.write_bytes(pixels)

    def eye(x):
        return dict(shape="polygon", points=[dict(x=x, y=.3), dict(x=x+.2, y=.3),
                                             dict(x=x+.2, y=.4), dict(x=x, y=.4)])

    rectangle = dict(shape="rect", region=dict(x=.79, y=.79, width=.1, height=.1))
    alpha_cases = []
    for constraint in ["accessories", "hair", "ears", "protection", "selection"]:
        params = dict(adjustments=dict(eyeSize=100),
                      semanticMasks=dict(leftEye=eye(.25), rightEye=eye(.65)))
        if constraint == "protection":
            params["protectMask"] = rectangle
        elif constraint == "selection":
            params["mask"] = dict(shape="rect", region=dict(x=.15, y=.15, width=.75, height=.45))
        else:
            params["semanticMasks"][constraint] = rectangle
        graph = OUT / ("alpha-" + constraint + ".json")
        graph.write_text(json.dumps(dict(operations=[dict(type="feature.adjust", params=params)])))
        target = OUT / ("alpha-" + constraint + "-native.png")
        render(source, graph, target)
        actual = Image.open(target).convert("RGBA").tobytes()
        native_raw = target.with_suffix(".rgba")
        native_raw.write_bytes(actual)
        assert actual != pixels, "Must exercise a real edit"
        checked = 0
        for y in range(76, 85):
            for x in range(76, 85):
                i = (y * width + x) * 4
                assert actual[i:i+4] == pixels[i:i+4], f"Frozen {constraint}: {x},{y}"
                checked += 1
        alpha_cases.append(dict(label=constraint, width=width, height=height, sourceRaw=str(raw),
                                graph=str(graph), nativeRaw=str(native_raw), frozenPixelsChecked=checked))
    (OUT / "alpha-cases.json").write_text(json.dumps(alpha_cases, indent=2))
    assert sha(BINARY) == frozen, "Immutable worker mutated"
    (DOC / "unchanged-output-hashes.json").write_text(json.dumps(dict(
        scope="Fresh V4b native PNG replay against immutable accepted V4 PNG hashes. 30 brow outputs plus 18 prior UI cases; repeated inputs/controls are not independent samples. Old 42 mapping diagnostics are retained as historical evidence for unchanged zero-angle fields, not rerun.",
        nativeSha256=frozen, rows=unchanged), indent=2))


if __name__ == "__main__":
    main()
