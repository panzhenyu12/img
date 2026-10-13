"""Native same-process first/hot timings; frozen binary hash before/after spawn."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / ".run/skin-detail-v2-bench")
    parser.add_argument("--engine-binary", type=Path, default=ROOT / ".run/skin-detail-v2-image-render")
    args = parser.parse_args()
    expected = sha(args.binary)
    engine_sha = sha(args.engine_binary)
    rows = []
    for sid in ["fixture-portrait", "portrait-dark", "portrait-hair", "portrait-makeup", "portrait-roll", "portrait-woman"]:
        for case in ["smooth-strong", "whiten"]:
            assert sha(args.binary) == expected
            result = subprocess.run([str(args.binary), str(ROOT / f"test/results/native-face-contour/{sid}-source.png"),
                                     str(ROOT / f"test/results/skin-detail/{sid}-{case}.json")],
                                    capture_output=True, text=True, check=True)
            assert sha(args.binary) == expected
            row = json.loads(result.stdout)
            row.update(sample=sid, case=case)
            rows.append(row)
    assert sha(args.binary) == expected and sha(args.engine_binary) == engine_sha
    report = {"binary_sha256": expected, "engine_binary_sha256": engine_sha,
              "binary_checks": "sha256 before and after every spawn, then final hash",
              "model_initialization_ms": None, "model_detection_ms": None,
              "note": "No model invoked; simultaneous compilation and Chrome CPU load; not controlled acceptance timing.",
              "rows": rows}
    (ROOT / "doc/qa/skin-detail-timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"runs": len(rows), "hot_min_ms": min(r["hot_median_ms"] for r in rows),
                      "hot_max_ms": max(r["hot_median_ms"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
