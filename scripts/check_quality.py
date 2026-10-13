"""Run all real-photo diagnostics, including after an earlier suite fails."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
results = []
for args in [
    ["node", "scripts/check_portrait_quality.mjs", "--batch"],
    ["node", "scripts/check_body_quality.mjs"],
    ["node", "scripts/check_clothing_quality.mjs"],
]:
    results.append(subprocess.run(args, cwd=ROOT).returncode)
raise SystemExit(1 if any(results) else 0)
