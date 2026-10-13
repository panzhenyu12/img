#!/bin/sh
set -eu
background_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$background_dir"
if command -v uv >/dev/null 2>&1; then
  if [ ! -x .venv/bin/python ]; then
    uv python install 3.11
    uv venv --python 3.11 .venv
  fi
  uv pip install --python .venv/bin/python -r requirements.txt
else
  if [ ! -x .venv/bin/python ]; then python3.11 -m venv .venv; fi
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python scripts/download_models.py lama-torchscript
