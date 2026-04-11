#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv312}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/df_kwh_adjusted.csv}"
MODE="${MODE:-smoke}"
DEVICE="${DEVICE:-cuda}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/bootstrap_${MODE}}"

echo "[1/5] Preparing Python environment at $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

mkdir -p "$DATA_DIR"

if [[ "$SKIP_DOWNLOAD" != "1" && ! -f "$DATA_PATH" ]]; then
  echo "[2/5] Downloading Kaggle ELD mirror to $DATA_PATH"
  "$VENV_DIR/bin/python" - <<PY
from pathlib import Path
import shutil

import kagglehub

root = Path(kagglehub.dataset_download("eduardojst10/electricityloaddiagrams20112014"))
source = root / "df_kwh_adjusted.csv"
target = Path("$DATA_PATH")
target.parent.mkdir(parents=True, exist_ok=True)
if not source.exists():
    raise FileNotFoundError(f"Expected Kaggle dataset file at {source}")
if source.resolve() != target.resolve():
    shutil.copy2(source, target)
print(target)
PY
else
  echo "[2/5] Using existing dataset at $DATA_PATH"
fi

echo "[3/5] Checking CUDA availability"
"$VENV_DIR/bin/python" - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
PY

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "[4/5] Running smoke test"
  "$VENV_DIR/bin/python" -m src.pipeline.run_lira \
    --synthetic \
    --synthetic-users 10 \
    --synthetic-steps 240 \
    --epochs 2 \
    --num-shadow 2 \
    --num-runs 1 \
    --batch-size 64 \
    --num-train-users 2 \
    --num-val-users 2 \
    --num-test-users 2 \
    --num-aux-users 4 \
    --device "$DEVICE" \
    --output-dir "$ROOT_DIR/artifacts/bootstrap_smoke"
else
  echo "[4/5] Skipping smoke test"
fi

if [[ "$MODE" == "full" ]]; then
  echo "[5/5] Launching full ELD fallback run"
  "$VENV_DIR/bin/python" -m src.pipeline.run_lira \
    --paper-preset eld_lstm_record_lira_offline_mse_kaggle_fallback \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR"
else
  echo "[5/5] Bootstrap complete. Set MODE=full to launch the full experiment."
fi
