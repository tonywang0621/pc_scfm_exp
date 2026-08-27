#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
DATA_ROOT="$ROOT_DIR/data/ecg_baseline_wander"
PTBXL_ROOT="$DATA_ROOT/raw/PTBXL"
NOISE_DIR="$DATA_ROOT/raw/NSTDB"

CONFIG="${CONFIG:-configs/ecg_baseline_wander_preprocess_common.yaml}"
RECORDS_DIR="${RECORDS_DIR:-$PTBXL_ROOT/records100}"
METADATA_CSV="${METADATA_CSV:-$PTBXL_ROOT/ptbxl_database.csv}"

cd "$APP_DIR"

cmd=(
  python3 preprocess_ecg.py
  --config "$CONFIG"
  --input-dir "$RECORDS_DIR"
  --metadata-csv "$METADATA_CSV"
  --dataset-name ptbxl
  --overwrite
)

if find "$NOISE_DIR" -type f -print -quit 2>/dev/null | grep -q .; then
  cmd+=(--noise-dir "$NOISE_DIR")
elif [[ "${ALLOW_SYNTHETIC_FALLBACK:-0}" == "1" ]]; then
  echo "WARNING: NSTDB is missing; using synthetic random_low_frequency_drift because ALLOW_SYNTHETIC_FALLBACK=1." >&2
  cmd+=(--baseline-kind random_low_frequency_drift)
else
  echo "ERROR: NSTDB files are required for MECG-E/EDDM-aligned baseline-wander preprocessing." >&2
  echo "Place NSTDB records under $NOISE_DIR, or set ALLOW_SYNTHETIC_FALLBACK=1 only for smoke tests." >&2
  exit 1
fi

"${cmd[@]}" "$@"
