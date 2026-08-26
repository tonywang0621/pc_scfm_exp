#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
RUN_ROOT="$ROOT_DIR/runs/ecg_baseline_wander"
EPOCH_ROOT="$RUN_ROOT/epoch40"
DATA_DIR="$ROOT_DIR/data/ecg_baseline_wander/processed"
DATA_ROOT="$ROOT_DIR/data/ecg_baseline_wander"
PTBXL_ROOT="$DATA_ROOT/raw/PTBXL"
MITBIH_ROOT="$DATA_ROOT/raw/MITBIH"
if [[ ! -d "$MITBIH_ROOT" ]]; then
  MITBIH_ROOT="$DATA_ROOT/raw/MIT-BIH"
fi
CHAPMAN_ROOT="$DATA_ROOT/raw/Chapman"
CPSC_ROOT="$DATA_ROOT/raw/CPSC"
QTDB_ROOT="$DATA_ROOT/raw/QTDB"
NOISE_DIR="$DATA_ROOT/raw/NSTDB"

BATCH_SIZE="64"
DEVICE=""
FORCE=0
SKIP_INFERENCE=0
SKIP_ROBUSTNESS=0
SKIP_COMPLEXITY=0
LIMIT=""
ALPHA_VALUES="0.05,0.1,0.2,0.3,0.5"
EXP2_BASELINE_KIND="nstdb"
DATASETS="ptbxl,mit_bih,chapman,cpsc,qtdb"
CHECKPOINT_STEP="58080"
CHECKPOINT_EPOCH="40"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_epoch40_model_eval.sh MODEL_KEY [options]

Runs one model's epoch-40 / step-58080 checkpoint through:
  1. in-domain and out-domain inference metrics
  2. Experiment 2 baseline-strength robustness on four datasets
  3. model complexity profiling
  4. one AI-friendly aggregate CSV

Outputs are written under:
  runs/ecg_baseline_wander/epoch40/

Options:
  --batch-size N          Inference batch size. Default: 64
  --device DEVICE         e.g. cuda:0 or cpu. Default: auto
  --force                 Re-run and overwrite existing outputs
  --skip-existing         Skip existing outputs. Default behavior
  --skip-inference        Do not run regular inference
  --skip-robustness       Do not run robustness strength
  --skip-complexity       Do not run complexity profiling
  --limit N               Limit generated robustness windows for smoke tests
  --alpha-values CSV      Robustness strength alpha values. Default: 0.05,0.1,0.2,0.3,0.5
  --baseline-kind KIND    Robustness baseline kind. Default: nstdb
  --datasets CSV          Datasets to run. Default: ptbxl,mit_bih,chapman,cpsc,qtdb
  --checkpoint-step N     Checkpoint step. Default: 58080
  --checkpoint-epoch N    Label used in aggregate. Default: 40
  -h, --help              Show this help

MODEL_KEY must match scripts/experiment_models.sh.
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_KEY="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --skip-existing)
      FORCE=0
      shift
      ;;
    --skip-inference)
      SKIP_INFERENCE=1
      shift
      ;;
    --skip-robustness)
      SKIP_ROBUSTNESS=1
      shift
      ;;
    --skip-complexity)
      SKIP_COMPLEXITY=1
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --alpha-values)
      ALPHA_VALUES="$2"
      shift 2
      ;;
    --baseline-kind)
      EXP2_BASELINE_KIND="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --checkpoint-step)
      CHECKPOINT_STEP="$2"
      shift 2
      ;;
    --checkpoint-epoch)
      CHECKPOINT_EPOCH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

source "$ROOT_DIR/scripts/experiment_models.sh"

MODEL_CONFIG=""
MODEL_EXP_NAME=""
MODEL_DIR=""
for item in "${EXPERIMENT_MODELS[@]}"; do
  IFS="|" read -r model_key config exp_name model_dir <<< "$item"
  if [[ "$model_key" == "$MODEL_KEY" ]]; then
    MODEL_CONFIG="$config"
    MODEL_EXP_NAME="$exp_name"
    MODEL_DIR="$model_dir"
    break
  fi
done

if [[ -z "$MODEL_CONFIG" ]]; then
  echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
  echo "See scripts/experiment_models.sh for supported keys." >&2
  exit 2
fi

CHECKPOINT="$RUN_ROOT/checkpoint/$MODEL_EXP_NAME/$MODEL_DIR/model_step_${CHECKPOINT_STEP}.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

PROCESSED_DATASETS=(
  "ptbxl_fold10|$DATA_DIR/test.npz"
  "mit_bih|$DATA_DIR/mit_bih.npz"
  "chapman|$DATA_DIR/chapman.npz"
  "cpsc|$DATA_DIR/cpsc.npz"
  "qtdb|$DATA_DIR/qtdb.npz"
)

dataset_selected() {
  local requested_key="$1"
  IFS=',' read -ra requested_datasets <<< "$DATASETS"
  for requested in "${requested_datasets[@]}"; do
    requested="${requested//[[:space:]]/}"
    if [[ "$requested" == "$requested_key" ]]; then
      return 0
    fi
    if [[ "$requested" == "ptbxl" && "$requested_key" == "ptbxl_fold10" ]]; then
      return 0
    fi
  done
  return 1
}

have_nstdb() {
  find "$NOISE_DIR" -type f -print -quit 2>/dev/null | grep -q .
}

dataset_args() {
  local dataset="$1"
  case "$dataset" in
    ptbxl|ptbxl_fold10)
      if [[ -d "$PTBXL_ROOT/records100" && -f "$PTBXL_ROOT/ptbxl_database.csv" ]]; then
        printf '%s\n' ptbxl ptbxl test "$PTBXL_ROOT/records100" "$PTBXL_ROOT/ptbxl_database.csv"
        return 0
      fi
      return 1
      ;;
    mit_bih|mit-bih|mitbih)
      if [[ -d "$MITBIH_ROOT" ]]; then
        local metadata="$MITBIH_ROOT/metadata.csv"
        [[ -f "$metadata" ]] || metadata="__none__"
        printf '%s\n' mit_bih mit_bih mit_bih "$MITBIH_ROOT" "$metadata"
        return 0
      fi
      return 1
      ;;
    chapman)
      if [[ -d "$CHAPMAN_ROOT" ]]; then
        local metadata="$CHAPMAN_ROOT/metadata.csv"
        [[ -f "$metadata" ]] || metadata="__none__"
        printf '%s\n' chapman chapman chapman "$CHAPMAN_ROOT" "$metadata"
        return 0
      fi
      return 1
      ;;
    cpsc)
      if [[ -d "$CPSC_ROOT" ]]; then
        local metadata="$CPSC_ROOT/metadata.csv"
        [[ -f "$metadata" ]] || metadata="__none__"
        printf '%s\n' cpsc cpsc cpsc "$CPSC_ROOT" "$metadata"
        return 0
      fi
      return 1
      ;;
    qtdb|qt|qtdb-test)
      if [[ -d "$QTDB_ROOT" ]]; then
        local metadata="$QTDB_ROOT/metadata.csv"
        [[ -f "$metadata" ]] || metadata="__none__"
        printf '%s\n' qtdb qtdb qtdb "$QTDB_ROOT" "$metadata"
        return 0
      fi
      return 1
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
}

run_inference_if_available() {
  local dataset_key="$1"
  local dataset_path="$2"
  local output_dir="$EPOCH_ROOT/inference/${MODEL_KEY}_${dataset_key}"

  if ! dataset_selected "$dataset_key"; then
    return 0
  fi
  if [[ ! -f "$dataset_path" ]]; then
    echo "SKIP inference: $MODEL_KEY / $dataset_key: missing dataset $dataset_path"
    return 0
  fi
  if [[ "$FORCE" -eq 0 && -f "$output_dir/metrics_summary.csv" ]]; then
    echo "SKIP inference: $MODEL_KEY / $dataset_key: existing $output_dir/metrics_summary.csv"
    return 0
  fi
  if [[ "$FORCE" -eq 1 && -d "$output_dir" ]]; then
    echo "OVERWRITE inference: removing existing $output_dir"
    rm -rf "$output_dir"
  fi

  echo "RUN inference: $MODEL_KEY | $dataset_key"
  mkdir -p "$output_dir"
  local cmd=(
    python3 inference.py
    --config "$MODEL_CONFIG"
    --checkpoint "$CHECKPOINT"
    --input "$dataset_path"
    --output-dir "$output_dir"
    --batch-size "$BATCH_SIZE"
  )
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  "${cmd[@]}"
}

run_robustness_if_available() {
  local dataset="$1"
  local output_root="$EPOCH_ROOT/controlled_tests/$MODEL_KEY"

  if ! mapfile -t ds_args < <(dataset_args "$dataset"); then
    echo "SKIP robustness: $MODEL_KEY / $dataset: missing raw data"
    return 0
  fi
  if [[ "${#ds_args[@]}" -lt 5 ]]; then
    echo "SKIP robustness: $MODEL_KEY / $dataset: missing raw data"
    return 0
  fi
  if [[ "$EXP2_BASELINE_KIND" == "nstdb" ]] && ! have_nstdb; then
    echo "SKIP robustness: $MODEL_KEY / $dataset: missing NSTDB files under $NOISE_DIR"
    return 0
  fi

  local dataset_name="${ds_args[0]}"
  local dataset_label="${ds_args[1]}"
  local split_name="${ds_args[2]}"
  local input_dir="${ds_args[3]}"
  local metadata_csv="${ds_args[4]}"
  local summary="$output_root/exp2_strength/$dataset_label/summary.csv"

  if [[ "$FORCE" -eq 0 && -f "$summary" ]]; then
    echo "SKIP robustness: $MODEL_KEY / $dataset_label: existing $summary"
    return 0
  fi

  echo "RUN robustness strength: $MODEL_KEY | $dataset_label"
  local cmd=(
    python3 experiment_suite.py exp2-strength
    --config "$MODEL_CONFIG"
    --input-dir "$input_dir"
    --dataset-name "$dataset_name"
    --dataset-label "$dataset_label"
    --split-name "$split_name"
    --checkpoint "$CHECKPOINT"
    --output-root "$output_root"
    --batch-size "$BATCH_SIZE"
    --baseline-kind "$EXP2_BASELINE_KIND"
    --alpha-values "$ALPHA_VALUES"
  )
  if [[ "$metadata_csv" != "__none__" ]]; then
    cmd+=(--metadata-csv "$metadata_csv")
  fi
  if [[ "$EXP2_BASELINE_KIND" == "nstdb" ]]; then
    cmd+=(--noise-dir "$NOISE_DIR")
  fi
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  "${cmd[@]}"
}

run_complexity_if_needed() {
  local output_dir="$EPOCH_ROOT/complexity/$MODEL_KEY"
  if [[ "$FORCE" -eq 0 && -f "$output_dir/complexity_summary.csv" ]]; then
    echo "SKIP complexity: $MODEL_KEY: existing $output_dir/complexity_summary.csv"
    return 0
  fi
  if [[ "$FORCE" -eq 1 && -d "$output_dir" ]]; then
    echo "OVERWRITE complexity: removing existing $output_dir"
    rm -rf "$output_dir"
  fi

  echo "RUN complexity: $MODEL_KEY"
  mkdir -p "$output_dir"
  local cmd=(
    python3 profile_checkpoint_complexity.py
    --config "$MODEL_CONFIG"
    --checkpoint "$CHECKPOINT"
    --output-dir "$output_dir"
  )
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  "${cmd[@]}"
}

write_aggregate() {
  local aggregate="$EPOCH_ROOT/aggregate/epoch40_all_summary.csv"
  mkdir -p "$(dirname "$aggregate")"
  python3 aggregate_epoch40.py \
    --run-root "$EPOCH_ROOT" \
    --metadata-csv "$EPOCH_ROOT/metadata/epoch40_checkpoints.csv" \
    --output "$aggregate"
}

write_checkpoint_metadata() {
  local metadata="$EPOCH_ROOT/metadata/epoch40_checkpoints.csv"
  mkdir -p "$(dirname "$metadata")"
  python3 - "$metadata" "$MODEL_KEY" "$CHECKPOINT_EPOCH" "$CHECKPOINT_STEP" "$CHECKPOINT" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
new_row = {
    "model_key": sys.argv[2],
    "checkpoint_epoch": sys.argv[3],
    "checkpoint_step": sys.argv[4],
    "checkpoint_path": sys.argv[5],
}
rows = {}
if path.exists():
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model_key"):
                rows[row["model_key"]] = row
rows[new_row["model_key"]] = new_row
with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model_key", "checkpoint_epoch", "checkpoint_step", "checkpoint_path"],
    )
    writer.writeheader()
    for key in sorted(rows):
        writer.writerow(rows[key])
PY
}

mkdir -p "$EPOCH_ROOT/logs"
LOG_PATH="$EPOCH_ROOT/logs/${MODEL_KEY}_epoch${CHECKPOINT_EPOCH}_step${CHECKPOINT_STEP}_eval.log"
exec > >(tee -a "$LOG_PATH") 2>&1

cd "$APP_DIR"

echo "MODEL_KEY=$MODEL_KEY"
echo "CONFIG=$MODEL_CONFIG"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_ROOT=$EPOCH_ROOT"
echo "LOG=$LOG_PATH"

write_checkpoint_metadata

if [[ "$SKIP_INFERENCE" -eq 0 ]]; then
  for item in "${PROCESSED_DATASETS[@]}"; do
    IFS="|" read -r dataset_key dataset_path <<< "$item"
    run_inference_if_available "$dataset_key" "$dataset_path"
  done
fi

if [[ "$SKIP_ROBUSTNESS" -eq 0 ]]; then
  IFS=',' read -ra requested_datasets <<< "$DATASETS"
  for dataset in "${requested_datasets[@]}"; do
    dataset="${dataset//[[:space:]]/}"
    run_robustness_if_available "$dataset"
  done
fi

if [[ "$SKIP_COMPLEXITY" -eq 0 ]]; then
  run_complexity_if_needed
fi

write_aggregate
echo "Done. Aggregate: $EPOCH_ROOT/aggregate/epoch40_all_summary.csv"
