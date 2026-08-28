#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
DATA_ROOT="$ROOT_DIR/data/mecge_table1_repro"
RUN_ROOT="$ROOT_DIR/runs/mecge_table1_repro"

NV="${NV:-1}"
DEVICE="${DEVICE:-cuda:0}"
PKL_FILE="${PKL_FILE:-$DATA_ROOT/raw/dataset_bw_nv${NV}.pkl}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ROBUSTNESS="${SKIP_ROBUSTNESS:-0}"
SEEDS_MAIN="${SEEDS_MAIN:-42 43 44}"
SEEDS_EDDM="${SEEDS_EDDM:-42}"
ALPHAS="${ALPHAS:-0.2 0.6 1.0 1.5 2.0}"
TARGET_MODEL="${TARGET_MODEL:-all}"
TARGET_SEED="${TARGET_SEED:-}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_mecge_table1_repro.sh [options]

Options:
  --nv N                MECG-E noise version. Default: 1
  --pkl-file PATH       Official MECG-E dataset_bw_nv*.pkl path.
  --device DEVICE       Training/inference device. Default: cuda:0
  --model NAME          One of: all, main, eddm_1shot.
  --seed N              Run one seed only. Required for a single model/seed job.
  --skip-train          Only run robustness inference/aggregation from existing checkpoints.
  --skip-robustness     Only run train + QTDB pkl test.

Environment overrides:
  SEEDS_MAIN="42 43 44"
  SEEDS_EDDM="42"
  ALPHAS="0.2 0.6 1.0 1.5 2.0"

Single-job examples:
  bash scripts/run_mecge_table1_repro.sh --model main --seed 42 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_1shot --seed 42 --nv 1 --device cuda:0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nv)
      NV="$2"
      PKL_FILE="$DATA_ROOT/raw/dataset_bw_nv${NV}.pkl"
      shift 2
      ;;
    --pkl-file)
      PKL_FILE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --model)
      TARGET_MODEL="$2"
      shift 2
      ;;
    --seed)
      TARGET_SEED="$2"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-robustness)
      SKIP_ROBUSTNESS=1
      shift
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

if [[ ! -f "$PKL_FILE" ]]; then
  echo "Missing MECG-E pkl file: $PKL_FILE" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/controlled_tests" "$RUN_ROOT/analysis"

normalize_model() {
  case "$1" in
    all)
      printf '%s\n' "all"
      ;;
    main|mambattention|mambattention_stfrft_dualpath_dapp_cfm_unet_bd)
      printf '%s\n' "main"
      ;;
    eddm|eddm_1shot)
      printf '%s\n' "eddm_1shot"
      ;;
    *)
      echo "Unsupported --model '$1'. Expected one of: all, main, eddm_1shot." >&2
      exit 2
      ;;
  esac
}

TARGET_MODEL="$(normalize_model "$TARGET_MODEL")"

run_train() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local exp_name="$5"

  if [[ "$SKIP_TRAIN" == "1" ]]; then
    return 0
  fi

  (
    cd "$APP_DIR"
    python3 train_supervised.py \
      --config "$config" \
      "data_dir=$DATA_ROOT" \
      "root_dir=$RUN_ROOT/$result_model" \
      "checkpoint_dir=\${root_dir}/checkpoint" \
      "results_dir=\${root_dir}/results" \
      "log_dir=\${root_dir}/log" \
      "dataset.pkl_file=$PKL_FILE" \
      "dataset.test_label=qtdb_pkl_test" \
      "exp_name=$exp_name" \
      "seed=$seed" \
      "device=$DEVICE" \
      "training.resume=false"
  )
}

checkpoint_path() {
  local result_model="$1"
  local model_name="$2"
  local exp_name="$3"
  printf '%s\n' "$RUN_ROOT/$result_model/checkpoint/$exp_name/$model_name/best_model.pt"
}

alpha_label() {
  printf '%s\n' "$1" | sed 's/\./p/g; s/-/m/g'
}

run_exp2_generation() {
  if [[ "$SKIP_ROBUSTNESS" == "1" ]]; then
    return 0
  fi
  (
    cd "$APP_DIR"
    python3 mecge_table1_exp2_from_pkl.py \
      --pkl-file "$PKL_FILE" \
      --output-root "$DATA_ROOT/controlled_tests/nv${NV}" \
      --alpha-values $ALPHAS \
      --seed 42 \
      --overwrite
  )
}

run_exp2_inference() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local exp_name_train="$5"
  local ckpt
  ckpt="$(checkpoint_path "$result_model" "$model_name" "$exp_name_train")"

  if [[ "$SKIP_ROBUSTNESS" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint for robustness: $ckpt" >&2
    exit 1
  fi

  for alpha in $ALPHAS; do
    local label
    label="$(alpha_label "$alpha")"
    local input_npz="$DATA_ROOT/controlled_tests/nv${NV}/exp2_strength/qtdb_pkl_test/alpha_${label}/processed/test.npz"
    local result_name="${result_model}__qtdb_robustness_alpha_${label}__nv${NV}__seed${seed}"
    local output_dir="$RUN_ROOT/$result_model/controlled_tests/$result_name"
    (
      cd "$APP_DIR"
      python3 inference.py \
        --config "$config" \
        --checkpoint "$ckpt" \
        --input "$input_npz" \
        --output-dir "$output_dir" \
        --batch-size 64 \
        --device "$DEVICE"
    )
  done
}

run_one_job() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local exp_name="${result_model}__qtdb_train_qtdb_test__nv${NV}__seed${seed}"
  echo "RUN job: model=$result_model seed=$seed nv=nv${NV}"
  run_train "$config" "$result_model" "$model_name" "$seed" "$exp_name"
  run_exp2_inference "$config" "$result_model" "$model_name" "$seed" "$exp_name"
}

run_model_family() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seeds="$4"
  for seed in $seeds; do
    run_one_job "$config" "$result_model" "$model_name" "$seed"
  done
}

run_exp2_generation

MAIN_CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_cfm_unet_bd.yaml"
MAIN_RESULT_MODEL="mambattention_stfrft_dualpath_dapp_cfm_unet_bd"
MAIN_MODEL_NAME="mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg"
EDDM_CONFIG="configs/mecge_table1_repro_eddm_1shot.yaml"
EDDM_RESULT_MODEL="eddm_1shot"
EDDM_MODEL_NAME="eddm"

case "$TARGET_MODEL" in
  all)
    if [[ -n "$TARGET_SEED" ]]; then
      echo "--seed can only be used with --model main or --model eddm_1shot." >&2
      exit 2
    fi
    run_model_family "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "$SEEDS_MAIN"
    run_model_family "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "$SEEDS_EDDM"
    ;;
  main)
    if [[ -z "$TARGET_SEED" ]]; then
      echo "--model main requires --seed N for a single model/seed job." >&2
      exit 2
    fi
    run_one_job "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "$TARGET_SEED"
    ;;
  eddm_1shot)
    if [[ -z "$TARGET_SEED" ]]; then
      echo "--model eddm_1shot requires --seed N for a single model/seed job." >&2
      exit 2
    fi
    run_one_job "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "$TARGET_SEED"
    ;;
esac

(
  cd "$APP_DIR"
  python3 mecge_table1_collect_results.py \
    --run-root "$RUN_ROOT" \
    --noise-version "nv${NV}" \
    --output-dir "$RUN_ROOT/analysis"
)
