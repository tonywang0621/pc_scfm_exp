#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
DATA_ROOT="$ROOT_DIR/data/mecge_table1_repro"
RUN_ROOT="$ROOT_DIR/runs/mecge_table1_repro"

NV="${NV:-1}"
DEVICE="${DEVICE:-cuda:0}"
PKL_FILE="${PKL_FILE:-$DATA_ROOT/raw/dataset_bw_nv${NV}.pkl}"
QTDB_RAW="${QTDB_RAW:-$DATA_ROOT/raw/QTDB}"
NSTDB_RAW="${NSTDB_RAW:-$DATA_ROOT/raw/NSTDB}"
FORCE_PREPARE_RAW="${FORCE_PREPARE_RAW:-0}"
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
  --qtdb-raw PATH        Raw QTDB WFDB directory for 100% DeepFilter/MECG-E prep.
  --nstdb-raw PATH       Raw NSTDB WFDB directory for 100% DeepFilter/MECG-E prep.
  --prepare-raw          Recreate dataset_bw_nv*.pkl from raw QTDB/NSTDB before training.
  --device DEVICE       Training/inference device. Default: cuda:0
  --model NAME          One of: all, main, stable, eddm_fm, eddm_1shot.
  --seed N              Run one seed only. Required for a single model/seed job.
  --skip-train          Only run robustness inference/aggregation from existing checkpoints.
  --skip-robustness     Only run train + QTDB pkl test.

Environment overrides:
  SEEDS_MAIN="42 43 44"
  SEEDS_EDDM="42"
  QTDB_RAW="data/mecge_table1_repro/raw/QTDB"
  NSTDB_RAW="data/mecge_table1_repro/raw/NSTDB"

Single-job examples:
  bash scripts/run_mecge_table1_repro.sh --model main --seed 42 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stable --seed 42 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_fm --seed 42 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_1shot --seed 42 --nv 1 --device cuda:0

100% DeepFilter/MECG-E raw-prep example:
  bash scripts/run_mecge_table1_repro.sh --model main --seed 42 --nv 1 --prepare-raw --device cuda:0
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
    --qtdb-raw)
      QTDB_RAW="$2"
      shift 2
      ;;
    --nstdb-raw)
      NSTDB_RAW="$2"
      shift 2
      ;;
    --prepare-raw)
      FORCE_PREPARE_RAW=1
      shift
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

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/controlled_tests" "$RUN_ROOT/analysis"

normalize_model() {
  case "$1" in
    all)
      printf '%s\n' "all"
      ;;
    main|mambattention|mambattention_stfrft_dualpath_dapp_cfm_unet_bd)
      printf '%s\n' "main"
      ;;
    stable|mambattention_stfrft_dualpath_dapp_stable_cfm_unet)
      printf '%s\n' "stable"
      ;;
    eddm_fm|eddm_flow_matching)
      printf '%s\n' "eddm_fm"
      ;;
    eddm|eddm_1shot)
      printf '%s\n' "eddm_1shot"
      ;;
    *)
      echo "Unsupported --model '$1'. Expected one of: all, main, stable, eddm_fm, eddm_1shot." >&2
      exit 2
      ;;
  esac
}

prepare_raw_if_needed() {
  local rnd_test="$DATA_ROOT/raw/rnd_test_nv${NV}.npy"
  if [[ "$FORCE_PREPARE_RAW" != "1" && -f "$PKL_FILE" && ( "$SKIP_ROBUSTNESS" == "1" || -f "$rnd_test" ) ]]; then
    return 0
  fi
  if [[ -z "$QTDB_RAW" || -z "$NSTDB_RAW" ]]; then
    echo "Missing MECG-E pkl file: $PKL_FILE" >&2
    echo "Provide --pkl-file, or provide --qtdb-raw and --nstdb-raw with --prepare-raw." >&2
    exit 1
  fi
  (
    cd "$APP_DIR"
    python3 mecge_table1_prepare_deepfilter.py \
      --qtdb-root "$QTDB_RAW" \
      --nstdb-root "$NSTDB_RAW" \
      --output-dir "$DATA_ROOT/raw" \
      --noise-version "$NV" \
      --overwrite
  )
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
  local metrics_per_window="$RUN_ROOT/$result_model/results/$exp_name_train/$model_name/best_loss/metrics_per_window.csv"
  local rnd_test="$DATA_ROOT/raw/rnd_test_nv${NV}.npy"
  if [[ ! -f "$metrics_per_window" ]]; then
    echo "Missing QTDB test per-window metrics for robustness bins: $metrics_per_window" >&2
    exit 1
  fi
  if [[ ! -f "$rnd_test" ]]; then
    echo "Missing rnd_test for 100% MECG-E/DeepFilter robustness bins: $rnd_test" >&2
    echo "Run with --prepare-raw, or provide pkl generated by mecge_table1_prepare_deepfilter.py." >&2
    exit 1
  fi
  (
    cd "$APP_DIR"
    python3 mecge_table1_robustness_bins.py \
      --metrics-per-window "$metrics_per_window" \
      --rnd-test "$rnd_test" \
      --output-root "$RUN_ROOT/$result_model/controlled_tests" \
      --result-model "$result_model" \
      --noise-version "nv${NV}" \
      --seed "$seed"
  )
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

prepare_raw_if_needed

MAIN_CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_cfm_unet_bd.yaml"
MAIN_RESULT_MODEL="mambattention_stfrft_dualpath_dapp_cfm_unet_bd"
MAIN_MODEL_NAME="mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg"
STABLE_CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_stable_cfm_unet.yaml"
STABLE_RESULT_MODEL="mambattention_stfrft_dualpath_dapp_stable_cfm_unet"
STABLE_MODEL_NAME="mambattention_stfrft_dualpath_dapp_stable_cfm_unet_ecg"
EDDM_FM_CONFIG="configs/mecge_table1_repro_eddm_flow_matching.yaml"
EDDM_FM_RESULT_MODEL="eddm_flow_matching"
EDDM_FM_MODEL_NAME="eddm_flow_matching"
EDDM_CONFIG="configs/mecge_table1_repro_eddm_1shot.yaml"
EDDM_RESULT_MODEL="eddm_1shot"
EDDM_MODEL_NAME="eddm"

case "$TARGET_MODEL" in
  all)
    if [[ -n "$TARGET_SEED" ]]; then
      echo "--seed can only be used with --model main, stable, eddm_fm, or eddm_1shot." >&2
      exit 2
    fi
    run_model_family "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "$SEEDS_MAIN"
    run_model_family "$STABLE_CONFIG" "$STABLE_RESULT_MODEL" "$STABLE_MODEL_NAME" "$SEEDS_MAIN"
    run_model_family "$EDDM_FM_CONFIG" "$EDDM_FM_RESULT_MODEL" "$EDDM_FM_MODEL_NAME" "$SEEDS_MAIN"
    run_model_family "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "$SEEDS_EDDM"
    ;;
  main)
    if [[ -z "$TARGET_SEED" ]]; then
      echo "--model main requires --seed N for a single model/seed job." >&2
      exit 2
    fi
    run_one_job "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "$TARGET_SEED"
    ;;
  stable)
    if [[ -z "$TARGET_SEED" ]]; then
      echo "--model stable requires --seed N for a single model/seed job." >&2
      exit 2
    fi
    run_one_job "$STABLE_CONFIG" "$STABLE_RESULT_MODEL" "$STABLE_MODEL_NAME" "$TARGET_SEED"
    ;;
  eddm_fm)
    if [[ -z "$TARGET_SEED" ]]; then
      echo "--model eddm_fm requires --seed N for a single model/seed job." >&2
      exit 2
    fi
    run_one_job "$EDDM_FM_CONFIG" "$EDDM_FM_RESULT_MODEL" "$EDDM_FM_MODEL_NAME" "$TARGET_SEED"
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
