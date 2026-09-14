#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
DESCOD_DIR="${DESCOD_DIR:-$ROOT_DIR/references/Score-based-ECG-Denoising}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/work/tonyalpha1/pc_scfm_exp/runs/mecge_table1_repro/complexity_test}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INPUT_LENGTH="${INPUT_LENGTH:-512}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-20}"
TARGET_MODEL="${TARGET_MODEL:-all}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_mecge_table1_complexity_test.sh [options]

Options:
  --model NAME          One of: all, mecge, mecge_no_early_stop, mambattention, dualpath_dapp_cfm_unet_bd,
                        dualpath_dapp_cfm_unet_bd_step3, dualpath_dapp_cfm_unet_bd_step4,
                        dualpath_dapp_cfm_unet_bd_step5, dualpath_dapp_cfm_unet_bd_step8,
                        dualpath_dapp_cfm_unet_bd_no_attention,
                        lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience,
                        stfrft, main, stable, eddm_fm, eddm_fm_mamba,
                        eddm_1shot, eddm_3shot, eddm_5shot, eddm_10shot,
                        descod_1shot, descod_3shot, descod_5shot, descod_10shot.
  --output-root PATH    Output directory. Default:
                        /work/tonyalpha1/pc_scfm_exp/runs/mecge_table1_repro/complexity_test
  --device DEVICE       Profiling device. Default: cuda:0
  --batch-size N        Dummy input batch size. Default: 1
  --input-length N      Dummy ECG window length. Default: 512
  --warmup N            Latency warmup repeats. Default: 5
  --repeats N           Latency timing repeats. Default: 20
  --force               Recompute when output YAML already exists.
  -h, --help            Show this help.

Examples:
  bash scripts/run_mecge_table1_complexity_test.sh --model all --device cuda:0
  bash scripts/run_mecge_table1_complexity_test.sh --model dualpath_dapp_cfm_unet_bd_step5 --device cuda:0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      TARGET_MODEL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --input-length)
      INPUT_LENGTH="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --force)
      FORCE=1
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

normalize_model() {
  case "$1" in
    all)
      printf '%s\n' "all"
      ;;
    mecge|mecg_e)
      printf '%s\n' "mecge"
      ;;
    mecge_no_early_stop|mecg_e_no_early_stop|mecge_official_30epoch)
      printf '%s\n' "mecge_no_early_stop"
      ;;
    mambattention|mambattention_ecg)
      printf '%s\n' "mambattention"
      ;;
    dualpath_dapp_cfm_unet_bd|mambattention_dualpath_dapp_cfm_unet_bd|mambattention_dualpath_dapp_cfm_unet_bd_ecg)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd"
      ;;
    dualpath_dapp_cfm_unet_bd_step3|mambattention_dualpath_dapp_cfm_unet_bd_step3)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd_step3"
      ;;
    dualpath_dapp_cfm_unet_bd_step4|mambattention_dualpath_dapp_cfm_unet_bd_step4)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd_step4"
      ;;
    dualpath_dapp_cfm_unet_bd_step5|mambattention_dualpath_dapp_cfm_unet_bd_step5)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd_step5"
      ;;
    dualpath_dapp_cfm_unet_bd_step8|mambattention_dualpath_dapp_cfm_unet_bd_step8)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd_step8"
      ;;
    dualpath_dapp_cfm_unet_bd_no_attention|mambattention_dualpath_dapp_cfm_unet_bd_no_attention)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd_no_attention"
      ;;
    lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience|dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience)
      printf '%s\n' "lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience"
      ;;
    stfrft|mambattention_stfrft|mambattention_stfrft_ecg)
      printf '%s\n' "stfrft"
      ;;
    main|mambattention_stfrft_dualpath_dapp_cfm_unet_bd)
      printf '%s\n' "main"
      ;;
    stable|mambattention_stfrft_dualpath_dapp_stable_cfm_unet)
      printf '%s\n' "stable"
      ;;
    eddm_fm|eddm_flow_matching)
      printf '%s\n' "eddm_fm"
      ;;
    eddm_fm_mamba|eddm_flow_matching_mamba)
      printf '%s\n' "eddm_fm_mamba"
      ;;
    eddm|eddm_1shot|eddm-1|eddm_1)
      printf '%s\n' "eddm_1shot"
      ;;
    eddm_3shot|eddm-3|eddm_3)
      printf '%s\n' "eddm_3shot"
      ;;
    eddm_5shot|eddm-5|eddm_5)
      printf '%s\n' "eddm_5shot"
      ;;
    eddm_10shot|eddm-10|eddm_10)
      printf '%s\n' "eddm_10shot"
      ;;
    descod|descod_1shot|descod-1|descod1)
      printf '%s\n' "descod_1shot"
      ;;
    descod_3shot|descod-3|descod3)
      printf '%s\n' "descod_3shot"
      ;;
    descod_5shot|descod-5|descod5)
      printf '%s\n' "descod_5shot"
      ;;
    descod_10shot|descod-10|descod10)
      printf '%s\n' "descod_10shot"
      ;;
    *)
      echo "Unsupported --model '$1'." >&2
      usage >&2
      exit 2
      ;;
  esac
}

config_for_model() {
  case "$1" in
    mecge)
      printf '%s\n' "configs/mecge_table1_repro_mecg_e.yaml"
      ;;
    mecge_no_early_stop)
      printf '%s\n' "configs/mecge_table1_repro_mecg_e.yaml"
      ;;
    mambattention)
      printf '%s\n' "configs/mecge_table1_repro_mambattention.yaml"
      ;;
    dualpath_dapp_cfm_unet_bd)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd.yaml"
      ;;
    dualpath_dapp_cfm_unet_bd_step3)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step3.yaml"
      ;;
    dualpath_dapp_cfm_unet_bd_step4)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step4.yaml"
      ;;
    dualpath_dapp_cfm_unet_bd_step5)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step5.yaml"
      ;;
    dualpath_dapp_cfm_unet_bd_step8)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step8.yaml"
      ;;
    dualpath_dapp_cfm_unet_bd_no_attention)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_no_attention.yaml"
      ;;
    lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience)
      printf '%s\n' "configs/mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml"
      ;;
    stfrft)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_stfrft.yaml"
      ;;
    main)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_cfm_unet_bd.yaml"
      ;;
    stable)
      printf '%s\n' "configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_stable_cfm_unet.yaml"
      ;;
    eddm_fm)
      printf '%s\n' "configs/mecge_table1_repro_eddm_flow_matching.yaml"
      ;;
    eddm_fm_mamba)
      printf '%s\n' "configs/mecge_table1_repro_eddm_flow_matching_mamba.yaml"
      ;;
    eddm_1shot|eddm_3shot|eddm_5shot|eddm_10shot)
      printf '%s\n' "configs/mecge_table1_repro_eddm_1shot.yaml"
      ;;
  esac
}

overrides_for_model() {
  case "$1" in
    eddm_3shot)
      printf '%s\n' "model.num_shots=3"
      ;;
    eddm_5shot)
      printf '%s\n' "model.num_shots=5"
      ;;
    eddm_10shot)
      printf '%s\n' "model.num_shots=10"
      ;;
  esac
}

descod_shots_for_model() {
  case "$1" in
    descod_1shot)
      printf '%s\n' "1"
      ;;
    descod_3shot)
      printf '%s\n' "3"
      ;;
    descod_5shot)
      printf '%s\n' "5"
      ;;
    descod_10shot)
      printf '%s\n' "10"
      ;;
  esac
}

run_complexity() {
  local model_key="$1"
  local descod_shots
  descod_shots="$(descod_shots_for_model "$model_key")"
  if [[ -n "$descod_shots" ]]; then
    local output_yaml="$OUTPUT_ROOT/${model_key}_complexity_test.yaml"
    if [[ "$FORCE" != "1" && -f "$output_yaml" ]]; then
      echo "SKIP complexity: $model_key: existing $output_yaml"
      return 0
    fi
    if [[ ! -d "$DESCOD_DIR" ]]; then
      echo "Missing reference DeScoD directory: $DESCOD_DIR" >&2
      exit 1
    fi
    echo "RUN complexity: $model_key -> $output_yaml"
    (
      cd "$APP_DIR"
      python3 profile_reference_descod_complexity.py \
        --descod-dir "$DESCOD_DIR" \
        --output-yaml "$output_yaml" \
        --model-key "$model_key" \
        --num-shots "$descod_shots" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --input-length "$INPUT_LENGTH" \
        --warmup "$WARMUP" \
        --repeats "$REPEATS"
    )
    return 0
  fi

  local config
  config="$(config_for_model "$model_key")"
  local overrides=()
  local override
  while IFS= read -r override; do
    [[ -n "$override" ]] && overrides+=("$override")
  done < <(overrides_for_model "$model_key")
  local output_yaml="$OUTPUT_ROOT/${model_key}_complexity_test.yaml"
  if [[ "$FORCE" != "1" && -f "$output_yaml" ]]; then
    echo "SKIP complexity: $model_key: existing $output_yaml"
    return 0
  fi
  echo "RUN complexity: $model_key -> $output_yaml"
  (
    cd "$APP_DIR"
    python3 profile_config_complexity.py \
      --config "$config" \
      --output-yaml "$output_yaml" \
      --model-key "$model_key" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --input-length "$INPUT_LENGTH" \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
      "${overrides[@]}"
  )
}

TARGET_MODEL="$(normalize_model "$TARGET_MODEL")"

if [[ "$TARGET_MODEL" == "all" ]]; then
  MODELS=(
    mecge
    mecge_no_early_stop
    mambattention
    dualpath_dapp_cfm_unet_bd
    dualpath_dapp_cfm_unet_bd_step3
    dualpath_dapp_cfm_unet_bd_step4
    dualpath_dapp_cfm_unet_bd_step5
    dualpath_dapp_cfm_unet_bd_step8
    dualpath_dapp_cfm_unet_bd_no_attention
    lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience
    stfrft
    main
    stable
    eddm_fm
    eddm_fm_mamba
    eddm_1shot
    eddm_3shot
    eddm_5shot
    eddm_10shot
    descod_1shot
    descod_3shot
    descod_5shot
    descod_10shot
  )
else
  MODELS=("$TARGET_MODEL")
fi

mkdir -p "$OUTPUT_ROOT"
for model_key in "${MODELS[@]}"; do
  run_complexity "$model_key"
done
