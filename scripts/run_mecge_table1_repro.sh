#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
OFFICIAL_MECGE_DIR="$ROOT_DIR/references/MECG-E"
DATA_ROOT="$ROOT_DIR/data/mecge_table1_repro"
RUN_ROOT="$ROOT_DIR/runs/mecge_table1_repro"

NV="${NV:-all}"
DEVICE="${DEVICE:-cuda:0}"
PKL_FILE="${PKL_FILE:-}"
QTDB_RAW="${QTDB_RAW:-$DATA_ROOT/raw/QTDB}"
NSTDB_RAW="${NSTDB_RAW:-$DATA_ROOT/raw/NSTDB}"
if [[ -n "${RND_TEST_FILE+x}" ]]; then
  RND_TEST_FILE_USER_SET=1
else
  RND_TEST_FILE="$ROOT_DIR/references/MECG-E/rnd_test.npy"
  RND_TEST_FILE_USER_SET=0
fi
FORCE_PREPARE_RAW="${FORCE_PREPARE_RAW:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ROBUSTNESS="${SKIP_ROBUSTNESS:-0}"
RESUME="${RESUME:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SEEDS_MAIN="${SEEDS_MAIN:-3407}"
SEEDS_EDDM="${SEEDS_EDDM:-3407}"
TARGET_MODEL="${TARGET_MODEL:-all}"
TARGET_SEED="${TARGET_SEED:-}"
EXTRA_OVERRIDES=()
OFFICIAL_MECGE_RAN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_mecge_table1_repro.sh [options]

Options:
  --nv N|all            MECG-E noise version. Default: all (official nv1 and nv2).
  --pkl-file PATH       Official MECG-E dataset_bw_nv*.pkl path. With --nv all,
                        use a path containing {nv}, or omit to use data/mecge_table1_repro/raw.
  --qtdb-raw PATH        Raw QTDB WFDB directory for 100% DeepFilter/MECG-E prep.
  --nstdb-raw PATH       Raw NSTDB WFDB directory for 100% DeepFilter/MECG-E prep.
  --rnd-test PATH        Official MECG-E rnd_test.npy for robustness bins.
  --prepare-raw          Recreate dataset_bw_nv*.pkl from raw QTDB/NSTDB before training.
  --device DEVICE       Training/inference device. Default: cuda:0
  --model NAME          One of: all, mecge, mambattention, dualpath_dapp_cfm_unet_bd,
                        stfrft, main, stable, eddm_fm, eddm_fm_mamba, eddm_1shot.
  --seed N              Run one seed only. Default for single-model jobs: 3407.
  --skip-train          Only run robustness inference/aggregation from existing checkpoints.
  --skip-robustness     Only run train + QTDB pkl test.
  --resume              Resume training from training_state.pt for the selected run.
  --resume-checkpoint PATH
                        Resume training from an explicit training_state.pt path.
  key=value             Extra OmegaConf override passed to train_supervised.py.

Environment overrides:
  SEEDS_MAIN="3407"
  SEEDS_EDDM="3407"
  RND_TEST_FILE="references/MECG-E/rnd_test.npy"
  QTDB_RAW="data/mecge_table1_repro/raw/QTDB"
  NSTDB_RAW="data/mecge_table1_repro/raw/NSTDB"
  FORCE_RERUN=1 retrains even when an official-style result pkl already exists.

Single-job examples:
  bash scripts/run_mecge_table1_repro.sh --model main --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model mecge --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model mambattention --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stfrft --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stable --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_fm --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_fm_mamba --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_1shot --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stfrft --seed 3407 --nv 1 --resume --device cuda:0

100% DeepFilter/MECG-E raw-prep example:
  bash scripts/run_mecge_table1_repro.sh --model main --seed 3407 --nv all --prepare-raw --device cuda:0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nv)
      NV="$2"
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
    --rnd-test)
      RND_TEST_FILE="$2"
      RND_TEST_FILE_USER_SET=1
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
    --resume)
      RESUME=1
      shift
      ;;
    --resume-checkpoint)
      RESUME=1
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *=*)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$RND_TEST_FILE" != /* ]]; then
  RND_TEST_FILE="$ROOT_DIR/$RND_TEST_FILE"
fi

resolve_rnd_test_file() {
  local nv="$1"
  local prepared_rnd="$DATA_ROOT/raw/rnd_test_nv${nv}.npy"
  local reference_rnd="$OFFICIAL_MECGE_DIR/rnd_test_nv${nv}.npy"
  if [[ "$RND_TEST_FILE_USER_SET" == "0" && -f "$prepared_rnd" ]]; then
    printf '%s\n' "$prepared_rnd"
  elif [[ "$RND_TEST_FILE_USER_SET" == "0" && -f "$reference_rnd" ]]; then
    printf '%s\n' "$reference_rnd"
  else
    printf '%s\n' "$RND_TEST_FILE"
  fi
}

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/controlled_tests" "$RUN_ROOT/analysis"

validate_noise_version() {
  case "$1" in
    1|2)
      return 0
      ;;
    *)
      echo "Unsupported --nv '$1'. Expected 1, 2, or all." >&2
      exit 2
      ;;
  esac
}

noise_versions() {
  case "$NV" in
    all)
      printf '%s\n' 1 2
      ;;
    1|2)
      printf '%s\n' "$NV"
      ;;
    *)
      echo "Unsupported --nv '$NV'. Expected 1, 2, or all." >&2
      exit 2
      ;;
  esac
}

resolve_pkl_file() {
  local nv="$1"
  if [[ -n "$PKL_FILE" ]]; then
    if [[ "$PKL_FILE" == *"{nv}"* ]]; then
      printf '%s\n' "${PKL_FILE//\{nv\}/$nv}"
    else
      if [[ "$NV" == "all" ]]; then
        echo "--pkl-file with --nv all must contain '{nv}', or omit --pkl-file." >&2
        exit 2
      fi
      printf '%s\n' "$PKL_FILE"
    fi
  else
    printf '%s\n' "$DATA_ROOT/raw/dataset_bw_nv${nv}.pkl"
  fi
}

official_result_pkl() {
  local result_model="$1"
  local nv="$2"
  local seed="$3"
  printf '%s\n' "$RUN_ROOT/official_results/${result_model}__qtdb_train_qtdb_test__nv${nv}__seed${seed}.pkl"
}

normalize_model() {
  case "$1" in
    all)
      printf '%s\n' "all"
      ;;
    mecge|mecg_e)
      printf '%s\n' "mecge"
      ;;
    mambattention|mambattention_ecg)
      printf '%s\n' "mambattention"
      ;;
    dualpath_dapp_cfm_unet_bd|mambattention_dualpath_dapp_cfm_unet_bd|mambattention_dualpath_dapp_cfm_unet_bd_ecg)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd"
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
    eddm|eddm_1shot)
      printf '%s\n' "eddm_1shot"
      ;;
    *)
      echo "Unsupported --model '$1'. Expected one of: all, mecge, mambattention, dualpath_dapp_cfm_unet_bd, stfrft, main, stable, eddm_fm, eddm_fm_mamba, eddm_1shot." >&2
      exit 2
      ;;
  esac
}

prepare_raw_if_needed() {
  local nv="$1"
  local pkl_file="$2"
  if [[ "$FORCE_PREPARE_RAW" != "1" && -f "$pkl_file" ]]; then
    return 0
  fi
  if [[ -z "$QTDB_RAW" || -z "$NSTDB_RAW" ]]; then
    echo "Missing MECG-E pkl file: $pkl_file" >&2
    echo "Provide --pkl-file, or provide --qtdb-raw and --nstdb-raw with --prepare-raw." >&2
    exit 1
  fi
  (
    cd "$APP_DIR"
    python3 mecge_table1_prepare_deepfilter.py \
      --qtdb-root "$QTDB_RAW" \
      --nstdb-root "$NSTDB_RAW" \
      --output-dir "$DATA_ROOT/raw" \
      --noise-version "$nv" \
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
  local pkl_file="$6"

  if [[ "$SKIP_TRAIN" == "1" ]]; then
    return 0
  fi

  local resume_overrides=("training.resume=$RESUME")
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    resume_overrides+=("training.resume_checkpoint=$RESUME_CHECKPOINT")
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
      "dataset.pkl_file=$pkl_file" \
      "dataset.test_label=qtdb_pkl_test" \
      "dataset.pkl_validation_ratio=0.3" \
      "dataset.pkl_validation_random_state=1" \
      "exp_name=$exp_name" \
      "seed=$seed" \
      "device=$DEVICE" \
      "training.train_epochs=30" \
      "training.optimizer=AdamW" \
      "training.betas=[0.8,0.99]" \
      "training.weight_decay=0.01" \
      "training.scheduler=ExponentialLR" \
      "training.gamma=0.99" \
      "training.grad_clip_norm=null" \
      "training.eval_every_epochs=1" \
      "training.validation_metrics_every_epochs=1" \
      "training.early_stopping_patience_epochs=null" \
      "training.selection_metric=val_loss" \
      "training.num_workers=0" \
      "training.train_drop_last=true" \
      "training.val_drop_last=true" \
      "training.test_batch_size=64" \
      "evaluation.metric_protocol=mecge_official" \
      "${resume_overrides[@]}" \
      "${EXTRA_OVERRIDES[@]}"
  )
}

checkpoint_path() {
  local result_model="$1"
  local model_name="$2"
  local exp_name="$3"
  printf '%s\n' "$RUN_ROOT/$result_model/checkpoint/$exp_name/$model_name/best_model.pt"
}

write_official_metrics() {
  local result_pkl="$1"
  local result_model="$2"
  local model_name="$3"
  local exp_name="$4"
  local eval_dir="$RUN_ROOT/$result_model/results/$exp_name/$model_name/best_loss"

  (
    cd "$APP_DIR"
    python3 mecge_table1_official_result_metrics.py \
      --result-pkl "$result_pkl" \
      --output-dir "$eval_dir"
  )
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
  local nv="$6"
  local pkl_file="$7"
  local metrics_per_window="$RUN_ROOT/$result_model/results/$exp_name_train/$model_name/best_loss/metrics_per_window.csv"
  local rnd_test_file
  rnd_test_file="$(resolve_rnd_test_file "$nv")"

  if [[ "$SKIP_ROBUSTNESS" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$metrics_per_window" ]]; then
    echo "Missing official-style per-window metrics for robustness: $metrics_per_window" >&2
    exit 1
  fi
  if [[ ! -f "$rnd_test_file" ]]; then
    echo "Missing MECG-E rnd_test file for robustness bins: $rnd_test_file" >&2
    echo "Use --prepare-raw to create data/mecge_table1_repro/raw/rnd_test_nv*.npy, clone khhungg/MECG-E to references/MECG-E, or pass --rnd-test PATH." >&2
    exit 1
  fi

  (
    cd "$APP_DIR"
    python3 mecge_table1_robustness_bins.py \
      --metrics-per-window "$metrics_per_window" \
      --rnd-test "$rnd_test_file" \
      --output-root "$RUN_ROOT/$result_model/controlled_tests" \
      --result-model "$result_model" \
      --noise-version "nv${nv}" \
      --seed "$seed"
  )
}

run_mecge_pipeline_job() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local nv="$5"
  local pkl_file="$6"
  local exp_name="${result_model}__qtdb_train_qtdb_test__nv${nv}__seed${seed}"
  local config_path="$config"
  local config_name
  config_name="$(basename "$config_path" .yaml)"
  if [[ "$config_path" != /* ]]; then
    config_path="$APP_DIR/$config_path"
  fi
  local official_out
  official_out="$(official_result_pkl "$result_model" "$nv" "$seed")"
  local official_generated="$RUN_ROOT/$result_model/native_results/${config_name}_bw_nv${nv}.pkl"
  local training_state="$RUN_ROOT/$result_model/checkpoint/${exp_name}/$model_name/training_state.pt"
  local rnd_test_file
  rnd_test_file="$(resolve_rnd_test_file "$nv")"

  if [[ ! -d "$OFFICIAL_MECGE_DIR" ]]; then
    echo "Missing official MECG-E clone: $OFFICIAL_MECGE_DIR" >&2
    exit 1
  fi

  if [[ "$SKIP_TRAIN" != "1" && "$FORCE_RERUN" != "1" && -f "$official_generated" ]]; then
    echo "Found existing MECG-E-pipeline result for nv${nv}; skipping train/test: $official_generated"
  elif [[ "$SKIP_TRAIN" != "1" ]]; then
    mkdir -p "$OFFICIAL_MECGE_DIR/data" "$RUN_ROOT/$result_model/native_results" "$RUN_ROOT/$result_model/log"
    cp "$pkl_file" "$OFFICIAL_MECGE_DIR/data/dataset_bw_nv${nv}.pkl"
    (
      cd "$OFFICIAL_MECGE_DIR"
      job_resume="$RESUME"
      if [[ "$FORCE_RERUN" != "1" && -f "$training_state" ]]; then
        job_resume=1
        echo "Found existing training state for nv${nv}; resuming: $training_state"
      elif [[ "$RESUME" == "1" ]]; then
        job_resume=0
        echo "No training state for nv${nv}; starting this noise version from scratch."
      fi
      export MECGE_APP_DIR="$APP_DIR"
      export MECGE_DEVICE="$DEVICE"
      export MECGE_MODEL_WEIGHT_TEMPLATE="$RUN_ROOT/$result_model/checkpoint/${exp_name}/$model_name/best_model.pt"
      export MECGE_RESULTS_DIR="$RUN_ROOT/$result_model/native_results"
      export MECGE_LOGS_DIR="$RUN_ROOT/$result_model/log"
      export MECGE_RESUME="$job_resume"
      if [[ "$model_name" != "$MECGE_MODEL_NAME" ]]; then
        export MECGE_PROJECT_MODEL_NAME="$model_name"
      fi
      if [[ -n "$RESUME_CHECKPOINT" ]]; then
        export MECGE_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
      fi
      python3 main.py --n_type bw --nv "$nv" --config "$config_path" --device "$DEVICE"
    )
  fi

  if [[ ! -f "$official_generated" ]]; then
    echo "Missing MECG-E-pipeline result: $official_generated" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$official_out")"
  cp "$official_generated" "$official_out"
  write_official_metrics "$official_out" "$result_model" "$model_name" "$exp_name"

  if [[ "$SKIP_ROBUSTNESS" != "1" ]]; then
    if [[ ! -f "$rnd_test_file" ]]; then
      echo "Missing MECG-E rnd_test file for robustness bins: $rnd_test_file" >&2
      exit 1
    fi
    (
      cd "$APP_DIR"
      python3 mecge_table1_robustness_bins.py \
        --metrics-per-window "$RUN_ROOT/$result_model/results/$exp_name/$model_name/best_loss/metrics_per_window.csv" \
        --rnd-test "$rnd_test_file" \
        --output-root "$RUN_ROOT/$result_model/controlled_tests" \
        --result-model "$result_model" \
        --noise-version "nv${nv}" \
        --seed "$seed"
    )
  fi
}

run_official_mecge_for_nv() {
  local nv="$1"
  local pkl_file="$2"
  run_mecge_pipeline_job "$MECGE_CONFIG" "$MECGE_RESULT_MODEL" "$MECGE_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
}

run_mecge_pipeline_family() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seeds="$4"
  local nv="$5"
  local pkl_file="$6"
  for seed in $seeds; do
    run_mecge_pipeline_job "$config" "$result_model" "$model_name" "$seed" "$nv" "$pkl_file"
  done
}

run_one_job() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local nv="$5"
  local pkl_file="$6"
  local exp_name="${result_model}__qtdb_train_qtdb_test__nv${nv}__seed${seed}"
  local result_pkl
  result_pkl="$(official_result_pkl "$result_model" "$nv" "$seed")"
  echo "RUN job: model=$result_model seed=$seed nv=nv${nv}"
  run_train "$config" "$result_model" "$model_name" "$seed" "$exp_name" "$pkl_file"

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    (
      cd "$APP_DIR"
      python3 mecge_table1_export_local_result.py \
        --config "$config" \
        --checkpoint "$(checkpoint_path "$result_model" "$model_name" "$exp_name")" \
        --pkl-file "$pkl_file" \
        --output-pkl "$result_pkl" \
        --batch-size 64 \
        --device "$DEVICE"
    )
    write_official_metrics "$result_pkl" "$result_model" "$model_name" "$exp_name"
  fi

  run_exp2_inference "$config" "$result_model" "$model_name" "$seed" "$exp_name" "$nv" "$pkl_file"
}

run_model_family() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seeds="$4"
  local nv="$5"
  local pkl_file="$6"
  for seed in $seeds; do
    run_one_job "$config" "$result_model" "$model_name" "$seed" "$nv" "$pkl_file"
  done
}

MAIN_CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_cfm_unet_bd.yaml"
MAIN_RESULT_MODEL="mambattention_stfrft_dualpath_dapp_cfm_unet_bd"
MAIN_MODEL_NAME="mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg"
MECGE_CONFIG="configs/mecge_table1_repro_mecg_e.yaml"
MECGE_RESULT_MODEL="mecg_e"
MECGE_MODEL_NAME="mecg_e"
MAMBATTENTION_CONFIG="configs/mecge_table1_repro_mambattention.yaml"
MAMBATTENTION_RESULT_MODEL="mambattention"
MAMBATTENTION_MODEL_NAME="mambattention_ecg"
DUALPATH_DAPP_CFM_UNET_BD_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd.yaml"
DUALPATH_DAPP_CFM_UNET_BD_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd"
DUALPATH_DAPP_CFM_UNET_BD_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
STFRFT_CONFIG="configs/mecge_table1_repro_mambattention_stfrft.yaml"
STFRFT_RESULT_MODEL="mambattention_stfrft"
STFRFT_MODEL_NAME="mambattention_stfrft_ecg"
STABLE_CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_stable_cfm_unet.yaml"
STABLE_RESULT_MODEL="mambattention_stfrft_dualpath_dapp_stable_cfm_unet"
STABLE_MODEL_NAME="mambattention_stfrft_dualpath_dapp_stable_cfm_unet_ecg"
EDDM_FM_CONFIG="configs/mecge_table1_repro_eddm_flow_matching.yaml"
EDDM_FM_RESULT_MODEL="eddm_flow_matching"
EDDM_FM_MODEL_NAME="eddm_flow_matching"
EDDM_FM_MAMBA_CONFIG="configs/mecge_table1_repro_eddm_flow_matching_mamba.yaml"
EDDM_FM_MAMBA_RESULT_MODEL="eddm_flow_matching_mamba"
EDDM_FM_MAMBA_MODEL_NAME="eddm_flow_matching_mamba"
EDDM_CONFIG="configs/mecge_table1_repro_eddm_1shot.yaml"
EDDM_RESULT_MODEL="eddm_1shot"
EDDM_MODEL_NAME="eddm"

run_selected_models_for_nv() {
  local nv="$1"
  local pkl_file="$2"

  case "$TARGET_MODEL" in
    all)
      if [[ -n "$TARGET_SEED" ]]; then
        echo "--seed can only be used with a single --model target." >&2
        exit 2
      fi
      run_official_mecge_for_nv "$nv" "$pkl_file"
      run_mecge_pipeline_family "$MAMBATTENTION_CONFIG" "$MAMBATTENTION_RESULT_MODEL" "$MAMBATTENTION_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$DUALPATH_DAPP_CFM_UNET_BD_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$STFRFT_CONFIG" "$STFRFT_RESULT_MODEL" "$STFRFT_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$STABLE_CONFIG" "$STABLE_RESULT_MODEL" "$STABLE_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$EDDM_FM_CONFIG" "$EDDM_FM_RESULT_MODEL" "$EDDM_FM_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$EDDM_FM_MAMBA_CONFIG" "$EDDM_FM_MAMBA_RESULT_MODEL" "$EDDM_FM_MAMBA_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_mecge_pipeline_family "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "$SEEDS_EDDM" "$nv" "$pkl_file"
      ;;
    main)
      run_mecge_pipeline_job "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    mecge)
      run_official_mecge_for_nv "$nv" "$pkl_file"
      ;;
    mambattention)
      run_mecge_pipeline_job "$MAMBATTENTION_CONFIG" "$MAMBATTENTION_RESULT_MODEL" "$MAMBATTENTION_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd)
      run_mecge_pipeline_job "$DUALPATH_DAPP_CFM_UNET_BD_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    stfrft)
      run_mecge_pipeline_job "$STFRFT_CONFIG" "$STFRFT_RESULT_MODEL" "$STFRFT_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    stable)
      run_mecge_pipeline_job "$STABLE_CONFIG" "$STABLE_RESULT_MODEL" "$STABLE_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    eddm_fm)
      run_mecge_pipeline_job "$EDDM_FM_CONFIG" "$EDDM_FM_RESULT_MODEL" "$EDDM_FM_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    eddm_fm_mamba)
      run_mecge_pipeline_job "$EDDM_FM_MAMBA_CONFIG" "$EDDM_FM_MAMBA_RESULT_MODEL" "$EDDM_FM_MAMBA_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    eddm_1shot)
      run_mecge_pipeline_job "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
  esac
}

for nv in $(noise_versions); do
  validate_noise_version "$nv"
  pkl_file="$(resolve_pkl_file "$nv")"
  prepare_raw_if_needed "$nv" "$pkl_file"
done

for nv in $(noise_versions); do
  validate_noise_version "$nv"
  pkl_file="$(resolve_pkl_file "$nv")"
  run_selected_models_for_nv "$nv" "$pkl_file"
  (
    cd "$APP_DIR"
    python3 mecge_table1_collect_results.py \
      --run-root "$RUN_ROOT" \
      --noise-version "nv${nv}" \
      --output-dir "$RUN_ROOT/analysis"
  )
done

(
  cd "$APP_DIR"
  rnd_test_nv1="$(resolve_rnd_test_file 1)"
  rnd_test_nv2="$(resolve_rnd_test_file 2)"
  python3 mecge_table1_collect_results.py \
    --run-root "$RUN_ROOT" \
    --noise-version all \
    --output-dir "$RUN_ROOT/analysis"
  if [[ "$RND_TEST_FILE_USER_SET" == "0" && "$rnd_test_nv1" != "$RND_TEST_FILE" && "$rnd_test_nv2" != "$RND_TEST_FILE" ]]; then
    python3 mecge_table1_collect_official_protocol.py \
      --official-results-dir "$RUN_ROOT/official_results" \
      --rnd-test-nv1 "$rnd_test_nv1" \
      --rnd-test-nv2 "$rnd_test_nv2" \
      --output-dir "$RUN_ROOT/analysis"
  elif [[ -f "$RND_TEST_FILE" ]]; then
    python3 mecge_table1_collect_official_protocol.py \
      --official-results-dir "$RUN_ROOT/official_results" \
      --rnd-test "$RND_TEST_FILE" \
      --output-dir "$RUN_ROOT/analysis"
  fi
)
