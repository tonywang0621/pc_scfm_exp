#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
OFFICIAL_MECGE_DIR="$ROOT_DIR/references/MECG-E"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/mecge_table1_repro}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/mecge_table1_repro}"

NV="${NV:-all}"
DEVICE="${DEVICE:-cuda:0}"
PKL_FILE="${PKL_FILE:-}"
QTDB_RAW="${QTDB_RAW:-$DATA_ROOT/raw/QTDB}"
NSTDB_RAW="${NSTDB_RAW:-$DATA_ROOT/raw/NSTDB}"
if [[ -n "${RND_TEST_FILE+x}" ]]; then
  RND_TEST_FILE_USER_SET=1
else
  RND_TEST_FILE="$DATA_ROOT/raw/rnd_test.npy"
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
                        dualpath_dapp_cfm_unet_bd_step3, dualpath_dapp_cfm_unet_bd_step4,
                        dualpath_dapp_cfm_unet_bd_step5, dualpath_dapp_cfm_unet_bd_step8,
                        dualpath_dapp_cfm_unet_bd_no_attention,
                        dualpath_dapp_cfm_unet_bd_no_attention_v2,
                        stfrft, main, stable, baseline_sentry_lite,
                        baseline_sentry_flow, physio_freq_sentry_flow,
                        mecge_resflow_lite,
                        eddm_fm, eddm_fm_mamba, eddm_1shot.
  --seed N              Run one seed only. Default for single-model jobs: 3407.
                        Official MECG-E always uses the reference seed 3407.
  --skip-train          Only run robustness inference/aggregation from existing checkpoints.
  --skip-robustness     Only run train + QTDB pkl test.
  --resume              Resume training from training_state.pt for the selected run.
  --resume-checkpoint PATH
                        Resume training from an explicit training_state.pt path.
  key=value             Extra OmegaConf override passed to the selected training runner.

Environment overrides:
  SEEDS_MAIN="3407"
  SEEDS_EDDM="3407"
  RND_TEST_FILE="references/MECG-E/rnd_test.npy"
  QTDB_RAW="data/mecge_table1_repro/raw/QTDB"
  NSTDB_RAW="data/mecge_table1_repro/raw/NSTDB"
  DATA_ROOT="data/mecge_table1_repro"
  RUN_ROOT="runs/mecge_table1_repro"
  FORCE_RERUN=1 retrains even when an official-style result pkl already exists.

Single-job examples:
  bash scripts/run_mecge_table1_repro.sh --model main --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model mecge --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model mambattention --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd_step3 --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd_step4 --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd_step5 --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd_step8 --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd_no_attention --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model dualpath_dapp_cfm_unet_bd_no_attention_v2 --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stfrft --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stable --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_fm --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_fm_mamba --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model eddm_1shot --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model stfrft --seed 3407 --nv 1 --resume --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model baseline_sentry_lite --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model baseline_sentry_flow --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model physio_freq_sentry_flow --seed 3407 --nv 1 --device cuda:0
  bash scripts/run_mecge_table1_repro.sh --model mecge_resflow_lite --seed 3407 --nv 1 --device cuda:0

100% DeepFilter/MECG-E raw-prep example:
  bash scripts/run_mecge_table1_repro.sh --model mecge --nv all --prepare-raw --device cuda:0
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
  local prepared_shared_rnd="$DATA_ROOT/raw/rnd_test.npy"
  local reference_rnd="$OFFICIAL_MECGE_DIR/rnd_test_nv${nv}.npy"
  if [[ "$RND_TEST_FILE_USER_SET" == "0" && -f "$prepared_rnd" ]]; then
    printf '%s\n' "$prepared_rnd"
  elif [[ "$RND_TEST_FILE_USER_SET" == "0" && -f "$prepared_shared_rnd" ]]; then
    printf '%s\n' "$prepared_shared_rnd"
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
    dualpath_dapp_cfm_unet_bd_no_attention_v2|mambattention_dualpath_dapp_cfm_unet_bd_no_attention_v2)
      printf '%s\n' "dualpath_dapp_cfm_unet_bd_no_attention_v2"
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
    baseline_sentry_lite)
      printf '%s\n' "baseline_sentry_lite"
      ;;
    baseline_sentry_flow)
      printf '%s\n' "baseline_sentry_flow"
      ;;
    physio_freq_sentry_flow)
      printf '%s\n' "physio_freq_sentry_flow"
      ;;
    mecge_resflow_lite)
      printf '%s\n' "mecge_resflow_lite"
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
      echo "Unsupported --model '$1'. Expected one of: all, mecge, mambattention, dualpath_dapp_cfm_unet_bd, dualpath_dapp_cfm_unet_bd_step3, dualpath_dapp_cfm_unet_bd_step4, dualpath_dapp_cfm_unet_bd_step5, dualpath_dapp_cfm_unet_bd_step8, dualpath_dapp_cfm_unet_bd_no_attention, dualpath_dapp_cfm_unet_bd_no_attention_v2, stfrft, main, stable, baseline_sentry_lite, baseline_sentry_flow, physio_freq_sentry_flow, mecge_resflow_lite, eddm_fm, eddm_fm_mamba, eddm_1shot." >&2
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

if [[ "$TARGET_MODEL" == "mecge" || "$TARGET_MODEL" == "all" ]]; then
  if [[ "$TARGET_MODEL" == "mecge" && -n "$TARGET_SEED" && "$TARGET_SEED" != "3407" ]]; then
    echo "Official MECG-E reference pipeline hardcodes seed 3407; do not pass --seed for --model mecge." >&2
    exit 2
  fi
fi

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
      "training.train_epochs=50000" \
      "training.optimizer=AdamW" \
      "training.betas=[0.8,0.99]" \
      "training.weight_decay=0.01" \
      "training.scheduler=ExponentialLR" \
      "training.gamma=0.99" \
      "training.grad_clip_norm=null" \
      "training.eval_every_epochs=1" \
      "training.validation_metrics_every_epochs=1" \
      "training.early_stopping_patience_epochs=30" \
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

write_mecge_loss_artifacts() {
  local checkpoint_dir="$1"
  (
    cd "$APP_DIR"
    python3 mecge_table1_plot_mecge_loss.py \
      --checkpoint-dir "$checkpoint_dir"
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

run_official_mecge_reference_job() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local nv="$5"
  local pkl_file="$6"
  local exp_name="${result_model}__qtdb_train_qtdb_test__nv${nv}__seed${seed}"
  local config_name
  config_name="$(basename "$config" .yaml)"
  local official_out
  official_out="$(official_result_pkl "$result_model" "$nv" "$seed")"
  local official_generated="$RUN_ROOT/$result_model/native_results/${config_name}_bw_nv${nv}.pkl"
  local checkpoint_run_dir="$RUN_ROOT/$result_model/checkpoint/$exp_name/$model_name"
  local log_run_dir="$RUN_ROOT/$result_model/log/$exp_name/$model_name"
  local rnd_test_file
  local official_runner_args=()
  rnd_test_file="$(resolve_rnd_test_file "$nv")"

  if [[ ! -d "$OFFICIAL_MECGE_DIR" ]]; then
    echo "Missing official MECG-E clone: $OFFICIAL_MECGE_DIR" >&2
    exit 1
  fi
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    official_runner_args+=(--skip-train)
  fi
  if [[ "$RESUME" == "1" ]]; then
    official_runner_args+=(--resume)
  fi
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    official_runner_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi

  if [[ "$RESUME" != "1" && "$FORCE_RERUN" != "1" && -f "$official_generated" ]]; then
    echo "Found existing official MECG-E result for nv${nv}; skipping train/test: $official_generated"
  else
    mkdir -p "$RUN_ROOT/$result_model/native_results" "$checkpoint_run_dir" "$log_run_dir"
    (
      cd "$APP_DIR"
      python3 mecge_table1_run_official_reference.py \
        --mecge-dir "$OFFICIAL_MECGE_DIR" \
        --config "$config" \
        --n-type bw \
        --nv "$nv" \
        --dataset-pkl "$pkl_file" \
        --device "$DEVICE" \
        --output-pkl "$official_generated" \
        --checkpoint-dir "$checkpoint_run_dir" \
        --log-dir "$log_run_dir" \
        --max-epochs 50000 \
        --early-stopping-patience 30 \
        "${official_runner_args[@]}"
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
  run_official_mecge_reference_job "$MECGE_CONFIG" "$MECGE_RESULT_MODEL" "$MECGE_MODEL_NAME" "3407" "$nv" "$pkl_file"
}

run_official_local_model_job() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seed="$4"
  local nv="$5"
  local pkl_file="$6"
  local exp_name="${result_model}__qtdb_train_qtdb_test__nv${nv}__seed${seed}"
  local result_pkl
  result_pkl="$(official_result_pkl "$result_model" "$nv" "$seed")"
  local checkpoint_run_dir="$RUN_ROOT/$result_model/checkpoint/$exp_name/$model_name"
  local log_run_dir="$RUN_ROOT/$result_model/log/$exp_name/$model_name"
  local runner_args=()

  echo "RUN official-flow local job: model=$result_model seed=$seed nv=nv${nv}"
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    runner_args+=(--skip-train)
  fi
  if [[ "$RESUME" == "1" ]]; then
    runner_args+=(--resume)
  fi
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    runner_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi

  if [[ "$RESUME" != "1" && "$FORCE_RERUN" != "1" && -f "$result_pkl" ]]; then
    echo "Found existing official-flow local result for nv${nv}; skipping train/test: $result_pkl"
  else
    mkdir -p "$checkpoint_run_dir" "$log_run_dir" "$(dirname "$result_pkl")"
    (
      cd "$APP_DIR"
      python3 mecge_table1_run_official_local_model.py \
        --config "$config" \
        --dataset-pkl "$pkl_file" \
        --device "$DEVICE" \
        --output-pkl "$result_pkl" \
        --checkpoint-dir "$checkpoint_run_dir" \
        --log-dir "$log_run_dir" \
        --seed "$seed" \
        "${runner_args[@]}" \
        "${EXTRA_OVERRIDES[@]}"
    )
  fi

  if [[ ! -f "$result_pkl" ]]; then
    echo "Missing official-flow local result: $result_pkl" >&2
    exit 1
  fi
  write_official_metrics "$result_pkl" "$result_model" "$model_name" "$exp_name"
  run_exp2_inference "$config" "$result_model" "$model_name" "$seed" "$exp_name" "$nv" "$pkl_file"
}

run_official_local_model_family() {
  local config="$1"
  local result_model="$2"
  local model_name="$3"
  local seeds="$4"
  local nv="$5"
  local pkl_file="$6"
  for seed in $seeds; do
    run_official_local_model_job "$config" "$result_model" "$model_name" "$seed" "$nv" "$pkl_file"
  done
}

run_one_job() {
  run_official_local_model_job "$@"
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
MECGE_CONFIG="config/MECGE_phase.yaml"
MECGE_RESULT_MODEL="mecg_e"
MECGE_MODEL_NAME="mecg_e"
MAMBATTENTION_CONFIG="configs/mecge_table1_repro_mambattention.yaml"
MAMBATTENTION_RESULT_MODEL="mambattention"
MAMBATTENTION_MODEL_NAME="mambattention_ecg"
DUALPATH_DAPP_CFM_UNET_BD_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd.yaml"
DUALPATH_DAPP_CFM_UNET_BD_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd"
DUALPATH_DAPP_CFM_UNET_BD_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
DUALPATH_DAPP_CFM_UNET_BD_STEP3_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step3.yaml"
DUALPATH_DAPP_CFM_UNET_BD_STEP3_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd_step3"
DUALPATH_DAPP_CFM_UNET_BD_STEP3_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
DUALPATH_DAPP_CFM_UNET_BD_STEP4_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step4.yaml"
DUALPATH_DAPP_CFM_UNET_BD_STEP4_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd_step4"
DUALPATH_DAPP_CFM_UNET_BD_STEP4_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
DUALPATH_DAPP_CFM_UNET_BD_STEP5_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step5.yaml"
DUALPATH_DAPP_CFM_UNET_BD_STEP5_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd_step5"
DUALPATH_DAPP_CFM_UNET_BD_STEP5_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
DUALPATH_DAPP_CFM_UNET_BD_STEP8_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_step8.yaml"
DUALPATH_DAPP_CFM_UNET_BD_STEP8_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd_step8"
DUALPATH_DAPP_CFM_UNET_BD_STEP8_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_no_attention.yaml"
DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd_no_attention"
DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd_no_attention_v2.yaml"
DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_RESULT_MODEL="mambattention_dualpath_dapp_cfm_unet_bd_no_attention_v2"
DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
STFRFT_CONFIG="configs/mecge_table1_repro_mambattention_stfrft.yaml"
STFRFT_RESULT_MODEL="mambattention_stfrft"
STFRFT_MODEL_NAME="mambattention_stfrft_ecg"
STABLE_CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_stable_cfm_unet.yaml"
STABLE_RESULT_MODEL="mambattention_stfrft_dualpath_dapp_stable_cfm_unet"
STABLE_MODEL_NAME="mambattention_stfrft_dualpath_dapp_stable_cfm_unet_ecg"
BASELINE_SENTRY_LITE_CONFIG="configs/mecge_table1_repro_baseline_sentry_lite.yaml"
BASELINE_SENTRY_LITE_RESULT_MODEL="baseline_sentry_lite"
BASELINE_SENTRY_LITE_MODEL_NAME="baseline_sentry_lite"
BASELINE_SENTRY_FLOW_CONFIG="configs/mecge_table1_repro_baseline_sentry_flow.yaml"
BASELINE_SENTRY_FLOW_RESULT_MODEL="baseline_sentry_flow"
BASELINE_SENTRY_FLOW_MODEL_NAME="baseline_sentry_flow"
PHYSIO_FREQ_SENTRY_FLOW_CONFIG="configs/mecge_table1_repro_physio_freq_sentry_flow.yaml"
PHYSIO_FREQ_SENTRY_FLOW_RESULT_MODEL="physio_freq_sentry_flow"
PHYSIO_FREQ_SENTRY_FLOW_MODEL_NAME="physio_freq_sentry_flow"
MECGE_RESFLOW_LITE_CONFIG="configs/mecge_table1_repro_mecge_resflow_lite.yaml"
MECGE_RESFLOW_LITE_RESULT_MODEL="mecge_resflow_lite"
MECGE_RESFLOW_LITE_MODEL_NAME="mambattention_dualpath_dapp_cfm_unet_bd_ecg"
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
      run_model_family "$MAMBATTENTION_CONFIG" "$MAMBATTENTION_RESULT_MODEL" "$MAMBATTENTION_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$DUALPATH_DAPP_CFM_UNET_BD_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$DUALPATH_DAPP_CFM_UNET_BD_STEP3_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP3_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP3_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$DUALPATH_DAPP_CFM_UNET_BD_STEP4_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP4_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP4_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$DUALPATH_DAPP_CFM_UNET_BD_STEP5_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP5_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP5_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$DUALPATH_DAPP_CFM_UNET_BD_STEP8_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP8_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP8_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_official_local_model_family "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_official_local_model_family "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$STFRFT_CONFIG" "$STFRFT_RESULT_MODEL" "$STFRFT_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$STABLE_CONFIG" "$STABLE_RESULT_MODEL" "$STABLE_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$BASELINE_SENTRY_LITE_CONFIG" "$BASELINE_SENTRY_LITE_RESULT_MODEL" "$BASELINE_SENTRY_LITE_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$BASELINE_SENTRY_FLOW_CONFIG" "$BASELINE_SENTRY_FLOW_RESULT_MODEL" "$BASELINE_SENTRY_FLOW_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$PHYSIO_FREQ_SENTRY_FLOW_CONFIG" "$PHYSIO_FREQ_SENTRY_FLOW_RESULT_MODEL" "$PHYSIO_FREQ_SENTRY_FLOW_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$MECGE_RESFLOW_LITE_CONFIG" "$MECGE_RESFLOW_LITE_RESULT_MODEL" "$MECGE_RESFLOW_LITE_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$EDDM_FM_CONFIG" "$EDDM_FM_RESULT_MODEL" "$EDDM_FM_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$EDDM_FM_MAMBA_CONFIG" "$EDDM_FM_MAMBA_RESULT_MODEL" "$EDDM_FM_MAMBA_MODEL_NAME" "$SEEDS_MAIN" "$nv" "$pkl_file"
      run_model_family "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "$SEEDS_EDDM" "$nv" "$pkl_file"
      ;;
    main)
      run_one_job "$MAIN_CONFIG" "$MAIN_RESULT_MODEL" "$MAIN_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    mecge)
      run_official_mecge_for_nv "$nv" "$pkl_file"
      ;;
    mambattention)
      run_one_job "$MAMBATTENTION_CONFIG" "$MAMBATTENTION_RESULT_MODEL" "$MAMBATTENTION_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd)
      run_one_job "$DUALPATH_DAPP_CFM_UNET_BD_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd_step3)
      run_one_job "$DUALPATH_DAPP_CFM_UNET_BD_STEP3_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP3_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP3_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd_step4)
      run_one_job "$DUALPATH_DAPP_CFM_UNET_BD_STEP4_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP4_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP4_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd_step5)
      run_one_job "$DUALPATH_DAPP_CFM_UNET_BD_STEP5_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP5_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP5_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd_step8)
      run_one_job "$DUALPATH_DAPP_CFM_UNET_BD_STEP8_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_STEP8_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_STEP8_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd_no_attention)
      run_official_local_model_job "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    dualpath_dapp_cfm_unet_bd_no_attention_v2)
      run_official_local_model_job "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_CONFIG" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_RESULT_MODEL" "$DUALPATH_DAPP_CFM_UNET_BD_NO_ATTENTION_V2_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    stfrft)
      run_one_job "$STFRFT_CONFIG" "$STFRFT_RESULT_MODEL" "$STFRFT_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    stable)
      run_one_job "$STABLE_CONFIG" "$STABLE_RESULT_MODEL" "$STABLE_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    baseline_sentry_lite)
      run_one_job "$BASELINE_SENTRY_LITE_CONFIG" "$BASELINE_SENTRY_LITE_RESULT_MODEL" "$BASELINE_SENTRY_LITE_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    baseline_sentry_flow)
      run_one_job "$BASELINE_SENTRY_FLOW_CONFIG" "$BASELINE_SENTRY_FLOW_RESULT_MODEL" "$BASELINE_SENTRY_FLOW_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    physio_freq_sentry_flow)
      run_one_job "$PHYSIO_FREQ_SENTRY_FLOW_CONFIG" "$PHYSIO_FREQ_SENTRY_FLOW_RESULT_MODEL" "$PHYSIO_FREQ_SENTRY_FLOW_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    mecge_resflow_lite)
      run_one_job "$MECGE_RESFLOW_LITE_CONFIG" "$MECGE_RESFLOW_LITE_RESULT_MODEL" "$MECGE_RESFLOW_LITE_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    eddm_fm)
      run_one_job "$EDDM_FM_CONFIG" "$EDDM_FM_RESULT_MODEL" "$EDDM_FM_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    eddm_fm_mamba)
      run_one_job "$EDDM_FM_MAMBA_CONFIG" "$EDDM_FM_MAMBA_RESULT_MODEL" "$EDDM_FM_MAMBA_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
      ;;
    eddm_1shot)
      run_one_job "$EDDM_CONFIG" "$EDDM_RESULT_MODEL" "$EDDM_MODEL_NAME" "${TARGET_SEED:-3407}" "$nv" "$pkl_file"
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
