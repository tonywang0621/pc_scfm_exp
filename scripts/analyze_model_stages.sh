#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
# Registry fields: key | experiment model | model directory | config | analyzer.
source "$PROJECT_DIR/scripts/stage_analysis_models.sh"
MODEL_KEY=v7
FORWARDED=()
while (($#)); do
  case "$1" in
    --model)
      if (($# < 2)); then echo "Missing value for --model" >&2; exit 2; fi
      MODEL_KEY="$2"; shift 2 ;;
    --model=*) MODEL_KEY="${1#*=}"; shift ;;
    --list-models)
      printf 'Supported models:\n'
      for entry in "${STAGE_ANALYSIS_MODELS[@]}"; do
        IFS='|' read -r key model rest <<< "$entry"
        printf '  %s  %s\n' "$key" "$model"
      done
      exit 0 ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/analyze_model_stages.sh [--model v7] [options]

Without arguments: v7, NV1, seed3407, samples 0 10 100, automatic GPU/CPU.
  --list-models       List registered, supported models
  --model KEY         Choose model (default: v7)
  --nv 1|2            Choose noise version (default: 1)
  --seed N            Training seed used to locate checkpoint (default: 3407)
  --indices N [N ...] Choose test windows (zero-based)
  --checkpoint PATH  Override trained checkpoint
  --dataset-pkl PATH Override test dataset
  --config PATH      Override model config
  --device DEVICE    Override automatic GPU/CPU selection
  --output-dir PATH  New output directory
  --batch-size N     Inference batch size
  --sampling-rate HZ Use seconds on plots

RUN_ROOT and DATA_ROOT follow run_mecge_table1_repro.sh.
Outputs: runs/stage_waveform_analysis/<model>_<NV>_<seed>_<checkpoint>_<UTC>/
HELP
      exit 0 ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
FOUND=0
for entry in "${STAGE_ANALYSIS_MODELS[@]}"; do
  IFS='|' read -r key model model_dir config analyzer <<< "$entry"
  if [[ "$MODEL_KEY" == "$key" || "$MODEL_KEY" == "$model" ]]; then
    MODEL_KEY="$key"; MODEL="$model"; MODEL_DIR="$model_dir"
    CONFIG="$PROJECT_DIR/$config"; ANALYZER="$PROJECT_DIR/$analyzer"
    FOUND=1; break
  fi
done
if (( ! FOUND )); then
  echo "Unsupported model: $MODEL_KEY. Use --list-models to see supported models." >&2
  exit 2
fi
set -- "${FORWARDED[@]}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/mecge_table1_repro}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/mecge_table1_repro}"
NV=1
SEED=3407
HAS_CHECKPOINT=0
HAS_DATASET=0
HAS_DEVICE=0
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    -h|--help) exec "$PYTHON_BIN" -B "$ANALYZER" --help ;;
    --nv|--seed)
      if ((i+1 >= ${#ARGS[@]})); then
        echo "Missing value for ${ARGS[i]}" >&2; exit 2
      fi
      if [[ "${ARGS[i]}" == --nv ]]; then NV="${ARGS[i+1]}"; else SEED="${ARGS[i+1]}"; fi
      i=$((i+1)) ;;
    --nv=*) NV="${ARGS[i]#*=}" ;;
    --seed=*) SEED="${ARGS[i]#*=}" ;;
    --checkpoint|--checkpoint=*) HAS_CHECKPOINT=1 ;;
    --dataset-pkl|--dataset-pkl=*) HAS_DATASET=1 ;;
    --device|--device=*) HAS_DEVICE=1 ;;
  esac
done
if [[ "$NV" != 1 && "$NV" != 2 ]]; then echo "--nv must be 1 or 2" >&2; exit 2; fi
DEFAULTS=(--config "$CONFIG" --nv "$NV" --indices 0 10 100)
if (( ! HAS_CHECKPOINT )); then
  CHECKPOINT="$RUN_ROOT/$MODEL/checkpoint/${MODEL}__qtdb_train_qtdb_test__nv${NV}__seed${SEED}/$MODEL_DIR/best_model.pt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing trained $MODEL_KEY checkpoint: $CHECKPOINT" >&2
    echo "Set RUN_ROOT to your experiment root, or use --checkpoint PATH." >&2
    exit 1
  fi
  DEFAULTS+=(--checkpoint "$CHECKPOINT" --seed "$SEED")
fi
if (( ! HAS_DATASET )); then
  DATASET="$DATA_ROOT/raw/dataset_bw_nv${NV}.pkl"
  if [[ ! -f "$DATASET" ]]; then
    echo "Missing test dataset: $DATASET" >&2
    echo "Set DATA_ROOT to your dataset root, or use --dataset-pkl PATH." >&2
    exit 1
  fi
  DEFAULTS+=(--dataset-pkl "$DATASET")
fi
if (( ! HAS_DEVICE )); then
  DEVICE="$("$PYTHON_BIN" -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')"
  DEFAULTS+=(--device "$DEVICE")
fi
exec "$PYTHON_BIN" -B "$ANALYZER" "${DEFAULTS[@]}" "$@"
