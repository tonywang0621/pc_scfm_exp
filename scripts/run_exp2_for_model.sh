#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp2_for_model.sh MODEL_KEY [options]

Runs Experiment 2 baseline-strength robustness for one model across requested datasets.

Examples:
  bash scripts/run_exp2_for_model.sh mecge
  bash scripts/run_exp2_for_model.sh mambattention --datasets ptbxl
  bash scripts/run_exp2_for_model.sh mambattention_stfrft --datasets mit_bih,chapman,cpsc

MODEL_KEY must match scripts/experiment_models.sh, e.g. mecge, mambattention,
mambattention_stfrft, pc_scfm, fcn_dae, deepfilter, drnn.
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_KEY="$1"
shift

exec bash "$ROOT_DIR/scripts/run_available_exp2_exp3.sh" \
  --only-exp2 \
  --models "$MODEL_KEY" \
  "$@"
