#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "$ROOT_DIR/scripts/run_available_exp2_exp3.sh" --only-exp2 "$@"
