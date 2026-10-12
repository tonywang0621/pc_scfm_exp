#!/usr/bin/env bash
# Compatibility entry; use analyze_model_stages.sh for new commands.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/analyze_model_stages.sh" "$@" --model v7
