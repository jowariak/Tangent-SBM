#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?Usage: run.sh smoke|train|evaluate 13|14}"
EXP="${2:?Usage: run.sh smoke|train|evaluate 13|14}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -u "$HERE/check_package.py"
case "$MODE" in smoke|train|evaluate) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
case "$EXP" in
  13)
    SCRIPT="$HERE/run_sns_tsbm.py"
    RUN_ROOT=runs/sns_tsbm_official_v1
    python3 -m pip install ipdb==0.13.13
    ;;
  14)
    SCRIPT="$HERE/run_sns_spread_budget.py"
    RUN_ROOT=runs/sns_spread_budget_v1
    ;;
  *) echo "Experiment must be 13 or 14" >&2; exit 2;;
esac
mkdir -p "$RUN_ROOT/logs"
if [[ "$MODE" == smoke || "$MODE" == train ]]; then
  python3 -u "$SCRIPT" smoke 2>&1 | tee "$RUN_ROOT/logs/smoke.log"
fi
if [[ "$MODE" == train ]]; then
  if [[ "$EXP" == 14 ]]; then
    python3 -u "$SCRIPT" prepare 2>&1 | tee "$RUN_ROOT/logs/prepare.log"
  fi
  for SEED in 32 42 52; do
    python3 -u "$SCRIPT" train --seed "$SEED" 2>&1 | tee "$RUN_ROOT/logs/train_${SEED}.log"
  done
elif [[ "$MODE" == evaluate ]]; then
  python3 -u "$SCRIPT" evaluate 2>&1 | tee "$RUN_ROOT/logs/evaluate.log"
fi
