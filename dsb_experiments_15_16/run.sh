#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -u "$HERE/check_package.py"
MODE="${1:?Use train 15, evaluate 15, or run 16}"
EXP="${2:?Use train 15, evaluate 15, or run 16}"
if [[ "$EXP" == 15 ]]; then
  SCRIPT="$HERE/run_sns_local_budget.py"
  ROOT=runs/sns_local_budget_v1
  mkdir -p "$ROOT/logs"
  if [[ "$MODE" == train ]]; then
    python3 -u "$SCRIPT" smoke 2>&1 | tee "$ROOT/logs/smoke.log"
    python3 -u "$SCRIPT" prepare 2>&1 | tee "$ROOT/logs/prepare.log"
    for SEED in 32 42 52; do
      python3 -u "$SCRIPT" train --seed "$SEED" 2>&1 | tee "$ROOT/logs/train_${SEED}.log"
    done
  elif [[ "$MODE" == evaluate ]]; then
    python3 -u "$SCRIPT" evaluate 2>&1 | tee "$ROOT/logs/evaluate.log"
  else echo 'Use train 15 or evaluate 15' >&2; exit 2; fi
elif [[ "$EXP" == 16 && "$MODE" == run ]]; then
  SCRIPT="$HERE/run_sns_intervention_test.py"
  ROOT=runs/sns_intervention_selection_v1
  mkdir -p "$ROOT/logs"
  python3 -u "$SCRIPT" smoke 2>&1 | tee "$ROOT/logs/smoke.log"
  python3 -u "$SCRIPT" prepare 2>&1 | tee "$ROOT/logs/prepare.log"
  python3 -u "$SCRIPT" evaluate 2>&1 | tee "$ROOT/logs/evaluate.log"
else echo 'Use train 15, evaluate 15, or run 16' >&2; exit 2; fi
