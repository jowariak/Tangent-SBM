#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-run}"
EXP="${2:-}"
case "$EXP" in
  19) METHOD=gsbm ;;
  20) METHOD=tsbm ;;
  *) echo 'Usage: bash dsb_experiments_19_20/run.sh {smoke|train|evaluate|run} {19|20}'; exit 2 ;;
esac
case "$MODE" in smoke|train|evaluate|run) ;; *) echo 'Unknown mode'; exit 2 ;; esac
python3 -c 'import torch; from torch.func import vmap, grad, jacrev'
python3 -c 'import ipdb, tqdm' || python3 -m pip install ipdb tqdm
python3 "$HERE/fetch_official.py"
python3 "$HERE/fetch_tsbm.py"
COST=quadratic
  if [[ "$MODE" == smoke || "$MODE" == run || "$MODE" == train ]]; then
    python3 -u "$HERE/run_gaussian.py" smoke --method "$METHOD" --cost "$COST"
  fi
  if [[ "$MODE" == train || "$MODE" == run ]]; then
    for SEED in 32 42 52; do
      python3 -u "$HERE/run_gaussian.py" train --method "$METHOD" --cost "$COST" --seed "$SEED"
    done
  fi
# Evaluate the fixed nonzero-cost configuration after all three seeds finish.
if [[ "$MODE" == evaluate || "$MODE" == run ]]; then
    python3 -u "$HERE/run_gaussian.py" evaluate --method "$METHOD" --cost "$COST"
fi
