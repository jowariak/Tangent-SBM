#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -u "$HERE/check_package.py"
MODE="${1:-}"
EXP="${2:-}"
case "$EXP:$MODE" in
  17:smoke) python3 -u "$HERE/run_sns_figure.py" smoke ;;
  17:run)
    python3 -c 'import matplotlib' || python3 -m pip install 'matplotlib==3.7.5'
    python3 -u "$HERE/run_sns_figure.py" smoke
    python3 -u "$HERE/run_sns_figure.py" generate ;;
  17:render)
    python3 -c 'import matplotlib' || python3 -m pip install 'matplotlib==3.7.5'
    python3 -u "$HERE/run_sns_figure.py" render ;;
  18:smoke) python3 -u "$HERE/run_sns_unconditional.py" smoke ;;
  18:train)
    python3 -u "$HERE/run_sns_unconditional.py" smoke
    for SEED in 32 42 52; do
      python3 -u "$HERE/run_sns_unconditional.py" train --seed "$SEED"
    done ;;
  18:evaluate) python3 -u "$HERE/run_sns_unconditional.py" evaluate ;;
  *) echo "Usage: bash dsb_experiments_17_18/run.sh {smoke|run|render} 17 OR {smoke|train|evaluate} 18" >&2; exit 2 ;;
esac
