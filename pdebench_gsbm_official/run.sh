#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/.."
MODE=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
ROOT=${GSBM_RUN_ROOT:-runs/pdebench_gsbm_official_v1}
DEVICE=${GSBM_DEVICE:-cuda:0}
COMPOSE=(sudo docker compose -f pdebench_gsbm_official/compose.yaml)
case "$MODE" in setup|smoke|train|evaluate) ;; *) echo 'Usage: bash pdebench_gsbm_official/run.sh setup|smoke|train|evaluate [Python options]'; exit 2;; esac
sudo mkdir -p "$ROOT/logs"

run() {
  local label=$1
  shift
  "${COMPOSE[@]}" run --rm -T env python -u "$@" \
    2>&1 | sudo tee "$ROOT/logs/${label}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
}

if [[ "$MODE" == setup || "$MODE" == train || "$MODE" == smoke ]]; then
  "${COMPOSE[@]}" build env
  run fetch pdebench_gsbm_official/fetch_official.py
  run smoke pdebench_gsbm_official/run_pde_gsbm.py smoke --device "$DEVICE"
fi
if [[ "$MODE" == train ]]; then
  for SEED in 32 42 52; do
    run "train_seed_$SEED" pdebench_gsbm_official/run_pde_gsbm.py train \
      --seed "$SEED" --run-root "$ROOT" --device "$DEVICE" "$@"
  done
elif [[ "$MODE" == evaluate ]]; then
  run evaluate pdebench_gsbm_official/run_pde_gsbm.py evaluate \
    --run-root "$ROOT" --device "$DEVICE" "$@"
fi
