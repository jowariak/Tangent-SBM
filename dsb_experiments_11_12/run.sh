#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/.."
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash dsb_experiments_11_12/run.sh setup|smoke|train|evaluate|all 11|12 [Python options]'
  exit 2
fi
# Use the user's normal Compose service; no custom build or GPU allocation.
sudo docker compose run --rm env bash -c '
set -e
python3 -m pip install ipdb==0.13.13
exec python3 -u dsb_experiments_11_12/launch.py "$@"
' bash "$@"
