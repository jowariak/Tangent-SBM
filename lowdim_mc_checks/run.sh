#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/self_test.py"
python3 -u "$HERE/evaluate.py" --dataset gaussian --check-only "$@"
python3 -u "$HERE/evaluate.py" --dataset double_well --check-only "$@"
python3 -u "$HERE/evaluate.py" --dataset gaussian "$@"
python3 -u "$HERE/evaluate.py" --dataset double_well "$@"
