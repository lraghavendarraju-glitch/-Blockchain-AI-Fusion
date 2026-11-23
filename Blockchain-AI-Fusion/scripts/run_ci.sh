#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pytest -q
python -m blockai smoke --n 72 --epochs 2 --skip-baselines --output results/ci
python -m blockai audit --input results/ci/synthetic_workflows.jsonl.gz --limit 3 \
  --model results/ci/model --output results/ci/audit.json
python -m blockai artifact-store --input results/ci/synthetic_workflows.jsonl.gz \
  --limit 3 --output results/ci/offchain --key-file results/ci/secrets/aes.key
