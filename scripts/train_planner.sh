#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m src.planner.train   --train_path "$ROOT/data/planner/train.jsonl"   --val_path "$ROOT/data/planner/val.jsonl"   --output_dir "$ROOT/results/planner"   --model_name "google/gemma-3-4b"   --use_qlora

