#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m src.paper_parser.train   --train_path "$ROOT/data/paper_parser/train.jsonl"   --val_path "$ROOT/data/paper_parser/val.jsonl"   --output_dir "$ROOT/results/paper_parser"   --model_name "google/gemma-2b"   --use_qlora
