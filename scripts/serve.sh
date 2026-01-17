#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m src.serve.api   --project "paper_parser"   --model_name "google/gemma-3-4b"   --adapter_path "$ROOT/results/paper_parser/<run_id>/adapter"   --host "0.0.0.0"   --port 8000

