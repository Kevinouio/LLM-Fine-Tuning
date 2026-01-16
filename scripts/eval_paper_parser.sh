#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m src.paper_parser.eval   --data_path "$ROOT/data/paper_parser/test.jsonl"   --schema_path "$ROOT/schemas/paper_digest.schema.json"   --output_path "$ROOT/results/paper_parser/eval.json"
