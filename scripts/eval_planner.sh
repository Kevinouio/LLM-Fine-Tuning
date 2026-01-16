#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m src.planner.eval   --data_path "$ROOT/data/planner/test.jsonl"   --schema_path "$ROOT/schemas/planner_output.schema.json"   --output_path "$ROOT/results/planner/eval.json"
