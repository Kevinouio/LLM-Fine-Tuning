import argparse
from typing import Any, Dict, List

from src.common.io import read_jsonl, write_jsonl
from src.common.logging import setup_logging
from src.planner.prompts import build_prompt


logger = setup_logging("planner.infer")


def generate_stub_output(goal: str, state: str, tools: List[str]) -> Dict[str, Any]:
    state_lower = state.lower() if state else ""
    if not state or "missing" in state_lower or "no " in state_lower:
        return {
            "type": "clarify",
            "question": "What key details are missing to proceed?",
        }

    action = tools[0] if tools else "think"
    return {
        "type": "plan",
        "steps": [
            {
                "action": action,
                "args": {"goal": goal},
            }
        ],
    }


def infer_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs = []
    for record in records:
        goal = record.get("goal", "")
        state = record.get("state", "")
        tools = record.get("tools", []) or []
        _ = build_prompt(goal, state, tools)
        outputs.append({"id": record.get("id"), "output_json": generate_stub_output(goal, state, tools)})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer planner outputs (stub).")
    parser.add_argument("--input_path", help="JSONL with id, goal, state, tools")
    parser.add_argument("--goal", help="Single goal")
    parser.add_argument("--state", default="")
    parser.add_argument("--tools", default="")
    parser.add_argument("--output_path", default="results/planner/predictions.jsonl")
    args = parser.parse_args()

    if not args.input_path and not args.goal:
        parser.error("Provide --input_path or --goal")

    if args.input_path:
        records = read_jsonl(args.input_path)
    else:
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
        records = [{"id": "cli_001", "goal": args.goal, "state": args.state, "tools": tools}]

    outputs = infer_records(records)
    write_jsonl(args.output_path, outputs)
    logger.info("Wrote %d predictions to %s", len(outputs), args.output_path)


if __name__ == "__main__":
    main()
