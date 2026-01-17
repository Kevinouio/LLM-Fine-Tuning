import argparse
from typing import Any, Dict, List, Optional

from src.common.generation import generate_json
from src.common.hf import ModelSpec, load_model_for_inference, load_tokenizer
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


def infer_records(
    records: List[Dict[str, Any]],
    model_name: Optional[str],
    adapter_path: Optional[str],
    use_qlora: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_stub: bool,
) -> List[Dict[str, Any]]:
    outputs = []
    tokenizer = None
    model = None
    if not use_stub:
        if not model_name:
            raise ValueError("model_name is required unless --use_stub is set.")
        tokenizer = load_tokenizer(model_name)
        model = load_model_for_inference(
            ModelSpec(model_name=model_name, adapter_path=adapter_path, use_qlora=use_qlora)
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    for record in records:
        goal = record.get("goal", "")
        state = record.get("state", "")
        tools = record.get("tools", []) or []
        prompt = build_prompt(goal, state, tools)
        if use_stub:
            output = generate_stub_output(goal, state, tools)
        else:
            output = generate_json(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        outputs.append({"id": record.get("id"), "output_json": output})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer planner outputs.")
    parser.add_argument("--input_path", help="JSONL with id, goal, state, tools")
    parser.add_argument("--goal", help="Single goal")
    parser.add_argument("--state", default="")
    parser.add_argument("--tools", default="")
    parser.add_argument("--output_path", default="results/planner/predictions.jsonl")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_stub", action="store_true")
    args = parser.parse_args()

    if not args.input_path and not args.goal:
        parser.error("Provide --input_path or --goal")

    if args.input_path:
        records = read_jsonl(args.input_path)
    else:
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
        records = [{"id": "cli_001", "goal": args.goal, "state": args.state, "tools": tools}]

    outputs = infer_records(
        records,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_qlora=args.use_qlora,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_stub=args.use_stub,
    )
    write_jsonl(args.output_path, outputs)
    logger.info("Wrote %d predictions to %s", len(outputs), args.output_path)


if __name__ == "__main__":
    main()
