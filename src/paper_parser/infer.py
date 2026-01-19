import argparse
from typing import Any, Dict, List, Optional

from src.common.generation import generate_json
from src.common.hf import ModelSpec, load_model_for_inference, load_tokenizer
from src.common.io import read_jsonl, write_jsonl
from src.common.logging import setup_logging
from src.paper_parser.prompts import build_prompt


logger = setup_logging("paper_parser.infer")


def _extract_title(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return "Untitled"


def _extract_abstract(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.lower().startswith("abstract:"):
            return line.split(":", 1)[1].strip()
    return input_text[:200].strip()


def generate_stub_output(input_text: str) -> Dict[str, Any]:
    title = _extract_title(input_text)
    abstract = _extract_abstract(input_text)
    return {
        "quality_summary": "TODO: assess quality",
        "quality_scores": {
            "clarity": 3,
            "problem_understanding": 3,
            "novelty": 3,
            "method_validity": 3,
            "evidence_strength": 3,
            "evaluation_quality": 3,
            "limitations": 3,
            "impact": 3,
        },
        "quality_flags": ["TODO: list potential issues"],
        "paper_type": "other",
        "title": title,
        "abstract": abstract,
        "contributions": ["TODO: extract contributions"],
        "limitations": ["TODO: extract limitations"],
        "summary_simple": "TODO: write a simple summary",
        "methods": [],
        "datasets": [],
        "results": [],
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
        input_text = record.get("input_text", "")
        prompt = build_prompt(input_text)
        if use_stub:
            output = generate_stub_output(input_text)
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
    parser = argparse.ArgumentParser(description="Infer paper parser outputs.")
    parser.add_argument("--input_path", help="JSONL with id and input_text")
    parser.add_argument("--input_text", help="Single input text")
    parser.add_argument("--output_path", default="results/paper_parser/predictions.jsonl")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_stub", action="store_true")
    args = parser.parse_args()

    if not args.input_path and not args.input_text:
        parser.error("Provide --input_path or --input_text")

    if args.input_path:
        records = read_jsonl(args.input_path)
    else:
        records = [{"id": "cli_001", "input_text": args.input_text}]

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
