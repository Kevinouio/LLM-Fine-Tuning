import argparse
from typing import Any, Dict, List

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


def infer_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs = []
    for record in records:
        input_text = record.get("input_text", "")
        _ = build_prompt(input_text)
        outputs.append({"id": record.get("id"), "output_json": generate_stub_output(input_text)})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer paper parser outputs (stub).")
    parser.add_argument("--input_path", help="JSONL with id and input_text")
    parser.add_argument("--input_text", help="Single input text")
    parser.add_argument("--output_path", default="results/paper_parser/predictions.jsonl")
    args = parser.parse_args()

    if not args.input_path and not args.input_text:
        parser.error("Provide --input_path or --input_text")

    if args.input_path:
        records = read_jsonl(args.input_path)
    else:
        records = [{"id": "cli_001", "input_text": args.input_text}]

    outputs = infer_records(records)
    write_jsonl(args.output_path, outputs)
    logger.info("Wrote %d predictions to %s", len(outputs), args.output_path)


if __name__ == "__main__":
    main()
