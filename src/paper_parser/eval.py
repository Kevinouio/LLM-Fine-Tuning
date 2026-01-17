import argparse
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator

from src.common.io import load_json, read_jsonl, save_json
from src.common.logging import setup_logging
from src.paper_parser.infer import infer_records


logger = setup_logging("paper_parser.eval")


def validate_outputs(outputs: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    validator = Draft7Validator(schema)
    valid_count = 0
    invalid_count = 0
    for output in outputs:
        errors = list(validator.iter_errors(output))
        if errors:
            invalid_count += 1
        else:
            valid_count += 1
    total = len(outputs)
    return {
        "total": total,
        "schema_valid": valid_count,
        "schema_invalid": invalid_count,
        "schema_valid_rate": valid_count / total if total else 0,
    }


def load_predictions(predictions_path: str) -> Dict[str, Dict[str, Any]]:
    records = read_jsonl(predictions_path)
    return {record.get("id"): record.get("output_json") for record in records}


def _run_inference(
    data_records: List[Dict[str, Any]],
    model_name: Optional[str],
    adapter_path: Optional[str],
    use_qlora: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    predictions_path: str,
) -> List[Dict[str, Any]]:
    outputs = infer_records(
        data_records,
        model_name=model_name,
        adapter_path=adapter_path,
        use_qlora=use_qlora,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_stub=False,
    )
    from src.common.io import write_jsonl

    write_jsonl(predictions_path, outputs)
    return [record.get("output_json") for record in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper parser outputs.")
    parser.add_argument("--data_path", default="data/paper_parser/test.jsonl")
    parser.add_argument("--predictions_path", help="JSONL with id and output_json")
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--schema_path", default="schemas/paper_digest.schema.json")
    parser.add_argument("--output_path", default="results/paper_parser/eval.json")
    args = parser.parse_args()

    data_records = read_jsonl(args.data_path)
    schema = load_json(args.schema_path)

    outputs: List[Dict[str, Any]] = []
    missing = 0

    if args.run_inference:
        if not args.model_name:
            raise ValueError("--model_name is required when --run_inference is set.")
        predictions_path = args.predictions_path or "results/paper_parser/predictions.jsonl"
        outputs = _run_inference(
            data_records,
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            use_qlora=args.use_qlora,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            predictions_path=predictions_path,
        )
    elif args.predictions_path:
        predictions = load_predictions(args.predictions_path)
        for record in data_records:
            pred = predictions.get(record.get("id"))
            if pred is None:
                missing += 1
                continue
            outputs.append(pred)
    else:
        outputs = [record.get("output_json") for record in data_records]

    metrics = validate_outputs(outputs, schema)
    metrics["missing_predictions"] = missing

    save_json(args.output_path, metrics)
    logger.info("Eval metrics saved to %s", args.output_path)


if __name__ == "__main__":
    main()
