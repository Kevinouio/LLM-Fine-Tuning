import argparse
from typing import Any, Dict, List

from jsonschema import Draft7Validator

from src.common.io import load_json, read_jsonl, save_json
from src.common.logging import setup_logging


logger = setup_logging("planner.eval")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate planner outputs.")
    parser.add_argument("--data_path", default="data/planner/test.jsonl")
    parser.add_argument("--predictions_path", help="JSONL with id and output_json")
    parser.add_argument("--schema_path", default="schemas/planner_output.schema.json")
    parser.add_argument("--output_path", default="results/planner/eval.json")
    args = parser.parse_args()

    data_records = read_jsonl(args.data_path)
    schema = load_json(args.schema_path)

    outputs: List[Dict[str, Any]] = []
    missing = 0

    if args.predictions_path:
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
