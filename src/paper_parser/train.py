import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from src.common.config import TrainConfig
from src.common.io import read_jsonl, save_json, ensure_dir
from src.common.logging import setup_logging
from src.common.seeds import set_seed


logger = setup_logging("paper_parser.train")


REQUIRED_KEYS = {"id", "input_text", "output_json"}


def validate_records(records: List[Dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        missing = REQUIRED_KEYS - set(record.keys())
        if missing:
            raise ValueError(f"Record {index} missing keys: {sorted(missing)}")


def dataset_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(r.get("input_text", "")) for r in records]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    return {"count": len(records), "avg_input_length": avg_len}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paper parser (scaffold).")
    parser.add_argument("--train_path", default="data/paper_parser/train.jsonl")
    parser.add_argument("--val_path", default="data/paper_parser/val.jsonl")
    parser.add_argument("--output_dir", default="results/paper_parser")
    parser.add_argument("--model_name", default="google/gemma-2b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--use_qlora", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    train_records = read_jsonl(args.train_path)
    val_records = read_jsonl(args.val_path) if args.val_path else []
    validate_records(train_records)
    if val_records:
        validate_records(val_records)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    ensure_dir(run_dir)

    config = TrainConfig(
        project="paper_parser",
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=str(run_dir),
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        use_qlora=args.use_qlora,
    )
    config.save(run_dir / "config.json")

    stats = {
        "train": dataset_stats(train_records),
        "val": dataset_stats(val_records),
    }
    save_json(run_dir / "dataset_stats.json", stats)

    logger.info("Prepared run directory: %s", run_dir)
    logger.info("Training stub complete. Add LoRA/QLoRA training code here.")


if __name__ == "__main__":
    main()
