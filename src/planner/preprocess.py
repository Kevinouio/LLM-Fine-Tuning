import argparse
from pathlib import Path

from src.common.logging import setup_logging


logger = setup_logging("planner.preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess planner data.")
    parser.add_argument("--raw_dir", default="data/planner/raw")
    parser.add_argument("--output_path", default="data/planner/processed/processed.jsonl")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error("Raw directory does not exist: %s", raw_dir)
        return

    logger.info("Preprocess stub. Implement parsing for raw files in %s", raw_dir)
    logger.info("Would write processed JSONL to %s", args.output_path)


if __name__ == "__main__":
    main()
