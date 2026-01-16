import argparse
from pathlib import Path

from src.common.logging import setup_logging


logger = setup_logging("paper_parser.preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw paper text.")
    parser.add_argument("--raw_dir", default="data/paper_parser/raw")
    parser.add_argument("--output_path", default="data/paper_parser/processed/processed.jsonl")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error("Raw directory does not exist: %s", raw_dir)
        return

    logger.info("Preprocess stub. Implement parsing for raw files in %s", raw_dir)
    logger.info("Would write processed JSONL to %s", args.output_path)


if __name__ == "__main__":
    main()
