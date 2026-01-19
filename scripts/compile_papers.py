import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz

from src.common.io import ensure_dir, read_jsonl, write_jsonl
from src.common.logging import setup_logging


logger = setup_logging("compile_papers")

if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "set_verbosity"):
    fitz.TOOLS.set_verbosity(0)


def _short_id(openalex_id: str) -> str:
    return openalex_id.rsplit("/", 1)[-1]


def _extract_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    chunks: List[str] = []
    for page in doc:
        try:
            text = page.get_text("text")
        except Exception:
            continue
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _collect_records(input_root: Path, max_chars: int, max_papers: int, log_every: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for manifest in input_root.rglob("manifest.jsonl"):
        topic = manifest.parent.name
        pdf_dir = manifest.parent / "pdfs"
        for record in read_jsonl(manifest):
            if max_papers and len(records) >= max_papers:
                return records
            openalex_id = record.get("openalex_id")
            if not openalex_id:
                continue
            short_id = _short_id(openalex_id)
            pdf_path = pdf_dir / f"{short_id}.pdf"
            if not pdf_path.exists():
                continue
            try:
                text = _extract_pdf_text(pdf_path)
            except Exception as exc:
                logger.info("Skipping %s: extract_failed (%s)", short_id, exc)
                continue
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            records.append(
                {
                    "id": short_id,
                    "openalex_id": openalex_id,
                    "input_text": text,
                    "metadata": record,
                    "topic": topic,
                    "pdf_path": str(pdf_path),
                }
            )
            if log_every and len(records) % log_every == 0:
                logger.info("Compiled %d papers (latest %s)", len(records), short_id)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile PDFs + metadata into a single JSONL.")
    parser.add_argument("--input_root", default="data/openalex")
    parser.add_argument("--output_path", default="data/paper_parser/processed/compiled_papers.jsonl")
    parser.add_argument("--max_chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max_papers", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--log_every", type=int, default=50, help="Log progress every N papers.")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    records = _collect_records(Path(args.input_root), args.max_chars, args.max_papers, args.log_every)
    write_jsonl(output_path, records)
    logger.info("Wrote %d records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
