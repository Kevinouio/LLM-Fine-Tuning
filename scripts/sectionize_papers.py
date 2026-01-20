import argparse
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import ensure_dir, read_jsonl, write_jsonl
from src.common.logging import setup_logging


logger = setup_logging("sectionize_papers")


SECTION_HEADERS = {
    "abstract": {"abstract"},
    "introduction": {"introduction"},
    "methods": {"methods", "materials and methods", "methodology", "experimental", "materials"},
    "results": {"results"},
    "discussion": {"discussion"},
    "conclusion": {"conclusion", "conclusions", "summary"},
    "limitations": {"limitations", "threats to validity", "limitations and future work"},
}

SECTION_ORDER = [
    "title",
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "other",
]


def _normalize_heading(line: str) -> str:
    line = line.lower().strip()
    line = re.sub(r"^[\d\s\.\-:]+", "", line)
    line = re.sub(r"[^a-z\s]+", "", line)
    return re.sub(r"\s+", " ", line).strip()


def _clean_lines(text: str) -> List[str]:
    lines = []
    recent = deque(maxlen=10)
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if line.lower().startswith("page "):
            continue
        if line in recent:
            continue
        recent.append(line)
        lines.append(line)
    return lines


def _split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {key: [] for key in SECTION_ORDER}
    lines = _clean_lines(text)

    current = "other"
    seen_heading = False
    front_matter: List[str] = []

    for line in lines:
        heading = _normalize_heading(line)
        matched = None
        for section, headers in SECTION_HEADERS.items():
            if heading in headers:
                matched = section
                break
        if matched:
            current = matched
            seen_heading = True
            continue

        if not seen_heading:
            front_matter.append(line)

        if current == "abstract" and line.lower().startswith("abstract"):
            line = re.sub(r"^abstract[\s:.\-–—]+", "", line, flags=re.IGNORECASE).strip()

        sections[current].append(line)

    title = ""
    for line in front_matter:
        if line.lower().startswith("abstract"):
            continue
        title = line
        break

    if title:
        sections["title"] = [title]

    return {key: "\n".join(value).strip() for key, value in sections.items() if value or key in {"title", "abstract"}}


def _format_sectioned_input(sections: Dict[str, str]) -> str:
    parts = []
    for key in SECTION_ORDER:
        content = sections.get(key, "")
        if content:
            parts.append(f"{key.upper()}:\n{content}")
    return "\n\n".join(parts)


def _sectionize_record(record: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    input_text = record.get("input_text", "")
    if max_chars and len(input_text) > max_chars:
        input_text = input_text[:max_chars]

    sections = _split_sections(input_text)
    return {
        "id": record.get("id"),
        "openalex_id": record.get("openalex_id"),
        "sections": sections,
        "sectioned_input": _format_sectioned_input(sections),
        "metadata": record.get("metadata"),
        "topic": record.get("topic"),
        "pdf_path": record.get("pdf_path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sectionize compiled paper texts.")
    parser.add_argument("--input_jsonl", default="data/paper_parser/processed/compiled_papers.jsonl")
    parser.add_argument("--output_path", default="data/paper_parser/processed/sectioned_papers.jsonl")
    parser.add_argument("--max_chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    records = []
    for idx, record in enumerate(read_jsonl(Path(args.input_jsonl)), start=1):
        records.append(_sectionize_record(record, args.max_chars))
        if args.log_every and idx % args.log_every == 0:
            logger.info("Sectionized %d papers", idx)

    write_jsonl(output_path, records)
    logger.info("Wrote %d records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
