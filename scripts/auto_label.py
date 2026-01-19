import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz
from jsonschema import Draft7Validator

if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "set_verbosity"):
    fitz.TOOLS.set_verbosity(0)

from src.common.generation import generate_json
from src.common.hf import ModelSpec, load_model_for_inference, load_tokenizer
from src.common.io import ensure_dir, load_json, read_jsonl, write_jsonl
from src.common.logging import setup_logging

STRICT_TEMPLATE = (
    "{\n"
    '  "quality_summary": "string",\n'
    '  "quality_scores": {\n'
    '    "clarity": 1,\n'
    '    "problem_understanding": 1,\n'
    '    "novelty": 1,\n'
    '    "method_validity": 1,\n'
    '    "evidence_strength": 1,\n'
    '    "evaluation_quality": 1,\n'
    '    "limitations": 1,\n'
    '    "impact": 1\n'
    "  },\n"
    '  "quality_flags": ["string"],\n'
    '  "paper_type": "theoretical|empirical|survey|system|other",\n'
    '  "title": "string",\n'
    '  "abstract": "string",\n'
    '  "contributions": ["string"],\n'
    '  "limitations": ["string"],\n'
    '  "summary_simple": "string",\n'
    '  "methods": ["string"],\n'
    '  "datasets": ["string"],\n'
    '  "results": ["string"]\n'
    "}"
)


@dataclass
class Stats:
    total_records: int = 0
    attempted: int = 0
    processed: int = 0
    successes: int = 0
    missing_pdf: int = 0
    invalid_schema: int = 0
    generation_failures: int = 0


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


def _load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done = set()
    for record in read_jsonl(path):
        record_id = record.get("id")
        if record_id:
            done.add(record_id)
    return done


def _iter_manifests(root: Path) -> Iterable[Path]:
    return root.rglob("manifest.jsonl")


def _collect_records(input_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for manifest in _iter_manifests(input_root):
        topic = manifest.parent.name
        pdf_dir = manifest.parent / "pdfs"
        for record in read_jsonl(manifest):
            openalex_id = record.get("openalex_id")
            if not openalex_id:
                continue
            records.append(
                {
                    "openalex_id": openalex_id,
                    "metadata": record,
                    "topic": topic,
                    "pdf_dir": pdf_dir,
                }
            )
    return records


def _collect_from_compiled(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for record in read_jsonl(path):
        records.append(
            {
                "openalex_id": record.get("openalex_id"),
                "metadata": record.get("metadata"),
                "topic": record.get("topic"),
                "pdf_path": record.get("pdf_path"),
                "input_text": record.get("input_text", ""),
                "sections": record.get("sections"),
                "sectioned_input": record.get("sectioned_input"),
                "id": record.get("id"),
            }
        )
    return records


def _record_has_pdf(record: Dict[str, Any]) -> bool:
    if record.get("input_text"):
        return True
    pdf_path = record.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        return True
    pdf_dir = record.get("pdf_dir")
    if pdf_dir:
        openalex_id = record.get("openalex_id", "")
        short_id = record.get("id") or _short_id(openalex_id)
        return (Path(pdf_dir) / f"{short_id}.pdf").exists()
    return False


def _count_available(records: List[Dict[str, Any]]) -> int:
    return sum(1 for record in records if _record_has_pdf(record))


def _validate_output(payload: Dict[str, Any], validator: Draft7Validator) -> bool:
    errors = list(validator.iter_errors(payload))
    return not errors


def _schema_errors(payload: Dict[str, Any], validator: Draft7Validator) -> List[str]:
    errors = list(validator.iter_errors(payload))
    messages = []
    for err in errors[:5]:
        messages.append(err.message)
    return messages


def _format_sectioned_input(sections: Dict[str, str]) -> str:
    order = [
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
    parts = []
    for key in order:
        content = sections.get(key, "")
        if content:
            parts.append(f"{key.upper()}:\n{content}")
    return "\n\n".join(parts)


def _build_strict_prompt(input_text: str) -> str:
    return (
        "You are a research paper parser. Return JSON only. "
        "Follow the exact schema and numeric ranges. "
        "All quality_scores must be integers 1-10. "
        "Do not include extra keys. "
        "Do not use markdown or code fences. "
        "Start with '{' and end with '}'. "
        "If a field is unknown, use an empty string or empty array, not null.\n\n"
        "INPUT:\n"
        f"{input_text}\n\n"
        "OUTPUT JSON (strict template):\n"
        f"{STRICT_TEMPLATE}\n"
    )


def _format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _normalize_scores(raw_scores: Any) -> Dict[str, int]:
    required = [
        "clarity",
        "problem_understanding",
        "novelty",
        "method_validity",
        "evidence_strength",
        "evaluation_quality",
        "limitations",
        "impact",
    ]
    synonyms = {
        "technical_soundness": "method_validity",
        "soundness": "method_validity",
        "methodology": "method_validity",
        "completeness": "evaluation_quality",
        "evaluation": "evaluation_quality",
        "evidence": "evidence_strength",
        "evidence_quality": "evidence_strength",
    }
    scores: Dict[str, Any] = raw_scores if isinstance(raw_scores, dict) else {}
    normalized: Dict[str, Any] = {}

    for key, value in scores.items():
        target = synonyms.get(key, key)
        if target not in required or target in normalized:
            continue
        normalized[target] = value

    def coerce(val: Any) -> int:
        if isinstance(val, (int, float)):
            if 0 < val <= 1:
                val = round(val * 10)
            val = int(round(val))
        else:
            val = 5
        if val < 1:
            return 1
        if val > 10:
            return 10
        return val

    for key in required:
        normalized[key] = coerce(normalized.get(key))

    return normalized


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def _normalize_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    quality_summary = raw.get("quality_summary") or raw.get("summary_simple") or raw.get("digest") or ""
    summary_simple = raw.get("summary_simple") or raw.get("digest") or ""
    return {
        "quality_summary": str(quality_summary),
        "quality_scores": _normalize_scores(raw.get("quality_scores")),
        "quality_flags": _ensure_list(raw.get("quality_flags")),
        "paper_type": str(raw.get("paper_type") or "other"),
        "title": str(raw.get("title") or ""),
        "abstract": str(raw.get("abstract") or ""),
        "contributions": _ensure_list(raw.get("contributions")),
        "limitations": _ensure_list(raw.get("limitations")),
        "summary_simple": str(summary_simple),
        "methods": _ensure_list(raw.get("methods")),
        "datasets": _ensure_list(raw.get("datasets")),
        "results": _ensure_list(raw.get("results")),
    }


def _write_failure(path: Path, payload: Dict[str, Any]) -> None:
    write_jsonl(path, [payload])


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label papers into silver JSONL.")
    parser.add_argument("--input_root", default="data/openalex")
    parser.add_argument("--input_jsonl", default=None, help="Use a compiled JSONL instead of manifests.")
    parser.add_argument("--output_path", default="data/paper_parser/processed/auto_labeled.jsonl")
    parser.add_argument("--failures_path", default="data/paper_parser/processed/auto_label_failures.jsonl")
    parser.add_argument("--schema_path", default="schemas/paper_digest.schema.json")
    parser.add_argument("--model_name", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max_papers", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1, help="Log progress every N papers.")
    parser.add_argument("--print_json", action="store_true", help="Print generated JSON before validation.")
    parser.add_argument("--use_sections", action="store_true", help="Prefer sectioned input when available.")
    parser.add_argument("--log_file", default=None, help="Write logs to a file.")
    args = parser.parse_args()

    logger = setup_logging("auto_label", log_file=args.log_file)

    output_path = Path(args.output_path)
    failures_path = Path(args.failures_path)
    ensure_dir(output_path.parent)
    ensure_dir(failures_path.parent)

    schema = load_json(args.schema_path)
    validator = Draft7Validator(schema)

    tokenizer = load_tokenizer(args.model_name)
    model = load_model_for_inference(
        ModelSpec(model_name=args.model_name, adapter_path=args.adapter_path, use_qlora=args.use_qlora)
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    done_ids = _load_done_ids(output_path)
    if args.input_jsonl:
        records = _collect_from_compiled(Path(args.input_jsonl))
    else:
        records = _collect_records(Path(args.input_root))
    stats = Stats(total_records=_count_available(records))

    for record in records:
        if args.max_papers and stats.processed >= args.max_papers:
            break
        openalex_id = record.get("openalex_id", "")
        short_id = record.get("id") or _short_id(openalex_id)
        if short_id in done_ids:
            continue
        stats.attempted += 1

        input_text = record.get("input_text")
        if args.use_sections and record.get("sections"):
            input_text = record.get("sectioned_input") or _format_sectioned_input(record["sections"])
        if not input_text:
            pdf_path = Path(record["pdf_dir"]) / f"{short_id}.pdf"
            if not pdf_path.exists():
                stats.missing_pdf += 1
                logger.info("Skipping %s: missing pdf at %s", short_id, pdf_path)
                _write_failure(
                    failures_path,
                    {
                        "id": short_id,
                        "openalex_id": openalex_id,
                        "error": "missing_pdf",
                    },
                )
                continue

            try:
                input_text = _extract_pdf_text(pdf_path)
            except Exception as exc:
                stats.generation_failures += 1
                logger.info("Skipping %s: extract_failed (%s)", short_id, exc)
                _write_failure(
                    failures_path,
                    {
                        "id": short_id,
                        "openalex_id": openalex_id,
                        "error": f"extract_failed: {exc}",
                    },
                )
                continue

        if args.max_chars and len(input_text) > args.max_chars:
            input_text = input_text[: args.max_chars]

        if args.log_every and stats.attempted % args.log_every == 0:
            logger.info(
                "Processing %d/%d (successes=%d): %s",
                stats.attempted,
                stats.total_records,
                stats.successes,
                short_id,
            )

        prompt = _build_strict_prompt(input_text)
        prompt = _format_prompt(tokenizer, prompt)
        output_json = generate_json(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        if args.print_json and isinstance(output_json, dict):
            print(json.dumps(output_json, indent=2))

        if not isinstance(output_json, dict) or "raw_text" in output_json:
            stats.generation_failures += 1
            logger.info("Skipping %s: generation_invalid_json", short_id)
            strict_prompt = _build_strict_prompt(input_text)
            strict_prompt = _format_prompt(tokenizer, strict_prompt)
            output_json = generate_json(
                model=model,
                tokenizer=tokenizer,
                prompt=strict_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            if args.print_json and isinstance(output_json, dict):
                print(json.dumps(output_json, indent=2))
            if not isinstance(output_json, dict) or "raw_text" in output_json:
                _write_failure(
                    failures_path,
                    {
                        "id": short_id,
                        "openalex_id": openalex_id,
                        "error": "generation_invalid_json",
                        "raw_text": output_json.get("raw_text") if isinstance(output_json, dict) else None,
                    },
                )
                continue

        output_json = _normalize_output(output_json)

        if not _validate_output(output_json, validator):
            stats.invalid_schema += 1
            logger.info("Skipping %s: schema_invalid", short_id)
            errors = _schema_errors(output_json, validator)
            strict_prompt = _build_strict_prompt(input_text)
            strict_prompt = _format_prompt(tokenizer, strict_prompt)
            output_json_retry = generate_json(
                model=model,
                tokenizer=tokenizer,
                prompt=strict_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            if args.print_json and isinstance(output_json_retry, dict):
                print(json.dumps(output_json_retry, indent=2))
            output_json_retry = _normalize_output(output_json_retry)
            if _validate_output(output_json_retry, validator):
                output_json = output_json_retry
            else:
                _write_failure(
                    failures_path,
                    {
                        "id": short_id,
                        "openalex_id": openalex_id,
                        "error": "schema_invalid",
                        "schema_errors": errors,
                        "output_json": output_json,
                    },
                )
                continue

        write_jsonl(
            output_path,
            [
                {
                    "id": short_id,
                    "input_text": input_text,
                    "output_json": output_json,
                    "metadata": record["metadata"],
                    "topic": record["topic"],
                    "pdf_path": record.get("pdf_path"),
                }
            ],
        )
        stats.processed += 1
        stats.successes += 1
        done_ids.add(short_id)

    logger.info("Auto-label stats: %s", json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
