import argparse
import json
import hashlib
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
from src.common.io import append_jsonl, ensure_dir, load_json, read_jsonl, write_jsonl
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

QUALITY_TEMPLATE = (
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
    '  "quality_flags": ["string"]\n'
    "}"
)

QUALITY_REWRITE_TEMPLATE = "{\n  \"quality_summary\": \"string\"\n}"

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


def _load_failed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    failed = set()
    for record in read_jsonl(path):
        record_id = record.get("id")
        if record_id:
            failed.add(record_id)
    return failed


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


def _resolve_pdf_path(record: Dict[str, Any], short_id: str) -> Optional[Path]:
    pdf_path = record.get("pdf_path")
    if pdf_path:
        return Path(pdf_path)
    pdf_dir = record.get("pdf_dir")
    if pdf_dir:
        return Path(pdf_dir) / f"{short_id}.pdf"
    return None


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_existing_state(path: Path, dedupe_pdf_path: bool, dedupe_pdf_hash: bool) -> tuple[Set[str], Set[str], Set[str]]:
    done_ids = set()
    seen_paths: Set[str] = set()
    seen_hashes: Set[str] = set()
    if not path.exists():
        return done_ids, seen_paths, seen_hashes
    for record in read_jsonl(path):
        record_id = record.get("id")
        if record_id:
            done_ids.add(record_id)
        if dedupe_pdf_path:
            pdf_path = record.get("pdf_path")
            if pdf_path:
                seen_paths.add(str(Path(pdf_path)))
        if dedupe_pdf_hash:
            pdf_hash = record.get("pdf_sha256")
            if pdf_hash:
                seen_hashes.add(str(pdf_hash))
    return done_ids, seen_paths, seen_hashes


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


def _trim_text(text: str, limit: int) -> str:
    if not limit or len(text) <= limit:
        return text
    return text[:limit]


def _format_sectioned_input(
    sections: Dict[str, str], max_section_chars: int, max_other_chars: int
) -> str:
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
            limit = max_other_chars if key == "other" else max_section_chars
            content = _trim_text(content, limit)
            if content:
                parts.append(f"{key.upper()}:\n{content}")
    return "\n\n".join(parts)


def _build_strict_prompt_parts() -> tuple[str, str]:
    prefix = (
        "You are a research paper parser. Return JSON only. "
        "Follow the exact schema and numeric ranges. "
        "quality_summary must be 2-4 sentences that assess writing quality, "
        "methods, evidence, limitations, and impact. "
        "Mention at least one strength and one weakness. "
        "summary_simple must be 3-5 sentences covering the problem, approach, and main findings. "
        "All quality_scores must be integers 1-10. "
        "Do not include extra keys. "
        "Do not use markdown or code fences. "
        "Start with '{' and end with '}'. "
        "Replace all template placeholders with real values. "
        "If a field is unknown, use an empty string or empty array, not null.\n\n"
        "INPUT:\n"
    )
    suffix = "\n\nOUTPUT JSON (strict template):\n" f"{STRICT_TEMPLATE}\n"
    return prefix, suffix


def _build_quality_prompt_parts() -> tuple[str, str]:
    prefix = (
        "You are a research paper reviewer. Return JSON only. "
        "Write a quality_summary that critiques the paper quality, not a content summary. "
        "Do NOT describe the study aims, methods, or results unless you are criticizing them. "
        "Avoid phrases like 'This paper/study/research...'. "
        "Use 2-4 sentences and mention at least one strength and one weakness. "
        "All quality_scores must be integers 1-10. "
        "quality_flags should be short, specific issues or an empty array. "
        "Do not include extra keys. "
        "Do not use markdown or code fences. "
        "Start with '{' and end with '}'. "
        "Replace all template placeholders with real values. "
        "If evidence is missing, say so in quality_summary and add a flag. "
        "Do not give uniform scores or all 5s unless the paper is truly average.\n\n"
        "INPUT:\n"
    )
    suffix = "\n\nOUTPUT JSON (quality template):\n" f"{QUALITY_TEMPLATE}\n"
    return prefix, suffix


def _build_quality_rewrite_prompt_parts() -> tuple[str, str]:
    prefix = (
        "Rewrite the QUALITY_SUMMARY to be purely evaluative. "
        "Do not describe the study aims, methods, or results unless criticizing them. "
        "Avoid phrases like 'This paper/study/research...'. "
        "Use 2-3 sentences with one explicit strength and one explicit weakness. "
        "Return JSON only and do not add extra keys.\n\n"
        "QUALITY_SUMMARY:\n"
    )
    suffix = "\n\nOUTPUT JSON (rewrite template):\n" f"{QUALITY_REWRITE_TEMPLATE}\n"
    return prefix, suffix


def _compose_prompt(input_text: str, prefix: str, suffix: str) -> str:
    return f"{prefix}{input_text}{suffix}"


def _resolve_max_context_tokens(model, tokenizer, override: int) -> int:
    if override and override > 0:
        return override
    candidates: List[int] = []
    for attr in (
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "max_length",
    ):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < 1_000_000:
        candidates.append(tok_limit)
    if candidates:
        return max(candidates)
    return 8192


def _truncate_input_for_prompt(
    tokenizer,
    input_text: str,
    prefix: str,
    suffix: str,
    max_context_tokens: int,
    max_new_tokens: int,
    buffer_tokens: int,
) -> tuple[str, bool, int, int]:
    budget = max_context_tokens - max_new_tokens - buffer_tokens
    prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_tokens = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    available = budget - len(prefix_tokens) - len(suffix_tokens)
    input_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"] if input_text else []
    if available <= 0:
        return "", True, len(input_ids), max(0, available)
    if len(input_ids) <= available:
        return input_text, False, len(input_ids), available
    truncated_ids = input_ids[:available]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, True, len(input_ids), available


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


def _normalize_paper_type(value: Any) -> str:
    allowed = {"theoretical", "empirical", "survey", "system", "other"}
    if not value:
        return "other"
    text = str(value).strip().lower()
    if text in allowed:
        return text
    if "|" in text:
        for part in (p.strip() for p in text.split("|")):
            if part in allowed:
                return part
    mapping = {
        "experimental": "empirical",
        "experiment": "empirical",
        "review": "survey",
        "systematic review": "survey",
        "case study": "empirical",
        "case-study": "empirical",
        "method": "system",
        "methods": "system",
    }
    return mapping.get(text, "other")


def _normalize_quality_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    quality_summary = raw.get("quality_summary") or raw.get("quality_assessment") or raw.get("quality") or ""
    return {
        "quality_summary": str(quality_summary),
        "quality_scores": _normalize_scores(raw.get("quality_scores")),
        "quality_flags": _ensure_list(raw.get("quality_flags")),
    }


def _normalize_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    quality_summary = raw.get("quality_summary") or raw.get("quality_assessment") or raw.get("quality") or ""
    summary_simple = raw.get("summary_simple") or raw.get("digest") or ""
    return {
        "quality_summary": str(quality_summary),
        "quality_scores": _normalize_scores(raw.get("quality_scores")),
        "quality_flags": _ensure_list(raw.get("quality_flags")),
        "paper_type": _normalize_paper_type(raw.get("paper_type")),
        "title": str(raw.get("title") or ""),
        "abstract": str(raw.get("abstract") or ""),
        "contributions": _ensure_list(raw.get("contributions")),
        "limitations": _ensure_list(raw.get("limitations")),
        "summary_simple": str(summary_simple),
        "methods": _ensure_list(raw.get("methods")),
        "datasets": _ensure_list(raw.get("datasets")),
        "results": _ensure_list(raw.get("results")),
    }

def _is_placeholder_summary(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    placeholders = {"string", "tbd", "todo", "n/a", "na", "unknown"}
    return lowered in placeholders


def _looks_like_content_summary(text: str) -> bool:
    lowered = text.lower()
    summary_markers = (
        "this paper",
        "this study",
        "this research",
        "the study",
        "the paper",
        "we investigate",
        "we examine",
        "we show",
        "we found",
        "the results",
        "the findings",
        "the authors",
        "aims to",
        "investigates",
        "examines",
        "demonstrates",
        "reports",
        "in this work",
    )
    return any(marker in lowered for marker in summary_markers)


def _has_quality_language(text: str) -> tuple[bool, bool]:
    lowered = text.lower()
    eval_terms = (
        "clear",
        "unclear",
        "rigor",
        "rigorous",
        "methodology",
        "evidence",
        "support",
        "limitations",
        "limitation",
        "weakness",
        "strength",
        "bias",
        "robust",
        "valid",
        "reproduc",
        "generaliz",
        "impact",
        "novel",
        "original",
        "coherent",
    )
    weakness_terms = (
        "limitation",
        "weakness",
        "concern",
        "issue",
        "problem",
        "unclear",
        "insufficient",
        "lack",
        "missing",
        "thin",
    )
    has_eval = any(term in lowered for term in eval_terms)
    has_weakness = any(term in lowered for term in weakness_terms)
    return has_eval, has_weakness


def _needs_quality_refresh(
    payload: Dict[str, Any],
    min_chars: int,
    allow_content_summary: bool,
    allow_without_weakness: bool,
) -> bool:
    if min_chars <= 0:
        return False
    summary = payload.get("quality_summary") or ""
    if _is_placeholder_summary(summary):
        return True
    if len(summary.strip()) < min_chars:
        return True
    has_eval, has_weakness = _has_quality_language(summary)
    if not allow_content_summary and _looks_like_content_summary(summary):
        if not has_eval or not has_weakness:
            return True
    if not has_eval:
        return True
    if not allow_without_weakness and not has_weakness:
        return True
    return False


def _scores_need_refresh(scores: Dict[str, int]) -> bool:
    if not scores:
        return True
    values = list(scores.values())
    if not values:
        return True
    if len(set(values)) <= 1:
        return True
    if values.count(5) >= max(6, len(values) - 1):
        return True
    return False


def _write_failure(path: Path, payload: Dict[str, Any]) -> None:
    append_jsonl(path, [payload])


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
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_context_tokens", type=int, default=16384, help="0 means infer from model/tokenizer.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--retry_temperature", type=float, default=0.4)
    parser.add_argument("--retry_top_p", type=float, default=0.95)
    parser.add_argument("--prompt_buffer_tokens", type=int, default=256, help="Reserved prompt buffer for chat templates.")
    parser.add_argument("--max_chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max_section_chars", type=int, default=4000, help="Max chars per section (0 means no limit).")
    parser.add_argument("--max_other_chars", type=int, default=1000, help="Max chars for OTHER section (0 to drop).")
    parser.add_argument("--max_papers", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1, help="Log progress every N papers.")
    parser.add_argument("--min_quality_chars", type=int, default=40, help="Minimum length for quality_summary (0 to disable).")
    parser.add_argument(
        "--allow_quality_summary",
        action="store_true",
        help="Allow quality_summary to read like a content summary.",
    )
    parser.add_argument(
        "--allow_quality_without_weakness",
        action="store_true",
        help="Allow quality_summary without an explicit weakness/limitation.",
    )
    parser.add_argument("--print_json", action="store_true", help="Print generated JSON before validation.")
    parser.add_argument("--use_sections", action="store_true", help="Prefer sectioned input when available.")
    parser.add_argument("--log_file", default=None, help="Write logs to a file.")
    parser.add_argument("--dedupe_pdf_path", action="store_true", help="Skip PDFs already labeled by path.")
    parser.add_argument("--dedupe_pdf_hash", action="store_true", help="Skip PDFs already labeled by content hash.")
    parser.add_argument("--skip_failed", action="store_true", help="Skip IDs that previously failed.")
    args = parser.parse_args()

    logger = setup_logging("auto_label", log_file=args.log_file)
    if args.max_papers == 0:
        if not args.dedupe_pdf_path and not args.dedupe_pdf_hash:
            args.dedupe_pdf_path = True
            args.dedupe_pdf_hash = True
            logger.info("Enabled dedupe by path and hash because max_papers=0")

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
    max_context_tokens = _resolve_max_context_tokens(model, tokenizer, args.max_context_tokens)
    max_new_tokens = args.max_new_tokens
    max_new_tokens_limit = max(1, max_context_tokens - args.prompt_buffer_tokens - 1)
    if max_new_tokens > max_new_tokens_limit:
        logger.info(
            "Clamping max_new_tokens from %d to %d to fit context window %d",
            max_new_tokens,
            max_new_tokens_limit,
            max_context_tokens,
        )
        max_new_tokens = max_new_tokens_limit
    strict_prefix, strict_suffix = _build_strict_prompt_parts()
    quality_prefix, quality_suffix = _build_quality_prompt_parts()
    rewrite_prefix, rewrite_suffix = _build_quality_rewrite_prompt_parts()
    retry_temperature = max(args.retry_temperature, args.temperature)
    retry_top_p = max(min(args.retry_top_p, 1.0), 0.1)
    rewrite_temperature = max(args.temperature, 0.3)
    rewrite_top_p = min(max(args.top_p, 0.9), 0.95)

    done_ids, seen_paths, seen_hashes = _load_existing_state(
        output_path, args.dedupe_pdf_path, args.dedupe_pdf_hash
    )
    failed_ids = _load_failed_ids(Path(args.failures_path)) if args.skip_failed else set()
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
        if args.skip_failed and short_id in failed_ids:
            logger.info("Skipping %s: previously failed", short_id)
            continue
        stats.attempted += 1

        pdf_path = _resolve_pdf_path(record, short_id)
        if args.dedupe_pdf_path and pdf_path and str(pdf_path) in seen_paths:
            logger.info("Skipping %s: duplicate pdf_path %s", short_id, pdf_path)
            continue

        pdf_hash = None
        if args.dedupe_pdf_hash and pdf_path and pdf_path.exists():
            pdf_hash = _hash_file(pdf_path)
            if pdf_hash in seen_hashes:
                logger.info("Skipping %s: duplicate pdf_hash %s", short_id, pdf_hash)
                continue

        input_text = record.get("input_text")
        if args.use_sections and record.get("sections"):
            sections = dict(record["sections"])
            if not sections.get("title"):
                metadata = record.get("metadata") or {}
                if metadata.get("title"):
                    sections["title"] = str(metadata["title"])
            input_text = _format_sectioned_input(
                sections,
                args.max_section_chars,
                args.max_other_chars,
            )
        if not input_text:
            if not pdf_path or not pdf_path.exists():
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
        raw_input_text = input_text
        input_text, was_truncated, input_tokens, input_budget = _truncate_input_for_prompt(
            tokenizer,
            raw_input_text,
            strict_prefix,
            strict_suffix,
            max_context_tokens,
            max_new_tokens,
            args.prompt_buffer_tokens,
        )
        if raw_input_text and not input_text:
            stats.generation_failures += 1
            logger.info("Skipping %s: prompt_budget_too_small", short_id)
            _write_failure(
                failures_path,
                {
                    "id": short_id,
                    "openalex_id": openalex_id,
                    "error": "prompt_budget_too_small",
                    "max_context_tokens": max_context_tokens,
                    "max_new_tokens": max_new_tokens,
                },
            )
            continue
        if was_truncated:
            logger.info(
                "Truncated %s input from %d to %d tokens to fit prompt budget",
                short_id,
                input_tokens,
                input_budget,
            )

        if args.log_every and stats.attempted % args.log_every == 0:
            logger.info(
                "Processing %d/%d (successes=%d): %s",
                stats.attempted,
                stats.total_records,
                stats.successes,
                short_id,
            )

        prompt = _compose_prompt(input_text, strict_prefix, strict_suffix)
        prompt = _format_prompt(tokenizer, prompt)
        output_json = generate_json(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_input_tokens=max_context_tokens,
        )
        if args.print_json and isinstance(output_json, dict):
            print(json.dumps(output_json, indent=2))

        if not isinstance(output_json, dict) or "raw_text" in output_json:
            stats.generation_failures += 1
            logger.info("Skipping %s: generation_invalid_json", short_id)
            strict_prompt = _compose_prompt(input_text, strict_prefix, strict_suffix)
            strict_prompt = _format_prompt(tokenizer, strict_prompt)
            output_json = generate_json(
                model=model,
                tokenizer=tokenizer,
                prompt=strict_prompt,
                max_new_tokens=max_new_tokens,
                temperature=retry_temperature,
                top_p=retry_top_p,
                max_input_tokens=max_context_tokens,
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
            strict_prompt = _compose_prompt(input_text, strict_prefix, strict_suffix)
            strict_prompt = _format_prompt(tokenizer, strict_prompt)
            output_json_retry = generate_json(
                model=model,
                tokenizer=tokenizer,
                prompt=strict_prompt,
                max_new_tokens=max_new_tokens,
                temperature=retry_temperature,
                top_p=retry_top_p,
                max_input_tokens=max_context_tokens,
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

        if _needs_quality_refresh(
            output_json,
            args.min_quality_chars,
            args.allow_quality_summary,
            args.allow_quality_without_weakness,
        ) or _scores_need_refresh(output_json.get("quality_scores") or {}):
            quality_input, q_truncated, q_tokens, q_budget = _truncate_input_for_prompt(
                tokenizer,
                raw_input_text,
                quality_prefix,
                quality_suffix,
                max_context_tokens,
                min(768, max_new_tokens),
                args.prompt_buffer_tokens,
            )
            if q_truncated:
                logger.info(
                    "Truncated %s quality input from %d to %d tokens",
                    short_id,
                    q_tokens,
                    q_budget,
                )
            quality_prompt = _compose_prompt(quality_input, quality_prefix, quality_suffix)
            quality_prompt = _format_prompt(tokenizer, quality_prompt)
            quality_json = generate_json(
                model=model,
                tokenizer=tokenizer,
                prompt=quality_prompt,
                max_new_tokens=min(768, max_new_tokens),
                temperature=max(args.temperature, 0.2),
                top_p=min(args.top_p, 0.95),
                max_input_tokens=max_context_tokens,
            )
            if args.print_json and isinstance(quality_json, dict):
                print(json.dumps(quality_json, indent=2))
            if not isinstance(quality_json, dict) or "raw_text" in quality_json:
                stats.generation_failures += 1
                logger.info("Skipping %s: quality_generation_invalid_json", short_id)
                _write_failure(
                    failures_path,
                    {
                        "id": short_id,
                        "openalex_id": openalex_id,
                        "error": "quality_generation_invalid_json",
                        "raw_text": quality_json.get("raw_text") if isinstance(quality_json, dict) else None,
                    },
                )
                continue
            quality_json = _normalize_quality_output(quality_json)
            summary_needs = _needs_quality_refresh(
                quality_json,
                args.min_quality_chars,
                args.allow_quality_summary,
                args.allow_quality_without_weakness,
            )
            scores_needs = _scores_need_refresh(quality_json.get("quality_scores") or {})
            if summary_needs:
                rewrite_input = quality_json.get("quality_summary", "")
                rewrite_prompt = _compose_prompt(rewrite_input, rewrite_prefix, rewrite_suffix)
                rewrite_prompt = _format_prompt(tokenizer, rewrite_prompt)
                rewritten = generate_json(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rewrite_prompt,
                    max_new_tokens=min(256, max_new_tokens),
                    temperature=rewrite_temperature,
                    top_p=rewrite_top_p,
                    max_input_tokens=max_context_tokens,
                )
                if args.print_json and isinstance(rewritten, dict):
                    print(json.dumps(rewritten, indent=2))
                if not isinstance(rewritten, dict) or "raw_text" in rewritten:
                    stats.generation_failures += 1
                    logger.info("Skipping %s: quality_summary_rewrite_failed", short_id)
                    _write_failure(
                        failures_path,
                        {
                            "id": short_id,
                            "openalex_id": openalex_id,
                            "error": "quality_summary_rewrite_failed",
                            "raw_text": rewritten.get("raw_text") if isinstance(rewritten, dict) else None,
                        },
                    )
                    continue
                rewrite_summary = str(rewritten.get("quality_summary") or "")
                quality_json["quality_summary"] = rewrite_summary
                summary_needs = _needs_quality_refresh(
                    quality_json,
                    args.min_quality_chars,
                    args.allow_quality_summary,
                    args.allow_quality_without_weakness,
                )
            if summary_needs or scores_needs:
                stats.invalid_schema += 1
                logger.info("Skipping %s: quality_summary_not_critical_or_uniform_scores", short_id)
                _write_failure(
                    failures_path,
                    {
                        "id": short_id,
                        "openalex_id": openalex_id,
                        "error": "quality_summary_not_critical_or_uniform_scores",
                        "output_json": quality_json,
                    },
                )
                continue
            output_json.update(quality_json)

        append_jsonl(
            output_path,
            [
                {
                    "id": short_id,
                    "input_text": input_text,
                    "output_json": output_json,
                    "metadata": record["metadata"],
                    "topic": record["topic"],
                    "pdf_path": str(pdf_path) if pdf_path else None,
                    "pdf_sha256": pdf_hash,
                }
            ],
        )
        if args.dedupe_pdf_path and pdf_path:
            seen_paths.add(str(pdf_path))
        if args.dedupe_pdf_hash and pdf_hash:
            seen_hashes.add(pdf_hash)
        stats.processed += 1
        stats.successes += 1
        done_ids.add(short_id)

    logger.info("Auto-label stats: %s", json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
