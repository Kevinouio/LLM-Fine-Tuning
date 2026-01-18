import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_config(path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_fetch(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    name = entry["name"]
    out_dir = Path(args.out_root) / name
    pdf_dir = out_dir / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/openalex/openalex_fetch.py",
        "--n_low",
        str(entry["n_low"]),
        "--n_high",
        str(entry["n_high"]),
        "--out",
        str(out_dir / "manifest.jsonl"),
        "--pdf_dir",
        str(pdf_dir),
        "--mailto",
        args.mailto,
        "--seed",
        str(args.seed),
    ]

    if entry.get("field_filter"):
        cmd.extend(["--field_filter", entry["field_filter"]])
    elif entry.get("field"):
        cmd.extend(["--field", entry["field"]])

    if args.sleep_api is not None:
        cmd.extend(["--sleep_api", str(args.sleep_api)])
    if args.sleep_pdf is not None:
        cmd.extend(["--sleep_pdf", str(args.sleep_pdf)])
    if args.max_attempts is not None:
        cmd.extend(["--max_attempts", str(args.max_attempts)])
    if args.per_page is not None:
        cmd.extend(["--per_page", str(args.per_page)])
    if args.topic_scope:
        cmd.extend(["--topic_scope", args.topic_scope])
    if args.require_oa:
        cmd.extend(["--require_oa"])

    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenAlex papers across multiple fields.")
    parser.add_argument("--config", default="scripts/openalex/topics.json")
    parser.add_argument("--out_root", default="data/openalex")
    parser.add_argument("--mailto", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sleep_api", type=float, default=0.2)
    parser.add_argument("--sleep_pdf", type=float, default=0.5)
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--per_page", type=int, default=200)
    parser.add_argument("--topic_scope", choices=("primary", "any"), default="any")
    parser.add_argument("--require_oa", action="store_true")
    args = parser.parse_args()

    entries = _load_config(args.config)
    for entry in entries:
        _run_fetch(entry, args)


if __name__ == "__main__":
    main()
