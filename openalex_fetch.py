import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


OPENALEX_URL = "https://api.openalex.org/works"
OVERSAMPLE_FACTOR = 10
MAX_SAMPLE_SIZE = 10000


@dataclass
class Counters:
    metadata_saved: int = 0
    pdfs_downloaded: int = 0
    missing_pdf_url: int = 0
    failed_downloads: int = 0
    rate_limit_waits: int = 0
    duplicates_skipped: int = 0


def _short_id(openalex_id: str) -> str:
    return openalex_id.rsplit("/", 1)[-1]


def _select_fields(default_fields: str, override: Optional[str]) -> str:
    return override if override else default_fields


def _request_with_retry(
    url: str,
    params: dict,
    timeout: int,
    max_retries: int,
    base_backoff: float,
    sleep_api: float,
    counters: Counters,
) -> requests.Response:
    attempt = 0
    while True:
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = base_backoff * (2 ** (attempt - 1))
            time.sleep(wait)
            continue
        if response.status_code in {429, 500, 502, 503, 504}:
            attempt += 1
            if attempt > max_retries:
                response.raise_for_status()
            wait = base_backoff * (2 ** (attempt - 1))
            if response.status_code == 429:
                counters.rate_limit_waits += 1
            time.sleep(wait)
            continue
        response.raise_for_status()
        if sleep_api:
            time.sleep(sleep_api)
        return response


def _openalex_params(
    filter_text: str,
    per_page: int,
    seed: Optional[int],
    sample: Optional[int],
    page: Optional[int],
    mailto: Optional[str],
    select_fields: str,
) -> dict:
    params: Dict[str, object] = {
        "filter": filter_text,
        "per-page": per_page,
        "select": select_fields,
    }
    if seed is not None:
        params["seed"] = seed
    if sample is not None:
        params["sample"] = sample
    if page:
        params["page"] = page
    if mailto:
        params["mailto"] = mailto
    return params


def _fetch_openalex_batch(
    filter_text: str,
    per_page: int,
    seed: Optional[int],
    sample: Optional[int],
    page: Optional[int],
    mailto: Optional[str],
    select_fields: str,
    timeout: int,
    max_retries: int,
    base_backoff: float,
    sleep_api: float,
    counters: Counters,
) -> Tuple[List[dict], Dict[str, object]]:
    params = _openalex_params(
        filter_text=filter_text,
        per_page=per_page,
        seed=seed,
        sample=sample,
        page=page,
        mailto=mailto,
        select_fields=select_fields,
    )
    response = _request_with_retry(
        OPENALEX_URL,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        base_backoff=base_backoff,
        sleep_api=sleep_api,
        counters=counters,
    )
    payload = response.json()
    return payload.get("results", []), payload.get("meta", {})


def _extract_metadata(work: dict) -> dict:
    best_oa = work.get("best_oa_location") or {}
    open_access = work.get("open_access") or {}
    primary_topic = work.get("primary_topic") or {}
    topics = work.get("topics") or []
    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "best_oa_location": {
            "pdf_url": best_oa.get("pdf_url"),
            "landing_page_url": best_oa.get("landing_page_url"),
            "license": best_oa.get("license"),
            "version": best_oa.get("version"),
        },
        "open_access": {
            "is_oa": open_access.get("is_oa"),
            "oa_status": open_access.get("oa_status"),
        },
        "primary_topic": primary_topic.get("display_name"),
        "topics": [t.get("display_name") for t in topics if isinstance(t, dict)],
    }


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_pdf_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        return response.content[:4] == b"%PDF"
    return response.content[:4] == b"%PDF"


def _download_pdf(
    pdf_url: str,
    out_path: Path,
    timeout: int,
    max_retries: int,
    base_backoff: float,
    sleep_pdf: float,
    counters: Counters,
) -> Tuple[bool, Optional[str]]:
    attempt = 0
    while True:
        try:
            response = requests.get(pdf_url, timeout=timeout)
        except requests.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                return False, str(exc)
            time.sleep(base_backoff * (2 ** (attempt - 1)))
            continue

        if response.status_code in {429, 500, 502, 503, 504}:
            attempt += 1
            if attempt > max_retries:
                return False, f"HTTP {response.status_code}"
            if response.status_code == 429:
                counters.rate_limit_waits += 1
            time.sleep(base_backoff * (2 ** (attempt - 1)))
            continue

        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"

        if not _is_pdf_response(response):
            return False, "Response is not a PDF"

        out_path.write_bytes(response.content)
        if sleep_pdf:
            time.sleep(sleep_pdf)
        return True, None


def _sample_and_download_bucket(
    filter_text: str,
    target_count: int,
    seed: Optional[int],
    per_page: int,
    max_attempts: int,
    seen_ids: set[str],
    mailto: Optional[str],
    select_fields: str,
    manifest_path: Path,
    failures_path: Path,
    pdf_dir: Path,
    timeout: int,
    max_retries: int,
    base_backoff: float,
    sleep_api: float,
    sleep_pdf: float,
    counters: Counters,
) -> None:
    attempts = 0
    downloaded_for_bucket = 0
    sample_size = min(MAX_SAMPLE_SIZE, max(per_page, target_count * OVERSAMPLE_FACTOR))

    while downloaded_for_bucket < target_count and (max_attempts <= 0 or attempts < max_attempts):
        attempts += 1
        attempt_seed = seed + attempts if seed is not None else random.randint(1, 2_000_000_000)
        total_pages = max(1, (sample_size + per_page - 1) // per_page)
        for page in range(1, total_pages + 1):
            batch, _ = _fetch_openalex_batch(
                filter_text=filter_text,
                per_page=per_page,
                seed=attempt_seed,
                sample=sample_size,
                page=page,
                mailto=mailto,
                select_fields=select_fields,
                timeout=timeout,
                max_retries=max_retries,
                base_backoff=base_backoff,
                sleep_api=sleep_api,
                counters=counters,
            )
            if not batch:
                break

            random.shuffle(batch)
            for work in batch:
                metadata = _extract_metadata(work)
                openalex_id = metadata.get("openalex_id")
                if not openalex_id:
                    continue
                short_id = _short_id(openalex_id)
                print(json.dumps(metadata, indent=2))
                if short_id in seen_ids:
                    counters.duplicates_skipped += 1
                    print(f"Skipping {short_id}: duplicate")
                    continue
                seen_ids.add(short_id)

                pdf_url = metadata.get("best_oa_location", {}).get("pdf_url")
                if not pdf_url:
                    counters.missing_pdf_url += 1
                    _write_jsonl(failures_path, [{**metadata, "error": "Missing pdf_url"}])
                    print(f"Skipping {short_id}: missing pdf_url")
                    continue

                pdf_path = pdf_dir / f"{short_id}.pdf"
                if pdf_path.exists():
                    counters.duplicates_skipped += 1
                    print(f"Skipping {short_id}: pdf already exists")
                    continue

                ok, error = _download_pdf(
                    pdf_url=pdf_url,
                    out_path=pdf_path,
                    timeout=timeout,
                    max_retries=max_retries,
                    base_backoff=base_backoff,
                    sleep_pdf=sleep_pdf,
                    counters=counters,
                )
                if not ok:
                    counters.failed_downloads += 1
                    _write_jsonl(failures_path, [{**metadata, "error": error}])
                    print(f"Skipping {short_id}: download failed ({error})")
                    continue

                _write_jsonl(manifest_path, [metadata])
                counters.metadata_saved += 1
                counters.pdfs_downloaded += 1
                downloaded_for_bucket += 1

                if downloaded_for_bucket >= target_count:
                    break

            if downloaded_for_bucket >= target_count:
                break
        sample_size = min(MAX_SAMPLE_SIZE, sample_size * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenAlex papers and download PDFs.")
    parser.add_argument("--n_low", type=int, required=True)
    parser.add_argument("--n_high", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="manifest.jsonl")
    parser.add_argument("--pdf_dir", default="pdfs")
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--sleep_api", type=float, default=0.2)
    parser.add_argument("--sleep_pdf", type=float, default=0.5)
    parser.add_argument("--max_attempts", type=int, default=10, help="0 means unlimited attempts.")
    parser.add_argument("--per_page", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--base_backoff", type=float, default=1.0)
    parser.add_argument(
        "--select_fields",
        default=(
            "id,doi,title,publication_year,cited_by_count,"
            "best_oa_location,open_access,primary_topic,topics"
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.out)
    failures_path = manifest_path.with_name("failures.jsonl")
    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    counters = Counters()

    seen_ids: set[str] = set()

    _sample_and_download_bucket(
        filter_text="cited_by_count:>0,cited_by_count:<50",
        target_count=args.n_low,
        seed=args.seed,
        per_page=args.per_page,
        max_attempts=args.max_attempts,
        seen_ids=seen_ids,
        mailto=args.mailto,
        select_fields=_select_fields(args.select_fields, None),
        manifest_path=manifest_path,
        failures_path=failures_path,
        pdf_dir=pdf_dir,
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_backoff=args.base_backoff,
        sleep_api=args.sleep_api,
        sleep_pdf=args.sleep_pdf,
        counters=counters,
    )

    _sample_and_download_bucket(
        filter_text="cited_by_count:>49",
        target_count=args.n_high,
        seed=args.seed,
        per_page=args.per_page,
        max_attempts=args.max_attempts,
        seen_ids=seen_ids,
        mailto=args.mailto,
        select_fields=_select_fields(args.select_fields, None),
        manifest_path=manifest_path,
        failures_path=failures_path,
        pdf_dir=pdf_dir,
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_backoff=args.base_backoff,
        sleep_api=args.sleep_api,
        sleep_pdf=args.sleep_pdf,
        counters=counters,
    )

    summary = {
        "metadata_saved": counters.metadata_saved,
        "pdfs_downloaded": counters.pdfs_downloaded,
        "missing_pdf_url": counters.missing_pdf_url,
        "failed_downloads": counters.failed_downloads,
        "rate_limit_waits": counters.rate_limit_waits,
        "duplicates_skipped": counters.duplicates_skipped,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
