import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.common.io import ensure_dir, write_jsonl
from src.common.logging import setup_logging


logger = setup_logging("paper_parser.preprocess")


def _load_pdf(path: Path):
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF preprocessing. Install pymupdf.") from exc
    return fitz.open(path)


def _extract_pdf(
    path: Path,
    extract_images: bool,
    image_dir: Path,
    include_image_placeholders: bool,
) -> Dict[str, Any]:
    doc = _load_pdf(path)
    text_chunks: List[str] = []
    figures: List[Dict[str, Any]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text_chunks.append(page.get_text("text"))

        if not extract_images:
            continue

        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            pix = doc.extract_image(xref)
            if not pix:
                continue
            ext = pix.get("ext", "png")
            img_name = f"{path.stem}_p{page_index + 1}_{img_index}.{ext}"
            out_path = image_dir / img_name
            out_path.write_bytes(pix.get("image", b""))
            figures.append({"page": page_index + 1, "path": str(out_path)})
            if include_image_placeholders:
                text_chunks.append(f"[FIGURE:{img_name}]")

    return {
        "id": path.stem,
        "input_text": "\n".join(chunk for chunk in text_chunks if chunk),
        "figures": figures,
        "source_path": str(path),
    }


def _extract_text(path: Path) -> Dict[str, Any]:
    return {
        "id": path.stem,
        "input_text": path.read_text(encoding="utf-8"),
        "figures": [],
        "source_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw paper text.")
    parser.add_argument("--raw_dir", default="data/paper_parser/raw")
    parser.add_argument("--output_path", default="data/paper_parser/processed/processed.jsonl")
    parser.add_argument("--extract_images", action="store_true")
    parser.add_argument("--image_dir", default="data/paper_parser/processed/images")
    parser.add_argument("--include_image_placeholders", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error("Raw directory does not exist: %s", raw_dir)
        return

    image_dir = Path(args.image_dir)
    if args.extract_images:
        ensure_dir(image_dir)

    records: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            records.append(
                _extract_pdf(
                    path,
                    extract_images=args.extract_images,
                    image_dir=image_dir,
                    include_image_placeholders=args.include_image_placeholders,
                )
            )
        elif path.suffix.lower() in {".txt", ".md"}:
            records.append(_extract_text(path))

    if not records:
        logger.warning("No files found in %s", raw_dir)
        return

    write_jsonl(args.output_path, records)
    logger.info("Wrote %d records to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
