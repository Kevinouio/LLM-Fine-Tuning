import json
from pathlib import Path
from typing import Any, Iterable, List, Dict, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    if path.parent:
        ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "
")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    if path.parent:
        ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
