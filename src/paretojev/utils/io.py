from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON atomically so interrupted runs do not corrupt experiment state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonl_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(jsonl_line(row))


def append_jsonl_row(
    handle: TextIO,
    row: dict[str, Any],
    *,
    flush: bool = True,
    fsync: bool = False,
) -> None:
    """Append one JSONL record and optionally make progress durable."""

    handle.write(jsonl_line(row))
    if flush or fsync:
        handle.flush()
    if fsync:
        os.fsync(handle.fileno())


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL with line-numbered parse errors."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row at {path}:{line_number} must be an object, "
                    f"got {type(value).__name__}"
                )
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))
