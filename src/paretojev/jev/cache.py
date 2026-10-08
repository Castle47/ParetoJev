from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class DecisionCache:
    """SQLite-backed cache for expensive Jev/decision-model calls."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    cache_key TEXT PRIMARY KEY,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def canonical_json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def key_for(cls, provider: str, model: str, payload: Any) -> str:
        raw = f"{provider}\n{model}\n{cls.canonical_json(payload)}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, provider: str, model: str, payload: Any) -> dict[str, Any] | None:
        key = self.key_for(provider, model, payload)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM decisions WHERE cache_key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, provider: str, model: str, payload: Any, response: dict[str, Any]) -> None:
        key = self.key_for(provider, model, payload)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decisions
                (cache_key, request_json, response_json, provider, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    self.canonical_json(payload),
                    self.canonical_json(response),
                    provider,
                    model,
                    time.time(),
                ),
            )
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
