"""Persist attempted public-demo requests across interrupted scoring processes.

The caller must hold an exclusive run-directory lock. Reserve before sending:
failed or interrupted requests still consume the registered lifetime budget.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from paretojev.jev.client import DecisionAPIError
from paretojev.utils.io import append_jsonl_row
from paretojev.utils.provenance import canonical_hash


class DurableBudgetSession(requests.Session):
    def __init__(self, ledger_path: str | Path, maximum: int):
        super().__init__()
        self.ledger_path = Path(ledger_path)
        self.maximum = maximum
        if maximum <= 0:
            raise ValueError('Lifetime request budget must be positive')
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.used = 0
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                # Fail closed on an incomplete/corrupt reservation record.
                row = json.loads(line)
                if row['attempt'] != self.used + 1:
                    raise ValueError('Nonsequential request budget ledger')
                self.used += 1
        if self.used > maximum:
            raise ValueError('Ledger already exceeds the declared budget')

    def post(self, url, **kwargs):
        if self.used >= self.maximum:
            raise DecisionAPIError('Registered lifetime HTTP budget exhausted')
        row = {'attempt': self.used + 1, 'reserved_at_utc': datetime.now(timezone.utc).isoformat(),
               'endpoint': url, 'payload_sha256': canonical_hash(kwargs.get('json'))}
        with self.ledger_path.open('a', encoding='utf-8') as handle:
            append_jsonl_row(handle, row, flush=True, fsync=True)
        self.used += 1
        return super().post(url, **kwargs)
