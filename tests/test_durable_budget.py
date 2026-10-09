import json

import pytest
import requests

from paretojev.jev.client import DecisionAPIError
from paretojev.jev.durable_budget import DurableBudgetSession


def test_failed_attempt_consumes_budget_across_restart(tmp_path, monkeypatch):
    ledger = tmp_path / 'attempts.jsonl'
    def post(*args, **kwargs):
        assert json.loads(ledger.read_text())['attempt'] == 1
        raise requests.Timeout('sent but timed out')
    monkeypatch.setattr(requests.Session, 'post', post)
    first = DurableBudgetSession(ledger, 1)
    with pytest.raises(requests.Timeout):
        first.post('https://example.test', json={'q': 'test'})
    resumed = DurableBudgetSession(ledger, 1)
    with pytest.raises(DecisionAPIError, match='lifetime'):
        resumed.post('https://example.test', json={'q': 'test'})
    assert len(ledger.read_text().splitlines()) == 1


def test_corrupt_budget_ledger_fails_closed(tmp_path):
    ledger = tmp_path / 'attempts.jsonl'
    ledger.write_text('{"attempt":1}\n{"attempt":')
    with pytest.raises(json.JSONDecodeError):
        DurableBudgetSession(ledger, 3)
    ledger.write_text('{"attempt":2}\n')
    with pytest.raises(ValueError, match='Nonsequential'):
        DurableBudgetSession(ledger, 3)
