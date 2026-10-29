import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_emer_10k_pilot as evaluator
from prepare_emer_10k_pilot import select


def config():
    return {
        "selection": {
            "namespace": "test",
            "seed": 7,
            "stratum_quotas": [2, 1],
            "slates": 3,
            "heldout_remaining_slates": 3
        }
    }


def test_hash_selection_is_deterministic_stratified_and_ordered():
    strata = np.array([0, 1, 0, 1, 0, 1], dtype="int8")
    first = select(strata, config())
    second = select(strata.copy(), config())
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:-1] < first[1:])
    assert np.bincount(strata[first], minlength=2).tolist() == [2, 1]


def test_score_gate_precedes_candidate_or_label_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reached model or labels before complete scores")
    monkeypatch.setattr(evaluator, "verify", forbidden)
    monkeypatch.setattr(evaluator, "audit_scores", forbidden)
    with pytest.raises(ValueError, match="Complete pilot score directory required"):
        evaluator.evaluate(SimpleNamespace(scores=None), {})
    scores = tmp_path / "scores"
    scores.mkdir()
    for name in evaluator.SCORE_FILES:
        (scores / name).write_text("{}")
    (scores / "complete.json").write_text(json.dumps({"complete": True, "valid_decisions": 69999, "http_groups": 10000, "public_cache_mixed": False}))
    with pytest.raises(ValueError, match="All70000"):
        evaluator.evaluate(SimpleNamespace(scores=scores), {})


def test_mixed_cache_is_rejected(tmp_path):
    for name in evaluator.SCORE_FILES:
        (tmp_path / name).write_text("{}")
    (tmp_path / "complete.json").write_text(json.dumps({"complete": True, "valid_decisions": 70000, "http_groups": 10000, "public_cache_mixed": True}))
    with pytest.raises(ValueError, match="All70000"):
        evaluator.require_scores(tmp_path)
