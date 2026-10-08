from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paretojev.jev.prompts import select_objectives
from paretojev.jev.runner import ScoreJsonlConfig, run_score_jsonl
from paretojev.jev.scoring import DecisionScores
from paretojev.utils.io import read_jsonl, write_jsonl


class FakeClient:
    model = "fake-1"

    def __init__(self) -> None:
        self.calls = 0
        self.states = []

    def stats(self) -> dict[str, int]:
        return {"requests": self.calls, "cache_hits": 0, "http_calls": self.calls, "retries": 0}


def fake_score(client, state, candidates, objectives):
    client.calls += 1
    client.states.append(state)
    ids = tuple(str(candidate["candidate_id"]) for candidate in candidates)
    probabilities = np.zeros((len(ids), len(objectives)), dtype=np.float32)
    probabilities[:, :] = np.linspace(0.2, 0.8, len(ids))[:, None]
    entropy = np.full_like(probabilities, 0.5)
    confidence = 1.0 - entropy
    return DecisionScores(
        candidate_ids=ids,
        objective_names=tuple(objective.name for objective in objectives),
        probabilities=probabilities,
        confidence=confidence,
        entropy=entropy,
        raw_response={"model": "fake-1", "usage": {"input_tokens": 10}},
    )


def _input_rows() -> list[dict]:
    return [
        {
            "slate_id": index,
            "history": [{"candidate_id": "h", "caption": "history"}],
            "candidates": [
                {"candidate_id": f"{index}-a", "caption": "alpha"},
                {"candidate_id": f"{index}-b", "caption": "beta"},
            ],
        }
        for index in range(3)
    ]


def test_score_runner_is_durable_resumable_and_pointwise(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "scores.jsonl"
    write_jsonl(input_path, _input_rows())
    objectives = select_objectives(["click"])

    first_client = FakeClient()
    first = run_score_jsonl(
        input_path,
        output_path,
        client=first_client,
        score_function=fake_score,
        objectives=objectives,
        config=ScoreJsonlConfig(
            provider="fake",
            mode="noul",
            objective_names=("click",),
            noul_scope="pointwise",
            resume=True,
            max_new_rows=2,
        ),
    )
    assert first["newly_scored_rows"] == 2
    assert len(read_jsonl(output_path)) == 2
    assert all("candidates" not in state for state in first_client.states)

    second_client = FakeClient()
    second = run_score_jsonl(
        input_path,
        output_path,
        client=second_client,
        score_function=fake_score,
        objectives=objectives,
        config=ScoreJsonlConfig(
            provider="fake",
            mode="noul",
            objective_names=("click",),
            noul_scope="pointwise",
            resume=True,
        ),
    )
    assert second["newly_scored_rows"] == 1
    assert second["skipped_completed_rows"] == 2
    assert len(read_jsonl(output_path)) == 3


def test_score_runner_rejects_incompatible_resume(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "scores.jsonl"
    write_jsonl(input_path, _input_rows()[:1])
    objectives = select_objectives(["click"])
    run_score_jsonl(
        input_path,
        output_path,
        client=FakeClient(),
        score_function=fake_score,
        objectives=objectives,
        config=ScoreJsonlConfig(
            provider="fake",
            mode="noul",
            objective_names=("click",),
        ),
    )
    with pytest.raises(ValueError, match="incompatible"):
        run_score_jsonl(
            input_path,
            output_path,
            client=FakeClient(),
            score_function=fake_score,
            objectives=select_objectives(["like"]),
            config=ScoreJsonlConfig(
                provider="fake",
                mode="noul",
                objective_names=("like",),
            ),
        )
