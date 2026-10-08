import numpy as np

from paretojev.jev.prompts import DEFAULT_OBJECTIVES, build_state
from paretojev.jev.scoring import score_typesafe_choice, score_typesafe_noul


class FakeTypeSafeClient:
    def evaluate(self, state, questions):
        answers = {}
        for name, question in questions.items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                values = np.arange(len(keys), 0, -1, dtype=float)
                values /= values.sum()
                answers[name] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": dict(zip(keys, values, strict=True)),
                    "confidence": 0.7,
                }
            else:
                answers[name] = {"type": "noul", "noul": 0.8}
        return {"model": "fake", "answers": answers, "usage": {}}


def test_typesafe_scoring_parsers() -> None:
    candidates = [
        {"candidate_id": "a", "caption": "mountain"},
        {"candidate_id": "b", "caption": "cat"},
    ]
    state = build_state([], candidates)
    client = FakeTypeSafeClient()
    choice = score_typesafe_choice(client, state, candidates, DEFAULT_OBJECTIVES[:2])
    assert choice.probabilities.shape == (2, 2)
    assert np.allclose(choice.probabilities.sum(axis=0), 1.0)
    noul = score_typesafe_noul(client, state, candidates, DEFAULT_OBJECTIVES[:2])
    assert noul.probabilities.shape == (2, 2)
    assert np.allclose(noul.probabilities, 0.8)


def test_pointwise_noul_state_hides_competitors_and_keeps_candidate_text() -> None:
    from paretojev.jev.prompts import build_open_jev_noul_requests, select_objectives

    candidates = [
        {"candidate_id": "a", "caption": "mountain trail"},
        {"candidate_id": "b", "caption": "cat compilation"},
    ]
    state = build_state(
        [{"candidate_id": "h", "caption": "hiking"}],
        (),
        include_candidates=False,
    )
    assert "candidates" not in state
    requests = build_open_jev_noul_requests(
        state, candidates, select_objectives(["click"])
    )
    assert len(requests) == 2
    assert "mountain trail" in requests[0][2]
    assert "cat compilation" in requests[1][2]


def test_objective_selection_is_ordered_and_validated() -> None:
    import pytest

    from paretojev.jev.prompts import select_objectives

    selected = select_objectives(["like", "click"])
    assert [objective.name for objective in selected] == ["like", "click"]
    with pytest.raises(ValueError, match="Unknown objectives"):
        select_objectives(["not_an_objective"])
