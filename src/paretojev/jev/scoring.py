from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .client import OpenJevClient, TypeSafeJevClient
from .prompts import (
    DEFAULT_OBJECTIVES,
    DecisionObjective,
    build_open_jev_choice_requests,
    build_open_jev_noul_requests,
    build_typesafe_choice_questions,
    build_typesafe_noul_questions,
)


@dataclass
class DecisionScores:
    """Candidate-by-objective probabilities and reliability metadata."""

    candidate_ids: tuple[str, ...]
    objective_names: tuple[str, ...]
    probabilities: np.ndarray  # [K,M]
    confidence: np.ndarray  # [K,M]
    entropy: np.ndarray  # [K,M], normalized to [0,1]
    raw_response: Any

    def validate(self) -> None:
        expected = (len(self.candidate_ids), len(self.objective_names))
        for name in ("probabilities", "confidence", "entropy"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}")
        if np.any((self.probabilities < 0) | (self.probabilities > 1)):
            raise ValueError("probabilities must lie in [0,1]")


def _candidate_ids(candidates: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(item.get("candidate_id", item.get("item_id", item.get("video_id", index))))
        for index, item in enumerate(candidates)
    )


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _categorical_entropy(probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    denominator = math.log2(len(p)) if len(p) > 1 else 1.0
    return float(-(p * np.log2(p)).sum() / denominator)


def score_typesafe_choice(
    client: TypeSafeJevClient,
    state: Any,
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> DecisionScores:
    objectives = tuple(objectives)
    ids = _candidate_ids(candidates)
    response = client.evaluate(state, build_typesafe_choice_questions(candidates, objectives))
    answers = response.get("answers", {})
    probabilities = np.zeros((len(ids), len(objectives)), dtype=np.float32)
    confidence = np.zeros_like(probabilities)
    entropy = np.zeros_like(probabilities)
    for objective_index, objective in enumerate(objectives):
        answer = answers.get(objective.name)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"Malformed Choice answer for {objective.name}: {answer}")
        distribution = answer.get("probabilities", {})
        column = np.asarray([float(distribution.get(candidate_id, 0.0)) for candidate_id in ids])
        total = float(column.sum())
        if total <= 0:
            raise ValueError(f"Choice probabilities sum to zero for {objective.name}")
        column /= total
        probabilities[:, objective_index] = column
        confidence[:, objective_index] = float(answer.get("confidence", 1.0 - _categorical_entropy(column)))
        entropy[:, objective_index] = _categorical_entropy(column)
    result = DecisionScores(
        candidate_ids=ids,
        objective_names=tuple(x.name for x in objectives),
        probabilities=probabilities,
        confidence=confidence,
        entropy=entropy,
        raw_response=response,
    )
    result.validate()
    return result


def score_typesafe_noul(
    client: TypeSafeJevClient,
    state: Any,
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> DecisionScores:
    objectives = tuple(objectives)
    ids = _candidate_ids(candidates)
    response = client.evaluate(state, build_typesafe_noul_questions(candidates, objectives))
    answers = response.get("answers", {})
    probabilities = np.zeros((len(ids), len(objectives)), dtype=np.float32)
    for candidate_index, candidate_id in enumerate(ids):
        for objective_index, objective in enumerate(objectives):
            key = f"{objective.name}__{candidate_id}"
            answer = answers.get(key)
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise ValueError(f"Malformed Noul answer for {key}: {answer}")
            probabilities[candidate_index, objective_index] = float(answer["noul"])
    entropy = _binary_entropy(probabilities).astype(np.float32)
    # Noul does not expose a separate official confidence field. This is an
    # explicit concentration proxy, not TypeSafe's Choice confidence.
    confidence = (1.0 - entropy).astype(np.float32)
    result = DecisionScores(
        candidate_ids=ids,
        objective_names=tuple(x.name for x in objectives),
        probabilities=probabilities,
        confidence=confidence,
        entropy=entropy,
        raw_response=response,
    )
    result.validate()
    return result


def score_open_choice(
    client: OpenJevClient,
    state: Any,
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> DecisionScores:
    objectives = tuple(objectives)
    ids = _candidate_ids(candidates)
    requests = build_open_jev_choice_requests(state, candidates, objectives)
    responses = client.decide_many(requests)
    probabilities = np.zeros((len(ids), len(objectives)), dtype=np.float32)
    confidence = np.zeros_like(probabilities)
    entropy = np.zeros_like(probabilities)
    for objective_index, response in enumerate(responses):
        column = np.asarray([float(response.get(candidate_id, 0.0)) for candidate_id in ids])
        total = float(column.sum())
        if total <= 0:
            raise ValueError("Open Jev Choice probabilities sum to zero")
        column /= total
        probabilities[:, objective_index] = column
        h = _categorical_entropy(column)
        confidence[:, objective_index] = 1.0 - h
        entropy[:, objective_index] = h
    result = DecisionScores(
        candidate_ids=ids,
        objective_names=tuple(x.name for x in objectives),
        probabilities=probabilities,
        confidence=confidence,
        entropy=entropy,
        raw_response=responses,
    )
    result.validate()
    return result


def score_open_noul(
    client: OpenJevClient,
    state: Any,
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> DecisionScores:
    objectives = tuple(objectives)
    ids = _candidate_ids(candidates)
    requests = build_open_jev_noul_requests(state, candidates, objectives)
    responses = client.decide_many(requests)
    probabilities = np.zeros((len(ids), len(objectives)), dtype=np.float32)
    cursor = 0
    for candidate_index in range(len(ids)):
        for objective_index in range(len(objectives)):
            response = responses[cursor]
            cursor += 1
            # Community JEV servers typically return {"true": p, "false": 1-p}.
            true_probability = response.get("true", response.get("True"))
            if true_probability is None:
                raise ValueError(f"Open Jev Noul response lacks a true option: {response}")
            probabilities[candidate_index, objective_index] = float(true_probability)
    entropy = _binary_entropy(probabilities).astype(np.float32)
    confidence = (1.0 - entropy).astype(np.float32)
    result = DecisionScores(
        candidate_ids=ids,
        objective_names=tuple(x.name for x in objectives),
        probabilities=probabilities,
        confidence=confidence,
        entropy=entropy,
        raw_response=responses,
    )
    result.validate()
    return result
