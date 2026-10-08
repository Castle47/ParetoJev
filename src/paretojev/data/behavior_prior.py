from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from paretojev.data.kuairand import PreparedKuaiRandSlates
from paretojev.data.materialize import SlateScoreTensor
from paretojev.utils.io import atomic_write_json


@dataclass(frozen=True)
class HistoricalBayesConfig:
    """Leakage-safe empirical behavior prior for pipeline bootstrapping.

    This scorer is intentionally simple and CPU-friendly. It is not presented as
    a replacement for SASRec/MMoE; it provides a reproducible behavior signal so
    real KuaiRand/Jev materialization can run before the neural behavior branch is
    trained. Every current slate is scored from events strictly earlier than that
    slate's first exposure timestamp.
    """

    alpha: float = 1.0
    beta: float = 1.0
    user_strength: float = 20.0
    item_strength: float = 20.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if self.user_strength <= 0 or self.item_strength <= 0:
            raise ValueError("shrinkage strengths must be positive")


class _HistoricalState:
    def __init__(self, n_objectives: int, config: HistoricalBayesConfig) -> None:
        self.n_objectives = n_objectives
        self.config = config
        self.global_count = 0
        self.global_sum = np.zeros(n_objectives, dtype=np.float64)
        self.user: dict[int, tuple[int, np.ndarray]] = {}
        self.item: dict[int, tuple[int, np.ndarray]] = {}

    def update(self, user_id: int, item_id: int, labels: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=np.float64)
        if labels.shape != (self.n_objectives,):
            raise ValueError("event label shape mismatch")
        self.global_count += 1
        self.global_sum += labels
        user_count, user_sum = self.user.get(
            user_id, (0, np.zeros(self.n_objectives, dtype=np.float64))
        )
        item_count, item_sum = self.item.get(
            item_id, (0, np.zeros(self.n_objectives, dtype=np.float64))
        )
        self.user[user_id] = (user_count + 1, user_sum + labels)
        self.item[item_id] = (item_count + 1, item_sum + labels)

    def score(self, user_id: int, item_id: int) -> np.ndarray:
        config = self.config
        global_probability = (
            self.global_sum + config.alpha
        ) / (self.global_count + config.alpha + config.beta)
        user_count, user_sum = self.user.get(
            user_id, (0, np.zeros(self.n_objectives, dtype=np.float64))
        )
        item_count, item_sum = self.item.get(
            item_id, (0, np.zeros(self.n_objectives, dtype=np.float64))
        )
        user_probability = (
            user_sum + config.user_strength * global_probability
        ) / (user_count + config.user_strength)
        item_probability = (
            item_sum + config.item_strength * global_probability
        ) / (item_count + config.item_strength)
        user_reliability = user_count / (user_count + config.user_strength)
        item_reliability = item_count / (item_count + config.item_strength)
        probability = (
            global_probability
            + user_reliability * user_probability
            + item_reliability * item_probability
        ) / (1.0 + user_reliability + item_reliability)
        return np.clip(probability, 1e-5, 1.0 - 1e-5)


def _valid_events(
    slates: PreparedKuaiRandSlates,
) -> dict[tuple[int, int, int], np.ndarray]:
    events: dict[tuple[int, int, int], np.ndarray] = {}
    for slate_index, user_id in enumerate(slates.user_ids):
        for candidate_index in np.flatnonzero(slates.mask[slate_index]):
            key = (
                int(user_id),
                int(slates.item_ids[slate_index, candidate_index]),
                int(slates.timestamps[slate_index, candidate_index]),
            )
            labels = slates.labels[slate_index, candidate_index].astype(
                np.float64, copy=False
            )
            previous = events.get(key)
            if previous is not None and not np.array_equal(previous, labels):
                raise ValueError(f"Conflicting duplicate event labels for {key}")
            events[key] = labels
    return events


def historical_bayes_behavior_scores(
    slates: PreparedKuaiRandSlates,
    *,
    prefix_slates: PreparedKuaiRandSlates | None = None,
    config: HistoricalBayesConfig = HistoricalBayesConfig(),
) -> tuple[SlateScoreTensor, dict[str, Any]]:
    """Score every current slate using only strictly earlier observed events."""

    slates.validate()
    if len(slates.user_ids) == 0:
        raise ValueError("slates are empty")
    if not slates.mask.any(axis=1).all():
        raise ValueError("every slate must contain at least one valid candidate")
    n_objectives = len(slates.objective_names)
    state = _HistoricalState(n_objectives, config)
    prefix_event_count = 0
    if prefix_slates is not None:
        prefix_slates.validate()
        if prefix_slates.objective_names != slates.objective_names:
            raise ValueError("prefix/current objective names differ")
        prefix_events = _valid_events(prefix_slates)
        current_start = int(
            np.min(np.where(slates.mask, slates.timestamps, np.iinfo(np.int64).max))
        )
        if prefix_events:
            prefix_max = max(key[2] for key in prefix_events)
            if prefix_max >= current_start:
                raise ValueError(
                    "prefix_slates must end strictly before current slate timestamps"
                )
        for key in sorted(prefix_events, key=lambda value: (value[2], value[0], value[1])):
            user_id, item_id, _ = key
            state.update(user_id, item_id, prefix_events[key])
        prefix_event_count = len(prefix_events)

    n, k = slates.item_ids.shape
    probabilities = np.zeros((n, k, n_objectives), dtype=np.float32)
    start_times = np.where(
        slates.mask,
        slates.timestamps,
        np.iinfo(np.int64).max,
    ).min(axis=1)
    order = np.argsort(start_times, kind="stable")

    # Pending events are flushed only when event_time < next_slate_start. This
    # freezes all candidates in a pseudo-slate to the same pre-slate history and
    # prevents early outcomes in the slate from leaking into later candidates.
    pending: list[tuple[int, int, tuple[int, int, int], np.ndarray]] = []
    queued: dict[tuple[int, int, int], np.ndarray] = {}
    applied: set[tuple[int, int, int]] = set()
    sequence = 0

    for slate_index in order:
        slate_start = int(start_times[slate_index])
        while pending and pending[0][0] < slate_start:
            _, _, key, labels = heapq.heappop(pending)
            if key in applied:
                continue
            user_id, item_id, _ = key
            state.update(user_id, item_id, labels)
            applied.add(key)

        user_id = int(slates.user_ids[slate_index])
        for candidate_index in np.flatnonzero(slates.mask[slate_index]):
            item_id = int(slates.item_ids[slate_index, candidate_index])
            probabilities[slate_index, candidate_index] = state.score(user_id, item_id)

        for candidate_index in np.flatnonzero(slates.mask[slate_index]):
            item_id = int(slates.item_ids[slate_index, candidate_index])
            timestamp = int(slates.timestamps[slate_index, candidate_index])
            key = (user_id, item_id, timestamp)
            labels = slates.labels[slate_index, candidate_index].astype(
                np.float64, copy=False
            )
            previous = queued.get(key)
            if previous is not None:
                if not np.array_equal(previous, labels):
                    raise ValueError(f"Conflicting duplicate event labels for {key}")
                continue
            queued[key] = labels
            heapq.heappush(pending, (timestamp, sequence, key, labels))
            sequence += 1

    tensor = SlateScoreTensor(
        slate_ids=np.arange(n, dtype=np.int64),
        candidate_ids=slates.item_ids.copy(),
        probabilities=probabilities,
        mask=slates.mask.copy(),
        objective_names=slates.objective_names,
    )
    tensor.validate()
    valid = np.broadcast_to(slates.mask[..., None], slates.labels.shape)
    squared = (probabilities - slates.labels.astype(np.float32)) ** 2
    per_objective_brier = {
        name: float(squared[..., index][slates.mask].mean())
        for index, name in enumerate(slates.objective_names)
    }
    report = {
        "schema_version": "historical-bayes-behavior-v1",
        "config": asdict(config),
        "slates": int(n),
        "valid_candidates": int(slates.mask.sum()),
        "prefix_unique_events": int(prefix_event_count),
        "current_unique_events": int(len(queued)),
        "applied_current_events_during_scoring": int(len(applied)),
        "global_history_events_at_end_of_scoring": int(state.global_count),
        "pending_current_events_not_needed_for_scoring": int(len(queued) - len(applied)),
        "per_objective_brier": per_objective_brier,
        "overall_brier": float(squared[valid].mean()),
        "history_source": (
            "Only unique logged events represented by the prepared pseudo-slates "
            "are available to this bootstrap scorer."
        ),
        "brier_interpretation": (
            "Prequential diagnostic on current slates, not an independent held-out "
            "estimate and not a research baseline claim."
        ),
        "limitation": (
            "CPU bootstrap prior only; final experiments must replace or compare "
            "it with the SASRec+MMoE behavior branch."
        ),
    }
    return tensor, report


def score_and_save_historical_bayes(
    slates_path: str | Path,
    output_path: str | Path,
    *,
    prefix_slates_path: str | Path | None = None,
    config: HistoricalBayesConfig = HistoricalBayesConfig(),
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    slates = PreparedKuaiRandSlates.load(slates_path)
    prefix = (
        PreparedKuaiRandSlates.load(prefix_slates_path)
        if prefix_slates_path is not None
        else None
    )
    tensor, report = historical_bayes_behavior_scores(
        slates,
        prefix_slates=prefix,
        config=config,
    )
    tensor.save(output_path)
    report["output"] = str(output_path)
    if report_path is not None:
        atomic_write_json(report_path, report)
    return report
