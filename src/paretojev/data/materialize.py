from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from paretojev.data.kuairand import PreparedKuaiRandSlates
from paretojev.utils.io import atomic_write_json


# Jev prompt names intentionally differ from the source KuaiRand label names.
# Keeping the mapping explicit prevents silent objective permutations.
DEFAULT_OBJECTIVE_ALIASES: dict[str, tuple[str, ...]] = {
    "is_click": ("is_click", "click"),
    "long_view": ("long_view",),
    "is_like": ("is_like", "like"),
    "is_follow": ("is_follow", "follow"),
    "deep_engagement": ("deep_engagement",),
    "is_hate": ("is_hate", "negative_feedback", "hate"),
}


@dataclass(frozen=True)
class KuaiRandFeatureSpec:
    """Train-fitted, leakage-safe numeric candidate feature transform."""

    log_prior_mean: float
    log_prior_std: float
    log_duration_mean: float
    log_duration_std: float
    tab_values: tuple[int, ...]
    max_tab_values: int = 16
    include_sequence_position: bool = False
    schema_version: str = "kuairand-candidate-features-v2"

    def __post_init__(self) -> None:
        if self.log_prior_std <= 0 or self.log_duration_std <= 0:
            raise ValueError("Feature standard deviations must be positive")
        if self.max_tab_values <= 0:
            raise ValueError("max_tab_values must be positive")
        if len(self.tab_values) > self.max_tab_values:
            raise ValueError("tab_values exceeds max_tab_values")
        if len(set(self.tab_values)) != len(self.tab_values):
            raise ValueError("tab_values must be unique")

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [
            "z_log1p_item_prior_count",
            "z_log1p_duration_ms",
        ]
        if self.include_sequence_position:
            names.append("sequence_position_fraction")
        names.extend(f"tab::{value}" for value in self.tab_values)
        names.append("tab::__other__")
        return tuple(names)

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["tab_values"] = list(self.tab_values)
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "KuaiRandFeatureSpec":
        import json

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["tab_values"] = tuple(int(value) for value in payload["tab_values"])
        # v1 always included pseudo-slate sequence position. Keep old specs
        # replayable while v2 defaults to a permutation-safe feature set.
        if "include_sequence_position" not in payload:
            payload["include_sequence_position"] = True
        return cls(**payload)


@dataclass
class SlateScoreTensor:
    """Fixed-width candidate probabilities aligned by slate and candidate ID.

    This is the interchange contract for behavior-model predictions. It remains
    separate from the ParetoJev training NPZ so behavior training/scoring can run
    on different hardware and be audited before fusion.
    """

    slate_ids: np.ndarray
    candidate_ids: np.ndarray
    probabilities: np.ndarray
    mask: np.ndarray
    objective_names: tuple[str, ...]

    def validate(self) -> None:
        self.slate_ids = np.asarray(self.slate_ids, dtype=np.int64)
        self.candidate_ids = np.asarray(self.candidate_ids, dtype=np.int64)
        self.probabilities = np.asarray(self.probabilities, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.slate_ids.ndim != 1:
            raise ValueError("slate_ids must have shape [N]")
        n = len(self.slate_ids)
        if self.candidate_ids.ndim != 2:
            raise ValueError("candidate_ids must have shape [N,K]")
        if self.candidate_ids.shape[0] != n:
            raise ValueError("candidate_ids and slate_ids disagree on N")
        if self.mask.shape != self.candidate_ids.shape:
            raise ValueError("mask shape mismatch")
        if n and not self.mask.any(axis=1).all():
            raise ValueError("every score row must contain at least one valid candidate")
        if self.probabilities.shape[:2] != self.candidate_ids.shape:
            raise ValueError("probabilities must have shape [N,K,M]")
        if self.probabilities.shape[2] != len(self.objective_names):
            raise ValueError("probabilities objective dimension mismatch")
        if len(np.unique(self.slate_ids)) != n:
            raise ValueError("slate_ids must be unique")
        valid = np.broadcast_to(self.mask[..., None], self.probabilities.shape)
        selected = self.probabilities[valid]
        if not np.isfinite(selected).all():
            raise ValueError("valid probabilities must be finite")
        if np.any((selected < 0.0) | (selected > 1.0)):
            raise ValueError("probabilities must lie in [0,1]")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            slate_ids=self.slate_ids,
            candidate_ids=self.candidate_ids,
            probabilities=self.probabilities,
            mask=self.mask,
            objective_names=np.asarray(self.objective_names, dtype="U64"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SlateScoreTensor":
        with np.load(path, allow_pickle=False) as data:
            required = {
                "slate_ids",
                "candidate_ids",
                "probabilities",
                "mask",
                "objective_names",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Behavior score NPZ is missing keys: {sorted(missing)}")
            result = cls(
                slate_ids=data["slate_ids"],
                candidate_ids=data["candidate_ids"],
                probabilities=data["probabilities"],
                mask=data["mask"],
                objective_names=tuple(str(value) for value in data["objective_names"]),
            )
        result.validate()
        return result


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values))
    return value if value >= 1e-6 else 1.0


def _normalize_slate_ids(
    slates: PreparedKuaiRandSlates,
    slate_ids: Iterable[int],
) -> np.ndarray:
    result = np.asarray(tuple(int(value) for value in slate_ids), dtype=np.int64)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("At least one slate_id is required")
    if len(np.unique(result)) != len(result):
        raise ValueError("slate_ids must be unique")
    if result.min() < 0 or result.max() >= len(slates.user_ids):
        raise IndexError("slate_id outside PreparedKuaiRandSlates range")
    return result


def fit_kuairand_feature_spec(
    slates: PreparedKuaiRandSlates,
    slate_ids: Iterable[int],
    *,
    max_tab_values: int = 16,
    include_sequence_position: bool = False,
) -> KuaiRandFeatureSpec:
    """Fit candidate-feature statistics on training slates only."""

    slates.validate()
    ids = _normalize_slate_ids(slates, slate_ids)
    if slates.item_prior_count is None:
        raise ValueError("item_prior_count is required for leakage-safe popularity")
    mask = slates.mask[ids]
    log_prior = np.log1p(slates.item_prior_count[ids][mask].astype(np.float64))
    log_duration = np.log1p(
        np.maximum(slates.duration_ms[ids][mask].astype(np.float64), 0.0)
    )
    tabs = slates.tab[ids][mask].astype(np.int64)
    counts = Counter(int(value) for value in tabs)
    top_tabs = tuple(
        value
        for value, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[
            :max_tab_values
        ]
    )
    return KuaiRandFeatureSpec(
        log_prior_mean=float(log_prior.mean()),
        log_prior_std=_safe_std(log_prior),
        log_duration_mean=float(log_duration.mean()),
        log_duration_std=_safe_std(log_duration),
        tab_values=top_tabs,
        max_tab_values=max_tab_values,
        include_sequence_position=include_sequence_position,
    )


def build_kuairand_candidate_features(
    slates: PreparedKuaiRandSlates,
    slate_ids: Iterable[int],
    spec: KuaiRandFeatureSpec,
) -> np.ndarray:
    """Apply a train-fitted pre-exposure feature transform."""

    ids = _normalize_slate_ids(slates, slate_ids)
    if slates.item_prior_count is None:
        raise ValueError("item_prior_count is required")
    mask = slates.mask[ids]
    n, k = mask.shape
    log_prior = np.log1p(slates.item_prior_count[ids].astype(np.float64))
    log_duration = np.log1p(
        np.maximum(slates.duration_ms[ids].astype(np.float64), 0.0)
    )
    z_prior = (log_prior - spec.log_prior_mean) / spec.log_prior_std
    z_duration = (log_duration - spec.log_duration_mean) / spec.log_duration_std

    features = [z_prior[..., None], z_duration[..., None]]
    if spec.include_sequence_position:
        valid_count = mask.sum(axis=1, keepdims=True)
        denominator = np.maximum(valid_count - 1, 1)
        position = (
            np.broadcast_to(np.arange(k, dtype=np.float64), (n, k)) / denominator
        )
        features.append(position[..., None])
    tabs = slates.tab[ids].astype(np.int64)
    known = np.zeros((n, k), dtype=bool)
    for tab in spec.tab_values:
        indicator = tabs == tab
        known |= indicator
        features.append(indicator[..., None].astype(np.float64))
    features.append((~known)[..., None].astype(np.float64))
    result = np.concatenate(features, axis=-1).astype(np.float32)
    result[~mask] = 0.0
    if result.shape[-1] != len(spec.feature_names):
        raise AssertionError("Feature names and tensor dimension diverged")
    return result


def _source_objective_index(
    source_names: Sequence[str],
    target_name: str,
    aliases: dict[str, tuple[str, ...]] = DEFAULT_OBJECTIVE_ALIASES,
) -> int:
    candidates = aliases.get(target_name, (target_name,))
    matches = [index for index, name in enumerate(source_names) if name in candidates]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source objective for {target_name!r}; "
            f"aliases={candidates}, source={list(source_names)}"
        )
    return matches[0]


def align_behavior_scores(
    slates: PreparedKuaiRandSlates,
    slate_ids: Iterable[int],
    scores: SlateScoreTensor,
) -> np.ndarray:
    """Align behavior probabilities to selected prepared slates and objectives."""

    ids = _normalize_slate_ids(slates, slate_ids)
    scores.validate()
    row_by_slate = {int(value): index for index, value in enumerate(scores.slate_ids)}
    objective_indices = [
        _source_objective_index(scores.objective_names, target)
        for target in slates.objective_names
    ]
    output = np.zeros(
        (len(ids), slates.item_ids.shape[1], len(slates.objective_names)),
        dtype=np.float32,
    )
    for output_row, slate_id in enumerate(ids):
        if int(slate_id) not in row_by_slate:
            raise KeyError(f"Behavior scores missing slate_id={int(slate_id)}")
        source_row = row_by_slate[int(slate_id)]
        expected_mask = slates.mask[slate_id]
        if not np.array_equal(scores.mask[source_row], expected_mask):
            raise ValueError(f"Behavior mask mismatch for slate_id={int(slate_id)}")
        if not np.array_equal(
            scores.candidate_ids[source_row][expected_mask],
            slates.item_ids[slate_id][expected_mask],
        ):
            raise ValueError(f"Behavior candidate order mismatch for slate_id={int(slate_id)}")
        output[output_row] = scores.probabilities[source_row][:, objective_indices]
    output[~slates.mask[ids]] = 0.0
    return output


def align_decision_score_rows(
    slates: PreparedKuaiRandSlates,
    slate_ids: Iterable[int],
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Align score-jsonl rows to selected slates with strict ID validation."""

    ids = _normalize_slate_ids(slates, slate_ids)
    row_by_slate: dict[int, dict[str, Any]] = {}
    for row in rows:
        slate_id = int(row["slate_id"])
        if slate_id in row_by_slate:
            raise ValueError(f"Duplicate decision score slate_id={slate_id}")
        row_by_slate[slate_id] = row

    shape = (len(ids), slates.item_ids.shape[1], len(slates.objective_names))
    probabilities = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    entropy = np.zeros(shape, dtype=np.float32)
    providers: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    models: Counter[str] = Counter()

    for output_row, slate_id in enumerate(ids):
        row = row_by_slate.get(int(slate_id))
        if row is None:
            raise KeyError(f"Decision scores missing slate_id={int(slate_id)}")
        expected_ids = [
            str(int(value)) for value in slates.item_ids[slate_id][slates.mask[slate_id]]
        ]
        observed_ids = [str(value) for value in row.get("candidate_ids", [])]
        if observed_ids != expected_ids:
            raise ValueError(
                f"Decision candidate order mismatch for slate_id={int(slate_id)}: "
                f"expected={expected_ids}, observed={observed_ids}"
            )
        source_names = tuple(str(value) for value in row.get("objective_names", []))
        objective_indices = [
            _source_objective_index(source_names, target)
            for target in slates.objective_names
        ]
        valid_count = len(expected_ids)
        for key, destination in (
            ("probabilities", probabilities),
            ("confidence", confidence),
            ("entropy", entropy),
        ):
            source = np.asarray(row.get(key), dtype=np.float32)
            if source.shape != (valid_count, len(source_names)):
                raise ValueError(
                    f"{key} shape mismatch for slate_id={int(slate_id)}: "
                    f"{source.shape} != {(valid_count, len(source_names))}"
                )
            selected = source[:, objective_indices]
            if not np.isfinite(selected).all() or np.any(
                (selected < 0.0) | (selected > 1.0)
            ):
                raise ValueError(f"{key} must be finite and in [0,1]")
            destination[output_row, :valid_count] = selected
        providers[str(row.get("provider", "unknown"))] += 1
        modes[str(row.get("mode", "unknown"))] += 1
        models[str(row.get("resolved_model", row.get("requested_model", "unknown")))] += 1

    report = {
        "rows": len(ids),
        "provider_counts": dict(sorted(providers.items())),
        "mode_counts": dict(sorted(modes.items())),
        "resolved_model_counts": dict(sorted(models.items())),
    }
    return probabilities, confidence, entropy, report


def selected_slate_ids_from_manifest(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if "slate_id" not in frame:
        raise ValueError("Selection manifest must contain slate_id")
    if "sample_order" in frame:
        frame = frame.sort_values("sample_order", kind="stable")
    return frame["slate_id"].to_numpy(dtype=np.int64)


def materialize_kuairand_precomputed(
    slates: PreparedKuaiRandSlates,
    slate_ids: Iterable[int],
    *,
    behavior_scores: SlateScoreTensor,
    decision_rows: Sequence[dict[str, Any]],
    feature_spec: KuaiRandFeatureSpec,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create the feature-complete ParetoJev NPZ for selected KuaiRand slates."""

    slates.validate()
    ids = _normalize_slate_ids(slates, slate_ids)
    if slates.item_prior_count is None or slates.user_history_count is None:
        raise ValueError(
            "Prepared slates need item_prior_count and user_history_count; "
            "re-run prepare-kuairand with the current pipeline"
        )
    candidate_features = build_kuairand_candidate_features(slates, ids, feature_spec)
    behavior = align_behavior_scores(slates, ids, behavior_scores)
    jev, confidence, entropy, decision_report = align_decision_score_rows(
        slates, ids, decision_rows
    )
    mask = slates.mask[ids]
    labels = slates.labels[ids].astype(np.float32, copy=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidate_features=candidate_features,
        behavior_scores=behavior,
        jev_scores=jev,
        labels=labels,
        mask=mask,
        jev_confidence=confidence,
        jev_entropy=entropy,
        popularity=slates.item_prior_count[ids].astype(np.float32),
        history_length=slates.user_history_count[ids].astype(np.float32),
        slate_ids=ids,
        candidate_ids=slates.item_ids[ids],
        objective_names=np.asarray(slates.objective_names, dtype="U64"),
        feature_names=np.asarray(feature_spec.feature_names, dtype="U96"),
    )
    return {
        "output": str(output_path),
        "slates": int(len(ids)),
        "valid_candidates": int(mask.sum()),
        "slate_width": int(mask.shape[1]),
        "objectives": list(slates.objective_names),
        "candidate_feature_dim": int(candidate_features.shape[-1]),
        "candidate_feature_names": list(feature_spec.feature_names),
        "decision_scores": decision_report,
        "behavior_score_objectives": list(behavior_scores.objective_names),
        "slate_id_min": int(ids.min()),
        "slate_id_max": int(ids.max()),
    }
