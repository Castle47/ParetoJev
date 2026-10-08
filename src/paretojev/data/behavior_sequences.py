from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from paretojev.data.kuairand import KUAI_OBJECTIVES, PreparedKuaiRandSlates


@dataclass(frozen=True)
class ItemVocabulary:
    """Train-fitted mapping from raw item IDs to embedding indices.

    Index 0 is padding and index 1 is OOV. All remaining IDs are sorted so the
    mapping is deterministic and can be serialized compactly in a checkpoint.
    """

    raw_item_ids: tuple[int, ...]
    padding_idx: int = 0
    unknown_idx: int = 1

    def __post_init__(self) -> None:
        if self.padding_idx != 0 or self.unknown_idx != 1:
            raise ValueError("ItemVocabulary currently reserves padding=0 and OOV=1")
        if tuple(sorted(set(self.raw_item_ids))) != self.raw_item_ids:
            raise ValueError("raw_item_ids must be sorted and unique")

    @property
    def num_items(self) -> int:
        return len(self.raw_item_ids) + 2

    @classmethod
    def fit(cls, item_ids: Sequence[int] | np.ndarray) -> "ItemVocabulary":
        values = np.asarray(item_ids, dtype=np.int64)
        return cls(tuple(int(value) for value in np.unique(values)))

    def encode(
        self,
        values: Sequence[int] | np.ndarray,
        *,
        padding_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        raw = np.asarray(self.raw_item_ids, dtype=np.int64)
        values_array = np.asarray(values, dtype=np.int64)
        result = np.full(values_array.shape, self.unknown_idx, dtype=np.int64)
        if len(raw):
            positions = np.searchsorted(raw, values_array)
            in_range = positions < len(raw)
            matched = np.zeros(values_array.shape, dtype=bool)
            matched[in_range] = raw[positions[in_range]] == values_array[in_range]
            result[matched] = positions[matched] + 2
        if padding_mask is not None:
            mask = np.asarray(padding_mask, dtype=bool)
            if mask.shape != result.shape:
                raise ValueError("padding_mask shape mismatch")
            result[mask] = self.padding_idx
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_item_ids": list(self.raw_item_ids),
            "padding_idx": self.padding_idx,
            "unknown_idx": self.unknown_idx,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ItemVocabulary":
        return cls(
            raw_item_ids=tuple(int(value) for value in payload["raw_item_ids"]),
            padding_idx=int(payload.get("padding_idx", 0)),
            unknown_idx=int(payload.get("unknown_idx", 1)),
        )


@dataclass(frozen=True)
class BehaviorContextSpec:
    """Train-fitted transform for pre-exposure numeric behavior features."""

    log_prior_mean: float
    log_prior_std: float
    log_duration_mean: float
    log_duration_std: float
    log_history_mean: float
    log_history_std: float
    tab_values: tuple[int, ...]
    max_tab_values: int = 16
    schema_version: str = "kuairand-behavior-context-v1"

    def __post_init__(self) -> None:
        for name in ("log_prior_std", "log_duration_std", "log_history_std"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_tab_values <= 0:
            raise ValueError("max_tab_values must be positive")
        if len(self.tab_values) > self.max_tab_values:
            raise ValueError("tab_values exceeds max_tab_values")
        if len(set(self.tab_values)) != len(self.tab_values):
            raise ValueError("tab_values must be unique")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "z_log1p_item_prior_count",
            "z_log1p_duration_ms",
            "z_log1p_user_history_count",
            *(f"tab::{value}" for value in self.tab_values),
            "tab::__other__",
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def transform(
        self,
        *,
        item_prior_count: np.ndarray,
        duration_ms: np.ndarray,
        user_history_count: np.ndarray,
        tab: np.ndarray,
    ) -> np.ndarray:
        prior = np.asarray(item_prior_count, dtype=np.float64)
        duration = np.asarray(duration_ms, dtype=np.float64)
        history = np.asarray(user_history_count, dtype=np.float64)
        tabs = np.asarray(tab, dtype=np.int64)
        if not (prior.shape == duration.shape == history.shape == tabs.shape):
            raise ValueError("Behavior context inputs must have identical shapes")
        features: list[np.ndarray] = [
            ((np.log1p(np.maximum(prior, 0.0)) - self.log_prior_mean) / self.log_prior_std)[
                ..., None
            ],
            (
                (np.log1p(np.maximum(duration, 0.0)) - self.log_duration_mean)
                / self.log_duration_std
            )[..., None],
            (
                (np.log1p(np.maximum(history, 0.0)) - self.log_history_mean)
                / self.log_history_std
            )[..., None],
        ]
        known = np.zeros(tabs.shape, dtype=bool)
        for value in self.tab_values:
            indicator = tabs == value
            known |= indicator
            features.append(indicator[..., None].astype(np.float64))
        features.append((~known)[..., None].astype(np.float64))
        return np.concatenate(features, axis=-1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tab_values"] = list(self.tab_values)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BehaviorContextSpec":
        values = dict(payload)
        values["tab_values"] = tuple(int(value) for value in values["tab_values"])
        return cls(**values)


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values))
    return value if value >= 1e-6 else 1.0


def _ensure_objectives(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "deep_engagement" not in result:
        missing = {"is_comment", "is_forward"} - set(result.columns)
        if missing:
            raise ValueError(f"Behavior log is missing columns: {sorted(missing)}")
        result["deep_engagement"] = result[["is_comment", "is_forward"]].max(axis=1)
    required = {"user_id", "video_id", "time_ms", "duration_ms", "tab"} | set(
        KUAI_OBJECTIVES
    )
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Behavior log is missing columns: {sorted(missing)}")
    return result


@dataclass
class BehaviorEventTable:
    """Chronological events with target rows and strict pre-event history indices."""

    raw_item_ids: np.ndarray
    user_ids: np.ndarray
    timestamps: np.ndarray
    user_starts: np.ndarray
    history_ends: np.ndarray
    target_positions: np.ndarray
    target_labels: np.ndarray
    target_item_prior_count: np.ndarray
    target_user_history_count: np.ndarray
    target_duration_ms: np.ndarray
    target_tab: np.ndarray
    objective_names: tuple[str, ...] = KUAI_OBJECTIVES

    def validate(self) -> None:
        n = len(self.raw_item_ids)
        for name in ("user_ids", "timestamps", "user_starts", "history_ends"):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} shape mismatch")
        t = len(self.target_positions)
        if self.target_labels.shape != (t, len(self.objective_names)):
            raise ValueError("target_labels shape mismatch")
        if not np.isfinite(self.target_labels).all() or np.any(
            (self.target_labels < 0.0) | (self.target_labels > 1.0)
        ):
            raise ValueError("target_labels must be finite binary/probability values")
        for name in (
            "target_item_prior_count",
            "target_user_history_count",
            "target_duration_ms",
            "target_tab",
        ):
            if np.asarray(getattr(self, name)).shape != (t,):
                raise ValueError(f"{name} shape mismatch")
        if t and (self.target_positions.min() < 0 or self.target_positions.max() >= n):
            raise ValueError("target_positions outside event table")
        if np.any(self.user_starts > self.history_ends):
            raise ValueError("user_starts cannot exceed history_ends")


def build_behavior_event_table(
    current_log: pd.DataFrame,
    *,
    prefix_log: pd.DataFrame | None = None,
) -> BehaviorEventTable:
    """Prepare targets while keeping all histories strictly before target time."""

    current = _ensure_objectives(current_log)
    if len(current) == 0:
        raise ValueError("current_log is empty")
    parts: list[pd.DataFrame] = []
    if prefix_log is not None and len(prefix_log):
        prefix = _ensure_objectives(prefix_log)
        if int(prefix["time_ms"].max()) >= int(current["time_ms"].min()):
            raise ValueError("prefix_log must end strictly before current_log")
        prefix = prefix.copy()
        prefix["_target"] = False
        prefix["_target_order"] = -1
        parts.append(prefix)
    current = current.copy()
    current["_target"] = True
    current["_target_order"] = np.arange(len(current), dtype=np.int64)
    parts.append(current)
    combined = pd.concat(parts, ignore_index=True)
    combined["_event_id"] = np.arange(len(combined), dtype=np.int64)

    global_order = combined.sort_values(["time_ms", "_event_id"], kind="stable").copy()
    global_order["_item_prior_count"] = (
        global_order.groupby("video_id", sort=False).cumcount()
        - global_order.groupby(["video_id", "time_ms"], sort=False).cumcount()
    )
    item_prior = global_order.set_index("_event_id")["_item_prior_count"]
    combined["_item_prior_count"] = combined["_event_id"].map(item_prior).astype(np.int64)

    ordered = combined.sort_values(
        ["user_id", "time_ms", "_event_id"], kind="stable"
    ).reset_index(drop=True)
    user_rank = ordered.groupby("user_id", sort=False).cumcount().to_numpy(dtype=np.int64)
    same_time_rank = ordered.groupby(["user_id", "time_ms"], sort=False).cumcount().to_numpy(
        dtype=np.int64
    )
    positions = np.arange(len(ordered), dtype=np.int64)
    user_starts = positions - user_rank
    history_ends = positions - same_time_rank
    ordered["_user_prior_count"] = user_rank - same_time_rank

    target_frame = ordered.loc[ordered["_target"]].copy()
    # Keep targets in chronological user order; random-access training can shuffle
    # without carrying the original CSV row order into the model.
    target_positions = target_frame.index.to_numpy(dtype=np.int64)
    table = BehaviorEventTable(
        raw_item_ids=ordered["video_id"].to_numpy(dtype=np.int64),
        user_ids=ordered["user_id"].to_numpy(dtype=np.int64),
        timestamps=ordered["time_ms"].to_numpy(dtype=np.int64),
        user_starts=user_starts,
        history_ends=history_ends,
        target_positions=target_positions,
        target_labels=target_frame[list(KUAI_OBJECTIVES)].to_numpy(dtype=np.float32),
        target_item_prior_count=target_frame["_item_prior_count"].to_numpy(dtype=np.int64),
        target_user_history_count=target_frame["_user_prior_count"].to_numpy(dtype=np.int64),
        target_duration_ms=target_frame["duration_ms"].to_numpy(dtype=np.float32),
        target_tab=target_frame["tab"].to_numpy(dtype=np.int64),
    )
    table.validate()
    return table


def fit_behavior_context_spec(
    table: BehaviorEventTable,
    *,
    max_tab_values: int = 16,
) -> BehaviorContextSpec:
    table.validate()
    tabs, counts = np.unique(table.target_tab, return_counts=True)
    order = np.lexsort((tabs, -counts))
    top_tabs = tuple(int(value) for value in tabs[order][:max_tab_values])
    log_prior = np.log1p(table.target_item_prior_count.astype(np.float64))
    log_duration = np.log1p(np.maximum(table.target_duration_ms.astype(np.float64), 0.0))
    log_history = np.log1p(table.target_user_history_count.astype(np.float64))
    return BehaviorContextSpec(
        log_prior_mean=float(log_prior.mean()),
        log_prior_std=_safe_std(log_prior),
        log_duration_mean=float(log_duration.mean()),
        log_duration_std=_safe_std(log_duration),
        log_history_mean=float(log_history.mean()),
        log_history_std=_safe_std(log_history),
        tab_values=top_tabs,
        max_tab_values=max_tab_values,
    )


class KuaiRandBehaviorDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-efficient event dataset with histories sliced on demand."""

    def __init__(
        self,
        table: BehaviorEventTable,
        vocabulary: ItemVocabulary,
        context_spec: BehaviorContextSpec,
        *,
        max_history_length: int = 50,
    ) -> None:
        table.validate()
        if max_history_length <= 0:
            raise ValueError("max_history_length must be positive")
        self.table = table
        self.vocabulary = vocabulary
        self.context_spec = context_spec
        self.max_history_length = max_history_length
        self.mapped_items = vocabulary.encode(table.raw_item_ids)
        self.context = context_spec.transform(
            item_prior_count=table.target_item_prior_count,
            duration_ms=table.target_duration_ms,
            user_history_count=table.target_user_history_count,
            tab=table.target_tab,
        )

    def __len__(self) -> int:
        return len(self.table.target_positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        position = int(self.table.target_positions[index])
        history_end = int(self.table.history_ends[position])
        history_start = max(
            int(self.table.user_starts[position]), history_end - self.max_history_length
        )
        history_values = self.mapped_items[history_start:history_end]
        history = np.zeros(self.max_history_length, dtype=np.int64)
        if len(history_values):
            history[-len(history_values) :] = history_values
        return {
            "history_item_ids": torch.from_numpy(history),
            "candidate_item_id": torch.tensor(
                int(self.mapped_items[position]), dtype=torch.long
            ),
            "candidate_context": torch.from_numpy(self.context[index]),
            "labels": torch.from_numpy(self.table.target_labels[index].copy()),
        }


class UserHistoryIndex:
    """Per-user time index used to score prepared slates without future items."""

    def __init__(
        self,
        current_log: pd.DataFrame,
        vocabulary: ItemVocabulary,
        *,
        prefix_log: pd.DataFrame | None = None,
    ) -> None:
        current = _ensure_objectives(current_log)
        parts: list[pd.DataFrame] = []
        if prefix_log is not None and len(prefix_log):
            prefix = _ensure_objectives(prefix_log)
            if int(prefix["time_ms"].max()) >= int(current["time_ms"].min()):
                raise ValueError("prefix_log must end strictly before current_log")
            parts.append(prefix[["user_id", "video_id", "time_ms"]])
        parts.append(current[["user_id", "video_id", "time_ms"]])
        frame = pd.concat(parts, ignore_index=True).sort_values(
            ["user_id", "time_ms"], kind="stable"
        )
        self._users: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for user_id, group in frame.groupby("user_id", sort=False):
            times = group["time_ms"].to_numpy(dtype=np.int64)
            items = vocabulary.encode(group["video_id"].to_numpy(dtype=np.int64))
            self._users[int(user_id)] = (times, items)

    def history(
        self,
        user_id: int,
        timestamp: int,
        *,
        max_history_length: int,
    ) -> np.ndarray:
        result = np.zeros(max_history_length, dtype=np.int64)
        values = self._users.get(int(user_id))
        if values is None:
            return result
        times, items = values
        end = int(np.searchsorted(times, int(timestamp), side="left"))
        start = max(0, end - max_history_length)
        selected = items[start:end]
        if len(selected):
            result[-len(selected) :] = selected
        return result


def behavior_context_for_slates(
    slates: PreparedKuaiRandSlates,
    spec: BehaviorContextSpec,
) -> np.ndarray:
    slates.validate()
    if slates.item_prior_count is None or slates.user_history_count is None:
        raise ValueError("Prepared slates need strict item/user prior counts")
    history = np.broadcast_to(slates.user_history_count[:, None], slates.item_ids.shape)
    context = spec.transform(
        item_prior_count=slates.item_prior_count,
        duration_ms=slates.duration_ms,
        user_history_count=history,
        tab=slates.tab,
    )
    context[~slates.mask] = 0.0
    return context
