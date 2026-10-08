from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

KUAI_OBJECTIVES = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "deep_engagement",
    "is_hate",
)
KUAI_DIRECTIONS = (1, 1, 1, 1, 1, -1)

REQUIRED_LOG_COLUMNS = {
    "user_id",
    "video_id",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "is_rand",
    "tab",
}


@dataclass(frozen=True)
class KuaiRandSlateConfig:
    slate_size: int = 20
    min_slate_size: int = 5
    session_gap_minutes: int = 30
    stride: int | None = None
    objective_columns: tuple[str, ...] = KUAI_OBJECTIVES

    def __post_init__(self) -> None:
        if self.slate_size < 2:
            raise ValueError("slate_size must be at least 2")
        if not 2 <= self.min_slate_size <= self.slate_size:
            raise ValueError("min_slate_size must lie in [2, slate_size]")
        if self.session_gap_minutes <= 0:
            raise ValueError("session_gap_minutes must be positive")


@dataclass
class PreparedKuaiRandSlates:
    user_ids: np.ndarray
    item_ids: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    duration_ms: np.ndarray
    tab: np.ndarray
    is_random: np.ndarray
    watch_ratio: np.ndarray
    objective_names: tuple[str, ...]
    item_prior_count: np.ndarray | None = None
    user_history_count: np.ndarray | None = None

    def validate(self) -> None:
        n, k = self.item_ids.shape
        if self.user_ids.shape != (n,):
            raise ValueError("user_ids shape mismatch")
        for name in ("timestamps", "mask", "duration_ms", "tab", "is_random", "watch_ratio"):
            if getattr(self, name).shape != (n, k):
                raise ValueError(f"{name} shape mismatch")
        if self.labels.shape != (n, k, len(self.objective_names)):
            raise ValueError("labels shape mismatch")
        if self.item_prior_count is not None and self.item_prior_count.shape != (n, k):
            raise ValueError("item_prior_count shape mismatch")
        if self.user_history_count is not None and self.user_history_count.shape != (n,):
            raise ValueError("user_history_count shape mismatch")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "timestamps": self.timestamps,
            "labels": self.labels,
            "mask": self.mask,
            "duration_ms": self.duration_ms,
            "tab": self.tab,
            "is_random": self.is_random,
            "watch_ratio": self.watch_ratio,
            "objective_names": np.asarray(self.objective_names, dtype="U64"),
        }
        if self.item_prior_count is not None:
            payload["item_prior_count"] = self.item_prior_count
        if self.user_history_count is not None:
            payload["user_history_count"] = self.user_history_count
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "PreparedKuaiRandSlates":
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                user_ids=data["user_ids"],
                item_ids=data["item_ids"],
                timestamps=data["timestamps"],
                labels=data["labels"],
                mask=data["mask"],
                duration_ms=data["duration_ms"],
                tab=data["tab"],
                is_random=data["is_random"],
                watch_ratio=data["watch_ratio"],
                objective_names=tuple(str(x) for x in data["objective_names"]),
                item_prior_count=(
                    data["item_prior_count"] if "item_prior_count" in data.files else None
                ),
                user_history_count=(
                    data["user_history_count"]
                    if "user_history_count" in data.files
                    else None
                ),
            )
        result.validate()
        return result


@dataclass(frozen=True)
class KuaiRandSamplingConfig:
    """Balanced sampling policy for expensive decision-model annotation."""

    max_slates: int = 5_000
    per_cell: int = 250
    max_per_user: int | None = 2
    seed: int = 0
    cold_quantile: float = 0.25
    warm_quantile: float = 0.75
    positive_rich_threshold: int = 2

    def __post_init__(self) -> None:
        if self.max_slates <= 0:
            raise ValueError("max_slates must be positive")
        if self.per_cell <= 0:
            raise ValueError("per_cell must be positive")
        if self.max_per_user is not None and self.max_per_user <= 0:
            raise ValueError("max_per_user must be positive or None")
        if not 0.0 < self.cold_quantile < self.warm_quantile < 1.0:
            raise ValueError("Require 0 < cold_quantile < warm_quantile < 1")
        if self.positive_rich_threshold < 2:
            raise ValueError("positive_rich_threshold must be at least 2")


def find_kuairand_file(directory: str | Path, stem: str, variant: str = "pure") -> Path:
    directory = Path(directory)
    candidates = [
        directory / f"{stem}_{variant}.csv",
        directory / "data" / f"{stem}_{variant}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {stem}_{variant}.csv under {directory} or {directory / 'data'}"
    )


def read_kuairand_log(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=nrows)
    missing = REQUIRED_LOG_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"KuaiRand log is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["deep_engagement"] = frame[["is_comment", "is_forward"]].max(axis=1)
    duration = frame["duration_ms"].to_numpy(dtype=np.float64)
    play = frame["play_time_ms"].to_numpy(dtype=np.float64)
    frame["watch_ratio"] = np.divide(
        play,
        np.maximum(duration, 1.0),
        out=np.zeros_like(play, dtype=np.float64),
    ).clip(0.0, 5.0)
    return frame.sort_values(["user_id", "time_ms"], kind="stable")


def load_content_metadata(directory: str | Path) -> pd.DataFrame:
    """Load optional caption/category files and join on `final_video_id`."""

    directory = Path(directory)
    captions_path = directory / "kuairand_video_captions.csv"
    categories_path = directory / "kuairand_video_categories.csv"
    if not captions_path.exists() and not categories_path.exists():
        raise FileNotFoundError(
            "Expected kuairand_video_captions.csv and/or kuairand_video_categories.csv"
        )
    frames: list[pd.DataFrame] = []
    if captions_path.exists():
        frames.append(pd.read_csv(captions_path))
    if categories_path.exists():
        frames.append(pd.read_csv(categories_path))
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="final_video_id", how="outer", validate="one_to_one")
    return result


def _pad(values: Sequence[object], length: int, fill: object, dtype: np.dtype) -> np.ndarray:
    out = np.full(length, fill, dtype=dtype)
    out[: len(values)] = np.asarray(values, dtype=dtype)
    return out


def build_session_slates(
    log: pd.DataFrame,
    config: KuaiRandSlateConfig = KuaiRandSlateConfig(),
    *,
    prefix_log: pd.DataFrame | None = None,
) -> PreparedKuaiRandSlates:
    """Create exposed-session pseudo-slates from chronological KuaiRand logs.

    KuaiRand does not expose the platform's complete request candidate set. This
    function therefore groups consecutive *exposures* within a user session and
    chunks them into fixed-width pseudo-slates. Results must be reported under
    that name and not described as native request-level reranking.
    """

    missing = REQUIRED_LOG_COLUMNS - set(log.columns)
    if missing:
        raise ValueError(f"KuaiRand log is missing columns: {sorted(missing)}")
    item_base: pd.Series | None = None
    user_base: pd.Series | None = None
    if prefix_log is not None and len(prefix_log):
        prefix_missing = {"user_id", "video_id", "time_ms"} - set(prefix_log.columns)
        if prefix_missing:
            raise ValueError(
                f"KuaiRand prefix log is missing columns: {sorted(prefix_missing)}"
            )
        if int(prefix_log["time_ms"].max()) >= int(log["time_ms"].min()):
            raise ValueError(
                "prefix_log must end strictly before the first timestamp in log"
            )
        item_base = prefix_log.groupby("video_id", sort=False).size()
        user_base = prefix_log.groupby("user_id", sort=False).size()

    # Prefix counts are computed in global chronological order before switching
    # to user/session order. They are safe cold-start/history descriptors because
    # they exclude the current exposure and every future exposure.
    frame = log.copy().sort_values("time_ms", kind="stable")
    frame["_item_prior_count"] = frame.groupby("video_id", sort=False).cumcount()
    if item_base is not None:
        frame["_item_prior_count"] += (
            frame["video_id"].map(item_base).fillna(0).astype(np.int64)
        )
    frame = frame.sort_values(["user_id", "time_ms"], kind="stable")
    frame["_user_prior_count"] = frame.groupby("user_id", sort=False).cumcount()
    if user_base is not None:
        frame["_user_prior_count"] += (
            frame["user_id"].map(user_base).fillna(0).astype(np.int64)
        )
    if "deep_engagement" not in frame:
        frame["deep_engagement"] = frame[["is_comment", "is_forward"]].max(axis=1)
    if "watch_ratio" not in frame:
        duration = frame["duration_ms"].to_numpy(dtype=np.float64)
        play = frame["play_time_ms"].to_numpy(dtype=np.float64)
        frame["watch_ratio"] = np.divide(play, np.maximum(duration, 1.0)).clip(0.0, 5.0)

    gap_ms = config.session_gap_minutes * 60_000
    user_change = frame["user_id"].ne(frame["user_id"].shift())
    time_gap = frame["time_ms"].sub(frame["time_ms"].shift()).gt(gap_ms)
    frame["_session"] = (user_change | time_gap).cumsum()
    stride = config.stride or config.slate_size

    user_ids: list[int] = []
    item_ids: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    durations: list[np.ndarray] = []
    tabs: list[np.ndarray] = []
    random_flags: list[np.ndarray] = []
    watch_ratios: list[np.ndarray] = []
    item_prior_counts: list[np.ndarray] = []
    user_history_counts: list[int] = []

    for (_, _), session in frame.groupby(["user_id", "_session"], sort=False):
        session = session.sort_values("time_ms", kind="stable")
        for start in range(0, len(session), stride):
            chunk = session.iloc[start : start + config.slate_size]
            if len(chunk) < config.min_slate_size:
                continue
            valid = len(chunk)
            user_ids.append(int(chunk["user_id"].iloc[0]))
            item_ids.append(_pad(chunk["video_id"], config.slate_size, -1, np.int64))
            timestamps.append(_pad(chunk["time_ms"], config.slate_size, 0, np.int64))
            label_matrix = np.zeros((config.slate_size, len(config.objective_columns)), dtype=np.float32)
            label_matrix[:valid] = chunk[list(config.objective_columns)].to_numpy(dtype=np.float32)
            labels.append(label_matrix)
            mask = np.zeros(config.slate_size, dtype=bool)
            mask[:valid] = True
            masks.append(mask)
            durations.append(_pad(chunk["duration_ms"], config.slate_size, 0.0, np.float32))
            tabs.append(_pad(chunk["tab"], config.slate_size, -1, np.int16))
            random_flags.append(_pad(chunk["is_rand"], config.slate_size, 0, np.int8))
            watch_ratios.append(_pad(chunk["watch_ratio"], config.slate_size, 0.0, np.float32))
            item_prior_counts.append(
                _pad(chunk["_item_prior_count"], config.slate_size, 0, np.int64)
            )
            user_history_counts.append(int(chunk["_user_prior_count"].iloc[0]))

    if not user_ids:
        raise ValueError("No slates were produced; lower min_slate_size or inspect the input")
    result = PreparedKuaiRandSlates(
        user_ids=np.asarray(user_ids, dtype=np.int64),
        item_ids=np.stack(item_ids),
        timestamps=np.stack(timestamps),
        labels=np.stack(labels),
        mask=np.stack(masks),
        duration_ms=np.stack(durations),
        tab=np.stack(tabs),
        is_random=np.stack(random_flags),
        watch_ratio=np.stack(watch_ratios),
        objective_names=tuple(config.objective_columns),
        item_prior_count=np.stack(item_prior_counts),
        user_history_count=np.asarray(user_history_counts, dtype=np.int64),
    )
    result.validate()
    return result


def build_sampling_frame(
    slates: PreparedKuaiRandSlates,
    config: KuaiRandSamplingConfig = KuaiRandSamplingConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Describe slates for balanced, budget-aware Jev annotation.

    Popularity and history buckets use strictly prior counts materialized during
    slate construction. This avoids defining a supposedly cold item with future
    impressions from the same evaluation period.
    """

    slates.validate()
    if slates.item_prior_count is None or slates.user_history_count is None:
        raise ValueError(
            "Sampling requires item_prior_count and user_history_count. "
            "Rebuild the NPZ with the current prepare-kuairand command."
        )

    valid_count = slates.mask.sum(axis=1).astype(np.int64)
    log_popularity = np.where(
        slates.mask,
        np.log1p(slates.item_prior_count.astype(np.float64)),
        np.nan,
    )
    mean_log_popularity = np.nanmean(log_popularity, axis=1)
    cold_cut = float(np.quantile(mean_log_popularity, config.cold_quantile))
    warm_cut = float(np.quantile(mean_log_popularity, config.warm_quantile))
    popularity_bucket = np.where(
        mean_log_popularity <= cold_cut,
        "cold",
        np.where(mean_log_popularity >= warm_cut, "warm", "mid"),
    )

    history = slates.user_history_count.astype(np.int64)
    history_bucket = np.where(
        history <= 3,
        "short",
        np.where(history <= 10, "medium", "long"),
    )

    positive_objectives = [
        index
        for index, name in enumerate(slates.objective_names)
        if name != "is_hate"
    ]
    positive_candidate = (
        (slates.labels[:, :, positive_objectives] > 0).any(axis=2) & slates.mask
    )
    positive_count = positive_candidate.sum(axis=1).astype(np.int64)
    feedback_bucket = np.where(
        positive_count == 0,
        "none",
        np.where(
            positive_count >= config.positive_rich_threshold,
            "rich",
            "sparse",
        ),
    )
    random_fraction = np.divide(
        (slates.is_random * slates.mask).sum(axis=1),
        np.maximum(valid_count, 1),
    )
    exposure_bucket = np.where(random_fraction > 0, "random", "standard")

    frame = pd.DataFrame(
        {
            "slate_id": np.arange(len(slates.user_ids), dtype=np.int64),
            "user_id": slates.user_ids.astype(np.int64),
            "timestamp_ms": np.where(
                slates.mask,
                slates.timestamps,
                np.iinfo(np.int64).max,
            ).min(axis=1),
            "valid_candidates": valid_count,
            "user_history_count": history,
            "mean_log1p_item_prior_count": mean_log_popularity,
            "positive_candidate_count": positive_count,
            "random_fraction": random_fraction.astype(np.float64),
            "popularity_bucket": popularity_bucket,
            "history_bucket": history_bucket,
            "feedback_bucket": feedback_bucket,
            "exposure_bucket": exposure_bucket,
        }
    )
    if "is_hate" in slates.objective_names:
        hate_index = slates.objective_names.index("is_hate")
        frame["negative_feedback_count"] = (
            (slates.labels[:, :, hate_index] > 0) & slates.mask
        ).sum(axis=1)
    else:
        frame["negative_feedback_count"] = 0

    for objective_index, objective_name in enumerate(slates.objective_names):
        frame[f"positive_count::{objective_name}"] = (
            (slates.labels[:, :, objective_index] > 0) & slates.mask
        ).sum(axis=1)
    frame["sampling_cell"] = (
        frame["popularity_bucket"].astype(str)
        + "|"
        + frame["history_bucket"].astype(str)
        + "|"
        + frame["exposure_bucket"].astype(str)
        + "|"
        + frame["feedback_bucket"].astype(str)
    )
    report: dict[str, object] = {
        "slates": int(len(frame)),
        "unique_users": int(frame["user_id"].nunique()),
        "popularity_cutpoints": {
            "cold_quantile": config.cold_quantile,
            "cold_max_mean_log1p_prior": cold_cut,
            "warm_quantile": config.warm_quantile,
            "warm_min_mean_log1p_prior": warm_cut,
        },
        "cell_inventory": {
            str(key): int(value)
            for key, value in frame["sampling_cell"].value_counts().sort_index().items()
        },
    }
    return frame, report


def stratified_sample_slates(
    slates: PreparedKuaiRandSlates,
    config: KuaiRandSamplingConfig = KuaiRandSamplingConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a balanced slate subset under cell and per-user budgets."""

    metadata, report = build_sampling_frame(slates, config)
    rng = np.random.default_rng(config.seed)
    buckets: dict[str, list[int]] = {}
    for cell, group in metadata.groupby("sampling_cell", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        buckets[str(cell)] = positions[rng.permutation(len(positions))].tolist()

    cell_order = list(buckets)
    if cell_order:
        cell_order = [cell_order[index] for index in rng.permutation(len(cell_order))]
    cursors = {cell: 0 for cell in cell_order}
    taken_by_cell = {cell: 0 for cell in cell_order}
    taken_by_user: dict[int, int] = {}
    selected_positions: list[int] = []

    while len(selected_positions) < config.max_slates:
        progress = False
        for cell in cell_order:
            if len(selected_positions) >= config.max_slates:
                break
            if taken_by_cell[cell] >= config.per_cell:
                continue
            candidates = buckets[cell]
            cursor = cursors[cell]
            selected_position: int | None = None
            while cursor < len(candidates):
                position = candidates[cursor]
                cursor += 1
                user = int(metadata.at[position, "user_id"])
                if (
                    config.max_per_user is None
                    or taken_by_user.get(user, 0) < config.max_per_user
                ):
                    selected_position = position
                    break
            cursors[cell] = cursor
            if selected_position is None:
                continue
            selected_positions.append(selected_position)
            taken_by_cell[cell] += 1
            user = int(metadata.at[selected_position, "user_id"])
            taken_by_user[user] = taken_by_user.get(user, 0) + 1
            progress = True
        if not progress:
            break

    selected = metadata.loc[selected_positions].copy()
    selected.insert(0, "sample_order", np.arange(len(selected), dtype=np.int64))
    selected = selected.sort_values("sample_order", kind="stable").reset_index(drop=True)
    report.update(
        {
            "sampling_config": {
                "max_slates": config.max_slates,
                "per_cell": config.per_cell,
                "max_per_user": config.max_per_user,
                "seed": config.seed,
                "positive_rich_threshold": config.positive_rich_threshold,
            },
            "selected_slates": int(len(selected)),
            "selected_unique_users": int(selected["user_id"].nunique()),
            "shortfall": int(max(config.max_slates - len(selected), 0)),
            "selected_cell_counts": {
                str(key): int(value)
                for key, value in selected["sampling_cell"].value_counts().sort_index().items()
            },
            "selected_popularity_counts": {
                str(key): int(value)
                for key, value in selected["popularity_bucket"].value_counts().sort_index().items()
            },
            "selected_history_counts": {
                str(key): int(value)
                for key, value in selected["history_bucket"].value_counts().sort_index().items()
            },
            "selected_exposure_counts": {
                str(key): int(value)
                for key, value in selected["exposure_bucket"].value_counts().sort_index().items()
            },
            "selected_feedback_counts": {
                str(key): int(value)
                for key, value in selected["feedback_bucket"].value_counts().sort_index().items()
            },
        }
    )
    return selected, report


def build_history_records(
    log: pd.DataFrame,
    *,
    history_length: int = 10,
    positive_feedback: Iterable[str] = ("is_click", "long_view", "is_like"),
) -> dict[int, list[dict[str, object]]]:
    """Create recent-history records suitable for Jev prompt construction."""

    feedback_columns = tuple(positive_feedback)
    records: dict[int, list[dict[str, object]]] = {}
    for user, group in log.sort_values(["user_id", "time_ms"], kind="stable").groupby(
        "user_id", sort=False
    ):
        recent = group.tail(history_length)
        records[int(user)] = [
            {
                "video_id": int(row.video_id),
                "duration_ms": float(row.duration_ms),
                "feedback": {column: int(getattr(row, column)) for column in feedback_columns},
            }
            for row in recent.itertuples(index=False)
        ]
    return records


def export_slate_manifest(
    slates: PreparedKuaiRandSlates,
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> None:
    """Export an auditable JSONL manifest for decision-model scoring."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = len(slates.user_ids) if max_rows is None else min(max_rows, len(slates.user_ids))
    with path.open("w", encoding="utf-8") as file:
        for index in range(limit):
            valid = slates.mask[index]
            row = {
                "slate_id": index,
                "user_id": int(slates.user_ids[index]),
                "item_ids": [int(x) for x in slates.item_ids[index][valid]],
                "timestamps": [int(x) for x in slates.timestamps[index][valid]],
                "labels": slates.labels[index][valid].tolist(),
                "objective_names": list(slates.objective_names),
                "is_random": [int(x) for x in slates.is_random[index][valid]],
            }
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_jev_input_rows(
    slates: PreparedKuaiRandSlates,
    log: pd.DataFrame,
    content: pd.DataFrame,
    *,
    history_length: int = 10,
    max_rows: int | None = None,
    slate_ids: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Build text-rich JSON records consumable by `paretojev score-jsonl`.

    History is strictly earlier than the first exposure timestamp in each slate.
    Content is joined using `final_video_id`, as defined by the supplementary
    KuaiRand metadata files.
    """

    if "final_video_id" not in content:
        raise ValueError("content metadata must include final_video_id")
    content_map = content.set_index("final_video_id").to_dict(orient="index")
    ordered = log.sort_values(["user_id", "time_ms"], kind="stable")
    by_user = {int(user): group for user, group in ordered.groupby("user_id", sort=False)}
    if slate_ids is None:
        selected_ids = list(range(len(slates.user_ids)))
    else:
        selected_ids = [int(value) for value in slate_ids]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("slate_ids must be unique")
        invalid = [value for value in selected_ids if value < 0 or value >= len(slates.user_ids)]
        if invalid:
            raise ValueError(f"slate_ids out of range; first invalid value: {invalid[0]}")
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        selected_ids = selected_ids[:max_rows]
    rows: list[dict[str, object]] = []

    def item_record(video_id: int, feedback: dict[str, int] | None = None) -> dict[str, object]:
        metadata = content_map.get(video_id, {})
        row: dict[str, object] = {
            "item_id": video_id,
            "video_id": video_id,
            "caption": str(metadata.get("caption", "") or ""),
            "show_cover_text": str(metadata.get("show_cover_text", "") or ""),
            "duration_ms": float(metadata.get("duration", 0.0) or 0.0),
            "first_level_category_name": str(
                metadata.get("first_level_category_name", "") or ""
            ),
            "second_level_category_name": str(
                metadata.get("second_level_category_name", "") or ""
            ),
        }
        if feedback is not None:
            row["feedback"] = feedback
        return row

    for slate_id in selected_ids:
        user = int(slates.user_ids[slate_id])
        valid = slates.mask[slate_id]
        first_timestamp = int(slates.timestamps[slate_id][valid].min())
        user_log = by_user.get(user)
        if user_log is None:
            continue
        earlier = user_log[user_log["time_ms"] < first_timestamp].tail(history_length)
        history = [
            item_record(
                int(row.video_id),
                {
                    "is_click": int(row.is_click),
                    "long_view": int(row.long_view),
                    "is_like": int(row.is_like),
                    "is_follow": int(row.is_follow),
                    "is_comment": int(row.is_comment),
                    "is_forward": int(row.is_forward),
                    "is_hate": int(row.is_hate),
                },
            )
            for row in earlier.itertuples(index=False)
        ]
        candidates = []
        for candidate_position, video_id in enumerate(slates.item_ids[slate_id][valid]):
            candidate = item_record(int(video_id))
            candidate["candidate_id"] = str(int(video_id))
            candidate["position"] = candidate_position
            candidates.append(candidate)
        rows.append(
            {
                "slate_id": slate_id,
                "history": history,
                "candidates": candidates,
                "context": {
                    "dataset": "KuaiRand",
                    "user_id": user,
                    "tab": int(slates.tab[slate_id][valid][0]),
                    "timestamp_ms": first_timestamp,
                    "protocol": "exposed-session pseudo-slate",
                },
            }
        )
    return rows


def export_jev_input_rows(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
