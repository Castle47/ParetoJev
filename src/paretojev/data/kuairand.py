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

    def validate(self) -> None:
        n, k = self.item_ids.shape
        if self.user_ids.shape != (n,):
            raise ValueError("user_ids shape mismatch")
        for name in ("timestamps", "mask", "duration_ms", "tab", "is_random", "watch_ratio"):
            if getattr(self, name).shape != (n, k):
                raise ValueError(f"{name} shape mismatch")
        if self.labels.shape != (n, k, len(self.objective_names)):
            raise ValueError("labels shape mismatch")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            timestamps=self.timestamps,
            labels=self.labels,
            mask=self.mask,
            duration_ms=self.duration_ms,
            tab=self.tab,
            is_random=self.is_random,
            watch_ratio=self.watch_ratio,
            objective_names=np.asarray(self.objective_names, dtype="U64"),
        )

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
            )
        result.validate()
        return result


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
    frame = log.copy().sort_values(["user_id", "time_ms"], kind="stable")
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
    )
    result.validate()
    return result


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
    limit = len(slates.user_ids) if max_rows is None else min(max_rows, len(slates.user_ids))
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

    for slate_id in range(limit):
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
