from __future__ import annotations

import collections
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from paretojev.metrics.ranking import (
    TiePolicy,
    per_user_auc,
    positive_ranks,
    single_positive_metrics,
)

MICROLENS_BASE_URL = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/"
MICROLENS_FILES = (
    "MicroLens-100k_pairs.csv",
    "MicroLens-100k_title_en.csv",
    "MicroLens-100k_covers.zip",
)


@dataclass(frozen=True)
class MicroLensProtocol:
    """Candidate protocol used by the public JEV-27B-VL recommendation demo.

    ``upstream_occurrence`` reproduces the demo's negative sampler: draw from
    the time-window interaction *occurrences* with replacement and retain the
    first occurrence of each distinct item.  This intentionally gives popular
    items more sampling mass. ``uniform_unique`` is a cleaner research variant
    that samples uniformly from distinct eligible items, but it is not directly
    comparable to the published demo.
    """

    n_users: int = 200
    n_negatives: int = 19
    history_length: int = 5
    time_window_days: int = 3
    seed: int = 0
    negative_sampling: Literal["upstream_occurrence", "uniform_unique"] = (
        "upstream_occurrence"
    )

    def __post_init__(self) -> None:
        if self.n_users <= 0:
            raise ValueError("n_users must be positive")
        if self.n_negatives <= 0:
            raise ValueError("n_negatives must be positive")
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.time_window_days <= 0:
            raise ValueError("time_window_days must be positive")
        if self.negative_sampling not in {"upstream_occurrence", "uniform_unique"}:
            raise ValueError(f"Unknown negative_sampling: {self.negative_sampling}")

    @property
    def candidate_count(self) -> int:
        return self.n_negatives + 1


def _download_file(url: str, destination: Path, timeout_seconds: float = 600.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with temporary.open("wb") as out:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    out.write(chunk)
    temporary.replace(destination)


def _safe_extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        archive.extractall(destination)


def download_microlens(
    directory: str | Path,
    *,
    include_covers: bool = False,
    extract_covers: bool = False,
    force: bool = False,
) -> list[Path]:
    """Download official MicroLens-100K files.

    The pairs and English-title files are sufficient for the text/CF track. Covers
    are optional because the archive is large (roughly hundreds of MB).
    """

    directory = Path(directory)
    filenames = list(MICROLENS_FILES[:2])
    if include_covers:
        filenames.append(MICROLENS_FILES[2])

    downloaded: list[Path] = []
    for filename in filenames:
        destination = directory / filename
        if force or not destination.exists():
            _download_file(MICROLENS_BASE_URL + filename, destination)
        downloaded.append(destination)

    cover_zip = directory / MICROLENS_FILES[2]
    cover_dir = directory / "covers"
    if include_covers and extract_covers and (force or not cover_dir.exists()):
        _safe_extract_zip(cover_zip, cover_dir)
    return downloaded


def load_titles(directory: str | Path) -> dict[int, str]:
    path = Path(directory) / "MicroLens-100k_title_en.csv"
    frame = pd.read_csv(path, header=None, names=["item", "title"])
    return {int(row.item): str(row.title) for row in frame.itertuples(index=False)}


def _normalize_timestamp_milliseconds(values: pd.Series) -> pd.Series:
    """Normalize common Unix timestamp units to milliseconds.

    MicroLens mirrors have appeared with the same logical timestamp column but
    not always the same documented unit.  Candidate-window construction uses
    milliseconds internally, so detect seconds/milliseconds/microseconds/
    nanoseconds from the magnitude of epoch-like values.  Non-epoch ordinal
    values are left unchanged rather than guessed.
    """

    numeric = pd.to_numeric(values, errors="raise").astype("int64")
    non_zero = numeric[numeric != 0].abs()
    if non_zero.empty:
        return numeric

    magnitude = float(non_zero.median())
    if 1e8 <= magnitude < 1e11:  # Unix seconds
        return numeric * 1_000
    if 1e11 <= magnitude < 1e14:  # Unix milliseconds
        return numeric
    if 1e14 <= magnitude < 1e17:  # Unix microseconds
        return numeric // 1_000
    if magnitude >= 1e17:  # Unix nanoseconds
        return numeric // 1_000_000
    return numeric


def load_pairs(directory: str | Path) -> pd.DataFrame:
    """Load MicroLens interactions and normalize known schema variants.

    The official/legacy files use ``userID`` and ``videoID`` in some releases,
    while downstream projects often rename them to ``user`` and ``item``.
    Accept both forms so the documented public file works without a manual
    preprocessing step.
    """

    path = Path(directory) / "MicroLens-100k_pairs.csv"
    frame = pd.read_csv(path)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]

    aliases = {
        "userID": "user",
        "user_id": "user",
        "uid": "user",
        "videoID": "item",
        "video_id": "item",
        "item_id": "item",
        "time": "timestamp",
        "ts": "timestamp",
    }
    rename = {column: aliases[column] for column in frame.columns if column in aliases}
    frame = frame.rename(columns=rename)

    required = {"user", "item", "timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "MicroLens pairs file is missing canonical columns "
            f"{sorted(missing)}; observed columns: {sorted(frame.columns)}"
        )

    result = frame.loc[:, ["user", "item", "timestamp"]].copy()
    result["user"] = pd.to_numeric(result["user"], errors="raise").astype("int64")
    result["item"] = pd.to_numeric(result["item"], errors="raise").astype("int64")
    result["timestamp"] = _normalize_timestamp_milliseconds(result["timestamp"])
    return result.sort_values(["user", "timestamp"], kind="stable").reset_index(drop=True)


def prepare_microlens_candidates(
    directory: str | Path,
    protocol: MicroLensProtocol = MicroLensProtocol(),
) -> pd.DataFrame:
    """Build deterministic candidate slates comparable to JEV-27B-VL Demo 07.

    Each test user contributes the final item as the positive, the preceding N
    items as history, and negatives sampled from items watched by other users in
    a ±time-window around the positive timestamp. Item-CF is trained only on
    non-test users, matching the public demo's leakage controls.
    """

    pairs = load_pairs(directory)
    titles = load_titles(directory)
    grouped = pairs.groupby("user", sort=False).agg(list)
    grouped = grouped[grouped["item"].str.len() >= protocol.history_length + 1]
    if len(grouped) < protocol.n_users:
        raise ValueError(
            f"Only {len(grouped)} eligible users, but protocol requests {protocol.n_users}"
        )

    rng = random.Random(protocol.seed)
    users = rng.sample([int(x) for x in grouped.index], protocol.n_users)
    chronological = pairs.sort_values("timestamp", kind="stable")
    all_times = chronological["timestamp"].to_numpy()
    all_items = chronological["item"].to_numpy()
    window_ms = protocol.time_window_days * 86_400_000

    popularity_times: dict[int, list[int]] = collections.defaultdict(list)
    for item, timestamp in zip(all_items, all_times, strict=True):
        popularity_times[int(item)].append(int(timestamp))

    held_out_users = set(users)
    cooccurrence: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for user, row in grouped.iterrows():
        if int(user) in held_out_users:
            continue
        recent = [int(x) for x in row["item"][-20:]]
        for source in recent:
            for target in recent:
                if source != target:
                    cooccurrence[source][target] += 1

    rows: list[dict[str, object]] = []
    for user in users:
        items = [int(x) for x in grouped.loc[user, "item"]]
        times = [int(x) for x in grouped.loc[user, "timestamp"]]
        positive = items[-1]
        target_time = times[-1]
        history = items[-1 - protocol.history_length : -1]
        low = int(np.searchsorted(all_times, target_time - window_ms, side="left"))
        # The public demo uses NumPy's default ``side=left`` at both bounds.
        # Preserve that half-open interval for exact candidate comparability.
        high = int(np.searchsorted(all_times, target_time + window_ms, side="left"))
        seen = set(items)
        pool = [int(x) for x in all_items[low:high] if int(x) not in seen]
        unique_pool = list(dict.fromkeys(pool))
        if len(unique_pool) < protocol.n_negatives:
            raise ValueError(
                f"User {user} has only {len(unique_pool)} eligible negatives; "
                f"need {protocol.n_negatives}"
            )
        if protocol.negative_sampling == "upstream_occurrence":
            # Match upstream exactly: sample interaction occurrences (therefore
            # popularity weighted), accepting only the first draw of each item.
            negatives: list[int] = []
            while len(negatives) < protocol.n_negatives:
                candidate = int(rng.choice(pool))
                if candidate not in negatives:
                    negatives.append(candidate)
        else:
            negatives = rng.sample(unique_pool, protocol.n_negatives)
        candidates = [positive, *negatives]

        for position, candidate in enumerate(candidates):
            prior_count = int(
                np.searchsorted(
                    np.asarray(popularity_times[candidate], dtype=np.int64),
                    target_time,
                    side="left",
                )
            )
            rows.append(
                {
                    "user": user,
                    "item": candidate,
                    "position": position,
                    "y": int(candidate == positive),
                    "target_timestamp": target_time,
                    "hist_items": history,
                    "hist_titles": [titles.get(item, "") for item in history],
                    "title": titles.get(candidate, ""),
                    "pop": math.log1p(prior_count),
                    "itemcf": float(sum(cooccurrence[h][candidate] for h in history)),
                }
            )

    frame = pd.DataFrame(rows)
    relevant_items = set(frame["item"].astype(int))
    relevant_items.update(h for history in frame["hist_items"] for h in history)
    corpus = [str(titles.get(item, "")) for item in sorted(relevant_items)]
    vectorizer = TfidfVectorizer(min_df=1).fit(corpus)
    title_vectors = {
        item: vectorizer.transform([str(titles.get(item, ""))]) for item in relevant_items
    }
    similarities: list[float] = []
    for row in frame.itertuples(index=False):
        candidate_vector = title_vectors[int(row.item)]
        history_matrix = vectorizer.transform(
            [str(titles.get(int(item), "")) for item in row.hist_items]
        )
        similarities.append(float((candidate_vector @ history_matrix.T).max()))
    frame["title_sim"] = similarities
    return frame


def save_microlens_protocol(frame: pd.DataFrame, output_directory: str | Path) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    serializable["hist_items"] = serializable["hist_items"].map(json.dumps)
    if "hist_titles" in serializable:
        serializable["hist_titles"] = serializable["hist_titles"].map(json.dumps)
    serializable.to_csv(output_directory / "candidates.csv", index=False)

    payload: dict[str, dict[str, object]] = {}
    jev_rows: list[dict[str, object]] = []
    for user, group in frame.groupby("user", sort=False):
        history_items = [int(x) for x in group["hist_items"].iloc[0]]
        history_titles = [str(x) for x in group["hist_titles"].iloc[0]]
        payload[str(int(user))] = {
            "history": history_items,
            "candidates": [int(x) for x in group["item"]],
            "clicked": int(group.loc[group["y"] == 1, "item"].iloc[0]),
        }
        jev_rows.append(
            {
                "slate_id": int(user),
                "history": [
                    {"item_id": item, "caption": title, "feedback": {"watched": True}}
                    for item, title in zip(history_items, history_titles, strict=True)
                ],
                "candidates": [
                    {"candidate_id": int(row.item), "item_id": int(row.item), "caption": str(row.title)}
                    for row in group.itertuples(index=False)
                ],
                "context": {"dataset": "MicroLens-100K", "task": "next-item reranking"},
            }
        )
    with (output_directory / "candidates.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (output_directory / "jev_input.jsonl").open("w", encoding="utf-8") as file:
        for row in jev_rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_prepared_candidates(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "hist_items" in frame:
        frame["hist_items"] = frame["hist_items"].map(json.loads)
    if "hist_titles" in frame:
        frame["hist_titles"] = frame["hist_titles"].map(json.loads)
    return frame


def grouped_score_arrays(
    frame: pd.DataFrame,
    score_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = list(frame.groupby("user", sort=False))
    if not groups:
        raise ValueError("No users in candidate frame")
    sizes = {len(group) for _, group in groups}
    if len(sizes) != 1:
        raise ValueError(f"Candidate slate sizes are inconsistent: {sorted(sizes)}")
    users = np.asarray([int(user) for user, _ in groups], dtype=np.int64)
    scores = np.stack([group[score_column].to_numpy(dtype=np.float64) for _, group in groups])
    labels = np.stack([group["y"].to_numpy(dtype=np.int64) for _, group in groups])
    return users, scores, labels


def evaluate_score_column(
    frame: pd.DataFrame,
    score_column: str,
    ks: Iterable[int] = (1, 5, 10),
    *,
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> dict[str, float]:
    """Evaluate one score column under an explicit tie policy.

    The default mirrors the public MicroLens JEV demo. Pass ``stable`` for the
    conventional AUC half-credit treatment and stable input-order rank ties.
    """

    _, scores, labels = grouped_score_arrays(frame, score_column)
    return single_positive_metrics(
        scores, labels, ks=ks, tie_policy=tie_policy, tie_seed=tie_seed
    )


def _simplex_grid(n_models: int, step: float) -> list[np.ndarray]:
    if n_models < 2:
        return [np.ones(1, dtype=np.float64)]
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly")

    weights: list[np.ndarray] = []

    def recurse(prefix: list[int], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            weights.append(np.asarray([*prefix, remaining], dtype=np.float64) / units)
            return
        for value in range(remaining + 1):
            recurse([*prefix, value], remaining - value, dimensions - 1)

    recurse([], units, n_models)
    return weights


def _within_slate_rank_normalize(scores: np.ndarray) -> np.ndarray:
    # Equal scores carry equal information. Assigning distinct ranks by input
    # position can bias fusion when the held-out positive is stored first.
    ranks = pd.DataFrame(scores).rank(axis=1, method="average").to_numpy()
    return (ranks - 1.0) / max(scores.shape[1] - 1, 1)


def cross_validated_hybrid(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    folds: int = 5,
    weight_step: float = 0.05,
    objective: str = "balanced",
    seed: int = 0,
    output_column: str = "hybrid_cv",
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Tune a transparent linear hybrid without evaluating on its own users.

    Model weights are selected on K-1 folds and applied to the held-out fold. The
    default objective averages AUC, HR@5 and NDCG@10, matching the stated target
    of beating both Jev-like and CF signals rather than overfitting one metric.
    """

    if len(score_columns) < 2:
        raise ValueError("At least two score columns are required")
    users = np.asarray(sorted(frame["user"].unique()), dtype=np.int64)
    if folds < 2 or folds > len(users):
        raise ValueError("folds must be between 2 and number of users")
    rng = np.random.default_rng(seed)
    users = users[rng.permutation(len(users))]
    fold_users = np.array_split(users, folds)
    weight_grid = _simplex_grid(len(score_columns), weight_step)
    result = frame.copy()
    result[output_column] = np.nan
    reports: list[dict[str, object]] = []

    def objective_value(metrics: dict[str, float]) -> float:
        if objective == "auc":
            return metrics["auc"]
        if objective == "ndcg@10":
            return metrics["ndcg@10"]
        if objective == "balanced":
            return (metrics["auc"] + metrics["hr@5"] + metrics["ndcg@10"]) / 3.0
        raise ValueError(f"Unknown objective: {objective}")

    for fold_index, held_out in enumerate(fold_users):
        train_mask = ~result["user"].isin(set(int(x) for x in held_out))
        validation = result.loc[train_mask]
        _, labels_val = None, None
        normalized_columns: dict[str, np.ndarray] = {}
        groups = list(validation.groupby("user", sort=False))
        labels_val = np.stack([group["y"].to_numpy(dtype=np.int64) for _, group in groups])
        for column in score_columns:
            scores = np.stack(
                [group[column].to_numpy(dtype=np.float64) for _, group in groups]
            )
            normalized_columns[column] = _within_slate_rank_normalize(scores)

        best_score = -float("inf")
        best_weights: np.ndarray | None = None
        best_metrics: dict[str, float] | None = None
        for weights in weight_grid:
            combined = sum(
                weight * normalized_columns[column]
                for weight, column in zip(weights, score_columns, strict=True)
            )
            metrics = single_positive_metrics(
                combined,
                labels_val,
                ks=(1, 5, 10),
                tie_policy=tie_policy,
                tie_seed=tie_seed,
            )
            value = objective_value(metrics)
            if value > best_score + 1e-12:
                best_score = value
                best_weights = weights
                best_metrics = metrics

        assert best_weights is not None and best_metrics is not None
        for user in held_out:
            index = result.index[result["user"] == int(user)]
            raw = np.stack(
                [result.loc[index, column].to_numpy(dtype=np.float64) for column in score_columns],
                axis=0,
            )
            normalized = np.stack(
                [_within_slate_rank_normalize(row[None, :])[0] for row in raw], axis=0
            )
            result.loc[index, output_column] = best_weights @ normalized
        reports.append(
            {
                "fold": fold_index,
                "held_out_users": [int(x) for x in held_out],
                "weights": {
                    column: float(weight)
                    for column, weight in zip(score_columns, best_weights, strict=True)
                },
                "validation_metrics": best_metrics,
            }
        )

    if result[output_column].isna().any():
        raise RuntimeError("Some out-of-fold scores were not assigned")
    return result, reports


def per_user_metric_matrix(
    frame: pd.DataFrame,
    score_column: str,
    *,
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> np.ndarray:
    """Return [N,5]: AUC, HR@1, HR@5, NDCG@10, MRR for bootstrap tests."""

    _, scores, labels = grouped_score_arrays(frame, score_column)
    ranks = positive_ranks(
        scores, labels, tie_policy=tie_policy, tie_seed=tie_seed
    )
    auc = per_user_auc(
        scores, labels, tie_policy=tie_policy, tie_seed=tie_seed
    )
    return np.stack(
        [
            auc,
            (ranks <= 1).astype(np.float64),
            (ranks <= 5).astype(np.float64),
            np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0),
            1.0 / ranks,
        ],
        axis=1,
    )
