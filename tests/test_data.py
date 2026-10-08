import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paretojev.cli import main
from paretojev.data.kuairand import (
    KUAI_OBJECTIVES,
    KuaiRandSamplingConfig,
    KuaiRandSlateConfig,
    PreparedKuaiRandSlates,
    build_jev_input_rows,
    build_session_slates,
    read_kuairand_log,
    stratified_sample_slates,
)
from paretojev.data.microlens import (
    MicroLensProtocol,
    cross_validated_hybrid,
    load_pairs,
    prepare_microlens_candidates,
)


def test_kuairand_session_slates(tmp_path: Path) -> None:
    rows = []
    for user in (1, 2):
        for index in range(7):
            rows.append(
                {
                    "user_id": user,
                    "video_id": user * 100 + index,
                    "date": 20220421,
                    "hourmin": 100,
                    "time_ms": user * 10_000_000 + index * 60_000,
                    "is_click": index % 2,
                    "is_like": int(index == 3),
                    "is_follow": int(index == 4),
                    "is_comment": int(index == 5),
                    "is_forward": 0,
                    "is_hate": int(index == 6),
                    "long_view": int(index % 3 == 0),
                    "play_time_ms": 5000 + index,
                    "duration_ms": 10000,
                    "profile_stay_time": 0,
                    "comment_stay_time": 0,
                    "is_profile_enter": 0,
                    "is_rand": 0,
                    "tab": 1,
                }
            )
    path = tmp_path / "log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = read_kuairand_log(path)
    slates = build_session_slates(
        frame,
        KuaiRandSlateConfig(slate_size=5, min_slate_size=2, stride=5),
    )
    assert slates.labels.shape[-1] == 6
    assert len(slates.user_ids) == 4
    assert slates.mask.sum() == 14
    assert slates.item_prior_count is not None
    assert slates.user_history_count is not None
    assert slates.item_prior_count.shape == slates.item_ids.shape
    assert slates.user_history_count.shape == slates.user_ids.shape

    content = pd.DataFrame(
        {
            "final_video_id": sorted(frame["video_id"].unique()),
            "caption": [f"video {value}" for value in sorted(frame["video_id"].unique())],
        }
    )
    rows = build_jev_input_rows(slates, frame, content, slate_ids=[1, 0])
    assert [row["slate_id"] for row in rows] == [1, 0]


def test_kuairand_stratified_decision_sampling(tmp_path: Path) -> None:
    n, k, m = 12, 3, len(KUAI_OBJECTIVES)
    mask = np.ones((n, k), dtype=bool)
    labels = np.zeros((n, k, m), dtype=np.float32)
    for slate in range(n):
        feedback = slate % 3
        if feedback >= 1:
            labels[slate, 0, 0] = 1
        if feedback >= 2:
            labels[slate, 1, 2] = 1
    item_prior = np.zeros((n, k), dtype=np.int64)
    item_prior[4:8] = 10
    item_prior[8:] = 100
    slates = PreparedKuaiRandSlates(
        user_ids=np.repeat(np.arange(6), 2),
        item_ids=np.arange(n * k).reshape(n, k),
        timestamps=np.arange(n * k).reshape(n, k) * 1000 + 1,
        labels=labels,
        mask=mask,
        duration_ms=np.full((n, k), 10_000, dtype=np.float32),
        tab=np.ones((n, k), dtype=np.int16),
        is_random=np.repeat(np.arange(n)[:, None] % 2, k, axis=1).astype(np.int8),
        watch_ratio=np.zeros((n, k), dtype=np.float32),
        objective_names=KUAI_OBJECTIVES,
        item_prior_count=item_prior,
        user_history_count=np.asarray([1, 5, 20] * 4, dtype=np.int64),
    )
    selected, report = stratified_sample_slates(
        slates,
        KuaiRandSamplingConfig(
            max_slates=6,
            per_cell=2,
            max_per_user=1,
            seed=11,
        ),
    )
    assert 1 <= len(selected) <= 6
    assert selected["user_id"].value_counts().max() == 1
    assert selected["popularity_bucket"].nunique() >= 2
    assert report["selected_slates"] == len(selected)
    assert "selected_cell_counts" in report

    path = tmp_path / "slates.npz"
    slates.save(path)
    loaded = PreparedKuaiRandSlates.load(path)
    np.testing.assert_array_equal(loaded.item_prior_count, item_prior)
    np.testing.assert_array_equal(loaded.user_history_count, slates.user_history_count)


def test_kuairand_prefix_log_initializes_prior_counts() -> None:
    def row(user: int, video: int, timestamp: int) -> dict[str, int]:
        return {
            "user_id": user,
            "video_id": video,
            "time_ms": timestamp,
            "is_click": 0,
            "is_like": 0,
            "is_follow": 0,
            "is_comment": 0,
            "is_forward": 0,
            "is_hate": 0,
            "long_view": 0,
            "play_time_ms": 0,
            "duration_ms": 10_000,
            "is_rand": 0,
            "tab": 1,
        }

    prefix = pd.DataFrame([row(1, 10, 1_000), row(1, 11, 2_000)])
    current = pd.DataFrame(
        [
            row(1, 10, 3_000),
            row(1, 12, 4_000),
            row(2, 10, 5_000),
        ]
    )
    slates = build_session_slates(
        current,
        KuaiRandSlateConfig(slate_size=3, min_slate_size=2, stride=3),
        prefix_log=prefix,
    )
    assert slates.user_history_count is not None
    assert slates.item_prior_count is not None
    # User 1 had two prior events; video 10 had one prior event.
    assert slates.user_history_count[0] == 2
    assert slates.item_prior_count[0, 0] == 1



def test_microlens_official_header_aliases_and_second_timestamps(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"userID": 2, "videoID": 20, "timestamp": 1_700_000_002},
            {"userID": 1, "videoID": 11, "timestamp": 1_700_000_001},
            {"userID": 1, "videoID": 10, "timestamp": 1_700_000_000},
        ]
    ).to_csv(tmp_path / "MicroLens-100k_pairs.csv", index=False)

    frame = load_pairs(tmp_path)
    assert list(frame.columns) == ["user", "item", "timestamp"]
    assert frame[["user", "item"]].values.tolist() == [[1, 10], [1, 11], [2, 20]]
    assert frame["timestamp"].tolist() == [
        1_700_000_000_000,
        1_700_000_001_000,
        1_700_000_002_000,
    ]

def test_microlens_protocol_and_hybrid(tmp_path: Path) -> None:
    pairs = []
    timestamp = 1_000_000_000
    # Six users with disjoint histories ensure a broad negative pool.
    for user in range(6):
        for offset in range(5):
            pairs.append(
                {
                    "user": user,
                    "item": user * 10 + offset,
                    "timestamp": timestamp + offset * 1000 + user * 10,
                }
            )
    pd.DataFrame(pairs).to_csv(tmp_path / "MicroLens-100k_pairs.csv", index=False)
    with (tmp_path / "MicroLens-100k_title_en.csv").open("w", encoding="utf-8") as file:
        for item in range(60):
            file.write(f"{item},topic {item % 5} item {item}\n")
    frame = prepare_microlens_candidates(
        tmp_path,
        MicroLensProtocol(n_users=4, n_negatives=3, history_length=2, seed=0),
    )
    assert len(frame) == 16
    assert frame.groupby("user")["y"].sum().eq(1).all()
    # Construct a second signal with different ranking so CV code is exercised.
    rng = np.random.default_rng(0)
    frame["jev_mock"] = frame["y"] * 0.6 + rng.normal(0, 0.3, len(frame))
    result, reports = cross_validated_hybrid(
        frame, ["itemcf", "jev_mock"], folds=2, weight_step=0.5
    )
    assert result["hybrid_cv"].notna().all()
    assert len(reports) == 2



def _upstream_reference_candidates(
    pairs: pd.DataFrame,
    *,
    n_users: int,
    n_negatives: int,
    history_length: int,
    seed: int,
) -> dict[int, list[int]]:
    ordered = pairs.sort_values(["user", "timestamp"])
    sequences = ordered.groupby("user").agg(list)
    sequences = sequences[sequences["item"].str.len() >= history_length + 1]
    rng = random.Random(seed)
    users = rng.sample(list(sequences.index), n_users)
    chronological = ordered.sort_values("timestamp")
    timestamps = chronological["timestamp"].to_numpy()
    items = chronological["item"].to_numpy()
    day = 86_400_000
    result: dict[int, list[int]] = {}
    for user in users:
        user_items = sequences.loc[user, "item"]
        user_times = sequences.loc[user, "timestamp"]
        target = int(user_items[-1])
        target_time = int(user_times[-1])
        low = np.searchsorted(timestamps, target_time - 3 * day)
        high = np.searchsorted(timestamps, target_time + 3 * day)
        seen = set(user_items)
        pool = [int(value) for value in items[low:high] if int(value) not in seen]
        negatives: list[int] = []
        while len(negatives) < n_negatives:
            candidate = int(rng.choice(pool))
            if candidate not in negatives:
                negatives.append(candidate)
        result[int(user)] = [target, *negatives]
    return result


def test_microlens_upstream_candidate_conformance(tmp_path: Path) -> None:
    rows = []
    base = 1_700_000_000_000
    for user in range(12):
        for offset in range(7):
            # Disjoint user histories but overlapping time windows. Repeated
            # occurrence frequencies across users exercise upstream weighting.
            rows.append(
                {
                    "user": user,
                    "item": user * 100 + offset,
                    "timestamp": base + offset * 1000 + user,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "MicroLens-100k_pairs.csv", index=False)
    with (tmp_path / "MicroLens-100k_title_en.csv").open("w", encoding="utf-8") as file:
        for user in range(12):
            for offset in range(7):
                item = user * 100 + offset
                file.write(f"{item},user {user} clip {offset}\n")

    loaded = load_pairs(tmp_path)
    expected = _upstream_reference_candidates(
        loaded,
        n_users=5,
        n_negatives=6,
        history_length=3,
        seed=0,
    )
    actual = prepare_microlens_candidates(
        tmp_path,
        MicroLensProtocol(
            n_users=5,
            n_negatives=6,
            history_length=3,
            seed=0,
            negative_sampling="upstream_occurrence",
        ),
    )
    observed = {
        int(user): [int(value) for value in group["item"]]
        for user, group in actual.groupby("user", sort=False)
    }
    assert observed == expected


def test_kuairand_prior_counts_exclude_all_same_timestamp_exposures() -> None:
    def row(user: int, video: int, timestamp: int) -> dict[str, int]:
        return {
            "user_id": user,
            "video_id": video,
            "time_ms": timestamp,
            "is_click": 0,
            "is_like": 0,
            "is_follow": 0,
            "is_comment": 0,
            "is_forward": 0,
            "is_hate": 0,
            "long_view": 0,
            "play_time_ms": 0,
            "duration_ms": 10_000,
            "is_rand": 0,
            "tab": 1,
        }

    current = pd.DataFrame(
        [
            row(1, 10, 1_000),
            row(1, 10, 1_000),
            row(1, 11, 1_000),
            row(1, 10, 2_000),
        ]
    )
    slates = build_session_slates(
        current,
        KuaiRandSlateConfig(slate_size=4, min_slate_size=2, stride=4),
    )
    assert slates.item_prior_count is not None
    assert slates.user_history_count is not None
    # The first three rows are simultaneous: none may count another as history.
    assert slates.item_prior_count[0, :3].tolist() == [0, 0, 0]
    assert slates.user_history_count[0] == 0
    # The later repeat sees both earlier item-10 exposures.
    assert slates.item_prior_count[0, 3] == 2


def test_kuairand_temporal_split_preserves_equal_timestamp_groups() -> None:
    from paretojev.data.kuairand import temporal_split_log

    frame = pd.DataFrame(
        {
            "time_ms": [100, 100, 200, 300, 300, 400, 500, 600],
            "value": np.arange(8),
        }
    )
    splits, report = temporal_split_log(
        frame,
        (0.5, 0.25, 0.25),
        names=("behavior_train", "behavior_validation", "reranker_train"),
    )
    assert sum(len(part) for part in splits.values()) == len(frame)
    assert set(splits["behavior_train"]["time_ms"]).isdisjoint(
        set(splits["behavior_validation"]["time_ms"])
    )
    assert set(splits["behavior_validation"]["time_ms"]).isdisjoint(
        set(splits["reranker_train"]["time_ms"])
    )
    assert (
        splits["behavior_train"]["time_ms"].max()
        < splits["behavior_validation"]["time_ms"].min()
        < splits["reranker_train"]["time_ms"].min()
    )
    assert report["rows"] == len(frame)



def test_split_kuairand_cli_writes_strict_prefixes(tmp_path: Path, capsys) -> None:
    rows: list[dict[str, int]] = []
    timestamps = [100, 100, 200, 300, 300, 400, 500, 600, 700, 700, 800, 900]
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "user_id": 1 + index % 3,
                "video_id": 10 + index % 5,
                "time_ms": timestamp,
                "is_click": index % 2,
                "is_like": int(index % 4 == 0),
                "is_follow": int(index % 7 == 0),
                "is_comment": int(index % 5 == 0),
                "is_forward": int(index % 6 == 0),
                "is_hate": int(index % 11 == 0),
                "long_view": int(index % 3 == 0),
                "play_time_ms": 5_000,
                "duration_ms": 10_000,
                "is_rand": 0,
                "tab": 1,
            }
        )
    source = tmp_path / "all.csv"
    output = tmp_path / "splits"
    report = tmp_path / "split_report.json"
    pd.DataFrame(rows).to_csv(source, index=False)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "split-kuairand-log",
                str(source),
                str(output),
                "--fractions",
                "0.5,0.25,0.25",
                "--names",
                "behavior_train,behavior_validation,reranker_train",
                "--write-prefixes",
                "--report",
                str(report),
            ]
        )
    assert exit_info.value.code == 0
    capsys.readouterr()

    train = read_kuairand_log(output / "behavior_train.csv")
    validation = read_kuairand_log(output / "behavior_validation.csv")
    reranker = read_kuairand_log(output / "reranker_train.csv")
    validation_prefix = read_kuairand_log(
        output / "prefix_before_behavior_validation.csv"
    )
    reranker_prefix = read_kuairand_log(output / "prefix_before_reranker_train.csv")
    assert train["time_ms"].max() < validation["time_ms"].min()
    assert validation["time_ms"].max() < reranker["time_ms"].min()
    assert len(validation_prefix) == len(train)
    assert len(reranker_prefix) == len(train) + len(validation)
    payload = json.loads(report.read_text())
    assert payload["rows"] == len(rows)
    assert set(payload["prefix_outputs"]) == {
        "behavior_validation",
        "reranker_train",
    }
