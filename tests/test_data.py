from pathlib import Path

import numpy as np
import pandas as pd

from paretojev.data.kuairand import KuaiRandSlateConfig, build_session_slates, read_kuairand_log
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
