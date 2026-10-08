from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paretojev.cli import build_parser
from paretojev.data.behavior_sequences import (
    ItemVocabulary,
    KuaiRandBehaviorDataset,
    UserHistoryIndex,
    build_behavior_event_table,
    fit_behavior_context_spec,
)
from paretojev.data.kuairand import (
    KUAI_OBJECTIVES,
    KuaiRandSlateConfig,
    PreparedKuaiRandSlates,
    build_session_slates,
    read_kuairand_log,
)
from paretojev.data.materialize import SlateScoreTensor
from paretojev.training.behavior import (
    BehaviorTrainingConfig,
    run_kuairand_behavior_training,
    score_kuairand_behavior_slates,
)


def _row(user: int, item: int, timestamp: int, index: int) -> dict[str, int]:
    return {
        "user_id": user,
        "video_id": item,
        "time_ms": timestamp,
        "is_click": int((item + user) % 2 == 0),
        "is_like": int(index % 4 == 0),
        "is_follow": int(index % 7 == 0),
        "is_comment": int(index % 5 == 0),
        "is_forward": int(index % 6 == 0),
        "is_hate": int(index % 11 == 0),
        "long_view": int((item + index) % 3 == 0),
        "play_time_ms": 5_000,
        "duration_ms": 10_000 + item,
        "is_rand": int(index % 9 == 0),
        "tab": 1 + item % 3,
    }


def _write_logs(tmp_path: Path) -> tuple[Path, Path]:
    train_rows: list[dict[str, int]] = []
    validation_rows: list[dict[str, int]] = []
    for user in range(1, 9):
        for index in range(8):
            item = 1 + (user * 3 + index) % 12
            train_rows.append(_row(user, item, 1_000 + index * 10 + user, index))
        for index in range(4):
            item = 1 + (user * 5 + index) % 15
            validation_rows.append(
                _row(user, item, 10_000 + index * 10 + user, index + 20)
            )
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    return train_path, validation_path


def test_behavior_event_history_is_strict_at_equal_timestamps() -> None:
    frame = pd.DataFrame(
        [
            _row(1, 10, 100, 0),
            _row(1, 11, 100, 1),
            _row(1, 12, 200, 2),
        ]
    )
    table = build_behavior_event_table(frame)
    vocabulary = ItemVocabulary.fit(frame["video_id"].to_numpy())
    spec = fit_behavior_context_spec(table)
    dataset = KuaiRandBehaviorDataset(
        table,
        vocabulary,
        spec,
        max_history_length=4,
    )

    assert dataset[0]["history_item_ids"].count_nonzero().item() == 0
    assert dataset[1]["history_item_ids"].count_nonzero().item() == 0
    later_history = dataset[2]["history_item_ids"].numpy()
    assert np.count_nonzero(later_history) == 2
    assert later_history[:2].tolist() == [0, 0]
    assert later_history[-2:].tolist() == vocabulary.encode(np.array([10, 11])).tolist()

    index = UserHistoryIndex(frame, vocabulary)
    assert np.count_nonzero(index.history(1, 100, max_history_length=4)) == 0
    assert np.count_nonzero(index.history(1, 200, max_history_length=4)) == 2


def test_item_vocabulary_uses_explicit_padding_and_oov() -> None:
    vocabulary = ItemVocabulary.fit(np.array([30, 10, 20, 10]))
    encoded = vocabulary.encode(
        np.array([10, 99, -1]),
        padding_mask=np.array([False, False, True]),
    )
    assert vocabulary.raw_item_ids == (10, 20, 30)
    assert encoded.tolist() == [2, 1, 0]


def test_train_and_score_neural_kuairand_behavior_branch(tmp_path: Path) -> None:
    train_path, validation_path = _write_logs(tmp_path)
    checkpoint = tmp_path / "behavior.pt"
    metrics_path = tmp_path / "behavior_metrics.json"
    config = BehaviorTrainingConfig(
        seed=3,
        batch_size=16,
        epochs=2,
        patience=2,
        learning_rate=2e-3,
        max_history_length=5,
        embedding_dim=8,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        n_experts=2,
        expert_dim=8,
        dropout=0.0,
        max_tab_values=4,
        device="cpu",
        max_train_examples=48,
        max_validation_examples=24,
    )
    report = run_kuairand_behavior_training(
        train_path,
        validation_path,
        config=config,
        checkpoint_path=checkpoint,
        output_path=metrics_path,
    )
    assert checkpoint.exists()
    assert report["status"] == "complete"
    assert report["data"]["train_events_used"] == 48
    assert len(report["validation"]["per_objective"]) == len(KUAI_OBJECTIVES)

    train_log = read_kuairand_log(train_path)
    validation_log = read_kuairand_log(validation_path)
    slates = build_session_slates(
        validation_log,
        KuaiRandSlateConfig(slate_size=4, min_slate_size=2, stride=4),
        prefix_log=train_log,
    )
    slates_path = tmp_path / "validation_slates.npz"
    scores_path = tmp_path / "validation_behavior.npz"
    score_report_path = tmp_path / "validation_behavior.report.json"
    slates.save(slates_path)
    score_report = score_kuairand_behavior_slates(
        checkpoint,
        slates_path,
        validation_path,
        scores_path,
        prefix_log_path=train_path,
        batch_size=8,
        report_path=score_report_path,
    )
    scores = SlateScoreTensor.load(scores_path)
    assert scores.probabilities.shape == slates.labels.shape
    assert scores.objective_names == KUAI_OBJECTIVES
    assert np.isfinite(scores.probabilities[slates.mask]).all()
    assert score_report["valid_candidates"] == int(slates.mask.sum())
    assert score_report_path.exists()


def test_behavior_cli_commands_are_registered() -> None:
    parser = build_parser()
    train = parser.parse_args(
        ["train-kuairand-behavior", "train.csv", "valid.csv", "model.pt"]
    )
    score = parser.parse_args(
        [
            "score-kuairand-behavior",
            "model.pt",
            "slates.npz",
            "valid.csv",
            "scores.npz",
        ]
    )
    assert train.func.__name__ == "_command_train_kuairand_behavior"
    assert score.func.__name__ == "_command_score_kuairand_behavior"


def test_behavior_informed_sampling_cli_uses_aligned_scores(tmp_path: Path) -> None:
    n, k, m = 5, 3, len(KUAI_OBJECTIVES)
    slates = PreparedKuaiRandSlates(
        user_ids=np.arange(n, dtype=np.int64),
        item_ids=np.arange(n * k, dtype=np.int64).reshape(n, k) + 1,
        timestamps=np.arange(n * k, dtype=np.int64).reshape(n, k) * 100 + 1,
        labels=np.zeros((n, k, m), dtype=np.float32),
        mask=np.ones((n, k), dtype=bool),
        duration_ms=np.full((n, k), 10_000, dtype=np.float32),
        tab=np.ones((n, k), dtype=np.int16),
        is_random=np.zeros((n, k), dtype=np.int8),
        watch_ratio=np.zeros((n, k), dtype=np.float32),
        objective_names=KUAI_OBJECTIVES,
        item_prior_count=np.ones((n, k), dtype=np.int64),
        user_history_count=np.full(n, 5, dtype=np.int64),
    )
    probabilities = np.full((n, k, m), 0.05, dtype=np.float32)
    probabilities[:4, 0, :] = 0.95
    probabilities[4] = 0.5
    slates_path = tmp_path / "slates.npz"
    scores_path = tmp_path / "scores.npz"
    output_path = tmp_path / "sample.csv"
    report_path = tmp_path / "sample.report.json"
    slates.save(slates_path)
    SlateScoreTensor(
        slate_ids=np.arange(n, dtype=np.int64),
        candidate_ids=slates.item_ids.copy(),
        probabilities=probabilities,
        mask=slates.mask.copy(),
        objective_names=KUAI_OBJECTIVES,
    ).save(scores_path)

    parser = build_parser()
    args = parser.parse_args(
        [
            "sample-kuairand-jev",
            str(slates_path),
            str(output_path),
            "--max-slates",
            "1",
            "--per-cell",
            "1",
            "--max-per-user",
            "0",
            "--behavior-scores",
            str(scores_path),
            "--behavior-priority-strength",
            "1",
            "--report",
            str(report_path),
        ]
    )
    assert args.func(args) == 0
    selected = pd.read_csv(output_path)
    assert selected["slate_id"].tolist() == [4]
    assert report_path.exists()
