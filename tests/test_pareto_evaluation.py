from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from paretojev.cli import build_parser
from paretojev.data.synthetic import SyntheticConfig, SyntheticSlateDataset, collate_slates
from paretojev.evaluation import (
    ParetoEvaluationConfig,
    build_preference_set,
    evaluate_pareto_model,
    objective_utility_metrics,
)
from paretojev.models.pareto_reranker import ParetoJevConfig, ParetoJevReranker


def _small_model(feature_dim: int = 6, n_objectives: int = 3) -> ParetoJevReranker:
    return ParetoJevReranker(
        ParetoJevConfig(
            candidate_feature_dim=feature_dim,
            n_objectives=n_objectives,
            hidden_dim=12,
            n_heads=3,
            n_layers=1,
            ff_dim=24,
            dropout=0.0,
            fusion_hidden_dim=8,
        )
    )


def test_preference_set_contains_uniform_one_hot_pairwise_and_random() -> None:
    config = ParetoEvaluationConfig(
        random_preferences=2,
        include_pairwise_preferences=True,
        seed=19,
    )
    preferences, names = build_preference_set(3, config)
    assert preferences.shape == (1 + 3 + 3 + 2, 3)
    assert names[:4] == ["uniform", "objective_0", "objective_1", "objective_2"]
    assert np.allclose(preferences.sum(axis=1), 1.0)
    assert np.all(preferences >= 0.0)


def test_objective_utility_metrics_aligns_maximize_and_minimize_targets() -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    labels = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    )
    mask = np.ones((2, 3), dtype=bool)
    values, coverage = objective_utility_metrics(
        scores,
        labels,
        mask,
        [1, -1],
        top_k=1,
    )
    assert np.allclose(values, [1.0, 1.0])
    assert np.allclose(coverage, [1.0, 1.0])

    cold = np.array([[True, False, False], [False, True, False]])
    cold_values, cold_coverage = objective_utility_metrics(
        scores,
        labels,
        mask,
        [1, -1],
        top_k=1,
        cold_mask=cold,
    )
    assert np.isclose(cold_values[0], 1.0)
    assert np.isnan(cold_values[1])
    assert np.isclose(cold_coverage[0], 1.0)


def test_evaluate_pareto_model_emits_target_namespace() -> None:
    dataset = SyntheticSlateDataset(
        SyntheticConfig(
            n_slates=12,
            slate_size=5,
            n_objectives=3,
            feature_dim=6,
            seed=23,
            hate_objective=2,
        )
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_slates)
    report = evaluate_pareto_model(
        _small_model(),
        loader,
        directions=[1, 1, -1],
        config=ParetoEvaluationConfig(
            top_k=2,
            random_preferences=2,
            include_pairwise_preferences=True,
            hypervolume_samples=2_000,
            seed=7,
        ),
        objective_names=["click", "like", "hate"],
    )

    target = report["kuairand"]["paretojev"]
    assert report["schema_version"] == "pareto-evaluation-v1"
    assert report["data"]["slates"] == 12
    assert set(report["methods"]) == {"paretojev", "behavior", "jev", "fused", "oracle"}
    assert np.isfinite(target["hypervolume_relative_gain"])
    assert np.isfinite(target["cold_item_ndcg_relative_gain"])
    assert np.isfinite(target["oracle_hypervolume_ratio"])
    assert target["oracle_hypervolume_ratio"] >= 0.0
    assert set(target["paired_dominance_rate"]) == {"behavior", "jev", "fused"}


def test_evaluate_pareto_cli_loads_checkpoint_and_writes_report(tmp_path: Path) -> None:
    dataset = SyntheticSlateDataset(
        SyntheticConfig(
            n_slates=10,
            slate_size=4,
            n_objectives=3,
            feature_dim=6,
            seed=29,
            hate_objective=2,
        )
    )
    arrays = {key: value.numpy() for key, value in dataset.arrays.items()}
    validation = tmp_path / "validation.npz"
    np.savez_compressed(validation, **arrays)

    model = _small_model()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "directions": (1, 1, -1),
            "objective_names": ("click", "like", "hate"),
        },
        checkpoint,
    )
    output = tmp_path / "pareto.json"

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-pareto",
            str(checkpoint),
            str(validation),
            "--batch-size",
            "5",
            "--top-k",
            "2",
            "--random-preferences",
            "1",
            "--hypervolume-samples",
            "1000",
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["checkpoint"]["directions_source"] == "checkpoint"
    assert report["objective_names"] == ["click", "like", "hate"]
    assert report["provenance"]["inputs"][0]["sha256"]
