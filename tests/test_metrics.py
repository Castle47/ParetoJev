import numpy as np

from paretojev.metrics.calibration import brier_score, expected_calibration_error
from paretojev.metrics.pareto import (
    controllability_spearman,
    hypervolume_2d,
    pareto_front_mask,
)
from paretojev.metrics.ranking import single_positive_metrics


def test_single_positive_metrics() -> None:
    scores = np.array([[0.1, 0.9, 0.2], [0.8, 0.2, 0.1]])
    labels = np.array([[0, 1, 0], [1, 0, 0]])
    metrics = single_positive_metrics(scores, labels, ks=(1, 2))
    assert metrics["auc"] == 1.0
    assert metrics["hr@1"] == 1.0
    assert metrics["ndcg@2"] == 1.0


def test_pareto_front_and_hypervolume() -> None:
    points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.0, 1.0]])
    mask = pareto_front_mask(points)
    assert mask.tolist() == [True, True, True, False]
    assert np.isclose(hypervolume_2d(points, np.array([0.0, 0.0])), 6.0)


def test_calibration_and_controllability() -> None:
    probabilities = np.array([0.1, 0.9])
    labels = np.array([0.0, 1.0])
    assert np.isclose(brier_score(probabilities, labels), 0.01)
    assert expected_calibration_error(probabilities, labels, n_bins=2) >= 0
    weights = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    achieved = weights.copy()
    result = controllability_spearman(weights, achieved)
    assert np.isclose(result["mean"], 1.0)
