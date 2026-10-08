from pathlib import Path

import numpy as np

from paretojev.data.precomputed import PrecomputedSlateDataset
from paretojev.training.engine import PrecomputedTrainingConfig, run_precomputed_training


def _write_npz(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, k, f, m = 16, 5, 6, 3
    target = rng.uniform(0.05, 0.95, size=(n, k, m)).astype(np.float32)
    np.savez_compressed(
        path,
        candidate_features=rng.normal(size=(n, k, f)).astype(np.float32),
        behavior_scores=np.clip(target + rng.normal(0, 0.1, size=target.shape), 0, 1).astype(np.float32),
        jev_scores=np.clip(target + rng.normal(0, 0.1, size=target.shape), 0, 1).astype(np.float32),
        labels=rng.binomial(1, target).astype(np.float32),
        mask=np.ones((n, k), dtype=bool),
        jev_confidence=np.full((n, k, m), 0.7, dtype=np.float32),
        jev_entropy=np.full((n, k, m), 0.3, dtype=np.float32),
        popularity=rng.uniform(0, 1, size=(n, k)).astype(np.float32),
        history_length=rng.integers(1, 20, size=n).astype(np.float32),
        target_probabilities=target,
    )


def test_precomputed_dataset_and_training(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    valid = tmp_path / "valid.npz"
    _write_npz(train, 1)
    _write_npz(valid, 2)
    dataset = PrecomputedSlateDataset(train)
    assert dataset.feature_dim == 6
    assert dataset.n_objectives == 3
    result = run_precomputed_training(
        train,
        valid,
        directions=[1, 1, -1],
        config=PrecomputedTrainingConfig(
            batch_size=8,
            epochs=1,
            hidden_dim=24,
            n_heads=4,
            n_layers=1,
            ff_dim=48,
        ),
    )
    assert result["data"]["train_slates"] == 16
    assert np.isfinite(result["training"]["best_validation_loss"])
