from paretojev.training.engine import SyntheticSmokeConfig, run_synthetic_smoke


def test_synthetic_smoke_runs() -> None:
    result = run_synthetic_smoke(
        SyntheticSmokeConfig(
            n_train_slates=32,
            n_eval_slates=16,
            slate_size=6,
            n_objectives=3,
            feature_dim=6,
            batch_size=8,
            epochs=1,
        )
    )
    assert result["checks"]["finite_loss"]
    assert result["training"]["final_eval_loss"] > 0
