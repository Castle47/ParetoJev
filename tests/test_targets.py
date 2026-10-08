from paretojev.training.targets import evaluate_targets, target_report


def test_target_checker() -> None:
    metrics = {"training": {"loss_reduction_fraction": 0.1}, "fusion": {"brier": 0.02}}
    config = {
        "targets": {
            "loss": {
                "metric": "training.loss_reduction_fraction",
                "op": ">=",
                "threshold": 0.05,
            },
            "brier": {
                "metric": "fusion.brier",
                "op": "<=",
                "threshold": 0.03,
            },
        }
    }
    report = target_report(evaluate_targets(metrics, config))
    assert report["passed"]
