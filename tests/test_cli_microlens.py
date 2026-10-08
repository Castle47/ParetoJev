from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paretojev.cli import build_parser


def test_evaluate_microlens_cli_emits_target_namespace(tmp_path: Path) -> None:
    rng = np.random.default_rng(31)
    rows: list[dict[str, float | int]] = []
    for user in range(12):
        slate_size = 5
        positive = user % slate_size
        itemcf = rng.normal(size=slate_size)
        jev_logits = rng.normal(size=slate_size)
        itemcf[positive] += 0.8
        jev_logits[positive] += 1.1
        exp = np.exp(jev_logits - jev_logits.max())
        jev = exp / exp.sum()
        for position in range(slate_size):
            rows.append(
                {
                    "user": user,
                    "item": user * 10 + position,
                    "position": position,
                    "y": int(position == positive),
                    "itemcf": float(itemcf[position]),
                    "jev_click": float(jev[position]),
                }
            )

    candidates = tmp_path / "candidates.csv"
    report_path = tmp_path / "report.json"
    evaluated_path = tmp_path / "evaluated.csv"
    pd.DataFrame(rows).to_csv(candidates, index=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-microlens",
            str(candidates),
            "--scores",
            "itemcf",
            "jev_click",
            "--choice-temperature",
            "jev_click",
            "--hybrid",
            "itemcf",
            "jev_click",
            "--folds",
            "3",
            "--weight-step",
            "0.5",
            "--reference",
            "itemcf",
            "--bootstrap-samples",
            "30",
            "--output",
            str(report_path),
            "--output-scores",
            str(evaluated_path),
        ]
    )
    assert args.func(args) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["microlens"] == report["metrics"]
    assert "hybrid_cv" in report["microlens"]
    assert "jev_click_temp_cv" in report["calibration"]
    assert "hybrid_cv" in report["pairwise_vs_reference"]
    assert evaluated_path.exists()
