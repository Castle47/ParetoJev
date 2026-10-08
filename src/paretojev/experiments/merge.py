from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def merge_decision_scores(
    candidates: pd.DataFrame,
    score_rows: Iterable[dict[str, Any]],
    *,
    objective: str,
    slate_column: str = "user",
    item_column: str = "item",
    score_column: str | None = None,
) -> pd.DataFrame:
    """Merge JSONL decision outputs back into candidate-level experiment rows."""

    score_column = score_column or f"jev_{objective}"
    lookup: dict[tuple[str, str], tuple[float, float, float]] = {}
    for row in score_rows:
        objective_names = [str(x) for x in row["objective_names"]]
        if objective not in objective_names:
            raise ValueError(f"Objective {objective!r} missing from score row")
        objective_index = objective_names.index(objective)
        candidate_ids = [str(x) for x in row["candidate_ids"]]
        probabilities = row["probabilities"]
        confidence = row["confidence"]
        entropy = row["entropy"]
        if not (len(candidate_ids) == len(probabilities) == len(confidence) == len(entropy)):
            raise ValueError("Decision score row has inconsistent candidate dimensions")
        slate = str(row.get("slate_id"))
        for index, candidate_id in enumerate(candidate_ids):
            lookup[(slate, candidate_id)] = (
                float(probabilities[index][objective_index]),
                float(confidence[index][objective_index]),
                float(entropy[index][objective_index]),
            )

    result = candidates.copy()
    keys = [
        (str(slate), str(item))
        for slate, item in zip(result[slate_column], result[item_column], strict=True)
    ]
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"Missing decision scores for {len(missing)} rows; first key: {missing[0]}")
    values = [lookup[key] for key in keys]
    result[score_column] = [value[0] for value in values]
    result[f"{score_column}_confidence"] = [value[1] for value in values]
    result[f"{score_column}_entropy"] = [value[2] for value in values]
    return result
