"""Post-hoc tie sensitivity audit; does not replace registered primary metrics.

The reason for this diagnostic is the material change in Item-CF AUC after the
predeclared per-user candidate permutation, despite identical users/items/scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paretojev.data.microlens import grouped_score_arrays
from paretojev.metrics.ranking import per_user_auc, paired_bootstrap_difference
from paretojev.utils.io import atomic_write_json


def main() -> None:
    original = pd.read_csv("data/processed/microlens-real/candidates.csv", float_precision="round_trip")
    frame = pd.read_csv("artifacts/microlens-real-public-200/evaluated_candidates.csv", float_precision="round_trip")
    for column in ("itemcf", "y"):
        left = original.set_index(["user", "item"])[column].sort_index()
        right = frame.set_index(["user", "item"])[column].sort_index()
        if not left.index.equals(right.index) or not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise ValueError("User/item/label/Item-CF values differ; tie comparison is invalid")
    columns = ["itemcf", "jev_click", "hybrid_fixed_05", "hybrid_cv"]
    auc = {}
    for column in columns:
        _, scores, labels = grouped_score_arrays(frame, column)
        auc[column] = per_user_auc(scores, labels, tie_policy="stable")
    _, old_scores, old_labels = grouped_score_arrays(original, "itemcf")
    _, behavior, labels = grouped_score_arrays(frame, "itemcf")
    positive_scores = behavior[np.arange(len(behavior)), labels.argmax(axis=1)]
    ties = (behavior == positive_scores[:, None]).sum(axis=1) - 1
    subsets = {"with_positive_itemcf_ties": ties > 0, "without_positive_itemcf_ties": ties == 0,
               "positive_itemcf_zero": positive_scores == 0}
    report = {
        "diagnostic_only": True,
        "posthoc_reason": __doc__.strip(),
        "itemcf_scores_and_labels_identical_by_user_item": True,
        "auc_definition": "one credit when positive > negative; half credit for exact ties; no jitter",
        "auc_half_credit": {name: float(values.mean()) for name, values in auc.items()},
        "itemcf_original_order_auc_half_credit": float(per_user_auc(old_scores, old_labels, tie_policy="stable").mean()),
        "positive_itemcf_tie_count": int((ties > 0).sum()),
        "positive_itemcf_tied_negative_pairs": int(ties.sum()),
        "positive_itemcf_zero_count": int((positive_scores == 0).sum()),
        "paired_half_credit_auc_vs_itemcf": {}, "tie_groups": {},
    }
    for column in columns[1:]:
        difference, low, high = paired_bootstrap_difference(auc[column], auc["itemcf"], n_bootstrap=5000, seed=0)
        report["paired_half_credit_auc_vs_itemcf"][column] = {"difference": difference, "low": low, "high": high}
    for name, mask in subsets.items():
        report["tie_groups"][name] = {"users": int(mask.sum()),
            "itemcf_auc_half_credit": float(auc["itemcf"][mask].mean()) if mask.any() else None,
            "hybrid_cv_auc_half_credit": float(auc["hybrid_cv"][mask].mean()) if mask.any() else None}
    atomic_write_json("reports/microlens-real-200-2026-10-08/tie_diagnostic.json", report)
    import json
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
