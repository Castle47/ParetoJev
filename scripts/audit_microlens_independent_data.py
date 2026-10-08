"""Post-evaluation identity/history checks and policy weak-order comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paretojev.data.microlens import load_pairs, load_prepared_candidates
from paretojev.experiments.merge import merge_decision_scores
from paretojev.experiments.tie_validation import policy_keys
from paretojev.utils.io import read_json, read_jsonl, atomic_write_json


def weak_order(keys):
    tied = np.ones((len(keys), len(keys)), dtype=bool)
    better = np.zeros_like(tied)
    for column in range(keys.shape[1]):
        better |= tied & (keys[:, None, column] > keys[None, :, column])
        tied &= keys[:, None, column] == keys[None, :, column]
    return better, tied


def main() -> None:
    root = Path("reports/microlens-independent-2026-10-08")
    registration = read_json(root / "registration.json")
    frame = load_prepared_candidates("data/processed/microlens-independent-200/candidates.csv")
    frame = merge_decision_scores(frame, read_jsonl("artifacts/microlens-real-independent-200/scores.jsonl"), objective="click", score_column="jev_click")
    equal = {"jev": 0, "title": 0}
    for _, group in frame.groupby("user", sort=False):
        keys = policy_keys(group.itemcf.to_numpy(), group.title_sim.to_numpy(), group.jev_click.to_numpy())
        for family in equal:
            a, b = weak_order(keys[family + "_ties"]), weak_order(keys[family + "_05"])
            equal[family] += int(all(np.array_equal(x, y) for x, y in zip(a, b, strict=True)))
    pairs = load_pairs("data/raw/microlens")
    violations = [int(user) for user, group in pairs[pairs.user.isin(frame.user.unique())].groupby("user")
                  if (group.timestamp.iloc[-6:-1] >= group.timestamp.iloc[-1]).any()]
    result = {"users": int(frame.user.nunique()), "rows": len(frame),
              "overlap_previous_evaluation_users": len(set(frame.user) & set(registration["excluded_previous_evaluation_users"])),
              "itemcf_excluded_users_count": len(registration["itemcf_training_excluded_users"]),
              "strict_history_violations": violations,
              "blank_candidate_titles": int(frame.title.fillna("").str.strip().eq("").sum()),
              "identical_full_weak_order_users_ties_vs_05": equal,
              "note": "Weak-order equivalence is a post-evaluation mechanism diagnostic, not an additional success test."}
    atomic_write_json(root / "data_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
