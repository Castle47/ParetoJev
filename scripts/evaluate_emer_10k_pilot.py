#!/usr/bin/env python3
"""Audit and evaluate the fixed 10K new-date pilot after complete real scoring."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deployment/autodl"))
sys.path.insert(0, str(REPO / "deployment/emer_candidate"))
from score_confirmation import canonical, validate_group
from candidate_runtime import Candidate
from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.experiments.comparison import _output_lock
from paretojev.training.behavior_calibration import apply_calibration
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file
from run_emer_jev import paired_report
from run_emer_balanced_transfer import screen
from run_emer_scale import immutable_json


SCORE_FILES = ("complete.json", "registration.json", "scores.npz", "native.sqlite3")


def require_scores(path):
    missing = list(SCORE_FILES) if path is None else [name for name in SCORE_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError("Complete pilot score directory required; missing: " + ", ".join(missing))
    complete = read_json(path / "complete.json")
    if (not complete.get("complete") or complete.get("public_cache_mixed") or
            complete.get("valid_decisions") != 70000 or complete.get("http_groups") != 10000):
        raise ValueError("All70000 real decisions/10000 groups are required")


def audit_scores(args, config, parent):
    complete = read_json(args.scores / "complete.json")
    registration = read_json(args.scores / "registration.json")
    transfer = read_json(args.pilot / "transfer.json")
    development = read_json(args.development / "jev-native-v9-fp16/registration.json")
    if complete["registration_sha256"] != sha256_file(args.scores / "registration.json") or complete["scores_sha256"] != sha256_file(args.scores / "scores.npz"):
        raise ValueError("Pilot score artifacts changed")
    if registration["source"] != development["source"] or not registration["test_opened"]:
        raise ValueError("Pilot producer differs from frozen development producer")
    if registration["development_registration_sha256"] != sha256_file(args.development / "jev-native-v9-fp16/registration.json"):
        raise ValueError("Development producer receipt differs")
    if registration["script_sha256"] != sha256_file(REPO / "deployment/autodl/score_pilot.py") or registration["collector_library_sha256"] != sha256_file(REPO / "deployment/autodl/score_confirmation.py"):
        raise ValueError("Pilot collector implementation differs")
    if (registration["slates"] != 10000 or registration["confirmation_config_sha256"] != sha256_file(args.config) or
            registration["transfer_manifest_sha256"] != sha256_file(args.pilot / "transfer.json")):
        raise ValueError("Pilot scoring registration differs")
    if transfer["confirmation_config_sha256"] != sha256_file(args.config) or not transfer["pilot_subset"] or transfer["slates"] != 10000:
        raise ValueError("Pilot transfer scope differs")
    compressed = args.pilot / "jev_input.jsonl.gz"
    if registration["input_sha256"] != transfer["compressed_sha256"] or sha256_file(compressed) != transfer["compressed_sha256"]:
        raise ValueError("Pilot compressed input differs")
    with gzip.open(compressed, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != transfer["source_input_sha256"]:
            raise ValueError("Decoded pilot input differs")
    with (args.pilot / "jev_input.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    if (len(rows) != 10000 or any(row["slate_id"] != i or row["split"] != "test" or
            row["questions"] != parent["questions"] for i, row in enumerate(rows))):
        raise ValueError("Pilot prompt rows differ")
    with np.load(args.scores / "scores.npz", allow_pickle=False) as saved:
        scores = {key: saved[key] for key in saved.files}
    probabilities = scores["probabilities"]
    if (probabilities.shape != (10000, 10, 7) or not np.isfinite(probabilities).all() or
            (probabilities < 0).any() or (probabilities > 1).any() or
            not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-5)):
        raise ValueError("Invalid pilot categorical probabilities")
    if scores["objective_names"].tolist() != config["objectives"] or not np.array_equal(scores["slate_ids"], np.arange(10000)):
        raise ValueError("Pilot score row/goal order differs")
    if str(scores["model"]) != config["jev"]["model"] or str(scores["revision"]) != config["jev"]["revision"]:
        raise ValueError("Pilot tensor producer differs")
    entropy = np.clip(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum(1) / np.log(10), 0, 1)
    for name, expected in (("entropy", np.broadcast_to(entropy[:, None, :], probabilities.shape)),
                           ("confidence", np.broadcast_to(1 - entropy[:, None, :], probabilities.shape))):
        if scores[name].shape != probabilities.shape or not np.allclose(scores[name], expected, rtol=0, atol=1e-10):
            raise ValueError("Pilot signal summaries differ")
    with sqlite3.connect("file:" + str((args.scores / "native.sqlite3").resolve()) + "?mode=ro", uri=True) as database:
        if database.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Pilot score database integrity failed")
        children = iter(database.execute("SELECT row_id,goal,payload_hash,raw_json FROM results ORDER BY row_id,goal"))
        groups = decisions = 0
        for row_id, key, text in database.execute("SELECT row_id,payload_hash,raw_json FROM groups ORDER BY row_id"):
            if row_id != groups:
                raise ValueError("Pilot group coverage/order differs")
            row = rows[row_id]
            if not np.array_equal(scores["candidate_ids"][row_id], row["candidate_ids"]):
                raise ValueError("Pilot candidate IDs differ")
            wire = {"kind": "choice", "state": row["state"], "questions": row["questions"], "options": row["options"], "thinking": "off"}
            if key != hashlib.sha256(canonical(wire).encode()).hexdigest():
                raise ValueError("Pilot group request hash differs")
            group = json.loads(text)
            values = validate_group(group, row)
            for goal, value in enumerate(values):
                saved = next(children, None)
                payload = {"kind": "choice", "state": row["state"], "question": row["questions"][goal], "options": row["options"], "thinking": "off"}
                if (saved is None or saved[:2] != (row_id, goal) or saved[2] != hashlib.sha256(canonical(payload).encode()).hexdigest() or
                        json.loads(saved[3]) != value or not np.array_equal(value["probabilities"], probabilities[row_id, :, goal])):
                    raise ValueError("Pilot child response differs")
                decisions += 1
            groups += 1
        if groups != 10000 or decisions != 70000 or next(children, None) is not None:
            raise ValueError("Pilot raw response coverage differs")
    receipt = {
        "groups": groups,
        "decisions": decisions,
        "exact_http_response_alignment": True,
        "scores_sha256": complete["scores_sha256"],
        "database_sha256": sha256_file(args.scores / "native.sqlite3"),
        "producer_id": registration["source"]["producer_id"]
    }
    immutable_json(args.output / "response_audit.json", receipt)
    return scores


def prepare(args, config):
    candidate = Candidate(args.bundle)
    if sha256_file(args.bundle / "manifest.json") != config["candidate_manifest_sha256"]:
        raise ValueError("Pilot candidate manifest differs")
    required = {
        str(args.config): sha256_file(args.config),
        str(args.pilot / "registration.json"): sha256_file(args.pilot / "registration.json"),
        str(args.pilot / "complete.json"): sha256_file(args.pilot / "complete.json"),
        str(args.pilot / "selection.npz"): sha256_file(args.pilot / "selection.npz"),
        str(args.pilot / "raw_slates.npz"): sha256_file(args.pilot / "raw_slates.npz"),
        str(args.pilot / "transfer.json"): sha256_file(args.pilot / "transfer.json"),
        str(args.pilot / "jev_input.jsonl.gz"): sha256_file(args.pilot / "jev_input.jsonl.gz"),
        str(args.bundle / "manifest.json"): sha256_file(args.bundle / "manifest.json"),
        str(REPO / "deployment/autodl/score_pilot.py"): sha256_file(REPO / "deployment/autodl/score_pilot.py"),
        str(REPO / "deployment/autodl/score_confirmation.py"): sha256_file(REPO / "deployment/autodl/score_confirmation.py"),
        str(Path(__file__)): sha256_file(Path(__file__))
    }
    with np.load(args.pilot / "selection.npz") as saved:
        pilot = saved["pilot_original_slate_ids"]
        heldout = saved["heldout_original_slate_ids"]
    if len(pilot) != 10000 or len(heldout) != 40000 or len(np.intersect1d(pilot, heldout)):
        raise ValueError("Pilot/heldout split differs")
    with np.load(args.pilot / "raw_slates.npz") as raw:
        users = raw["user"]
        features = raw["x"]
    routes = candidate.predict if False else None
    preflight = {
        "slates": 10000,
        "users": int(len(np.unique(users))),
        "raw_feature_shape": list(features.shape),
        "pilot_original_id_sha256": hashlib.sha256(pilot.tobytes()).hexdigest(),
        "heldout_original_id_sha256": hashlib.sha256(heldout.tobytes()).hexdigest(),
        "labels_or_model_outputs_used_for_selection": False,
        "model_performance_computed": False
    }
    immutable_json(args.output / "input_preflight.json", preflight)
    required[str(args.output / "input_preflight.json")] = sha256_file(args.output / "input_preflight.json")
    immutable_json(args.output / "registration.json", {
        "config": config,
        "inputs": required,
        "candidate_manifest_sha256": config["candidate_manifest_sha256"],
        "pilot_and_heldout_frozen_before_scoring": True,
        "new_training": False,
        "model_performance_computed": False
    })
    missing = list(SCORE_FILES) if args.scores is None else [name for name in SCORE_FILES if not (args.scores / name).is_file()]
    atomic_write_json(args.output / "readiness.json", {
        "local_preflight_passed": True,
        "score_artifacts_missing": missing,
        "required_real_groups": 10000,
        "required_real_decisions": 70000,
        "pilot_metrics_computed": False
    })
    print(read_json(args.output / "readiness.json"), flush=True)


def verify(args, config):
    registration = read_json(args.output / "registration.json")
    if registration["config"] != config:
        raise ValueError("Pilot evaluation config changed")
    for path, expected in registration["inputs"].items():
        if sha256_file(path) != expected:
            raise ValueError("Pilot registered input/source changed")
    if sha256_file(args.bundle / "manifest.json") != config["candidate_manifest_sha256"]:
        raise ValueError("Pilot candidate changed")
    return Candidate(args.bundle)


def evaluate(args, config):
    require_scores(args.scores)
    candidate = verify(args, config)
    parent = read_json(args.parent_config)
    scores = audit_scores(args, config, parent)
    with np.load(args.pilot / "raw_slates.npz", allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    behavior_path = args.root / "behavior-v2/run-v2/upload_metadata_assumed_static.joblib"
    if sha256_file(behavior_path) != candidate.manifest["upstream"]["behavior_checkpoint_sha256"]:
        raise ValueError("Pilot behavior model changed")
    tree = joblib.load(behavior_path)
    flat = raw["x"].reshape(-1, 26)
    behavior = apply_calibration(np.column_stack([model.predict_proba(flat)[:, 1] for model in tree["models"]]), tree["calibration"]).reshape(-1, 10, 7)
    inputs = {
        "raw_features": raw["x"],
        "behavior_scores": behavior.astype("float32"),
        "jev_scores": scores["probabilities"].astype("float32"),
        "jev_entropy": scores["entropy"].astype("float32"),
        "jev_confidence": scores["confidence"].astype("float32"),
        "user_ids": raw["user"],
        "candidate_ids": raw["items"],
        "mask": np.ones(raw["items"].shape, dtype=bool),
        "popularity": raw["popularity"],
        "history_length": raw["history_count"]
    }
    predictions = {}
    route_ids = None
    for seed in config["training"]["seeds"]:
        for arm in ("emer", "emer_jev", "anchor"):
            result = candidate.predict(inputs, arm=arm, seed=seed)
            predictions[f"{arm}/{seed}"] = result["scores"]
            route_ids = result["fold_ids"]
            print("Pilot predicted:", arm, seed, flush=True)
    data = SimpleNamespace(
        arrays={"labels": torch.from_numpy(raw["labels"].astype("float32")), "mask": torch.ones(raw["items"].shape, dtype=torch.bool)},
        users=raw["user"], sample_weights=raw["weight"]
    )
    metric_config = dict(config, evaluation=dict(config["evaluation"], click_maximum_absolute_ndcg_drop=.005, hate_maximum_absolute_risk_increase=.005))
    comparisons = {"signal": ("emer", "emer_jev"), "original_reference": ("anchor", "emer_jev"), "training_control": ("anchor", "emer")}
    def compare(values, subset):
        reports = {}
        for name, (base, full) in comparisons.items():
            pair = {f"{arm}/{seed}": values[f"{source}/{seed}"] for arm, source in (("emer", base), ("emer_jev", full)) for seed in config["training"]["seeds"]}
            report = paired_report(pair, subset, metric_config)
            report.pop("gates")
            report.pop("all_effect_gates")
            reports[name] = report
        return reports
    pooled = compare(predictions, data)
    by_route = {name: {} for name in comparisons}
    for route in range(3):
        use = route_ids == route
        subset = SimpleNamespace(arrays={key: value[use] for key, value in data.arrays.items()}, users=data.users[use], sample_weights=data.sample_weights[use])
        for name, report in compare({key: value[use] for key, value in predictions.items()}, subset).items():
            by_route[name][str(route)] = report
    conditions = {name: screen(pooled[name], by_route[name], metric_config) for name in config["evaluation"]["required_comparisons"]}
    labels = raw["labels"]
    days = (raw["decision"] + 8 * 3600000) // 86400000
    result = {
        "role": config["role"],
        "pooled": pooled,
        "by_route": by_route,
        "conditions": conditions,
        "screen_passed": all(all(values.values()) for values in conditions.values()),
        "support": {
            "slates": 10000,
            "users": int(len(np.unique(raw["user"]))),
            "candidate_exposures": 100000,
            "positive_exposures": labels.sum((0, 1)).astype(int).tolist(),
            "positive_slates": (labels.sum(1) > 0).sum(0).tolist(),
            "weight_kish_effective_slates": float(raw["weight"].sum() ** 2 / np.square(raw["weight"]).sum())
        },
        "dates": [str(np.datetime64(int(day), "D")) for day in np.unique(days)],
        "candidate_manifest_sha256": config["candidate_manifest_sha256"],
        "config_sha256": sha256_file(args.config),
        "response_audit_sha256": sha256_file(args.output / "response_audit.json"),
        "remaining_heldout_slates_not_evaluated": 40000,
        "limitations": config["limitations"],
        "online_business_success": False
    }
    if args.stage == "replay":
        with np.load(args.output / "predictions.npz") as saved:
            if set(saved.files) != set(predictions) or any(not np.array_equal(saved[key], value) for key, value in predictions.items()):
                raise ValueError("Pilot prediction replay failed")
        if read_json(args.output / "evaluation.json") != result:
            raise ValueError("Pilot metric replay failed")
        atomic_write_json(args.output / "replay.json", {
            "exact_predictions_and_metrics": True,
            "evaluation_sha256": sha256_file(args.output / "evaluation.json"),
            "predictions_sha256": sha256_file(args.output / "predictions.npz"),
            "all_raw_responses_audited_again": True
        })
    else:
        if (args.output / "evaluation.json").exists():
            raise ValueError("Use replay for an existing pilot evaluation")
        np.savez_compressed(args.output / "predictions.npz", **predictions)
        immutable_json(args.output / "evaluation.json", result)
    print("Pilot conditions:", conditions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/emer_10k_pilot_v1.json")
    parser.add_argument("--parent-config", type=Path, default=REPO / "configs/emer_multiobjective_confirmation.json")
    parser.add_argument("--stage", choices=["prepare", "evaluate", "replay"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    with _output_lock(args.output), threadpool_limits(limits=config["training"]["cpu_threads"]):
        prepare(args, config) if args.stage == "prepare" else evaluate(args, config)
