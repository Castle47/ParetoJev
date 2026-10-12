"""Registered, resumable equal-budget baseline experiments.

There is no hyperparameter search or test-driven retry in this runner. All seeds,
budgets and selection rules are fixed before training; failures remain recorded.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from paretojev.config import load_yaml
from paretojev.data.comparison import PROTOCOL_VERSION, load_comparison_data
from paretojev.data.synthetic import collate_slates
from paretojev.evaluation.pareto import ParetoEvaluationConfig, build_preference_set
from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.models.comparison import COMPARISON_LOSS_WEIGHTS, COMPARISON_METHODS, build_comparison_model, comparison_training_loss
from paretojev.models.pareto_reranker import ParetoJevConfig
from paretojev.training.engine import PrecomputedTrainingConfig
from paretojev.training.targets import evaluate_targets, target_report
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import canonical_hash, runtime_provenance, sha256_file
from paretojev.utils.seed import seed_everything


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(package)): sha256_file(p) for p in sorted(package.rglob("*.py"))}
    repo = package.parents[1]
    result = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False)
    return {"source_sha256": canonical_hash(files), "files": files,
            "git_dirty": bool(result.stdout.strip()) if result.returncode == 0 else None}


def _assert_source_unchanged(expected: str) -> None:
    if _source_identity()["source_sha256"] != expected:
        raise ValueError("Source changed during the experiment; use a frozen checkout and a new output directory")


@contextmanager
def _output_lock(output_dir: str | Path):
    """Process-scoped advisory lock; the OS releases it after a crash.

    Keep the sidecar inode in place after release: unlinking it lets waiting
    processes and new processes lock different inodes for the same run.
    """
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / f".{directory.name}.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Another comparison process owns this output directory") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another comparison process owns this output directory") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        # File handles avoid archive filename restrictions and give reproducible
        # serialization independent of the random temporary filename.
        with open(name, "wb") as handle:
            torch.save(payload, handle)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_comparison_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Not a comparison harness checkpoint")
    model = build_comparison_model(checkpoint["method"], ParetoJevConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def _validation_score(model, loader, directions, evaluation, device) -> float:
    """Common checkpoint criterion: preference-weighted observed utility.

    Each objective is normalized by the existing evaluator's fixed [0,1]
    definition. Missing validation objective coverage is an error, not a zero.
    """
    preferences, _ = build_preference_set(len(directions), evaluation)
    model.eval()
    rows = []
    for preference in preferences:
        scores, labels, masks = [], [], []
        for batch in loader:
            batch = batch.to(device)
            weights = torch.as_tensor(preference, device=device, dtype=batch.labels.dtype)
            output = model(replace(batch, preference_weights=weights.expand(batch.batch_size, -1)))
            scores.append(output["score_mean"].cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
            masks.append(batch.mask.cpu().numpy())
        points, coverage = comparison_objective_metrics(np.concatenate(scores), np.concatenate(labels),
                                                    np.concatenate(masks), directions, top_k=evaluation.top_k,
                                                    sample_weights=getattr(loader.dataset, "sample_weights", None))
        if not np.isfinite(points).all() or (coverage <= 0).any():
            raise ValueError("Validation must cover every registered objective")
        rows.append(float(np.dot(preference, points)))
    return float(np.mean(rows))


def _train_one(method, datasets, *, seed, training, evaluation, model_config, directory, run_id, directions):
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    model = build_comparison_model(method, model_config).to(training.device)
    train_loader = DataLoader(datasets["train"], batch_size=training.batch_size, shuffle=True,
                              collate_fn=collate_slates, generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(datasets["validation"], batch_size=training.batch_size, shuffle=False,
                              collate_fn=collate_slates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    initial = _validation_score(model, valid_loader, directions, evaluation, training.device)
    best_score, best_epoch = initial, 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = []
    start = time.perf_counter()
    for epoch in range(1, training.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(training.device)
            losses = comparison_training_loss(method, model, batch, directions=directions)
            loss = losses["total_loss"]
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {method} training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * batch.batch_size
            count += batch.batch_size
        score = _validation_score(model, valid_loader, directions, evaluation, training.device)
        history.append({"epoch": epoch, "train_loss": total / count, "validation_selection_score": score})
        if score > best_score:  # Exact ties retain the earliest checkpoint.
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = directory / f"{method}.pt"
    payload = {"protocol_version": PROTOCOL_VERSION, "run_id": run_id, "method": method,
               "seed": seed, "model_config": asdict(model_config), "model_state_dict": best_state,
               "directions": list(directions), "objective_names": datasets["train"].metadata["objective_names"].tolist(),
               "best_epoch": best_epoch}
    _save_checkpoint(checkpoint_path, payload)
    reloaded, _ = load_comparison_checkpoint(checkpoint_path, device=training.device)
    probe = next(iter(valid_loader)).to(training.device)
    with torch.no_grad():
        expected, actual = model(probe)["score_mean"], reloaded(probe)["score_mean"]
    if not torch.equal(expected, actual):
        raise ValueError(f"{method} checkpoint predictions differ after reload")
    metrics = {"method": method, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
               "epochs_executed": training.epochs, "best_epoch": best_epoch,
               "initial_validation_score": initial, "best_validation_score": best_score,
               "history": history, "elapsed_seconds": time.perf_counter() - start,
               "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_reload_equal": True}
    atomic_write_json(directory / f"{method}.training.json", metrics)
    return reloaded, metrics


def _mean_present(values):
    if not values or any(x is None or not np.isfinite(x) for x in values):
        return None
    return float(np.mean(values))


def run_comparison(manifest_path: str | Path, output_dir: str | Path, *,
                   training: PrecomputedTrainingConfig = PrecomputedTrainingConfig(),
                   evaluation: ParetoEvaluationConfig = ParetoEvaluationConfig(),
                   seeds: tuple[int, ...] = (7, 17, 29), bootstrap_samples: int = 200,
                   target_config_path: str | Path | None = None, resume: bool = False,
                   cpu_threads: int = 1) -> dict[str, Any]:
    with _output_lock(output_dir):
        return _run_comparison(manifest_path, output_dir, training=training,
            evaluation=evaluation, seeds=seeds, bootstrap_samples=bootstrap_samples,
            target_config_path=target_config_path, resume=resume, cpu_threads=cpu_threads)


def _run_comparison(manifest_path: str | Path, output_dir: str | Path, *,
                    training: PrecomputedTrainingConfig,
                    evaluation: ParetoEvaluationConfig,
                    seeds: tuple[int, ...], bootstrap_samples: int,
                    target_config_path: str | Path | None, resume: bool,
                    cpu_threads: int) -> dict[str, Any]:
    from paretojev.evaluation.comparison import evaluate_comparison

    if not seeds or len(set(seeds)) != len(seeds) or any(type(x) is not int or x < 0 or x >= 2**32 for x in seeds):
        raise ValueError("seeds must be unique integers in [0,2**32)")
    if training.epochs <= 0 or training.batch_size <= 0 or cpu_threads <= 0 or bootstrap_samples < 0:
        raise ValueError("epochs, batch_size, cpu_threads must be positive; bootstrap_samples nonnegative")
    if not np.isfinite(training.learning_rate) or training.learning_rate <= 0 or not np.isfinite(training.weight_decay) or training.weight_decay < 0:
        raise ValueError("Invalid optimizer parameters")
    # Seed is controlled solely by the registered seed list, not two conflicting knobs.
    training = replace(training, seed=0)
    datasets, data_info = load_comparison_data(manifest_path)
    train = datasets["train"]
    directions = tuple(int(x) for x in train.metadata["directions"])
    model_config = ParetoJevConfig(train.feature_dim, train.n_objectives,
                                  hidden_dim=training.hidden_dim, n_heads=training.n_heads,
                                  n_layers=training.n_layers, ff_dim=training.ff_dim,
                                  dropout=training.dropout, fusion_hidden_dim=training.fusion_hidden_dim)
    cold_cut = float(np.quantile(train.arrays["popularity"].numpy()[train.arrays["mask"].numpy()], evaluation.cold_quantile))
    provenance = runtime_provenance(repository=Path(__file__).resolve().parents[3])
    source = _source_identity()
    targets = load_yaml(target_config_path) if target_config_path else None
    registration = {
        "protocol_version": PROTOCOL_VERSION, "data": data_info,
        "training": asdict(training), "evaluation": asdict(evaluation), "seeds": list(seeds),
        "methods": list(COMPARISON_METHODS), "model_config": asdict(model_config),
        "loss_weights": COMPARISON_LOSS_WEIGHTS,
        "selection": "maximum validation preference-weighted utility; earliest tie; includes epoch zero",
        "budget": "equal epochs, data order, AdamW settings and clipping; no hyperparameter search",
        "deterministic_algorithms": "required; unsupported nondeterministic kernels fail",
        "comparison_baseline": "emer_like", "cold_popularity_threshold_from_train": cold_cut,
        "required_comparators": ["emer_like", "cmr_like"],
        "bootstrap_samples": bootstrap_samples, "cpu_threads": cpu_threads,
        "source_sha256": source["source_sha256"], "packages": provenance["packages"],
        "python": provenance["python"], "platform": provenance["platform"], "targets": targets,
        "aggregation": "arithmetic mean across all preregistered seeds; missing stays null",
    }
    run_id = canonical_hash(registration)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "registry.json"
    if registry_path.exists():
        if not resume:
            raise ValueError("Output already registered; use --resume or a new output directory")
        registry = read_json(registry_path)
        if registry.get("run_id") != run_id or registry.get("registration") != registration:
            raise ValueError("Resume refused: source, inputs, environment, targets or configuration changed")
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Output directory must be empty for a new experiment")
        registry = {"run_id": run_id, "registration": registration, "provenance": provenance,
                    "source": source, "created_at": _now(), "status": "registered", "completed": {}}
        atomic_write_json(registry_path, registry)
    evidence = {"data_kind": data_info["manifest"]["data_kind"],
                "real_data": data_info["manifest"]["data_kind"] == "real",
                "dataset": data_info["manifest"]["dataset"], "protocol_version": PROTOCOL_VERSION,
                "comparison_baseline": "emer_like", "provenance_basis": "validated input manifest; upstream evidence remains auditable"}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        registry.update(status="running", updated_at=_now())
        registry.pop("error", None)
        atomic_write_json(registry_path, registry)
        results = []
        for seed in seeds:
            directory = output_dir / f"seed-{seed}"
            directory.mkdir(exist_ok=True)
            models, training_reports = {}, {}
            for method in COMPARISON_METHODS:
                key = f"{seed}/{method}"
                checkpoint_path = directory / f"{method}.pt"
                metrics_path = directory / f"{method}.training.json"
                saved = registry["completed"].get(key)
                if saved:
                    if sha256_file(checkpoint_path) != saved["checkpoint_sha256"] or sha256_file(metrics_path) != saved["metrics_sha256"]:
                        raise ValueError(f"Resume artifact hash mismatch: {key}")
                    model, checkpoint = load_comparison_checkpoint(checkpoint_path, device=training.device)
                    if checkpoint["run_id"] != run_id or checkpoint["method"] != method or checkpoint["seed"] != seed:
                        raise ValueError(f"Checkpoint registration mismatch: {key}")
                    train_report = read_json(metrics_path)
                else:
                    model, train_report = _train_one(method, datasets, seed=seed, training=training,
                        evaluation=evaluation, model_config=model_config, directory=directory,
                        run_id=run_id, directions=directions)
                    _assert_source_unchanged(source["source_sha256"])
                    registry["completed"][key] = {"checkpoint_sha256": sha256_file(checkpoint_path),
                                                  "metrics_sha256": sha256_file(metrics_path)}
                    registry["updated_at"] = _now()
                    atomic_write_json(registry_path, registry)
                models[method], training_reports[method] = model, train_report
            loader = DataLoader(datasets["test"], batch_size=training.batch_size, shuffle=False, collate_fn=collate_slates)
            report = evaluate_comparison(models, loader, directions=directions,
                objective_names=train.metadata["objective_names"].tolist(), config=evaluation,
                cold_popularity_threshold=cold_cut, device=training.device,
                group_ids=datasets["test"].metadata["user_ids"].tolist(),
                bootstrap_samples=bootstrap_samples, audit_path=directory / "predictions.npz")
            # Row/candidate identity accompanies predictions but never model features.
            np.savez_compressed(directory / "prediction_identity.npz", **datasets["test"].metadata)
            report.update(seed=seed, evidence=evidence, training=training_reports, run_id=run_id)
            _assert_source_unchanged(source["source_sha256"])
            if targets:
                report["target_report"] = target_report(evaluate_targets(report, targets))
            json.dumps(report, allow_nan=False)
            atomic_write_json(directory / "evaluation.json", report)
            results.append(report)
        target_keys = ("hypervolume", "hypervolume_relative_gain", "cold_item_ndcg_relative_gain",
                       "worst_objective_regret", "controllability_spearman_mean")
        aggregate = {key: _mean_present([r["kuairand"]["paretojev"].get(key) for r in results]) for key in target_keys}
        aggregate_cmr = {key: _mean_present([r["kuairand"]["paretojev_vs_cmr"].get(key) for r in results]) for key in target_keys}
        methods = {}
        for method in COMPARISON_METHODS:
            values = [r["methods"][method].get("hypervolume") for r in results]
            methods[method] = {"hypervolume_mean": _mean_present(values),
                               "hypervolume_per_seed": values,
                               "parameters": results[0]["training"][method]["parameters"]}
        summary = {"schema_version": PROTOCOL_VERSION, "run_id": run_id, "status": "complete",
                   "evidence": evidence, "seeds": list(seeds), "methods": methods,
                   "kuairand": {"paretojev": aggregate, "paretojev_vs_cmr": aggregate_cmr},
                   "checks": {"all_models_trained": len(registry["completed"]) == len(seeds) * len(COMPARISON_METHODS),
                              "all_checkpoint_reload_equal": all(t["checkpoint_reload_equal"] for r in results for t in r["training"].values())},
                   "limitations": ["EMER-like/CMR-like are documented project variants, not paper reproductions.",
                       "Equal training budget does not imply equal parameter count or FLOPs.",
                       "Hypervolume uses Monte Carlo integration; near-threshold gains require higher precision and paired uncertainty analysis.",
                       "Mean seed metrics are descriptive point estimates, not proof of a statistically significant win."]}
        if targets:
            summary["target_report"] = target_report(evaluate_targets(summary, targets))
        json.dumps(summary, allow_nan=False)
        _assert_source_unchanged(source["source_sha256"])
        atomic_write_json(output_dir / "summary.json", summary)
        registry.update(status="complete", updated_at=_now(), summary_sha256=sha256_file(output_dir / "summary.json"))
        atomic_write_json(registry_path, registry)
        return summary
    except Exception as exc:
        registry.update(status="failed", updated_at=_now(), error={"type": type(exc).__name__, "message": str(exc)})
        atomic_write_json(registry_path, registry)
        raise
    finally:
        torch.set_num_threads(previous_threads)
