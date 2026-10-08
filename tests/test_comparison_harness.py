from dataclasses import replace
import math
import shutil

import numpy as np
import pytest
import torch

from paretojev.data.comparison import (
    ComparisonSlateDataset,
    load_comparison_data,
    write_synthetic_comparison_data,
)
from paretojev.evaluation.pareto import ParetoEvaluationConfig
from paretojev.experiments import comparison as runner
from paretojev.models.comparison import COMPARISON_METHODS
from paretojev.training.engine import PrecomputedTrainingConfig
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import sha256_file


@pytest.fixture
def manifest(tmp_path):
    return write_synthetic_comparison_data(
        tmp_path / "data", seed=23, sizes=(8, 6, 6), slate_size=4,
        n_objectives=2, feature_dim=4,
    )


def _rewrite_split(manifest, split, mutate, *, register_hash=True):
    registration = read_json(manifest)
    path = manifest.parent / registration["splits"][split]["path"]
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    if register_hash:
        registration["splits"][split]["sha256"] = sha256_file(path)
        atomic_write_json(manifest, registration)
    return path


def test_comparison_data_loads_shared_protocol_and_zeroes_padding(manifest):
    datasets, provenance = load_comparison_data(manifest)
    assert set(datasets) == {"train", "validation", "test"}
    for dataset in datasets.values():
        assert "target_probabilities" not in dataset.arrays
        assert dataset.metadata["directions"].tolist() == [1, -1]
        assert dataset.metadata["objective_names"].tolist() == ["objective_0", "objective_1"]
        for value in dataset.arrays.values():
            if value.ndim == 3:
                assert (value[~dataset.arrays["mask"]] == 0).all()
    assert provenance["manifest_sha256"] == sha256_file(manifest)


@pytest.mark.parametrize("key", ["objective_names", "directions", "feature_names", "feature_spec_hash"])
def test_comparison_rejects_cross_split_schema_drift(manifest, key):
    def mutate(arrays):
        if key == "feature_spec_hash":
            arrays[key] = np.asarray("f" * 64)
        else:
            arrays[key] = arrays[key][::-1]
    _rewrite_split(manifest, "validation", mutate)
    with pytest.raises(ValueError, match=key):
        load_comparison_data(manifest)


def test_comparison_rejects_hash_tampering_before_loading(manifest):
    _rewrite_split(manifest, "test", lambda arrays: arrays["labels"].fill(0), register_hash=False)
    with pytest.raises(ValueError, match="test file hash mismatch"):
        load_comparison_data(manifest)


def test_comparison_rejects_temporal_boundary_ties(manifest):
    datasets, _ = load_comparison_data(manifest)
    train = datasets["train"]
    last_time = train.metadata["timestamps"][train.arrays["mask"].numpy()].max()
    def mutate(arrays):
        arrays["timestamps"][0, 0] = last_time
    _rewrite_split(manifest, "validation", mutate)
    with pytest.raises(ValueError, match="chronological"):
        load_comparison_data(manifest)


@pytest.mark.parametrize("mutation, message", [
    (lambda a: a.pop("user_ids"), "metadata missing"),
    (lambda a: a["directions"].fill(0), "directions"),
    (lambda a: a["mask"].fill(False), "valid candidates"),
    (lambda a: a.update(mask=a["mask"].astype(np.int32)), "boolean"),
    (lambda a: a.update(target_probabilities=a["labels"].copy()), "privileged"),
    (lambda a: a["jev_scores"].fill(1.01), r"\[0,1\]"),
    (lambda a: a["candidate_features"].fill(float("nan")), "finite"),
])
def test_comparison_rejects_invalid_input_contract(manifest, mutation, message):
    path = _rewrite_split(manifest, "train", mutation)
    with pytest.raises(ValueError, match=message):
        ComparisonSlateDataset(path)


def test_comparison_padding_is_not_required_to_have_valid_probability_targets(manifest):
    def mutate(arrays):
        padding = ~arrays["mask"]
        for key in ("labels", "behavior_scores", "jev_scores", "jev_entropy", "jev_confidence"):
            arrays[key][padding] = 3.0
    _rewrite_split(manifest, "train", mutate)
    datasets, _ = load_comparison_data(manifest)
    assert (datasets["train"].arrays["labels"][~datasets["train"].arrays["mask"]] == 0).all()


def test_synthetic_generator_supports_minimum_two_candidate_slates(tmp_path):
    manifest = write_synthetic_comparison_data(
        tmp_path / "minimal", sizes=(2, 2, 2), slate_size=2,
        n_objectives=2, feature_dim=1,
    )
    datasets, _ = load_comparison_data(manifest)
    assert all((dataset.arrays["mask"].sum(dim=1) == 2).all() for dataset in datasets.values())


def test_real_data_requires_upstream_evidence_documents(manifest):
    for split in ("train", "validation", "test"):
        _rewrite_split(manifest, split, lambda arrays: arrays.update(data_kind=np.asarray("real")))
    registration = read_json(manifest)
    registration.update(data_kind="real", dataset="kuairand")
    atomic_write_json(manifest, registration)
    with pytest.raises(ValueError, match="upstream feature_spec evidence"):
        load_comparison_data(manifest)


def _small_configs():
    training = PrecomputedTrainingConfig(
        batch_size=4, epochs=2, learning_rate=0.002, hidden_dim=8, n_heads=2,
        n_layers=1, ff_dim=16, dropout=0, fusion_hidden_dim=4,
    )
    evaluation = ParetoEvaluationConfig(
        top_k=2, random_preferences=1, hypervolume_samples=128, seed=31,
    )
    return training, evaluation


@pytest.fixture
def frozen_source_identity(monkeypatch):
    # Isolate these reproducibility assertions from concurrently edited project
    # files; a dedicated test below changes this exact registered fingerprint.
    identity = {"source_sha256": "0" * 64, "files": {}, "git_dirty": False}
    monkeypatch.setattr(runner, "_source_identity", lambda: identity.copy())
    return identity


def test_harness_trains_each_model_reproducibly_and_resumes_without_retraining(
    manifest, tmp_path, monkeypatch, frozen_source_identity,
):
    training, evaluation = _small_configs()
    calls = {method: [] for method in COMPARISON_METHODS}
    original_loss = runner.comparison_training_loss
    def record_loss(method, model, batch, directions):
        first_parameter = next(model.parameters())
        calls[method].append((id(model), first_parameter.detach().clone()))
        return original_loss(method, model, batch, directions)
    monkeypatch.setattr(runner, "comparison_training_loss", record_loss)
    options = dict(training=training, evaluation=evaluation, seeds=(7,), bootstrap_samples=4)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = runner.run_comparison(manifest, first_dir, **options)
    per_method_steps = training.epochs * math.ceil(8 / training.batch_size)
    assert first["checks"]["all_models_trained"]
    assert first["checks"]["all_checkpoint_reload_equal"]
    assert first["evidence"]["data_kind"] == "synthetic"
    assert first["evidence"]["real_data"] is False
    # Every method really performs its own optimization, even if the common
    # validation criterion later selects its epoch-zero checkpoint.
    for method in COMPARISON_METHODS:
        assert len(calls[method]) == per_method_steps
        assert len({model_id for model_id, _ in calls[method]}) == 1
        assert not torch.equal(calls[method][0][1], calls[method][-1][1])
    assert len({calls[method][0][0] for method in COMPARISON_METHODS}) == 3
    second = runner.run_comparison(manifest, second_dir, **options)
    assert first == second
    assert set(first["methods"]) == set(COMPARISON_METHODS)
    for method in COMPARISON_METHODS:
        a, a_checkpoint = runner.load_comparison_checkpoint(first_dir / "seed-7" / f"{method}.pt")
        b, b_checkpoint = runner.load_comparison_checkpoint(second_dir / "seed-7" / f"{method}.pt")
        assert a_checkpoint["best_epoch"] == b_checkpoint["best_epoch"]
        assert a_checkpoint["run_id"] == b_checkpoint["run_id"] == first["run_id"]
        assert a_checkpoint["method"] == method
        assert sum(p.numel() for p in a.parameters()) == first["methods"][method]["parameters"]
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        a_metrics = read_json(first_dir / "seed-7" / f"{method}.training.json")
        b_metrics = read_json(second_dir / "seed-7" / f"{method}.training.json")
        assert a_metrics["history"] == b_metrics["history"]
        candidates = [(0, a_metrics["initial_validation_score"])] + [
            (row["epoch"], row["validation_selection_score"]) for row in a_metrics["history"]
        ]
        assert a_metrics["best_epoch"] == max(candidates, key=lambda row: row[1])[0]
    # Alter only held-out labels. Training and checkpoint selection must remain
    # identical; manifest identity and evaluation evidence correctly change.
    perturbed_data = tmp_path / "perturbed-data"
    shutil.copytree(manifest.parent, perturbed_data)
    perturbed_manifest = perturbed_data / "manifest.json"
    _rewrite_split(perturbed_manifest, "test", lambda arrays: arrays.update(labels=1 - arrays["labels"]))
    perturbed_output = tmp_path / "perturbed-run"
    perturbed = runner.run_comparison(perturbed_manifest, perturbed_output, **options)
    assert perturbed["run_id"] != first["run_id"]
    for method in COMPARISON_METHODS:
        a, _ = runner.load_comparison_checkpoint(first_dir / "seed-7" / f"{method}.pt")
        b, _ = runner.load_comparison_checkpoint(perturbed_output / "seed-7" / f"{method}.pt")
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        a_metrics = read_json(first_dir / "seed-7" / f"{method}.training.json")
        b_metrics = read_json(perturbed_output / "seed-7" / f"{method}.training.json")
        assert a_metrics["history"] == b_metrics["history"]
        assert a_metrics["best_epoch"] == b_metrics["best_epoch"]
    def forbid_retraining(*args, **kwargs):
        raise AssertionError("resume retrained a completed method")
    monkeypatch.setattr(runner, "_train_one", forbid_retraining)
    resumed = runner.run_comparison(manifest, first_dir, resume=True, **options)
    assert resumed == first
    with pytest.raises(ValueError, match="already registered"):
        runner.run_comparison(manifest, first_dir, **options)


def test_resume_rejects_config_source_and_artifact_tampering(manifest, tmp_path, monkeypatch, frozen_source_identity):
    training, evaluation = _small_configs()
    options = dict(training=training, evaluation=evaluation, seeds=(3,), bootstrap_samples=0)
    output = tmp_path / "run"
    runner.run_comparison(manifest, output, **options)
    with pytest.raises(ValueError, match="Resume refused"):
        runner.run_comparison(manifest, output, **{**options, "training": replace(training, epochs=3)}, resume=True)
    frozen_source_identity["source_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="Resume refused"):
        runner.run_comparison(manifest, output, **options, resume=True)
    frozen_source_identity["source_sha256"] = "0" * 64
    for suffix in ("pt", "training.json"):
        changed = tmp_path / f"tampered-{suffix}"
        shutil.copytree(output, changed)
        artifact = changed / "seed-3" / f"emer_like.{suffix}"
        artifact.write_bytes(artifact.read_bytes() + b" ")
        with pytest.raises(ValueError, match="artifact hash mismatch"):
            runner.run_comparison(manifest, changed, **options, resume=True)
        registry = read_json(changed / "registry.json")
        assert registry["status"] == "failed"
        assert registry["error"]["type"] == "ValueError"
    mismatched = tmp_path / "wrong-registration"
    shutil.copytree(output, mismatched)
    checkpoint_path = mismatched / "seed-3" / "emer_like.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["run_id"] = "wrong-run"
    torch.save(checkpoint, checkpoint_path)
    registry = read_json(mismatched / "registry.json")
    registry["completed"]["3/emer_like"]["checkpoint_sha256"] = sha256_file(checkpoint_path)
    atomic_write_json(mismatched / "registry.json", registry)
    with pytest.raises(ValueError, match="Checkpoint registration mismatch"):
        runner.run_comparison(manifest, mismatched, **options, resume=True)


def test_runner_rejects_invalid_seed_schedule_before_creating_output(manifest, tmp_path, frozen_source_identity):
    with pytest.raises(ValueError, match="seeds"):
        runner.run_comparison(manifest, tmp_path / "run", seeds=(7, 7))
    assert not (tmp_path / "run").exists()


def test_concurrent_same_output_is_locked_and_released_after_completion(tmp_path, monkeypatch):
    import threading

    entered = threading.Event()
    release = threading.Event()
    errors = []
    calls = []

    def hold_run(*args, **kwargs):
        calls.append(kwargs.get("resume", False))
        entered.set()
        if not release.wait(timeout=10):
            raise AssertionError("test did not release the running comparison")
        return {"status": "complete"}

    monkeypatch.setattr(runner, "_run_comparison", hold_run)
    output = tmp_path / "shared"

    def first_run():
        try:
            runner.run_comparison(tmp_path / "unused-manifest.json", output)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=first_run)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(ValueError, match="Another comparison process"):
            runner.run_comparison(tmp_path / "unused-manifest.json", output, resume=True)
        assert calls == [False]
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors
    assert runner.run_comparison(tmp_path / "unused-manifest.json", output, resume=True) == {"status": "complete"}
    assert calls == [False, True]


def test_midrun_source_drift_cannot_register_completed_checkpoint(
    manifest, tmp_path, monkeypatch, frozen_source_identity,
):
    training, evaluation = _small_configs()
    original = runner._train_one

    def change_source_after_training(*args, **kwargs):
        result = original(*args, **kwargs)
        frozen_source_identity["source_sha256"] = "1" * 64
        return result

    monkeypatch.setattr(runner, "_train_one", change_source_after_training)
    output = tmp_path / "source-drift"
    options = dict(training=training, evaluation=evaluation, seeds=(7,), bootstrap_samples=0)
    with pytest.raises(ValueError, match="Source changed during"):
        runner.run_comparison(manifest, output, **options)
    registry = read_json(output / "registry.json")
    assert registry["status"] == "failed"
    assert registry["completed"] == {}
    assert registry["error"]["type"] == "ValueError"
    assert not (output / "summary.json").exists()
    # Restoring the originally registered source allows a clean method restart;
    # the unregistered checkpoint written before the drift is never reused.
    frozen_source_identity["source_sha256"] = "0" * 64
    monkeypatch.setattr(runner, "_train_one", original)
    recovered = runner.run_comparison(manifest, output, resume=True, **options)
    assert recovered["checks"]["all_models_trained"]


def test_byte_metadata_is_decoded_before_model_and_json_use(manifest):
    import json

    def encode_metadata(arrays):
        for key in ("slate_ids", "user_ids", "candidate_ids", "feature_names", "feature_spec_hash", "data_kind"):
            arrays[key] = arrays[key].astype("S")
        arrays["objective_names"] = np.asarray([name.encode("utf-8") for name in ("点击", "风险")])

    for split in ("train", "validation", "test"):
        _rewrite_split(manifest, split, encode_metadata)
    datasets, _ = load_comparison_data(manifest)
    for data in datasets.values():
        assert data.metadata["objective_names"].tolist() == ["点击", "风险"]
        for key in ("slate_ids", "user_ids", "candidate_ids", "objective_names", "feature_names", "feature_spec_hash", "data_kind"):
            assert data.metadata[key].dtype.kind == "U"
        json.dumps({key: value.tolist() for key, value in data.metadata.items()}, allow_nan=False)


@pytest.mark.parametrize("key", ["objective_names", "feature_names", "feature_spec_hash", "data_kind"])
def test_numeric_text_metadata_is_rejected_before_training(manifest, key):
    def mutate(arrays):
        arrays[key] = np.arange(arrays[key].size).reshape(arrays[key].shape)

    path = _rewrite_split(manifest, "train", mutate)
    with pytest.raises(ValueError, match=f"{key} must contain text"):
        ComparisonSlateDataset(path)


@pytest.mark.parametrize("key", ["user_ids", "candidate_ids", "feature_names"])
def test_empty_byte_metadata_is_rejected(manifest, key):
    def mutate(arrays):
        arrays[key] = arrays[key].astype("S")
        arrays[key].flat[0] = b""

    path = _rewrite_split(manifest, "train", mutate)
    with pytest.raises(ValueError):
        ComparisonSlateDataset(path)
