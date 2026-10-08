from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from paretojev.data.behavior_sequences import (
    BehaviorContextSpec,
    ItemVocabulary,
    KuaiRandBehaviorDataset,
    UserHistoryIndex,
    behavior_context_for_slates,
    build_behavior_event_table,
    fit_behavior_context_spec,
)
from paretojev.data.kuairand import PreparedKuaiRandSlates, read_kuairand_log
from paretojev.data.materialize import SlateScoreTensor
from paretojev.models.behavior import SASRecMMoEConfig, SASRecMMoERanker
from paretojev.utils.io import atomic_write_json
from paretojev.utils.seed import seed_everything


@dataclass(frozen=True)
class BehaviorTrainingConfig:
    seed: int = 42
    batch_size: int = 256
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_history_length: int = 50
    embedding_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    n_experts: int = 4
    expert_dim: int = 64
    dropout: float = 0.1
    max_tab_values: int = 16
    device: str = "cpu"
    num_workers: int = 0
    max_train_examples: int | None = None
    max_validation_examples: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.max_history_length <= 0:
            raise ValueError("max_history_length must be positive")
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        for name in ("max_train_examples", "max_validation_examples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")


def _deterministic_subset(
    dataset: Dataset[dict[str, torch.Tensor]],
    maximum: int | None,
    *,
    seed: int,
) -> Dataset[dict[str, torch.Tensor]]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=maximum, replace=False))
    return Subset(dataset, indices.tolist())


def _batch_loss(model: SASRecMMoERanker, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        batch["history_item_ids"],
        batch["candidate_item_id"][:, None],
        batch["candidate_context"][:, None, :],
    ).squeeze(1)
    per_objective = F.binary_cross_entropy_with_logits(
        logits,
        batch["labels"],
        reduction="none",
    ).mean(dim=0)
    # Every objective contributes equally. We intentionally avoid class-weighted
    # BCE here because its sigmoid output is not an absolute probability without
    # a separate leakage-safe calibration split.
    return per_objective.mean()


def _evaluate_behavior(
    model: SASRecMMoERanker,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    objective_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    loss_sum = 0.0
    loss_elements = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            logits = model(
                batch["history_item_ids"],
                batch["candidate_item_id"][:, None],
                batch["candidate_context"][:, None, :],
            ).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["labels"], reduction="sum"
            )
            loss_sum += float(loss.item())
            loss_elements += int(batch["labels"].numel())
            logits_rows.append(logits.cpu().numpy())
            label_rows.append(batch["labels"].cpu().numpy())
    logits_array = np.concatenate(logits_rows)
    labels = np.concatenate(label_rows)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -40.0, 40.0)))
    per_objective: dict[str, dict[str, float | None]] = {}
    valid_auc_values: list[float] = []
    valid_ap_values: list[float] = []
    for index, name in enumerate(objective_names):
        y = labels[:, index]
        p = probabilities[:, index]
        if len(np.unique(y)) >= 2:
            auc: float | None = float(roc_auc_score(y, p))
            ap: float | None = float(average_precision_score(y, p))
        else:
            auc = None
            ap = None
        if auc is not None:
            valid_auc_values.append(auc)
        if ap is not None:
            valid_ap_values.append(ap)
        per_objective[name] = {
            "prevalence": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "auc": auc,
            "average_precision": ap,
        }
    return {
        "bce": float(loss_sum / max(loss_elements, 1)),
        "overall_brier": float(np.mean((probabilities - labels) ** 2)),
        "macro_auc": (
            float(np.mean(valid_auc_values)) if valid_auc_values else None
        ),
        "macro_average_precision": (
            float(np.mean(valid_ap_values)) if valid_ap_values else None
        ),
        "per_objective": per_objective,
        "examples": int(len(labels)),
    }


def run_kuairand_behavior_training(
    train_log_path: str | Path,
    validation_log_path: str | Path,
    *,
    config: BehaviorTrainingConfig = BehaviorTrainingConfig(),
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train SASRec+MMoE on chronological KuaiRand exposure events."""

    seed_everything(config.seed)
    train_log = read_kuairand_log(train_log_path)
    validation_log = read_kuairand_log(validation_log_path)
    train_table = build_behavior_event_table(train_log)
    validation_table = build_behavior_event_table(
        validation_log,
        prefix_log=train_log,
    )
    vocabulary = ItemVocabulary.fit(train_log["video_id"].to_numpy(dtype=np.int64))
    context_spec = fit_behavior_context_spec(
        train_table,
        max_tab_values=config.max_tab_values,
    )
    train_full = KuaiRandBehaviorDataset(
        train_table,
        vocabulary,
        context_spec,
        max_history_length=config.max_history_length,
    )
    validation_full = KuaiRandBehaviorDataset(
        validation_table,
        vocabulary,
        context_spec,
        max_history_length=config.max_history_length,
    )
    train_dataset = _deterministic_subset(
        train_full, config.max_train_examples, seed=config.seed
    )
    validation_dataset = _deterministic_subset(
        validation_full, config.max_validation_examples, seed=config.seed + 1
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model_config = SASRecMMoEConfig(
        num_items=vocabulary.num_items,
        n_objectives=len(train_table.objective_names),
        max_history_length=config.max_history_length,
        embedding_dim=config.embedding_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ff_dim=config.ff_dim,
        n_experts=config.n_experts,
        expert_dim=config.expert_dim,
        context_dim=context_spec.feature_dim,
        dropout=config.dropout,
        padding_idx=vocabulary.padding_idx,
    )
    device = torch.device(config.device)
    model = SASRecMMoERanker(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    initial_validation = _evaluate_behavior(
        model,
        validation_loader,
        objective_names=train_table.objective_names,
        device=device,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        for raw_batch in train_loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            loss = _batch_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation = _evaluate_behavior(
            model,
            validation_loader,
            objective_names=train_table.objective_names,
            device=device,
        )
        validation_loss = float(validation["bce"])
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_bce": float(np.mean(train_losses)),
                "validation_bce": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Behavior training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    final_validation = _evaluate_behavior(
        model,
        validation_loader,
        objective_names=train_table.objective_names,
        device=device,
    )

    validation_raw_items = validation_log["video_id"].to_numpy(dtype=np.int64)
    validation_mapped = vocabulary.encode(validation_raw_items)
    result: dict[str, Any] = {
        "status": "complete",
        "train_log": str(train_log_path),
        "validation_log": str(validation_log_path),
        "config": asdict(config),
        "model_config": asdict(model_config),
        "objective_names": list(train_table.objective_names),
        "data": {
            "train_events_total": len(train_full),
            "validation_events_total": len(validation_full),
            "train_events_used": len(train_dataset),
            "validation_events_used": len(validation_dataset),
            "vocabulary_items": len(vocabulary.raw_item_ids),
            "validation_oov_fraction": float(
                np.mean(validation_mapped == vocabulary.unknown_idx)
            ),
            "context_feature_names": list(context_spec.feature_names),
        },
        "training": {
            "initial_validation_bce": float(initial_validation["bce"]),
            "best_validation_bce": best_loss,
            "epochs_completed": len(history),
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
        },
        "validation": final_validation,
        "probability_note": (
            "Unweighted BCE sigmoid probabilities; no validation-label calibration "
            "is fitted in this command."
        ),
    }
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "objective_names": train_table.objective_names,
            "vocabulary": vocabulary.to_dict(),
            "context_spec": context_spec.to_dict(),
            "metrics": result,
        },
        checkpoint,
    )
    result["checkpoint"] = str(checkpoint)
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result


def score_kuairand_behavior_slates(
    checkpoint_path: str | Path,
    slates_path: str | Path,
    current_log_path: str | Path,
    output_path: str | Path,
    *,
    prefix_log_path: str | Path | None = None,
    batch_size: int = 256,
    device_name: str = "cpu",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score every prepared pseudo-slate with a trained behavior checkpoint."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "model_config",
        "objective_names",
        "vocabulary",
        "context_spec",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Behavior checkpoint is missing keys: {sorted(missing)}")
    model_config = SASRecMMoEConfig(**checkpoint["model_config"])
    vocabulary = ItemVocabulary.from_dict(checkpoint["vocabulary"])
    context_spec = BehaviorContextSpec.from_dict(checkpoint["context_spec"])
    objective_names = tuple(str(value) for value in checkpoint["objective_names"])
    if vocabulary.num_items != model_config.num_items:
        raise ValueError("Checkpoint vocabulary size does not match model_config")
    if context_spec.feature_dim != model_config.context_dim:
        raise ValueError("Checkpoint context spec does not match model_config")
    slates = PreparedKuaiRandSlates.load(slates_path)
    if len(slates.user_ids) == 0:
        raise ValueError("prepared slate file is empty")
    if objective_names != slates.objective_names:
        raise ValueError(
            "Behavior checkpoint and prepared slates use different objectives: "
            f"{objective_names} != {slates.objective_names}"
        )
    current_log = read_kuairand_log(current_log_path)
    prefix_log = read_kuairand_log(prefix_log_path) if prefix_log_path is not None else None
    history_index = UserHistoryIndex(
        current_log,
        vocabulary,
        prefix_log=prefix_log,
    )
    context = behavior_context_for_slates(slates, context_spec)
    encoded_candidates = vocabulary.encode(
        slates.item_ids,
        padding_mask=~slates.mask,
    )
    if not slates.mask.any(axis=1).all():
        raise ValueError("every prepared slate must contain a valid candidate")
    start_times = np.where(
        slates.mask,
        slates.timestamps,
        np.iinfo(np.int64).max,
    ).min(axis=1)
    histories = np.stack(
        [
            history_index.history(
                int(user_id),
                int(timestamp),
                max_history_length=model_config.max_history_length,
            )
            for user_id, timestamp in zip(slates.user_ids, start_times, strict=True)
        ]
    )

    device = torch.device(device_name)
    model = SASRecMMoERanker(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    outputs: list[np.ndarray] = []
    batch_latencies_ms: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(slates.user_ids), batch_size):
            end = min(start + batch_size, len(slates.user_ids))
            history_batch = torch.from_numpy(histories[start:end]).to(device)
            candidate_batch = torch.from_numpy(encoded_candidates[start:end]).to(device)
            context_batch = torch.from_numpy(context[start:end]).to(device)
            tick = time.perf_counter()
            probabilities = model.probabilities(
                history_batch,
                candidate_batch,
                context_batch,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_latencies_ms.append((time.perf_counter() - tick) * 1000.0)
            outputs.append(probabilities.cpu().numpy())
    probability_array = np.concatenate(outputs).astype(np.float32, copy=False)
    probability_array[~slates.mask] = 0.0
    tensor = SlateScoreTensor(
        slate_ids=np.arange(len(slates.user_ids), dtype=np.int64),
        candidate_ids=slates.item_ids.copy(),
        probabilities=probability_array,
        mask=slates.mask.copy(),
        objective_names=objective_names,
    )
    tensor.save(output_path)

    valid = np.broadcast_to(slates.mask[..., None], slates.labels.shape)
    squared = (probability_array - slates.labels.astype(np.float32)) ** 2
    raw_valid_candidates = slates.item_ids[slates.mask]
    mapped_valid_candidates = vocabulary.encode(raw_valid_candidates)
    valid_history = histories[histories != vocabulary.padding_idx]
    history_oov_fraction = (
        float(np.mean(valid_history == vocabulary.unknown_idx))
        if len(valid_history)
        else 0.0
    )
    elapsed_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema_version": "kuairand-neural-behavior-score-v1",
        "checkpoint": str(checkpoint_path),
        "slates": int(len(slates.user_ids)),
        "valid_candidates": int(slates.mask.sum()),
        "objective_names": list(objective_names),
        "output": str(output_path),
        "overall_brier": float(squared[valid].mean()),
        "per_objective_brier": {
            name: float(squared[..., index][slates.mask].mean())
            for index, name in enumerate(objective_names)
        },
        "candidate_oov_fraction": float(
            np.mean(mapped_valid_candidates == vocabulary.unknown_idx)
        ),
        "history_oov_fraction": history_oov_fraction,
        "elapsed_seconds": elapsed_seconds,
        "throughput": {
            "slates_per_second": float(
                len(slates.user_ids) / max(elapsed_seconds, 1e-12)
            ),
            "valid_candidates_per_second": float(
                slates.mask.sum() / max(elapsed_seconds, 1e-12)
            ),
        },
        "batch_latency_ms": {
            "p50": float(np.quantile(batch_latencies_ms, 0.50)),
            "p95": float(np.quantile(batch_latencies_ms, 0.95)),
            "batches": len(batch_latencies_ms),
        },
        "brier_interpretation": (
            "Diagnostic against labels in the scored slate file; independence depends "
            "on whether this split was held out from behavior training and tuning."
        ),
        "probability_note": checkpoint.get("metrics", {}).get(
            "probability_note",
            "No post-hoc calibrator recorded.",
        ),
    }
    if report_path is not None:
        atomic_write_json(report_path, report)
    return report
