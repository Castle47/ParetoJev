from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from paretojev.data.schema import SlateBatch
from paretojev.data.synthetic import SyntheticConfig, SyntheticSlateDataset, collate_slates
from paretojev.losses.pareto import (
    augmented_tchebycheff,
    brier_loss,
    candidate_distribution_kl,
    gaussian_pareto_pairwise_loss,
    objective_pairwise_losses,
    pareto_pairwise_loss,
)
from paretojev.models.pareto_reranker import ParetoJevConfig, ParetoJevReranker
from paretojev.utils.io import atomic_write_json
from paretojev.utils.seed import seed_everything


@dataclass(frozen=True)
class TrainingLossConfig:
    pareto_pairwise: float = 1.0
    tchebycheff: float = 1.0
    uncertainty: float = 0.25
    jev_distillation: float = 0.35
    calibration: float = 0.25
    fusion_calibration: float = 0.5
    tchebycheff_rho: float = 0.05
    distillation_temperature: float = 1.0


@dataclass(frozen=True)
class SyntheticSmokeConfig:
    seed: int = 7
    n_train_slates: int = 192
    n_eval_slates: int = 96
    slate_size: int = 12
    n_objectives: int = 6
    feature_dim: int = 10
    batch_size: int = 24
    epochs: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


def utility_probabilities(
    probabilities: torch.Tensor,
    directions: Iterable[int],
) -> torch.Tensor:
    direction = torch.as_tensor(
        tuple(directions), device=probabilities.device, dtype=probabilities.dtype
    )
    if direction.shape != (probabilities.shape[-1],):
        raise ValueError("directions shape mismatch")
    return torch.where(direction > 0, probabilities, 1.0 - probabilities)


def compute_training_losses(
    model: ParetoJevReranker,
    batch: SlateBatch,
    *,
    directions: Iterable[int],
    config: TrainingLossConfig = TrainingLossConfig(),
) -> dict[str, torch.Tensor]:
    output = model(batch)
    directions = tuple(directions)
    targets = batch.target_probabilities if batch.target_probabilities is not None else batch.labels
    utility_targets = utility_probabilities(targets, directions)
    utility_jev = utility_probabilities(batch.jev_scores, directions)

    pareto = pareto_pairwise_loss(
        output["score_mean"], batch.labels, batch.mask, directions=directions
    )
    objective_losses = objective_pairwise_losses(
        output["objective_logits"], utility_targets, batch.mask, directions=None
    )
    mean_preference = batch.preference_weights.mean(dim=0)
    tchebycheff = augmented_tchebycheff(
        objective_losses,
        mean_preference,
        rho=config.tchebycheff_rho,
    )
    if "score_logvar" in output:
        uncertainty = gaussian_pareto_pairwise_loss(
            output["score_mean"],
            output["score_logvar"],
            batch.labels,
            batch.mask,
            directions=directions,
        )
    else:
        uncertainty = output["score_mean"].sum() * 0.0
    distillation = candidate_distribution_kl(
        output["objective_logits"],
        utility_jev,
        batch.mask,
        temperature=config.distillation_temperature,
    )
    calibration = brier_loss(
        torch.sigmoid(output["objective_logits"]), utility_targets, batch.mask
    )
    # This explicitly trains the gate to choose the lower-error raw probability
    # source; negative-feedback remains a risk probability at this stage.
    fusion_calibration = brier_loss(output["fused_scores"], targets, batch.mask)

    total = (
        config.pareto_pairwise * pareto
        + config.tchebycheff * tchebycheff
        + config.uncertainty * uncertainty
        + config.jev_distillation * distillation
        + config.calibration * calibration
        + config.fusion_calibration * fusion_calibration
    )
    return {
        "total": total,
        "pareto_pairwise": pareto,
        "tchebycheff": tchebycheff,
        "uncertainty": uncertainty,
        "jev_distillation": distillation,
        "calibration": calibration,
        "fusion_calibration": fusion_calibration,
        "objective_losses": objective_losses,
        **output,
    }


class AdvantageEvaluator:
    """EMER-inspired metric feedback for stabilizing objective weights.

    External preference weights remain the controlling signal. This evaluator
    only increases pressure on objectives whose current validation metric has
    fallen relative to a reference/EMA metric.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        minimum: float = 0.5,
        maximum: float = 2.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.minimum = minimum
        self.maximum = maximum
        self.epsilon = epsilon

    def adjust(
        self,
        preference_weights: torch.Tensor,
        current_metrics: torch.Tensor,
        reference_metrics: torch.Tensor,
    ) -> torch.Tensor:
        if current_metrics.shape != reference_metrics.shape:
            raise ValueError("metric vectors must have the same shape")
        if preference_weights.shape[-1] != current_metrics.shape[-1]:
            raise ValueError("objective dimension mismatch")
        ratio = (reference_metrics + self.epsilon) / (current_metrics + self.epsilon)
        advantage = ratio.clamp(self.minimum, self.maximum).pow(self.gamma)
        adjusted = preference_weights * advantage
        return adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)


def _batch_average_loss(
    model: ParetoJevReranker,
    loader: DataLoader[SlateBatch],
    directions: tuple[int, ...],
    loss_config: TrainingLossConfig,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            values.append(float(compute_training_losses(
                model, batch, directions=directions, config=loss_config
            )["total"].item()))
    return float(np.mean(values))


def _fusion_metrics(
    model: ParetoJevReranker,
    loader: DataLoader[SlateBatch],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    behavior: list[np.ndarray] = []
    jev: list[np.ndarray] = []
    fused: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    popularity: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            behavior.append(batch.behavior_scores.cpu().numpy())
            jev.append(batch.jev_scores.cpu().numpy())
            fused.append(output["fused_scores"].cpu().numpy())
            target = batch.target_probabilities if batch.target_probabilities is not None else batch.labels
            targets.append(target.cpu().numpy())
            gates.append(output["fusion_gate"].cpu().numpy())
            popularity.append(batch.popularity.cpu().numpy())
    behavior_array = np.concatenate(behavior)
    jev_array = np.concatenate(jev)
    fused_array = np.concatenate(fused)
    target_array = np.concatenate(targets)
    gate_array = np.concatenate(gates)
    popularity_array = np.concatenate(popularity)
    cold = popularity_array[..., 0] <= np.quantile(popularity_array[..., 0], 0.25)
    warm = popularity_array[..., 0] >= np.quantile(popularity_array[..., 0], 0.75)

    def mse(values: np.ndarray, selection: np.ndarray | None = None) -> float:
        squared = (values - target_array) ** 2
        if selection is None:
            return float(squared.mean())
        return float(squared[selection].mean())

    return {
        "behavior_brier": mse(behavior_array),
        "jev_brier": mse(jev_array),
        "fused_brier": mse(fused_array),
        "cold_behavior_brier": mse(behavior_array, cold),
        "cold_jev_brier": mse(jev_array, cold),
        "cold_fused_brier": mse(fused_array, cold),
        "warm_behavior_brier": mse(behavior_array, warm),
        "warm_jev_brier": mse(jev_array, warm),
        "warm_fused_brier": mse(fused_array, warm),
        "cold_mean_jev_gate": float(gate_array[cold].mean()),
        "warm_mean_jev_gate": float(gate_array[warm].mean()),
    }


def run_synthetic_smoke(
    config: SyntheticSmokeConfig = SyntheticSmokeConfig(),
    *,
    loss_config: TrainingLossConfig = TrainingLossConfig(),
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end CPU smoke test for data, model, losses and target metrics."""

    seed_everything(config.seed)
    device = torch.device(config.device)
    directions = tuple(1 if index != config.n_objectives - 1 else -1 for index in range(config.n_objectives))
    train_dataset = SyntheticSlateDataset(
        SyntheticConfig(
            n_slates=config.n_train_slates,
            slate_size=config.slate_size,
            n_objectives=config.n_objectives,
            feature_dim=config.feature_dim,
            seed=config.seed,
            hate_objective=config.n_objectives - 1,
        )
    )
    eval_dataset = SyntheticSlateDataset(
        SyntheticConfig(
            n_slates=config.n_eval_slates,
            slate_size=config.slate_size,
            n_objectives=config.n_objectives,
            feature_dim=config.feature_dim,
            seed=config.seed + 10_000,
            hate_objective=config.n_objectives - 1,
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_slates,
        generator=torch.Generator().manual_seed(config.seed),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_slates,
    )
    model = ParetoJevReranker(
        ParetoJevConfig(
            candidate_feature_dim=config.feature_dim,
            n_objectives=config.n_objectives,
            hidden_dim=48,
            n_heads=4,
            n_layers=1,
            ff_dim=96,
            dropout=0.05,
            fusion_hidden_dim=24,
            predict_uncertainty=True,
        )
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial_loss = _batch_average_loss(model, eval_loader, directions, loss_config, device)
    start = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            losses = compute_training_losses(
                model, batch, directions=directions, config=loss_config
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(losses["total"].item()))
        eval_loss = _batch_average_loss(model, eval_loader, directions, loss_config, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(epoch_losses)),
                "eval_loss": eval_loss,
            }
        )
    elapsed = time.perf_counter() - start
    final_loss = history[-1]["eval_loss"]
    fusion = _fusion_metrics(model, eval_loader, device)
    result: dict[str, Any] = {
        "status": "passed",
        "config": asdict(config),
        "loss_config": asdict(loss_config),
        "directions": list(directions),
        "training": {
            "initial_eval_loss": initial_loss,
            "final_eval_loss": final_loss,
            "loss_reduction_fraction": (initial_loss - final_loss) / max(initial_loss, 1e-12),
            "elapsed_seconds": elapsed,
            "history": history,
        },
        "fusion": fusion,
        "checks": {
            "finite_loss": bool(np.isfinite(final_loss)),
            "loss_decreased": bool(final_loss < initial_loss),
            "fused_beats_average_source": bool(
                fusion["fused_brier"]
                < 0.5 * (fusion["behavior_brier"] + fusion["jev_brier"])
            ),
            "jev_gate_higher_on_cold": bool(
                fusion["cold_mean_jev_gate"] > fusion["warm_mean_jev_gate"]
            ),
        },
    }
    if not all(result["checks"].values()):
        result["status"] = "warning"
    if output_path is not None:
        atomic_write_json(output_path, result)
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model.config),
                "smoke_config": asdict(config),
                "directions": directions,
            },
            checkpoint,
        )
    return result


@dataclass(frozen=True)
class PrecomputedTrainingConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    fusion_hidden_dim: int = 32
    device: str = "cpu"


def run_precomputed_training(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    directions: Iterable[int],
    config: PrecomputedTrainingConfig = PrecomputedTrainingConfig(),
    loss_config: TrainingLossConfig = TrainingLossConfig(),
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train ParetoJev from feature-complete NPZ files."""

    from paretojev.data.precomputed import PrecomputedSlateDataset

    seed_everything(config.seed)
    train_dataset = PrecomputedSlateDataset(train_path, seed=config.seed)
    validation_dataset = PrecomputedSlateDataset(validation_path, seed=config.seed + 1)
    if train_dataset.feature_dim != validation_dataset.feature_dim:
        raise ValueError("train and validation feature dimensions differ")
    if train_dataset.n_objectives != validation_dataset.n_objectives:
        raise ValueError("train and validation objective counts differ")
    directions = tuple(int(value) for value in directions)
    if len(directions) != train_dataset.n_objectives:
        raise ValueError("directions count must equal objective count")

    device = torch.device(config.device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_slates,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_slates,
    )
    model = ParetoJevReranker(
        ParetoJevConfig(
            candidate_feature_dim=train_dataset.feature_dim,
            n_objectives=train_dataset.n_objectives,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            fusion_hidden_dim=config.fusion_hidden_dim,
            predict_uncertainty=True,
        )
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    initial_loss = _batch_average_loss(
        model, validation_loader, directions, loss_config, device
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        values: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            losses = compute_training_losses(
                model, batch, directions=directions, config=loss_config
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values.append(float(losses["total"].item()))
        validation_loss = _batch_average_loss(
            model, validation_loader, directions, loss_config, device
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(values)),
                "validation_loss": validation_loss,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    fusion = _fusion_metrics(model, validation_loader, device)
    result: dict[str, Any] = {
        "status": "complete",
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "config": asdict(config),
        "loss_config": asdict(loss_config),
        "directions": list(directions),
        "data": {
            "train_slates": len(train_dataset),
            "validation_slates": len(validation_dataset),
            "slate_size": train_dataset.slate_size,
            "feature_dim": train_dataset.feature_dim,
            "n_objectives": train_dataset.n_objectives,
        },
        "training": {
            "initial_validation_loss": initial_loss,
            "best_validation_loss": best_loss,
            "loss_reduction_fraction": (initial_loss - best_loss) / max(initial_loss, 1e-12),
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
        },
        "fusion": fusion,
    }
    if output_path is not None:
        atomic_write_json(output_path, result)
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model.config),
                "training_config": asdict(config),
                "directions": directions,
                "metrics": result,
            },
            checkpoint,
        )
    return result
