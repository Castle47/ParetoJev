from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

import numpy as np

from paretojev.utils.io import append_jsonl_row, atomic_write_json, iter_jsonl
from paretojev.utils.provenance import canonical_hash, runtime_provenance

from .prompts import DecisionObjective, build_state
from .scoring import DecisionScores

ScoreMode = Literal["choice", "noul"]
NoulScope = Literal["pointwise", "slate"]


class DecisionClientProtocol(Protocol):
    model: str

    def stats(self) -> dict[str, int]: ...


ScoreFunction = Callable[
    [DecisionClientProtocol, Any, list[dict[str, Any]], Iterable[DecisionObjective]],
    DecisionScores,
]


@dataclass(frozen=True)
class ScoreJsonlConfig:
    provider: str
    mode: ScoreMode
    objective_names: tuple[str, ...]
    noul_scope: NoulScope = "pointwise"
    resume: bool = True
    overwrite: bool = False
    continue_on_error: bool = False
    durable_every: int = 1
    start_index: int = 0
    max_new_rows: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"choice", "noul"}:
            raise ValueError(f"Unknown score mode: {self.mode}")
        if self.noul_scope not in {"pointwise", "slate"}:
            raise ValueError(f"Unknown Noul scope: {self.noul_scope}")
        if not self.objective_names:
            raise ValueError("At least one objective is required")
        if self.durable_every <= 0:
            raise ValueError("durable_every must be positive")
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.max_new_rows is not None and self.max_new_rows <= 0:
            raise ValueError("max_new_rows must be positive")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite cannot both be enabled")


def _slate_key(value: Any) -> str:
    if value is None:
        raise ValueError("Every scoring input row needs a non-null slate_id")
    return canonical_hash(value)


def _signature(config: ScoreJsonlConfig, model: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "provider": config.provider,
        "mode": config.mode,
        "noul_scope": config.noul_scope if config.mode == "noul" else None,
        "objective_names": list(config.objective_names),
        "requested_model": model,
    }


def _validate_existing_row(row: dict[str, Any], expected: dict[str, Any]) -> None:
    observed = {
        "schema_version": int(row.get("schema_version", 1)),
        "provider": row.get("provider"),
        "mode": row.get("mode"),
        "noul_scope": row.get("noul_scope"),
        "objective_names": list(row.get("objective_names", [])),
        "requested_model": row.get("requested_model"),
    }
    if observed != expected:
        raise ValueError(
            "Existing score output is incompatible with this run. "
            f"Observed {observed}, expected {expected}. Use --overwrite or a new output path."
        )


def _existing_completed(
    output_path: Path,
    expected_signature: dict[str, Any],
) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    for row in iter_jsonl(output_path):
        _validate_existing_row(row, expected_signature)
        key = _slate_key(row.get("slate_id"))
        if key in completed:
            raise ValueError(f"Duplicate slate_id in existing output: {row.get('slate_id')!r}")
        completed.add(key)
    return completed


def _usage_add(total: dict[str, float], usage: Any, prefix: str = "") -> None:
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[full_key] = total.get(full_key, 0.0) + float(value)
        elif isinstance(value, dict):
            _usage_add(total, value, full_key)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "max_ms": float(array.max()),
    }


def run_score_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    client: DecisionClientProtocol,
    score_function: ScoreFunction,
    objectives: tuple[DecisionObjective, ...],
    config: ScoreJsonlConfig,
    error_path: str | Path | None = None,
    report_path: str | Path | None = None,
    cache_count: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Score slates incrementally with durable resume semantics.

    One successful slate is appended immediately. An interrupted API run can be
    restarted without repaying completed calls, even if the HTTP cache itself is
    unavailable. Existing rows are accepted only when the scoring signature is
    identical, preventing accidental mixtures of models, modes or objectives.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    error_path = Path(error_path) if error_path else output_path.with_suffix(
        output_path.suffix + ".errors.jsonl"
    )
    report_path = Path(report_path) if report_path else output_path.with_suffix(
        output_path.suffix + ".report.json"
    )
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    expected_signature = _signature(config, client.model)
    if config.overwrite:
        output_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
    elif output_path.exists() and not config.resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Enable resume or use --overwrite."
        )

    completed = _existing_completed(output_path, expected_signature) if config.resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    processed = 0
    skipped_completed = 0
    skipped_before_start = 0
    failed = 0
    examined = 0
    latencies_ms: list[float] = []
    usage_total: dict[str, float] = {}
    fatal_error: BaseException | None = None

    output_handle = output_path.open("a", encoding="utf-8")
    error_handle = error_path.open("a", encoding="utf-8")
    try:
        for input_index, row in enumerate(iter_jsonl(input_path)):
            examined += 1
            if input_index < config.start_index:
                skipped_before_start += 1
                continue
            if config.max_new_rows is not None and processed >= config.max_new_rows:
                break

            slate_id = row.get("slate_id")
            key = _slate_key(slate_id)
            if key in completed:
                skipped_completed += 1
                continue
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(
                    f"Input index {input_index} needs a candidates list with at least two entries"
                )

            include_candidates = config.mode == "choice" or config.noul_scope == "slate"
            state = build_state(
                row.get("history", []),
                candidates if include_candidates else (),
                row.get("context"),
                include_candidates=include_candidates,
            )
            score_started = time.perf_counter()
            try:
                scored = score_function(client, state, candidates, objectives)
                latency_ms = (time.perf_counter() - score_started) * 1000.0
                raw = scored.raw_response
                resolved_model = raw.get("model") if isinstance(raw, dict) else None
                usage = raw.get("usage") if isinstance(raw, dict) else None
                record = {
                    **expected_signature,
                    "slate_id": slate_id,
                    "input_index": input_index,
                    "candidate_ids": list(scored.candidate_ids),
                    "probabilities": scored.probabilities.tolist(),
                    "confidence": scored.confidence.tolist(),
                    "entropy": scored.entropy.tolist(),
                    "resolved_model": resolved_model or client.model,
                    "usage": usage,
                    "latency_ms": latency_ms,
                }
                processed += 1
                completed.add(key)
                latencies_ms.append(latency_ms)
                _usage_add(usage_total, usage)
                durable = processed % config.durable_every == 0
                append_jsonl_row(
                    output_handle,
                    record,
                    flush=True,
                    fsync=durable,
                )
            except Exception as exc:
                failed += 1
                error_record = {
                    **expected_signature,
                    "slate_id": slate_id,
                    "input_index": input_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                }
                append_jsonl_row(error_handle, error_record, flush=True, fsync=True)
                if not config.continue_on_error:
                    fatal_error = exc
                    break
    except BaseException as exc:
        fatal_error = exc
    finally:
        output_handle.flush()
        error_handle.flush()
        os.fsync(output_handle.fileno())
        os.fsync(error_handle.fileno())
        output_handle.close()
        error_handle.close()

        elapsed = time.perf_counter() - started
        report: dict[str, Any] = {
            "status": "failed" if fatal_error else ("partial" if failed else "completed"),
            "input": str(input_path),
            "output": str(output_path),
            "errors": str(error_path),
            "signature": expected_signature,
            "config": asdict(config),
            "examined_input_rows": examined,
            "newly_scored_rows": processed,
            "skipped_completed_rows": skipped_completed,
            "skipped_before_start": skipped_before_start,
            "failed_rows": failed,
            "completed_rows_total": len(completed),
            "elapsed_seconds": elapsed,
            "latency": _latency_summary(latencies_ms),
            "usage_totals": usage_total,
            "client_stats": client.stats(),
            "cache_entries": cache_count() if cache_count else None,
            "provenance": runtime_provenance(
                input_paths=[input_path],
                config={"signature": expected_signature, "runner": asdict(config)},
            ),
        }
        atomic_write_json(report_path, report)

    if fatal_error is not None:
        raise fatal_error
    return report
