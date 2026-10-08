from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from paretojev.config import load_yaml
from paretojev.utils.io import atomic_write_json, read_json


@dataclass(frozen=True)
class TargetResult:
    name: str
    metric: str
    operator: str
    threshold: float
    actual: float | None
    required: bool
    passed: bool
    detail: str = ""


_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
}


def get_dotted(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for component in dotted.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(dotted)
        value = value[component]
    return value


def evaluate_targets(
    metrics: dict[str, Any],
    target_config: dict[str, Any],
) -> list[TargetResult]:
    targets = target_config.get("targets", target_config)
    if not isinstance(targets, dict):
        raise ValueError("target config must contain a 'targets' mapping")
    results: list[TargetResult] = []
    for name, specification in targets.items():
        if not isinstance(specification, dict):
            raise ValueError(f"Target {name} must be a mapping")
        metric = str(specification.get("metric", name))
        op = str(specification.get("op", ">="))
        if op not in _OPERATORS:
            raise ValueError(f"Unsupported operator {op!r} for target {name}")
        threshold = float(specification["threshold"])
        required = bool(specification.get("required", True))
        detail = str(specification.get("description", ""))
        try:
            actual = float(get_dotted(metrics, metric))
            passed = bool(_OPERATORS[op](actual, threshold))
        except (KeyError, TypeError, ValueError):
            actual = None
            passed = False
        results.append(
            TargetResult(
                name=str(name),
                metric=metric,
                operator=op,
                threshold=threshold,
                actual=actual,
                required=required,
                passed=passed,
                detail=detail,
            )
        )
    return results


def target_report(results: list[TargetResult]) -> dict[str, Any]:
    required = [result for result in results if result.required]
    return {
        "passed": all(result.passed for result in required),
        "required_passed": sum(result.passed for result in required),
        "required_total": len(required),
        "targets": [result.__dict__ for result in results],
    }


def check_target_files(
    metrics_path: str | Path,
    targets_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    config = load_yaml(targets_path)
    report = target_report(evaluate_targets(metrics, config))
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
