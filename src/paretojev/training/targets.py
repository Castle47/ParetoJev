from __future__ import annotations

import math
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
    status: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        # Keep existing callers which construct a result from a passed boolean.
        if not self.status:
            object.__setattr__(self, "status", "passed" if self.passed else "failed")
        if self.status not in {"passed", "failed", "not_evaluable"}:
            raise ValueError(f"Unsupported target status: {self.status!r}")
        if self.passed != (self.status == "passed"):
            raise ValueError("Target status and passed flag disagree")


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


def _finite_number(value: Any) -> float:
    # bool is a subclass of int, but a success flag is not a measured metric.
    if isinstance(value, bool):
        raise ValueError("boolean values are not numeric metrics")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric value must be finite")
    return number


def _evidence_issues(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Validate declared evidence, without inferring real data from good scores.

    evidence_requirements maps dotted report paths to exact scalar values.
    required_methods lists method keys which must have nonempty report mappings.
    These are provenance checks, not independent verification of input origins.
    """
    requirements = config.get("evidence_requirements", {})
    if not isinstance(requirements, dict):
        raise ValueError("evidence_requirements must be a mapping")
    issues: list[str] = []
    for path, expected in requirements.items():
        if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
            raise ValueError("evidence requirement paths must be nonempty dotted strings")
        if not isinstance(expected, (str, bool, int, float)):
            raise ValueError(f"Evidence requirement {path!r} must have a scalar expected value")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                _finite_number(expected)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Evidence requirement {path!r} must be finite") from exc
        try:
            actual = get_dotted(metrics, path)
        except KeyError:
            issues.append(f"missing required evidence {path}")
            continue
        # In particular, 1 or the string 'true' cannot impersonate real_data=true.
        if type(actual) is not type(expected) or actual != expected:
            issues.append(f"evidence {path} must equal {expected!r}; got {actual!r}")
    required_methods = config.get("required_methods", [])
    if (
        not isinstance(required_methods, list)
        or any(not isinstance(method, str) or not method for method in required_methods)
        or len(set(required_methods)) != len(required_methods)
    ):
        raise ValueError("required_methods must be a list of distinct nonempty strings")
    methods = metrics.get("methods", {})
    for method in required_methods:
        if not isinstance(methods, dict) or not isinstance(methods.get(method), dict) or not methods[method]:
            issues.append(f"missing evaluated method {method}")
    return issues


def evaluate_targets(
    metrics: dict[str, Any],
    target_config: dict[str, Any],
) -> list[TargetResult]:
    if not isinstance(metrics, dict) or not isinstance(target_config, dict):
        raise ValueError("metrics and target config must be mappings")
    targets = target_config.get("targets", target_config)
    if not isinstance(targets, dict):
        raise ValueError("target config must contain a 'targets' mapping")
    evidence_issues = _evidence_issues(metrics, target_config)
    results: list[TargetResult] = []
    for name, specification in targets.items():
        if not isinstance(specification, dict):
            raise ValueError(f"Target {name} must be a mapping")
        metric = specification.get("metric", name)
        if not isinstance(metric, str) or not metric or any(not part for part in metric.split(".")):
            raise ValueError(f"Target {name} must have a nonempty metric path")
        op = str(specification.get("op", ">="))
        if op not in _OPERATORS:
            raise ValueError(f"Unsupported operator {op!r} for target {name}")
        try:
            threshold = _finite_number(specification["threshold"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Target {name} requires a finite numeric threshold") from exc
        required = specification.get("required", True)
        if not isinstance(required, bool):
            raise ValueError(f"Target {name} required must be a boolean")
        detail = str(specification.get("description", ""))
        reasons = list(evidence_issues)
        actual = None
        try:
            actual = _finite_number(get_dotted(metrics, metric))
        except KeyError:
            reasons.append(f"missing metric {metric}")
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"metric {metric} is undefined, nonnumeric, or nonfinite")
        if reasons:
            passed = False
            status = "not_evaluable"
        else:
            assert actual is not None
            passed = bool(_OPERATORS[op](actual, threshold))
            status = "passed" if passed else "failed"
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
                status=status,
                reason="; ".join(reasons),
            )
        )
    return results


def target_report(results: list[TargetResult]) -> dict[str, Any]:
    required = [result for result in results if result.required]
    if not required or any(result.status == "not_evaluable" for result in required):
        status = "not_evaluable"
    elif all(result.passed for result in required):
        status = "passed"
    else:
        status = "failed"
    return {
        "passed": status == "passed",
        "status": status,
        "reason": "No required targets were configured." if not required else "",
        "required_passed": sum(result.passed for result in required),
        "required_total": len(required),
        "required_not_evaluable": sum(result.status == "not_evaluable" for result in required),
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
