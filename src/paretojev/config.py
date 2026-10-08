from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is malformed."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ConfigError(f"direction for {self.name!r} must be -1 or 1")


def objective_specs(raw: list[dict[str, Any]]) -> list[ObjectiveSpec]:
    specs = [ObjectiveSpec(**item) for item in raw]
    names = [x.name for x in specs]
    if len(names) != len(set(names)):
        raise ConfigError("objective names must be unique")
    return specs
