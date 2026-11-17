"""Load experiment configs (YAML) with simple dotted overrides.

Example:
    cfg = load_config("configs/video_effnet.yaml", overrides=["train.epochs=5"])
    cfg["train"]["epochs"]  # -> 5
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def _parse_value(raw: str) -> Any:
    """Turn an override string into a Python value using YAML rules ("5" -> 5, "true" -> True)."""
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Return a copy of cfg with 'a.b.c=value' overrides applied."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {key}: {part} is not a mapping")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)
