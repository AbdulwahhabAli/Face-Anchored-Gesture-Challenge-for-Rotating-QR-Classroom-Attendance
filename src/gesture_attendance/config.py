"""Config loading: YAML file + dotted-key overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse ["pipeline.mode=full", "models.device=0"] into a dict (values YAML-typed)."""
    out: dict[str, Any] = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def resolve_path(p: str | Path | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def merged(cfg: dict, **section_overrides: dict) -> dict:
    out = copy.deepcopy(cfg)
    for section, vals in section_overrides.items():
        out.setdefault(section, {}).update(vals)
    return out
