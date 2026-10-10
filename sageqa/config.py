"""YAML config loading with recursive `extends` support and repository-root paths."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_one(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError(f"Cyclic config extends detected at {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    seen = set(seen)
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    merged: dict[str, Any] = {}
    for parent in extends:
        merged = deep_merge(merged, _load_one((path.parent / parent), seen))
    return deep_merge(merged, raw)


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists() or ((candidate / "scripts").is_dir() and (candidate / "configs").is_dir()):
            return candidate
    return Path.cwd().resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).expanduser().resolve()
    cfg = _load_one(path, set())
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(find_repo_root(path.parent))
    return cfg, Path(cfg["_repo_root"])


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()
