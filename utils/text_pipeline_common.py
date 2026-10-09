#!/usr/bin/env python3
"""Common helpers for the config-driven UniProp text pipeline."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config_dir = config_path.parent
    sage_root_value = config.get("paths", {}).get("sage_root", "../..")
    sage_root_path = Path(sage_root_value).expanduser()
    sage_root = (config_dir / sage_root_path).resolve() if not sage_root_path.is_absolute() else sage_root_path.resolve()
    return config, config_path, sage_root


def config_path(config: dict[str, Any], key: str, sage_root: Path) -> Path:
    value = Path(config["paths"][key]).expanduser()
    return value.resolve() if value.is_absolute() else (sage_root / value).resolve()


def selected_scale(config: dict[str, Any], override: str | None = None) -> tuple[str, int | None]:
    dataset = config.get("dataset", {})
    scale = override or dataset.get("scale", "smoke")
    limits = dataset.get("scale_limits", {})
    if scale not in limits:
        raise ValueError(f"Unknown scale {scale!r}; available: {', '.join(limits)}")
    limit = limits[scale]
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError(f"Scale limit must be >= 0, got {limit}")
    return str(scale), limit


def normalize_sentence(value: Any, normalize_cfg: dict[str, Any] | None = None) -> str | None:
    if value is None:
        return None
    cfg = normalize_cfg or {}
    text = unicodedata.normalize(str(cfg.get("unicode", "NFKC")), str(value))
    if cfg.get("collapse_whitespace", True):
        text = re.sub(r"\s+", " ", text)
    if cfg.get("strip", True):
        text = text.strip()
    if cfg.get("casefold", False):
        text = text.casefold()
    return text or None


def infer_split(path: Path, root: Path, split_names: list[str]) -> str:
    """Infer split from path components (train/val/test); return 'all' if absent."""
    aliases = {"validation": "val", "valid": "val"}
    names = {name.lower() for name in split_names}
    try:
        parts = path.resolve().relative_to(root.resolve()).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    for part in parts:
        candidate = aliases.get(part.lower(), part.lower())
        if candidate in names:
            return candidate
    return "all"


def allocate_quotas(counts: dict[str, int], limit: int | None) -> dict[str, int]:
    """Allocate a global row cap proportionally over detected split folders."""
    counts = {k: max(0, int(v)) for k, v in counts.items()}
    total = sum(counts.values())
    if limit is None or limit >= total:
        return dict(counts)
    if limit <= 0 or total == 0:
        return {k: 0 for k in counts}
    raw = {k: limit * v / total for k, v in counts.items()}
    quotas = {k: int(raw[k]) for k in counts}
    remainder = limit - sum(quotas.values())
    order = sorted(counts, key=lambda k: (raw[k] - quotas[k], counts[k], k), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp.replace(path)


def json_fingerprint(value: dict[str, Any]) -> str:
    import hashlib
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
