"""Checkpoint helpers with resolved config and runtime metadata."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


def git_revision(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, epoch: int, global_step: int, best_metric: float, cfg: dict[str, Any], repo_root: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "config": cfg,
        "git_revision": git_revision(repo_root),
        "rng_state": {
            "python": random.getstate(),
            "numpy": __import__("numpy").random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def write_run_metadata(run_dir: Path, cfg: dict[str, Any], model_summary: dict[str, Any], repo_root: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    meta = {"git_revision": git_revision(repo_root), **model_summary}
    (run_dir / "model_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
