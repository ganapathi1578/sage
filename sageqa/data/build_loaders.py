"""Build streaming loaders from the resolved experiment configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .collate import FeatureCollator
from .qa_dataset import QAParquetStream


def build_loader(cfg: dict[str, Any], repo_root: Path, split: str):
    data_cfg = cfg["data"]
    scale = str(data_cfg.get("scale", "10k"))
    analysis_root = Path(data_cfg["analysis_root"])
    if not analysis_root.is_absolute():
        analysis_root = (repo_root / analysis_root).resolve()
    compact_dir = str(data_cfg.get("compact_dir", "compact"))
    split_root = analysis_root / scale / compact_dir / split

    text_root = Path(data_cfg["text_token_store_root"])
    if not text_root.is_absolute():
        text_root = repo_root / text_root
    text_root = (text_root / scale).resolve()

    video_root = Path(data_cfg["video_store"])
    if not video_root.is_absolute():
        video_root = repo_root / video_root
    video_root = video_root.resolve()

    train_cfg = cfg.get("training", {})
    workers = max(0, int(train_cfg.get("num_workers", 0)))
    ddp_active = (
        split == "train"
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    if ddp_active and workers != 0:
        raise ValueError("DDP with QAParquetStream currently requires training.num_workers=0")

    rank = dist.get_rank() if ddp_active else 0
    world_size = dist.get_world_size() if ddp_active else 1
    per_rank_batch = int(train_cfg.get("batch_size", 2))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    ds = QAParquetStream(
        split_root,
        split,
        data_cfg,
        shuffle=(split == "train"),
        seed=int(cfg.get("seed", 42)),
        rank=rank,
        world_size=world_size,
        per_rank_batch_size=per_rank_batch if ddp_active else 1,
        accumulation_steps=accumulation if ddp_active else 1,
    )
    collator = FeatureCollator(
        text_root, video_root, data_cfg,
        training=(split == "train"), seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(
        ds,
        batch_size=per_rank_batch,
        num_workers=workers,
        pin_memory=bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
        drop_last=bool(split == "train" and train_cfg.get("drop_last", False)),
        collate_fn=collator,
        persistent_workers=bool(workers and train_cfg.get("persistent_workers", False)),
    )
    return loader, ds, collator
