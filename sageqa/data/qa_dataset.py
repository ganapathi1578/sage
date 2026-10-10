"""Streaming reader for Sage ID-coded multiple-choice QA Parquet shards."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .targets import resolve_target

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised in target environment
    raise RuntimeError("pyarrow is required; install requirements-training.txt") from exc


class QAParquetStream(IterableDataset):
    """Memory-bounded Parquet stream; shuffles shards and Arrow batches, not split membership."""

    def __init__(self, root: str | Path, split: str, cfg: dict[str, Any], *, shuffle: bool, seed: int = 42):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.cfg = cfg
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if not self.root.is_dir():
            raise FileNotFoundError(f"Compact split directory not found: {self.root}")
        self.files = sorted(self.root.rglob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No Parquet shards found beneath {self.root}")
        self.batch_rows = max(1, int(cfg.get("parquet_read_batch_rows", 2048)))
        self.columns = cfg.get("columns", {})
        self.target_cfg = cfg.get("target", {})
        self.num_rows = 0
        for path in self.files:
            self.num_rows += int(pq.ParquetFile(path).metadata.num_rows)

    def __len__(self) -> int:
        return self.num_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _convert(self, row: dict[str, Any], path: Path) -> dict[str, Any]:
        video_col = str(self.columns.get("video_id", "video_id"))
        query_col = str(self.columns.get("query_id", "query_sentence_id"))
        options_col = str(self.columns.get("option_ids", "option_sentence_ids"))
        target_col = str(self.target_cfg.get("column", "labels"))
        for col in (video_col, query_col, options_col, target_col):
            if col not in row:
                raise KeyError(
                    f"Required column {col!r} missing from {path}. Available: {sorted(row)}. "
                    "Update configs/training/base.yaml to match the generated schema."
                )
        video_id = row[video_col]
        if video_id is None or not str(video_id).strip():
            raise ValueError(f"Missing video ID in {path}")
        options = row[options_col]
        if not isinstance(options, (list, tuple)) or not options:
            raise ValueError(f"{options_col} must be a non-empty list; got {options!r}")
        target_labels = resolve_target(row[target_col], list(options), self.target_cfg, row)
        if len(target_labels) != len(options):
            raise ValueError(f"Target label count {len(target_labels)} differs from {len(options)} options")
        meta_cols = list(self.cfg.get("metadata_columns", []))
        metadata = {name: row.get(name) for name in meta_cols if name in row}
        return {
            "video_id": str(video_id),
            "query_sentence_id": 0 if row[query_col] is None else int(row[query_col]),
            "option_sentence_ids": [0 if item is None else int(item) for item in options],
            "target_labels": target_labels,
            "metadata": metadata,
            "split": self.split,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        files = self.files[worker_id::worker_count]
        rng = random.Random(self.seed + 1000003 * self.epoch + 97 * worker_id)
        files = list(files)
        if self.shuffle:
            rng.shuffle(files)
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=self.batch_rows):
                rows = batch.to_pylist()
                if self.shuffle:
                    rng.shuffle(rows)
                for row in rows:
                    yield self._convert(row, path)
