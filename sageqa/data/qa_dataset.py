"""Streaming reader for Sage ID-coded multiple-choice QA Parquet shards.

In DDP mode all ranks reproduce the same shuffled global stream, then yield
non-overlapping rows by global stream position. This keeps rank partitions
consistent while retaining bounded-memory shuffling.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from .targets import resolve_target, target_mode_for_record

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised in target environment
    raise RuntimeError("pyarrow is required; install requirements-training.txt") from exc


class QAParquetStream(IterableDataset):
    """Memory-bounded Parquet stream with optional buffered shuffle and DDP sharding.

    For training DDP, ranks must use the same seed/epoch and ``num_workers=0``.
    Every rank walks the same shuffled stream but receives a disjoint strided
    partition. A tail is dropped globally to make rows/rank divisible by both
    the local batch size and the accumulation count, preventing rank step-count
    mismatches and partial accumulation updates.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        cfg: dict[str, Any],
        *,
        shuffle: bool,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        per_rank_batch_size: int = 1,
        accumulation_steps: int = 1,
    ):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.cfg = cfg
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid DDP rank/world_size: {self.rank}/{self.world_size}")
        if self.world_size > 1 and not self.shuffle:
            raise ValueError("Distributed sharding is only enabled for shuffled training data")

        if not self.root.is_dir():
            raise FileNotFoundError(f"Compact split directory not found: {self.root}")
        self.files = sorted(self.root.rglob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No Parquet shards found beneath {self.root}")
        self.batch_rows = max(1, int(cfg.get("parquet_read_batch_rows", 2048)))
        self.shuffle_buffer_rows = max(1, int(cfg.get("shuffle_buffer_rows", 16384)))
        self.columns = cfg.get("columns", {})
        self.target_cfg = cfg.get("target", {})

        self.global_num_rows = sum(
            int(pq.ParquetFile(path).metadata.num_rows) for path in self.files
        )
        self.usable_global_rows = self.global_num_rows
        self.num_rows = self.global_num_rows
        if self.world_size > 1:
            quantum = self.world_size * max(1, int(per_rank_batch_size)) * max(1, int(accumulation_steps))
            self.usable_global_rows = (self.global_num_rows // quantum) * quantum
            self.num_rows = self.usable_global_rows // self.world_size
            if self.num_rows == 0:
                raise ValueError(
                    f"Only {self.global_num_rows} examples available; not enough for "
                    f"world_size={self.world_size}, batch={per_rank_batch_size}, "
                    f"accumulation={accumulation_steps}."
                )

    def __len__(self) -> int:
        # In DDP, DataLoader needs the per-rank length for update scheduling.
        return self.num_rows

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive a deterministic, different shuffle."""
        self.epoch = int(epoch)

    def _convert(self, row: dict[str, Any], path: Path) -> dict[str, Any]:
        video_col = str(self.columns.get("video_id", "video_id"))
        query_col = str(self.columns.get("query_id", "query_sentence_id"))
        options_col = str(self.columns.get("option_ids", "option_sentence_ids"))
        target_col = str(self.target_cfg.get("column", "labels"))
        for col in (video_col, query_col, options_col, target_col):
            if col not in row:
                raise KeyError(
                    f"Required column {col!r} missing from {path}. "
                    f"Available: {sorted(row)}. "
                    "Update configs/training/base.yaml to match the generated schema."
                )
        video_id = row[video_col]
        if video_id is None or not str(video_id).strip():
            raise ValueError(f"Missing video ID in {path}")
        options = row[options_col]
        if not isinstance(options, (list, tuple)) or not options:
            raise ValueError(f"{options_col} must be a non-empty list; got {options!r}")
        target_mode = target_mode_for_record(row, self.target_cfg)
        target_labels = resolve_target(row[target_col], list(options), self.target_cfg, row)
        if len(target_labels) != len(options):
            raise ValueError(
                f"Target label count {len(target_labels)} differs from {len(options)} options"
            )
        meta_cols = list(self.cfg.get("metadata_columns", []))
        metadata = {name: row.get(name) for name in meta_cols if name in row}
        return {
            "video_id": str(video_id),
            "query_sentence_id": 0 if row[query_col] is None else int(row[query_col]),
            "option_sentence_ids": [0 if item is None else int(item) for item in options],
            "target_labels": target_labels,
            "target_mode": target_mode,
            "metadata": metadata,
            "split": self.split,
        }

    def _iter_raw_rows(
        self, files: list[Path], rng: random.Random
    ) -> Iterator[tuple[dict[str, Any], Path]]:
        for path in files:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=self.batch_rows):
                rows = batch.to_pylist()
                if self.shuffle:
                    rng.shuffle(rows)
                for row in rows:
                    yield row, path

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        if self.world_size > 1 and worker_count != 1:
            raise RuntimeError("DDP global shuffle currently requires DataLoader num_workers=0")

        files = list(self.files[worker_id::worker_count])
        # Deliberately exclude rank from the shuffle seed so all DDP ranks
        # reproduce the same global order before taking disjoint rows.
        rng = random.Random(self.seed + 1000003 * self.epoch + 97 * worker_id)
        if self.shuffle:
            rng.shuffle(files)
        rows = self._iter_raw_rows(files, rng)

        if not self.shuffle:
            for row, path in rows:
                yield self._convert(row, path)
            return

        def buffered_shuffle() -> Iterator[tuple[dict[str, Any], Path]]:
            buffer: list[tuple[dict[str, Any], Path]] = []
            buffer_size = self.shuffle_buffer_rows
            for item in rows:
                if len(buffer) < buffer_size:
                    buffer.append(item)
                    continue
                selected_idx = rng.randrange(len(buffer))
                selected = buffer[selected_idx]
                buffer[selected_idx] = item
                yield selected
            rng.shuffle(buffer)
            yield from buffer

        for global_idx, (row, path) in enumerate(buffered_shuffle()):
            if self.world_size > 1 and global_idx >= self.usable_global_rows:
                break
            if self.world_size > 1 and (global_idx % self.world_size) != self.rank:
                continue
            yield self._convert(row, path)
