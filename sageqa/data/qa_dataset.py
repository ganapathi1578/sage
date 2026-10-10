"""Streaming reader for Sage ID-coded multiple-choice QA Parquet shards."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised in target environment
    raise RuntimeError("pyarrow is required; install requirements-training.txt") from exc


def resolve_target(value: Any, option_ids: list[Any], target_cfg: dict[str, Any], record: dict[str, Any]) -> int:
    """Resolve configured target representation to a zero-based candidate index.

    Supported formats:
      - index: scalar zero-based index
      - one_hot / option_labels: one label per option, exactly one positive
      - auto: scalar index or a one-positive per-option label vector
      - correct_text: string matching one of the raw options (requires options column)
    """
    fmt = str(target_cfg.get("format", "auto")).lower()
    k = len(option_ids)
    if fmt == "correct_text":
        raw_options = record.get(str(target_cfg.get("options_text_column", "options")))
        if not isinstance(raw_options, (list, tuple)) or not isinstance(value, str):
            raise ValueError("target.format=correct_text needs a text target and options_text_column list")
        matches = [i for i, option in enumerate(raw_options) if str(option) == value]
        if len(matches) != 1:
            raise ValueError(f"Correct answer text matched {len(matches)} options; expected exactly one")
        return matches[0]

    if fmt in {"index", "auto"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_index = int(value)
        if float(value) != raw_index:
            raise ValueError(f"Target index must be an integer, got {value!r}")
        idx = raw_index - int(target_cfg.get("index_base", 0))
        if not 0 <= idx < k:
            raise ValueError(
                f"Target index {raw_index} (index_base={target_cfg.get('index_base', 0)}) resolves to {idx}, "
                f"outside [0, {k}). Inspect labels and configure data.target.index_base/format."
            )
        return idx

    if fmt not in {"one_hot", "option_labels", "auto"}:
        raise ValueError(f"Unsupported target.format={fmt!r}")
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Target column is not a per-option list (format={fmt!r}, type={type(value).__name__}). "
            "Inspect the original labels column and set data.target.format explicitly."
        )
    if len(value) != k:
        raise ValueError(f"Label list length {len(value)} differs from option count {k}")
    positive_value = target_cfg.get("positive_value", 1)
    positive_strings = {"true", "yes", "correct", "positive"}
    positive: list[int] = []
    for i, label in enumerate(value):
        if isinstance(label, bool):
            is_positive = label
        elif isinstance(label, (int, float)):
            is_positive = label == positive_value
        elif isinstance(label, str):
            is_positive = label.strip().lower() in positive_strings or label == str(positive_value)
        else:
            is_positive = False
        if is_positive:
            positive.append(i)
    if len(positive) != 1:
        raise ValueError(
            f"Expected exactly one positive option label, found {positive}. "
            f"target={value!r}; set data.target.format/positive_value to match your schema."
        )
    return positive[0]


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
        target_index = resolve_target(row[target_col], list(options), self.target_cfg, row)
        if target_index >= len(options):
            raise ValueError(f"Target index {target_index} invalid for {len(options)} options")
        meta_cols = list(self.cfg.get("metadata_columns", []))
        metadata = {name: row.get(name) for name in meta_cols if name in row}
        return {
            "video_id": str(video_id),
            "query_sentence_id": 0 if row[query_col] is None else int(row[query_col]),
            "option_sentence_ids": [0 if item is None else int(item) for item in options],
            "target_index": target_index,
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
