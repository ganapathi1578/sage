#!/usr/bin/env python3
"""Memory-mapped lookup for sentence embeddings addressed by global sentence_id."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np


class SentenceEmbeddingStore:
    """Resolve IDs into FP16/FP32 vectors without loading the full table into RAM.

    ID 0 or any non-positive ID is treated as missing and returns a zero vector.
    IDs 1..N map to contiguous rows across shared/part-XXXXXX.npy files.
    """

    def __init__(self, embedding_root: str | Path, max_open_parts: int = 8):
        self.root = Path(embedding_root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if not self.manifest.get("complete", False):
            raise RuntimeError(f"Embedding store is incomplete: {self.root}")
        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.num_sentences = int(self.manifest.get("num_sentences", self.manifest.get("dictionary_rows", 0)))
        self.rows_per_chunk = int(self.manifest["rows_per_chunk"])
        self.dtype = np.dtype(self.manifest["storage_dtype"])
        self.shared_root = self.root / self.manifest.get("embedding_store", "shared")
        self.max_open_parts = max(1, int(max_open_parts))
        self._parts: OrderedDict[int, np.ndarray] = OrderedDict()

    def _load_part(self, part_index: int) -> np.ndarray:
        arr = self._parts.get(part_index)
        if arr is not None:
            self._parts.move_to_end(part_index)
            return arr
        path = self.shared_root / f"part-{part_index:06d}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing embedding shard for sentence IDs starting near {part_index * self.rows_per_chunk + 1}: {path}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        self._parts[part_index] = arr
        while len(self._parts) > self.max_open_parts:
            self._parts.popitem(last=False)
        return arr

    def get(self, sentence_ids: Iterable[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(sentence_ids, dtype=object)
        original_shape = ids.shape
        flat = np.fromiter(
            (0 if value is None else int(value) for value in ids.reshape(-1)),
            dtype=np.int64,
            count=ids.size,
        )
        result = np.zeros((flat.size, self.embedding_dim), dtype=self.dtype)
        valid_mask = flat > 0
        if np.any(flat[valid_mask] > self.num_sentences):
            bad = int(flat[valid_mask][flat[valid_mask] > self.num_sentences][0])
            raise IndexError(f"sentence_id {bad} exceeds dictionary size {self.num_sentences}")
        valid_positions = np.flatnonzero(valid_mask)
        if valid_positions.size:
            part_ids = (flat[valid_positions] - 1) // self.rows_per_chunk
            for part_index in np.unique(part_ids):
                positions = valid_positions[part_ids == part_index]
                arr = self._load_part(int(part_index))
                local_rows = flat[positions] - (int(part_index) * self.rows_per_chunk + 1)
                result[positions] = arr[local_rows]
        return result.reshape((*original_shape, self.embedding_dim))

    def close(self) -> None:
        self._parts.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
