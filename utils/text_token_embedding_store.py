"""Memory-mapped lookup for ragged, pre-pooling token embeddings.

Example:
    store = TokenEmbeddingStore("data/embeddings/text/all-MiniLM-L6-v2-tokens/10k")
    seq = store.get(12)  # ndarray [sequence_length, 384]
    batch, attention_mask, lengths = store.get_batch([12, 31, 52])

Sentence ID 0 / None resolves to an empty sequence. Special tokens retained by
 the tokenizer are part of each sequence; storage padding is removed.
"""
from __future__ import annotations

import bisect
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


class TokenEmbeddingStore:
    """Retrieve variable-length token vectors by sentence ID without loading all data into RAM."""

    def __init__(self, root: str | Path, *, mmap_mode: str = "r") -> None:
        self.root = Path(root).expanduser().resolve()
        candidate = self.root / "manifest.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"Embedding manifest not found: {candidate}")
        with candidate.open("r", encoding="utf-8") as handle:
            pointer = json.load(handle)
        # Accept either the scale root (main manifest) or a train/val/test folder
        # containing a split manifest that points back to the scale root.
        if "parts" in pointer:
            self.manifest = pointer
            self.scale_root = self.root
        elif pointer.get("embedding_manifest") and pointer.get("embedding_store_root"):
            main_manifest_path = (self.root / pointer["embedding_manifest"]).resolve()
            with main_manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
            self.scale_root = main_manifest_path.parent
        else:
            raise ValueError(f"Unrecognized token embedding manifest: {candidate}")

        if self.manifest.get("format") != "ragged_token_embeddings_v1":
            raise ValueError(
                f"Expected ragged_token_embeddings_v1, got {self.manifest.get('format')!r}. "
                "This store does not read pooled sentence embeddings."
            )
        if not self.manifest.get("complete", False):
            raise RuntimeError(f"Token embedding store is incomplete: {self.scale_root}")

        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.dtype = np.dtype(self.manifest.get("storage_dtype", "float16"))
        self.mmap_mode = mmap_mode
        self.shared_root = self.scale_root / str(self.manifest.get("embedding_store", "shared"))
        self.parts = sorted(self.manifest.get("parts", []), key=lambda p: int(p["start_id"]))
        self.starts = [int(p["start_id"]) for p in self.parts]
        self.ends = [int(p["end_id"]) for p in self.parts]
        self.num_sentences = int(self.manifest.get("num_sentences", 0))

    @lru_cache(maxsize=8)
    def _load_part(self, part_index: int) -> tuple[np.ndarray, np.ndarray]:
        part = self.parts[part_index]
        token_path = self.shared_root / str(part["token_file"])
        offsets_path = self.shared_root / str(part["offset_file"])
        tokens = np.load(token_path, mmap_mode=self.mmap_mode, allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode=self.mmap_mode, allow_pickle=False)
        if tokens.ndim != 2 or tokens.shape[1] != self.embedding_dim:
            raise ValueError(f"Invalid token feature shape in {token_path}: {tokens.shape}")
        if offsets.ndim != 1 or offsets.shape[0] != int(part["rows"]) + 1:
            raise ValueError(f"Invalid offsets shape in {offsets_path}: {offsets.shape}")
        if int(offsets[0]) != 0 or int(offsets[-1]) != tokens.shape[0]:
            raise ValueError(f"Invalid offset boundaries in {offsets_path}")
        return tokens, offsets

    def get(self, sentence_id: int | None) -> np.ndarray:
        """Return one sentence as [L, D]. ID 0 or None returns an empty [0, D] array."""
        if sentence_id is None or int(sentence_id) == 0:
            return np.empty((0, self.embedding_dim), dtype=self.dtype)
        sid = int(sentence_id)
        if sid < 0:
            raise ValueError(f"Sentence ID must be non-negative, got {sid}")
        part_index = bisect.bisect_right(self.starts, sid) - 1
        if part_index < 0 or sid > self.ends[part_index]:
            raise KeyError(f"Sentence ID {sid} is not in this embedding store")
        tokens, offsets = self._load_part(part_index)
        local_index = sid - int(self.parts[part_index]["start_id"])
        start = int(offsets[local_index])
        end = int(offsets[local_index + 1])
        return tokens[start:end]

    def get_batch(
        self,
        sentence_ids: Iterable[int | None],
        *,
        max_length: int | None = None,
        truncate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return padded token vectors, boolean attention mask and original/effective lengths.

        Shapes are [B, Lmax, D], [B, Lmax], and [B]. By default, max_length is
        not applied. If supplied, overlength sequences raise unless truncate=True.
        """
        sequences = [self.get(sid) for sid in sentence_ids]
        original_lengths = np.asarray([seq.shape[0] for seq in sequences], dtype=np.int32)
        if max_length is not None:
            if max_length < 0:
                raise ValueError("max_length must be >= 0")
            too_long = original_lengths > max_length
            if np.any(too_long) and not truncate:
                raise ValueError(
                    f"A sequence has {int(original_lengths.max())} tokens, exceeding max_length={max_length}. "
                    "Set truncate=True only if truncation is intended."
                )
            sequences = [seq[:max_length] for seq in sequences]
        lengths = np.asarray([seq.shape[0] for seq in sequences], dtype=np.int32)
        max_len = int(lengths.max()) if lengths.size else 0
        batch = np.zeros((len(sequences), max_len, self.embedding_dim), dtype=self.dtype)
        mask = np.zeros((len(sequences), max_len), dtype=np.bool_)
        for row, seq in enumerate(sequences):
            length = int(seq.shape[0])
            if length:
                batch[row, :length] = seq
                mask[row, :length] = True
        return batch, mask, lengths

    def close(self) -> None:
        """Drop cached mmap references. NumPy closes mappings when references are released."""
        self._load_part.cache_clear()

    def __enter__(self) -> "TokenEmbeddingStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
