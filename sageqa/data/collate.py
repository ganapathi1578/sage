"""Feature-aware collation with variable-length text/video padding."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import get_worker_info

from .token_embedding_store import TokenEmbeddingStore
from .video_feature_store import VideoFeatureStore


class FeatureCollator:
    """Resolve sentence IDs and video IDs into tensors at batch time.

    Stores are opened lazily in each worker, which avoids sharing Zarr/NumPy handles
    across multiprocessing processes.
    """
    def __init__(self, text_root: str | Path, video_root: str | Path, data_cfg: dict[str, Any], *, training: bool, seed: int = 42):
        self.text_root = Path(text_root)
        self.video_root = Path(video_root)
        self.cfg = data_cfg
        self.training = training
        self.seed = int(seed)
        self._text_store = None
        self._video_store = None
        self._counter = 0

    def _stores(self):
        if self._text_store is None:
            self._text_store = TokenEmbeddingStore(self.text_root)
        if self._video_store is None:
            self._video_store = VideoFeatureStore(self.video_root)
        return self._text_store, self._video_store

    @staticmethod
    def _pad_text(sequences: list[np.ndarray], dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = np.asarray([s.shape[0] for s in sequences], dtype=np.int64)
        max_len = max(1, int(lengths.max(initial=0)))
        out = np.zeros((len(sequences), max_len, dim), dtype=np.float32)
        mask = np.zeros((len(sequences), max_len), dtype=np.bool_)
        for i, seq in enumerate(sequences):
            if seq.ndim != 2 or seq.shape[1] != dim:
                raise ValueError(f"Expected token embeddings [L,{dim}], got {seq.shape}")
            n = len(seq)
            if n:
                out[i, :n] = seq
                mask[i, :n] = True
        return torch.from_numpy(out), torch.from_numpy(mask), torch.from_numpy(lengths)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("Cannot collate an empty batch")
        text_store, video_store = self._stores()
        data_cfg = self.cfg
        max_video_frames = int(data_cfg.get("max_video_frames", 32))
        target_fps = data_cfg.get("target_fps", 1.0)
        sampling = str(data_cfg.get("video_sampling", "random_contiguous"))
        worker = get_worker_info()
        worker_offset = 0 if worker is None else 1000003 * worker.id
        rng = np.random.default_rng(self.seed + worker_offset + self._counter)
        self._counter += 1

        videos = [video_store.get(
            row["video_id"], max_frames=max_video_frames, target_fps=target_fps,
            training=self.training, sampling=sampling, rng=rng,
        ) for row in rows]
        spatial_pooling = str(data_cfg.get("spatial_pooling", "none")).lower()
        if spatial_pooling not in {"none", "mean", "max"}:
            raise ValueError("data.spatial_pooling must be one of: none, mean, max")
        if spatial_pooling != "none":
            for item in videos:
                f = item["features"]
                if spatial_pooling == "mean":
                    item["features"] = f.mean(axis=1, keepdims=True)
                else:
                    item["features"] = f.max(axis=1, keepdims=True)
                item["spatial_xy"] = item["spatial_xy"].mean(axis=0, keepdims=True)
        b = len(rows)
        max_t = max(len(v["features"]) for v in videos)
        p = int(videos[0]["features"].shape[1])
        video_dim = int(videos[0]["features"].shape[2])
        video = np.zeros((b, max_t, p, video_dim), dtype=np.float32)
        video_xy = np.zeros((b, p, 2), dtype=np.float32)
        video_times = np.zeros((b, max_t), dtype=np.float32)
        video_mask = np.zeros((b, max_t), dtype=np.bool_)
        for i, v in enumerate(videos):
            features = v["features"]
            if features.shape[1:] != (p, video_dim):
                raise ValueError("All videos in one store must share the same patch count and channel dimension")
            t = len(features)
            video[i, :t] = features
            video_xy[i] = v["spatial_xy"]
            video_times[i, :t] = v["temporal_positions"]
            video_mask[i, :t] = True

        # Each dictionary ID is looked up independently; repeated IDs map to the exact same cached sequence.
        query_limit = data_cfg.get("max_query_tokens")
        option_limit = data_cfg.get("max_option_tokens")
        query_seqs = [text_store.get(row["query_sentence_id"]) for row in rows]
        if query_limit is not None:
            query_seqs = [seq[:int(query_limit)] for seq in query_seqs]
        text_dim = int(text_store.embedding_dim)
        q, q_mask, q_lengths = self._pad_text(query_seqs, text_dim)

        max_k = max(len(row["option_sentence_ids"]) for row in rows)
        option_seqs: list[list[np.ndarray]] = []
        option_ids = np.zeros((b, max_k), dtype=np.int64)
        option_mask = np.zeros((b, max_k), dtype=np.bool_)
        for i, row in enumerate(rows):
            ids = row["option_sentence_ids"]
            seqs = [text_store.get(sid) for sid in ids]
            if option_limit is not None:
                seqs = [seq[:int(option_limit)] for seq in seqs]
            option_seqs.append(seqs)
            option_ids[i, :len(ids)] = ids
            option_mask[i, :len(ids)] = True
        flat_options = [seq for seqs in option_seqs for seq in seqs]
        opt_flat, opt_mask_flat, opt_lengths_flat = self._pad_text(flat_options, text_dim)
        max_opt_len = opt_flat.shape[1]
        option_tokens = np.zeros((b, max_k, max_opt_len, text_dim), dtype=np.float32)
        option_token_mask = np.zeros((b, max_k, max_opt_len), dtype=np.bool_)
        cursor = 0
        for i, seqs in enumerate(option_seqs):
            for j, seq in enumerate(seqs):
                n = int(opt_lengths_flat[cursor])
                if n:
                    option_tokens[i, j, :n] = opt_flat[cursor, :n].numpy()
                    option_token_mask[i, j, :n] = True
                cursor += 1

        target_labels = np.zeros((b, max_k), dtype=np.float32)
        for i, row in enumerate(rows):
            labels = np.asarray(row["target_labels"], dtype=np.float32)
            n_options = len(row["option_sentence_ids"])
            if labels.ndim != 1 or labels.shape[0] != n_options:
                raise ValueError(
                    f"Example target_labels has shape {labels.shape}, but {n_options} options exist"
                )
            if not np.isin(labels, [0.0, 1.0]).all():
                raise ValueError("target_labels must be multi-hot values containing only 0/1")
            if labels.sum() < 1:
                raise ValueError("Every example must contain at least one correct option")
            target_labels[i, :n_options] = labels
        target_tensor = torch.from_numpy(target_labels)
        target_counts = target_tensor.sum(dim=1).to(torch.long)
        return {
            "video_features": torch.from_numpy(video),
            "video_xy": torch.from_numpy(video_xy),
            "video_times": torch.from_numpy(video_times),
            "video_mask": torch.from_numpy(video_mask),
            "query_tokens": q,
            "query_mask": q_mask,
            "query_lengths": q_lengths,
            "option_tokens": torch.from_numpy(option_tokens),
            "option_token_mask": torch.from_numpy(option_token_mask),
            "option_mask": torch.from_numpy(option_mask),
            "target_labels": target_tensor,
            "target_counts": target_counts,
            "video_ids": [row["video_id"] for row in rows],
            "query_sentence_ids": [int(row["query_sentence_id"]) for row in rows],
            "option_sentence_ids": option_ids.tolist(),
            "metadata": [row.get("metadata", {}) for row in rows],
        }

    def close(self) -> None:
        if self._text_store is not None:
            self._text_store.close()
        if self._video_store is not None:
            self._video_store.close()
        self._text_store = self._video_store = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_text_store"] = None
        state["_video_store"] = None
        return state
