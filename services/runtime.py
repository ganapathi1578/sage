"""Checkpoint discovery/loading and single-process Sage model inference."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sageqa.config import load_config
from sageqa.data.token_embedding_store import TokenEmbeddingStore
from sageqa.data.video_feature_store import VideoFeatureStore
from sageqa.models.registry import build_model
from services import settings
from services.catalog import DatasetCatalog


class TextTokenEncoder:
    """Create pre-pooling MiniLM token features for text absent from the dictionary."""

    def __init__(self) -> None:
        self.model = None
        self.base_max_seq_length: int | None = None
        self.lock = threading.RLock()
        self.cache: OrderedDict[tuple[str, int | None], np.ndarray] = OrderedDict()
        self.cache_size = 2048

    def _load(self, max_seq_length: int | None = None) -> None:
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(settings.TEXT_MODEL_ID, device=settings.TEXT_DEVICE)
            self.model.eval()
            self.base_max_seq_length = int(self.model.max_seq_length)
        # A prior checkpoint may have requested a smaller input cap. Always set the
        # effective limit so a later uncapped model cannot accidentally reuse it.
        effective = self.base_max_seq_length if max_seq_length is None else int(max_seq_length)
        if effective is not None:
            self.model.max_seq_length = int(effective)

    def _encode_uncached(self, text: str, max_seq_length: int | None = None) -> np.ndarray:
        key = (text, max_seq_length)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached.copy()
        self._load(max_seq_length)
        model = self.model
        features = model.tokenize([text])
        features = {k: v.to(settings.TEXT_DEVICE) if torch.is_tensor(v) else v for k, v in features.items()}
        with torch.inference_mode():
            output = model[0](features)
            states = output.get("token_embeddings")
            attention = features.get("attention_mask")
            if states is None or attention is None:
                raise RuntimeError("MiniLM did not return token_embeddings and attention_mask")
            sequence = states[0][attention[0].bool()].detach().to(device="cpu", dtype=torch.float32).numpy()
        array = np.asarray(sequence, dtype=np.float32, order="C")
        self.cache[key] = array.copy()
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return array

    def encode_many(self, texts: list[str], max_seq_length: int | None = None) -> list[np.ndarray]:
        # Small batched encoding reduces overhead and uses exactly the pre-pooling
        # Transformer outputs used by scripts/encode_text_embeddings.py.
        with self.lock:
            self._load(max_seq_length)
            result: list[np.ndarray | None] = [None] * len(texts)
            pending: list[tuple[int, tuple[str, int | None]]] = []
            for i, text in enumerate(texts):
                key = (text, max_seq_length)
                cached = self.cache.get(key)
                if cached is not None:
                    self.cache.move_to_end(key)
                    result[i] = cached.copy()
                else:
                    pending.append((i, key))
            if pending:
                pending_texts = [key[0] for _, key in pending]
                features = self.model.tokenize(pending_texts)
                features = {k: v.to(settings.TEXT_DEVICE) if torch.is_tensor(v) else v for k, v in features.items()}
                with torch.inference_mode():
                    output = self.model[0](features)
                    states = output.get("token_embeddings")
                    attention = features.get("attention_mask")
                    if states is None or attention is None:
                        raise RuntimeError("MiniLM did not return token_embeddings and attention_mask")
                    for j, (original_index, key) in enumerate(pending):
                        seq = states[j][attention[j].bool()].detach().to(device="cpu", dtype=torch.float32).numpy()
                        array = np.asarray(seq, dtype=np.float32, order="C")
                        result[original_index] = array
                        self.cache[key] = array.copy()
                        self.cache.move_to_end(key)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            return [np.asarray(item, dtype=np.float32) for item in result if item is not None]


class FastViTHDCustomEncoder:
    """Lazily load Sage's exact FastViTHD extraction code for uploaded clips."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.vision = None
        self.projector = None
        self.device = None
        self.cfg = None
        self.model_commit = None

    def _ensure_loaded(self, device: torch.device) -> None:
        if self.vision is not None and self.device == device:
            return
        import scripts.extract_fastvithd_embeddings as extractor

        with settings.FASTVITHD_CONFIG.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["runtime"]["device"] = str(device)
        cfg["runtime"]["frame_batch_size"] = min(int(cfg["runtime"].get("frame_batch_size", 8)), 8)
        extractor.validate_config(cfg)
        vision, projector, commit = extractor.load_fastvithd(cfg, device)
        self.vision, self.projector, self.device = vision, projector, device
        self.cfg, self.model_commit = cfg, commit

    def encode(self, video_path: Path, device: torch.device, max_frames: int) -> dict[str, np.ndarray]:
        import cv2
        import scripts.extract_fastvithd_embeddings as extractor

        with self.lock:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                raise ValueError("OpenCV could not open the uploaded video")
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if source_fps <= 0 or frame_count <= 0:
                raise ValueError("Uploaded video metadata is invalid")
            duration_preflight = frame_count / source_fps
            if duration_preflight > settings.MAX_UPLOAD_DURATION_SECONDS:
                raise ValueError(
                    f"Uploaded video is {duration_preflight:.1f}s long; maximum allowed is "
                    f"{settings.MAX_UPLOAD_DURATION_SECONDS}s."
                )
            self._ensure_loaded(device)
            cfg = self.cfg
            duration, sample_count, batches = extractor.frame_batches(video_path, cfg)
            if duration > settings.MAX_UPLOAD_DURATION_SECONDS:
                close = getattr(batches, "close", None)
                if close:
                    close()
                raise ValueError(
                    f"Uploaded video is {duration:.1f}s long; maximum allowed is "
                    f"{settings.MAX_UPLOAD_DURATION_SECONDS}s."
                )
            fps = float(cfg["sampling"]["sample_fps"])
            wanted = np.unique(np.linspace(0, sample_count - 1, num=min(max_frames, sample_count)).round().astype(np.int64))
            wanted_set = set(int(i) for i in wanted.tolist())
            features_by_index: dict[int, np.ndarray] = {}
            output_grid: tuple[int, int] | None = None
            for images, sample_ids, _source_ids in batches:
                pick = np.asarray([i for i, sid in enumerate(sample_ids.tolist()) if int(sid) in wanted_set], dtype=np.int64)
                if not len(pick):
                    continue
                encoded, _native_grid, output_grid = extractor.encode_image_batch(
                    images[pick], self.vision, self.projector, device, cfg
                )
                for j, source_index in enumerate(sample_ids[pick].tolist()):
                    features_by_index[int(source_index)] = encoded[j]
            if not features_by_index:
                raise RuntimeError("No frames from the uploaded video could be encoded")
            actual_indices = sorted(features_by_index)
            feats = np.stack([features_by_index[i] for i in actual_indices]).astype(np.float32, copy=False)
            grid = output_grid or tuple(cfg["embeddings"].get("output_grid_hw", [4, 4]))
            xy = extractor.make_spatial_xy(int(grid[0]), int(grid[1]))
            frame_times = np.asarray(actual_indices, dtype=np.float32) / fps
            unit = float(cfg["sampling"].get("sample_fps", 1.0))
            positions = (frame_times - frame_times[0]) * unit
            return {
                "features": feats,
                "spatial_xy": xy.astype(np.float32),
                "timestamps_sec": frame_times,
                "temporal_positions": positions.astype(np.float32),
                "duration_sec": np.asarray([duration], dtype=np.float32),
            }


class ModelRuntime:
    def __init__(self, catalog: DatasetCatalog) -> None:
        self.catalog = catalog
        self.lock = threading.RLock()
        self.active: dict[str, Any] | None = None
        self.text_encoder = TextTokenEncoder()
        self.custom_encoder = FastViTHDCustomEncoder()
        self.text_stores: dict[tuple[str, str], TokenEmbeddingStore] = {}
        self.video_stores: dict[str, VideoFeatureStore] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        self.device = self._resolve_device()

    @staticmethod
    def _resolve_device() -> torch.device:
        requested = settings.DEVICE
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAGE_PLAYGROUND_DEVICE requests CUDA but CUDA is unavailable")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible")
        return device

    def list_models(self) -> dict[str, Any]:
        runs = []
        root = settings.OUTPUT_ROOT
        if not root.is_dir():
            return {"runs": [], "output_root": str(root)}
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cfg_path = run_dir / "config.resolved.yaml"
            summary_path = run_dir / "model_summary.json"
            cfg: dict[str, Any] = {}
            summary: dict[str, Any] = {}
            if cfg_path.is_file():
                try:
                    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    cfg = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = {}
            checkpoints = []
            for filename in ("best_checkpoint.pt", "last_checkpoint.pt"):
                p = run_dir / filename
                if p.is_file() and not p.name.endswith(".tmp"):
                    checkpoints.append({
                        "name": filename,
                        "size_bytes": p.stat().st_size,
                        "modified_at": int(p.stat().st_mtime),
                    })
            if not checkpoints:
                continue
            latest_metrics = None
            metrics_path = run_dir / "metrics.jsonl"
            if metrics_path.is_file():
                try:
                    with metrics_path.open("rb") as handle:
                        handle.seek(0, os.SEEK_END)
                        size = handle.tell()
                        handle.seek(max(0, size - 8192))
                        lines = handle.read().decode("utf-8", "replace").splitlines()
                    for line in reversed(lines):
                        try:
                            latest_metrics = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                except OSError:
                    pass
            model_cfg = cfg.get("model", {})
            data_cfg = cfg.get("data", {})
            runs.append({
                "run": run_dir.name,
                "architecture": summary.get("architecture", model_cfg.get("name", "unknown")),
                "dataset_scale": data_cfg.get("scale"),
                "parameters": summary.get("parameters"),
                "trainable_parameters": summary.get("trainable_parameters"),
                "configured_epochs": cfg.get("training", {}).get("epochs"),
                "latest_epoch": latest_metrics.get("epoch") if latest_metrics else None,
                "latest_validation_accuracy": (latest_metrics.get("val") or {}).get("accuracy") if latest_metrics else None,
                "checkpoints": checkpoints,
            })
        return {"runs": runs, "output_root": str(root)}

    def _checkpoint_path(self, run: str, checkpoint: str) -> Path:
        if not run or Path(run).name != run or run in {".", ".."}:
            raise ValueError("Invalid experiment name")
        if checkpoint not in {"best_checkpoint.pt", "last_checkpoint.pt"}:
            raise ValueError("Choose best_checkpoint.pt or last_checkpoint.pt")
        path = (settings.OUTPUT_ROOT / run / checkpoint).resolve()
        if settings.OUTPUT_ROOT not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {run}/{checkpoint}")
        return path

    def _load_model_locked(self, run: str, checkpoint_name: str) -> dict[str, Any]:
        ckpt_path = self._checkpoint_path(run, checkpoint_name)
        active = self.active
        if active and active["run"] == run and active["checkpoint_name"] == checkpoint_name and active["checkpoint_mtime_ns"] == ckpt_path.stat().st_mtime_ns:
            return self.active_info()
        # Free the previous checkpoint before moving a new one onto a single shared GPU.
        if self.active is not None:
            del self.active["model"]
            self.active = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = state.get("config")
        if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict):
            cfg_path = ckpt_path.parent / "config.resolved.yaml"
            if not cfg_path.is_file():
                raise ValueError("Checkpoint has no embedded config and config.resolved.yaml is missing")
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        model_cfg = cfg["model"]
        model = build_model(model_cfg)
        model.load_state_dict(state["model"])
        model.to(self.device)
        model.eval()
        self.active = {
            "run": run,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_mtime_ns": ckpt_path.stat().st_mtime_ns,
            "model": model,
            "config": cfg,
            "epoch": int(state.get("epoch", -1)) + 1,
            "global_step": int(state.get("global_step", 0)),
            "best_metric": state.get("best_metric"),
            "git_revision": state.get("git_revision"),
            "loaded_at": time.time(),
            "amp": str(cfg.get("training", {}).get("amp", "none")).lower(),
        }
        del state
        return self.active_info()

    def load_model(self, run: str, checkpoint_name: str) -> dict[str, Any]:
        with self.lock:
            return self._load_model_locked(run, checkpoint_name)

    def active_info(self) -> dict[str, Any]:
        if self.active is None:
            return {"loaded": False, "device": str(self.device)}
        cfg = self.active["config"]
        return {
            "loaded": True,
            "run": self.active["run"],
            "checkpoint": self.active["checkpoint_name"],
            "architecture": cfg.get("model", {}).get("name", "unknown"),
            "dataset_scale": cfg.get("data", {}).get("scale"),
            "epoch": self.active["epoch"],
            "global_step": self.active["global_step"],
            "best_validation_score": self.active["best_metric"],
            "git_revision": self.active["git_revision"],
            "device": str(self.device),
            "amp": self.active["amp"],
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            info = self.active_info()
        info["cuda_available"] = torch.cuda.is_available()
        active_cfg = (self.active or {}).get("config", {})
        active_data_cfg = active_cfg.get("data", {}) if isinstance(active_cfg, dict) else {}
        configured_store = active_data_cfg.get("video_store")
        effective_store = (settings.VIDEO_STORE if os.environ.get("SAGE_PLAYGROUND_VIDEO_STORE") else
                           settings.repo_path(configured_store) if configured_store else settings.VIDEO_STORE)
        info["video_store_path"] = str(effective_store)
        info["video_store_ready"] = effective_store.exists()
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(self.device if self.device.type == "cuda" else 0)
                if self.device.type == "cuda":
                    info["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(self.device))
                    info["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(self.device))
            except Exception:
                pass
        return info

    def _dictionary_value(self, scale: str, sentence_id: int | None) -> str | None:
        if sentence_id is None:
            return None
        return self.catalog.dictionary(scale).get(int(sentence_id))

    def _text_store(self, scale: str, data_cfg: dict[str, Any]) -> TokenEmbeddingStore:
        root = settings.resolve_token_store_root(scale, data_cfg)
        if root is None:
            candidates = [str(path / scale / "manifest.json") for path in settings.token_store_candidates(data_cfg)]
            raise FileNotFoundError(
                "No complete ragged token-level MiniLM store was found for this scale. "
                "Checked: " + ", ".join(candidates)
            )
        key = (str(root), scale)
        if key not in self.text_stores:
            self.text_stores[key] = TokenEmbeddingStore(root / scale)
        return self.text_stores[key]

    def _tokens_for(
        self, scale: str, text: str, sentence_id: int | None,
        original_text: str | None, max_seq_length: int | None, data_cfg: dict[str, Any],
    ) -> np.ndarray:
        if sentence_id and original_text is not None and text.strip() == original_text.strip():
            try:
                token_store = self._text_store(scale, data_cfg)
                seq = token_store.get(int(sentence_id))
                if len(seq):
                    seq = np.asarray(seq, dtype=np.float32)
                    return seq[:max_seq_length] if max_seq_length else seq
            except (FileNotFoundError, RuntimeError, KeyError, ValueError):
                # Use the same encoder online if this scale's cache is absent or invalid.
                pass
        seq = self.text_encoder._encode_uncached(text, max_seq_length)
        return seq[:max_seq_length] if max_seq_length else seq

    @staticmethod
    def _pad_sequences(sequences: list[np.ndarray], dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(1, max((int(x.shape[0]) for x in sequences), default=0))
        padded = torch.zeros((len(sequences), max_len, dim), dtype=torch.float32)
        masks = torch.zeros((len(sequences), max_len), dtype=torch.bool)
        for i, seq in enumerate(sequences):
            seq = np.asarray(seq, dtype=np.float32)
            if seq.ndim != 2 or seq.shape[1] != dim:
                raise ValueError(f"Text feature must have shape [L,{dim}], got {seq.shape}")
            if len(seq):
                padded[i, :len(seq)] = torch.from_numpy(seq.copy())
                masks[i, :len(seq)] = True
        return padded, masks

    def _cached_video_features(self, video_id: str, data_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
        configured_store = data_cfg.get("video_store")
        # An explicit service override wins; otherwise follow the selected experiment's config.
        path = (settings.VIDEO_STORE if os.environ.get("SAGE_PLAYGROUND_VIDEO_STORE") else
                settings.repo_path(configured_store) if configured_store else settings.VIDEO_STORE)
        key = str(path)
        if key not in self.video_stores:
            self.video_stores[key] = VideoFeatureStore(path)
        feature_store = self.video_stores[key]
        max_frames = min(int(data_cfg.get("max_video_frames", 32)), settings.MAX_VIDEO_FRAMES)
        target_fps = data_cfg.get("target_fps", 1.0)
        video = feature_store.get(
            video_id,
            max_frames=max_frames,
            target_fps=None if target_fps is None else float(target_fps),
            training=False,
            sampling="uniform",
            rng=np.random.default_rng(42),
        )
        return self._apply_spatial_pooling(video, str(data_cfg.get("spatial_pooling", "none")))

    @staticmethod
    def _apply_spatial_pooling(video: dict[str, np.ndarray], pooling: str) -> dict[str, np.ndarray]:
        result = dict(video)
        features = result["features"]
        mode = pooling.lower()
        if mode == "mean":
            result["features"] = features.mean(axis=1, keepdims=True)
            result["spatial_xy"] = result["spatial_xy"].mean(axis=0, keepdims=True)
        elif mode == "max":
            result["features"] = features.max(axis=1, keepdims=True)
            result["spatial_xy"] = result["spatial_xy"].mean(axis=0, keepdims=True)
        elif mode != "none":
            raise ValueError(f"Unsupported spatial pooling mode {pooling!r}")
        return result

    def _batch_from_inputs(
        self, *, scale: str, video: dict[str, np.ndarray], question: str,
        options: list[str], question_sentence_id: int | None,
        option_sentence_ids: list[int] | None, original_question: str | None,
        original_options: list[str] | None, data_cfg: dict[str, Any], model_cfg: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        max_q = data_cfg.get("max_query_tokens")
        max_o = data_cfg.get("max_option_tokens")
        max_q = int(max_q) if max_q is not None else None
        max_o = int(max_o) if max_o is not None else None
        q_seq = self._tokens_for(scale, question, question_sentence_id, original_question, max_q, data_cfg)
        q, q_mask = self._pad_sequences([q_seq], int(model_cfg.get("text_dim", 384)))

        ids = option_sentence_ids or []
        original_options = original_options or []
        opt_seqs: list[np.ndarray] = []
        for i, option in enumerate(options):
            sid = ids[i] if i < len(ids) else None
            original = original_options[i] if i < len(original_options) else None
            opt_seqs.append(self._tokens_for(scale, option, sid, original, max_o, data_cfg))
        opt_flat, opt_mask_flat = self._pad_sequences(opt_seqs, int(model_cfg.get("text_dim", 384)))
        max_opt = opt_flat.shape[1]
        option_tokens = opt_flat.unsqueeze(0)
        option_token_mask = opt_mask_flat.unsqueeze(0)

        vf = np.asarray(video["features"], dtype=np.float32)
        xy = np.asarray(video["spatial_xy"], dtype=np.float32)
        times = np.asarray(video["temporal_positions"], dtype=np.float32)
        if vf.ndim != 3:
            raise ValueError(f"Expected video features [T,P,C], got {vf.shape}")
        if vf.shape[2] != int(model_cfg.get("video_dim", vf.shape[2])):
            raise ValueError(
                f"Video feature dimension {vf.shape[2]} does not match this checkpoint's expected "
                f"dimension {model_cfg.get('video_dim')}. Check the selected video-feature cache/model."
            )
        video_features = torch.from_numpy(vf.copy()).unsqueeze(0)
        video_mask = torch.ones((1, vf.shape[0]), dtype=torch.bool)
        video_times = torch.from_numpy(times.copy()).unsqueeze(0)
        video_xy = torch.from_numpy(xy.copy()).unsqueeze(0)
        option_mask = torch.ones((1, len(options)), dtype=torch.bool)
        batch = {
            "video_features": video_features,
            "video_xy": video_xy,
            "video_times": video_times,
            "video_mask": video_mask,
            "query_tokens": q,
            "query_mask": q_mask,
            "option_tokens": option_tokens,
            "option_token_mask": option_token_mask,
            "option_mask": option_mask,
        }
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _uploaded_features(self, upload_id: str, username: str) -> dict[str, np.ndarray]:
        """Return unpooled features cached independently of the active checkpoint."""
        upload = self.uploads.get(upload_id)
        if not upload or upload.get("username") != username or not Path(upload.get("path", "")).is_file():
            raise FileNotFoundError("Uploaded video has expired or is not available in this session. Upload it again.")
        now = time.time()
        if upload.get("features") is not None:
            upload["last_used"] = now
            return upload["features"]
        path = Path(upload["path"])
        # Cache a service-capped, uniformly sampled representation. Model-specific
        # temporal caps and spatial pooling are applied to a copy in predict().
        features = self.custom_encoder.encode(path, self.device, max(1, settings.MAX_VIDEO_FRAMES))
        upload["features"] = features
        upload["last_used"] = now
        return features

    @staticmethod
    def _prepare_uploaded_features(
        video: dict[str, np.ndarray], data_cfg: dict[str, Any], max_frames_limit: int,
    ) -> dict[str, np.ndarray]:
        """Apply the selected experiment's FPS/frame cap without mutating the raw cache."""
        result = dict(video)
        features = np.asarray(video["features"])
        times = np.asarray(video["timestamps_sec"], dtype=np.float32)
        original_positions = np.asarray(video.get("temporal_positions", times), dtype=np.float32)
        max_frames_limit = max(1, int(max_frames_limit))
        max_frames = min(max_frames_limit, max(1, int(data_cfg.get("max_video_frames", max_frames_limit))))

        target_fps = data_cfg.get("target_fps", 1.0)
        if target_fps is not None:
            target_fps = float(target_fps)
            if target_fps <= 0:
                raise ValueError("data.target_fps must be positive or null")
            step = 1.0 / target_fps
            desired = np.arange(float(times[0]), float(times[-1]) + step * 0.25, step, dtype=np.float32)
            right = np.searchsorted(times, desired, side="left").clip(0, len(times) - 1)
            left = np.clip(right - 1, 0, len(times) - 1)
            nearest = np.where(np.abs(times[left] - desired) <= np.abs(times[right] - desired), left, right)
            indices = np.unique(nearest.astype(np.int64))
        else:
            indices = np.arange(len(times), dtype=np.int64)

        if len(indices) > max_frames:
            indices = indices[np.linspace(0, len(indices) - 1, num=max_frames).round().astype(np.int64)]
            indices = np.unique(indices)
        features = features[indices]
        selected_times = times[indices]
        if target_fps is None:
            selected_positions = original_positions[indices] - original_positions[indices[0]]
        else:
            selected_positions = (selected_times - selected_times[0]) * target_fps
        result["features"] = np.asarray(features, dtype=np.float32)
        result["timestamps_sec"] = selected_times.astype(np.float32)
        result["temporal_positions"] = np.asarray(selected_positions, dtype=np.float32)
        return result

    def predict(
        self, *, run: str, checkpoint: str, scale: str, split: str,
        video_id: str | None, upload_id: str | None, username: str,
        question_text: str, options: list[str], question_sentence_id: int | None = None,
        option_sentence_ids: list[int] | None = None, example: dict[str, Any] | None = None,
        expected_indices: list[int] | None = None, task_mode: str = "single_choice",
    ) -> dict[str, Any]:
        with self.lock:
            self._load_model_locked(run, checkpoint)
            assert self.active is not None
            active = self.active
            cfg = active["config"]
            data_cfg = cfg.get("data", {})
            model_cfg = cfg.get("model", {})
            if not question_text.strip():
                raise ValueError("Question text cannot be empty")
            if not 2 <= len(options) <= settings.MAX_OPTIONS:
                raise ValueError(f"Provide between 2 and {settings.MAX_OPTIONS} answer options")
            if any(not option.strip() for option in options):
                raise ValueError("Answer options cannot be empty")

            dataset_video_matches = bool(example and video_id and example.get("video_id") == video_id and not upload_id)
            dictionary = self.catalog.dictionary(scale) if example and not upload_id else {}
            original_question = dictionary.get(int(example["query_sentence_id"])) if example else None
            original_options = [dictionary.get(int(sid), "") for sid in example["option_sentence_ids"]] if example else None
            unchanged = bool(
                dataset_video_matches
                and original_question is not None
                and question_text.strip() == original_question.strip()
                and original_options is not None
                and [x.strip() for x in options] == [x.strip() for x in original_options]
            )
            if upload_id:
                video = self._uploaded_features(upload_id, username)
                video = self._prepare_uploaded_features(video, data_cfg, settings.MAX_VIDEO_FRAMES)
                video = self._apply_spatial_pooling(video, str(data_cfg.get("spatial_pooling", "none")))
            else:
                if not video_id:
                    raise ValueError("Select a dataset video or upload a video before inference")
                video = self._cached_video_features(video_id, data_cfg)

            batch = self._batch_from_inputs(
                scale=scale, video=video, question=question_text, options=options,
                question_sentence_id=question_sentence_id if unchanged else None,
                option_sentence_ids=option_sentence_ids if unchanged else None,
                original_question=original_question if unchanged else None,
                original_options=original_options if unchanged else None,
                data_cfg=data_cfg, model_cfg=model_cfg,
            )
            amp = active["amp"]
            amp_enabled = self.device.type == "cuda" and amp in {"fp16", "bf16"}
            amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
            with torch.inference_mode():
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = active["model"](batch).float()[0, :len(options)]
            scores = logits.detach().cpu().numpy().astype(np.float64)
            shifted = scores - scores.max()
            softmax_probs = np.exp(shifted) / max(float(np.exp(shifted).sum()), 1e-12)
            sigmoid_probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))
            top1 = int(np.argmax(scores))
            threshold = float(cfg.get("training", {}).get("multi_label_threshold", 0.5))
            effective_task_mode = (example or {}).get("target_mode", task_mode) if unchanged else task_mode
            multi = effective_task_mode == "multi_label"
            threshold_selected = [i for i, prob in enumerate(sigmoid_probs.tolist()) if prob >= threshold] if multi else [top1]
            results_options = [
                {
                    "index": i,
                    "text": text,
                    "logit": round(float(scores[i]), 6),
                    "relative_probability": round(float(softmax_probs[i]), 6),
                    "sigmoid_probability": round(float(sigmoid_probs[i]), 6),
                    "selected_by_threshold": bool(i in threshold_selected),
                    "is_top1": bool(i == top1),
                }
                for i, text in enumerate(options)
            ]

            ground_truth_indices: list[int] | None = None
            ground_truth_source: str | None = None
            if unchanged and example is not None:
                ground_truth_indices = [i for i, label in enumerate(example.get("target_labels", [])) if int(label) == 1]
                ground_truth_source = f"dataset {scale}/{split} example {example['example_id']}"
            elif expected_indices:
                ground_truth_indices = sorted({int(i) for i in expected_indices if 0 <= int(i) < len(options)})
                ground_truth_source = "manually supplied expected answer"

            correctness = None
            exact_match = None
            top1_correct = None
            if ground_truth_indices is not None:
                top1_correct = top1 in ground_truth_indices
                correctness = bool(top1_correct)
                if multi:
                    exact_match = set(threshold_selected) == set(ground_truth_indices)
                else:
                    exact_match = top1_correct

            return {
                "run": run,
                "checkpoint": checkpoint,
                "checkpoint_epoch": active["epoch"],
                "global_step": active["global_step"],
                "architecture": model_cfg.get("name", "unknown"),
                "device": str(self.device),
                "amp": active["amp"],
                "scale": scale,
                "split": split,
                "video_id": video_id,
                "uploaded_video": bool(upload_id),
                "question": question_text,
                "task_mode": effective_task_mode,
                "prediction_index": top1,
                "prediction_text": options[top1],
                "threshold_selected_indices": threshold_selected,
                "threshold_selected_texts": [options[i] for i in threshold_selected],
                "threshold": threshold,
                "options": results_options,
                "ground_truth_indices": ground_truth_indices,
                "ground_truth_texts": [options[i] for i in ground_truth_indices] if ground_truth_indices is not None else None,
                "ground_truth_source": ground_truth_source,
                "top1_correct": top1_correct,
                "exact_match": exact_match,
                "is_original_dataset_example": unchanged,
                "metadata": (example or {}).get("metadata", {}) if unchanged and example else {},
                "video_duration_seconds": float(np.asarray(video.get("duration_sec", [0])).reshape(-1)[0]) if upload_id else None,
            }

    def upload_video(self, upload_id: str, username: str, path: Path, filename: str) -> None:
        with self.lock:
            self.uploads[upload_id] = {
                "username": username, "path": str(path), "filename": filename,
                "features": None, "created_at": time.time(), "last_used": time.time(),
            }

    def get_upload(self, upload_id: str, username: str) -> dict[str, Any] | None:
        self.cleanup_uploads()
        with self.lock:
            item = self.uploads.get(upload_id)
            if not item or item.get("username") != username:
                return None
            item["last_used"] = time.time()
            return item

    def cleanup_uploads(self, max_age_seconds: int = 4 * 60 * 60) -> None:
        now = time.time()
        with self.lock:
            expired = [key for key, item in self.uploads.items() if now - float(item.get("last_used", item.get("created_at", now))) > max_age_seconds]
            for key in expired:
                path = Path(self.uploads[key].get("path", ""))
                path.unlink(missing_ok=True)
                self.uploads.pop(key, None)
