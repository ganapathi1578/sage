#!/usr/bin/env python3
"""Extract reusable, timestamped FastViTHD frame-token embeddings into a Zarr v2 store.

Designed for use from a Sage repository root. Frames are encoded at a configured maximum FPS
and saved per frame, without video windows or temporal pooling. Training can later resample the
cached token sequence at 0.25, 0.4, 1, 2, 4, or any FPS <= the extraction FPS.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr
from numcodecs import Blosc

PIPELINE_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/embeddings/fastvithd_charades.yaml",
        help="YAML path relative to the repository root (the parent of scripts/), unless absolute.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N videos (smoke test).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing output groups.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip compatible completed groups.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0, cuda:2, cpu, auto.")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA GPU index, e.g. --gpu 2 (overrides --device).")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    prep = cfg["preprocessing"]
    sampling = cfg["sampling"]
    emb = cfg["embeddings"]
    storage = cfg["storage"]
    runtime = cfg["runtime"]

    image_size = int(prep["image_size"])
    if image_size < 128 or image_size % 64 != 0:
        raise ValueError("preprocessing.image_size should be >=128 and divisible by 64 (e.g. 256, 512, 768, 1024).")
    if str(prep.get("color_order", "RGB")).upper() != "RGB":
        raise ValueError("Only preprocessing.color_order='RGB' is supported.")
    if prep.get("resize_mode", "letterbox") not in {"letterbox", "center_crop"}:
        raise ValueError("preprocessing.resize_mode must be 'letterbox' or 'center_crop'.")
    if prep.get("interpolation", "bicubic") not in {"bicubic", "bilinear", "area"}:
        raise ValueError("preprocessing.interpolation must be bicubic, bilinear, or area.")
    if float(sampling["sample_fps"]) <= 0:
        raise ValueError("sampling.sample_fps must be > 0.")
    if int(emb.get("spatial_pool_grid_size", 0)) < 0:
        raise ValueError("spatial_pool_grid_size must be 0 (keep native grid) or a positive integer.")
    if emb.get("save_dtype", "float16") not in {"float16", "float32"}:
        raise ValueError("embeddings.save_dtype must be float16 or float32.")
    if model.get("feature_space", "raw_vision") not in {"raw_vision", "fastvlm_projected"}:
        raise ValueError("model.feature_space must be raw_vision or fastvlm_projected.")
    if storage.get("format", "zarr_v2") != "zarr_v2":
        raise ValueError("This extractor implements storage.format='zarr_v2'.")
    if int(storage.get("chunk_frames", 32)) < 1:
        raise ValueError("storage.chunk_frames must be positive.")
    if int(runtime.get("frame_batch_size", 8)) < 1:
        raise ValueError("runtime.frame_batch_size must be positive.")
    if runtime.get("amp_dtype", "float16") not in {"float16", "bfloat16"}:
        raise ValueError("runtime.amp_dtype must be float16 or bfloat16.")
    if runtime.get("amp_dtype", "float16") == "bfloat16" and torch.cuda.is_available():
        # Allow only where hardware supports it; Turing (RTX 2080 Ti) does not.
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 AMP was selected, but this GPU does not support BF16. Use float16.")


def load_fastvithd(cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module | None, str]:
    """Load FastVLM's trained FastViTHD tower; keep only the vision tower on the selected device."""
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise RuntimeError(
            "This extractor needs a Transformers release with native FastVLM support. "
            "Install requirements-fastvithd.txt (transformers>=5.0.0, timm>=1.0.24)."
        ) from exc

    model_cfg = cfg["model"]
    model_id = str(model_cfg["model_id"])
    revision = str(model_cfg.get("revision", "main"))
    use_remote_code = bool(model_cfg.get("trust_remote_code", False))
    load_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading checkpoint: {model_id}@{revision}", flush=True)
    print("Loading model weights on CPU first; only the vision tower will be moved to the selected device.", flush=True)

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": use_remote_code,
        "low_cpu_mem_usage": True,
    }
    # Transformers 5.x uses dtype; fallback keeps this script usable with some 4.x model builds.
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=load_dtype, **load_kwargs)
    except TypeError as first_exc:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=load_dtype, **load_kwargs)
        except Exception:
            raise first_exc

    model_root = getattr(model, "model", model)
    vision = getattr(model_root, "vision_tower", None)
    projector = getattr(model_root, "multi_modal_projector", None)
    if vision is None:
        raise RuntimeError(
            f"Loaded {model_id!r}, but model.model.vision_tower was not found. "
            "This checkpoint/API may not expose the FastViTHD tower through Transformers' native FastVLM class."
        )

    vision_cfg = getattr(vision, "config", None)
    architecture = str(getattr(vision_cfg, "architecture", "unknown"))
    # `fastvit_mci3` is the timm architecture used by the native FastVLM FastViTHD wrapper.
    if architecture != "unknown" and "fastvit" not in architecture.lower():
        raise RuntimeError(
            f"Checkpoint vision architecture is {architecture!r}, not a FastViT/FastViTHD architecture. "
            "Check model.model_id. Some legacy apple/FastVLM-0.5B revisions point to mobileclip_l_1024."
        )

    vision.eval().to(device=device, dtype=load_dtype)
    for parameter in vision.parameters():
        parameter.requires_grad_(False)

    feature_space = str(model_cfg.get("feature_space", "raw_vision"))
    if feature_space == "fastvlm_projected":
        if projector is None:
            raise RuntimeError("feature_space=fastvlm_projected requires model.model.multi_modal_projector.")
        projector.eval().to(device=device, dtype=load_dtype)
        for parameter in projector.parameters():
            parameter.requires_grad_(False)
    else:
        projector = None

    commit_hash = str(
        getattr(getattr(model, "config", None), "_commit_hash", None)
        or getattr(getattr(vision, "config", None), "_commit_hash", None)
        or revision
    )
    print(f"Vision architecture: {architecture}; feature_space={feature_space}; device={device}; dtype={load_dtype}", flush=True)

    # The full model's language model and LM head are not needed after extracting these module references.
    del model, model_root
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vision, projector, commit_hash


def cv2_interpolation(name: str) -> int:
    return {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
    }[name]


def preprocess_frame(frame_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Convert BGR HWC uint8 to square RGB uint8. Letterbox preserves the full scene."""
    prep = cfg["preprocessing"]
    image_size = int(prep["image_size"])
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    interpolation = cv2_interpolation(str(prep.get("interpolation", "bicubic")))

    if prep.get("resize_mode", "letterbox") == "letterbox":
        scale = min(image_size / max(1, width), image_size / max(1, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        left = (image_size - new_w) // 2
        top = (image_size - new_h) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return np.ascontiguousarray(canvas)

    # Resize shorter side to the requested input size, then center crop the long side.
    scale = image_size / min(height, width)
    new_w = max(image_size, int(round(width * scale)))
    new_h = max(image_size, int(round(height * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    crop = resized[top:top + image_size, left:left + image_size]
    return np.ascontiguousarray(crop)


def encode_image_batch(
    images_rgb: np.ndarray,
    vision: torch.nn.Module,
    projector: torch.nn.Module | None,
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return features [B, tokens, channels], native grid (H,W), output grid (H,W)."""
    image_tensor = torch.from_numpy(images_rgb).to(device=device, dtype=torch.float32, non_blocking=True)
    image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous()
    image_tensor.mul_(float(cfg["preprocessing"].get("rescale_factor", 1.0 / 255.0)))

    runtime = cfg["runtime"]
    amp_enabled = bool(runtime.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.float16 if runtime.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    feature_space = str(cfg["model"].get("feature_space", "raw_vision"))
    pool_grid = int(cfg["embeddings"].get("spatial_pool_grid_size", 0))

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = vision(pixel_values=image_tensor, return_dict=True)
            feature_map = outputs.last_hidden_state
            if feature_map.ndim != 4:
                raise RuntimeError(
                    "Expected FastViTHD to return a [B,C,H,W] feature map, got "
                    f"{tuple(feature_map.shape)}. Verify the Transformers/timm versions and checkpoint."
                )
            batch, channels, native_h, native_w = feature_map.shape
            if feature_space == "fastvlm_projected":
                assert projector is not None
                token_features = projector(feature_map.flatten(2).transpose(1, 2))
                feature_map_for_pool = token_features.transpose(1, 2).reshape(
                    batch, token_features.shape[-1], native_h, native_w
                )
            else:
                feature_map_for_pool = feature_map

            if pool_grid > 0:
                if pool_grid > min(native_h, native_w):
                    raise ValueError(
                        f"spatial_pool_grid_size={pool_grid} exceeds native feature grid {native_h}x{native_w}. "
                        "Use a smaller pool grid, image_size=512+, or 0 to retain the native grid."
                    )
                feature_map_for_pool = F.adaptive_avg_pool2d(feature_map_for_pool, (pool_grid, pool_grid))
                output_h = output_w = pool_grid
            else:
                output_h, output_w = native_h, native_w
            features = feature_map_for_pool.flatten(2).transpose(1, 2).contiguous()

    dtype_name = str(cfg["embeddings"].get("save_dtype", "float16"))
    np_dtype = np.float16 if dtype_name == "float16" else np.float32
    return features.float().cpu().numpy().astype(np_dtype, copy=False), (native_h, native_w), (output_h, output_w)


def make_spatial_xy(grid_h: int, grid_w: int) -> np.ndarray:
    """Normalized (x,y) centers for the stored spatial token grid."""
    ys = (np.arange(grid_h, dtype=np.float32) + 0.5) / float(grid_h)
    xs = (np.arange(grid_w, dtype=np.float32) + 0.5) / float(grid_w)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)


def frame_batches(video_path: Path, cfg: dict[str, Any]) -> tuple[float, int, Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Return video duration/sample count and a streaming iterator of sampled frame batches."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(source_fps) or source_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: fps={source_fps}, frames={frame_count}")

    duration = frame_count / source_fps
    requested_fps = float(cfg["sampling"]["sample_fps"])
    n_samples = max(1, int(math.ceil(duration * requested_fps - 1e-9)))
    target_times = np.arange(n_samples, dtype=np.float64) / requested_fps
    source_indices = np.rint(target_times * source_fps).astype(np.int64)
    source_indices = np.clip(source_indices, 0, frame_count - 1)
    source_to_targets: dict[int, list[int]] = {}
    for sample_i, source_i in enumerate(source_indices.tolist()):
        source_to_targets.setdefault(int(source_i), []).append(sample_i)

    max_source = int(source_indices[-1])
    batch_size = int(cfg["runtime"].get("frame_batch_size", 8))

    def iterator() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        images: list[np.ndarray] = []
        sample_ids: list[int] = []
        source_ids: list[int] = []
        frame_i = 0
        try:
            while frame_i <= max_source:
                ok, frame = cap.read()
                if not ok:
                    break
                wanted = source_to_targets.get(frame_i)
                if wanted:
                    processed = preprocess_frame(frame, cfg)
                    for sample_i in wanted:
                        images.append(processed)
                        sample_ids.append(sample_i)
                        source_ids.append(frame_i)
                        if len(images) >= batch_size:
                            yield (
                                np.stack(images, axis=0),
                                np.asarray(sample_ids, dtype=np.int64),
                                np.asarray(source_ids, dtype=np.int64),
                            )
                            images.clear()
                            sample_ids.clear()
                            source_ids.clear()
                frame_i += 1
            if images:
                yield (
                    np.stack(images, axis=0),
                    np.asarray(sample_ids, dtype=np.int64),
                    np.asarray(source_ids, dtype=np.int64),
                )
        finally:
            cap.release()

    # First batch must be fetched only after caller has decided to process this video.
    return duration, n_samples, iterator()


def config_signature(cfg: dict[str, Any], model_commit: str) -> str:
    signature_fields = {
        "pipeline_version": PIPELINE_VERSION,
        "model_id": cfg["model"]["model_id"],
        "model_revision": cfg["model"].get("revision", "main"),
        "model_commit": model_commit,
        "feature_space": cfg["model"].get("feature_space", "raw_vision"),
        "image_size": cfg["preprocessing"]["image_size"],
        "resize_mode": cfg["preprocessing"].get("resize_mode", "letterbox"),
        "interpolation": cfg["preprocessing"].get("interpolation", "bicubic"),
        "sample_fps": cfg["sampling"]["sample_fps"],
        "spatial_pool_grid_size": cfg["embeddings"].get("spatial_pool_grid_size", 0),
        "save_dtype": cfg["embeddings"].get("save_dtype", "float16"),
        "rescale_factor": cfg["preprocessing"].get("rescale_factor", 1.0 / 255.0),
    }
    canonical = json.dumps(signature_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def video_signature(global_signature: str, video_path: Path, relative_id: str) -> str:
    stat = video_path.stat()
    payload = f"{global_signature}|{relative_id}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_compressor(cfg: dict[str, Any]) -> Any:
    name = str(cfg["storage"].get("compression", "zstd")).lower()
    if name == "none":
        return None
    level = int(cfg["storage"].get("compression_level", 3))
    if name not in {"zstd", "lz4"}:
        raise ValueError("storage.compression must be zstd, lz4, or none")
    return Blosc(cname=name, clevel=level, shuffle=Blosc.BITSHUFFLE)


def remove_group_if_present(root: zarr.hierarchy.Group, key: str) -> None:
    if key in root:
        del root[key]


def is_group_complete_and_compatible(root: zarr.hierarchy.Group, key: str, signature: str) -> bool:
    if key not in root:
        return False
    group = root[key]
    return group.attrs.get("status") == "complete" and group.attrs.get("video_signature") == signature


def extract_one_video(
    video_path: Path,
    relative_id: str,
    key: str,
    root: zarr.hierarchy.Group,
    vision: torch.nn.Module,
    projector: torch.nn.Module | None,
    device: torch.device,
    cfg: dict[str, Any],
    global_signature: str,
    compressor: Any,
) -> dict[str, Any]:
    signature = video_signature(global_signature, video_path, relative_id)
    resume = bool(cfg["run"].get("resume", True))
    overwrite = bool(cfg["run"].get("overwrite", False))

    if key in root:
        existing = root[key]
        if resume and not overwrite and existing.attrs.get("status") == "complete":
            if existing.attrs.get("video_signature") == signature:
                return {"video_id": relative_id, "status": "existing", "duration_sec": existing.attrs.get("duration_sec", ""), "num_frames": existing.attrs.get("num_frames", 0), "num_tokens_per_frame": existing.attrs.get("num_tokens_per_frame", 0), "embedding_dim": existing.attrs.get("embedding_dim", 0), "output_key": key}
            raise RuntimeError(
                f"Existing embeddings for {relative_id!r} were made with different extraction settings or a changed video. "
                "Use --overwrite or change storage.output_store to avoid accidental replacement."
            )
        remove_group_if_present(root, key)

    duration, n_samples, batches = frame_batches(video_path, cfg)
    group = root.create_group(key)
    meta_cap = cv2.VideoCapture(str(video_path))
    source_fps_meta = float(meta_cap.get(cv2.CAP_PROP_FPS))
    frame_count_meta = int(meta_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_cap.release()
    group.attrs.update({
        "status": "writing",
        "video_id": relative_id,
        "source_path": str(video_path.resolve()),
        "duration_sec": float(duration),
        "source_fps": source_fps_meta,
        "requested_sample_fps": float(cfg["sampling"]["sample_fps"]),
        "image_size": int(cfg["preprocessing"]["image_size"]),
        "resize_mode": str(cfg["preprocessing"].get("resize_mode", "letterbox")),
        "feature_space": str(cfg["model"].get("feature_space", "raw_vision")),
        "model_id": str(cfg["model"]["model_id"]),
        "model_revision": str(cfg["model"].get("revision", "main")),
        "model_commit": str(cfg.get("_resolved_model_commit", "unknown")),
        "config_signature": global_signature,
        "video_signature": signature,
        "num_frames_expected": int(n_samples),
        "status_version": PIPELINE_VERSION,
    })

    dtype_name = str(cfg["embeddings"].get("save_dtype", "float16"))
    zarr_dtype = np.dtype("float16" if dtype_name == "float16" else "float32")
    chunk_frames = max(1, min(int(cfg["storage"].get("chunk_frames", 32)), n_samples))
    store_features = None
    store_pooled = None
    cursor = 0
    native_grid = (0, 0)
    output_grid = (0, 0)
    n_tokens = 0
    feature_dim = 0
    started = time.time()

    try:
        for image_batch, sample_ids, source_ids in batches:
            # CUDA OOM retry: split the current batch recursively and concatenate in the original order.
            def encode_with_oom_retry(images: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
                try:
                    return encode_image_batch(images, vision, projector, device, cfg)
                except RuntimeError as exc:
                    is_oom = "out of memory" in str(exc).lower() and device.type == "cuda"
                    if not is_oom or len(images) <= 1:
                        raise
                    torch.cuda.empty_cache()
                    mid = len(images) // 2
                    left = encode_with_oom_retry(images[:mid])
                    right = encode_with_oom_retry(images[mid:])
                    if left[1:] != right[1:]:
                        raise RuntimeError("Feature shape changed between sub-batches after CUDA OOM retry.")
                    return np.concatenate([left[0], right[0]], axis=0), left[1], left[2]

            batch_features, native_grid, output_grid = encode_with_oom_retry(image_batch)
            if store_features is None:
                n_tokens, feature_dim = int(batch_features.shape[1]), int(batch_features.shape[2])
                store_features = group.create_dataset(
                    "features",
                    shape=(n_samples, n_tokens, feature_dim),
                    chunks=(chunk_frames, n_tokens, feature_dim),
                    dtype=zarr_dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                store_features.attrs.update({"axes": ["time", "spatial_token", "channel"], "dtype": dtype_name})
                group.create_dataset("timestamps_sec", data=(np.arange(n_samples, dtype=np.float32) / float(cfg["sampling"]["sample_fps"])), dtype="f4", compressor=compressor, overwrite=True)
                # Compute the intended source frame indices from the same FPS assumptions as frame_batches.
                expected_t = np.arange(n_samples, dtype=np.float64) / float(cfg["sampling"]["sample_fps"])
                expected_source_idx = np.clip(np.rint(expected_t * source_fps_meta).astype(np.int64), 0, max(0, frame_count_meta - 1))
                group.create_dataset("source_frame_indices", data=expected_source_idx, dtype="i8", compressor=compressor, overwrite=True)
                group.create_dataset("spatial_xy", data=make_spatial_xy(*output_grid), dtype="f4", compressor=None, overwrite=True)
                if bool(cfg["embeddings"].get("save_frame_pooled", True)):
                    store_pooled = group.create_dataset(
                        "frame_pooled",
                        shape=(n_samples, feature_dim),
                        chunks=(chunk_frames, feature_dim),
                        dtype=zarr_dtype,
                        compressor=compressor,
                        overwrite=True,
                    )
                    store_pooled.attrs.update({"pooling": "mean_over_spatial_tokens", "axes": ["time", "channel"]})
                group.attrs.update({
                    "native_grid_hw": [int(native_grid[0]), int(native_grid[1])],
                    "output_grid_hw": [int(output_grid[0]), int(output_grid[1])],
                    "num_tokens_per_frame": n_tokens,
                    "embedding_dim": feature_dim,
                })

            if batch_features.ndim != 3 or batch_features.shape[1:] != (n_tokens, feature_dim):
                raise RuntimeError(f"Feature shape changed within video: got {batch_features.shape}; expected (*,{n_tokens},{feature_dim}).")
            # Sample ids are monotonic for regular FPS sampling and are written contiguously, but use ids
            # explicitly to keep correctness if the sampling implementation is changed later.
            assert store_features is not None
            if len(sample_ids) and (len(sample_ids) == 1 or np.all(np.diff(sample_ids) == 1)):
                lo, hi = int(sample_ids[0]), int(sample_ids[-1]) + 1
                store_features[lo:hi, :, :] = batch_features
                group["source_frame_indices"][lo:hi] = source_ids
                if store_pooled is not None:
                    store_pooled[lo:hi, :] = batch_features.mean(axis=1).astype(zarr_dtype, copy=False)
            else:
                # Defensive fallback for non-contiguous indices.
                store_features.oindex[sample_ids, :, :] = batch_features
                group["source_frame_indices"].oindex[sample_ids] = source_ids
                if store_pooled is not None:
                    store_pooled.oindex[sample_ids, :] = batch_features.mean(axis=1).astype(zarr_dtype, copy=False)
            cursor += len(sample_ids)

        if store_features is None or cursor == 0:
            raise RuntimeError("No frames were decoded/encoded for this video.")
        if cursor < n_samples:
            # Some decoders stop before reported duration/frame count. Shrink every time-indexed array
            # to only the frames actually encoded, never expose uninitialized trailing rows to training.
            store_features.resize((cursor, n_tokens, feature_dim))
            group["timestamps_sec"].resize((cursor,))
            group["source_frame_indices"].resize((cursor,))
            if store_pooled is not None:
                store_pooled.resize((cursor, feature_dim))
        group.attrs.update({
            "status": "complete",
            "num_frames": int(cursor),
            "num_frames_expected": int(n_samples),
            "sampling_rate_hz": float(cfg["sampling"]["sample_fps"]),
            "native_grid_hw": [int(native_grid[0]), int(native_grid[1])],
            "output_grid_hw": [int(output_grid[0]), int(output_grid[1])],
            "num_tokens_per_frame": int(n_tokens),
            "embedding_dim": int(feature_dim),
            "feature_dtype": dtype_name,
            "spatial_xy_normalized": True,
            "timestamps_are_seconds_from_video_start": True,
            "elapsed_sec": round(time.time() - started, 3),
        })
        return {
            "video_id": relative_id,
            "status": "ok",
            "duration_sec": round(duration, 4),
            "num_frames": cursor,
            "num_tokens_per_frame": n_tokens,
            "embedding_dim": feature_dim,
            "output_key": key,
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception:
        group.attrs["status"] = "incomplete"
        group.attrs["num_frames_written"] = int(cursor)
        raise


def build_manifest(root: zarr.hierarchy.Group, manifest_path: Path) -> None:
    fields = [
        "video_id", "source_path", "duration_sec", "num_frames", "sampling_rate_hz",
        "num_tokens_per_frame", "embedding_dim", "native_grid_hw", "output_grid_hw", "feature_dtype", "status", "output_key",
    ]
    rows: list[dict[str, Any]] = []
    for key in sorted(root.group_keys()):
        group = root[key]
        attrs = dict(group.attrs)
        if attrs.get("status") != "complete":
            continue
        rows.append({
            "video_id": attrs.get("video_id", ""),
            "source_path": attrs.get("source_path", ""),
            "duration_sec": attrs.get("duration_sec", ""),
            "num_frames": attrs.get("num_frames", ""),
            "sampling_rate_hz": attrs.get("sampling_rate_hz", ""),
            "num_tokens_per_frame": attrs.get("num_tokens_per_frame", ""),
            "embedding_dim": attrs.get("embedding_dim", ""),
            "native_grid_hw": json.dumps(attrs.get("native_grid_hw", [])),
            "output_grid_hw": json.dumps(attrs.get("output_grid_hw", [])),
            "feature_dtype": attrs.get("feature_dtype", ""),
            "status": attrs.get("status", ""),
            "output_key": key,
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, manifest_path)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    cfg = load_yaml(config_path)
    validate_config(cfg)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    torch.set_num_threads(int(cfg["runtime"].get("num_threads", 4)))

    requested_device = args.device or str(cfg["runtime"].get("device", "auto"))
    if args.gpu is not None:
        requested_device = f"cuda:{args.gpu}"
    device = choose_device(requested_device)

    input_dir = resolve_path(repo_root, cfg["input"]["video_dir"])
    store_path = resolve_path(repo_root, cfg["storage"]["output_store"])
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(store_path) + ".manifest.csv")
    failures_path = Path(str(store_path) + ".failures.jsonl")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input video directory does not exist: {input_dir}")

    extensions = {str(ext).lower() for ext in cfg["input"].get("extensions", [".mp4"])}
    recursive = bool(cfg["input"].get("recursive", True))
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    video_paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in extensions)
    limit = args.limit if args.limit is not None else cfg["run"].get("limit_videos")
    if limit is not None:
        video_paths = video_paths[:int(limit)]
    if not video_paths:
        raise FileNotFoundError(f"No video files found under {input_dir} with extensions {sorted(extensions)}")

    cfg["run"]["resume"] = bool(cfg["run"].get("resume", True)) and not args.no_resume
    cfg["run"]["overwrite"] = bool(cfg["run"].get("overwrite", False)) or args.overwrite

    print(f"Repository root: {repo_root}", flush=True)
    print(f"Config:          {config_path}", flush=True)
    print(f"Input:           {input_dir}", flush=True)
    print(f"Videos:          {len(video_paths)}", flush=True)
    print(f"Output Zarr:     {store_path}", flush=True)
    print(f"Device:          {device}", flush=True)
    print(f"Image size:      {cfg['preprocessing']['image_size']} x {cfg['preprocessing']['image_size']}", flush=True)
    print(f"Sampling:        {cfg['sampling']['sample_fps']} FPS (frame-level, no temporal pooling)", flush=True)
    print(f"Spatial grid:    {cfg['embeddings'].get('spatial_pool_grid_size', 0)} (0=native)", flush=True)
    print(f"Feature space:   {cfg['model'].get('feature_space', 'raw_vision')}", flush=True)

    vision, projector, model_commit = load_fastvithd(cfg, device)
    cfg["_resolved_model_commit"] = model_commit
    global_sig = config_signature(cfg, model_commit)
    store_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(store_path), mode="a")
    root.attrs.update({
        "pipeline": "fastvithd_frame_embeddings",
        "pipeline_version": PIPELINE_VERSION,
        "model_id": str(cfg["model"]["model_id"]),
        "model_revision": str(cfg["model"].get("revision", "main")),
        "model_commit": model_commit,
        "global_config_signature": global_sig,
        "feature_space": str(cfg["model"].get("feature_space", "raw_vision")),
        "image_size": int(cfg["preprocessing"]["image_size"]),
        "sample_fps": float(cfg["sampling"]["sample_fps"]),
        "spatial_pool_grid_size": int(cfg["embeddings"].get("spatial_pool_grid_size", 0)),
    })
    compressor = build_compressor(cfg)

    started = time.time()
    newly_done = resumed = failed = 0
    progress_every = max(1, int(cfg["run"].get("progress_every_videos", 10)))
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    for index, video_path in enumerate(video_paths, start=1):
        relative_id = video_path.relative_to(input_dir).with_suffix("").as_posix()
        key = quote(relative_id, safe="-_.")
        signature = video_signature(global_sig, video_path, relative_id)
        if cfg["run"]["resume"] and not cfg["run"]["overwrite"] and is_group_complete_and_compatible(root, key, signature):
            resumed += 1
        else:
            try:
                record = extract_one_video(
                    video_path=video_path,
                    relative_id=relative_id,
                    key=key,
                    root=root,
                    vision=vision,
                    projector=projector,
                    device=device,
                    cfg=cfg,
                    global_signature=global_sig,
                    compressor=compressor,
                )
                if record.get("status") == "existing":
                    resumed += 1
                else:
                    newly_done += 1
                    print(
                        f"[{index}/{len(video_paths)}] OK {relative_id} | frames={record.get('num_frames')} "
                        f"tokens/frame={record.get('num_tokens_per_frame')} dim={record.get('embedding_dim')} "
                        f"time={record.get('elapsed_sec', 0):.1f}s",
                        flush=True,
                    )
            except Exception as exc:
                failed += 1
                failure = {
                    "video_id": relative_id,
                    "source_path": str(video_path.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time_unix": time.time(),
                }
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(video_paths)}] ERROR {relative_id}: {type(exc).__name__}: {exc}", flush=True)

        if index % progress_every == 0 or index == len(video_paths):
            elapsed = time.time() - started
            processed = newly_done + resumed + failed
            speed = processed / max(elapsed, 1e-9)
            print(
                f"Progress {index}/{len(video_paths)} | new={newly_done} resumed={resumed} failed={failed} "
                f"| {speed:.2f} videos/s | elapsed={elapsed / 60:.1f} min",
                flush=True,
            )
            build_manifest(root, manifest_path)

    build_manifest(root, manifest_path)
    elapsed = time.time() - started
    print("\nExtraction summary", flush=True)
    print(f"  considered:      {len(video_paths)}", flush=True)
    print(f"  newly generated: {newly_done}", flush=True)
    print(f"  resumed:         {resumed}", flush=True)
    print(f"  failed:          {failed}", flush=True)
    print(f"  elapsed:         {elapsed / 60:.2f} minutes", flush=True)
    print(f"  Zarr store:      {store_path}", flush=True)
    print(f"  manifest:        {manifest_path}", flush=True)
    if failed:
        print(f"  failures:        {failures_path}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
