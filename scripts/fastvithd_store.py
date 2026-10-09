"""Read and temporally resample frame-level FastViTHD features from a Zarr v2 store."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr


def open_store(path: str | Path, mode: str = "r") -> zarr.hierarchy.Group:
    """Open the pipeline's Zarr v2 store."""
    return zarr.open_group(str(Path(path).expanduser()), mode=mode)


def encode_video_key(video_id: str) -> str:
    """Encode a relative video ID so it is a single Zarr group name."""
    from urllib.parse import quote

    return quote(video_id.replace("\\", "/"), safe="-_.")


def get_video_group(store: zarr.hierarchy.Group, video_id: str) -> zarr.hierarchy.Group:
    """Get one video's group by its metadata video_id."""
    key = encode_video_key(video_id)
    if key not in store:
        raise KeyError(f"Video ID not found in store: {video_id!r} (key={key!r})")
    group = store[key]
    if group.attrs.get("status") != "complete":
        raise RuntimeError(f"Embeddings are incomplete for {video_id!r}")
    return group


def resample_indices(
    timestamps_sec: np.ndarray,
    target_fps: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest stored-frame indices and target times for a temporal interval.

    The store should be extracted at >= target_fps. Target times are relative to the video's
    absolute timeline. Each chosen index is the nearest cached timestamp; source frames are not
    decoded again. For exact reproducibility, use target FPS values that divide the extraction FPS.
    """
    times = np.asarray(timestamps_sec, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("timestamps_sec must be a non-empty 1D array")
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if end_sec is None:
        end_sec = float(times[-1] + 0.5 / max(target_fps, 1e-12))
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    step = 1.0 / float(target_fps)
    # Use an exclusive end; never create a target timestamp at the boundary.
    n = max(0, int(np.ceil((end_sec - start_sec) / step - 1e-9)))
    target_times = start_sec + np.arange(n, dtype=np.float64) * step
    target_times = target_times[target_times < end_sec - 1e-9]
    if len(target_times) == 0:
        return np.empty(0, dtype=np.int64), target_times.astype(np.float32)

    right = np.searchsorted(times, target_times, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    use_left = np.abs(times[left] - target_times) <= np.abs(times[right] - target_times)
    chosen = np.where(use_left, left, right).astype(np.int64)
    return chosen, target_times.astype(np.float32)


def load_video_features(
    store_path: str | Path,
    video_id: str,
    target_fps: float | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    include_spatial_xy: bool = True,
    include_frame_pooled: bool = False,
) -> dict[str, Any]:
    """Load all features or resample one video at a requested FPS.

    Returns:
      features: [T, P, C] for patch tokens, or [T, C] if include_frame_pooled=True
      timestamps_sec: original cached timestamps for the selected indices
      target_timestamps_sec: requested temporal grid when resampling, otherwise same as cached
      spatial_xy: [P, 2] normalized token centers (optional)
      source_frame_indices: source video frame index for each selected timestamp
      attrs: a copy of group attributes

    For use in a training Dataset, call this once per sampled video/window and apply the returned
    timestamp values as temporal-position input. A target FPS lower than the extraction FPS only
    selects cached embeddings; it does not run the vision encoder again.
    """
    store = open_store(store_path, "r")
    group = get_video_group(store, video_id)
    all_times = np.asarray(group["timestamps_sec"][:], dtype=np.float32)
    all_source_idx = np.asarray(group["source_frame_indices"][:], dtype=np.int64)
    stored_fps = float(group.attrs.get("sampling_rate_hz", group.attrs.get("requested_sample_fps", 0.0)))
    if target_fps is not None and target_fps > stored_fps + 1e-6:
        raise ValueError(f"target_fps={target_fps} exceeds cached extraction FPS={stored_fps}; re-extract at a higher FPS.")
    if target_fps is None:
        lo = np.searchsorted(all_times, start_sec, side="left")
        hi = len(all_times) if end_sec is None else np.searchsorted(all_times, end_sec, side="left")
        indices = np.arange(lo, hi, dtype=np.int64)
        target_times = all_times[indices].copy()
    else:
        if end_sec is None:
            duration = float(group.attrs["duration_sec"])
            end_sec = duration
        indices, target_times = resample_indices(all_times, target_fps, start_sec, end_sec)
        # target times outside the actual cached range must not silently clamp to the last frame.
        valid = (target_times <= all_times[-1] + 1e-6) & (target_times < float(group.attrs["duration_sec"]))
        indices = indices[valid]
        target_times = target_times[valid]

    def read_indexed(array: Any, idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.empty((0,) + tuple(array.shape[1:]), dtype=np.float32)
        if len(idx) == 1 or np.all(np.diff(idx) == 1):
            return np.asarray(array[int(idx[0]):int(idx[-1]) + 1], dtype=np.float32)
        return np.asarray(array.oindex[idx], dtype=np.float32)

    if include_frame_pooled:
        if "frame_pooled" not in group:
            raise KeyError("This store was created with save_frame_pooled=false")
        features = read_indexed(group["frame_pooled"], indices)
    else:
        features = read_indexed(group["features"], indices)

    result: dict[str, Any] = {
        "video_id": str(group.attrs["video_id"]),
        "features": features,
        "timestamps_sec": all_times[indices],
        "target_timestamps_sec": target_times,
        "source_frame_indices": all_source_idx[indices],
        "attrs": dict(group.attrs),
    }
    if include_spatial_xy and not include_frame_pooled:
        result["spatial_xy"] = np.asarray(group["spatial_xy"][:], dtype=np.float32)
    return result


def list_video_ids(store_path: str | Path, complete_only: bool = True) -> list[str]:
    store = open_store(store_path, "r")
    ids = []
    for key in store.group_keys():
        group = store[key]
        if complete_only and group.attrs.get("status") != "complete":
            continue
        value = group.attrs.get("video_id")
        if value is not None:
            ids.append(str(value))
    return sorted(ids)


def flatten_to_spatiotemporal_tokens(
    sample: dict[str, Any],
    clip_start_sec: float | None = None,
    normalize_time_by_seconds: float | None = None,
) -> dict[str, np.ndarray]:
    """Flatten `[T,P,C]` frame features and attach `[x,y,t]` positions.

    `t` is elapsed seconds from `clip_start_sec` (or the first selected timestamp by default).
    If `normalize_time_by_seconds` is positive, t is divided by that value; for example 32.0
    maps a 32-second clip onto approximately [0,1]. This helper does not add positional embeddings;
    the downstream transformer can encode these coordinates with an MLP, Fourier features, or 3D RoPE.
    """
    features = np.asarray(sample["features"], dtype=np.float32)
    times = np.asarray(sample["timestamps_sec"], dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected patch features [T,P,C], got {features.shape}. Do not use include_frame_pooled=True.")
    xy = np.asarray(sample["spatial_xy"], dtype=np.float32)
    if xy.shape != (features.shape[1], 2):
        raise ValueError(f"spatial_xy shape {xy.shape} does not match P={features.shape[1]} tokens.")
    if clip_start_sec is None:
        clip_start_sec = float(times[0]) if len(times) else 0.0
    t = times - float(clip_start_sec)
    if normalize_time_by_seconds is not None:
        if normalize_time_by_seconds <= 0:
            raise ValueError("normalize_time_by_seconds must be positive or None.")
        t = t / float(normalize_time_by_seconds)
    positions = np.concatenate(
        [
            np.broadcast_to(xy[None, :, :], (len(times), len(xy), 2)),
            np.broadcast_to(t[:, None, None], (len(times), len(xy), 1)),
        ],
        axis=-1,
    )
    return {
        "features": features.reshape(-1, features.shape[-1]),
        "positions_xyt": positions.reshape(-1, 3).astype(np.float32, copy=False),
        "timestamps_sec": times,
    }
