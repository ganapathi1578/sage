"""Lazy reader for the Sage FastViTHD Zarr v2 cache."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


def _encode_video_key(video_id: str) -> str:
    return quote(video_id.replace("\\", "/"), safe="-_ .".replace(" ", ""))


class VideoFeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Video feature store not found: {self.root}")
        self._store = None

    def _open(self):
        if self._store is None:
            try:
                import zarr
            except ImportError as exc:
                raise RuntimeError("zarr<3 is required to read the FastViTHD feature cache") from exc
            self._store = zarr.open_group(str(self.root), mode="r")
        return self._store

    def get(
        self,
        video_id: str,
        *,
        max_frames: int,
        target_fps: float | None,
        training: bool,
        sampling: str,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        store = self._open()
        key = _encode_video_key(video_id)
        if key not in store:
            raise KeyError(f"Video ID {video_id!r} (key={key!r}) not found in {self.root}")
        group = store[key]
        if group.attrs.get("status") != "complete":
            raise RuntimeError(f"Video feature extraction is incomplete for {video_id!r}")
        times = np.asarray(group["timestamps_sec"][:], dtype=np.float32)
        if times.size == 0:
            raise ValueError(f"No frame timestamps found for video {video_id!r}")
        stored_fps = float(group.attrs.get("sampling_rate_hz", group.attrs.get("requested_sample_fps", 0.0)))
        if target_fps is not None:
            if target_fps <= 0:
                raise ValueError("data.target_fps must be positive or null")
            if stored_fps and target_fps > stored_fps + 1e-6:
                raise ValueError(f"Requested {target_fps} FPS exceeds cached FPS {stored_fps} for {video_id!r}")
            step = 1.0 / float(target_fps)
            desired = np.arange(float(times[0]), float(times[-1]) + step * 0.25, step, dtype=np.float32)
            right = np.searchsorted(times, desired, side="left").clip(0, len(times) - 1)
            left = np.clip(right - 1, 0, len(times) - 1)
            use_left = np.abs(times[left] - desired) <= np.abs(times[right] - desired)
            indices = np.where(use_left, left, right).astype(np.int64)
            indices = np.unique(indices)
        else:
            indices = np.arange(len(times), dtype=np.int64)
        if max_frames > 0 and len(indices) > max_frames:
            if training and sampling == "random_contiguous":
                start = int(rng.integers(0, len(indices) - max_frames + 1))
                indices = indices[start:start + max_frames]
            elif sampling == "uniform" or not training:
                positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
                indices = indices[positions]
            elif sampling == "random_uniform":
                positions = np.sort(rng.choice(len(indices), size=max_frames, replace=False))
                indices = indices[positions]
            else:
                raise ValueError(f"Unknown data.video_sampling={sampling!r}")
        frame_times = times[indices]
        features_array = group["features"]
        if len(indices) == 1:
            features = np.asarray(features_array[int(indices[0])], dtype=np.float32)[None, ...]
        elif len(indices) and np.all(np.diff(indices) == 1):
            features = np.asarray(features_array[int(indices[0]):int(indices[-1]) + 1], dtype=np.float32)
        else:
            features = np.asarray(features_array.oindex[indices], dtype=np.float32)
        spatial_xy = np.asarray(group["spatial_xy"][:], dtype=np.float32)
        if features.ndim != 3:
            raise ValueError(f"Expected video features [T,P,C], got {features.shape} for {video_id!r}")
        if spatial_xy.shape != (features.shape[1], 2):
            raise ValueError(f"spatial_xy {spatial_xy.shape} does not match patch count {features.shape[1]}")
        # Relative temporal position in approximately sampled-frame units, stable to absolute clip offset.
        unit = float(target_fps or stored_fps or 1.0)
        temporal_positions = (frame_times - frame_times[0]) * unit
        return {
            "features": features,
            "spatial_xy": spatial_xy,
            "timestamps_sec": frame_times,
            "temporal_positions": temporal_positions.astype(np.float32),
        }

    def close(self) -> None:
        self._store = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store"] = None
        return state
