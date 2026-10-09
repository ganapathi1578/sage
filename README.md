# FastViTHD frame-level embedding extraction for Sage

This package replaces the VideoMamba window extractor with **FastViTHD frame-level feature extraction**. It is designed to be copied into the `~/sage` repository root.

## What it stores

For each video, the Zarr v2 group stores:

- `features`: `[num_frames, num_spatial_tokens, feature_dim]`, default `float16`.
- `timestamps_sec`: requested sample timestamps measured from the start of the video.
- `source_frame_indices`: decoded source-video frame index selected for each timestamp.
- `spatial_xy`: normalized `(x, y)` center for each stored spatial token.
- `frame_pooled` (optional): mean over spatial tokens, `[num_frames, feature_dim]`.
- Zarr attributes: video ID and path, duration, source/sample FPS, preprocessing, model ID/revision/commit, feature space, native/output feature grid, dtype, and completion status.

No temporal pooling or fixed-length windows are applied during extraction. This is intentional: a training run can use the same cached features at 0.25, 0.4, 1, 2, 4, or 8 FPS without running FastViTHD again. The loader chooses the nearest stored timestamp. Rates such as 0.25, 0.4, 1, 2, and 4 divide an 8 FPS extraction grid exactly; arbitrary rates may use nearest-frame selection.

## Why Zarr instead of LMDB?

Zarr is the default because your loader needs **time-slice reads** from per-video arrays. Chunked arrays can read just the temporal range needed by a training sample. LMDB is excellent for fast key/value lookup, but if each video's entire feature tensor is one value, the loader generally has to fetch/decode the whole value even when it needs only a short segment. HDF5 is also viable but one-writer/concurrent-writer behavior needs care. This script uses one Zarr writer and supports resuming video-by-video.

For parallel multi-GPU extraction, use one independent output store per GPU/process and combine stores afterward; do not let multiple independent writers mutate the same Zarr directory concurrently. For now, `--gpu 0`, `--gpu 1`, etc. selects one GPU per process.

## 1. Copy into Sage

From this package's extracted folder, copy the two directories and config into the Sage repository:

```bash
cp -r scripts ~/sage/
mkdir -p ~/sage/configs/embeddings
cp configs/embeddings/fastvithd_charades.yaml ~/sage/configs/embeddings/
cp README_FastViTHD.md ~/sage/
cp requirements-fastvithd.txt ~/sage/
```

If `~/sage/scripts` already exists, copy the new scripts into it rather than replacing the directory.

## 2. Install dependencies

Use a Python environment with your Sage dependencies, then run. Since Transformers 5.x is a major dependency change, use a dedicated environment if the existing Sage environment is pinned to Transformers 4.x:

```bash
cd ~/sage
pip install -r requirements-fastvithd.txt
```

The extractor uses the native Transformers FastVLM implementation and requires a Transformers version that exposes `AutoModelForImageTextToText` and `FastVlmForConditionalGeneration`. The default model ID is `KamilaMila/FastVLM-0.5B`, matching the native Transformers integration. Some older `apple/FastVLM-0.5B` repository revisions have a legacy LLaVA config that names `mobileclip_l_1024`; the script checks that the loaded tower is FastViT and stops instead of silently extracting from the wrong encoder.

The first run downloads the checkpoint into the Hugging Face cache. Pin `model.revision` to a commit hash after choosing the version you want, so later runs remain reproducible.

## 3. Configure extraction

Edit `configs/embeddings/fastvithd_charades.yaml`:

- `input.video_dir`: video directory.
- `input.recursive`: recursively search subdirectories.
- `preprocessing.image_size`: `256`, `512`, `768`, or `1024` are reasonable starting sizes (the script checks for a multiple of 64).
- `preprocessing.resize_mode`: `letterbox` preserves the whole scene and is the default for videos; `center_crop` removes content along the long dimension.
- `sampling.sample_fps`: extraction sampling rate. Choose the **maximum FPS you plan to train on**.
- `embeddings.spatial_pool_grid_size`: `4` stores 16 spatial tokens/frame; `0` retains the full native grid (more storage); `2` stores four coarse spatial tokens/frame.
- `model.feature_space`: `raw_vision` (default, better if you want to train your own projection) or `fastvlm_projected` (use FastVLM's trained vision-to-language projector, which produces features in its language-model space).
- `runtime.device`: `cuda:0`, `cuda:1`, `cuda:2`, `cpu`, or `auto`.
- `runtime.frame_batch_size`: reduce it if you hit CUDA OOM; the encoder splits OOM batches automatically as a fallback.
- `storage.output_store`: Zarr directory path.

**Keep `raw_vision` if the downstream projector must be trainable.** Use `fastvlm_projected` only if you intentionally want to store features after the pretrained FastVLM projector. The `feature_dim` is discovered from the loaded checkpoint and printed at runtime; do not hardcode it in the training model.

## 4. Smoke test, then full run

From `~/sage`:

```bash
python -u scripts/extract_fastvithd_embeddings.py \
  --config configs/embeddings/fastvithd_charades.yaml \
  --limit 10
```

If the shape/dtype and sample rate look correct, run the full set:

```bash
python -u scripts/extract_fastvithd_embeddings.py \
  --config configs/embeddings/fastvithd_charades.yaml
```

Choose a GPU explicitly:

```bash
python -u scripts/extract_fastvithd_embeddings.py --gpu 1
```

Background run:

```bash
mkdir -p logs
nohup python -u scripts/extract_fastvithd_embeddings.py \
  --config configs/embeddings/fastvithd_charades.yaml \
  > logs/fastvithd_charades_8fps.log 2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f logs/fastvithd_charades_8fps.log
```

The script resumes completed groups. If you change the model, input size, sample FPS, feature space, pooling grid, or dtype, use a new `output_store` path. `--overwrite` explicitly rebuilds outputs in place.

## 5. Load a video at another FPS during training

```python
from scripts.fastvithd_store import load_video_features

sample = load_video_features(
    "data/embeddings/charades_v1_480/fastvithd_512_8fps.zarr",
    video_id="Charades_v1_XXXXXX",  # group ID is the path relative to video_dir, without extension
    target_fps=0.4,
    start_sec=0.0,
    end_sec=32.0,
)

features = sample["features"]                  # [T, spatial_tokens, feature_dim], float32 for training
source_times = sample["timestamps_sec"]       # time where the selected cached features came from
requested_times = sample["target_timestamps_sec"]
spatial_xy = sample["spatial_xy"]             # [spatial_tokens, 2], normalized x/y centers
```

If the video ID has subdirectories, include them using forward slashes, e.g. `nested/Charades_v1_XXXXXX`. Use the actual IDs printed in the manifest. `target_timestamps_sec` describes the desired sampling grid; `timestamps_sec` describes the actual cached frames selected. Use actual timestamps for faithful temporal encoding, particularly when using non-divisor FPS values.

To get just the cheaper spatially pooled features:

```python
sample = load_video_features(
    "data/embeddings/charades_v1_480/fastvithd_512_8fps.zarr",
    video_id="Charades_v1_XXXXXX",
    target_fps=1.0,
    start_sec=0.0,
    end_sec=32.0,
    include_frame_pooled=True,
)
frame_vectors = sample["features"]  # [T, feature_dim]
```

## 6. Output and logs

The default outputs are:

```text
data/embeddings/charades_v1_480/fastvithd_512_8fps.zarr/
data/embeddings/charades_v1_480/fastvithd_512_8fps.zarr.manifest.csv
data/embeddings/charades_v1_480/fastvithd_512_8fps.zarr.failures.jsonl   # only if errors occur
```

Each video group is marked `status=complete` only after all its arrays have been written. Interrupted or incomplete groups are rebuilt on the next run. The manifest reports the exact token count and feature dimension produced by the loaded encoder.

## Storage planning

Approximate uncompressed feature bytes:

`duration_seconds * sample_fps * tokens_per_frame * feature_dim * bytes_per_value`

At 8 FPS, 16 stored tokens/frame, feature dimension 3072 and FP16, one frame is 96 KiB and a 30-second clip is about 22.5 MiB before compression. If you retain 64 tokens/frame instead, that is about four times larger. Use a lower spatial pool grid or smaller `image_size` when storage is constrained; use `spatial_pool_grid_size: 0` only when you want to preserve all native spatial tokens.
