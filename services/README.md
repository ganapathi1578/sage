# Sage Model Playground

An authenticated, shared-network web UI for browsing Sage video-QA datasets, loading trained experiment checkpoints, editing questions/options for one-off inference, and testing uploaded videos. It reads Sage's existing datasets, embeddings, checkpoints, and Zarr video-feature cache in place; it does not copy those large artifacts into `services/`.

## Features

- Shared-account sign-in with signed sessions, CSRF checks, origin checks, and login rate limiting.
- Model/checkpoint discovery from `outputs/experiments/` and lazy loading of a single model onto the configured device.
- Dataset scale/split browser for `UniProp/data/analysis/<scale>/compact/{train,val,test}`.
- Lazy, resumable SQLite index per scale/split for fast video/question/filter browsing. The first index can take time and disk space, especially for 1M/10M scales; the SQLite index is cached in `services/cache/catalogs/` and reused while source shard fingerprints remain unchanged.
- Original-video preview from explicitly configured video roots; cached-feature inference from the existing Sage Zarr feature store.
- Question/option editing and custom question inference without writing changes into Parquet.
- Temporary custom-video uploads; FastViTHD features are extracted lazily on first inference and are not added to the dataset.
- Option scores, top-1 answer, multi-label threshold selection, and ground-truth comparison for unmodified dataset examples.

## Requirements

Run inside the existing Sage Conda environment where Sage's training dependencies are installed. In particular, the application needs PyTorch, FastAPI, Uvicorn, `python-multipart`, PyArrow, Zarr v2, OpenCV, Transformers/timm, Hugging Face Hub, and SentenceTransformers. Avoid reinstalling/replacing the existing PyTorch/CUDA stack just to add the UI dependencies.

From the repository root:

```bash
conda activate sage
python -m pip install -r services/requirements.txt
```

## Configure credentials and paths

Copy `services/.env.example` to `services/.env` and change **all** credential/secret values before starting. `services/start.sh` loads this file. The password is one shared login for the research group; each browser receives its own signed session. Do not commit `services/.env`.

At minimum set:

- `SAGE_PLAYGROUND_USERNAME`: shared research account username.
- `SAGE_PLAYGROUND_PASSWORD`: a long, unique password.
- `SAGE_PLAYGROUND_SECRET_KEY`: random secret with at least 32 UTF-8 bytes. Generate one with `python -c 'import secrets; print(secrets.token_urlsafe(48))'`.
- `SAGE_PLAYGROUND_VIDEO_ROOTS`: absolute path(s) to raw dataset videos, separated with `:` on Linux.

Defaults point to the repository's usual Sage paths:

- checkpoints: `outputs/experiments/`
- compact Parquet QA data and sentence dictionary: `UniProp/data/analysis/`
- MiniLM token embeddings: `data/embeddings/text/all-MiniLM-L6-v2-tokens/` (or the compatible path recorded in the selected experiment config)
- FastViTHD feature cache: `data/embeddings/charades_v1_480/fastvithd_256_1fps.zarr`
- FastViTHD preprocessing config: `configs/embeddings/fastvithd_charades.yaml`

The selected experiment config is used for its `data.video_store` and `data.text_token_store_root` paths unless an explicit service environment override is set. Token caches must use `ragged_token_embeddings_v1`; pooled sentence embeddings are not accepted. Change these with `SAGE_PLAYGROUND_OUTPUT_ROOT`, `SAGE_PLAYGROUND_ANALYSIS_ROOT`, `SAGE_PLAYGROUND_TEXT_STORE_ROOT`, `SAGE_PLAYGROUND_VIDEO_STORE`, `SAGE_PLAYGROUND_FASTVITHD_CONFIG`, `SAGE_PLAYGROUND_VIDEO_ROOTS`, `SAGE_PLAYGROUND_INDEX_ROOT`, and `SAGE_PLAYGROUND_UPLOAD_ROOT` if your server uses different locations. Relative path values are resolved against the Sage repository root; use absolute paths for video roots.

## Start

```bash
cd ~/sage
conda activate sage
chmod 700 services/start.sh
chmod 600 services/.env
./services/start.sh
```

By default the server listens on `0.0.0.0:8765` so approved clients on the same network can connect. Find the server's LAN IP with `hostname -I`, then open `http://<server-lan-ip>:8765` from a client on that network. Set `SAGE_PLAYGROUND_PORT` to change the port.

### Shared-network security

- Restrict port 8765 at the host/network firewall to the intended campus/research subnet or VPN. Do not expose this development server directly to the public Internet.
- For networks you do not fully trust, place it behind HTTPS (and then set `SAGE_PLAYGROUND_COOKIE_SECURE=1`). Plain HTTP does not protect credentials/session cookies from observers on an untrusted network.
- Use a long shared password; rotate it if a team member leaves. The app currently has one shared account rather than a multi-user identity/role system.
- Only one checkpoint is held on the configured inference device at a time. Switching checkpoints unloads the previous model, and inference requests serialize through a model lock to avoid overlapping GPU allocations.
- Uploads are limited by `SAGE_PLAYGROUND_MAX_UPLOAD_BYTES` (default 500 MiB) and `SAGE_PLAYGROUND_MAX_UPLOAD_DURATION_SECONDS` (default 600s). Temporary videos are removed after inactivity and never incorporated into dataset files.
- The UI currently allows browsing the configured train/val/test splits because this is a research inspection tool. Avoid providing access to untrusted users if the test split is intended to remain hidden.

## First use

1. Sign in.
2. Choose a model experiment and `best_checkpoint.pt` or `last_checkpoint.pt`, then load it.
3. Choose a dataset scale and split.
4. Wait for the progress indicator while a SQLite index is built the first time for that split.
5. Pick a video and question. You can filter by task type or reasoning family.
6. Edit the question/options or upload a custom video, then run inference.

For original dataset items, the UI displays correctness only when the question, options, and video remain unchanged. Edited questions and uploaded videos are treated as custom inference and are not written to the dataset. You may optionally mark one expected answer in the UI for a custom test.

## Troubleshooting

- **No model listed:** check that the experiment directory contains `best_checkpoint.pt` or `last_checkpoint.pt` under the configured checkpoint root.
- **Dataset index errors:** verify PyArrow is installed and that the scale contains the compact train/val/test Parquet shards and `sentence_dictionary.parquet`.
- **Text or video features missing:** check the text manifest and configured Zarr store path. The model's expected video dimension must match the configured feature cache (the supplied FastViTHD cache is 3072-dimensional raw-vision features).
- **Uploaded-video inference fails:** the first custom video requires the original FastViTHD extraction dependencies and a working model download/cache; check that the running server can access any needed Hugging Face model files. Precompute/cache the model where the server has network access if the environment is offline.
- **CUDA unavailable:** set `SAGE_PLAYGROUND_DEVICE=cpu` for a CPU-capable debug run, or fix the Sage CUDA environment. Custom FastViTHD extraction on CPU may be very slow.
- **Dataset index failed:** check the error shown in the UI/server log, fix the underlying source/dependency, then restart the service to reset the in-process failed status. The SQLite index can resume partial indexing on startup.

## Validation performed in the build environment

The service modules pass Python bytecode compilation, the frontend passes `node --check`, and service smoke/security tests pass. The build container does not contain the actual Sage dataset/checkpoints and lacks some runtime packages (`pyarrow`, `zarr`, `sentence-transformers`), so full real-checkpoint inference could not be end-to-end verified here. A synthetic feature batch did pass a forward pass through the configured `llama_bidir` model, but this does not replace a real-checkpoint smoke test on the server. Validate one known cached QA example on the Sage server before opening the UI to the group.
