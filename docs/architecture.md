# Sage video-QA experiment architecture

## Task contract

Each row is a multiple-choice video-QA problem. The data adapter reads `video_id`, `query_sentence_id`, `option_sentence_ids`, and a target label. A collator retrieves cached video features `[T,P,3072]` and token-level MiniLM features `[L,384]` for the query and every option. The target is a zero-based correct option index.

The target label's source representation must be confirmed in the generated Parquet schema. Run `scripts.inspect_qa_schema.py` first. The default config sets `target.column: labels` and `target.format: auto`; `auto` accepts a scalar option index (zero-based unless `index_base` is changed) or exactly one positive value in a per-option label list. If your labels use another format, configure `data.target.format`, `index_base`, and/or `positive_value` explicitly instead of silently guessing. The compacted files do not normally retain the original `options` strings, so `correct_text` targets require changing preprocessing to preserve/map that information.

## Shared input serialization

For each candidate, the scorer uses a shared, candidate-independent serialized stream:

`<video> visual_tokens </video> <query> query_tokens </query> <option> option_tokens </option> <mask>`

These are learned model marker embeddings, not new MiniLM token IDs. The final `<mask>` hidden state is passed to a scalar scoring head. A single model instance scores each option; it returns `[B,K]` candidate logits for cross-entropy.

## Order properties

- Candidate options: the same scorer evaluates every candidate independently. Permuting candidate rows permutes the output score columns by the same permutation. There is no candidate-index embedding. This is **permutation equivariance of scores**, not invariance of the score vector's column order.
- Spatial patches: each video patch is paired with its `(x,y)` coordinate. Permuting patches and their coordinates together is a symmetry of the input serialization/attention computation.
- Time: video patches share a frame timestamp/temporal index, and RoPE is applied to query/key channels. Temporal order is not discarded: changing time coordinates can change attention scores. The model is intentionally *not invariant to time reversal* because action QA may depend on event order.

## Architectures

### `llama_bidir`

A LLaMA-inspired, not checkpoint-compatible, encoder: RMSNorm, bidirectional scaled-dot-product attention, RoPE, SwiGLU, pre-norm residual blocks, learned `<mask>` scorer. Video features are projected from 3072 to `d_model` and receive a Fourier-feature MLP encoding of spatial `(x,y)`. RoPE uses per-token positions; visual tokens share their frame's temporal position while text tokens retain their own order.

### `channel_vector`

An EqCipher-inspired channel/vector representation without encryption, keys, rotation sessions, or cryptographic guarantees. By default the configured research variant has `channels=384`, `vector_dim=8`, so each hidden token has flattened width 3072. Visual input is projected `3072 -> 3072`; each MiniLM token is projected `384 -> 3072`; tensors are reshaped to `[channels, vector_dim]`. Channel-linear layers mix channels while preserving the vector axis. Attention logits use vector dot products in `vector_multihead` mode, temporal RoPE rotates per-head channels, and spatial coordinates enter as scalar channel gates. `vector_multihead` additionally uses relative XY attention bias; the `invariant_linear` ablation drops pairwise XY bias to keep attention linear in sequence length, while retaining the coordinate gates. This design aims to preserve the channel/vector structure inside its blocks; it is not a formal privacy guarantee.

The channel-vector configuration is computationally heavier. Begin with the LLaMA-inspired 384-width run and use small batch/frame caps for the channel-vector model; increase only after measuring peak memory.

## Padding and attention

All valid sequence positions can attend to all other valid positions (bidirectional attention). Padding keys are masked. Visual frame masks become patch-token masks; text masks come from valid token lengths in the ragged token store. The final `<mask>` marker is always valid.

## Data paths

- ID-coded Parquet: `UniProp/data/analysis/<scale>/compact/<split>/`
- Token embeddings: `data/embeddings/text/all-MiniLM-L6-v2-tokens/<scale>/`
- FastViTHD Zarr: configured by `data.video_store`, normally `data/embeddings/charades_v1_480/fastvithd_256_1fps.zarr`
- Run outputs/checkpoints: `outputs/experiments/<experiment.name>/`

The sentence IDs are scoped to each scale's sentence dictionary. Never use a dictionary/store from another scale. Video IDs must match the `video_id` attr used by the Zarr extractor.

Run this first to inspect the real `labels`, video ID, and option-ID schema:

```bash
python -m scripts.inspect_qa_schema --config configs/training/experiments/llama_10k.yaml --scale 10k --split train --rows 5
```

## Useful commands

```bash
python -m scripts.validate_training_data --config configs/training/experiments/llama_10k.yaml --split train --batches 2
python -m scripts.train --config configs/training/experiments/llama_10k.yaml --set training.max_steps_per_epoch=10
python -m scripts.evaluate --config configs/training/experiments/llama_10k.yaml --checkpoint outputs/experiments/llama_bidir_10k_seed42/best_checkpoint.pt --split test
python -m scripts.run_sweep --config configs/training/sweeps/architecture.yaml --dry-run
python -m scripts.summarize_experiments --root outputs/experiments --output outputs/experiment_summary.csv
```

## Research protocol

1. Validate label semantics and ID/video mapping before training.
2. Run a 10k smoke training and check loss decreases.
3. Compare widths/depths and position/fusion choices with fixed seeds and data artifacts.
4. Select model/config with validation only; use test once for final comparison.
5. Record the resolved configuration, Git revision, counts, best checkpoint, runtime, accuracy, and category-wise metrics.

## Ablations exposed through configuration

`model.use_temporal_rope` and `model.use_xy_encoding` can be disabled independently. `model.attention` in the channel-vector configuration can be `vector_multihead` or `invariant_linear`. `data.spatial_pooling` can be `none`, `mean`, or `max`; `data.target_fps` can select a lower rate than the cached FPS; `data.max_video_frames` caps clip length. Every ablation gets a distinct `experiment.name`.

Examples:

```bash
python -m scripts.train --config configs/training/experiments/llama_10k.yaml --set model.num_layers=2 --set experiment.name=llama_2layers_seed42
python -m scripts.train --config configs/training/experiments/llama_10k.yaml --set model.use_temporal_rope=false --set experiment.name=llama_no_rope_seed42
python -m scripts.train --config configs/training/experiments/channel_vector_10k.yaml --set model.attention=invariant_linear --set experiment.name=channel_linear_attention_seed42
```
