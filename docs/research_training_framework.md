# SageQA research training framework

A config-driven supervised training/evaluation framework for multiple-choice video QA, using cached FastViTHD features and cached token-level MiniLM representations. It includes two research architectures: `llama_bidir` and `channel_vector`.

**This is an integration starter for the Sage repo, not a claim that the current server data schema has been end-to-end verified.** Confirm the configured label column/format and video ID mapping using `scripts/validate_training_data.py` before the first run.

See [`architecture.md`](architecture.md) for the data contract and architectural choices.

## Install

From Sage root:

```bash
python -m pip install -r requirements-training.txt
```

The current repo's `utils/text_token_embedding_store.py` must be the token-level version that reads `ragged_token_embeddings_v1` artifacts. Do not point this training framework at pooled `[384]` stores. The adapter prefers that canonical Sage utility and includes a fallback only to keep this new package testable on its own.

Append the entries in `GITIGNORE_APPEND.txt` to the existing `.gitignore`; do not overwrite Sage's current ignore rules.

Run this first to inspect the real `labels`, video ID, and option-ID schema:

```bash
python -m scripts.inspect_qa_schema --config configs/training/experiments/llama_10k.yaml --scale 10k --split train --rows 5
```

## Validate and train

```bash
python -m scripts.validate_training_data --config configs/training/experiments/llama_10k.yaml --split all --batches 1
python -m scripts.train --config configs/training/experiments/llama_10k.yaml
```

For a quick initial check use `--set training.max_steps_per_epoch=5 --set training.epochs=1`. Start with `llama_bidir`; the `channels=384, vector_dim=8` channel-vector model is substantially more memory-intensive.

## Candidate order and sequence construction

Each option is scored independently by the same network. The score matrix is `[B,K]`; permuting options permutes scores in the same way. Video, question, and candidate tokens are concatenated with learned markers `video_open`, `video_close`, `query_open`, `query_close`, `option_open`, `option_close`, and a final decision `[MASK]` token. Attention is bidirectional with key padding masks; it is not a causal language-modeling setup.

## Mixed single-answer and multi-answer supervision

UniProp may contain ordinary single-correct-option questions and multi-correct questions (for example, prompts such as “Select all objects…”). The dataset adapter therefore converts every row to a multi-hot `target_labels` vector aligned with the option order. A row must have at least one positive label; more than one positive is valid.

The default `training.target_loss: mixed_ce_bce` selects the loss per example:

- exactly one positive option: cross-entropy over valid options;
- multiple positive options: binary cross-entropy averaged over that example's valid options.

Padded options are excluded using `option_mask`. Set `training.target_loss: bce` for an all-BCE ablation. The model interface is unchanged: models return one logit per option in `[B, K]`.

Evaluation reports top-1-any-correct, single-choice accuracy, and (for multi-answer examples) thresholded micro precision/recall/F1 and exact-set match. The default `selection_score` is the mean of single-choice accuracy and multi-answer F1 when both types are present, so checkpoint selection does not judge multi-answer examples only by their top-ranked option. Tune `training.multi_label_threshold` using validation data only. For final test evaluation, freeze the threshold chosen on validation.
