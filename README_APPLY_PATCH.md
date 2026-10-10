# Sage mixed-target semantics fix

This patch fixes handling of valid all-zero labels in `task_type: multi_label` rows while preserving strict single-choice validation.

## Changes

- Target interpretation now uses `data.target.task_type_column` and `multi_label_task_types`, not the number of positive labels.
- A multi-label row may contain zero, one, or several positives. By default, an empty target is accepted only when `truth_state` explicitly indicates false.
- Non-multi-label rows must contain exactly one positive target.
- The collator emits `target_modes` and `is_multi_label_task` for each batch.
- `mixed_ce_bce` applies cross-entropy only to single-choice tasks and masked BCE to every multi-label task, including one-positive and zero-positive cases.
- Metrics handle empty target sets; an empty prediction can exactly match an empty target. Model selection uses single-choice accuracy and a multi-label score averaging micro-F1 and exact-set match.
- Validator output reports actual task modes and empty multi-label rows.
- Adds `scripts/audit_qa_targets.py` for a dataset-wide target audit without reading feature stores.

## Apply

Extract this ZIP at the Sage repository root, merging/overwriting only the listed files. It contains no generated datasets, embeddings, model checkpoints, logs, or environment directories.

Review first with Git:

```bash
git status --short
# after extracting
git diff --check
git diff --stat
```

## Validate before training

```bash
python -m pytest -q
python -m scripts.audit_qa_targets \
  --config configs/training/experiments/llama_10k.yaml \
  --scale 10k --split all
python -m scripts.validate_training_data \
  --config configs/training/experiments/llama_10k.yaml \
  --scale 10k --split val --batches 4
```

Then use a new run name (do not resume a checkpoint trained with the earlier target semantics):

```bash
python -m scripts.train \
  --config configs/training/experiments/llama_10k.yaml \
  --set training.max_steps_per_epoch=10 \
  --set training.epochs=1 \
  --run-name llama_bidir_mixed_targets_smoke_10k_v2
```

## Validation status

The included unit test suite passes in the build environment. Real-data audit and the training smoke run must still be performed against the actual Sage feature stores.
